#+build darwin
package gpu

import "core:mem"
import "core:slice"
import "core:log"
import "base:runtime"
import "core:sync"
import "core:dynlib"
import "core:container/priority_queue"
import "core:strings"
import "core:fmt"
import intr "base:intrinsics"

import mtl "darwodin/Metal"
import ca "darwodin/QuartzCore"
import ns "darwodin/Foundation"
import cf "darwodin/CoreFoundation"
import cg "darwodin/CoreGraphics"

// NOTE: We don't really need this dependency. It is only here to 
// supply a hacky solution, so once that hack is gone, this will
// disappear as well.
import "core:unicode/utf8"

@(private="file")
Max_Textures :: 65536
@(private="file")
Max_Samplers :: 256
@(private="file")
Max_BVHs :: 16

@(private="file")
Bytes_Per_Texture_Slot :: size_of(mtl.ResourceID)
@(private="file")
Bytes_Per_Sampler_Slot :: size_of(mtl.ResourceID)
// TODO: Figure out how to support raytracing...
@(private="file")
Bytes_Per_BVH_Slot :: 0

@(private="file")
Graphics_Shader_Push_Constants :: struct #packed {
    vert_data: rawptr,
    frag_data: rawptr,
    indirect_data: rawptr,
}

@(private="file")
Compute_Shader_Push_Constants :: struct #packed {
    compute_data: rawptr,
}

/*
    NOTE: Metal doesn't support aliased shader buffers the same way spirv
    does, at least not directly. However, we can hack together something
    that operates similar to that model, by simply making several distinct
    buffers on the shader side and then binding all of them to the same
    buffer address through the CPU calls. Each buffer then will become a
    simple typed-container for determining how to interpret the data in
    the shared buffer given to them.
*/

@(private="file")
Shader_Texture_Descriptor_Indices := [?]ns.UInteger {
    1, // 2DHeap
    2, // 1DHeap
    3, // 3DHeap
    4, // CubeHeap
    5, // 2DArrayHeap
    6, // CubeArrayHeap
    7, // 1DArrayHeap
}

@(private="file")
Shader_Texture_RW_Descriptor_Indices := [?]ns.UInteger {
    8,  // 2DHeapRW
    9,  // 1DHeapRW
    10, // 3DHeapRW
    11, // 2DArrayHeapRW
    12, // 1DArrayHeapRW
}

@(private="file")
Shader_Sampler_Descriptor_Index : ns.UInteger : 13

@(private="file")
Shader_BVH_Descriptor_Index : ns.UInteger : 14

@(private="file")
Alloc_Handle :: distinct Handle

@(private="file")
Context :: struct
{
    validation: bool,
    features: Features,

    // Common resources
    device: ^mtl.Device,    
    physical_properties: Physical_Properties,
    allocation_set: ^mtl.ResidencySet,
    shader_compiler: ^mtl.MTL4Compiler,

    // Resource pools
    allocs: Resource_Pool(Alloc_Handle, Alloc_Info),
    queues: [Queue]^mtl.MTL4CommandQueue,
    textures: Resource_Pool(Texture_Handle, Texture_Info),
    bvhs: Resource_Pool(BVH, BVH_Info),
    shaders: Resource_Pool(Shader, Shader_Info),
    command_buffers: Resource_Pool(Command_Buffer, Command_Buffer_Info),
    semaphores: Resource_Pool(Semaphore, ^mtl.SharedEvent),
    desc_heaps: Resource_Pool(Descriptor_Heap, Descriptor_Heap_Info),

    cmd_bufs_sem_vals: [Queue]Semaphore_Value,

    // Swapchain
    swapchain: Swapchain,
    swapchain_image_idx: u32,
    frames_in_flight: u32,

    lock: sync.Atomic_Mutex, // Ensures thread-safe access to ctx and VK operations
    tls_contexts: [dynamic]^Thread_Local_Context,
}

@(private="file")
Free_Command_Buffer :: struct
{
    pool_handle: Command_Buffer,
    timeline_value: u64, // Duplicated information from Command_Buffer_Info to avoid locking during search
}

@(private="file")
Thread_Local_Context :: struct
{
    pools: [Queue]^mtl.MTL4CommandAllocator,
    buffers: [Queue][dynamic]Command_Buffer,
    free_buffers: [Queue]priority_queue.Priority_Queue(Free_Command_Buffer),
    samplers: [dynamic]Sampler_Info,  // Samplers are interned but have permanent lifetime
}

@(private="file")
Physical_Properties :: struct
{
    // TODO: Figure out what information may be relevant in the case of 
    // raytracing on metal...

    // bvh_props: vk.PhysicalDeviceAccelerationStructurePropertiesKHR,
    // props2: vk.PhysicalDeviceProperties2,
}

@(private="file")
BVH_Info :: struct
{
    // TODO: Figure out what may be relevant for raytracing on metal...
    
    /*
    handle: vk.AccelerationStructureKHR,
    mem: rawptr,
    is_blas: bool,
    shapes: [dynamic]BVH_Shape_Desc,  // Only used if BLAS.
    blas_desc: BLAS_Desc,
    tlas_desc: TLAS_Desc,
    */
}

@(private="file")
Alloc_Info :: struct
{
    buf_handle: ^mtl.Buffer,
    cpu: rawptr,
    gpu: rawptr,
    align: u32,
    buf_size: u64,
}

Alloc_Impl_Info :: struct
{
    range_end: rawptr,
    handle: Alloc_Handle,
}

@(private="file")
Texture_Info :: struct
{
    handle: ^mtl.Texture,
    owns_image: bool,
}

@(private="file")
Sampler_Info :: struct
{
    info: Sampler_Desc,
    sampler: ^mtl.SamplerState,
}

@(private="file")
Shader_Info :: struct {
    handle: ^mtl.MTL4LibraryFunctionDescriptor,
    current_workgroup_size: [3]u32,
    is_compute: bool,
    graphics_type: Shader_Type_Graphics,
}

@(private="file")
Command_Buffer_Info :: struct {
    handle: ^mtl.CommandBuffer,
    timeline_value: u64,
    thread_id: int,
    queue: Queue,
    compute_shader: Maybe(Shader),
    recording: bool,
    pool_handle: Command_Buffer,

    wait_sems: [dynamic]Semaphore_Value,
    signal_sems: [dynamic]Semaphore_Value,
}

@(private="file")
Descriptor_Heap_Info :: struct
{
    residency_set: ^mtl.ResidencySet,
    textures: ^mtl.Buffer,
    textures_rw: ^mtl.Buffer,
    samplers: ^mtl.Buffer,
    bvhs: ^mtl.Buffer,
}

@(private="file")
Semaphore_Value :: struct
{
    sem: Semaphore,
    val: u64,
}

// Initialization

@(private="file")
ctx: Context

@(private="file")
mtl_logger: log.Logger

@(require_results)
_init :: proc(validation := true, loc := #caller_location) -> bool
{
    scratch, _ := acquire_scratch()

    mtl_logger = context.logger
    ctx.validation = validation

    // For metal, validation layers are set up through environment variables.
    // NOTE: Only enabling on macOS, not the subtarget (which is IOS).
    when ODIN_PLATFORM_SUBTARGET == .Default {
        os.set_env("MTL_DEBUG_LAYER",       "1" if validation else "0");
        os.set_env("MTL_SHADER_VALIDATION", "1" if validation else "0");
    }

    ctx.device = mtl.CreateSystemDefaultDevice()

    ensure((ctx.device)->supportsFamily(.Metal4), "Support for Metal 4 is required for 'no_gfx'").
    
    // If we want to do proper `bindless` rendering techniques, we need our 
    // argument buffers to be tier 2.
    ensure((ctx.device)->argumentBuffersSupport() == ._2, "Tier 2 Argument Buffer support is expected for 'no_gfx' on the Metal Backend")
    
    ensure((ctx.device)->readWriteTextureSupport() == ._2, "Tier 2 read/write texture support is expected for 'no_gfx' on the Metal Backend")

    // Set random initial capacity (need a better metric for this later...)
    ctx.allocation_set = mtl_create_residency_set(32, "allocation")
    if ctx.allocation_set == nil {
        return false
    }
    
    // Set up individual queues for different tasks.
    #unroll for queue in Queue {
        mtl4_queue := (ctx.device)->newMTL4CommandQueue()

        mtl4_queue->addResidencySet(ctx.allocation_set)
        
        ctx.queues[queue] = mtl4_queue
    }

    // Set up queue specific semaphores for command buffers.
    #unroll for queue in Queue {
        ctx.cmd_bufs_sem_vals[queue] = {
            sem = semaphore_create(0),
            val = 0,
        }
    }

    // TODO: Implement logic relating to raytracing support beyond the following...
    if (ctx.device)->supportsRaytracing() /* && ctx.device->supportsFunctionPointers() /* (maybe?) */ */ {
        ctx.features += { .Raytracing }
    }

    // Resource pools 
    pool_init(&ctx.allocs)
    pool_init(&ctx.textures)
    pool_init(&ctx.bvhs)
    pool_init(&ctx.shaders)
    pool_init(&ctx.command_buffers)
    pool_init(&ctx.semaphores)
    pool_init(&ctx.desc_heaps)

    // Initialize the new MTL4 Compiler that handles compiling shaders
    // for the render and compute pipelines. (Although, this is probably 
    // a bad description for what this really is)
    {
        compiler_desc: ^mtl.MTL4CompilerDescriptor = (mtl.MTL4CompilerDescriptor{})->alloc()
        defer compiler_desc->release()

        err: ^ns.Error
        ctx.shader_compiler = (ctx.device)->newCompilerWithDescriptor(compiler_desc, &err)
        mtl_ensure(err, "Unable to initialize MTL4Compiler!\n")
    }

    /*
        NOTE:
        - Residency set per descriptor heap (textures, samplers, bvhs)
        - A single residency set for all buffers allocated through the raw 
          procedures.
    */

    return true;
}

@(private="file")
get_tls :: proc() -> ^Thread_Local_Context
{
    @(thread_local)
    tls_ctx: ^Thread_Local_Context

    if tls_ctx != nil do return tls_ctx

    tls_ctx = new(Thread_Local_Context)

    for queue in Queue
    {
        tls_ctx.pools[queue] = (ctx.device)->newCommandAllocator()
        
        priority_queue.init(
            &tls_ctx.free_buffers[queue],
            less = proc(a,b: Free_Command_Buffer) -> bool {
                return a.timeline_value < b.timeline_value
            },
            swap = proc(q: []Free_Command_Buffer, i, j: int) {
                q[i], q[j] = q[j], q[i]
            }
        )
    }

    if sync.guard(&ctx.lock) do append(&ctx.tls_contexts, tls_ctx)

    return tls_ctx
}

@(private="file")
mtl_create_residency_set :: proc(initial_capacity: u32, $set_type: string) -> ^mtl.ResidencySet
{
    set_desc: ^mtl.ResidencySetDescriptor
    set_desc = (mtl.ResidencySetDescriptor{})->alloc()
    defer set_desc->release()

    set_desc->setInitialCapacity(ns.UInteger(initial_capacity))

    set: ^mtl.ResidencySet

    {
        err: ^ns.Error
        set = (ctx.device)->newResidencySetWithDescriptor(set_desc, &error)

        mtl_ensure(err, "Failed to create " + set_type + " residency set for metal backend.\n")
    }

    return set
}

_cleanup :: proc(loc := #caller_location)
{
    scratch, _ := acquire_scratch()

    {
        // Cleanup all TLS contexts
        for tls_context in ctx.tls_contexts {
            if tls_context != nil {
                for type in Queue {
                    buffers := make([dynamic]vk.CommandBuffer, len(tls_context.buffers[type]), scratch)
                    for buf in tls_context.buffers[type] {
                        cmd_buf_info := pool_get(&ctx.command_buffers, buf)
                        delete(cmd_buf_info.wait_sems)
                        delete(cmd_buf_info.signal_sems)
                        append(&buffers, cmd_buf_info.handle)
                    }

                    if len(buffers) > 0 {
                        vk.FreeCommandBuffers(ctx.device, tls_context.pools[type], u32(len(buffers)), raw_data(buffers))
                    }

                    vk.DestroyCommandPool(ctx.device, tls_context.pools[type], nil)
                    priority_queue.destroy(&tls_context.free_buffers[type])
                    delete(tls_context.buffers[type])
                }

                for sampler in tls_context.samplers
                {
                    vk.DestroySampler(ctx.device, sampler.sampler, nil)
                }

                free(tls_context)
            }
        }

        delete(ctx.tls_contexts)
        ctx.tls_contexts = {}
    }

    destroy_swapchain(&ctx.swapchain)

    // Common resources
    {
        for &layout in ctx.desc_layouts {
            vk.DestroyDescriptorSetLayout(ctx.device, layout, nil)
        }
        delete(ctx.desc_layouts)

        vk.DestroyPipelineLayout(ctx.device, ctx.common_pipeline_layout_graphics, nil)
        vk.DestroyPipelineLayout(ctx.device, ctx.common_pipeline_layout_compute, nil)
    }

    for semaphore in ctx.cmd_bufs_sem_vals {
        semaphore_destroy(semaphore.sem)
    }

    vma.destroy_allocator(ctx.vma_allocator)

    // Check for leaked resources
    can_destroy_device := true
    if ctx.validation
    {
        {
            sb := strings.builder_make_none()
            defer strings.builder_destroy(&sb)

            leaked_allocs := pool_get_alive_list(&ctx.allocs, scratch)
            if len(leaked_allocs) > 0
            {
                strings.write_string(&sb, "There are leaked allocations present:\n")
                can_destroy_device = false

                for leaked, i in leaked_allocs
                {
                    fmt.sbprintf(&sb, "Allocated at: %v", leaked.meta.created_at)
                    if i < len(leaked_allocs) - 1 {
                        fmt.sbprintln(&sb, "")
                    }
                }

                log.error(strings.to_string(sb), location = loc)
            }
        }

        print_leaked_resources(&ctx.textures,   "Texture_Handle", &can_destroy_device, loc)
        print_leaked_resources(&ctx.bvhs,       "BVH",            &can_destroy_device, loc)
        print_leaked_resources(&ctx.shaders,    "Shader",         &can_destroy_device, loc)
        print_leaked_resources(&ctx.semaphores, "Semaphore",      &can_destroy_device, loc)

        print_leaked_resources :: proc(pool: ^Resource_Pool($Handle_T, $Info_T), handle_name: string, can_destroy_device: ^bool, loc: runtime.Source_Code_Location)
        {
            sb := strings.builder_make_none()
            defer strings.builder_destroy(&sb)

            scratch, _ := acquire_scratch()
            leaked_res := pool_get_alive_list(pool, scratch)
            if len(leaked_res) > 0
            {
                fmt.sbprintfln(&sb, "There are leaked %vs present:", handle_name)
                can_destroy_device^ = false

                for leaked, i in leaked_res
                {
                    if leaked.meta.name == "" {
                        fmt.sbprintf(&sb, "(no name), Created at: %v", leaked.meta.created_at)
                    } else {
                        fmt.sbprintf(&sb, "\"%v\", Created at: %v", leaked.meta.name, leaked.meta.created_at)
                    }

                    if i < len(leaked_res) - 1 {
                        fmt.sbprintln(&sb, "")
                    }
                }

                log.error(strings.to_string(sb), location = loc)
            }
        }

        // Destroy pools
        pool_destroy(&ctx.allocs)
        pool_destroy(&ctx.textures)
        pool_destroy(&ctx.bvhs)
        pool_destroy(&ctx.shaders)
        pool_destroy(&ctx.command_buffers)
        pool_destroy(&ctx.semaphores)
    }

    if can_destroy_device {
        vk.DestroyDevice(ctx.device, nil)
    } else {
        runtime.debug_trap()  // Break here so user has a chance to read the last error logs.
    }
}

_wait_idle :: proc()
{
    sync.guard(&ctx.lock)

    // TODO: Implement a "waitForEvent" call making use of something related 
    // to queue submission (?)
    // 
    // I'm not sure how to effectively replicate the device wait idle in metal
    // just yet.
}

_swapchain_init :: proc(_surface: rawptr, init_size: [2]u32, frames_in_flight: u32)
{
    layer := (^ca.MetalLayer)(_surface);

    layer->setDevice(ctx.device)
    layer->setDrawableSize({
        width = cg.Float(max(init_size[0], 1)) * layer->contentsScale(),
        height = cg.Float(max(init_size[1], 1)) * layer->contentsScale(),
    })
    
    if sync.guard(&ctx.lock) {
        // NOTE(MP): According to the metal documentation, it seems that the
        // max number of drawables is 2-3 only.
        assert(frames_in_flight >= 2 && frames_in_flight <= 3, "When initializing the swapchain, frames in flight can only be between 2 and 3 on the metal backend.")
        
        ctx.frames_in_flight = frames_in_flight

        ctx.swapchain.layer = layer
        ctx.swapchain.acquired = false
        ctx.swapchain.current_drawable = nil
        ctx.swapchain.current_texture_handle = nil
    }

    // NOTE(MP): Setting this field to 'false' allows us to wait indefinitely
    // on the call to 'nextDrawable', rather than timeout after 1 second, as is
    // the default behavior.
    // 
    // The main reason we do this is to ensure that the behavior is as close to
    // the vulkan implementation as possible.
    layer->setAllowsNextDrawableTimeout(false)

    layer->setMaximumDrawableCount(ns.UInteger(frames_in_flight))
}

_swapchain_resize :: proc(size: [2]u32)
{
    queue_wait_idle(.Main)

    ctx.swapchain.layer->setDrawableSize({
        width = cg.Float(max(size[0], 1)),
        height = cg.Float(max(size[1], 1)),
    })
}

_swapchain_acquire_next :: proc() -> Texture
{
    assert(ctx.swapchain.layer != nil, "Before calling `swapchain_acquire_next`, make sure to initialize the swapchain with `swapchain_init`.")

    sync.guard(&ctx.swapchain_lock)
    
    assert(!ctx.swapchain.acquired, "You cannot acquire multiple swapchain images at the same time. Present your current work with `swapchain_present` and then you should be able to acquire another image.")

    // NOTE(MP): Because we set `allowsNextDrawableTimeout` to `false` in the
    // init method for the swapchain, this'll wait indefinitely until we get
    // the next drawable.
    drawable := (ctx.swapchain.layer)->nextDrawable()

    texture := (^mtl.Texture)(drawable->texture())
    
    ctx.swapchain.acquired = true
    ctx.swapchain.current_drawable = drawable
    ctx.swapchain.current_texture_handle = texture

    tex_handle := pool_add(&ctx.textures, Texture_Info { handle = texture, owns_image = false }, {})

    // TODO: Handle the 'format' present on the texture to account for 
    // possible different pixel formats for the 'swapchain'
    // 
    // Likely will need a helper method to convert `mtl.PixelFormat` -> `Texture_Format`
    
    return Texture {
        type = .D2,
        dimensions = { u32(texture->width()), u32(texture->height()), 1 },
        format = .BGRA8_Unorm,
        mip_count = u32(texture->mipmapLevelCount()),
        layer_count = 1,
        sample_count = u32(texture->sampleCount()),
        handle = tex_handle,
    }
}

_swapchain_present :: proc(queue: Queue, sem_wait: Semaphore, wait_value: u64)
{
    sync.guard(&ctx.swapchain_lock)
    
    assert(ctx.swapchain.acquired, "Expected to have acquired the swapchain image before calling `swapchain_present` (make sure to call `swapchain_acquire_next` first and use the resulting swapchain image!)")
    
    mtl_queue := ctx.queues[queue]
    mtl_sem := pool_get(&ctx.semaphores, sem_wait)

    mtl_queue->waitForEvent(mtl_sem, auto_cast wait_value)

    (ctx.swapchain.current_drawable)->present()

    // TODO: Complete this logic...
        
    vk_sem_wait := pool_get(&ctx.semaphores, sem_wait)

    present_semaphore := ctx.swapchain.present_semaphores[ctx.swapchain_image_idx]

    // NOTE: Workaround for the fact that swapchain presentation
    // only supports binary semaphores.
    // wait on sem_wait on wait_value and signal ctx.binary_sem
    {
        // Switch to optimal layout for presentation (this is mandatory)
        cmd_buf: Command_Buffer
        {
            cmd_buf = vk_acquire_cmd_buf(queue)
            cmd_buf_info := pool_get(&ctx.command_buffers, cmd_buf)
            vk_cmd_buf := cmd_buf_info.handle

            cmd_buf_bi := vk.CommandBufferBeginInfo {
                sType = .COMMAND_BUFFER_BEGIN_INFO,
                flags = { .ONE_TIME_SUBMIT },
            }
            vk_check(vk.BeginCommandBuffer(vk_cmd_buf, &cmd_buf_bi))

            transition := vk.ImageMemoryBarrier2 {
                sType = .IMAGE_MEMORY_BARRIER_2,
                image = ctx.swapchain.images[ctx.swapchain_image_idx],
                subresourceRange = {
                    aspectMask = { .COLOR },
                    levelCount = 1,
                    layerCount = 1,
                },
                oldLayout = .GENERAL,
                newLayout = .PRESENT_SRC_KHR,
                srcStageMask = { .ALL_COMMANDS },
                srcAccessMask = { .MEMORY_WRITE },
                dstStageMask = { .COLOR_ATTACHMENT_OUTPUT },
                dstAccessMask = { .MEMORY_READ },
            }
            vk.CmdPipelineBarrier2(vk_cmd_buf, &vk.DependencyInfo {
                sType = .DEPENDENCY_INFO,
                imageMemoryBarrierCount = 1,
                pImageMemoryBarriers = &transition,
            })

            vk_check(vk.EndCommandBuffer(vk_cmd_buf))
        }

        // NOTE: Submissions must be performed in order w.r.t the timeline value used.
        sync.guard(&ctx.lock)

        if cmd_buf_info, r_lock := pool_get_mut(&ctx.command_buffers, cmd_buf); sync.guard(r_lock) {
            cmd_buf_info.timeline_value = sync.atomic_add(&ctx.cmd_bufs_sem_vals[cmd_buf_info.queue].val, 1) + 1
        }

        cmd_buf_info := pool_get(&ctx.command_buffers, cmd_buf)
        vk_cmd_buf := cmd_buf_info.handle
        queue_sem := ctx.cmd_bufs_sem_vals[cmd_buf_info.queue].sem
        vk_queue_sem := pool_get(&ctx.semaphores, queue_sem)

        wait_stage_flags := vk.PipelineStageFlags { .COLOR_ATTACHMENT_OUTPUT }
        next: rawptr
        next = &vk.TimelineSemaphoreSubmitInfo {
            sType = .TIMELINE_SEMAPHORE_SUBMIT_INFO,
            pNext = next,
            waitSemaphoreValueCount = 1,
            pWaitSemaphoreValues = raw_data([]u64 {
                wait_value,
            }),
            signalSemaphoreValueCount = 2,
            pSignalSemaphoreValues = raw_data([]u64 {
                {},
                cmd_buf_info.timeline_value,
            })
        }
        submit_info := vk.SubmitInfo {
            sType = .SUBMIT_INFO,
            pNext = next,
            commandBufferCount = 1,
            pCommandBuffers = &vk_cmd_buf,
            waitSemaphoreCount = 1,
            pWaitSemaphores = raw_data([]vk.Semaphore {
                vk_sem_wait,
            }),
            pWaitDstStageMask = raw_data([]vk.PipelineStageFlags {
                wait_stage_flags,
            }),
            signalSemaphoreCount = 2,
            pSignalSemaphores = raw_data([]vk.Semaphore {
                present_semaphore,
                vk_queue_sem,
            }),
        }

        vk_check(vk.QueueSubmit(vk_queue, 1, &submit_info, {}))

        recycle_cmd_buf(cmd_buf)
    }

    sync.guard(&ctx.lock)
    res := vk.QueuePresentKHR(vk_queue, &{
        sType = .PRESENT_INFO_KHR,
        swapchainCount = 1,
        waitSemaphoreCount = 1,
        pWaitSemaphores = &present_semaphore,
        pSwapchains = &ctx.swapchain.handle,
        pImageIndices = &ctx.swapchain_image_idx,
    })
    if res == .SUBOPTIMAL_KHR do log.warn("Suboptimal swapchain acquire!")
    if res != .SUCCESS && res != .SUBOPTIMAL_KHR {
        vk_check(res)
    }
}

_features_available :: proc() -> Features
{
    return ctx.features
}

_device_limits :: proc() -> Device_Limits
{
    return {
        // NOTE(MP): This seems to be the cap for metal according to the docs.
        max_anisotropy = 16.0,
    }
}

// Memory

_mem_alloc_raw :: proc(#any_int el_size, #any_int el_count, #any_int align: i64, mem_type := Memory.Default, loc := #caller_location) -> ptr
{
    requested_bytes := el_size * el_count
    if requested_bytes == 0 do return {}

    resource_options: mtl.ResourceOptions

    switch mem_type 
    {
        case .Default:
        {
            // Defaults to shared memory storage Cull_Mode
            resource_options = {}
        }
        case .GPU:
        {
            resource_options = { .StorageModePrivate }
        }
        case .Readback:
        {
            resource_options = { .CPUCacheModeWriteCombined }
        }
    }

    // Manually align the bytes
    mem_requirements := (ctx.device)->heapBufferSizeAndAlignWithLength(ns.UInteger(bytes), resource_options)

    required_align: i64 = max(i64(mem_requirements.align), align)

    bytes := (requested_bytes + required_align - 1) & ~(required_align - 1)

    // Finally allocate the memory.
    buf := (ctx.device)->newBufferWithLength_options(ns.UInteger(bytes), resource_options)

    // Append to residency set!
    if sync.guard(&ctx.lock) {
        (ctx.allocation_set)->addAllocation(buf)
        // NOTE(MP): Doing this for *every* allocation is probably a bad idea
        // for performance, but for simplicity, this should work fine...
        // 
        // We'll likely have to find a place where we could sync the residency
        // set at, but I don't want to think about that for a while.
        (ctx.allocation_set)->commit()
    }
    
    p: ptr
    if mem_type != .GPU 
    {
        p.cpu = buf->contents()
    }
    
    p.gpu.ptr = cast(rawptr) cast(uintptr) buf->gpuAddress()

    alloc_info := Alloc_Info {
        buf_handle = buf,

        cpu = p.cpu,
        gpu = p.gpu.ptr,

        align = u32(align),
        buf_size = u64(bytes),
    }

    alloc_handle := pool_add(&ctx.allocs, alloc_info, { created_at = loc })
    end_ptr := rawptr(uintptr(p.gpu.ptr) + uintptr(bytes))
    alloc_impl := Alloc_Impl_Info { end_ptr, alloc_handle }
    p.gpu._impl = transmute([2]u64) alloc_impl
    return p
}

_mem_suballoc :: proc(addr: ptr, offset, el_size, el_count: i64, loc := #caller_location) -> ptr
{
    bytes := el_size * el_count

    if ctx.validation
    {
        ok := true
        if bytes != 0 {
            ok &= check_ptr(addr, "addr", loc)
        }
        if !ok do return {}
    }

    if bytes == 0 do return {}

    suballoc_p := addr
    if suballoc_p.cpu != nil {
        suballoc_p.cpu = auto_cast(uintptr(suballoc_p.cpu) + uintptr(offset))
    }
    suballoc_p.gpu.ptr = auto_cast(uintptr(suballoc_p.gpu.ptr) + uintptr(offset))

    // Update internal _impl.
    addr_impl := transmute(Alloc_Impl_Info) addr._impl
    addr_impl.range_end = rawptr(uintptr(suballoc_p.gpu.ptr) + uintptr(bytes))
    suballoc_p._impl = transmute([2]u64) addr_impl

    return suballoc_p
}

_mem_free_raw :: proc(addr: gpuptr, loc := #caller_location)
{
    alloc_impl := transmute(Alloc_Impl_Info) addr._impl
    alloc := alloc_impl.handle

    if ctx.validation
    {
        ok := true
        if addr != {} {
            ok &= check_ptr(addr, "addr", loc)
            ok &= check_ptr_must_not_be_suballoc(addr, "addr", loc)
        }
        if !ok do return
    }

    if addr == {} do return

    alloc_info := pool_get(&ctx.allocs, alloc)
    (alloc_info.buf_handle)->release()
    pool_remove(&ctx.allocs, alloc)
}

// Textures
_texture_size_and_align :: proc(desc: Texture_Desc, loc := #caller_location) -> (size: u64, align: u64)
{
    desc_clean := texture_desc_cleanup(desc)

    tex_desc := to_mtl_texture_descriptor(desc_clean)
    defer tex_desc->release()

    tex_requirements := (ctx.device)->heapTextureSizeAndAlignWithDescriptor(tex_desc)

    return u64(tex_requirements.size), u64(tex_requirements.align)
}

_texture_create :: proc(desc: Texture_Desc, storage: gpuptr, queue: Queue = .Main, signal_sem: Semaphore = {}, signal_value: u64 = 0, name := "", loc := #caller_location) -> Texture
{
    if ctx.validation
    {
        ok := true
        ok &= check_ptr(storage, "storage", loc)
        if !ok do return {}
    }

    desc_clean := texture_desc_cleanup(desc)

    mtl_queue := ctx.queues[queue]
    alloc_impl := transmute(Alloc_Impl_Info) storage._impl
    alloc_info := pool_get(&ctx.allocs, alloc_impl.handle)

    mtl_buf := alloc_info.buf_handle

    offset := uintptr(storage.ptr) - uintptr(alloc_info.gpu)
    
    tex_desc := to_mtl_texture_descriptor(desc_clean)
    defer tex_desc->release()

    bytes_per_row := mtl_helper_bytes_per_row(desc.format, desc.dimensions[0])

    mtl_tex := mtl_buf->newTextureWithDescriptor(tex_desc, ns.UInteger(offset), ns.UInteger(bytes_per_row))
    
    if signal_sem != {} {
        mtl_sem := pool_get(&ctx.semaphores, signal_sem)
        
        mtl_queue->signalEvent(mtl_sem, auto_cast signal_value)
    }

    debug_name_objc := to_mtl_string(name)
    defer debug_name_objc->release()
    
    mtl_tex->setLabel(debug_name_objc)

    tex_info := Texture_Info { handle = image, owns_image = true }
    return Texture {
        type = desc_clean.type,
        dimensions = desc_clean.dimensions,
        format = desc_clean.format,
        mip_count = desc_clean.mip_count,
        sample_count = desc_clean.sample_count,
        layer_count = desc_clean.layer_count,
        handle = pool_add(&ctx.textures, tex_info, { name = name, created_at = loc } )
    }
}

_texture_destroy :: proc(texture: Texture, loc := #caller_location)
{
    if ctx.validation
    {
        ok := true
        ok &= pool_check(&ctx.textures, texture.handle, "texture", loc)
        if !ok do return
    }

    tex_info := pool_get(&ctx.textures, texture.handle)
    (tex_info.handle)->release()
    pool_remove(&ctx.textures, texture.handle)
}

@(private="file")
get_or_add_sampler :: proc(info: Sampler_Desc) -> ^mtl.SamplerState
{
    tls := get_tls()

    for sampler in tls.samplers
    {
        if sampler.info == info {
            return sampler.sampler
        }
    }

    desc: ^mtl.SamplerDescriptor = (mtl.SamplerDescriptor{})->alloc()
    desc->init()
    defer desc->release()

    desc->setMagFilter(to_mtl_filter(info.mag_filter))
    desc->setMinFilter(to_mtl_filter(info.min_filter))
    desc->setMipFilter(to_mtl_filter(info.mip_filter))

    desc->setSAddressMode(to_mtl_address_mode(info.address_mode_u))
    desc->setTAddressMode(to_mtl_address_mode(info.address_mode_v))
    desc->setRAddressMode(to_mtl_address_mode(info.address_mode_w))

    desc->setLodBias(info.mip_lod_bias)
    desc->setLodMinClamp(info.min_lod)
    desc->setLodMaxClamp(info.max_lod)

    desc->setMaxAnisotropy(info.max_anisotropy)

    sampler := (ctx.device)->newSamplerState(desc)
    
    append(&tls.samplers, Sampler_Info { info, sampler })
    return sampler
}

_texture_descriptor :: proc(texture: Texture, view_desc: Texture_View_Desc, loc := #caller_location) -> Texture_Descriptor
{
    if ctx.validation
    {
        ok := true
        ok &= pool_check(&ctx.textures, texture.handle, "texture", loc)
        if !ok do return {}
    }

    tex_info := pool_get(&ctx.textures, texture.handle)

    desc: Texture_Descriptor
    desc[0] = u64(uintptr(tex_info.handle))

    return desc
}

_texture_rw_descriptor :: proc(texture: Texture, view_desc: Texture_View_Desc, loc := #caller_location) -> Texture_Descriptor
{
    // NOTE(MP): The distinction in how this resource is accessed is at texture
    // creation, thus we don't need to specify anything special for the shader to 
    // recognize this as a texture we can read and write to.
    return _texture_view_descriptor(
        texture,
        view_desc,
        loc = loc,
    )
}

_sampler_descriptor :: proc(sampler_desc: Sampler_Desc, loc := #caller_location) -> Sampler_Descriptor
{
    if sampler_desc.max_anisotropy != 0.0 {
        ensure(
            sampler_desc.max_anisotropy >= 1.0 &&
            // NOTE(MP): This is the cap on the metal backend.
            sampler_desc.max_anisotropy <= 16.0,
            "Sampler anisotropy out of range. Call gpu.device_limits() to get the supported maximum anisotropy.",
        )
    }

    sampler := get_or_add_sampler(sampler_desc)

    return Sampler_Descriptor(sampler)
}

_desc_heap_create :: proc(texture_count: u32 = 65536,
                          texture_rw_count: u32 = 65536,
                          sampler_count: u32 = 32,
                          bvh_count: u32 = 16,
                          name := "", loc := #caller_location) -> Descriptor_Heap
{
    if ctx.validation
    {
        ok := true
        if texture_count > Max_Textures {
            log.errorf("'texture_count' is %v and is greater than the maximum allowed by no_gfx (%v)", texture_count, Max_Textures, location = loc)
            ok = false
        }
        if texture_rw_count > Max_Textures {
            log.errorf("'texture_rw_count' is %v and is greater than the maximum allowed by no_gfx (%v)", texture_rw_count, Max_Textures, location = loc)
            ok = false
        }
        if sampler_count > Max_Samplers {
            log.errorf("'sampler_count' is %v and is greater than the maximum allowed by no_gfx (%v)", sampler_count, Max_Samplers, location = loc)
            ok = false
        }
        if bvh_count > Max_BVHs {
            log.errorf("'bvh_count' is %v and is greater than the maximum allowed by no_gfx (%v)", bvh_count, Max_BVHs, location = loc)
            ok = false
        }
        if !ok do return {}
    }

    textures := (ctx.device)->newBufferWithLength_options(ns.UInteger(Bytes_Per_Texture_Slot * texture_count), {})
    textures_rw := (ctx.device)->newBufferWithLength_options(ns.UInteger(Bytes_Per_Texture_Slot * texture_rw_count), {})
    samplers := (ctx.device)->newBufferWithLength_options(ns.UInteger(Bytes_Per_Sampler_Slot * sampler_count), {})
    bvhs := (ctx.device)->newBufferWithLength_options(ns.UInteger(Bytes_Per_BVH_Slot * bvh_count), {})

    residency_set := mtl_create_residency_set(texture_count + texture_rw_count + sampler_count + bvh_count, "descriptor heap")
    
    ensure(residency_set != nil)
    ensure(textures != nil)
    ensure(textures_rw != nil)
    ensure(samplers != nil)
    ensure(bvhs != nil)
    
    desc_heap_info := Descriptor_Heap_Info {
        residency_set = residency_set,
        textures = textures,
        textures_rw = textures_rw,
        samplers = samplers,
        bvhs = bvhs,
    }
    
    return pool_add(&ctx.desc_heaps, desc_heap_info, { created_at = loc, name = name })
}

_desc_heap_destroy :: proc(heap: Descriptor_Heap, loc := #caller_location)
{
    if ctx.validation
    {
        ok := true
        ok &= pool_check(&ctx.desc_heaps, heap, "heap", loc)
        if !ok do return
    }

    heap_info := pool_get(&ctx.desc_heaps, heap)

    (heap_info.residency_set)->endResidency()

    (heap_info.residency_set)->release()
    (heap_info.textures)->release()
    (heap_info.textures_rw)->release()
    (heap_info.samplers)->release()
    (heap_info.bvhs)->release()
    
    pool_remove(&ctx.desc_heaps, heap)
}

_desc_heap_set_textures :: proc(heap: Descriptor_Heap, start_idx: u32, textures: []Texture_Descriptor, loc := #caller_location)
{
    if ctx.validation
    {
        ok := true
        ok &= pool_check(&ctx.desc_heaps, heap, "heap", loc)
        for desc, i in textures {
            ok &= check_texture_descriptor(desc, "textures", i, loc)
        }
        if !ok do return
    }

    heap_info := pool_get(&ctx.desc_heaps, heap)

    scratch, _ := acquire_scratch()
    texture_ids: []mtl.ResourceID = make([]mtl.ResourceID, len(textures), allocator = scratch)
    for &id, i in texture_ids
    {
        texture := (^mtl.Texture)(uintptr(textures[i][0]))

        (heap_info.residency_set)->addAllocation(texture)
        
        // Get the resource id from the descriptor
        id = texture->gpuResourceID()
    }

    heap_textures := cast([^]mtl.ResourceID)(heap_info.textures)->contents()

    mem.copy(
        heap_textures[start_idx:],
        raw_data(texture_ids),
        int(Bytes_Per_Texture_Slot * len(textures))
    )
}

_desc_heap_set_textures_rw :: proc(heap: Descriptor_Heap, start_idx: u32, textures: []Texture_Descriptor, loc := #caller_location)
{
    if ctx.validation
    {
        ok := true
        ok &= pool_check(&ctx.desc_heaps, heap, "heap", loc)
        if !ok do return
    }

    heap_info := pool_get(&ctx.desc_heaps, heap)

    scratch, _ := acquire_scratch()
    texture_ids: []mtl.ResourceID = make([]mtl.ResourceID, len(textures), allocator = scratch)
    for &id, i in texture_ids
    {
        texture := (^mtl.Texture)(uintptr(textures[i][0]))

        (heap_info.residency_set)->addAllocation(texture)
        
        // Get the resource id from the descriptor
        id = texture->gpuResourceID()
    }

    heap_textures_rw := cast([^]mtl.ResourceID)(heap_info.textures_rw)->contents()

    mem.copy(
        heap_textures_rw[start_idx:],
        raw_data(texture_ids),
        int(Bytes_Per_Texture_Slot * len(textures))
    )
}

_desc_heap_set_samplers :: proc(heap: Descriptor_Heap, start_idx: u32, samplers: []Sampler_Descriptor, loc := #caller_location)
{
    if ctx.validation
    {
        ok := true
        ok &= pool_check(&ctx.desc_heaps, heap, "heap", loc)
        if !ok do return
    }

    heap_info := pool_get(&ctx.desc_heaps, heap)

    scratch, _ := acquire_scratch()
    sampler_ids: []mtl.ResourceID = make([]mtl.ResourceID, len(samplers), allocator = scratch)
    for &id, i in sampler_ids
    {
        sampler := (^mtl.SamplerState)(samplers[i])
        
        // Get the resource id from the descriptor
        id = sampler->gpuResourceID()
    }

    heap_samplers := cast([^]mtl.ResourceID)(heap_info.samplers)->contents()

    mem.copy(
        heap_samplers[start_idx:],
        raw_data(sampler_ids),
        int(Bytes_Per_Texture_Slot * len(samplers))
    )
}

_desc_heap_set_bvhs :: proc(heap: Descriptor_Heap, start_idx: u32, bvhs: []BVH, loc := #caller_location)
{
    if ctx.validation
    {
        ok := true
        ok &= pool_check(&ctx.desc_heaps, heap, "heap", loc)
        if !ok do return
    }

    heap_info := pool_get(&ctx.desc_heaps, heap)

    scratch, _ := acquire_scratch()
    /*
    vk_bvhs := make([]vk.AccelerationStructureKHR, len(bvhs), allocator = scratch)
    for i in 0..<len(bvhs) {
        vk_bvhs[i] = pool_get(&ctx.bvhs, bvhs[i]).handle
    }

    write_bvh := vk.WriteDescriptorSetAccelerationStructureKHR {
        sType = .WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR,
        accelerationStructureCount = u32(len(bvhs)),
        pAccelerationStructures = raw_data(vk_bvhs),
    }
    write := vk.WriteDescriptorSet {
        sType = .WRITE_DESCRIPTOR_SET,
        pNext = &write_bvh,
        dstSet = heap_info.bvhs,
        dstBinding = 0,
        dstArrayElement = start_idx,
        descriptorCount = u32(len(bvhs)),
        descriptorType = .ACCELERATION_STRUCTURE_KHR,
    }
    vk.UpdateDescriptorSets(ctx.device, 1, &write, 0, nil)
    */
}

// Shaders
@(private="file")
_shader_create_internal :: proc(code: []u32, is_compute: bool, graphics_stage: Shader_Type_Graphics, entry_point_name := "main", group_size_x: u32 = 1, group_size_y: u32 = 1, group_size_z: u32 = 1, name: string, loc: runtime.Source_Code_Location) -> Shader
{
    // TODO: Implement a better way of handling shader source input for both
    // metal and vulkan backends. 
    // 
    // Right now, this is just a hack to avoid messing with the API boundary
    // and to afford quick testing as a result. I don't want to mess with the
    // API because I don't feel like dealing with a potential merge conflict
    // at this moment in time...
    source_runes := transmute([]rune)code
    source := utf8.runes_to_string(source_runes)
    defer delete(source)

    // TODO: Implement a hash table of sorts to cache libraries coming from
    // the same source code to prevent us compiling the same library code
    // multiple times. Use something like SHA256 for the hash.

    library: ^mtl.Library
    {
        lib_desc: ^mtl.MTL4LibraryDescriptor = (mtl.MTL4LibraryDescriptor{})->alloc()
        defer lib_desc->release()
    
        source_objc := to_mtl_string(source)
        defer source_objc->release()
        
        lib_desc->setSource(source_objc)

        err: ^ns.Error
        library = (ctx.shader_compiler)->newLibraryWithDescriptor_error(lib_desc, &err)

        mtl_ensure(err, "%v shader creation failed during library/source compilation.", "Compute" if is_compute else "Graphics")
    }

    shader: ^mtl.MTL4LibraryFunctionDescriptor = (mtl.MTL4LibraryFunctionDescriptor{})->alloc()

    shader->setLibrary(library)

    objc_entry_point_name := to_mtl_string(entry_point_name)
    defer objc_entry_point_name->release()

    shader->setName(objc_entry_point_name)

    si := Shader_Info {
        handle = shader,
        current_workgroup_size = { group_size_x, group_size_y, group_size_z },
        is_compute = is_compute,
        graphics_type = graphics_stage,
    }

    return pool_add(&ctx.shaders, si, { created_at = loc, name = name })
}

_shader_create :: proc(code: []u32, type: Shader_Type_Graphics, entry_point_name := "main", name := "", loc := #caller_location) -> Shader
{
    return _shader_create_internal(code, false, type, entry_point_name, name = name, loc = loc)
}

_shader_create_compute :: proc(code: []u32, group_size_x: u32, group_size_y: u32 = 1, group_size_z: u32 = 1, entry_point_name := "main", name := "", loc := #caller_location) -> Shader
{
    return _shader_create_internal(code, true, .Vertex, entry_point_name, group_size_x, group_size_y, group_size_z, name = name, loc = loc)
}

_shader_destroy :: proc(shader: Shader, loc := #caller_location)
{
    if ctx.validation
    {
        ok := true
        ok &= pool_check(&ctx.shaders, shader, "shader", loc)
        if !ok do return
    }

    shader_info := pool_get(&ctx.shaders, shader)
    mtl_shader := shader_info.handle
    mtl_shader->release()
    pool_remove(&ctx.shaders, shader)
}

// Semaphores
_semaphore_create :: proc(init_value: u64 = 0, name := "", loc := #caller_location) -> Semaphore
{
    event := (ctx.device)->newSharedEvent()

    event->setSignaledValue(init_value)

    if name != ""
    {
        objc_name := to_mtl_string(name)
        defer objc_name->release()
    
        event->setLabel(objc_name)
    }

    return pool_add(&ctx.semaphores, event, { name = name, created })
}

_semaphore_get_value :: proc(sem: Semaphore, loc := #caller_location) -> u64
{
    if ctx.validation
    {
        ok := true
        ok &= pool_check(&ctx.semaphores, sem, "sem", loc)
        if !ok do return {}
    }

    mtl_sem := pool_get(&ctx.semaphores, sem)

    return u64(mtl_sem->signaledValue())
}

_semaphore_wait :: proc(sem: Semaphore, wait_value: u64, loc := #caller_location)
{
    if ctx.validation
    {
        ok := true
        ok &= pool_check(&ctx.semaphores, sem, "sem", loc)
        if !ok do return
    }

    mtl_sem := pool_get(&ctx.semaphores, sem)

    mtl_sem->waitUntilSignaledValue(wait_value, max(u64))
}

_semaphore_destroy :: proc(sem: Semaphore, loc := #caller_location)
{
    if ctx.validation
    {
        ok := true
        ok &= pool_check(&ctx.semaphores, sem, "sem", loc)
        if !ok do return
    }

    mtl_sem := pool_get(&ctx.semaphores, sem)
    mtl_sem->release()
    pool_remove(&ctx.semaphores, sem)
}

// Raytracing
_blas_size_and_align :: proc(desc: BLAS_Desc, loc := #caller_location) -> (size: u64, align: u64)
{
    // TODO(Raytracing)
    return 0, 0
    // return u64(get_vk_blas_size_info(desc).accelerationStructureSize), 16
}

_blas_create :: proc(desc: BLAS_Desc, storage: gpuptr, name := "", loc := #caller_location) -> BVH
{
    if ctx.validation
    {
        ok := true
        ok &= check_ptr(storage, "storage", loc)
        if !ok do return {}
    }

    // TODO(Raytracing)
    return {}

    /*
    storage_buf, storage_offset, _ := get_buf_offset_from_gpu_ptr(storage)
    size_info := get_vk_blas_size_info(desc)

    bvh_handle: vk.AccelerationStructureKHR
    blas_ci := vk.AccelerationStructureCreateInfoKHR {
        sType = .ACCELERATION_STRUCTURE_CREATE_INFO_KHR,
        buffer = storage_buf,
        offset = vk.DeviceSize(storage_offset),
        size = size_info.accelerationStructureSize,
        type = .BOTTOM_LEVEL,
    }
    vk_check(vk.CreateAccelerationStructureKHR(ctx.device, &blas_ci, nil, &bvh_handle))

    vk_set_debug_name(name, u64(bvh_handle), .ACCELERATION_STRUCTURE_KHR)

    new_desc := desc
    cloned_shapes := slice.clone_to_dynamic(new_desc.shapes)
    new_desc.shapes = cloned_shapes[:]
    bvh_info := BVH_Info {
        handle = bvh_handle,
        mem = storage.ptr,
        is_blas = true,
        shapes = cloned_shapes,
        blas_desc = desc,
    }
    return pool_add(&ctx.bvhs, bvh_info, { created_at = loc, name = name })
    */
}

_blas_build_scratch_buffer_size_and_align :: proc(desc: BLAS_Desc, loc := #caller_location) -> (size: u64, align: u64)
{
    // TODO(Raytracing)
    return 0, 0
    // return u64(get_vk_blas_size_info(desc).buildScratchSize), u64(ctx.physical_properties.bvh_props.minAccelerationStructureScratchOffsetAlignment)
}

_tlas_size_and_align :: proc(desc: TLAS_Desc, loc := #caller_location) -> (size: u64, align: u64)
{
    // TODO(Raytracing)
    return 0, 0
    // return u64(get_vk_tlas_size_info(desc).accelerationStructureSize), 1
}

_tlas_create :: proc(desc: TLAS_Desc, storage: gpuptr, name := "", loc := #caller_location) -> BVH
{
    if ctx.validation
    {
        ok := true
        ok &= check_ptr(storage, "storage", loc)
        if !ok do return {}
    }

    // TODO(Raytracing)
    return {}

    /*
    storage_buf, storage_offset, _ := get_buf_offset_from_gpu_ptr(storage)
    size_info := get_vk_tlas_size_info(desc)

    bvh_handle: vk.AccelerationStructureKHR
    tlas_ci := vk.AccelerationStructureCreateInfoKHR {
        sType = .ACCELERATION_STRUCTURE_CREATE_INFO_KHR,
        buffer = storage_buf,
        offset = vk.DeviceSize(storage_offset),
        size = size_info.accelerationStructureSize,
        type = .TOP_LEVEL,
    }
    vk_check(vk.CreateAccelerationStructureKHR(ctx.device, &tlas_ci, nil, &bvh_handle))

    vk_set_debug_name(name, u64(bvh_handle), .ACCELERATION_STRUCTURE_KHR)

    bvh_info := BVH_Info {
        handle = bvh_handle,
        mem = storage.ptr,
        is_blas = false,
        tlas_desc = desc
    }
    return pool_add(&ctx.bvhs, bvh_info, { created_at = loc, name = name })
    */
}

_tlas_build_scratch_buffer_size_and_align :: proc(desc: TLAS_Desc, loc := #caller_location) -> (size: u64, align: u64)
{
    // TODO(Raytracing)
    return 0, 0
    // return u64(get_vk_tlas_size_info(desc).buildScratchSize), u64(ctx.physical_properties.bvh_props.minAccelerationStructureScratchOffsetAlignment)
}

_bvh_root_ptr :: proc(bvh: BVH, loc := #caller_location) -> rawptr
{
    if ctx.validation
    {
        ok := true
        ok &= pool_check(&ctx.bvhs, bvh, "bvh", loc)
        if !ok do return nil
    }

    // TODO(Raytracing)
    return nil

    /*
    bvh_info := pool_get(&ctx.bvhs, bvh)

    return transmute(rawptr) vk.GetAccelerationStructureDeviceAddressKHR(ctx.device, & {
        sType = .ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR,
        accelerationStructure = bvh_info.handle
    })
    */
}

_bvh_destroy :: proc(bvh: BVH, loc := #caller_location)
{
    if ctx.validation
    {
        ok := true
        ok &= pool_check(&ctx.bvhs, bvh, "bvh", loc)
        if !ok do return
    }

    // TODO(Raytracing)
    /*
    bvh_info := pool_get(&ctx.bvhs, bvh)
    vk.DestroyAccelerationStructureKHR(ctx.device, bvh_info.handle, nil)
    pool_remove(&ctx.bvhs, bvh)
    */
}

/*
@(private="file")
get_vk_blas_size_info :: proc(desc: BLAS_Desc) -> vk.AccelerationStructureBuildSizesInfoKHR
{
    scratch, _ := acquire_scratch()

    primitive_counts := make([]u32, len(desc.shapes), allocator = scratch)
    for shape, i in desc.shapes
    {
        switch s in shape
        {
            case BVH_Mesh_Desc: primitive_counts[i] = s.tri_count
            case BVH_AABB_Desc: primitive_counts[i] = s.aabb_count
        }
    }

    build_info := to_vk_blas_desc(desc, scratch)

    size_info := vk.AccelerationStructureBuildSizesInfoKHR { sType = .ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR }
    vk.GetAccelerationStructureBuildSizesKHR(ctx.device, .DEVICE, &build_info, raw_data(primitive_counts), &size_info)
    return size_info
}

@(private="file")
get_vk_tlas_size_info :: proc(desc: TLAS_Desc) -> vk.AccelerationStructureBuildSizesInfoKHR
{
    scratch, _ := acquire_scratch()

    build_info := to_vk_tlas_desc(desc, scratch)

    size_info := vk.AccelerationStructureBuildSizesInfoKHR { sType = .ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR }
    primitive_count := desc.instance_count
    vk.GetAccelerationStructureBuildSizesKHR(ctx.device, .DEVICE, &build_info, &primitive_count, &size_info)
    return size_info
}
*/

// Command buffer

_queue_wait_idle :: proc(queue: Queue)
{
    sync.guard(&ctx.lock)

    // TODO: Implement a basic semaphore lock here where it increments the
    // counter, and encodes a signal on the queue, and finally waits on
    // the semaphore until that signaled value is present.
    // 
    // h := create_semaphore(0)
    // incr := 0 + 1
    // 
    // queue->signalEvent(h.sem, incr)
    // 
    // h->waitUntilSignaledValue(incr)
    // 
    // Something akin to the above
    // 
    // Then for the entire device, we could do the same thing, but with
    // a loop across the all queues.
    
    tls_ctx := get_tls()

    if sync.guard(&ctx.lock) do vk.QueueWaitIdle(ctx.queues[queue].handle)
}

_commands_begin :: proc(queue: Queue, loc := #caller_location) -> Command_Buffer
{
    cmd_buf := vk_acquire_cmd_buf(queue)
    cmd_buf_info := pool_get(&ctx.command_buffers, cmd_buf)

    cmd_buf_bi := vk.CommandBufferBeginInfo {
        sType = .COMMAND_BUFFER_BEGIN_INFO,
        flags = { .ONE_TIME_SUBMIT },
    }
    vk_cmd_buf := cmd_buf_info.handle
    vk_check(vk.BeginCommandBuffer(vk_cmd_buf, &cmd_buf_bi))

    return cmd_buf
}

_queue_submit :: proc(queue: Queue, cmd_bufs: []Command_Buffer, loc := #caller_location)
{
    if ctx.validation
    {
        ok := true
        for cmd_buf, i in cmd_bufs
        {
            cmd_buf_info := pool_get(&ctx.command_buffers, cmd_buf)
            if cmd_buf_info.queue != queue {
                log.errorf("'queue' does not match the queue associated with 'cmd_bufs[%v]'.", i)
                ok = false
            }

            if cmd_buf_info.thread_id != sync.current_thread_id() {
                log.errorf("Attempting to submit 'cmd_bufs[%v]' on thread '%v' even though it was created on thread '%v'. This is not allowed.",
                           i, sync.current_thread_id(), cmd_buf_info.thread_id)
                ok = false
            }
        }

        // TODO: Check that all wait sems and signal sems are still valid here.

        if !ok do return
    }

    for cmd_buf in cmd_bufs
    {
        cmd_buf_info := pool_get(&ctx.command_buffers, cmd_buf)
        vk_cmd_buf := cmd_buf_info.handle
        vk_check(vk.EndCommandBuffer(vk_cmd_buf))
    }

    vk_submit_cmd_bufs(cmd_bufs)

    for cmd_buf in cmd_bufs {
        clear_cmd_buf(cmd_buf)
    }
}

@(private="file")
clear_cmd_buf :: proc(cmd_buf: Command_Buffer)
{
    cmd_buf_info_mut, r_lock := pool_get_mut(&ctx.command_buffers, cmd_buf); sync.guard(r_lock)
    cmd_buf_info_mut.compute_shader = {}
    cmd_buf_info_mut.recording = false
    clear(&cmd_buf_info_mut.wait_sems)
    clear(&cmd_buf_info_mut.signal_sems)
}

// Commands

_cmd_mem_copy_raw :: proc(cmd_buf: Command_Buffer, dst, src: gpuptr, #any_int bytes: i64, loc := #caller_location)
{
    if ctx.validation
    {
        ok := true
        ok &= pool_check(&ctx.command_buffers, cmd_buf, "cmd_buf", loc)
        if bytes > 0
        {
            ok &= check_ptr_range(dst, bytes, "dst", loc)
            ok &= check_ptr_range(src, bytes, "src", loc)
        }
        if !ok do return
    }

    if bytes == 0 do return

    cmd_buf_info := pool_get(&ctx.command_buffers, cmd_buf)

    src_alloc_impl := transmute(Alloc_Impl_Info) src._impl
    src_alloc_info := pool_get(&ctx.allocs, src_alloc_impl.handle)
    dst_alloc_impl := transmute(Alloc_Impl_Info) dst._impl
    dst_alloc_info := pool_get(&ctx.allocs, dst_alloc_impl.handle)

    src_buf, src_offset, _ := get_buf_offset_from_gpu_ptr(src)
    dst_buf, dst_offset, _ := get_buf_offset_from_gpu_ptr(dst)

    // Clamp copy regions
    to_copy: uintptr
    if uintptr(src_offset) > uintptr(src_alloc_info.buf_size) || uintptr(dst_offset) > uintptr(dst_alloc_info.buf_size) {
        to_copy = 0
    } else {
        to_copy = min(uintptr(bytes), min(uintptr(src_alloc_info.buf_size) - uintptr(src_offset), uintptr(dst_alloc_info.buf_size) - uintptr(dst_offset)))
    }

    if to_copy > 0
    {
        copy_regions := []vk.BufferCopy {
            {
                srcOffset = vk.DeviceSize(src_offset),
                dstOffset = vk.DeviceSize(dst_offset),
                size = vk.DeviceSize(to_copy),
            }
        }
        vk.CmdCopyBuffer(cmd_buf_info.handle, src_buf, dst_buf, u32(len(copy_regions)), raw_data(copy_regions))
    }
}

_cmd_copy_to_texture :: proc(cmd_buf: Command_Buffer, dst: Texture, src: gpuptr, region: Texture_Region = {}, loc := #caller_location)
{
    if ctx.validation
    {
        ok := true
        ok &= pool_check(&ctx.command_buffers, cmd_buf, "cmd_buf", loc)
        ok &= pool_check(&ctx.textures, dst.handle, "dst", loc)
        ok &= check_ptr(src, "src", loc)
        if !ok do return
    }

    cmd_buf_info := pool_get(&ctx.command_buffers, cmd_buf)
    tex_info := pool_get(&ctx.textures, dst.handle)

    src_buf, src_offset, ok_s := get_buf_offset_from_gpu_ptr(src)
    assert(ok_s)

    plane_aspect := to_vk_image_aspect_flags(dst.format)
    is_compressed := is_block_compressed(dst.format)

    mip_width := max(1, dst.dimensions.x >> region.mip_level)
    mip_height := max(1, dst.dimensions.y >> region.mip_level)
    mip_depth := max(1, dst.dimensions.z >> region.mip_level)

    copy := vk.BufferImageCopy{
        bufferOffset = vk.DeviceSize(src_offset),
        bufferRowLength = 0 if is_compressed else mip_width,
        bufferImageHeight = 0 if is_compressed else mip_height,
        imageSubresource = {
            aspectMask = plane_aspect,
            mipLevel = region.mip_level,
            baseArrayLayer = region.base_layer,
            layerCount = max(1, region.layer_count),
        },
        imageOffset = {},
        imageExtent = { mip_width, mip_height, mip_depth },
    }

    vk.CmdCopyBufferToImage(cmd_buf_info.handle, src_buf, tex_info.handle, .GENERAL, 1, &copy)
}

// TODO: Missing: cmd_copy_from_texture

_cmd_blit_texture :: proc(cmd_buf: Command_Buffer, dst: Texture, dst_rect: Blit_Rect, src: Texture, src_rect: Blit_Rect, filter: Filter, loc := #caller_location)
{
    if ctx.validation
    {
        ok := true
        ok &= pool_check(&ctx.command_buffers, cmd_buf, "cmd_buf", loc)
        ok &= check_cmd_buf_must_be_graphics(cmd_buf, "cmd_buf", loc)
        if !ok do return
    }

    cmd_buf_info := pool_get(&ctx.command_buffers, cmd_buf)
    src_info := pool_get(&ctx.textures, src.handle)
    dst_info := pool_get(&ctx.textures, dst.handle)

    vk_filter := to_vk_filter(filter)

    src_dimensions := [3]i32 { i32(src.dimensions.x), i32(src.dimensions.y), i32(src.dimensions.z) }
    dst_dimensions := [3]i32 { i32(dst.dimensions.x), i32(dst.dimensions.y), i32(dst.dimensions.z) }

    src_offsets := [2][3]i32 { src_rect.offset_a, src_rect.offset_b }
    if src_offsets == ([2][3]i32 { { 0, 0, 0 }, { 0, 0, 0 } }) {
        src_offsets[1] = get_mip_dimensions_i32(src_dimensions, src_rect.mip_level)
    }

    dst_offsets := [2][3]i32 { dst_rect.offset_a, dst_rect.offset_b }
    if dst_offsets == ([2][3]i32 { { 0, 0, 0 }, { 0, 0, 0 } }) {
        dst_offsets[1] = get_mip_dimensions_i32(dst_dimensions, dst_rect.mip_level)
    }

    region := vk.ImageBlit {
        srcSubresource = {
            aspectMask = { .COLOR },
            mipLevel = src_rect.mip_level,
            baseArrayLayer = src_rect.base_layer,
            layerCount = src_rect.layer_count if src_rect.layer_count > 0 else 1,  // TODO
        },
        srcOffsets = {
            { src_offsets[0].x, src_offsets[0].y, src_offsets[0].z },
            { src_offsets[1].x, src_offsets[1].y, src_offsets[1].z },
        },
        dstSubresource = {
            aspectMask = { .COLOR },
            mipLevel = dst_rect.mip_level,
            baseArrayLayer = dst_rect.base_layer,
            layerCount = dst_rect.layer_count if dst_rect.layer_count > 0 else 1,  // TODO
        },
        dstOffsets = {
            { dst_offsets[0].x, dst_offsets[0].y, dst_offsets[0].z },
            { dst_offsets[1].x, dst_offsets[1].y, dst_offsets[1].z },
        }
    }

    vk.CmdBlitImage(cmd_buf_info.handle, src_info.handle, .GENERAL, dst_info.handle, .GENERAL, 1, &region, vk_filter)
}

_cmd_set_desc_heap :: proc(cmd_buf: Command_Buffer, heap: Descriptor_Heap, loc := #caller_location)
{
    if ctx.validation
    {
        ok := true
        ok &= pool_check(&ctx.command_buffers, cmd_buf, "cmd_buf", loc)
        ok &= pool_check(&ctx.desc_heaps, heap, "heap", loc)
        if !ok do return
    }

    cmd_buf_info := pool_get(&ctx.command_buffers, cmd_buf)
    vk_cmd_buf := cmd_buf_info.handle

    heap_info := pool_get(&ctx.desc_heaps, heap)

    sets := [4]vk.DescriptorSet {
        heap_info.textures,
        heap_info.textures_rw,
        heap_info.samplers,
        heap_info.bvhs,
    }
    vk.CmdBindDescriptorSets(vk_cmd_buf, .GRAPHICS, ctx.common_pipeline_layout_graphics, 0, u32(len(ctx.desc_layouts)), &sets[0], 0, nil)
    vk.CmdBindDescriptorSets(vk_cmd_buf, .COMPUTE, ctx.common_pipeline_layout_compute, 0, u32(len(ctx.desc_layouts)), &sets[0], 0, nil)
}

_cmd_add_wait_semaphore :: proc(cmd_buf: Command_Buffer, sem: Semaphore, wait_value: u64, loc := #caller_location)
{
    if ctx.validation
    {
        ok := true
        ok &= pool_check(&ctx.command_buffers, cmd_buf, "cmd_buf", loc)
        ok &= check_cmd_buf_must_be_recording(cmd_buf, "cmd_buf", loc)
        ok &= pool_check(&ctx.semaphores, sem, "sem", loc)
        if !ok do return
    }

    cmd_buf_info, r_lock := pool_get_mut(&ctx.command_buffers, cmd_buf); sync.guard(r_lock)
    append(&cmd_buf_info.wait_sems, Semaphore_Value { sem = sem, val = wait_value })
}

_cmd_add_signal_semaphore :: proc(cmd_buf: Command_Buffer, sem: Semaphore, signal_value: u64, loc := #caller_location)
{
    if ctx.validation
    {
        ok := true
        ok &= pool_check(&ctx.command_buffers, cmd_buf, "cmd_buf", loc)
        ok &= check_cmd_buf_must_be_recording(cmd_buf, "cmd_buf", loc)
        ok &= pool_check(&ctx.semaphores, sem, "sem", loc)
        if !ok do return
    }

    cmd_buf_info, r_lock := pool_get_mut(&ctx.command_buffers, cmd_buf); sync.guard(r_lock)
    append(&cmd_buf_info.signal_sems, Semaphore_Value { sem = sem, val = signal_value })
}

_cmd_barrier :: proc(cmd_buf: Command_Buffer, before: Stage, after: Stage, hazards: Hazard_Flags = {}, loc := #caller_location)
{
    if ctx.validation
    {
        ok := true
        ok &= pool_check(&ctx.command_buffers, cmd_buf, "cmd_buf", loc)
        if !ok do return
    }

    cmd_buf := pool_get(&ctx.command_buffers, cmd_buf)

    vk_cmd_buf := cmd_buf.handle

    vk_before := to_vk_stage(before)
    vk_after  := to_vk_stage(after)

    // Determine access masks based on hazards
    src_access: vk.AccessFlags
    dst_access: vk.AccessFlags

    if .Draw_Arguments in hazards
    {
        // When compute shader writes draw arguments, ensure they're visible to indirect draw commands
        // Source: compute shader writes
        src_access += { .SHADER_WRITE }
        // Destination: indirect command read (for draw/dispatch indirect)
        dst_access += { .INDIRECT_COMMAND_READ }
    }
    if .Descriptors in hazards
    {
        // When descriptors are updated, ensure visibility
        src_access += { .SHADER_WRITE }
        dst_access += { .SHADER_READ }
    }
    if .Depth_Stencil in hazards
    {
        // Depth/stencil attachment synchronization
        src_access += { .DEPTH_STENCIL_ATTACHMENT_WRITE }
        dst_access += { .DEPTH_STENCIL_ATTACHMENT_READ, .DEPTH_STENCIL_ATTACHMENT_WRITE }
    }
    if .BVHs in hazards
    {
        src_access += { .ACCELERATION_STRUCTURE_WRITE_KHR }
        dst_access += { .ACCELERATION_STRUCTURE_READ_KHR }
    }

    // If no specific hazards, use generic memory barrier
    if card(hazards) == 0
    {
        src_access = { .MEMORY_WRITE }
        dst_access = { .MEMORY_READ, .MEMORY_WRITE }
    }

    barrier := vk.MemoryBarrier {
        sType = .MEMORY_BARRIER,
        srcAccessMask = src_access,
        dstAccessMask = dst_access,
    }
    vk.CmdPipelineBarrier(vk_cmd_buf, vk_before, vk_after, {}, 1, &barrier, 0, nil, 0, nil)
}

_cmd_set_shaders :: proc(cmd_buf: Command_Buffer, vert_shader: Shader, frag_shader: Shader, loc := #caller_location)
{
    if ctx.validation
    {
        ok := true
        ok &= pool_check(&ctx.command_buffers, cmd_buf, "cmd_buf", loc)
        ok &= pool_check(&ctx.shaders, vert_shader, "vert_shader", loc)
        ok &= pool_check(&ctx.shaders, frag_shader, "frag_shader", loc)
        if !ok do return
    }

    cmd_buf := pool_get(&ctx.command_buffers, cmd_buf)
    vert_shader := pool_get(&ctx.shaders, vert_shader)
    frag_shader := pool_get(&ctx.shaders, frag_shader)

    vk_cmd_buf := cmd_buf.handle
    vk_vert_shader := vert_shader.handle
    vk_frag_shader := frag_shader.handle

    shader_stages := []vk.ShaderStageFlags { { .VERTEX }, { .FRAGMENT } }
    to_bind := []vk.ShaderEXT { vk_vert_shader, vk_frag_shader }
    assert(len(shader_stages) == len(to_bind))
    vk.CmdBindShadersEXT(vk_cmd_buf, u32(len(shader_stages)), raw_data(shader_stages), raw_data(to_bind))
}

_cmd_set_depth_state :: proc(cmd_buf: Command_Buffer, state: Depth_State, loc := #caller_location)
{
    if ctx.validation
    {
        ok := true
        ok &= pool_check(&ctx.command_buffers, cmd_buf, "cmd_buf", loc)
        if !ok do return
    }

    cmd_buf := pool_get(&ctx.command_buffers, cmd_buf)

    vk_cmd_buf := cmd_buf.handle

    vk.CmdSetDepthCompareOp(vk_cmd_buf, to_vk_compare_op(state.compare))
    vk.CmdSetDepthTestEnable(vk_cmd_buf, .Read in state.mode)
    vk.CmdSetDepthWriteEnable(vk_cmd_buf, .Write in state.mode)
    vk.CmdSetDepthBiasEnable(vk_cmd_buf, false)
    vk.CmdSetDepthClipEnableEXT(vk_cmd_buf, true)
    vk.CmdSetStencilTestEnable(vk_cmd_buf, false)
}

_cmd_set_raster_state :: proc(cmd_buf: Command_Buffer, state: Raster_State, loc := #caller_location)
{
    if ctx.validation
    {
        ok := true
        ok &= pool_check(&ctx.command_buffers, cmd_buf, "cmd_buf", loc)
        if !ok do return
    }

    cmd_buf := pool_get(&ctx.command_buffers, cmd_buf)
    vk_cmd_buf := cmd_buf.handle

    vk.CmdSetPrimitiveTopology(vk_cmd_buf, to_vk_topology(state.topology))
    vk.CmdSetCullMode(vk_cmd_buf, to_vk_cull_mode(state.cull_mode))
    vk.CmdSetAlphaToCoverageEnableEXT(vk_cmd_buf, b32(state.alpha_to_coverage))
}

_cmd_set_blend_state :: proc(cmd_buf: Command_Buffer, state: Blend_State, loc := #caller_location)
{
    if ctx.validation
    {
        ok := true
        ok &= pool_check(&ctx.command_buffers, cmd_buf, "cmd_buf", loc)
        if !ok do return
    }

    cmd_buf := pool_get(&ctx.command_buffers, cmd_buf)
    vk_cmd_buf := cmd_buf.handle

    enable_b32 := b32(state.enable)
    vk.CmdSetColorBlendEnableEXT(vk_cmd_buf, 0, 1, &enable_b32)

    vk.CmdSetColorBlendEquationEXT(vk_cmd_buf, 0, 1, &vk.ColorBlendEquationEXT {
        srcColorBlendFactor = to_vk_blend_factor(state.src_color_factor),
        dstColorBlendFactor = to_vk_blend_factor(state.dst_color_factor),
        colorBlendOp        = to_vk_blend_op(state.color_op),
        srcAlphaBlendFactor = to_vk_blend_factor(state.src_alpha_factor),
        dstAlphaBlendFactor = to_vk_blend_factor(state.dst_alpha_factor),
        alphaBlendOp        = to_vk_blend_op(state.alpha_op),
    })

    color_write_mask := transmute(vk.ColorComponentFlags) cast(u32) transmute(u8) state.color_write_mask
    vk.CmdSetColorWriteMaskEXT(vk_cmd_buf, 0, 1, &color_write_mask)
}

_cmd_set_viewport :: proc(cmd_buf: Command_Buffer, viewport: Viewport, loc := #caller_location)
{
    if ctx.validation
    {
        ok := true
        ok &= pool_check(&ctx.command_buffers, cmd_buf, "cmd_buf", loc)
        if viewport.size.x <= 0 || viewport.size.y <= 0 {
            log.error("Viewport width and height must be > 0.", location = loc)
            ok = false
        }
        if !ok do return
    }

    vk_cmd_buf := pool_get(&ctx.command_buffers, cmd_buf).handle
    vk_viewport := to_vk_viewport(viewport)
    vk.CmdSetViewportWithCount(vk_cmd_buf, 1, &vk_viewport)
}

_cmd_set_scissor :: proc(cmd_buf: Command_Buffer, scissor: Rect_2D, loc := #caller_location)
{
    if ctx.validation
    {
        ok := true
        ok &= pool_check(&ctx.command_buffers, cmd_buf, "cmd_buf", loc)
        if !ok do return
    }

    vk_cmd_buf := pool_get(&ctx.command_buffers, cmd_buf).handle
    vk_scissor := to_vk_rect_2D(scissor)
    vk.CmdSetScissorWithCount(vk_cmd_buf, 1, &vk_scissor)
}

_cmd_set_compute_shader :: proc(cmd_buf: Command_Buffer, compute_shader: Shader, loc := #caller_location)
{
    if ctx.validation
    {
        ok := true
        ok &= pool_check(&ctx.command_buffers, cmd_buf, "cmd_buf", loc)
        ok &= pool_check(&ctx.shaders, compute_shader, "compute_shader", loc)
        if !ok do return
    }

    shader_info := pool_get(&ctx.shaders, compute_shader)
    vk_shader_info := shader_info.handle

    cmd_buf_info, r_lock := pool_get_mut(&ctx.command_buffers, cmd_buf); sync.guard(r_lock)
    vk_cmd_buf := cmd_buf_info.handle

    shader_stages := []vk.ShaderStageFlags { { .COMPUTE } }
    to_bind := []vk.ShaderEXT { vk_shader_info }
    assert(len(shader_stages) == len(to_bind))
    vk.CmdBindShadersEXT(vk_cmd_buf, u32(len(shader_stages)), raw_data(shader_stages), raw_data(to_bind))

    cmd_buf_info.compute_shader = compute_shader
}

_cmd_dispatch :: proc(cmd_buf: Command_Buffer, compute_data: gpuptr, num_groups_x: u32, num_groups_y: u32 = 1, num_groups_z: u32 = 1, loc := #caller_location)
{
    if ctx.validation
    {
        ok := true
        ok &= pool_check(&ctx.command_buffers, cmd_buf, "cmd_buf", loc)
        ok &= check_ptr(compute_data, "compute_data", loc)
        ok &= check_cmd_buf_has_compute_shader_set(cmd_buf, "cmd_buf", loc)
        if !ok do return
    }

    cmd_buf_info := pool_get(&ctx.command_buffers, cmd_buf)
    vk_cmd_buf := cmd_buf_info.handle

    push_constants := Compute_Shader_Push_Constants {
        compute_data = compute_data.ptr,
    }

    vk.CmdPushConstants(vk_cmd_buf, ctx.common_pipeline_layout_compute, { .COMPUTE }, 0, size_of(Compute_Shader_Push_Constants), &push_constants)

    vk.CmdDispatch(vk_cmd_buf, num_groups_x, num_groups_y, num_groups_z)
}

_cmd_dispatch_indirect_raw :: proc(cmd_buf: Command_Buffer, compute_data, arguments: gpuptr, loc := #caller_location)
{
    if ctx.validation
    {
        ok := true
        ok &= pool_check(&ctx.command_buffers, cmd_buf, "cmd_buf", loc)
        ok &= check_ptr(compute_data, "compute_data", loc)
        ok &= check_ptr(arguments, "arguments", loc)
        ok &= check_cmd_buf_has_compute_shader_set(cmd_buf, "cmd_buf", loc)
        if !ok do return
    }

    cmd_buf_info := pool_get(&ctx.command_buffers, cmd_buf)
    vk_cmd_buf := cmd_buf_info.handle

    arguments_buf, arguments_offset, ok_a := get_buf_offset_from_gpu_ptr(arguments)
    assert(ok_a)

    push_constants := Compute_Shader_Push_Constants {
        compute_data = compute_data.ptr,
    }

    vk.CmdPushConstants(vk_cmd_buf, ctx.common_pipeline_layout_compute, { .COMPUTE }, 0, size_of(Compute_Shader_Push_Constants), &push_constants)

    vk.CmdDispatchIndirect(vk_cmd_buf, arguments_buf, vk.DeviceSize(arguments_offset))
}

_cmd_begin_render_pass :: proc(cmd_buf: Command_Buffer, desc: Render_Pass_Desc, loc := #caller_location)
{
    if ctx.validation
    {
        ok := true
        ok &= pool_check(&ctx.command_buffers, cmd_buf, "cmd_buf", loc)
        ok &= check_cmd_buf_must_be_graphics(cmd_buf, "cmd_buf", loc)
        if !ok do return
    }

    cmd_buf_info := pool_get(&ctx.command_buffers, cmd_buf)
    vk_cmd_buf := cmd_buf_info.handle

    scratch, _ := acquire_scratch()

    // Compute sample count
    sample_count := u32(1)
    {
        for attachment in desc.color_attachments {
            sample_count = max(sample_count, attachment.texture.sample_count)
        }
        if desc.depth_attachment != nil {
            sample_count = max(sample_count, desc.depth_attachment.?.texture.sample_count)
        }
    }

    vk_color_attachments := make([]vk.RenderingAttachmentInfo, len(desc.color_attachments), allocator = scratch)
    for &vk_attach, i in vk_color_attachments {
        vk_attach = to_vk_render_attachment(desc.color_attachments[i])
    }

    vk_depth_attachment: vk.RenderingAttachmentInfo
    vk_depth_attachment_ptr: ^vk.RenderingAttachmentInfo
    if desc.depth_attachment != nil
    {
        vk_depth_attachment = to_vk_render_attachment(desc.depth_attachment.?)
        vk_depth_attachment_ptr = &vk_depth_attachment
    }

    width := desc.render_area_size.x
    if width == {} {
        width = desc.color_attachments[0].texture.dimensions.x
    }
    height := desc.render_area_size.y
    if height == {} {
        height = desc.color_attachments[0].texture.dimensions.y
    }
    layer_count := desc.layer_count
    if layer_count == 0 {
        layer_count = 1
    }

    rendering_info := vk.RenderingInfo {
        sType = .RENDERING_INFO,
        renderArea = {
            offset = { desc.render_area_offset.x, desc.render_area_offset.y },
            extent = { width, height }
        },
        layerCount = layer_count,
        colorAttachmentCount = u32(len(vk_color_attachments)),
        pColorAttachments = raw_data(vk_color_attachments),
        pDepthAttachment = vk_depth_attachment_ptr,
    }
    vk.CmdBeginRendering(vk_cmd_buf, &rendering_info)

    // Blend state
    vk.CmdSetStencilTestEnable(vk_cmd_buf, false)
    color_attachment_count := u32(len(vk_color_attachments))
    if color_attachment_count > 0 {
        // Set blend enable for all attachments
        blend_enables := make([]b32, color_attachment_count, allocator = scratch)
        for i in 0 ..< color_attachment_count {
            blend_enables[i] = false
        }
        vk.CmdSetColorBlendEnableEXT(vk_cmd_buf, 0, color_attachment_count, raw_data(blend_enables))

        // Set color write mask for all attachments
        color_mask := vk.ColorComponentFlags { .R, .G, .B, .A }
        color_masks := make([]vk.ColorComponentFlags, color_attachment_count, allocator = scratch)
        for i in 0 ..< color_attachment_count {
            color_masks[i] = color_mask
        }
        vk.CmdSetColorWriteMaskEXT(vk_cmd_buf, 0, color_attachment_count, raw_data(color_masks))
    }

    // Raster state
    vk.CmdSetRasterizationSamplesEXT(vk_cmd_buf, to_vk_sample_count(sample_count))
    vk.CmdSetPrimitiveTopology(vk_cmd_buf, .TRIANGLE_LIST)
    vk.CmdSetPolygonModeEXT(vk_cmd_buf, .FILL)
    vk.CmdSetCullMode(vk_cmd_buf, { .BACK })
    vk.CmdSetFrontFace(vk_cmd_buf, .COUNTER_CLOCKWISE)

    // Depth state
    vk.CmdSetDepthCompareOp(vk_cmd_buf, .LESS)
    vk.CmdSetDepthTestEnable(vk_cmd_buf, false)
    vk.CmdSetDepthWriteEnable(vk_cmd_buf, false)
    vk.CmdSetDepthBiasEnable(vk_cmd_buf, false)
    vk.CmdSetDepthClipEnableEXT(vk_cmd_buf, true)

    // Viewport
    viewport := vk.Viewport {
        x = 0, y = 0,
        width = f32(width), height = f32(height),
        minDepth = 0.0, maxDepth = 1.0,
    }
    vk.CmdSetViewportWithCount(vk_cmd_buf, 1, &viewport)
    scissor := vk.Rect2D {
        offset = {
            x = 0, y = 0
        },
        extent = {
            width = width, height = height,
        }
    }
    vk.CmdSetScissorWithCount(vk_cmd_buf, 1, &scissor)
    vk.CmdSetRasterizerDiscardEnable(vk_cmd_buf, false)
    vk.CmdSetColorBlendEquationEXT(vk_cmd_buf, 0, 1, &vk.ColorBlendEquationEXT {
        srcColorBlendFactor = .SRC_ALPHA,
        dstColorBlendFactor = .ONE_MINUS_SRC_ALPHA,
        colorBlendOp        = .ADD,
        srcAlphaBlendFactor = .ONE,
        dstAlphaBlendFactor = .ONE_MINUS_SRC_ALPHA,
        alphaBlendOp        = .ADD,
    })

    // Unused
    vk.CmdSetVertexInputEXT(vk_cmd_buf, 0, nil, 0, nil)
    vk.CmdSetPrimitiveRestartEnable(vk_cmd_buf, false)

    sample_mask := vk.SampleMask(0xFF)
    vk.CmdSetSampleMaskEXT(vk_cmd_buf, to_vk_sample_count(sample_count), &sample_mask)
    vk.CmdSetAlphaToCoverageEnableEXT(vk_cmd_buf, false)
}

_cmd_end_render_pass :: proc(cmd_buf: Command_Buffer, loc := #caller_location)
{
    if ctx.validation
    {
        ok := true
        ok &= pool_check(&ctx.command_buffers, cmd_buf, "cmd_buf", loc)
        if !ok do return
    }

    cmd_buf := pool_get(&ctx.command_buffers, cmd_buf)
    vk_cmd_buf := cmd_buf.handle
    vk.CmdEndRendering(vk_cmd_buf)
}

_cmd_draw :: proc(cmd_buf: Command_Buffer, vertex_data, fragment_data: gpuptr,
                  vertex_count: u32, instance_count: u32 = 1, loc := #caller_location)
{
    if ctx.validation
    {
        ok := true
        ok &= pool_check(&ctx.command_buffers, cmd_buf, "cmd_buf", loc)
        ok &= check_ptr_allow_nil(vertex_data, "vertex_data", loc)
        ok &= check_ptr_allow_nil(fragment_data, "fragment_data", loc)
        if !ok do return
    }

    cmd_buf := pool_get(&ctx.command_buffers, cmd_buf)
    vk_cmd_buf := cmd_buf.handle

    push_constants := Graphics_Shader_Push_Constants {
        vert_data = vertex_data.ptr,
        frag_data = fragment_data.ptr,
        indirect_data = nil,
    }
    vk.CmdPushConstants(vk_cmd_buf, ctx.common_pipeline_layout_graphics, { .VERTEX, .FRAGMENT }, 0, size_of(Graphics_Shader_Push_Constants), &push_constants)

    vk.CmdDraw(vk_cmd_buf, vertex_count, instance_count, 0, 0)
}

_cmd_draw_indexed_raw :: proc(cmd_buf: Command_Buffer, vertex_data, fragment_data, indices: gpuptr,
                              index_format: Index_Format, index_count: u32, instance_count: u32 = 1, loc := #caller_location)
{

    if ctx.validation
    {
        index_size: u32 = 4 if index_format == .U32 else 2
        ok := true
        ok &= pool_check(&ctx.command_buffers, cmd_buf, "cmd_buf", loc)
        ok &= check_ptr_allow_nil(vertex_data, "vertex_data", loc)
        ok &= check_ptr_allow_nil(fragment_data, "fragment_data", loc)
        ok &= check_ptr_range(indices, index_size * index_count, "indices", loc)
        if !ok do return
    }

    cmd_buf := pool_get(&ctx.command_buffers, cmd_buf)
    vk_cmd_buf := cmd_buf.handle

    indices_buf, indices_offset, ok_i := get_buf_offset_from_gpu_ptr(indices)
    assert(ok_i)

    push_constants := Graphics_Shader_Push_Constants {
        vert_data = vertex_data.ptr,
        frag_data = fragment_data.ptr,
        indirect_data = nil,
    }
    vk.CmdPushConstants(vk_cmd_buf, ctx.common_pipeline_layout_graphics, { .VERTEX, .FRAGMENT }, 0, size_of(Graphics_Shader_Push_Constants), &push_constants)

    vk.CmdBindIndexBuffer(vk_cmd_buf, indices_buf, vk.DeviceSize(indices_offset), to_vk_index_format(index_format))
    vk.CmdDrawIndexed(vk_cmd_buf, index_count, instance_count, 0, 0, 0)
}

_cmd_draw_indexed_indirect_raw :: proc(cmd_buf: Command_Buffer, vertex_data, fragment_data, indices: gpuptr, index_format: Index_Format, indirect_arguments: gpuptr, loc := #caller_location)
{
    if ctx.validation
    {
        ok := true
        ok &= pool_check(&ctx.command_buffers, cmd_buf, "cmd_buf", loc)
        ok &= check_ptr_allow_nil(vertex_data, "vertex_data", loc)
        ok &= check_ptr_allow_nil(fragment_data, "fragment_data", loc)
        ok &= check_ptr_allow_nil(indices, "indices", loc)
        ok &= check_ptr(indirect_arguments, "indirect_arguments", loc)
        if !ok do return
    }

    cmd_buf_info := pool_get(&ctx.command_buffers, cmd_buf)
    vk_cmd_buf := cmd_buf_info.handle

    indices_buf, indices_offset, _ := get_buf_offset_from_gpu_ptr(indices)
    arguments_buf, arguments_offset, _ := get_buf_offset_from_gpu_ptr(indirect_arguments)

    push_constants := Graphics_Shader_Push_Constants {
        vert_data = vertex_data.ptr,
        frag_data = fragment_data.ptr,
        indirect_data = indirect_arguments.ptr,
    }
    vk.CmdPushConstants(vk_cmd_buf, ctx.common_pipeline_layout_graphics, { .VERTEX, .FRAGMENT }, 0, size_of(Graphics_Shader_Push_Constants), &push_constants)

    vk.CmdBindIndexBuffer(vk_cmd_buf, indices_buf, vk.DeviceSize(indices_offset), to_vk_index_format(index_format))
    vk.CmdDrawIndexedIndirect(vk_cmd_buf, arguments_buf, vk.DeviceSize(arguments_offset), 1, 0)
}

_cmd_draw_indexed_indirect_multi_raw :: proc(cmd_buf: Command_Buffer, vertex_data, fragment_data, indices: gpuptr,
                                             index_format: Index_Format, indirect_arguments: gpuptr, stride: u32, draw_count: gpuptr, loc := #caller_location)
{
    if ctx.validation
    {
        ok := true
        ok &= pool_check(&ctx.command_buffers, cmd_buf, "cmd_buf", loc)
        ok &= check_ptr_allow_nil(vertex_data, "vertex_data", loc)
        ok &= check_ptr_allow_nil(fragment_data, "fragment_data", loc)
        ok &= check_ptr_allow_nil(indices, "indices", loc)
        ok &= check_ptr(indirect_arguments, "indirect_arguments", loc)
        ok &= check_ptr(draw_count, "draw_count", loc)
        if !ok do return
    }

    cmd_buf := pool_get(&ctx.command_buffers, cmd_buf)

    vk_cmd_buf := cmd_buf.handle

    indices_buf, indices_offset, _ := get_buf_offset_from_gpu_ptr(indices)
    arguments_buf, arguments_offset, _ := get_buf_offset_from_gpu_ptr(indirect_arguments)
    draw_count_buf, draw_count_offset, _ := get_buf_offset_from_gpu_ptr(draw_count)

    // vertex_data and fragment_data are shared data for vertex and fragment shaders
    // indirect_arguments points to the unified indirect data array containing both command and user data
    // The stride is the size of the combined struct { IndirectDrawCommand cmd; UserData data; }
    push_constants := Graphics_Shader_Push_Constants {
        vert_data = vertex_data.ptr,
        frag_data = fragment_data.ptr,
        indirect_data = indirect_arguments.ptr,
    }
    vk.CmdPushConstants(vk_cmd_buf, ctx.common_pipeline_layout_graphics, { .VERTEX, .FRAGMENT }, 0, size_of(Graphics_Shader_Push_Constants), &push_constants)

    vk.CmdBindIndexBuffer(vk_cmd_buf, indices_buf, vk.DeviceSize(indices_offset), to_vk_index_format(index_format))

    max_draw_count := max(u32)
    buf_size, ok_size := get_buf_size_from_gpu_ptr(indirect_arguments)
    if ok_size && buf_size > vk.DeviceSize(arguments_offset)
    {
        available_size := buf_size - vk.DeviceSize(arguments_offset)
        max_draw_count = u32(available_size / vk.DeviceSize(stride))
    }

    vk.CmdDrawIndexedIndirectCount(vk_cmd_buf, arguments_buf, vk.DeviceSize(arguments_offset), draw_count_buf, vk.DeviceSize(draw_count_offset), max_draw_count, stride)
}

_cmd_build_blas :: proc(cmd_buf: Command_Buffer, bvh: BVH, scratch_storage: gpuptr, shapes: []BVH_Shape, loc := #caller_location)
{
    if ctx.validation
    {
        ok := true
        ok &= pool_check(&ctx.command_buffers, cmd_buf, "cmd_buf", loc)
        ok &= check_ptr(scratch_storage, "scratch_storage", loc)
        ok &= check_bvh_must_be_blas(bvh, "bvh", loc)
        if !ok do return
    }

    cmd_buf_info := pool_get(&ctx.command_buffers, cmd_buf)
    vk_cmd_buf := cmd_buf_info.handle
    bvh_info := pool_get(&ctx.bvhs, bvh)

    if len(shapes) != len(bvh_info.blas_desc.shapes)
    {
        log.error("Length used in the shapes argument and length used in the shapes supplied during the creation of this BVH don't match.")
        return
    }

    // TODO: Check for mismatching types.
    /*
    for shape, i in shapes
    {
        switch s in shape
        {
            case BVH_Mesh: {}
            case BVH_AABBs: {}
        }
    }
    */

    scratch, _ := acquire_scratch()

    build_info := to_vk_blas_desc(bvh_info.blas_desc, arena = scratch)
    build_info.dstAccelerationStructure = bvh_info.handle
    build_info.scratchData.deviceAddress = transmute(vk.DeviceAddress) scratch_storage.ptr
    assert(u32(len(shapes)) == build_info.geometryCount)

    range_infos := make([]vk.AccelerationStructureBuildRangeInfoKHR, len(shapes), allocator = scratch)

    // Fill in actual data in shapes
    for i in 0..<build_info.geometryCount
    {
        range_infos[i] = {
            // primitiveCount = primitive_count,
            primitiveOffset = 0,
            firstVertex = 0,
            transformOffset = 0,
        }

        geom := &build_info.pGeometries[i]
        switch s in shapes[i]
        {
            case BVH_Mesh:
            {
                geom.geometry.triangles.vertexData.deviceAddress = transmute(vk.DeviceAddress) s.verts
                geom.geometry.triangles.indexData.deviceAddress = transmute(vk.DeviceAddress) s.indices
                range_infos[i].primitiveCount = bvh_info.blas_desc.shapes[i].(BVH_Mesh_Desc).tri_count
            }
            case BVH_AABBs:
            {
                geom.geometry.aabbs.data.deviceAddress = transmute(vk.DeviceAddress) s.data
            }
        }
    }

    // Vulkan expects an array of pointers (to arrays), one pointer per BVH to build.
    // We always build one at a time, so we only need a pointer to an array (double pointer).
    range_infos_ptr := raw_data(range_infos)
    vk.CmdBuildAccelerationStructuresKHR(vk_cmd_buf, 1, &build_info, &range_infos_ptr)
}

_cmd_build_tlas :: proc(cmd_buf: Command_Buffer, bvh: BVH, scratch_storage, instances: gpuptr, loc := #caller_location)
{
    if ctx.validation
    {
        ok := true
        ok &= pool_check(&ctx.command_buffers, cmd_buf, "cmd_buf", loc)
        ok &= check_ptr(scratch_storage, "scratch_storage", loc)
        ok &= check_ptr(instances, "instances", loc)
        ok &= check_bvh_must_be_tlas(bvh, "bvh", loc)
        if !ok do return
    }

    cmd_buf_info := pool_get(&ctx.command_buffers, cmd_buf)
    bvh_info := pool_get(&ctx.bvhs, bvh)
    vk_cmd_buf := cmd_buf_info.handle

    scratch, _ := acquire_scratch()

    build_info := to_vk_tlas_desc(bvh_info.tlas_desc, arena = scratch)
    build_info.dstAccelerationStructure = bvh_info.handle
    build_info.scratchData.deviceAddress = transmute(vk.DeviceAddress) scratch_storage.ptr
    assert(build_info.geometryCount == 1)

    // Fill in actual data
    build_info.pGeometries[0].geometry.instances.data.deviceAddress = transmute(vk.DeviceAddress) instances.ptr

    // Vulkan expects an array of pointers (to arrays), one pointer per BVH to build.
    // We always build one at a time, and a TLAS always has only one geometry.
    range_info := []vk.AccelerationStructureBuildRangeInfoKHR {
        {
            primitiveCount = bvh_info.tlas_desc.instance_count
        }
    }
    range_info_ptr := raw_data(range_info)
    vk.CmdBuildAccelerationStructuresKHR(vk_cmd_buf, 1, &build_info, &range_info_ptr)
}

_cmd_begin_debug_label :: proc(cmd_buf: Command_Buffer, name: string, color: [4]f32, loc := #caller_location)
{
    if ctx.validation
    {
        ok := true
        ok &= pool_check(&ctx.command_buffers, cmd_buf, "cmd_buf", loc)
        if !ok do return
    }

    scratch, _ := acquire_scratch()
    name_cstr := strings.clone_to_cstring(name, allocator = scratch)

    vk_cmd_buf := pool_get(&ctx.command_buffers, cmd_buf).handle
    vk.CmdBeginDebugUtilsLabelEXT(vk_cmd_buf, &vk.DebugUtilsLabelEXT {
        sType = .DEBUG_UTILS_LABEL_EXT,
        pLabelName = name_cstr,
        color = color,
    })
}

_cmd_end_debug_label :: proc(cmd_buf: Command_Buffer, loc := #caller_location)
{
    if ctx.validation
    {
        ok := true
        ok &= pool_check(&ctx.command_buffers, cmd_buf, "cmd_buf", loc)
        if !ok do return
    }

    vk_cmd_buf := pool_get(&ctx.command_buffers, cmd_buf).handle
    vk.CmdEndDebugUtilsLabelEXT(vk_cmd_buf)
}

_cmd_insert_debug_label :: proc(cmd_buf: Command_Buffer, name: string, color: [4]f32, loc := #caller_location)
{
    if ctx.validation
    {
        ok := true
        ok &= pool_check(&ctx.command_buffers, cmd_buf, "cmd_buf", loc)
        if !ok do return
    }

    scratch, _ := acquire_scratch()
    name_cstr := strings.clone_to_cstring(name, allocator = scratch)

    vk_cmd_buf := pool_get(&ctx.command_buffers, cmd_buf).handle
    vk.CmdInsertDebugUtilsLabelEXT(vk_cmd_buf, &vk.DebugUtilsLabelEXT {
        sType = .DEBUG_UTILS_LABEL_EXT,
        pLabelName = name_cstr,
        color = color,
    })
}

@(private="file")
mtl_ensure :: proc(err: ^ns.Error, msg := "", args: ..any, location := #caller_location)
{
    if err != nil {
        objc_str := err->localizedFailureReason()
        log.fatalf(msg, ..args, location = location)
        fatal_error("Metal failure: %v", objc_str->UTF8String(), location = location)
    }
}

@(private="file")
vk_check :: proc(result: vk.Result, location := #caller_location)
{
    if result != .SUCCESS {
        fatal_error("Vulkan failure: %v", result, location = location)
    }
}

@(private="file")
vk_debug_callback :: proc "system" (severity: vk.DebugUtilsMessageSeverityFlagsEXT,
                                    types: vk.DebugUtilsMessageTypeFlagsEXT,
                                    callback_data: ^vk.DebugUtilsMessengerCallbackDataEXT,
                                    user_data: rawptr) -> b32
{
    context = runtime.default_context()
    context.logger = vk_logger

    level: log.Level
    if .ERROR in severity        do level = .Error
    else if .WARNING in severity do level = .Warning
    else if .INFO in severity    do level = .Info
    else                         do level = .Debug
    log.log(level, callback_data.pMessage)

    return false
}

@(private="file")
create_swapchain :: proc(width: u32, height: u32, frames_in_flight: u32) -> Swapchain
{
    scratch, _ := acquire_scratch()

    res: Swapchain

    surface_caps: vk.SurfaceCapabilitiesKHR
    vk_check(vk.GetPhysicalDeviceSurfaceCapabilitiesKHR(ctx.phys_device, ctx.surface, &surface_caps))

    image_count := max(max(2, surface_caps.minImageCount), frames_in_flight)
    if surface_caps.maxImageCount != 0 do assert(image_count <= surface_caps.maxImageCount)

    surface_format_count: u32
    vk_check(vk.GetPhysicalDeviceSurfaceFormatsKHR(ctx.phys_device, ctx.surface, &surface_format_count, nil))
    surface_formats := make([]vk.SurfaceFormatKHR, surface_format_count, allocator = scratch)
    vk_check(vk.GetPhysicalDeviceSurfaceFormatsKHR(ctx.phys_device, ctx.surface, &surface_format_count, raw_data(surface_formats)))

    surface_format := surface_formats[0]
    for candidate in surface_formats
    {
        if candidate == { .B8G8R8A8_UNORM, .SRGB_NONLINEAR }
        {
            surface_format = candidate
            break
        }
    }

    present_mode_count: u32
    vk_check(vk.GetPhysicalDeviceSurfacePresentModesKHR(ctx.phys_device, ctx.surface, &present_mode_count, nil))
    present_modes := make([]vk.PresentModeKHR, present_mode_count, allocator = scratch)
    vk_check(vk.GetPhysicalDeviceSurfacePresentModesKHR(ctx.phys_device, ctx.surface, &present_mode_count, raw_data(present_modes)))

    present_mode := vk.PresentModeKHR.FIFO
    for candidate in present_modes {
        if candidate == .MAILBOX {
            present_mode = candidate
            break
        }
    }

    res.width = width
    res.height = height

    swapchain_ci := vk.SwapchainCreateInfoKHR {
        sType = .SWAPCHAIN_CREATE_INFO_KHR,
        surface = ctx.surface,
        minImageCount = image_count,
        imageFormat = surface_format.format,
        imageColorSpace = surface_format.colorSpace,
        imageExtent = { res.width, res.height },
        imageArrayLayers = 1,
        imageUsage = { .COLOR_ATTACHMENT },
        preTransform = surface_caps.currentTransform,
        compositeAlpha = { .OPAQUE },
        presentMode = present_mode,
        clipped = true,
    }
    vk_check(vk.CreateSwapchainKHR(ctx.device, &swapchain_ci, nil, &res.handle))

    vk_check(vk.GetSwapchainImagesKHR(ctx.device, res.handle, &image_count, nil))
    res.images = make([]vk.Image, image_count, context.allocator)
    res.texture_handles = make([]Texture_Handle, image_count, context.allocator)
    vk_check(vk.GetSwapchainImagesKHR(ctx.device, res.handle, &image_count, raw_data(res.images)))

    res.image_views = make([]vk.ImageView, image_count, context.allocator)
    for image, i in res.images
    {
        image_view_ci := vk.ImageViewCreateInfo {
            sType = .IMAGE_VIEW_CREATE_INFO,
            image = image,
            viewType = .D2,
            format = surface_format.format,
            subresourceRange = {
                aspectMask = { .COLOR },
                levelCount = 1,
                layerCount = 1,
            },
        }
        vk_check(vk.CreateImageView(ctx.device, &image_view_ci, nil, &res.image_views[i]))

        tex_info := Texture_Info { handle = image }
        append(&tex_info.views, Image_View_Info { info = image_view_ci, view = res.image_views[i] })
        res.texture_handles[i] = pool_add(&ctx.textures, tex_info, {})
    }

    res.present_semaphores = make([]vk.Semaphore, image_count, context.allocator)

    semaphore_ci := vk.SemaphoreCreateInfo { sType = .SEMAPHORE_CREATE_INFO }
    for &semaphore in res.present_semaphores {
        vk_check(vk.CreateSemaphore(ctx.device, &semaphore_ci, nil, &semaphore))
    }

    return res
}

@(private="file")
destroy_swapchain :: proc(swapchain: ^Swapchain)
{
    delete(swapchain.images)
    for semaphore in swapchain.present_semaphores {
        vk.DestroySemaphore(ctx.device, semaphore, nil)
    }
    delete(swapchain.present_semaphores)
    for image_view in swapchain.image_views {
        vk.DestroyImageView(ctx.device, image_view, nil)
    }
    delete(swapchain.image_views)
    vk.DestroySwapchainKHR(ctx.device, swapchain.handle, nil)

    for handle in swapchain.texture_handles
    {
        tex_info := pool_get(&ctx.textures, handle)
        // Vulkan objects for views are already destroyed by destroying swapchain.image_views
        delete(tex_info.views)
        pool_remove(&ctx.textures, handle)
    }
    delete(swapchain.texture_handles)

    swapchain^ = {}
}

@(private="file")
Swapchain :: struct
{
    layer: ^ca.MetalLayer,
    acquired: bool,

    current_drawable: ^mtl.Drawable,
    current_texture_handle: ^mtl.Texture,
    
    handle: vk.SwapchainKHR,
    width, height: u32,
    images: []vk.Image,
    texture_handles: []Texture_Handle,
    image_views: []vk.ImageView,
    present_semaphores: []vk.Semaphore,
}

@(private="file")
get_buf_offset_from_gpu_ptr :: proc(p: gpuptr) -> (buf: vk.Buffer, offset: u32, ok: bool)
{
    if p == {} do return {}, {}, false

    alloc_impl := transmute(Alloc_Impl_Info) p._impl
    alloc_info := pool_get(&ctx.allocs, alloc_impl.handle)

    buf = alloc_info.buf_handle
    offset = u32(uintptr(p.ptr) - uintptr(alloc_info.gpu))
    return buf, offset, true
}

@(private="file")
get_buf_size_from_gpu_ptr :: proc(p: gpuptr) -> (size: vk.DeviceSize, ok: bool)
{
    if p == {} do return {}, false

    alloc_impl := transmute(Alloc_Impl_Info) p._impl
    alloc_info := pool_get(&ctx.allocs, alloc_impl.handle)
    return alloc_info.buf_size, true
}

// Command buffers
@(private="file")
vk_acquire_cmd_buf :: proc(queue: Queue) -> Command_Buffer
{
    tls_ctx := get_tls()
    sync.guard(&ctx.lock)

    // Check whether there is a free command buffer available with a timeline value that is less than or equal to the current semaphore value
    if handle, ok := priority_queue.pop_safe(&tls_ctx.free_buffers[queue]); ok {
        cmd_buf_info, r_lock := pool_get_mut(&ctx.command_buffers, handle.pool_handle); sync.guard(r_lock)

        assert(!cmd_buf_info.recording)

        vk_sem := pool_get(&ctx.semaphores, ctx.cmd_bufs_sem_vals[queue].sem)

        current_semaphore_value: u64
        vk_check(vk.GetSemaphoreCounterValue(ctx.device, vk_sem, &current_semaphore_value))

        if current_semaphore_value >= cmd_buf_info.timeline_value {
            cmd_buf_info.recording = true
            cmd_buf_info.queue = queue
            cmd_buf_info.compute_shader = {}
            cmd_buf_info.thread_id = sync.current_thread_id()
            return handle.pool_handle
        } else {
            priority_queue.push(&tls_ctx.free_buffers[queue], handle)
        }
    }

    cmd_buf_info := Command_Buffer_Info {
        recording = true,
        queue = queue,
        compute_shader = {},
        thread_id = sync.current_thread_id(),
    }

    // If no free command buffer is available, create a new one
    cmd_buf_ai := vk.CommandBufferAllocateInfo {
        sType = .COMMAND_BUFFER_ALLOCATE_INFO,
        commandPool = tls_ctx.pools[queue],
        level = .PRIMARY,
        commandBufferCount = 1,
    }

    vk_check(vk.AllocateCommandBuffers(ctx.device, &cmd_buf_ai, &cmd_buf_info.handle))

    cmd_buf := pool_add(&ctx.command_buffers, cmd_buf_info, {})
    if cmd_buf_info_mut, r_lock := pool_get_mut(&ctx.command_buffers, cmd_buf); sync.guard(r_lock)
    {
        cmd_buf_info_mut.pool_handle = cmd_buf
        append(&tls_ctx.buffers[queue], cmd_buf_info_mut.pool_handle)
    }

    return cmd_buf
}

@(private="file")
vk_submit_cmd_bufs :: proc(cmd_bufs: []Command_Buffer)
{
    if len(cmd_bufs) <= 0 do return

    // NOTE: Submissions must be performed in order w.r.t the timeline value used.
    sync.guard(&ctx.lock)

    for cmd_buf in cmd_bufs
    {
        cmd_buf_info_mut, _ := pool_get_mut(&ctx.command_buffers, cmd_buf)
        intr.volatile_store(&cmd_buf_info_mut.timeline_value, sync.atomic_add(&ctx.cmd_bufs_sem_vals[cmd_buf_info_mut.queue].val, 1) + 1)
    }

    scratch, _ := acquire_scratch()
    submit_infos := make([dynamic]vk.SubmitInfo, allocator = scratch)
    queue: Queue
    for cmd_buf in cmd_bufs
    {
        cmd_buf_info := pool_get(&ctx.command_buffers, cmd_buf)
        cmd_buf_lock := pool_get_lock(&ctx.command_buffers, cmd_buf)
        sync.guard(cmd_buf_lock)

        queue = cmd_buf_info.queue
        queue_sem := ctx.cmd_bufs_sem_vals[queue].sem
        vk_queue_sem := pool_get(&ctx.semaphores, queue_sem)
        assert(cmd_buf_info.recording)
        assert(cmd_buf_info.thread_id == sync.current_thread_id())

        wait_count := len(cmd_buf_info.wait_sems)
        signal_count := len(cmd_buf_info.signal_sems) + 1
        wait_sems := make([]vk.Semaphore, wait_count, allocator = scratch)
        wait_values := make([]u64, wait_count, allocator = scratch)
        wait_stages := make([]vk.PipelineStageFlags, wait_count, allocator = scratch)
        signal_sems := make([]vk.Semaphore, signal_count, allocator = scratch)
        signal_values := make([]u64, signal_count, allocator = scratch)
        for wait_sem, i in cmd_buf_info.wait_sems
        {
            wait_sems[i] = pool_get(&ctx.semaphores, wait_sem.sem)
            wait_stages[i] = { .ALL_COMMANDS }
            wait_values[i] = wait_sem.val
        }
        for signal_sem, i in cmd_buf_info.signal_sems
        {
            signal_sems[i] = pool_get(&ctx.semaphores, signal_sem.sem)
            signal_values[i] = signal_sem.val
        }

        signal_sems[signal_count - 1] = vk_queue_sem
        signal_values[signal_count - 1] = cmd_buf_info.timeline_value

        to_submit := make([]vk.CommandBuffer, 1, allocator = scratch)
        to_submit[0] = cmd_buf_info.handle

        next := new(vk.TimelineSemaphoreSubmitInfo, allocator = scratch)
        next^ = {
            sType = .TIMELINE_SEMAPHORE_SUBMIT_INFO,
            waitSemaphoreValueCount = u32(len(wait_values)),
            pWaitSemaphoreValues = raw_data(wait_values),
            signalSemaphoreValueCount = u32(len(signal_values)),
            pSignalSemaphoreValues = raw_data(signal_values),
        }
        submit_info := vk.SubmitInfo {
            sType = .SUBMIT_INFO,
            pNext = next,
            commandBufferCount = u32(len(to_submit)),
            pCommandBuffers = raw_data(to_submit),
            waitSemaphoreCount = u32(len(wait_sems)),
            pWaitSemaphores = raw_data(wait_sems),
            pWaitDstStageMask = raw_data(wait_stages),
            signalSemaphoreCount = u32(len(signal_sems)),
            pSignalSemaphores = raw_data(signal_sems),
        }
        append(&submit_infos, submit_info)
    }

    queue_info := ctx.queues[queue]
    vk_queue := queue_info.handle
    vk_check(vk.QueueSubmit(vk_queue, u32(len(submit_infos)), raw_data(submit_infos), {}))

    for cmd_buf in cmd_bufs {
        recycle_cmd_buf(cmd_buf)
    }
}

@(private="file")
recycle_cmd_buf :: proc(cmd_buf: Command_Buffer)
{
    tls_ctx := get_tls()

    clear_cmd_buf(cmd_buf)

    cmd_buf_info := pool_get(&ctx.command_buffers, cmd_buf)
    priority_queue.push(&tls_ctx.free_buffers[cmd_buf_info.queue], Free_Command_Buffer { pool_handle = cmd_buf_info.pool_handle, timeline_value = cmd_buf_info.timeline_value })
}

// Interop

_vk_get_instance :: proc() -> vk.Instance
{
    return ctx.instance
}

_vk_get_physical_device :: proc() -> vk.PhysicalDevice
{
    return ctx.phys_device
}

_vk_get_device :: proc() -> vk.Device
{
    return ctx.device
}

_vk_get_queue :: proc(queue: Queue) -> vk.Queue
{
    return ctx.queues[queue].handle
}

_vk_get_queue_family :: proc(queue: Queue) -> u32
{
    return ctx.queues[queue].family_idx
}

_vk_get_command_buffer :: proc(cmd_buf: Command_Buffer) -> vk.CommandBuffer
{
    cmd_buf := pool_get(&ctx.command_buffers, cmd_buf)
    return cmd_buf.handle
}

_vk_get_swapchain_image_count :: proc() -> u32
{
    return u32(len(ctx.swapchain.images))
}

_vk_get_image :: proc(texture: Texture) -> vk.Image
{
    image := pool_get(&ctx.textures, texture.handle)
    return image.handle
}

_vk_get_buffer :: proc(addr: gpuptr) -> (vk.Buffer, u32)
{
    buf, offset, ok := get_buf_offset_from_gpu_ptr(addr)
    ensure(ok)
    return buf, offset
}

_vk_wrap_image :: proc(image: vk.Image, desc: Texture_Desc, name := "", loc := #caller_location) -> Texture
{
    desc_clean := texture_desc_cleanup(desc)

    if ctx.validation {
        ensure(image != {}, "Cannot wrap a nil VkImage.")
    }

    tex_info := Texture_Info {
        handle = image,
        owns_image = false,
    }

    return Texture {
        type = desc_clean.type,
        dimensions = desc_clean.dimensions,
        format = desc_clean.format,
        mip_count = desc_clean.mip_count,
        sample_count = desc_clean.sample_count,
        layer_count = desc_clean.layer_count,
        handle = pool_add(&ctx.textures, tex_info, { name = name, created_at = loc }),
    }
}

@(thread_local) EXTRA_OPT_DEVICE_EXTENSIONS: [dynamic]cstring
_vk_add_opt_device_extension :: proc(extension: cstring)
{
    append(&EXTRA_OPT_DEVICE_EXTENSIONS, extension)
}

@(thread_local) EXTRA_DEVICE_EXTENSIONS: [dynamic]cstring
_vk_add_device_extension :: proc(extension: cstring)
{
    append(&EXTRA_DEVICE_EXTENSIONS, extension)
}

_vk_move_semaphore :: proc(semaphore: vk.Semaphore, loc := #caller_location) -> Semaphore
{
    return pool_add(&ctx.semaphores, semaphore, { name = "", created_at = loc })
}

@(private)
to_vk_render_attachment :: #force_inline proc(attach: Render_Attachment) -> vk.RenderingAttachmentInfo
{
    view_desc := attach.view
    texture := attach.texture
    resolve_texture := attach.resolve_texture
    resolve_view_desc := attach.resolve_view

    has_output := texture != {}
    vk_image := pool_get(&ctx.textures, texture.handle).handle if has_output else vk.Image(0)
    has_resolve := resolve_texture != {}
    vk_resolve_image := pool_get(&ctx.textures, resolve_texture.handle).handle if has_resolve else vk.Image(0)

    view_desc_clean := texture_view_desc_cleanup(texture, view_desc)
    resolve_view_desc_clean := texture_view_desc_cleanup(resolve_texture, resolve_view_desc)

    plane_aspect := to_vk_image_aspect_flags(view_desc_clean.format)

    view: vk.ImageView
    if has_output
    {
        image_view_ci := vk.ImageViewCreateInfo {
            sType = .IMAGE_VIEW_CREATE_INFO,
            image = vk_image,
            viewType = to_vk_texture_view_type(view_desc_clean.type),
            format = to_vk_texture_format(view_desc_clean.format),
            subresourceRange = {
                aspectMask = plane_aspect,
                levelCount = 1,
                layerCount = 1,
            }
        }
        view = get_or_add_image_view(texture.handle, image_view_ci)
    }

    resolve_view: vk.ImageView
    if has_resolve
    {
        resolve_image_view_ci := vk.ImageViewCreateInfo {
            sType = .IMAGE_VIEW_CREATE_INFO,
            image = vk_resolve_image,
            viewType = to_vk_texture_view_type(resolve_view_desc_clean.type),
            format = to_vk_texture_format(resolve_view_desc_clean.format),
            subresourceRange = {
                aspectMask = plane_aspect,
                levelCount = 1,
                layerCount = 1,
            }
        }
        resolve_view = get_or_add_image_view(resolve_texture.handle, resolve_image_view_ci)
    }

    vk_store_op, vk_resolve_mode := to_vk_store_op(attach.store_op)

    return {
        sType = .RENDERING_ATTACHMENT_INFO,
        imageView = view,
        imageLayout = .GENERAL,
        loadOp = to_vk_load_op(attach.load_op),
        storeOp = vk_store_op,
        clearValue = { color = { float32 = attach.clear_color } },
        resolveMode = vk_resolve_mode,
        resolveImageView = resolve_view,
        resolveImageLayout = .GENERAL if has_resolve else {},
    }
}

//////////////////////////////////////
// Validation

@(private="file")
check_ptr :: proc(p: gpuptr, name: string, loc: runtime.Source_Code_Location) -> bool
{
    if p == {} {
        log.errorf("'%v' address is nil.", name, location = loc)
        return false
    }

    alloc_impl := transmute(Alloc_Impl_Info) p._impl
    alloc_handle := alloc_impl.handle
    if !pool_check_no_message(&ctx.allocs, alloc_handle) {
        log.errorf("'%v' address is stale, has been freed before.", name, location = loc)
        return false
    }
    alloc_info := pool_get(&ctx.allocs, alloc_handle)

    if uintptr(p.ptr) > uintptr(alloc_info.gpu) + uintptr(alloc_info.buf_size) || uintptr(p.ptr) < uintptr(alloc_info.gpu) {
        log.errorf("'%v' address is out of range for the designated allocation. %v bytes were allocated, but you're attempting to access offset %v.",
                   name, alloc_info.buf_size, i64(uintptr(p.ptr)) - i64(uintptr(alloc_info.gpu)), location = loc)
        return false
    }

    return true
}

@(private="file")
check_ptr_allow_nil :: proc(p: gpuptr, name: string, loc: runtime.Source_Code_Location) -> bool
{
    if p == {} {
        return true
    }

    alloc_impl := transmute(Alloc_Impl_Info) p._impl
    alloc_handle := alloc_impl.handle
    if !pool_check_no_message(&ctx.allocs, alloc_handle) {
        log.errorf("'%v' address is stale, has been freed before.", name, location = loc)
        return false
    }
    alloc_info := pool_get(&ctx.allocs, alloc_handle)

    if uintptr(p.ptr) > uintptr(alloc_info.gpu) + uintptr(alloc_info.buf_size) || uintptr(p.ptr) < uintptr(alloc_info.gpu) {
        log.errorf("'%v' address is out of range for the designated allocation. %v bytes were allocated, but you're attempting to access offset %v.",
                   name, alloc_info.buf_size, i64(uintptr(p.ptr)) - i64(uintptr(alloc_info.gpu)), location = loc)
        return false
    }

    return true
}

@(private="file")
check_ptr_range :: proc(p: gpuptr, #any_int size: i64, name: string, loc: runtime.Source_Code_Location) -> bool
{
    if p == {} {
        log.errorf("'%v' address is nil.", name, location = loc)
        return false
    }

    alloc_impl := transmute(Alloc_Impl_Info) p._impl
    alloc_handle := alloc_impl.handle
    if !pool_check_no_message(&ctx.allocs, alloc_handle) {
        log.errorf("'%v' address is stale, has been freed before.", name, location = loc)
        return false
    }
    alloc_info := pool_get(&ctx.allocs, alloc_handle)

    if uintptr(p.ptr) + uintptr(size) > uintptr(alloc_impl.range_end) || uintptr(p.ptr) < uintptr(alloc_info.gpu) {
        log.errorf("'%v' address is out of range for the designated allocation. %v bytes were allocated, but you're attempting to access [0, %v].",
                   name, i64(uintptr(alloc_impl.range_end)) - i64(uintptr(p.ptr)), size, location = loc)
        return true  // Proceed with execution, make sure to clamp accesses.
    }

    return true
}

check_ptr_must_not_be_suballoc :: proc(p: gpuptr, name: string, loc: runtime.Source_Code_Location) -> bool
{
    if p == {} {
        log.errorf("'%v' address is nil.", name, location = loc)
        return false
    }

    alloc_impl := transmute(Alloc_Impl_Info) p._impl
    alloc_handle := alloc_impl.handle
    if !pool_check_no_message(&ctx.allocs, alloc_handle) {
        log.errorf("'%v' address is stale, has been freed before.", name, location = loc)
        return false
    }
    alloc_info := pool_get(&ctx.allocs, alloc_handle)

    end_ptr := rawptr(uintptr(alloc_info.gpu) + uintptr(alloc_info.buf_size))
    if uintptr(alloc_impl.range_end) < uintptr(end_ptr) || uintptr(p.ptr) > uintptr(alloc_info.gpu) {
        log.errorf("'%v' address was suballocated, need an actual allocation here.", name, location = loc)
        return false
    }

    return true
}

@(private="file")
check_cmd_buf_has_compute_shader_set :: proc(cmd_buf: Command_Buffer, name: string, loc: runtime.Source_Code_Location) -> bool
{
    if !pool_check_no_message(&ctx.command_buffers, cmd_buf) do return false

    cmd_buf_info := pool_get(&ctx.command_buffers, cmd_buf)

    if cmd_buf_info.compute_shader == nil {
        log.errorf("'%v' does not have an associated compute shader. Call cmd_set_compute_shader first.", name, location = loc)
        return false
    }

    return true
}

@(private="file")
check_cmd_buf_must_be_graphics :: proc(cmd_buf: Command_Buffer, name: string, loc: runtime.Source_Code_Location) -> bool
{
    if !pool_check_no_message(&ctx.command_buffers, cmd_buf) do return false

    cmd_buf_info := pool_get(&ctx.command_buffers, cmd_buf)
    if cmd_buf_info.queue != .Main {
        log.errorf("'%v' must be of type '%v', got type '%v'.", name, Queue.Main, cmd_buf_info.queue, location = loc)
        return false
    }

    return true
}

@(private="file")
check_cmd_buf_must_be_recording :: proc(cmd_buf: Command_Buffer, name: string, loc: runtime.Source_Code_Location) -> bool
{
    if !pool_check_no_message(&ctx.command_buffers, cmd_buf) do return false

    cmd_buf_info := pool_get(&ctx.command_buffers, cmd_buf)
    if !cmd_buf_info.recording {
        log.errorf("'%v' must be in a recording state, it's illegal to reuse a command buffer after submit. Command buffers are temporary handles.", name, location = loc)
        return false
    }

    return true
}

@(private="file")
check_bvh_must_be_tlas :: proc(bvh: BVH, name: string, loc: runtime.Source_Code_Location) -> bool
{
    bvh_info := pool_get(&ctx.bvhs, bvh)
    if bvh_info.is_blas {
        log.errorf("'%v' must be a TLAS.", name, location = loc)
        return false
    }

    return true
}

@(private="file")
check_bvh_must_be_blas :: proc(bvh: BVH, name: string, loc: runtime.Source_Code_Location) -> bool
{
    bvh_info := pool_get(&ctx.bvhs, bvh)
    if !bvh_info.is_blas {
        log.errorf("'%v' must be a BLAS.", name, location = loc)
        return false
    }

    return true
}

@(private="file")
check_texture_descriptor :: proc(desc: Texture_Descriptor, name: string, index: int, loc: runtime.Source_Code_Location) -> bool
{
    if !pool_check_no_message(&ctx.textures, texture_descriptor_get_handle(desc)) {
        log.errorf("'%v[%v]' texture descriptor is stale, the underlying texture has been freed.", name, index, location = loc)
        return false
    }

    return true
}

@(private="file")
texture_descriptor_get_handle :: #force_inline proc(desc: Texture_Descriptor) -> Texture_Handle
{
    return transmute(Texture_Handle) (cast([2]u64)desc)[0]
}

@(private="file")
texture_descriptor_get_vk_view :: #force_inline proc(desc: Texture_Descriptor) -> vk.ImageView
{
    return cast(vk.ImageView) (cast([2]u64)desc)[1]
}

vk_set_debug_name :: proc(name: string, handle: u64, type: vk.ObjectType)
{
    if name == "" || !ctx.validation do return

    scratch, _ := acquire_scratch()
    name_cstr := strings.clone_to_cstring(name, allocator = scratch)

    vk.SetDebugUtilsObjectNameEXT(ctx.device, &vk.DebugUtilsObjectNameInfoEXT {
        sType = .DEBUG_UTILS_OBJECT_NAME_INFO_EXT,
        objectType = type,
        objectHandle = handle,
        pObjectName = name_cstr,
    })
}
