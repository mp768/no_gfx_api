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
import "core:hash/xxhash"
import intr "base:intrinsics"

// Temp: Just do avoid compilation errors
import vk "vendor:vulkan"

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
Total_Descriptor_Heap_Buffer_Count :: 14

@(private="file")
Render_Pipeline_Handle :: distinct u64

@(private="file")
Compute_Pipeline_Handle :: distinct Shader

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
    idle_queue_sem_vals: [Queue]Semaphore_Value,

    // cached pipelines
    render_pipelines: map[Render_Pipeline_Handle]Render_Pipeline_Info,
    compute_pipelines: map[Compute_Pipeline_Handle]^mtl.ComputePipelineState,

    // cached depth stencil state
    depth_stencil_states: map[Depth_State]^mtl.DepthStencilState,

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
Shader_Info :: struct 
{
    handle: ^mtl.MTL4LibraryFunctionDescriptor,
    current_workgroup_size: [3]u32,
    is_compute: bool,
    graphics_type: Shader_Type_Graphics,
}

@(private="file")
Render_State_Info :: struct 
{
    vert_shader: Maybe(Shader),
    frag_shader: Maybe(Shader),

    depth_state: Depth_State,
    raster_state: Raster_State,
    blend_state: Blend_State,
    viewport: Viewport,
    scissor: Rect_2D,

    render_sample_count: u32,
    
    graphics_shader_source_location: runtime.Source_Code_Location,
}

@(private="file")
Render_Pipeline_Info :: struct
{
    state: ^mtl.RenderPipelineState,
    depth_stencil: ^mtl.DepthStencilState,
}

@(private="file")
Push_Constant_Buffer_Max_Size :: max(size_of(Graphics_Shader_Push_Constants), size_of(Compute_Shader_Push_Constants)) * 32

@(private="file")
Command_Buffer_Info :: struct 
{
    handle: ^mtl.MTL4CommandBuffer,
    allocator: ^mtl.MTL4CommandAllocator,

    argument_table: ^mtl.MTL4ArgumentTable,
    push_constant_buffer: ptr,

    timeline_value: u64,
    thread_id: int,
    queue: Queue,
    recording: bool,
    
    encoder_type: enum u8 { None = 0, Compute, Render },
    encoder: struct #raw_union {
        base: ^mtl.MTL4CommandEncoder,
        // NOTE: Compute encoder handles compute, blit, and acceleration 
        compute: ^mtl.MTL4ComputeCommandEncoder,
        render: ^mtl.MTL4RenderCommandEncoder,
    }, 

    compute_shader: Maybe(Shader),
    using render_state: Render_State_Info,
    
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
    has_changed: bool,
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
    #unroll for queue in Queue 
    {
        mtl4_queue := (ctx.device)->newMTL4CommandQueue()

        mtl4_queue->addResidencySet(ctx.allocation_set)
        
        ctx.queues[queue] = mtl4_queue
    }

    // Set up queue specific semaphores for command buffers and idle procedures.
    #unroll for queue in Queue 
    {
        ctx.cmd_bufs_sem_vals[queue] = {
            sem = semaphore_create(0),
            val = 0,
        }

        ctx.idle_queue_sem_vals[queue] = {
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
        compiler_desc := objc_alloc(mtl.MTL4CompilerDescriptor)
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
    set_desc := objc_alloc(mtl.ResidencySetDescriptor)
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

    #unroll for queue in Queue 
    {
        mtl_queue := ctx.queues[queue]
        sem_val := &ctx.idle_queue_sem_vals[queue]
        mtl_sem := pool_get(&ctx.semaphores, sem_val.sem)

        sem_val.val += 1

        mtl_queue->signalEvent(mtl_sem, sem_val.val)

        mtl_sem->waitUntilSignaledValue(sem_val.val, max(u64))
    }
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
    bytes_per_row := mtl_helper_bytes_per_row(desc.format, desc.dimensions[0])
    
    tex_desc := to_mtl_texture_descriptor(desc_clean)
    defer tex_desc->release()

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

    desc := objc_alloc(mtl.SamplerDescriptor)
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
    desc[1] = u64(uintptr(texture.handle))

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
        has_changed = false,
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

    if heap_info, heap_lock := pool_get_mut(&ctx.desc_heaps, heap); sync.guard(heap_lock) 
    {
        heap_info.has_changed = true
    }

    heap_info := pool_get(&ctx.desc_heaps, heap)

    scratch, _ := acquire_scratch()
    texture_ids: []mtl.ResourceID = make([]mtl.ResourceID, len(textures), allocator = scratch)
    for &id, i in texture_ids
    {
        texture := texture_descriptor_get_mtl_texture(textures[i])

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

    if heap_info, heap_lock := pool_get_mut(&ctx.desc_heaps, heap); sync.guard(heap_lock) 
    {
        heap_info.has_changed = true
    }

    heap_info := pool_get(&ctx.desc_heaps, heap)

    scratch, _ := acquire_scratch()
    texture_ids: []mtl.ResourceID = make([]mtl.ResourceID, len(textures), allocator = scratch)
    for &id, i in texture_ids
    {
        texture := texture_descriptor_get_mtl_texture(textures[i])

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

    if heap_info, heap_lock := pool_get_mut(&ctx.desc_heaps, heap); sync.guard(heap_lock) 
    {
        heap_info.has_changed = true
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
        lib_desc := objc_alloc(mtl.MTL4LibraryDescriptor)
        defer lib_desc->release()
    
        source_objc := to_mtl_string(source)
        defer source_objc->release()
        
        lib_desc->setSource(source_objc)

        err: ^ns.Error
        library = (ctx.shader_compiler)->newLibraryWithDescriptor_error(lib_desc, &err)

        mtl_ensure(err, "%v shader creation failed during library/source compilation.", "Compute" if is_compute else "Graphics")
    }

    shader := objc_alloc(mtl.MTL4LibraryFunctionDescriptor)

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

    mtl_queue := ctx.queues[queue]
    
    sem_val := &ctx.idle_queue_sem_vals[queue]
    mtl_sem := pool_get(&ctx.semaphores, sem_val.sem)

    sem_val.val += 1

    mtl_queue->signalEvent(mtl_sem, sem_val.val)

    mtl_sem->waitUntilSignaledValue(sem_val.val, max(u64))
}

_commands_begin :: proc(queue: Queue, loc := #caller_location) -> Command_Buffer
{
    cmd_buf := mtl_acquire_cmd_buf(queue)
    cmd_buf_info := pool_get(&ctx.command_buffers, cmd_buf)

    mtl_cmd_buf := cmd_buf_info.handle
    allocator := cmd_buf_info.allocator

    mtl_cmd_buf->beginCommandBufferWithAllocator_(allocator)

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
        mtl_cmd_buf := cmd_buf_info.handle

        mtl_cmd_buf->endCommandBuffer()
    }

    // NOTE: Submission already recycles cmd buffers for us
    mtl_submit_cmd_bufs(cmd_bufs)
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

@(private="file")
mtl_get_compute_encoder :: proc(cmd_buf: Command_Buffer, loc := #caller_location) -> ^mtl.MTL4ComputeCommandEncoder
{
    cmd_buf_info, cmd_buf_sync := pool_get_mut(&ctx.command_buffers, cmd_buf); sync.guard(cmd_buf_sync)

    if cmd_buf_info.encoder_type == .Compute {
        return cmd_buf_info.encoder.compute
    }

    ensure(cmd_buf_info.encoder_type != .Render, "Cannot perform compute, blit, or accleration work within a render pass. Please consider doing this operation elsewhere.", loc = loc)

    mtl_cmd_buf := cmd_buf_info.handle
    
    cmd_buf_info.encoder_type = .Compute
    cmd_buf_info.encoder.compute = mtl_cmd_buf->computeCommandEncoder()
   
    return cmd_buf_info.encoder.compute 
}

@(private="file")
mtl_get_render_encoder :: proc(cmd_buf: Command_Buffer, loc := #caller_location) -> ^mtl.MTL4RenderCommandEncoder
{
    cmd_buf_info := pool_get(&ctx.command_buffers, cmd_buf)

    ensure(cmd_buf_info.encoder_type == .Render, "Cannot perform this command outside of a render pass. Call 'cmd_begin_render_pass' to start one.", loc = loc)

    return cmd_buf_info.encoder.render
}

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

    encoder := mtl_get_compute_encoder(cmd_buf, loc = loc)

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
        encoder->copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
            src_buf, ns.UInteger(src_offset),
            dst_buf, ns.UInteger(dst_offset),
            ns.UInteger(to_copy)
        )
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

    encoder := mtl_get_compute_encoder(cmd_buf, loc = loc)

    cmd_buf_info := pool_get(&ctx.command_buffers, cmd_buf)
    tex_info := pool_get(&ctx.textures, dst.handle)

    src_buf, src_offset, ok_s := get_buf_offset_from_gpu_ptr(src)
    assert(ok_s)

    mip_width := max(1, dst.dimensions.x >> region.mip_level)
    mip_height := max(1, dst.dimensions.y >> region.mip_level)
    mip_depth := max(1, dst.dimensions.z >> region.mip_level)

    is_3d := dst.type == .D3

    bytes_per_row := mtl_helper_bytes_per_row(dst.format, mip_width)
    bytes_per_image := mtl_helper_bytes_per_image(dst.format, mip_width, mip_height, 1)

    block_width := mtl_helper_texture_format_block_width(dst.format)
    block_height := mtl_helper_texture_format_block_height(dst.format)

    // Align each dimension to the block size of the destination texture 
    aligned_width := ((mip_width + block_width - 1) / block_width) * block_width
    aligned_height := ((mip_height + block_height - 1) / block_height) * block_height
    
    source_size: mtl.Size
    source_size.width = ns.UInteger(min(aligned_width, mip_width))
    source_size.height = ns.UInteger(min(aligned_height, mip_height))

    // Depth corresponds to the layer count of an image array when the image is not 3D
    source_size.depth = ns.UInteger(mip_depth if is_3d else region.layer_count)

    // NOTE: This is (0, 0, 0), since we're only performing a full copy to the
    // texture. If we were to implement the code using 'region.rect', this would have 
    // to change and so would the 'source_size' measurement.
    destination_origin: mtl.Origin
    destination_origin.x = 0
    destination_origin.y = 0
    destination_origin.z = 0

    src_bytes_per_image := ns.UInteger(0)
    if is_3d || region.layer_count > 1
    {
        src_bytes_per_image = ns.UInteger(bytes_per_image)
    }

    encoder->copyFromBuffer_sourceOffset_sourceBytesPerRow_sourceBytesPerImage_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(
        src_buf, 
        ns.UInteger(src_offset), 
        ns.UInteger(bytes_per_row), 
        src_bytes_per_image,
        source_size,
        tex_info.handle,
        ns.UInteger(region.base_layer),
        ns.UInteger(region.mip_level),
        destination_origin
    )
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

    encoder := mtl_get_compute_encoder(cmd_buf, loc = loc)

    // TODO: Implement this

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

    if heap_info, heap_lock := pool_get_mut(&ctx.desc_heaps, heap); sync.guard(heap_lock)
    {
        // Commit any changes made to the descriptor heap here
        if heap_info.has_changed 
        {
            heap_info.has_changed = false
            (heap_info.residency_set)->commit()
            (heap_info.residency_set)->requestResidency()
        }
    }

    cmd_buf_info := pool_get(&ctx.command_buffers, cmd_buf)
    mtl_cmd_buf := cmd_buf_info.handle

    heap_info := pool_get(&ctx.desc_heaps, heap)

    for index in Shader_Texture_Descriptor_Indices
    {
        (cmd_buf_info.argument_table)->setAddress_atIndex((heap_info.textures)->gpuAddress(), index)
    }

    for index in Shader_Texture_RW_Descriptor_Indices
    {
        (cmd_buf_info.argument_table)->setAddress_atIndex((heap_info.textures_rw)->gpuAddress(), index)
    }

    (cmd_buf_info.argument_table)->setAddress_atIndex((heap_info.samplers)->gpuAddress(), Shader_Sampler_Descriptor_Index)
    (cmd_buf_info.argument_table)->setAddress_atIndex((heap_info.bvhs)->gpuAddress(), Shader_BVH_Descriptor_Index)

    mtl_cmd_buf->useResidencySet(heap_info.residency_set)
    
    // NOTE: Unlike the vulkan implementation, we'll bind the argument table
    // at the draw/dispatch site instead (if it isn't already bound).
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

    cmd_buf_info := pool_get(&ctx.command_buffers, cmd_buf)

    mtl_before := to_mtl_stage(before)
    mtl_after  := to_mtl_stage(after)

    ensure(cmd_buf_info.encoder_type != .None, "Cannot place a barrier without being in an active encoding pass", loc = loc)

    (cmd_buf_info.encoder.base)->barrierAfterStages(mtl_after, mtl_before, {})
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

    // TODO: For the render pipeline, hold off on creating it until we get to a draw
    // call. Until we call a draw procedure, we'll propagate the values passed into
    // these procedure, '_cmd_set_depth_state', '_cmd_set_raster_state', and so on.
    // 
    // Then for the Render_Pipeline_Handle, we'll unfortunately have to make it
    // a hash of the current set of configurations. We'll do a map[u64]^mtl.RenderPipelineState,
    // for the caching. 
    // 
    // For the hash we'll use, we can just do something like this:
    // ```
    // import "core:hash/xxhash"
    // import "core:mem"
    // 
    // My_Struct :: struct { ... }
    // s: My_Struct = ...
    // 
    // bytes := mem.slice_from_ptr(cast([^]u8)&s, size_of(My_Struct))
	//
	// hash: u64 = xxhash.XXH64(bytes)
    // ```
    // 
    // IMPORTANT: Don't forget to include sync guards to prevent multi-threading
    // issues when it comes time to append or look-up render pipelines from the 
    // cache.
    // 
    // We can apply the same with compute pipelines if necessary. However, I feel
    // it most likely won't be. Compared to the render pipeline, the compute pass
    // shouldn't have as many inline configuration options.

    cmd_buf_info, cmd_sync := pool_get_mut(&ctx.command_buffers, cmd_buf); sync.guard(cmd_sync)

    cmd_buf_info.vert_shader = vert_shader
    cmd_buf_info.frag_shader = frag_shader

    cmd_buf_info.graphics_shader_source_location = loc

    /*
    pipeline_handle: Render_Pipeline_Handle
    pipeline_handle[0] = rawptr(vert_shader)
    pipeline_handle[1] = rawptr(frag_shader)

    render_pipeline: ^mtl.RenderPipelineState
    if pipeline_handle not_in ctx.render_pipelines
    {
        vert_shader := pool_get(&ctx.shaders, vert_shader)
        frag_shader := pool_get(&ctx.shaders, frag_shader)

        assert(!vert_shader.is_compute, "Expected the first shader given to be a vertex shader. Instead, we got a compute shader.", loc = loc)
        assert(!frag_shader.is_compute, "Expected the second shader given to be a fragment shader. Instead, we got a compute shader.", loc = loc)
        assert(vert_shader.graphics_type == .Vertex, "Expected the first shader given to be a vertex shader. Instead, we got a fragment shader.", loc = loc)
        assert(frag_shader.graphics_type == .Fragment, "Expected the second shader given to be a fragment shader. Instead, we got a vertex shader.", loc = loc)

        pipe_desc := objc_alloc(mtl.MTL4RenderPipelineDescriptor)
        defer pipe_desc->release()

        pipe_desc->setVertexFunctionDescriptor(vert_shader.handle)
        pipe_desc->setFragmentFunctionDescriptor(frag_shader.handle)

        // TODO: Figure out a proper value for this.
        pipe_desc->setRasterSampleCount(1)

        compile_task_options := objc_alloc(mtl.MTL4CompilerTaskOptions)
        defer compile_task_options->release()
        
        err: ^ns.Error
        render_pipeline = (ctx.shader_compiler)->newRenderPipelineStateWithDescriptor_compilerTaskOptions_error(pipe_desc, compile_task_options, &err)

        mtl_ensure(err, "Unable to create new render pipeline with given shaders on line %v, col %v, in '%v'", loc.line, loc.column, loc.file_path)

        ctx.render_pipelines[pipeline_handle] = render_pipeline
    }
    else 
    {
        render_pipeline, _ = ctx.render_pipelines[pipeline_handle]
    }

    encoder->setRenderPipelineState(render_pipeline)
    */

    /*
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
    */
}

_cmd_set_depth_state :: proc(cmd_buf: Command_Buffer, state: Depth_State, loc := #caller_location)
{
    if ctx.validation
    {
        ok := true
        ok &= pool_check(&ctx.command_buffers, cmd_buf, "cmd_buf", loc)
        if !ok do return
    }

    cmd_buf_info, cmd_sync := pool_get(&ctx.command_buffers, cmd_buf); sync.guard(cmd_sync)

    cmd_buf_info.depth_state = state
}

_cmd_set_raster_state :: proc(cmd_buf: Command_Buffer, state: Raster_State, loc := #caller_location)
{
    if ctx.validation
    {
        ok := true
        ok &= pool_check(&ctx.command_buffers, cmd_buf, "cmd_buf", loc)
        if !ok do return
    }

    cmd_buf_info, cmd_sync := pool_get(&ctx.command_buffers, cmd_buf); sync.guard(cmd_sync)

    cmd_buf_info.raster_state = state
}

_cmd_set_blend_state :: proc(cmd_buf: Command_Buffer, state: Blend_State, loc := #caller_location)
{
    if ctx.validation
    {
        ok := true
        ok &= pool_check(&ctx.command_buffers, cmd_buf, "cmd_buf", loc)
        if !ok do return
    }

    cmd_buf_info, cmd_sync := pool_get(&ctx.command_buffers, cmd_buf); sync.guard(cmd_sync)

    cmd_buf_info.blend_state = state
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

    cmd_buf_info, cmd_sync := pool_get(&ctx.command_buffers, cmd_buf); sync.guard(cmd_sync)

    cmd_buf_info.viewport = viewport
}

_cmd_set_scissor :: proc(cmd_buf: Command_Buffer, scissor: Rect_2D, loc := #caller_location)
{
    if ctx.validation
    {
        ok := true
        ok &= pool_check(&ctx.command_buffers, cmd_buf, "cmd_buf", loc)
        if !ok do return
    }

    cmd_buf_info, cmd_sync := pool_get(&ctx.command_buffers, cmd_buf); sync.guard(cmd_sync)

    cmd_buf_info.scissor = scissor
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

    encoder := mtl_get_compute_encoder(cmd_buf, loc = loc)

    shader_info := pool_get(&ctx.shaders, compute_shader)
    mtl_shader_info := shader_info.handle

    compute_pipeline: ^mtl.ComputePipelineState
    if sync.guard(&ctx.lock)
    {
        pipeline_handle := Compute_Pipeline_Handle(compute_shader)
        
        if pipeline_handle not_in ctx.compute_pipelines
        {
            assert(shader_info.is_compute, "Expected the shader given to be a compute shader. Instead, we got a vertex or fragment shader.", loc = loc)

            pipe_desc := objc_alloc(mtl.MTL4ComputePipelineDescriptor)
            defer pipe_desc->release()

            pipe_desc->setComputeFunctionDescriptor(mtl_shader_info)

            compile_task_options := objc_alloc(mtl.MTL4CompilerTaskOptions)
            defer compile_task_options->release()

            err: ^ns.Error
            compute_pipeline = (ctx.shader_compiler)->newComputePipelineStateWithDescriptor_compilerTaskOptions_error(pipe_desc, compile_task_options, &err)

            mtl_ensure(err, "Unable to create new compute pipeline with given compute shader.", loc = loc)
            ctx.compute_pipelines[pipeline_handle] = compute_pipeline
        }
        else
        {
            compute_pipeline, _ = ctx.compute_pipelines[pipeline_handle]
        }
    }

    encoder->setComputePipelineState(compute_pipeline)

    if cmd_buf_info, cmd_lock := pool_get_mut(&ctx.command_buffers, cmd_buf); sync.guard(cmd_lock)
    {
        cmd_buf_info.compute_shader = compute_shader
    }
}

@(private="file")
mtl_push_constant :: proc(cmd_buf: Command_Buffer, constant: $T, loc := #caller_location)
{
    cmd_buf_info, cmd_sync := pool_get(&ctx.command_buffers, cmd_buf); sync.guard(cmd_sync)

    if !check_ptr(cnd_buf_info.push_constant_buffer, "cmd_buf.push_constant_buffer", loc)
    {
        ensure(false, "Unable to push additional data to the internal 'push constant' buffer of the current command buffer. Too many things have been passed in.", loc = loc)
    }

    constant := constant

    mem.copy(
        cmd_buf_info.push_constant_buffer.cpu,
        &constant,
        size_of(T)
    )

    push_constant_gpu_address := cast(mtl.GPUAddress) cast(uintptr) cmd_buf_info.push_constant_buffer.gpu.ptr

    (cmd_buf_info.argument_table)->setAddress_atIndex(push_constant_gpu_address, 0)
    
    cmd_buf_info.push_constant_buffer = ptr_advance(cmd_buf_info.push_constant_buffer, size_of(T))
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

    encoder := mtl_get_compute_encoder(cmd_buf, loc = loc)

    push_constants := Compute_Shader_Push_Constants {
        compute_data = compute_data.ptr,
    }

    mtl_push_constant(cmd_buf, push_constants)
    
    cmd_buf_info := pool_get(&ctx.command_buffers, cmd_buf)
    mtl_cmd_buf := cmd_buf_info.handle

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

    cmd_buf_info, cmd_sync := pool_get_mut(&ctx.command_buffers, cmd_buf); sync.guard(cmd_sync)
    mtl_cmd_buf := cmd_buf_info.handle

    if cmd_buf_info.encoder_type == .Compute
    {
        (cmd_buf_info.encoder.compute)->endEncoding()
    }

    render_pass_desc := objc_alloc(mtl.MTL4RenderPassDescriptor)
    defer render_pass_desc->release()

    color_attachments := render_pass_desc->colorAttachments()
    
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
    cmd_buf_info.render_sample_count = sample_count

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

    cmd_buf_info.scissor = {
        offset = desc.render_area_offset,
        size = { width, height },
    }

    cmd_buf_info.viewport = {
        origin = { 0, 0 },
        size = { f32(width), f32(height) },
        depth_min = 0,
        depth_max = 0,
    }

    cmd_buf_info.raster_state = {
        topology = .Triangle_List,
        cull_mode = .Cull_CW,
        alpha_to_coverage = false,
    }

    cmd_buf_info.depth_state = {
        mode = {},
        compare = .Less,
    }

    cmd_buf_info.blend_state = {
        enable = false,
        color_op = .Add,
        src_color_factor = .Src_Alpha,
        dst_color_factor = .One_Minus_Src_Alpha,
        alpha_op = .Add,
        src_alpha_factor = .One,
        dst_alpha_factor = .One_Minus_Src_Alpha,
        color_write_mask = { .R, .G, .B, .A }
    }

    render_pass_desc->setDefaultRasterSampleCount(ns.UInteger(sampler_count))
    render_pass_desc->setRenderTargetArrayLength(ns.UInteger(layer_count))

    for attach, i in desc.color_attachments
    {
        mtl_attach := color_attachments->objectAtIndexedSubscript(ns.UInteger(i))
        mtl_update_render_pass_attachment(attach, mtl_attach)

        mtl_attach->setClearColor({ 
            f64(attach.clear_color.r),
            f64(attach.clear_color.g),
            f64(attach.clear_color.b),
            f64(attach.clear_color.a),
        })
    }

    if depth_attach, has_attach := desc.depth_attachment.?; has_attach
    {
        mtl_depth_attach := render_pass_desc->depthAttachment()
        mtl_update_render_pass_attachment(depth_attach, mtl_depth_attach)
    }

    if stencil_attach, has_attach := desc.stencil_attachment.?; has_attach
    {
        mtl_stencil_attach := render_pass_desc->stencilAttachment()
        mtl_update_render_pass_attachment(stencil_attach, mtl_stencil_attach)
    }

    cmd_buf_info.encoder.render = mtl_cmd_buf->renderCommandEncoderWithDescriptor_(render_pass_desc)
    cmd_buf_info.encoder_type = .Render

    // Raster state
    (cmd_buf_info.encoder.render)->setFrontFacingWinding(.CounterClockwise)
    (cmd_buf_info.encoder.render)->setTriangleFillMode(.Fill)
    (cmd_buf_info.encoder.render)->setDepthClipMode(.Clip)

    // NOTE: Setting this to nil to ensure we receive proper errors if we attempt
    // to invoke any compute shader related calls after the render pass (or even
    // during it)
    cmd_buf_info.compute_shader = nil
}

_cmd_end_render_pass :: proc(cmd_buf: Command_Buffer, loc := #caller_location)
{
    if ctx.validation
    {
        ok := true
        ok &= pool_check(&ctx.command_buffers, cmd_buf, "cmd_buf", loc)
        if !ok do return
    }

    encoder := mtl_get_render_encoder(cmd_buf, loc = loc)

    encoder->endEncoding()

    cmd_buf_info, cmd_sync := pool_get_mut(&ctx.command_buffers, cmd_buf); sync.guard(cmd_sync)

    cmd_buf_info.encoder_type = .None
}

@(private="file")
mtl_set_up_render_state :: proc(cmd_buf_info: Command_Buffer_Info, encoder: ^mtl.MTL4RenderCommandEncoder, loc := #caller_location)
{
    pipeline_handle := hash_cmd_buf_render_state(cmd_buf_info)

    render_pipeline: ^mtl.RenderPipelineState
    depth_stencil_state: ^mtl.DepthStencilState

    if sync.guard(&ctx.lock)
    {
        if pipeline_handle not_in ctx.render_pipelines
        {
            assert(
                cmf_buf_info.vert_shader != nil &&
                cmd_buf_info.frag_shader != nil,
                "Before you can issue this draw call, You have to set the vertex and fragment shaders that'll be used with `cmd_set_shaders`!",
                loc = loc
            )
            
            vert_shader := pool_get(&ctx.shaders, cmd_buf_info.vert_shader.? or_else panic("You haven't set the vertex shader for this render pass!"))
            frag_shader := pool_get(&ctx.shaders, cmd_buf_info.frag_shader.? or_else panic("You haven't set hte fragment shader for this render pass!"))

            shader_source_location := cmd_buf_info.graphics_shader_source_location
    
            assert(!vert_shader.is_compute, "Expected the first shader given to be a vertex shader. Instead, we got a compute shader.", loc = shader_source_location)
            assert(!frag_shader.is_compute, "Expected the second shader given to be a fragment shader. Instead, we got a compute shader.", loc = shader_source_location)
            assert(vert_shader.graphics_type == .Vertex, "Expected the first shader given to be a vertex shader. Instead, we got a fragment shader.", loc = shader_source_location)
            assert(frag_shader.graphics_type == .Fragment, "Expected the second shader given to be a fragment shader. Instead, we got a vertex shader.", loc = shader_source_location)
    
            pipe_desc := objc_alloc(mtl.MTL4RenderPipelineDescriptor)
            defer pipe_desc->release()
    
            pipe_desc->setVertexFunctionDescriptor(vert_shader.handle)
            pipe_desc->setFragmentFunctionDescriptor(frag_shader.handle)
    
            pipe_desc->setRasterSampleCount(ns.UInteger(cmd_buf_info.render_sample_count))

            pipe_desc->setAlphaToCoverageState(.Enabled if cmd_buf_info.raster_state.alpha_to_coverage else .Disabled)

            // Set up blend state
            {
                color_attachments := pipe_desc->colorAttachments()
                blend_attach := color_attachments->objectAtIndexedSubscript(0)
                blend_state := cmd_buf_info.blend_state
    
                blend_attach->setBlendingState(.Enabled if blend_state.enable else .Disabled)

                // RGB Config
                blend_attach->setSourceRGBBlendFactor(to_mtl_blend_factor(blend_state.src_color_factor))
                blend_attach->setDestinationRGBBlendFactor(to_mtl_blend_factor(blend_state.dst_color_factor))
                blend_attach->setRgbBlendOperation(to_mtl_blend_op(blend_state.color_op))

                // Alpha Config
                blend_attach->setSourceAlphaBlendFactor(to_mtl_blend_factor(blend_state.src_alpha_factor))
                blend_attach->setDestinationAlphaBlendFactor(to_mtl_blend_factor(blend_state.dst_alpha_factor))
                blend_attach->setAlphaBlendOperation(to_mtl_blend_op(blend_state.alpha_op))

                blend_attach->setWriteMask(to_mtl_write_mask(blend_state.color_write_mask))
            }

            // NOTE: Needed for compilation, but not really useful for our purposes.
            compile_task_options := objc_alloc(mtl.MTL4CompilerTaskOptions)
            defer compile_task_options->release()
            
            err: ^ns.Error
            render_pipeline = (ctx.shader_compiler)->newRenderPipelineStateWithDescriptor_compilerTaskOptions_error(pipe_desc, compile_task_options, &err)
    
            mtl_ensure(err, "Unable to create new render pipeline with given shaders on line %v, col %v, in '%v'", loc.line, loc.column, loc.file_path)
    
            ctx.render_pipelines[pipeline_handle] = render_pipeline
        }
        else
        {
            render_pipeline, _ = ctx.render_pipelines[pipeline_handle]
        }

        if cmd_buf_info.depth_state not_in ctx.depth_stencil_states
        {
            depth_stencil_desc := objc_alloc(mtl.DepthStencilDescriptor)
            defer depth_stencil_desc->release()

            depth_state := cmd_buf_info.depth_state

            // NOTE: Unlike in vulkan, there does not exist a separate procedure
            // used to mark if depth testing is enabled. Instead, metal makes use 
            // of the '.Always' flag to indicate that it's disabled, simply through 
            // the logically conclusion that if everything passes then it's not 
            // doing it's main job, marking it effectively as disabled.
            depth_stencil_desc->setDepthCompareFunction(
                to_mtl_compare_op(cmd_buf_info.depth_state.compare) if .Read in depth_state.mode else .Always
            )
            depth_stencil_desc->setDepthWriteEnabled(.Write in depth_state.mode)

            depth_stencil_state := (ctx.device)->newDepthStencilStateWithDescriptor(depth_stencil_desc)

            ctx.depth_stencil_states[depth_state] = depth_stencil_state
        }
        else
        {
            depth_stencil_state, _ = ctx.depth_stencil_states[cmd_buf_info.depth_state]
        }
    }

    encoder->setCullMode(to_mtl_cull_mode(cmd_buf_info.raster_state.cull_mode))

    encoder->setRenderPipelineState(render_pipeline)
    encoder->setDepthStencilState(depth_stencil_state)

    encoder->setViewport(mtl.Viewport {
        originX = f64(cmd_buf_info.viewport.origin.x),
        originY = f64(cmd_buf_info.viewport.origin.y),
        
        width   = f64(cmd_buf_info.viewport.size.x),
        height  = f64(cmd_buf_info.viewport.size.y),

        znear   = f64(cmd_buf_info.viewport.depth_min),
        zfar    = f64(cmd_buf_info.viewport.depth_max),
    })

    encoder->setScissorRect(mtl.ScissorRect {
        x      = ns.UInteger(cmd_buf_info.scissor.offset.x),
        y      = ns.UInteger(cmd_buf_info.scissor.offset.y),
        
        width  = ns.UInteger(cmd_buf_info.scissor.size.x),
        height = ns.UInteger(cmd_buf_info.scissor.size.y),
    })

    encoder->setArgumentTable(cmd_buf_info.argument_table, { .StageVertex, .StageFragment })
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

    encoder := mtl_get_render_encoder(cmd_buf, loc = loc)

    cmd_buf_info := pool_get(&ctx.command_buffers, cmd_buf)

    push_constants := Graphics_Shader_Push_Constants {
        vert_data = vertex_data.ptr,
        frag_data = fragment_data.ptr,
        indirect_data = nil,
    }
    mtl_push_constant(cmd_buf, push_constants, loc = loc)
    mtl_set_up_render_state(cmd_buf_info, encoder, loc = loc)

    encoder->drawPrimitives_vertexStart_vertexCount_instanceCount_baseInstance(
        to_mtl_topology(cmd_buf_info.raster_state.topology),
        0,
        ns.UInteger(vertex_count),
        ns.UInteger(instance_count),
        0
    )
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

    /*
        TODO: For this, here is what I'm thinking:
    
        I'll likely have to record a simple draw call making use of the 
        indirect argument buffer, that'll then be stored in a metal
        indirect command buffer (icb). Then to execute it X times from the
        GPU, like we can in Vulkan, we'll have to change the count buffer
        to be structured like this:

        On metal:
        ```
        Draw_Indirect_Count :: struct
        {
            _: u64,
            count: u64,
        }
        ```

        On vulkan:
        ```
        Draw_Indirect_Count :: struct
        {
            count: u32,
        }
        ```

        With maybe a specific type to help "ground" the type you're
        talking about when casting to the 'count' field on this struct.

        This'll prevent us from having to do a blit to sync between the
        vulkan and metal version, as the struct detailed above for metal
        is required for it to be valid (it has to map to a NSRange)
    */
    
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
Swapchain :: struct
{
    acquired: bool,
    layer: ^ca.MetalLayer,

    current_drawable: ^mtl.Drawable,
    current_texture_handle: ^mtl.Texture,
}

@(private="file")
get_buf_offset_from_gpu_ptr :: proc(p: gpuptr) -> (buf: ^mtl.Buffer, offset: u32, ok: bool)
{
    if p == {} do return {}, {}, false

    alloc_impl := transmute(Alloc_Impl_Info) p._impl
    alloc_info := pool_get(&ctx.allocs, alloc_impl.handle)

    buf = alloc_info.buf_handle
    offset = u32(uintptr(p.ptr) - uintptr(alloc_info.gpu))
    return buf, offset, true
}

@(private="file")
get_buf_size_from_gpu_ptr :: proc(p: gpuptr) -> (size: u64, ok: bool)
{
    if p == {} do return {}, false

    alloc_impl := transmute(Alloc_Impl_Info) p._impl
    alloc_info := pool_get(&ctx.allocs, alloc_impl.handle)
    return alloc_info.buf_size, true
}

@(private="file")
reset_advanced_ptr :: proc(p: ^ptr) 
{
    if p == nil do return
    if p^ == {} do return

    alloc_impl := transmute(Alloc_Impl_Info) p._impl
    alloc_info := pool_get(&ctx.allocs, alloc_impl.handle)

    offset := uintptr(p.ptr) - uintptr(alloc_info.gpu)

    if p.cpu != nil
    {
        p.cpu = rawptr(uintptr(p.cpu) - offset)
    }
    p.gpu.ptr = rawptr(uintptr(p.gpu.ptr) - offset)
}

// Command buffers
@(private="file")
mtl_acquire_cmd_buf :: proc(queue: Queue) -> Command_Buffer
{
    tls_ctx := get_tls()
    sync.guard(&ctx.lock)

    // Check whether there is a free command buffer available with a timeline value that is less than or equal to the current semaphore value
    if handle, ok := priority_queue.pop_safe(&tls_ctx.free_buffers[queue]); ok {
        cmd_buf_info, r_lock := pool_get_mut(&ctx.command_buffers, handle.pool_handle); sync.guard(r_lock)

        assert(!cmd_buf_info.recording)

        current_semaphore_value := _semaphore_get_value(ctx.cmd_bufs_sem_vals[queue].sem)

        if current_semaphore_value >= cmd_buf_info.timeline_value {
            cmd_buf_info.recording = true
            cmd_buf_info.queue = queue
            cmd_buf_info.compute_shader = nil
            cmd_buf_info.thread_id = sync.current_thread_id()

            reset_advanced_ptr(&cmd_buf_info.push_constant_buffer)

            // NOTE: When we want to reuse a command buffer, we reset
            // it's associated allocator to make use of the already existing
            // memory pool present.
            // 
            // Additionally, if we reach this point, then the commands stored
            // have most likely already been submitted and executed on the GPU.
            (cmd_buf_info.allocator)->reset()
            
            return handle.pool_handle
        } else {
            priority_queue.push(&tls_ctx.free_buffers[queue], handle)
        }
    }

    // If no free command buffer is available, create a new one
    // 
    cmd_buf_info: Command_Buffer_Info 
    cmd_buf_info.recording = true
    cmd_buf_info.queue = queue
    cmd_buf_info.compute_shader = nil
    cmd_buf_info.thread_id = sync.current_thread_id()
    cmd_buf_info.handle = (ctx.device)->newCommandBuffer()
    cmd_buf_info.allocator = (ctx.device)->newCommandAllocator()

    argument_table_desc := objc_alloc(mtl.MTL4ArgumentTableDescriptor)
    defer argument_table_desc->release()

    // The extra (+1) is for the push constant buffer.
    argument_table_desc->setMaxBufferBindCount(Total_Descriptor_Heap_Buffer_Count + 1)
    
    cmd_buf_info.argument_table = (ctx.device)->newArgumentTableWithDescriptor(argument_table_desc)

    cmd_buf_info.push_constant_buffer = _mem_alloc_raw(Push_Constant_Buffer_Max_Size, 1, 1)
    
    cmd_buf := pool_add(&ctx.command_buffers, cmd_buf_info, {})
    if cmd_buf_info_mut, r_lock := pool_get_mut(&ctx.command_buffers, cmd_buf); sync.guard(r_lock)
    {
        cmd_buf_info_mut.pool_handle = cmd_buf
        append(&tls_ctx.buffers[queue], cmd_buf_info_mut.pool_handle)
    }

    return cmd_buf
}

@(private="file")
mtl_submit_cmd_bufs :: proc(cmd_bufs: []Command_Buffer)
{
    if len(cmd_bufs) <= 0 do return

    // NOTE: Submissions must be performed in order w.r.t the timeline value used.
    sync.guard(&ctx.lock)

    for cmd_buf in cmd_bufs
    {
        cmd_buf_info_mut, _ := pool_get_mut(&ctx.command_buffers, cmd_buf)
        intr.volatile_store(&cmd_buf_info_mut.timeline_value, sync.atomic_add(&ctx.cmd_bufs_sem_vals[cmd_buf_info_mut.queue].val, 1) + 1)
    }

    for cmd_buf in cmd_bufs 
    {
        cmd_buf_info := pool_get(&ctx.command_buffers, cmd_buf)
        cmd_buf_lock := pool_get_lock(&ctx.command_buffers, cmd_buf)
        sync.guard(cmd_buf_lock)

        queue_sem := ctx.cmd_bufs_sem_vals[cmd_buf_info.queue].sem
        mtl_queue_sem := pool_get(&ctx.semaphores, queue_sem)
        mtl_queue := ctx.queues[cmd_buf_info.queue]
        assert(cmd_buf_info.recording)
        assert(cmd_buf_info.thread_id == sync.current_thread_id())

        for wait_sem in cmd_buf_info.wait_sems
        {
            mtl_wait_sem := pool_get(&ctx.semaphores, wait_sem.sem)
            mtl_queue->waitForEvent(mtl_wait_sem, wait_sem.val)
        }

        mtl_queue->commit_count(&cmd_buf_info.handle, 1)

        for signal_sem in cmd_buf_info.signal_sems
        {
            mtl_signal_sem := pool_get(&ctx.semaphores, signal_sem.sem)
            mtl_queue->signalEvent(mtl_signal_sem, signal_sem.val)
        }

        mtl_queue->signalEvent(mtl_queue_sem, u64(cmd_buf_info.timeline_value))
    }

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

@(private="file")
hash_cmd_buf_render_state :: proc(cmd_buf_info: Command_Buffer_Info) -> Render_Pipeline_Handle
{
    render_state := cmd_buf_info.render_state

    // Nullifying some fields so they don't effect the hash when they're 
    // changed by user code. These are fields not relevant towards determining
    // if a new render pipeline needs to be made.
    render_state.depth_state = {}
    render_state.raster_state.topology = .Triangle_List
    render_state.raster_state.cull_mode = .None
    render_state.viewport = {}
    render_state.scissor = {}
    render_state.graphics_shader_source_location = {}

	bytes := mem.ptr_to_bytes(&render_state)

	return Render_Pipeline_Handle(xxhash.XXH64(bytes))
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

@(private="file")
mtl_update_render_pass_attachment :: proc(attach: Render_Attachment, mtl_attach: ^mtl.RenderPassAttachmentDescriptor)
{
    has_output := attach.texture != {}
    has_resolve := attach.resolve_texture != {}

    if has_output
    {
        texture_info := pool_get(&ctx.textures, attach.texture.handle)
        mtl_attach->setTexture(texture_info.handle)
    }

    if has_resolve
    {
        texture_info := pool_get(&ctx.textures, attach.resolve_texture.handle)
        mtl_attach->setResolveTexture(texture_info.handle)
    }

    mtl_attach->setLoadAction(to_mtl_load_op(attach.load_op))
    mtl_attach->setStoreAction(to_mtl_store_op(attach.store_op))
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
    return transmute(Texture_Handle) (cast([2]u64)desc)[1]
}

@(private="file")
texture_descriptor_get_mtl_texture :: #force_inline proc(desc: Texture_Descriptor) -> ^mtl.Texture
{
    return cast(^mtl.Texture) uintptr((cast([2]u64)desc)[0])
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
