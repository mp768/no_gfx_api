#+build darwin
package gpu

import "core:slice"
import "core:log"
import "base:runtime"
import "core:sync"
import "core:dynlib"
import "core:container/priority_queue"
import "core:strings"
import "core:fmt"
import "core:os"
import intr "base:intrinsics"

import mtl "vendor:darwin/Metal"
import ca "vendor:darwin/QuartzCore"
import ns "core:sys/darwin/Foundation"
import cf "core:sys/darwin/CoreFoundation"

@(private="file")
Max_Textures :: 65535
@(private="file")
Max_BVHs :: 65535

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

@(private="file")
Alloc_Handle :: distinct Handle

@(private="file")
Context :: struct
{
    validation: bool,
    device: ^mtl.Device,

    features: Features,

    // Resource pools
    allocs: Resource_Pool(Alloc_Handle, Alloc_Info),
    queues: [Queue]^mtl.CommandQueue,
    semaphores: Resource_Pool(Semaphore, ^mtl.SharedEvent),
    textures: Resource_Pool(Texture_Handle, Texture_Info),
    shaders: Resource_Pool(Shader, Shader_Info),
    command_buffers: Resource_Pool(Command_Buffer, Command_Buffer_Info),
    samplers: [dynamic]Sampler_Info,
    residency_set: ^MTLResidencySet,

    // Swapchain (MetalLayer)
    swapchain: Swapchain,
    frames_in_flight: u32,

    lock: sync.Atomic_Mutex, // Ensures thread-safe access to ctx and MTL operations

    // Ensures thread-safe access to resources relevant to the swapchain.
    // 
    // Separate from the regular lock to allow other work to be done while waiting
    // on swapchain (MetalLayer) specific operations.
    swapchain_lock: sync.Atomic_Mutex, 

    tls_contexts: [dynamic]^Thread_Local_Context,
}

@(private="file")
Thread_Local_Context :: struct 
{
    current_shaders: [enum { Compute, Vertex, Fragment }]Shader_Info,
}

@(private="file")
Texture_Info :: struct { handle: ^mtl.Texture }

@(private="file")
Sampler_Info :: struct { handle: ^mtl.SamplerState }

@(private="file")
Shader_Info :: struct 
{
    handle: ^mtl.Function,

    // Metadata...
    is_compute: bool,
    group_size_x, group_size_y, group_size_z: u32,
    
    graphics_type: Shader_Type_Graphics, 
}

@(private="file")
Command_Buffer_Info :: struct 
{
    handle: ^mtl.CommandBuffer,
    encoder: union {
        ^mtl.BlitCommandEncoder,
        ^mtl.ComputeCommandEncoder,
        ^mtl.RenderCommandEncoder,
    },
    thread_id: int,
    queue: Queue,

    shader_set: bool,

    // These are fields related to setting up the descriptor heap for the 
    // current command buffer. This allow us to have a handle on these resources in
    // our shaders.
    textures, textures_rw, samplers, bvhs: ^mtl.Buffer,

    wait_sems, signal_sems: [dynamic]Semaphore_Value,
}

@(private="file")
Semaphore_Value :: struct
{
    sem: Semaphore,
    val: u64,
}

@(private="file")
Swapchain :: struct 
{
    // Before we can do anthing with presentation, we MUST acquire an image
    // from the swapchain.
    acquired: bool,
    
    layer: ^ca.MetalLayer,
    current_drawable: ^ca.MetalDrawable,
    current_texture_handle: Texture_Handle,
}

@(private="file")
Alloc_Info :: struct
{
    buf_handle: ^mtl.Buffer,
    heap_handle: ^mtl.Heap, // Only populated for `Allocation_Type.Descriptors` allocations
    
    cpu, gpu: rawptr,
    
    align: u32,
    
    alloc_type: Allocation_Type,
    buf_size: u64,
}

Alloc_Impl_Info :: struct
{
    range_end: rawptr,
    handle: Alloc_Handle,
}

@(private="file")
ctx: Context

@(private="file")
mtl_logger: log.Logger

@(require_results)
_init :: proc(validation := true, loc := #caller_location) -> bool
{
    // NOTE: Useful for taking care of resources managed by the objective-c
    // runtime, which is pretty much most stuff with Metal.

    // ns.scoped_autoreleasepool()
    
    scratch, _ := acquire_scratch()

    mtl_logger = context.logger
    ctx.validation = validation

    // For metal, validation layers are set up through environment variables.
    when ODIN_PLATFORM_SUBTARGET == .Default {
        os.set_env("MTL_DEBUG_LAYER",       "1" if validation else "0");
        os.set_env("MTL_SHADER_VALIDATION", "1" if validation else "0");
    }
        
    ctx.device = mtl.CreateSystemDefaultDevice()

    // If we want to do proper `bindless` rendering techniques, we need our 
    // argument buffers to be tier 2.
    ensure(ctx.device->argumentBuffersSupport() == .Tier2, "Tier 2 Argument Buffer support is expected for `no_gfx`")

    {
        tex_rw_support := ctx.device->readWriteTextureSupport()
        // NOTE(MP): Do we want to have tier2 support only? It supports a larger subset
        // of image formats, but tier1 supports some at least (which may be good enough).
        // I don't know at this point...
        ensure(
            tex_rw_support == .Tier1 ||
            tex_rw_support == .Tier2,
            "Tier 1 or Higher is expected for read/write texture support with `no_gfx`"
        )
    }

    // Set up individual queues for different tasks.
    {
        ctx.queues[.Main]     = ctx.device->newCommandQueue()
        ctx.queues[.Compute]  = ctx.device->newCommandQueue()
        ctx.queues[.Transfer] = ctx.device->newCommandQueue()
    }
    
    // TODO: Properly implement logic for checking raytracing support
    // beyond this...
    if ctx.device->supportsRaytracing() /* && ctx.device->supportsFunctionPointers() /* (maybe?) */ */ {
        ctx.features += { .Raytracing }
    }

    // Resource pools
    pool_init(&ctx.allocs)
    pool_init(&ctx.semaphores)
    pool_init(&ctx.textures)
    pool_init(&ctx.shaders)

    residency_desc := MTLResidencySetDescriptor_alloc()
    residency_desc->init()
    defer residency_desc->release()

    ctx.residency_set = MTLDevice_makeResidencySet(residency_desc)
}

_cleanup :: proc(loc := #caller_location) 
{
    for queue in ctx.queues {
        queue->release()
    }
    
    ctx.device->release()
} 

_queue_wait_idle :: proc(queue: Queue)
{
    // NOTE(MP): This is a remanant of porting over the logic from the vulkan
    // implementation. I'm not too sure why we would guard around this resource,
    // since it's already thread-safe by itself. It likely has something to do
    // with signaling that no work should be done while we tell it to idle.
    sync.guard(&ctx.lock)
    
    // NOTE(MP): This is a weird way of forcing the queue to wait, where we create
    // a buffer, submit bogus work, and then wait on that to complete (and since
    // it's the latest submission, it'll be at the end of the queue, thus it'll
    // wait on everything before it to complete too).
    cmd_buf := ctx.queues[queue]->commandBuffer()

    cmd_buf->commit()

    cmd_buf->waitUntilCompleted()
}

_wait_idle :: proc() 
{   
    sync.guard(&ctx.lock)
    
    // NOTE(MP): A sort of hack that forces every queue we have to wait by submitting
    // a bogus piece of work that then will wait on all previous things submitted
    // as well. 
    // 
    // At least, that's the hope. I haven't tried it unfortunately, but this is
    // my best guess at attempting to replicate vk.DeviceWaitIdle's functionality
    // here in metal.
    for queue in ctx.queues {
        cmd_buf := queue->commandBuffer()
    
        cmd_buf->commit()
    
        cmd_buf->waitUntilCompleted()
    }
}

_swapchain_init :: proc(surface_ptr: rawptr, init_size: [2]u32, frames_in_flight: u32) 
{
    mtl_layer := cast(^ca.MetalLayer)surface_ptr;

    mtl_layer->setDrawableSize({ 
        width = cast(cf.CGFloat)init_size[0], 
        height = cast(cf.CGFloat)init_size[1] 
    })

    mtl_layer->setDevice(ctx.device)
    
    if sync.guard(&ctx.lock) {
        // NOTE(MP): According to the metal documentation, t seems that the
        // max number of drawables is 2-3 only. So, we can't have `frames_in_flight`
        // above 3.
        assert(frames_in_flight >= 2 && frames_in_flight <= 3, "When initializing the swapchain, frames in flight can only be between 2 and 3 on the metal backend.")
        ctx.frames_in_flight = frames_in_flight

        mtl_layer->setMaximumDrawableCount(ns.UInteger(frames_in_flight))

        // NOTE(MP): Sets this field to `false` on the CAMetalLayer so that it'll
        // wait indefinitely on a call to `nextDrawable`, rather than timeout after
        // 1 seocnd as is the default behavior.
        // 
        // We do this to replicate the behavior present in the vulkan implementation.
        objcMsgSend(
            nil,
            mtl_layer,
            "setAllowsNextDrawableTimeout:",
            false
        )        
        
        ctx.swapchain.layer = mtl_layer
        ctx.swapchain.acquired = false
        ctx.swapchain.current_drawable = nil
        ctx.swapchain.current_texture_handle = nil
    }
} 

_swapchain_resize :: proc(size: [2]u32) 
{
    queue_wait_idle(.Main)
    ctx.swapchain.layer->setDrawableSize({
        width = cast(cf.CGFloat)size[0],
        height = cast(cf.CGFloat)size[1],
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

    texture := drawable->texture()
    // ensure(texture->pixelFormat() == .BGRA8Unorm, "The pixel format for the drawable's texture should be in BGRA8Unorm without SRGB")

    tex_handle := pool_add(&ctx.textures, { handle = texture }, {})

    ctx.swapchain.acquired = true
    ctx.swapchain.current_drawable = drawable
    ctx.swapchain.current_texture_handle = tex_handle

    return Texture {
        dimensions = { u32(texture->width()), u32(texture->height()), 1 },
        format = .BGRA8_Unorm,
        mip_count = u32(texture->mipmapLevelCount()),
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

    // NOTE(MP): Sort of hacky solution to force the queue of commands to wait 
    // on presenting to the drawable until specific conditions are met.

    cmd_buf := mtl_queue->commandBuffer()

    cmd_buf->encodeWaitForEvent(mtl_sem, u64(wait_value))

    // Present our beautiful image to the world!!!!!!
    cmd_buf->presentDrawable(ctx.swapchain.current_drawable)

    cmd_buf->commit()

    // Clean up
    {
        ctx.swapchain.acquired = false
        ctx.swapchain.current_drawable->release()
    
        pool_remove(&ctx.textures, ctx.swapchain.current_texture_handle)
    }
}

_features_available :: proc() -> Features 
{
    return ctx.features
}

_device_limits :: proc() -> Device_Limits
{
    return Device_Limits {
        // NOTE(MP): This seems to be the cap for metal.
        max_anisotropy = 16.0,
    }
}

// Memory

_mem_alloc_raw :: proc(#any_int el_size, #any_int el_count, #any_int align: i64, mem_type := Memory.Default, alloc_type := Allocation_Type.Default, loc := #caller_location) -> ptr
{
    bytes := el_size * el_count
    if bytes == 0 do return {}

    resource_options: mtl.ResourceOptions

    switch mem_type
    {
        case .Default:
        {
            resource_options = mtl.ResourceStorageModeShared
        }
        case .GPU:
        {
            resource_options = { .StorageModePrivate }
        }
        case .Readback:
        {
            resource_options = mtl.ResourceOptionCPUCacheModeDefault
        }
    }

    buf : ^mtl.Buffer = nil
    heap: ^mtl.Heap   = nil

    switch alloc_type
    {
        case .Default:
        {
            buf = (ctx.device)->newBufferWithLength(ns.UInteger(bytes), resource_options)
        }
        case .Descriptors:
        {
            heap_desc := mtl.HeapDescriptor_alloc()->init()
            defer heap_desc->release()

            heap_desc->setResourceOptions(resource_options)
            heap_desc->setStorageMode(.Shared)
            heap_desc->setSize(ns.UInteger(bytes))
            
            heap = (ctx.device)->newHeap(heap_desc)

            buf = heap->newBufferWithLength(ns.UInteger(bytes), resource_options)
        }
    }

    p: ptr
    if mem_type != .GPU 
    {
        p.cpu = buf->contentsPointer()
    }

    p.gpu.ptr = cast(rawptr) cast(uintptr) buf->gpuAddress()

    alloc_info := Alloc_Info {
        buf_handle = buf,
        heap_handle = heap,
        
        cpu = p.cpu,
        gpu = p.gpu.ptr,
        
        align = u32(align),
        buf_size = u64(bytes),

        alloc_type = alloc_type,
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

    if alloc_info.alloc_type == .Descriptors 
    {
        (alloc_info.heap_handle)->release()
    }
    
    pool_remove(&ctx.allocs, alloc)
}

// Textures

_texture_size_and_align :: proc(desc: Texture_Desc, loc := #caller_location) -> (size: u64, align: u64)
{
    desc_clean := texture_desc_cleanup(desc)

    tex_desc := to_mtl_texture_descriptor(desc_clean)
    defer tex_desc->release()

    _size, _align := (ctx.device)->heapTextureSizeAndAlignWithDescriptor(tex_desc)

    return u64(_size), u64(_align)
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

    alloc_impl := transmute(Alloc_Impl_Info) storage._impl
    alloc_info := pool_get(&ctx.allocs, alloc_impl.handle)

    mtl_buf := alloc_info.buf_handle

    offset := uintptr(storage.ptr) - uintptr(alloc_info.gpu)
    tex_desc := to_mtl_texture_descriptor(desc_clean)
    defer tex_desc->release()

    bytes_per_row := mtl_helper_bytes_per_row(desc.format, desc.dimensions[0])

    // TODO: Figure out something semantically equivalent to the `cmd_add_signal_semaphore`
    // usage in the original vulkan implementaton. Since we don't need to make use of a
    // separate command buffer to handle the creation of an image in the format we want,
    // I don't see why we'd have to wait on a semaphore to perform texture creation on this
    // backend.
    
    mtl_tex := (mtl_buf)->newTexture(tex_desc, ns.UInteger(offset), ns.UInteger(bytes_per_row))

    debug_name_objc := to_mtl_string(name)
    defer debug_name_objc->release()
    
    mtl_tex->setLabel(debug_name_objc)
    
    tex_info := Texture_Info { handle = mtl_tex }
    return Texture {
        dimensions = desc_clean.dimensions,
        format = desc_clean.format,
        mip_count = desc_clean.mip_count,
        sample_count = desc_clean.sample_count,
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

_texture_view_descriptor :: proc(texture: Texture, view_desc: Texture_View_Desc, loc := #caller_location) -> Texture_Descriptor
{
    if ctx.validation
    {
        ok := true
        ok &= pool_check(&ctx.textures, texture.handle, "texture", loc)
        if !ok do return {}
    }

    tex_info := pool_get(&ctx.textures, texture.handle)

    (ctx.residency_set)->addAllocation(tex_info.handle)
    
    return Texture_Descriptor {
        resource_id = u64((tex_info.handle)->gpuResourceID()) 
    }
}

_texture_rw_view_descriptor :: proc(texture: Texture, view_desc: Texture_View_Desc, loc := #caller_location) -> Texture_Descriptor
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
            // NOTE(MP): This seems to be the cap for metal.
            sampler_desc.max_anisotropy <= 16.0,
            "Sampler anisotropy out of range. Call gpu.device_limits() to get the supported maximum anisotropy.",
        )
    }

    desc := mtl.SamplerDescriptor_alloc()
    desc->init()
    defer desc->release()

    desc->setMagFilter(to_mtl_filter(sampler_desc.mag_filter))
    desc->setMinFilter(to_mtl_filter(sampler_desc.min_filter))
    desc->setMipFilter(to_mtl_filter(sampler_desc.mip_filter))

    desc->setSAddressMode(to_mtl_address_mode(sampler_desc.address_mode_u))
    desc->setTAddressMode(to_mtl_address_mode(sampler_desc.address_mode_v))
    desc->setRAddressMode(to_mtl_address_mode(sampler_desc.address_mode_w))

    MTLSamplerDescriptor_setLodBias(desc, ns.Float(sampler_desc.mip_lod_bias))
    desc->setLodMinClamp(sampler_desc.min_lod)
    desc->setLodMaxClamp(sampler_desc.max_lod)

    desc->setMaxAnisotropy(sampler_desc.max_anisotropy)

    sampler := (ctx.device)->newSamplerState(desc)

    if sync.guard(&ctx.lock) {
        append(&ctx.samplers, sampler)
    }

    return Sampler_Descriptor {
        resource_id = u64(sampler->gpuResourceID()) 
    }
}

_texture_view_descriptor_size :: proc() -> u32
{
    return size_of(Texture_Descriptor)
}

_texture_rw_view_descriptor_size :: proc() -> u32 
{
    return size_of(Texture_Descriptor)
}

_sampler_descriptor_size :: proc() -> u32 
{
    return size_of(Sampler_Descriptor)
}

// Shaders 

@(private="file")
_shader_create_internal :: proc(code: []u32, is_compute: bool, graphics_type: Shader_Type_Graphics, entry_point_name := "main", group_size_x: u32 = 1, group_size_y: u32 = 1, group_size_z: u32 = 1, name: string, loc: runtime.Source_Code_Location) -> Shader
{
    // NOTE(MP): I hate this solution, but it's the only way... (maybe)
    shader_source_code := string(slice.reinterpret([]u8, code))

    shader_source_code_objc := ns.String_alloc()
    defer shader_source_code_objc->release()

    shader_source_code_objc->initWithOdinString(shader_source_code)
    
    compile_options := mtl.CompileOptions_alloc()
    compile_options->init()
    defer compile_options->release()

    lib: ^mtl.Library
    err: ^ns.Error
    lib, err = (ctx.device)->newLibraryWithSource(shader_source_code_objc, compile_options)

    if err != nil {
        description := err->localizedDescription()
        if is_compute {
            log.fatalf("Compute shader creation failed for '%v' entry point. Reason: %v\n", entry_point_name, description->odinString())
        } else {
            log.fatalf("Graphics shader creation failed for '%v' entry point on %v stage. Reason: %v\n", entry_point_name, graphics_type, description->odinString())
        }
        return nil
    }
    
    entry_point_name_objc := ns.String_alloc()
    defer entry_point_name_objc->release()

    entry_point_name_objc->initWithOdinString(entry_point_name)
    
    function := lib->newFunctionWithName(entry_point_name_objc)
    defer lib->release()
    
    debug_name_objc := to_mtl_string(name)
    defer debug_name_objc->release()

    function->setLabel(debug_name_objc)

    shader_ref: Shader = pool_add(&ctx.shaders, Shader_Info {
        handle = function,
        is_compute = is_compute,
        group_size_x = group_size_x,
        group_size_y = group_size_y,
        group_size_z = group_size_z,
        graphics_type = graphics_type
    })

    return shader_ref
}

_shader_create :: proc(code: []u32, type: Shader_Type_Graphics, entry_point_name := "main", name := "", loc := #caller_location) -> Shader
{
    return _shader_create_internal(code, false, type, entry_point_name, name=name, loc=loc)
}

_shader_create_compute :: proc(code: []u32, group_size_x: u32, group_size_y: u32 = 1, group_size_z: u32 = 1, entry_point_name := "main", name := "", loc := #caller_location) -> Shader
{
    return _shader_create_internal(code, true, .Vertex, entry_point_name, group_size_x, group_size_y, group_size_z, name=name, loc=loc)
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

    (shader_info.handle)->release()

    pool_remove(&ctx.shaders, shader)
}

// Semaphores

_semaphore_create :: proc(init_value: u64 = 0, name := "", loc := #caller_location) -> Semaphore
{
    event := (ctx.device)->newSharedEvent()
    
    event->setSignaledValue(init_value)
    event->setLabel(ns.String_alloc()->initWithOdinString(name))

    return pool_add(&ctx.semaphores, event, { name = name, created_at = loc })
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
        if !ok do return {}
    }

    mtl_sem := pool_get(&ctx.semaphores, sem)

    SharedEvent_wait(mtl_sem, wait_value, max(u64))
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

// Command buffer

_commands_begin :: proc(queue: Queue, loc := #caller_location) -> Command_Buffer
{
    mtl_queue := ctx.queues[queue]

    mtl_cmd := mtl_queue->commandBuffer()

    cmd_buf_info := Command_Buffer_Info { 
        handle = mtl_cmd,
        encoder = nil,
        thread_id = sync.current_thread_id(),
        queue = queue,

        shader_set = false,
    }

    return pool_add(&ctx.command_buffers, cmd_buf_info)
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

    blit_encoder := mtl_get_blit_encoder(cmd_buf)

    src_buf, src_offset, _ := get_buf_offset_from_gpu_ptr(src)
    dst_buf, dst_offset, _ := get_buf_offset_from_gpu_ptr(dst)

    // Clamp copy regions
    to_copy: uintptr
    if uintptr(src_offset) > uintptr(src_alloc_info.buf_size) || uintptr(dst_offset) > uintptr(dst_alloc_info.buf_size) {
        to_copy = 0
    } else {
        to_copy = min(uintptr(bytes), min(uintptr(src_alloc_info.buf_size) - uintptr(src_offset), uintptr(dst_alloc_info.buf_size) - uintptr(dst_offset)))
    }

    if to_copy <= 0 do return

    blit_encoder->copyFromBuffer(
        src_buf, ns.UInteger(src_offset), 
        dst_buf, ns.UInteger(dst_offset), 
        ns.UInteger(to_copy)
    )
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

    blit_encoder := mtl_get_blit_encoder(cmd_buf)
    tex_info := pool_get(&ctx.textures, dst.handle)

    src_buf, src_offset, ok_s := get_buf_offset_from_gpu_ptr(src)
    assert(ok_s)

    is_compressed := is_block_compressed(dst.format)

    mip_width := max(1, dst.dimensions.x >> region.mip_level)
    mip_height := max(1, dst.dimensions.y >> region.mip_level)
    mip_depth := max(1, dst.dimensions.z >> region.mip_level)

    bytes_per_row := mtl_helper_bytes_per_row(dst.format, dst.dimensions[0])

    bytes_per_image := mtl_helper_bytes_per_image(dst.format, dst.dimensions)

    new_dimensions := mtl_helper_dimensions_by_texture_format(dst.format, { mip_width, mip_height, mip_depth })

    // TODO: Somebody please look over this... I don't know if I've implemented this
    // correctly at all...
    for _index in 0..<region.layer_count { 
        layer := region.base_layer + _index
        
        blit_encoder->copyFromBufferEx(
            src_buf, ns.UInteger(src_offset), 
            ns.UInteger(bytes_per_row), ns.UInteger(bytes_per_image),
            mtl.Size { 
                width = ns.Integer(new_dimensions[0]), 
                height = ns.Integer(new_dimensions[1]),
                depth = ns.Integer(new_dimensions[2]),
            },
    
            tex_info.handle, 
            ns.UInteger(layer),
            ns.UInteger(region.mip_level),
            mtl.Origin { 0, 0, 0 }
        )
    }
}

_cmd_blit_texture :: proc(cmd_buf: Command_Buffer, dst: Texture, dst_rect: Blit_Rect, src: Texture, src_rect: Blit_Rect, filter: Filter, loc := #caller_location)
{
    if ctx.validation
    {
        ok := true
        ok &= pool_check(&ctx.command_buffers, cmd_buf, "cmd_buf", loc)
        ok &= check_cmd_buf_must_be_graphics(cmd_buf, "cmd_buf", loc)
        if !ok do return
    }

    blit_encoder := mtl_get_blit_encoder(cmd_buf)
    src_info := pool_get(&ctx.textures, src.handle)
    dst_info := pool_get(&ctx.textures, dst.handle)

    mtl_filter := to_mtl_filter(filter)

    // TODO(MP): Complete this method later... It's stressing me out. I'll hopefully
    // have a better grasp of this texture stuff later.
    
    // blit_encoder->copyFromTextureWithDestinationOrigin(
        // 
    // )
}

_cmd_set_desc_heap :: proc(cmd_buf: Command_Buffer, textures, textures_rw, samplers, bvhs: gpuptr, loc := #caller_location)
{
    if ctx.validation
    {
        ok := true
        ok &= pool_check(&ctx.command_buffers, cmd_buf, "cmd_buf", loc)
        ok &= check_ptr_allow_nil(textures, "textures", loc)
        ok &= check_ptr_allow_nil(textures_rw, "textures_rw", loc)
        ok &= check_ptr_allow_nil(samplers, "samplers", loc)
        ok &= check_ptr_allow_nil(bvhs, "bvhs", loc)
        if !ok do return
    }

    cmd_info, lock := pool_get_mut(&ctx.command_buffers, cmd_buf); sync.guard(lock)

    if textures != {}
    {
        impl_info := transmute(Alloc_Impl_Info)textures._impl
        alloc_info := pool_get(&ctx.allocs, impl_info.handle)

        cmd_info.textures = alloc_info.buf_handle
    } else do cmd_info.textures = nil

    if textures_rw != {}
    {
        impl_info := transmute(Alloc_Impl_Info)textures_rw._impl
        alloc_info := pool_get(&ctx.allocs, impl_info.handle)

        cmd_info.textures_rw = alloc_info.buf_handle
    } else do cmd_info.textures_rw = nil

    if samplers != {}
    {
        impl_info := transmute(Alloc_Impl_Info)samplers._impl
        alloc_info := pool_get(&ctx.allocs, impl_info.handle)

        cmd_info.samplers = alloc_info.buf_handle
    } else do cmd_info.samplers = nil

    if bvhs != {} && .Raytracing in ctx.features
    {
        impl_info := transmute(Alloc_Impl_Info)bvhs._impl
        alloc_info := pool_get(&ctx.allocs, impl_info.handle)

        cmd_info.bvhs = alloc_info.buf_handle
    } else do cmd_info.bvhs = nil
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

}

//////////////////////////////////////
// Command Helpers

// NOTE(MP): These "get_X_encoder" procs are used because we don't have a set
// "pass" that these types of commands can occur in. Thus, we have to figure out
// if the current command belongs alongside various commands before it on the fly.

@(private="file")
mtl_get_compute_encoder :: proc(cmd: Command_Buffer) -> ^mtl.ComputeCommandEncoder
{
    info, lock := pool_get_mut(&ctx.command_buffers, cmd); sync.guard(lock)

    switch encoder in info.encoder
    {
        case: // do nothing
        case ^mtl.BlitCommandEncoder: encoder->endEncoding()
        case ^mtl.RenderCommandEncoder: encoder->endEncoding()
        case ^mtl.ComputeCommandEncoder:
            return encoder
    }
    
    compute_encoder := (info.handle)->computeCommandEncoder()

    if info.textures != nil {
        compute_encoder->setBuffer(info.textures, 0, 0)
    }

    if info.textures_rw != nil {
        compute_encoder->setBuffer(info.textures_rw, 0, 1)
    }

    if info.samplers != nil {
        compute_encoder->setBuffer(info.samplers, 0, 2)
    }

    if info.bvhs != nil {
        compute_encoder->setBuffer(info.bvhs, 0, 3)
    }
    
    info.encoder = compute_encoder
    return compute_encoder
}

@(private="file")
mtl_get_blit_encoder :: proc(cmd: Command_Buffer) -> ^mtl.BlitCommandEncoder
{
    info, lock := pool_get_mut(&ctx.command_buffers, cmd); sync.guard(lock)

    switch encoder in info.encoder
    {
        case: // do nothing
        case ^mtl.ComputeCommandEncoder: encoder->endEncoding()
        case ^mtl.RenderCommandEncoder: encoder->endEncoding()
        case ^mtl.BlitCommandEncoder:
            return encoder
    }
    
    blit_encoder := (info.handle)->blitCommandEncoder()
    info.encoder = blit_encoder
    return blit_encoder
}

@(private="file")
mtl_end_cmd_buf_encoding :: proc(cmd: Command_Buffer) 
{
    info, lock := pool_get_mut(&ctx.command_buffers, cmd); sync.guard(lock)

    // Unfortunately, since the encoders are wrapped in a union,
    // we can't make use of the generic `endEncoding` method present
    // on all encoders easily (at least not in a safe manner). So,
    // this is how we handle it. 
    switch encoder in info.encoder
    {
        case: // Do nothing
        case ^mtl.BlitCommandEncoder: encoder->endEncoding()
        case ^mtl.RenderCommandEncoder: encoder->endEncoding()
        case ^mtl.ComputeCommandEncoder: encoder->endEncoding()
    }

    info.encoder = nil
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