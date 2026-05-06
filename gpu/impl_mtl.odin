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
Texture_Info :: struct { handle: ^mtl.Texture }

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

    // Swapchain (MetalLayer)
    swapchain: Swapchain,
    frames_in_flight: u32,

    lock: sync.Atomic_Mutex, // Ensures thread-safe access to ctx and MTL operations

    // Ensures thread-safe access to resources relevant to the swapchain.
    // 
    // Separate from the regular lock to allow other work to be done while waiting
    // on swapchain (MetalLayer) specific operations.
    swapchain_lock: sync.Atomic_Mutex, 
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
    heap_handle: ^mtl.Heap, // Only populated for `.Descriptors` allocations
    
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
        intr.objc_send(
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
    alloc := transmute(Alloc_Handle) addr._impl[0]

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




// Semaphores

// A helper procedure since this wasn't already implemented.
@(private="file")
SharedEvent_wait :: #force_inline proc "c" (self: ^mtl.SharedEvent, untilSignaledValue: u64, timeoutMS: u64) -> mtl.BOOL {
	return intr.objc_send(mtl.BOOL, self, "waitUntilSignaledValue:timeoutMS:", untilSignaledValue, timeoutMS)
}

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