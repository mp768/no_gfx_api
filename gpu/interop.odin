#+build !darwin
package gpu

import vk "vendor:vulkan"

vk_get_instance: proc() -> vk.Instance : _vk_get_instance
vk_get_physical_device: proc() -> vk.PhysicalDevice : _vk_get_physical_device
vk_get_device: proc() -> vk.Device : _vk_get_device
vk_get_queue: proc(queue: Queue) -> vk.Queue : _vk_get_queue
vk_get_queue_family: proc(queue: Queue) -> u32 : _vk_get_queue_family
vk_get_swapchain_image_count: proc() -> u32 : _vk_get_swapchain_image_count
vk_get_command_buffer: proc(cmd_buf: Command_Buffer) -> vk.CommandBuffer : _vk_get_command_buffer
vk_get_image: proc(texture: Texture) -> vk.Image : _vk_get_image
vk_get_buffer: proc(addr: gpuptr) -> (vk.Buffer, u32) : _vk_get_buffer

// To be called before gpu.init(). These extra arguments are thread-local.
vk_add_opt_device_extension: proc(extension: cstring) : _vk_add_opt_device_extension
vk_add_device_extension: proc(extension: cstring) : _vk_add_device_extension
