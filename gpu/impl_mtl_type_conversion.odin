#+build darwin 
#+private
package gpu

import "base:intrinsics"
import mtl "vendor:darwin/Metal"
import ca "vendor:darwin/QuartzCore"
import ns "core:sys/darwin/Foundation"
import cf "core:sys/darwin/CoreFoundation"

to_mtl_texture_format :: proc(format: Texture_Format) -> mtl.PixelFormat 
{
    switch format 
    {
        case .Default: panic("Implementation bug!")
        case .RGBA8_Unorm: return .RGBA8Unorm
        case .BGRA8_Unorm: return .BGRA8Unorm
        case .RGBA8_SRGB: return .RGBA8Unorm_sRGB
        case .D32_Float: return .Depth32Float
        case .RGBA16_Float: return .RGBA16Float
        case .RGBA32_Float: return .RGBA32Float
        case .BC1_RGBA_Unorm: return .BC1_RGBA
        case .BC3_RGBA_Unorm: return .BC3_RGBA
        case .BC7_RGBA_Unorm: return .BC7_RGBAUnorm
        case .ASTC_4x4_RGBA_Unorm: return .ASTC_4x4_sRGB // NOTE(MP): Is there a better choice here???
        case .ETC2_RGB8_Unorm: return .ETC2_RGB8
        case .ETC2_RGBA8_Unorm: return .ETC2_RGB8A1 // NOTE(MP): Not exactly the same, but closest match
        case .EAC_R11_Unorm: return .EAC_R11Unorm 
        case .EAC_RG11_Unorm: return .EAC_RG11Unorm
    }
    
    return .Invalid
}

mtl_helper_bytes_per_row :: proc(format: Texture_Format, width: u32) -> u64
{
    switch format 
    {
        case .Default: panic("Implementation bug!")
        case .RGBA8_Unorm, .BGRA8_Unorm, .RGBA8_SRGB: 
            return size_of(u8) * 4 * width
        case .D32_Float:
            return size_of(f32) * 1 * width
        case .RGBA16_Float: 
            return size_of(f16) * 4 * width 
        case .RGBA32_Float: 
            return size_of(f32) * 4 * width

        // NOTE(MP): Don't know enough to implement these.
        case .BC1_RGBA_Unorm: panic("Bytes per row for BC1_RGBA_Unorm is unimplemented!")
        case .BC3_RGBA_Unorm: panic("Bytes per row for BC3_RGBA_Unorm is unimplemented!")
        case .BC7_RGBA_Unorm: panic("Bytes per row for BC7_RGBA_Unorm is unimplemented!")
        case .ASTC_4x4_RGBA_Unorm: panic("Bytes per row for ASTC_4x4_RGBA_Unorm is unimplemented!")
        case .ETC2_RGB8_Unorm: panic("Bytes per row for ETC2_RGB8_Unorm is unimplemented!")
        case .ETC2_RGBA8_Unorm: panic("Bytes per row for ETC2_RGBA8_Unorm is unimplemented!")
        case .EAC_R11_Unorm: panic("Bytes per row for EAC_R11_Unorm is unimplemented!")
        case .EAC_RG11_Unorm: panic("Bytes per row for EAC_RG11_Unorm is unimplemented!")
    }
    
    return 0
}

mtl_helper_bytes_per_image :: proc(format: Texture_Format, dimensions: [3]u32) -> u64
{  
    // We don't need to calculate this for anything other than 3d images.
    if dimensions.z <= 1
    {
        return 0
    }
    
    switch format 
    {
        case .Default: panic("Implementation bug!")
        case .RGBA8_Unorm, .BGRA8_Unorm, .RGBA8_SRGB: 
            return size_of(u8) * 4 * dimensions.x * dimensions.y
        case .D32_Float:
            return size_of(f32) * 1 * dimensions.x * dimensions.y
        case .RGBA16_Float: 
            return size_of(f16) * 4 * dimensions.x * dimensions.y 
        case .RGBA32_Float: 
            return size_of(f32) * 4 * dimensions.x * dimensions.y

        // NOTE(MP): Don't know enough to implement these.
        case .BC1_RGBA_Unorm: panic("Bytes per image for BC1_RGBA_Unorm is unimplemented!")
        case .BC3_RGBA_Unorm: panic("Bytes per image for BC3_RGBA_Unorm is unimplemented!")
        case .BC7_RGBA_Unorm: panic("Bytes per image for BC7_RGBA_Unorm is unimplemented!")
        case .ASTC_4x4_RGBA_Unorm: panic("Bytes per image for ASTC_4x4_RGBA_Unorm is unimplemented!")
        case .ETC2_RGB8_Unorm: panic("Bytes per image for ETC2_RGB8_Unorm is unimplemented!")
        case .ETC2_RGBA8_Unorm: panic("Bytes per image for ETC2_RGBA8_Unorm is unimplemented!")
        case .EAC_R11_Unorm: panic("Bytes per image for EAC_R11_Unorm is unimplemented!")
        case .EAC_RG11_Unorm: panic("Bytes per image for EAC_RG11_Unorm is unimplemented!")
    }
    
    return 0
}

to_mtl_filter :: proc(filter: Filter) -> mtl.SamplerMinMagFilter
{
    switch filter
    {
        case .Linear: return .Linear
        case .Nearest: return .Nearest
    }
    
    return .Nearest
}

to_mtl_address_mode :: proc(mode: Address_Mode) -> mtl.SamplerAddressMode
{
    switch mode
    {
        case .Repeat: return .Repeat
        case .Mirrored_Repeat: return .MirrorRepeat
        case .Clamp_To_Edge: return .ClampToEdge
    }

    return .ClampToZero
}

to_mtl_string :: proc(str: string) -> ^ns.String 
{
    objc_str := ns.String_alloc()
    return objc_str->initWithOdinString(str)
}

to_mtl_texture_usage :: proc(usage: Usage_Flags) -> mtl.TextureUsage
{
    res: mtl.TextureUsage

    // NOTE(MP): Don't know about this one
    if .Transfer_Src in usage do res += {   }
    
    if .Sampled in usage do                  res += { .ShaderRead }
    if .Storage in usage do                  res += { .ShaderWrite }
    if .Color_Attachment in usage do         res += { .RenderTarget }
    if .Depth_Stencil_Attachment in usage do res += { .RenderTarget }
    
    return res
}

to_mtl_texture_descriptor :: proc(desc: Texture_Desc) -> ^mtl.TextureDescriptor
{
    res := mtl.TextureDescriptor_alloc()
    res->init()

    res->setMipmapLevelCount(ns.UInteger(desc.mip_count))
    res->setSampleCount(ns.UInteger(desc.sample_count))
    res->setPixelFormat(to_mtl_texture_format(desc.format))

    res->setWidth(NS.UInteger(max(1, desc.dimensions[0])))
    res->setHeight(NS.UInteger(max(1, desc.dimensions[1])))
    res->setDepth(NS.UInteger(max(1, desc.dimensions[2])))

    res->setUsage(to_mtl_texture_usage(desc.usage))

    is_array := desc.layer_count > 1

    switch desc.type 
    {
        case .D1:
            if is_array {
                res->setTextureType(.Type1DArray)
                res->setArrayLength(ns.UInteger(desc.layer_count))
            } else {
                res->setTextureType(.Type1D)
            }
            
        case .D2:
            if is_array {
                res->setTextureType(.Type2DArray)
                res->setArrayLength(ns.UInteger(desc.layer_count))
            } else {
                res->setTextureType(.Type2D)
            }

        case .D3:
            // NOTE(MP): Don't know if this is correct...
            if is_array {
                res->setTextureType(.TypeCubeArray)
                res->setArrayLength(ns.UInteger(desc.layer_count))
            } else {
                res->setTextureType(.TypeCube)
            }
    }

    return res
}