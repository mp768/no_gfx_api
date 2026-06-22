#+build darwin
#+private
package gpu

// darowind

import "core:log"
import "core:fmt"

import "base:intrinsics"
import mtl "darwodin/Metal"
import ca "darwodin/QuartzCore"
import ns "darwodin/Foundation"
import cf "darwodin/CoreFoundation"

objc_alloc :: proc "contextless" ($T: typeid) -> ^T
{
    return (T{})->alloc()
}

to_mtl_texture_format :: proc(format: Texture_Format) -> mtl.PixelFormat
{
    no_equivalent :: proc(format: Texture_Format, loc := #caller_location) -> !
    {
        panic(fmt.tprintf("No equivalent for texture format '%v' on the metal backend!"), loc=loc)
    }
    
    switch format
    {
        case .Default: panic("Implementation bug!")
        case .R8_Unorm: return .R8Unorm
        case .RG8_Unorm: return .RG8Unorm
        case .RGBA8_Unorm: return .RGBA8Unorm
        case .ABGR8_Unorm: no_equivalent(format)
        case .BGRA8_Unorm: return .BGRA8Unorm
        case .R8_SRGB: return .R8Unorm_sRGB
        case .RG8_SRGB: return .RG8Unorm_sRGB
        case .RGBA8_SRGB: return .RGBA8Unorm_sRGB
        case .ABGR8_SRGB: no_equivalent(format)
        case .BGRA8_SRGB: return .BGRA8Unorm_sRGB
        case .R16_Unorm: return .R16Unorm
        case .RG16_Unorm: return .RG16Unorm
        case .RGBA16_Unorm: return .RGBA16Unorm
        case .D16_Unorm: return .Depth16Unorm
        case .D16_Unorm_S8_Uint: no_equivalent(format)
        case .D24_Unorm_Pack32: no_equivalent(format)
        
        case .D24_Unorm_S8_Uint: 
            log.warn("macOS has depreciated the 'D24_Unorm_S8_Uint' texture format"); 
            return .Depth24Unorm_Stencil8
            
        case .D32_Float: return .Depth32Float
        case .R16_Float: return .R16Float
        case .RG16_Float: return .RG16Float
        case .RGBA16_Float: return .RGBA16Float
        case .R32_Float: return .R32Float
        case .RG32_Float: return .RG32Float
        case .RGBA32_Float: return .RGBA32Float
        case .BC1_RGBA_Unorm: return .BC1_RGBA
        case .BC3_RGBA_Unorm: return .BC3_RGBA
        case .BC4_R_Unorm: return .BC4_RUnorm
        case .BC5_RG_Unorm: return .BC5_RGUnorm
        case .BC6H_RGB_Float: return .BC6H_RGBFloat
        case .BC7_RGBA_Unorm: return .BC7_RGBAUnorm
        case .BC7_RGBA_SRGB: return .BC7_RGBAUnorm_sRGB
        case .ASTC_4x4_RGBA_Unorm: return .ASTC_4x4_LDR
        case .ETC2_RGB8_Unorm: return .ETC2_RGB8
        case .ETC2_RGBA8_Unorm: no_equivalent(format)
        case .EAC_R11_Unorm: return .EAC_R11Unorm
        case .EAC_RG11_Unorm: return .EAC_RG11Unorm
    }
    
    return .Invalid
}

mtl_helper_dimensions_by_texture_format :: proc(format: Texture_Format, dimensions: [3]u32) -> [3]u32
{
    switch format 
    {
        case .Default: panic("Implementation bug!")

        // These don't need to be altered...
        case .RGBA8_Unorm, .BGRA8_Unorm, .RGBA8_SRGB, 
             .D32_Float, .RGBA16_Float, .RGBA32_Float: 
            return dimensions

        // NOTE(MP): Don't know enough to implement these.
        case .BC1_RGBA_Unorm: panic("Dimensions based on texture format for BC1_RGBA_Unorm is unimplemented!")
        case .BC3_RGBA_Unorm: panic("Dimensions based on texture format for BC3_RGBA_Unorm is unimplemented!")
        case .BC7_RGBA_Unorm: panic("Dimensions based on texture format for BC7_RGBA_Unorm is unimplemented!")
        case .ASTC_4x4_RGBA_Unorm: panic("Dimensions based on texture format for ASTC_4x4_RGBA_Unorm is unimplemented!")
        case .ETC2_RGB8_Unorm: panic("Dimensions based on texture format for ETC2_RGB8_Unorm is unimplemented!")
        case .ETC2_RGBA8_Unorm: panic("Dimensions based on texture format for ETC2_RGBA8_Unorm is unimplemented!")
        case .EAC_R11_Unorm: panic("Dimensions based on texture format for EAC_R11_Unorm is unimplemented!")
        case .EAC_RG11_Unorm: panic("Dimensions based on texture format for EAC_RG11_Unorm is unimplemented!")
    }
    
    return dimensions
}

to_mtl_filter :: proc "contextless" (filter: Filter) -> mtl.SamplerMinMagFilter
{
    switch filter
    {
        case .Linear: return .Linear
        case .Nearest: return .Nearest
    }
    
    return .Nearest
}

to_mtl_stage :: #force_inline proc "contextless" (stage: Stage) -> mtl.Stages
{
    switch stage
    {
        case .Transfer: return { .StageBlit }
        case .Compute: return { .StageDispatch }
        case .Raster_Color_Out: return { .StageFragment }
        case .Fragment_Shader: return { .StageFragment }
        case .Vertex_Shader: return { .StageVertex }
        case .Build_BVH: return { .StageAccelerationStructure }
        case .All: return { .StageAll }
    }

    return {}
}

to_mtl_address_mode :: proc "contextless" (mode: Address_Mode) -> mtl.SamplerAddressMode
{
    switch mode
    {
        case .Repeat: return .Repeat
        case .Mirrored_Repeat: return .MirrorRepeat
        case .Clamp_To_Edge: return .ClampToEdge
    }

    return .ClampToZero
}

to_mtl_load_op :: proc "contextless" (op: Load_Op) -> mtl.LoadAction
{
    switch op
    {
        case .Clear: return .Clear  
        case .Load: return .Load    
        case .Dont_Care: return .DontCare   
    }

    return .DontCare
}

to_mtl_store_op :: proc "contextless" (op: Store_Op) -> mtl.StoreAction
{
    switch op
    {
        case .Store: return .Store
        case .Dont_Care: return .DontCare
        case .Resolve: return .MultisampleResolve
        case .Resolve_And_Store: return .StoreAndMultisampleResolve
    }

    return .Unknown
}

to_mtl_topology :: proc(topology: Topology) -> mtl.PrimitiveType
{
    switch topology
    {
        case .Triangle_List: return .Triangle
        case .Triangle_Strip: return .TriangleStrip
        case .Triangle_Fan: panic("'Topology.Triangle_Fan' does not have an equivalent on the metal 4 backend!")
    }

    return .Point
}

to_mtl_blend_factor :: proc "contextless" (blend_factor: Blend_Factor) -> mtl.BlendFactor
{
    switch blend_factor
    {
        case .Zero: return .Zero
        case .One: return .One
        case .Src_Color: return .SourceColor
        case .Dst_Color: return .DestinationColor
        case .Src_Alpha: return .SourceAlpha
        case .Dst_Alpha: return .DestinationAlpha
        case .One_Minus_Src_Alpha: return .OneMinusSourceAlpha
        case .One_Minus_Src_Color: return .OneMinusSourceColor
        case .One_Minus_Dst_Alpha: return .OneMinusDestinationAlpha
        case .One_Minus_Dst_Color: return .OneMinusDestinationColor
    }

    return .Zero
}

to_mtl_blend_op :: proc "contextless" (blend_op: Blend_Op) -> mtl.BlendOperation
{
    switch blend_op
    {
        case .Add: return .Add
        case .Subtract: return .Subtract
        case .Rev_Subtract: return .ReverseSubtract
        case .Min: return .Min
        case .Max: return .Max
    }

    return .Unspecialized
}

to_mtl_write_mask :: proc "contextless" (mask: Color_Component_Flags) -> mtl.ColorWriteMasks
{
    res: mtl.ColorWriteMasks
    
    if .R in mask do res |= .Red
    if .G in mask do res |= .Green
    if .B in mask do res |= .Blue
    if .A in mask do res != .Alpha

    return res
}

to_mtl_compare_op :: proc "contextless" (op: Compare_Op) -> mtl.CompareFunction
{
    switch op
    {
        case .Never: return .Never
        case .Less: return .Less
        case .Equal: return .Equal
        case .Less_Equal: return .LessEqual
        case .Greater: return .Greater
        case .Not_Equal: return .NotEqual
        case .Greater_Equal: return .GreaterEqual
        case .Always: return .Always
    }

    return .Never
}

to_mtl_cull_mode :: proc "contextless" (cull_mode: Cull_Mode) -> mtl.CullMode
{
    switch cull_mode
    {
        case .Cull_CW: return .Back
        case .Cull_CCW: return .Front
        case .None: return .None
        case .All: panic("'Cull_Mode.All' does not have an equivalent on the metal 4 backend!")
    }

    return .None
}

to_mtl_string :: proc(str: string) -> ^ns.String 
{
    objc_str: ^ns.String = ns.String_string()

    objc_str->initWithBytes(raw_data(transmute([]byte)str), ns.UInteger(len(str)), ns.UTF8StringEncoding)
    
    return objc_str
}

to_mtl_texture_usage :: proc "contextless" (usage: Usage_Flags) -> mtl.TextureUsage
{
    res: mtl.TextureUsage

    // NOTE(MP): Don't know about this one
    // if .Transfer_Src in usage do res += {   }
    
    if .Sampled in usage do                  res |= .ShaderRead
    if .Storage in usage do                  res |= .ShaderWrite & .ShaderRead
    if .Color_Attachment in usage do         res |= .RenderTarget
    if .Depth_Stencil_Attachment in usage do res |= .RenderTarget
    
    return res
}

to_mtl_texture_descriptor :: proc "contextless" (desc: Texture_Desc) -> ^mtl.TextureDescriptor
{
    res: ^mtl.TextureDescriptor = (mtl.TextureDescriptor{})->alloc()

    res->init()

    res->setMipmapLevelCount(ns.UInteger(desc.mip_count))
    res->setSampleCount(ns.UInteger(desc.sample_count))
    res->setPixelFormat(to_mtl_texture_format(desc.format))

    res->setWidth(ns.UInteger(max(1, desc.dimensions[0])))
    res->setHeight(ns.UInteger(max(1, desc.dimensions[1])))
    res->setDepth(ns.UInteger(max(1, desc.dimensions[2])))

    res->setUsage(to_mtl_texture_usage(desc.usage))

    switch desc.type 
    {
        case .D1_Array:
        {
            res->setTextureType(._1DArray)
            res->setArrayLength(ns.UInteger(desc.layer_count))
        }
        case .D1:
        {
            res->setTextureType(._1D)
        }

        case .D2_Array:
        {
            res->setTextureType(._2DArray)
            res->setArrayLength(ns.UInteger(desc.layer_count))
        }
        case .D2, .Default:
        {
            res->setTextureType(._2D)
        }

        case .D3:
        {
            res->setTextureType(._3D)
        }

        case .Cube_Array:
        {
            res->setTextureType(.CubeArray)
            res->setArrayLength(ns.UInteger(desc.layer_count))
        }
        case .Cube:
        {
            res->setTextureType(.Cube)
        }
    }

    // NOTE(MP): Working with metal 4, resources are going to be treated as untracked anyways...
    res->setHazardTrackingMode(.Untracked)

    return res
}

// The following code below was adapted/ported from the SDL3 codebase. As such, the
// original copyright notice is maintained for this section.
/*
  Simple DirectMedia Layer
  Copyright (C) 1997-2026 Sam Lantinga <slouken@libsdl.org>

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software
     in a product, an acknowledgment in the product documentation would be
     appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.
*/

// adapted from:
// https://github.com/libsdl-org/SDL/blob/7bbd9d5c2c6925e5824de4288a1ed0bed0c0e5f7/src/gpu/SDL_sysgpu.h#L604
mtl_helper_bytes_per_row :: proc "contextless" (format: Texture_Format, width: u32) -> u32 
{
    block_width := mtl_helper_texture_format_block_width(format)
    blocks_per_row := (width + block_width - 1) / block_width

    texel_size := mtl_helper_texture_format_texel_block_byte_size(format)

    return blocks_per_row * texel_size
}

// adapted from:
// https://github.com/libsdl-org/SDL/blob/7bbd9d5c2c6925e5824de4288a1ed0bed0c0e5f7/src/gpu/SDL_gpu.c#L3518
mtl_helper_bytes_per_image :: proc "contextless" (format: Texture_Format, width, height, depth_or_layers: u32) -> u32
{
    block_width := max(mtl_helper_texture_format_block_width(format), 1)
    block_height := max(mtl_helper_texture_format_block_height(format), 1)

    blocks_per_row := (width + block_width - 1) / block_width
    blocks_per_column := (width + block_height - 1) / block_height

    texel_size := mtl_helper_texture_format_texel_block_byte_size(format)

    return depth_or_layers * blocks_per_row * blocks_per_column * texel_size
}

// adapted from:
// https://github.com/libsdl-org/SDL/blob/7bbd9d5c2c6925e5824de4288a1ed0bed0c0e5f7/src/gpu/SDL_sysgpu.h#L157
mtl_helper_texture_format_block_width :: proc "contextless" (format: Texture_Format) -> u32 
{
    switch format
    {
        case .Default: panic("implementation bug!")
    
        case .R8_Unorm, .RG8_Unorm, .RGBA8_Unorm, .ABGR8_Unorm, .BGRA8_Unorm,
            .R8_SRGB, .RG8_SRGB, .RGBA8_SRGB, .ABGR8_SRGB, .BGRA8_SRGB, .R16_Unorm, 
            .RG16_Unorm, .RGBA16_Unorm, .D16_Unorm, .D16_Unorm_S8_Uint, .D24_Unorm_Pack32,
            .D24_Unorm_S8_Uint, .D32_Float, .R16_Float, .RG16_Float, .RGBA16_Float, .R32_Float,
            .RG32_Float, .RGBA32_Float:
        return 1

        case .BC1_RGBA_Unorm, .BC3_RGBA_Unorm, .BC4_R_Unorm, .BC5_RG_Unorm, .BC6H_RGB_Float,
            .BC7_RGBA_Unorm, .BC7_RGBA_SRGB, .ASTC_4x4_RGBA_Unorm, .ETC2_RGB8_Unorm, .ETC2_RGBA8_Unorm,
            .EAC_R11_Unorm, .EAC_RG11_Unorm:
        return 4
    }

    return 0
}

// adapted from:
// https://github.com/libsdl-org/SDL/blob/7bbd9d5c2c6925e5824de4288a1ed0bed0c0e5f7/src/gpu/SDL_sysgpu.h#L278
mtl_helper_texture_format_block_height :: proc "contextless" (format: Texture_Format) -> u32
{
    switch format
    {
        case .R8_Unorm, .RG8_Unorm, .RGBA8_Unorm, .ABGR8_Unorm, .BGRA8_Unorm,
            .R8_SRGB, .RG8_SRGB, .RGBA8_SRGB, .ABGR8_SRGB, .BGRA8_SRGB, .R16_Unorm, 
            .RG16_Unorm, .RGBA16_Unorm, .D16_Unorm, .D16_Unorm_S8_Uint, .D24_Unorm_Pack32,
            .D24_Unorm_S8_Uint, .D32_Float, .R16_Float, .RG16_Float, .RGBA16_Float, .R32_Float,
            .RG32_Float, .RGBA32_Float:
        return 1

        case .BC1_RGBA_Unorm, .BC3_RGBA_Unorm, .BC4_R_Unorm, .BC5_RG_Unorm, .BC6H_RGB_Float,
            .BC7_RGBA_Unorm, .BC7_RGBA_SRGB, .ASTC_4x4_RGBA_Unorm, .ETC2_RGB8_Unorm, .ETC2_RGBA8_Unorm,
            .EAC_R11_Unorm, .EAC_RG11_Unorm:
        return 4
    }

    return 0
}

// adapted from:
// https://github.com/libsdl-org/SDL/blob/7bbd9d5c2c6925e5824de4288a1ed0bed0c0e5f7/src/gpu/SDL_gpu.c#L814
mtl_helper_texture_format_texel_block_byte_size :: proc "contextless" (format: Texture_Format) -> u32
{
    switch format 
    {
        case .Default: panic("Implementation bug!")

        case .R8_Unorm, .R8_SRGB:
            return 1

        case .RG8_Unorm, .RG8_SRGB, .R16_Unorm, .D16_Unorm, .R16_Float:
            return 2

        case .RGBA8_Unorm, .ABGR8_Unorm, .BGRA8_Unorm, .RGBA8_SRGB,
            .ABGR8_SRGB, .BGRA8_SRGB, .RG16_Unorm, .D16_Unorm_S8_Uint, 
            .D24_Unorm_Pack32, .D24_Unorm_S8_Uint, .D32_Float, .RG16_Float,
            .R32_Float:
            return 4
        
        case .RGBA16_Unorm, .RGBA16_Float, .RG32_Float:
            return 8

        case .RGBA32_Float:
            return 16

        case .BC1_RGBA_Unorm, .BC3_RGBA_Unorm, .BC4_R_Unorm:
            return 8

        case .BC5_RG_Unorm, .BC6H_RGB_Float, .BC7_RGBA_Unorm, .BC7_RGBA_SRGB:
            return 16
            
        case .ASTC_4x4_RGBA_Unorm:
            return 16

        // Relevant sizing information came from: 
        // https://github.com/KhronosGroup/3D-Formats-Guidelines/blob/main/KTXDeveloperGuide.md
        case .ETC2_RGB8_Unorm, .EAC_R11_Unorm:
            return 8

        case .ETC2_RGBA8_Unorm, .EAC_RG11_Unorm:
            return 16
    }
    
    return 0
}