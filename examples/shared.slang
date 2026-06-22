// Standard no_gfx_api bindings, you can copy this into your project

module shared;

#define TEXTURE_HEAP_BINDING 0
#define RWTEXTURE_HEAP_BINDING 1
#define SAMPLER_HEAP_BINDING 2
#define BVH_HEAP_BINDING 3

#pragma warning(disable: 39001)

#if defined(__metal__)

// Bindings for metal buffers

// Already the default on metal and it's usage is also disallowed
// when targeting the metal shading language. 
#define NonUniformResourceIndex(idx) idx

[[vk::binding(1,  0)]] public ParameterBlock<Texture2D[]> texture2DHeap : register(t1);
[[vk::binding(2,  0)]] public ParameterBlock<Texture1D[]> texture1DHeap : register(t2);
[[vk::binding(3,  0)]] public ParameterBlock<Texture3D[]> texture3DHeap : register(t3);
[[vk::binding(4,  0)]] public ParameterBlock<TextureCube[]> textureCubeHeap : register(t4);
[[vk::binding(5,  0)]] public ParameterBlock<Texture2DArray[]> texture2DArrayHeap : register(t5);
[[vk::binding(6,  0)]] public ParameterBlock<TextureCubeArray[]> textureCubeArrayHeap : register(t6);
[[vk::binding(7,  0)]] public ParameterBlock<Texture1DArray[]> texture1DArrayHeap : register(t7);

[[vk::binding(8,  0)]] public ParameterBlock<RWTexture2D<float4>[]> texture2DHeapRW : register(t8);
[[vk::binding(9,  0)]] public ParameterBlock<RWTexture1D<float4>[]> texture1DHeapRW : register(t9);
[[vk::binding(10, 0)]] public ParameterBlock<RWTexture3D<float4>[]> texture3DHeapRW : register(t10);
[[vk::binding(11, 0)]] public ParameterBlock<RWTexture2DArray<float4>[]> texture2DArrayHeapRW : register(t11);
[[vk::binding(12, 0)]] public ParameterBlock<RWTexture1DArray<float4>[]> texture1DArrayHeapRW : register(t12);

[[vk::binding(13, 0)]] public ParameterBlock<SamplerState[]> samplerHeap : register(t13);
[[vk::binding(14, 0)]] ParameterBlock<RaytracingAccelerationStructure[]> bvhHeap : register(t14);

#else

// Bindings for vulkan descriptor sets

[[vk::binding(0, TEXTURE_HEAP_BINDING)]] public Texture2D texture2DHeap[];
[[vk::binding(0, TEXTURE_HEAP_BINDING)]] public Texture1D texture1DHeap[];
[[vk::binding(0, TEXTURE_HEAP_BINDING)]] public Texture3D texture3DHeap[];
[[vk::binding(0, TEXTURE_HEAP_BINDING)]] public TextureCube textureCubeHeap[];
[[vk::binding(0, TEXTURE_HEAP_BINDING)]] public Texture2DArray texture2DArrayHeap[];
[[vk::binding(0, TEXTURE_HEAP_BINDING)]] public TextureCubeArray textureCubeArrayHeap[];
[[vk::binding(0, TEXTURE_HEAP_BINDING)]] public Texture1DArray texture1DArrayHeap[];

[[vk::binding(0, RWTEXTURE_HEAP_BINDING)]] public RWTexture2D<float4> texture2DHeapRW[];
[[vk::binding(0, RWTEXTURE_HEAP_BINDING)]] public RWTexture1D<float4> texture1DHeapRW[];
[[vk::binding(0, RWTEXTURE_HEAP_BINDING)]] public RWTexture3D<float4> texture3DHeapRW[];
[[vk::binding(0, RWTEXTURE_HEAP_BINDING)]] public RWTexture2DArray<float4> texture2DArrayHeapRW[];
[[vk::binding(0, RWTEXTURE_HEAP_BINDING)]] public RWTexture1DArray<float4> texture1DArrayHeapRW[];

[[vk::binding(0, SAMPLER_HEAP_BINDING)]] public SamplerState samplerHeap[];
[[vk::binding(0, BVH_HEAP_BINDING)]] RaytracingAccelerationStructure bvhHeap[];

#endif

// Push constant blocks - different structures for graphics vs compute shaders
public struct __PushConstants_Graphics
{
    public uint64_t vertData;
    public uint64_t fragData;
    public uint64_t indirectData;
};

public struct __PushConstants_Compute
{
    public uint64_t computeData;
};

[[vk::push_constant]] __PushConstants_Graphics __push_graphics;
[[vk::push_constant]] __PushConstants_Compute __push_compute;

// Standard Vulkan indirect draw command structure
public struct DrawIndexedIndirectCommand
{
    uint indexCount;
    uint instanceCount;
    uint firstIndex;
    int  vertexOffset;
    uint firstInstance;
};

struct __DrawCommand<TUserData>
{
    DrawIndexedIndirectCommand cmd;
    TUserData data;
};

in int draw_id : SV_DrawIndex;

public struct GraphicsPipelineData<TVertexData = void, TFragmentData = void, TDrawData = void> {
    public property vert : TVertexData* {
        [ForceInline]
        get { return (TVertexData*)__push_graphics.vertData; }
    }

    public property frag : TFragmentData* {
        [ForceInline]
        get { return (TFragmentData*)__push_graphics.fragData; }
    }

    public property drawData : TDrawData {
        [ForceInline]
        get { 
            __DrawCommand<TDrawData>* buffer = (__DrawCommand<TDrawData>*)__push_graphics.indirectData;
            return buffer[draw_id].data; 
        }
    }

    public property drawCmd : DrawIndexedIndirectCommand {
        [ForceInline]
        get { 
            __DrawCommand<TDrawData>* buffer = (__DrawCommand<TDrawData>*)__push_graphics.indirectData;
            return buffer[draw_id].cmd; 
        }
    }
}

public struct ComputePipelineData<TData = void> {
    public property data : TData* {
        [ForceInline]
        get { return (TData*)__push_compute.computeData; }
    }
}

public SamplerState getSampler(uint samplerIndex) 
{
    return samplerHeap[NonUniformResourceIndex(samplerIndex)];
}

public Texture2D getTexture2D(uint textureIndex)
{
    return texture2DHeap[NonUniformResourceIndex(textureIndex)];
}

public Texture3D getTexture3D(uint textureIndex)
{
    return texture3DHeap[NonUniformResourceIndex(textureIndex)];
}

public Texture1D getTexture1D(uint textureIndex)
{
    return texture1DHeap[NonUniformResourceIndex(textureIndex)];
}

public TextureCube getTextureCube(uint textureIndex)
{
    return textureCubeHeap[NonUniformResourceIndex(textureIndex)];
}

public Texture2DArray getTexture2DArray(uint textureIndex)
{
    return texture2DArrayHeap[NonUniformResourceIndex(textureIndex)];
}

public TextureCubeArray getTextureCubeArray(uint textureIndex)
{
    return textureCubeArrayHeap[NonUniformResourceIndex(textureIndex)];
}

public Texture1DArray getTexture1DArray(uint textureIndex)
{
    return texture1DArrayHeap[NonUniformResourceIndex(textureIndex)];
}

public RWTexture2D<float4> getTexture2DRW(uint textureIndex)
{
    return texture2DHeapRW[NonUniformResourceIndex(textureIndex)];
}

public RWTexture1D<float4> getTexture1DRW(uint textureIndex)
{
    return texture1DHeapRW[NonUniformResourceIndex(textureIndex)];
}

public RWTexture3D<float4> getTexture3DRW(uint textureIndex)
{
    return texture3DHeapRW[NonUniformResourceIndex(textureIndex)];
}

public RWTexture2DArray<float4> getTexture2DArrayRW(uint textureIndex)
{
    return texture2DArrayHeapRW[NonUniformResourceIndex(textureIndex)];
}

public RWTexture1DArray<float4> getTexture1DArrayRW(uint textureIndex)
{
    return texture1DArrayHeapRW[NonUniformResourceIndex(textureIndex)];
}

public RaytracingAccelerationStructure getBVH(uint bvhIndex)
{
    return bvhHeap[NonUniformResourceIndex(bvhIndex)];
}
