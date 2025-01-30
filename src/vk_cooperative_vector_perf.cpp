/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>
#include <chrono>
#include <string.h>

#include <vulkan/vulkan.h>

using std::vector;

#define ARRAY_LENGTH(x) (sizeof(x) / sizeof(x[0]))

#define CHECK_RESULT(r) do {    \
    if ((r) != VK_SUCCESS) {    \
        printf("result = %d, line = %d\n", (r), __LINE__);  \
        throw;  \
    }   \
} while (0)

// pasted from Vulkan spec
int32_t findProperties(const VkPhysicalDeviceMemoryProperties* pMemoryProperties,
                       uint32_t memoryTypeBitsRequirement,
                       VkMemoryPropertyFlags requiredProperties) {
    const uint32_t memoryCount = pMemoryProperties->memoryTypeCount;
    for (uint32_t memoryIndex = 0; memoryIndex < memoryCount; ++memoryIndex) {
        const uint32_t memoryTypeBits = (1 << memoryIndex);
        const bool isRequiredMemoryType = memoryTypeBitsRequirement & memoryTypeBits;

        const VkMemoryPropertyFlags properties =
            pMemoryProperties->memoryTypes[memoryIndex].propertyFlags;
        const bool hasRequiredProperties =
            (properties & requiredProperties) == requiredProperties;

        if (isRequiredMemoryType && hasRequiredProperties)
            return static_cast<int32_t>(memoryIndex);
    }

    // failed to find memory type
    return -1;
}

struct {
    const char *typeName;
    uint32_t bits;
} componentTypeInfo[] =
{
    { "float16_t",  16 },
    { "float32_t",  32 },
    { "float64_t",  64 },
    { "int8_t",     8 },
    { "int16_t",    16 },
    { "int32_t",    32 },
    { "int64_t",    64 },
    { "uint8_t",    8 },
    { "uint16_t",   16 },
    { "uint32_t",   32 },
    { "uint64_t",   64 },
};

#define VK_NV_COOPERATIVE_VECTOR_SPEC_VERSION                        1
#define VK_NV_COOPERATIVE_VECTOR_EXTENSION_NAME                      "VK_NV_cooperative_vector"

#define VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_VECTOR_FEATURES_NV ((VkStructureType)1000491000)
#define VK_STRUCTURE_TYPE_CONVERT_COOPERATIVE_VECTOR_MATRIX_INFO_NV  ((VkStructureType)1000491004)
#define VK_COMPONENT_TYPE_FLOAT_E4M3_NV                              ((VkComponentTypeNV)1000491002)
#define VK_COMPONENT_TYPE_FLOAT_E5M2_NV                              ((VkComponentTypeNV)1000491003)

typedef struct VkPhysicalDeviceCooperativeVectorFeaturesNV {
    VkStructureType                       sType;
    void*                                 pNext;
    VkBool32                              cooperativeVector;
    VkBool32                              cooperativeVectorTraining;
} VkPhysicalDeviceCooperativeVectorFeaturesNV;

typedef enum VkCooperativeVectorMatrixLayoutNV {
    VK_COOPERATIVE_VECTOR_MATRIX_LAYOUT_ROW_MAJOR_NV = 0,
    VK_COOPERATIVE_VECTOR_MATRIX_LAYOUT_COLUMN_MAJOR_NV = 1,
    VK_COOPERATIVE_VECTOR_MATRIX_LAYOUT_INFERENCING_OPTIMAL_NV = 2,
    VK_COOPERATIVE_VECTOR_MATRIX_LAYOUT_TRAINING_OPTIMAL_NV = 3,
    VK_COOPERATIVE_VECTOR_MATRIX_LAYOUT_MAX_ENUM_NV = 0x7FFFFFFF
} VkCooperativeVectorMatrixLayoutNV;

typedef struct VkConvertCooperativeVectorMatrixInfoNV {
    VkStructureType                       sType;
    void const*                           pNext;
    size_t                                srcSize;
    VkDeviceOrHostAddressConstKHR         srcData;
    size_t*                               pDstSize;
    VkDeviceOrHostAddressKHR              dstData;
    VkComponentTypeKHR                    srcComponentType;
    VkComponentTypeKHR                    dstComponentType;
    uint32_t                              numRows;
    uint32_t                              numColumns;
    VkCooperativeVectorMatrixLayoutNV     srcLayout;
    size_t                                srcStride;
    VkCooperativeVectorMatrixLayoutNV     dstLayout;
    size_t                                dstStride;
} VkConvertCooperativeVectorMatrixInfoNV;

typedef VkResult (VKAPI_PTR *PFN_vkConvertCooperativeVectorMatrixNV)(VkDevice device, const VkConvertCooperativeVectorMatrixInfoNV *pInfo);

static void GetFloatExpManBits(VkComponentTypeNV dt, uint32_t &expBits, uint32_t &manBits, uint32_t &byteSize)
{
    switch (dt) {
    case VK_COMPONENT_TYPE_FLOAT16_NV:
        expBits = 5;
        manBits = 10;
        byteSize = 2;
        break;
    case VK_COMPONENT_TYPE_FLOAT_E4M3_NV:
        expBits = 4;
        manBits = 3;
        byteSize = 1;
        break;
    case VK_COMPONENT_TYPE_FLOAT_E5M2_NV:
        expBits = 5;
        manBits = 2;
        byteSize = 1;
        break;
    default:
        assert(0);
        break;
    }
}

void setDataFloat(void *ptr, VkComponentTypeNV dataType, uint32_t offset, uint32_t index, float value)
{
    uint8_t *p = (uint8_t *)ptr;
    p += offset;

    switch (dataType) {
    case VK_COMPONENT_TYPE_FLOAT32_NV:
        ((float *)p)[index] = value;
        break;
    case VK_COMPONENT_TYPE_FLOAT16_NV:
    case VK_COMPONENT_TYPE_FLOAT_E4M3_NV:
    case VK_COMPONENT_TYPE_FLOAT_E5M2_NV:
        {
            uint32_t expBits, manBits, byteSize;
            GetFloatExpManBits(dataType, expBits, manBits, byteSize);
            uint32_t signBit = manBits + expBits;

		    uint32_t intVal = *(uint32_t *)&value;
            uint32_t sign = intVal & 0x80000000;
            int32_t exp = intVal & 0x7F800000;
            uint32_t mantissa = intVal & 0x007FFFFF;
            exp >>= 23;
            exp -= (1<<(8-1)) - 1;
            exp += (1<<(expBits-1)) - 1;
            exp &= (1<<expBits) - 1;
            // RTNE:
            if (mantissa & (1<<(23 - manBits))) {
                mantissa++;
            }
            mantissa += (1<<(22 - manBits)) - 1;
            if (mantissa & (1<<23)) {
                exp++;
                mantissa = 0;
            }
            mantissa >>= 23 - manBits;
            sign >>= 31;
            sign <<= signBit;
            exp <<= manBits;
            uint32_t result = sign | exp | mantissa;
            assert(result < (1ULL << (byteSize*8)));
            if (value == 0) {
                memset(&((uint8_t *)p)[index*byteSize], 0, byteSize);
            } else {
                memcpy(&((uint8_t *)p)[index*byteSize], &result, byteSize);
            }
        }
        break;
    default:
        assert(0);
        break;
    }
}

float getDataFloat(void const *ptr, VkComponentTypeNV dataType, uint32_t offset, uint32_t index)
{
    uint8_t *p = (uint8_t *)ptr;
    p += offset;

    switch (dataType) {
    case VK_COMPONENT_TYPE_FLOAT32_NV:
        return ((float *)p)[index];
    case VK_COMPONENT_TYPE_FLOAT16_NV:
    case VK_COMPONENT_TYPE_FLOAT_E4M3_NV:
    case VK_COMPONENT_TYPE_FLOAT_E5M2_NV:
        {
            uint32_t expBits, manBits, byteSize;
            GetFloatExpManBits(dataType, expBits, manBits, byteSize);
		    uint32_t intVal = 0;
            memcpy(&intVal, &((uint8_t *)p)[index*byteSize], byteSize);

            uint32_t signBit = manBits + expBits;
            uint32_t signMask = 1 << signBit;
            uint32_t expMask = ((1 << expBits) - 1) << manBits;

            uint32_t sign = intVal & signMask;
            uint32_t mantissa = intVal & ((1 << manBits) - 1);
            int32_t exp = (intVal & expMask) >> manBits;
            exp -= (1<<(expBits-1)) - 1;
            exp += (1<<(8-1)) - 1;
            exp &= 0xFF;
            exp <<= 23;
            mantissa <<= 23 - manBits;
            sign <<= 31 - signBit;
            uint32_t result = sign | exp | mantissa;
            float ret = (intVal == 0 || intVal == signMask) ? 0.0f : *(float *)&result;
            return ret;
        }
    default:
        assert(0);
        return 0.f;
	}
}

struct Allocation
{
    VkComponentTypeNV dataType;
    VkDeviceSize bufferSize;

    // Create a host- and device-local buffer for each input and output.
    // Descriptors point at the device buffers.
    VkBuffer hostBuffer;
    VkDeviceMemory hostMemory;
    VkBuffer deviceBuffer;
    VkDeviceMemory deviceMemory;
    void *ptr;

    bool isFloatType() const
    {
        switch (dataType)
        {
        default:
            return false;
        case VK_COMPONENT_TYPE_FLOAT16_NV:
        case VK_COMPONENT_TYPE_FLOAT32_NV:
        case VK_COMPONENT_TYPE_FLOAT64_NV:
        case VK_COMPONENT_TYPE_FLOAT_E4M3_NV:
        case VK_COMPONENT_TYPE_FLOAT_E5M2_NV:
            return true;
        }
    }

    void setDataFloat(uint32_t offset, uint32_t index, float value)
    {
        ::setDataFloat(ptr, dataType, offset, index, value);
    }

    float getDataFloat(uint32_t offset, uint32_t index) const
    {
        return ::getDataFloat(ptr, dataType, offset, index);
    }

    void setDataInt(uint32_t i, uint32_t value)
    {
        assert(componentTypeInfo[dataType].bits == 8 || componentTypeInfo[dataType].bits == 32);
        switch (dataType) {
        default: assert(0); // fallthrough
        case VK_COMPONENT_TYPE_UINT8_NV:    ((uint8_t  *)ptr)[i] = (uint8_t)value; break;
        case VK_COMPONENT_TYPE_UINT32_NV:   ((uint32_t *)ptr)[i] = (uint32_t)value; break;
        case VK_COMPONENT_TYPE_SINT8_NV:    ((int8_t   *)ptr)[i] = (int8_t)value; break;
        case VK_COMPONENT_TYPE_SINT32_NV:   ((int32_t  *)ptr)[i] = (int32_t)value; break;
        }
    }

    int64_t getDataInt(uint32_t offset, uint32_t i) const
    {
        uint8_t *p = (uint8_t *)ptr;
        p += offset;
        assert(componentTypeInfo[dataType].bits == 8 || componentTypeInfo[dataType].bits == 32);
	    switch (dataType) {
	    default: assert(0); // fallthrough
	    case VK_COMPONENT_TYPE_UINT8_NV:	return ((uint8_t  *)p)[i];
	    case VK_COMPONENT_TYPE_UINT32_NV:	return ((uint32_t *)p)[i];
	    case VK_COMPONENT_TYPE_SINT8_NV:	return ((int8_t   *)p)[i];
	    case VK_COMPONENT_TYPE_SINT32_NV:	return ((int32_t  *)p)[i];
	    }
    }
};

// create storage for a matrix
void createAllocation(VkDevice device, VkPhysicalDeviceMemoryProperties &memoryProperties, Allocation &a, VkComponentTypeNV dt, uint32_t totalBytes)
{
    VkResult result;

    a.dataType = dt;
    a.bufferSize = totalBytes;

    VkBufferCreateInfo bufferCreateInfo = {
        VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        NULL,
        0,
        a.bufferSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT|VK_BUFFER_USAGE_TRANSFER_SRC_BIT|VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT_EXT,
        VK_SHARING_MODE_EXCLUSIVE,
        0u,
        NULL,
    };

    result = vkCreateBuffer(device, &bufferCreateInfo, NULL, &a.hostBuffer);
    CHECK_RESULT(result);
    result = vkCreateBuffer(device, &bufferCreateInfo, NULL, &a.deviceBuffer);
    CHECK_RESULT(result);

    VkMemoryRequirements memReqs;
    vkGetBufferMemoryRequirements(device, a.hostBuffer, &memReqs);

    int32_t hostIndex = findProperties(&memoryProperties, memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
    int32_t deviceIndex = findProperties(&memoryProperties, memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    VkMemoryAllocateFlagsInfo allocateFlagsInfo = {
        VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,
        NULL,
        VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT,
        0,
    };

    VkMemoryAllocateInfo memAllocateInfo = {
        VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        &allocateFlagsInfo,
        memReqs.size,
        (uint32_t)hostIndex,
    };

    result = vkAllocateMemory(device, &memAllocateInfo, NULL, &a.hostMemory);
    CHECK_RESULT(result);

    memAllocateInfo.memoryTypeIndex = deviceIndex;
    result = vkAllocateMemory(device, &memAllocateInfo, NULL, &a.deviceMemory);
    CHECK_RESULT(result);

    result = vkBindBufferMemory(device, a.hostBuffer, a.hostMemory, 0);
    CHECK_RESULT(result);

    result = vkBindBufferMemory(device, a.deviceBuffer, a.deviceMemory, 0);
    CHECK_RESULT(result);

    result = vkMapMemory(device, a.hostMemory, 0, a.bufferSize, 0, &a.ptr);
    CHECK_RESULT(result);
}

// destroy storage for a matrix
void destroyAllocation(VkDevice device, Allocation &a)
{
    vkDestroyBuffer(device, a.hostBuffer, NULL);
    vkDestroyBuffer(device, a.deviceBuffer, NULL);
    vkFreeMemory(device, a.hostMemory, NULL);
    vkFreeMemory(device, a.deviceMemory, NULL);
}

static uint32_t floorLog2(uint32_t x)
{
    uint32_t ret = 0;
    while (x > 1) {
        x >>= 1;
        ret++;
    }
    return ret;
}

int main(int argc, char *argv[])
{
    bool correctness = false;
    bool raygen = false;
    bool frag = false;
    bool int8 = false;
    bool fp8 = false;
    bool outerproduct = false;

    printf("usage: vk_cooperative_vector_perf.exe [--correctness] [--raygen] [--frag] [--int8] [--fp8] [--outerproduct]\n\n");

    for (int arg = 1; arg < argc; ++arg) {
        if (strcmp(argv[arg], "--correctness") == 0) {
            correctness = true;
            continue;
        }
        if (strcmp(argv[arg], "--raygen") == 0) {
            raygen = true;
            continue;
        }
        if (strcmp(argv[arg], "--frag") == 0) {
            frag = true;
            continue;
        }
        if (strcmp(argv[arg], "--int8") == 0) {
            int8 = true;
            continue;
        }
        if (strcmp(argv[arg], "--fp8") == 0) {
            fp8 = true;
            continue;
        }
        if (strcmp(argv[arg], "--outerproduct") == 0) {
            outerproduct = true;
            continue;
        }

        printf("unexpected option %s\n", argv[arg]);
        exit(-1);
    }

    if (outerproduct && (fp8 || int8)) {
        printf("outerproduct not supported with fp8/int8\n");
        exit(-1);
    }

    // Initialize Vulkan
    VkApplicationInfo applicationInfo = {
        VK_STRUCTURE_TYPE_APPLICATION_INFO,
        NULL,
        "Cooperative vector performance test",
        1,
        "none",
        0,
        VK_MAKE_VERSION(1, 3, 0),
    };

    const char *enabledInstanceExtensions[] = { VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME };
    VkInstanceCreateInfo instanceCreateInfo = {
        VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        NULL,
        0,
        &applicationInfo,
        0,
        NULL,
        1,
        enabledInstanceExtensions,
    };

    VkResult result;
    VkInstance instance;
    result = vkCreateInstance(&instanceCreateInfo, NULL, &instance);
    CHECK_RESULT(result);

    uint32_t numPhysicalDevices = 0;
    vector<VkPhysicalDevice> physicalDevices;

    result = vkEnumeratePhysicalDevices(instance, &numPhysicalDevices, NULL);
    CHECK_RESULT(result);

    physicalDevices.resize(numPhysicalDevices);
    result = vkEnumeratePhysicalDevices(instance, &numPhysicalDevices, &physicalDevices[0]);
    CHECK_RESULT(result);

    int physicalDeviceIndex = -1;

    for (uint32_t i = 0; i < numPhysicalDevices; ++i) {

        uint32_t numExtensions = 0;
        vector<VkExtensionProperties> extensions;

        result = vkEnumerateDeviceExtensionProperties(physicalDevices[i], NULL, &numExtensions, NULL);
        CHECK_RESULT(result);

        extensions.resize(numExtensions);
        result = vkEnumerateDeviceExtensionProperties(physicalDevices[i], NULL, &numExtensions, &extensions[0]);
        CHECK_RESULT(result);

        for (uint32_t j = 0; j < numExtensions; ++j) {
            if (strcmp(extensions[j].extensionName, VK_NV_COOPERATIVE_VECTOR_EXTENSION_NAME) == 0) {
                physicalDeviceIndex = i;
                break;
            }
        }
        if (physicalDeviceIndex != -1) {
            break;
        }
    }

    if (physicalDeviceIndex == -1) {
        printf("couldn't find physical device that supports VK_NV_cooperative_vector\n");
        return 0;
    }
    VkPhysicalDevice physicalDevice = physicalDevices[physicalDeviceIndex];

    PFN_vkConvertCooperativeVectorMatrixNV pfn_vkConvertCooperativeVectorMatrixNV =
        (PFN_vkConvertCooperativeVectorMatrixNV)vkGetInstanceProcAddr(instance, "vkConvertCooperativeVectorMatrixNV");

    VkPhysicalDeviceMemoryProperties memoryProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memoryProperties);

    uint32_t numQueueFamilies = 0;
    vector<VkQueueFamilyProperties> queueFamilies;

    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &numQueueFamilies, NULL);

    queueFamilies.resize(numQueueFamilies);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &numQueueFamilies, &queueFamilies[0]);

    int queueFamilyIndex = -1;

    for (uint32_t i = 0; i < numPhysicalDevices; ++i) {
        if (queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            queueFamilyIndex = i;
            break;
        }
    }
    if (queueFamilyIndex == -1) {
        printf("couldn't find compute queue\n");
        return 0;
    }

    float queuePriority = 1.0f;
    VkDeviceQueueCreateInfo deviceQueueCreateInfo = {
        VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        NULL,
        0,
        (uint32_t)queueFamilyIndex,
        1,
        &queuePriority,
    };

    VkPhysicalDeviceRayTracingPipelinePropertiesKHR rayTracingProperties { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR };
    VkPhysicalDeviceProperties2 properties2 { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &rayTracingProperties };

    vkGetPhysicalDeviceProperties2(physicalDevice, &properties2);

    VkPhysicalDeviceCooperativeVectorFeaturesNV coopVecFeatures = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_VECTOR_FEATURES_NV,
        NULL,
        VK_TRUE, // cooperativeVector
        VK_TRUE, // cooperativeVectorTraining
    };

    VkPhysicalDeviceVulkan12Features vk12features = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES };
    vk12features.pNext  = &coopVecFeatures;
    vk12features.shaderFloat16 = VK_TRUE;
    vk12features.storageBuffer8BitAccess = VK_TRUE;
    vk12features.bufferDeviceAddress = VK_TRUE;

    VkPhysicalDeviceVulkan13Features vk13features = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES };
    vk13features.pNext  = &vk12features;
    vk13features.maintenance4 = VK_TRUE;

    VkPhysicalDeviceRayTracingPipelineFeaturesKHR rayTracingFeatures = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR, // VkStructureType    sType;
        &vk13features, // void*              pNext;
        VK_TRUE, // VkBool32           rayTracingPipeline;
        VK_FALSE, // VkBool32           rayTracingPipelineShaderGroupHandleCaptureReplay;
        VK_FALSE, // VkBool32           rayTracingPipelineShaderGroupHandleCaptureReplayMixed;
        VK_FALSE, // VkBool32           rayTracingPipelineTraceRaysIndirect;
        VK_FALSE, // VkBool32           rayTraversalPrimitiveCulling;
    };

    const char *enabledDeviceExtensions[] = { VK_NV_COOPERATIVE_VECTOR_EXTENSION_NAME,
                                              VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
                                              VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
                                              VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,};
    VkDeviceCreateInfo deviceCreateInfo = {
        VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        &rayTracingFeatures,
        0,
        1,
        &deviceQueueCreateInfo,
        0,
        NULL,
        ARRAY_LENGTH(enabledDeviceExtensions),
        enabledDeviceExtensions,
        NULL,
    };

    VkDevice device;
    result = vkCreateDevice(physicalDevice, &deviceCreateInfo, NULL, &device);
    CHECK_RESULT(result);

    PFN_vkGetBufferDeviceAddress pfn_vkGetBufferDeviceAddress =
        (PFN_vkGetBufferDeviceAddress)vkGetDeviceProcAddr(device, "vkGetBufferDeviceAddress");

    PFN_vkCreateRayTracingPipelinesKHR pfn_vkCreateRayTracingPipelinesKHR =
        (PFN_vkCreateRayTracingPipelinesKHR)vkGetDeviceProcAddr(device, "vkCreateRayTracingPipelinesKHR");

    PFN_vkGetRayTracingShaderGroupHandlesKHR pfn_vkGetRayTracingShaderGroupHandlesKHR =
        (PFN_vkGetRayTracingShaderGroupHandlesKHR)vkGetDeviceProcAddr(device, "vkGetRayTracingShaderGroupHandlesKHR");

    PFN_vkCmdTraceRaysKHR pfn_vkCmdTraceRaysKHR =
        (PFN_vkCmdTraceRaysKHR)vkGetDeviceProcAddr(device, "vkCmdTraceRaysKHR");

    VkQueue queue;
    vkGetDeviceQueue(device, (uint32_t)queueFamilyIndex, 0, &queue);

    // The shaders use one UBO to pass in all the buffer addresses
    VkDescriptorSetLayoutBinding layoutBinding = {};
    layoutBinding.binding = 0;
    layoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    layoutBinding.descriptorCount = 1;
    layoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        NULL,
        0,
        1,
        &layoutBinding,
    };

    VkDescriptorSetLayout descriptorSetLayout;
    result = vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCreateInfo, NULL, &descriptorSetLayout);
    CHECK_RESULT(result);

    VkPushConstantRange pushConstantRange = {
        VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_FRAGMENT_BIT,
        0,
        sizeof(uint32_t),
    };
    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = {
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        NULL,
        0,
        1,
        &descriptorSetLayout,
        1,
        &pushConstantRange
    };

    VkPipelineLayout pipelineLayout;
    result = vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, NULL, &pipelineLayout);
    CHECK_RESULT(result);

    VkDescriptorPoolSize poolSizes[1] = { { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 } };

    VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        NULL,
        VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
        1,
        ARRAY_LENGTH(poolSizes),
        poolSizes,
    };

    VkDescriptorPool descriptorPool;
    result = vkCreateDescriptorPool(device, &descriptorPoolCreateInfo, NULL, &descriptorPool);
    CHECK_RESULT(result);

    VkDescriptorSetAllocateInfo setAllocateInfo = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        NULL,
        descriptorPool,
        1,
        &descriptorSetLayout,
    };

    VkDescriptorSet descriptorSet;
    result = vkAllocateDescriptorSets(device, &setAllocateInfo, &descriptorSet);
    CHECK_RESULT(result);

    VkCommandPoolCreateInfo commandPoolCreateInfo = {
        VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        NULL,
        VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        (uint32_t)queueFamilyIndex,
    };

    VkCommandPool commandPool;
    result = vkCreateCommandPool(device, &commandPoolCreateInfo, NULL, &commandPool);
    CHECK_RESULT(result);

    // The command buffers, one for initializing buffers, one for compute, one
    // for reading back the results. This lets us time the compute work more
    // precisely.
    VkCommandBufferAllocateInfo commandBufferAllocateInfo = {
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        NULL,
        commandPool,
        VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        3,
    };

    VkCommandBuffer commandBuffers[3];
    result = vkAllocateCommandBuffers(device, &commandBufferAllocateInfo, commandBuffers);
    CHECK_RESULT(result);

    VkBuffer sbtBuffer {};
    VkDeviceMemory sbtMemory {};
    void *sbtPtr {};
    VkDeviceAddress sbtDeviceAddress {};

    {
        VkBufferCreateInfo bufferCreateInfo = {
            VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            NULL,
            0,
            rayTracingProperties.shaderGroupHandleSize,
            VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            VK_SHARING_MODE_EXCLUSIVE,
            0u,
            NULL,
        };

        result = vkCreateBuffer(device, &bufferCreateInfo, NULL, &sbtBuffer);
        CHECK_RESULT(result);

        VkMemoryRequirements memReqs;
        vkGetBufferMemoryRequirements(device, sbtBuffer, &memReqs);

        int32_t hostIndex = findProperties(&memoryProperties, memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT);

        VkMemoryAllocateFlagsInfo allocateFlagsInfo = {
            VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,
            NULL,
            VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT,
            0,
        };

        VkMemoryAllocateInfo memAllocateInfo = {
            VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            &allocateFlagsInfo,
            memReqs.size,
            (uint32_t)hostIndex,
        };

        result = vkAllocateMemory(device, &memAllocateInfo, NULL, &sbtMemory);
        CHECK_RESULT(result);

        result = vkBindBufferMemory(device, sbtBuffer, sbtMemory, 0);
        CHECK_RESULT(result);

        result = vkMapMemory(device, sbtMemory, 0, bufferCreateInfo.size, 0, &sbtPtr);
        CHECK_RESULT(result);

        memset(sbtPtr, 0, rayTracingProperties.shaderGroupHandleSize);

        VkBufferDeviceAddressInfoEXT info = {
            VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO_EXT,
            NULL,
            0,
        };
        info.buffer = sbtBuffer;
        sbtDeviceAddress = pfn_vkGetBufferDeviceAddress(device, &info);
    }

    struct TestCase {
        uint32_t numLayers;
        uint32_t layerSize;
        uint32_t networkRepeatShift;
        uint32_t nonUniform;
        bool optimalLayout;
    };

    uint32_t networkRepeats[] = {31, 14, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0};

    std::vector<TestCase> testCases;
    if (outerproduct) {
        for (uint32_t layerSize = 128; layerSize >= 16; layerSize -= 16) {
        for (uint32_t nonUniform = 0; nonUniform <= 1; ++nonUniform) {

            testCases.push_back({1, layerSize, 0, nonUniform, true});
        }
        }
    } else {
        for (int32_t optimalLayout = 1; optimalLayout >= 1; --optimalLayout) {
        for (uint32_t layerSize = 64; layerSize >= 32; layerSize -= 32) {
        for (uint32_t numLayers = 2; numLayers <= 3; ++numLayers) {
        for (uint32_t nonUniform = 0; nonUniform <= 1; ++nonUniform) {
        for (uint32_t networkIdx = 0; networkIdx < ARRAY_LENGTH(networkRepeats); ++networkIdx) {

            testCases.push_back({numLayers, layerSize, networkRepeats[networkIdx], nonUniform, !!optimalLayout});
        }
        }
        }
        }
        }
    }

    for (uint32_t s = 0; s < testCases.size(); ++s) {

        std::string fileName = outerproduct ? std::string("shaders/outerproduct") : std::string("shaders/matvecmul");
        if (raygen) {
            fileName += "ray";
        }
        if (frag) {
            fileName += "frag";
        }
        if (int8) {
            fileName += "s8";
        } else if (fp8) {
            fileName += "fp8";
        }
        fileName = fileName + ".spv";

        // Load and create the shader module.
        std::ifstream spirvfile(fileName.c_str(), std::ios::binary | std::ios::ate);
        std::streampos spirvsize = spirvfile.tellg();
        if ((int)spirvsize == -1) {
            printf("%s not found!\n", fileName.c_str());
            throw;
        }
        spirvfile.seekg(0, std::ios::beg);

        vector<char> spirv(spirvsize);
        spirvfile.read(&spirv[0], spirvsize);

        VkShaderModuleCreateInfo shaderModuleCreateInfo = {
            VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            NULL,
            0,
            spirv.size(),
            (const uint32_t *)&spirv[0],
        };

        VkShaderModule shaderModule;
        result = vkCreateShaderModule(device, &shaderModuleCreateInfo, NULL, &shaderModule);
        CHECK_RESULT(result);

        fileName = "shaders/vert.spv";
        std::ifstream vertfile(fileName.c_str(), std::ios::binary | std::ios::ate);
        spirvsize = vertfile.tellg();
        if ((int)spirvsize == -1) {
            printf("%s not found!\n", fileName.c_str());
            throw;
        }
        vertfile.seekg(0, std::ios::beg);

        vector<char> vertspirv(spirvsize);
        vertfile.read(&vertspirv[0], spirvsize);

        VkShaderModuleCreateInfo vertShaderModuleCreateInfo = {
            VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            NULL,
            0,
            vertspirv.size(),
            (const uint32_t *)&vertspirv[0],
        };

        VkShaderModule vertShaderModule;
        result = vkCreateShaderModule(device, &vertShaderModuleCreateInfo, NULL, &vertShaderModule);
        CHECK_RESULT(result);

        uint32_t numLayers = testCases[s].numLayers;
        uint32_t layerSize = testCases[s].layerSize;
        uint32_t networkRepeatShift = testCases[s].networkRepeatShift;
        bool useOptimal = testCases[s].optimalLayout;
        uint32_t nonUniform = testCases[s].nonUniform;

        VkCooperativeVectorMatrixLayoutNV matrixLayout;
        if (useOptimal) {
            matrixLayout = outerproduct ? VK_COOPERATIVE_VECTOR_MATRIX_LAYOUT_TRAINING_OPTIMAL_NV : VK_COOPERATIVE_VECTOR_MATRIX_LAYOUT_INFERENCING_OPTIMAL_NV;
        } else {
            matrixLayout = VK_COOPERATIVE_VECTOR_MATRIX_LAYOUT_ROW_MAJOR_NV;
        }

        uint32_t workgroupSize = 64;
        uint32_t numWorkgroups = correctness ? 4 : 4096;

        // Scale down the workload for slower tests
        if (!correctness && networkRepeatShift < 8) {
            numWorkgroups /= (256 >> networkRepeatShift);
        }

        uint32_t N = 8;
        uint32_t K = 8;

        printf("size = %dx%d %s\tnonuniform %d\tinvocationsPerMatrix %-10.u \t", numLayers, layerSize, useOptimal ? "optimal" : "rowMajor", nonUniform, 1<<networkRepeatShift);

        // input, matrix, bias, output
        enum {ALLOC_A = 0, ALLOC_B = 1, ALLOC_C = 2, ALLOC_D = 3, NUM_ALLOC = 4};

        Allocation allocations[NUM_ALLOC];

        VkComponentTypeNV inputType = int8 ? VK_COMPONENT_TYPE_SINT8_NV : VK_COMPONENT_TYPE_FLOAT16_NV;
        VkComponentTypeNV matrixType = fp8 ? (VkComponentTypeNV)VK_COMPONENT_TYPE_FLOAT_E4M3_NV : int8 ? VK_COMPONENT_TYPE_SINT8_NV : VK_COMPONENT_TYPE_FLOAT16_NV;
        VkComponentTypeNV outputType = int8 ? VK_COMPONENT_TYPE_SINT32_NV : VK_COMPONENT_TYPE_FLOAT16_NV;

        uint32_t inputElementSize = int8 ? 1 : 2;
        uint32_t outputElementSize = int8 ? 4 : 2;
        uint32_t matrixElementSize = fp8 ? 1 : inputElementSize;
        uint32_t numInvocations = workgroupSize * numWorkgroups;
        uint32_t matrixSize = 0;
        uint32_t biasSize = 0;
        uint32_t inputVectorPaddedElements = ((K + (16/inputElementSize - 1)) & ~(16/inputElementSize - 1));
        uint32_t outputVectorPaddedElements = ((N + (16/outputElementSize - 1)) & ~(16/outputElementSize - 1));
        uint32_t inputSize = inputVectorPaddedElements * workgroupSize * inputElementSize;
        uint32_t outputSize;
        if (outerproduct) {
            size_t dstSize = 0;

            VkConvertCooperativeVectorMatrixInfoNV info =
            {
                VK_STRUCTURE_TYPE_CONVERT_COOPERATIVE_VECTOR_MATRIX_INFO_NV,            // VkStructureType                       sType;
                nullptr,                                                                // void const*                           pNext;
                ~0u,                                                                    // size_t                                srcSize;
                0,                                                                      // VkDeviceOrHostAddressConstKHR         srcData;
                &dstSize,                                                               // size_t*                               pDstSize;
                0,                                                                      // VkDeviceOrHostAddressKHR              dstData;
                matrixType,                                                             // VkComponentTypeKHR                    srcComponentType;
                matrixType,                                                             // VkComponentTypeKHR                    dstComponentType;
                layerSize,                                                              // uint32_t                              numRows;
                layerSize,                                                              // uint32_t                              numColumns;
                VK_COOPERATIVE_VECTOR_MATRIX_LAYOUT_ROW_MAJOR_NV,                       // VkCooperativeVectorMatrixLayoutNV     srcLayout;
                layerSize * matrixElementSize,                                          // size_t                                srcStride;
                matrixLayout,                                                           // VkCooperativeVectorMatrixLayoutNV     dstLayout;
                0,                                                                      // size_t                                dstStride;
            };

            result = pfn_vkConvertCooperativeVectorMatrixNV(device, &info);
            CHECK_RESULT(result);
            outputSize = (uint32_t)dstSize;
        } else {
            outputSize = outputVectorPaddedElements * numInvocations * outputElementSize;
        }

        uint32_t numNetworks = (numInvocations + (1 << networkRepeatShift) - 1) >> networkRepeatShift;

        uint32_t matStrides[4] = {};
        uint32_t matOffsets[4] = {};
        uint32_t matOffsetsOpt[4] = {};
        uint32_t biasOffsets[4] = {};

        for (uint32_t j = 0; j <= numLayers; ++j) {
            uint32_t inSize = (j == 0) ? K : layerSize;
            uint32_t outSize = (j == numLayers) ? N : layerSize;

            matOffsets[j] = matrixSize;
            biasOffsets[j] = biasSize;

            matStrides[j] = (inSize * matrixElementSize + 15) & ~15;

            matrixSize += matStrides[j] * outSize;
            matrixSize = (matrixSize + 63) & ~63;

            if (useOptimal) {
                matOffsetsOpt[j] = matrixSize;

                size_t dstSize = 0;

                VkConvertCooperativeVectorMatrixInfoNV info =
                {
                    VK_STRUCTURE_TYPE_CONVERT_COOPERATIVE_VECTOR_MATRIX_INFO_NV,            // VkStructureType                       sType;
                    nullptr,                                                                // void const*                           pNext;
                    matStrides[j] * outSize,                                                // size_t                                srcSize;
                    0,                                                                      // VkDeviceOrHostAddressConstKHR         srcData;
                    &dstSize,                                                               // size_t*                               pDstSize;
                    0,                                                                      // VkDeviceOrHostAddressKHR              dstData;
                    matrixType,                                                             // VkComponentTypeKHR                    srcComponentType;
                    matrixType,                                                             // VkComponentTypeKHR                    dstComponentType;
                    outSize,                                                                // uint32_t                              numRows;
                    inSize,                                                                 // uint32_t                              numColumns;
                    VK_COOPERATIVE_VECTOR_MATRIX_LAYOUT_ROW_MAJOR_NV,                       // VkCooperativeVectorMatrixLayoutNV     srcLayout;
                    matStrides[j],                                                          // size_t                                srcStride;
                    matrixLayout,                                                           // VkCooperativeVectorMatrixLayoutNV     dstLayout;
                    0,                                                                      // size_t                                dstStride;
                };

                result = pfn_vkConvertCooperativeVectorMatrixNV(device, &info);
                CHECK_RESULT(result);
                matrixSize += (uint32_t)dstSize;
            }
            biasSize += outSize * outputElementSize;
            biasSize = (biasSize + 15) & ~15;
        }

        uint32_t networkMatrixStride = matrixSize;
        uint32_t networkBiasStride = biasSize;

        matrixSize *= numNetworks;
        biasSize *= numNetworks;

        createAllocation(device, memoryProperties, allocations[ALLOC_A], inputType, inputSize);
        createAllocation(device, memoryProperties, allocations[ALLOC_B], matrixType, matrixSize);
        createAllocation(device, memoryProperties, allocations[ALLOC_C], outputType, biasSize);
        createAllocation(device, memoryProperties, allocations[ALLOC_D], outputType, outputSize);

        // Allocate buffer to hold device addresses for the four allocations
        VkBuffer paramBuffer;
        VkDeviceMemory paramMemory;
        void *paramPtr;

        VkBufferCreateInfo bufferCreateInfo = {
            VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            NULL,
            0,
            4*sizeof(VkDeviceAddress),
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VK_SHARING_MODE_EXCLUSIVE,
            0u,
            NULL,
        };

        result = vkCreateBuffer(device, &bufferCreateInfo, NULL, &paramBuffer);
        CHECK_RESULT(result);

        VkMemoryRequirements memReqs;
        vkGetBufferMemoryRequirements(device, paramBuffer, &memReqs);

        int32_t hostIndex = findProperties(&memoryProperties, memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT);

        VkMemoryAllocateFlagsInfo allocateFlagsInfo = {
            VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,
            NULL,
            VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT,
            0,
        };

        VkMemoryAllocateInfo memAllocateInfo = {
            VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            &allocateFlagsInfo,
            memReqs.size,
            (uint32_t)hostIndex,
        };

        result = vkAllocateMemory(device, &memAllocateInfo, NULL, &paramMemory);
        CHECK_RESULT(result);

        result = vkBindBufferMemory(device, paramBuffer, paramMemory, 0);
        CHECK_RESULT(result);

        result = vkMapMemory(device, paramMemory, 0, bufferCreateInfo.size, 0, &paramPtr);
        CHECK_RESULT(result);

        for (int i = 0; i < NUM_ALLOC; ++i) {
            Allocation &a = allocations[i];

            VkBufferDeviceAddressInfoEXT info = {
                VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO_EXT,
                NULL,
                0,
            };
            VkDeviceAddress *addrsInMemory = (VkDeviceAddress *)paramPtr;
            info.buffer = a.deviceBuffer;
            VkDeviceAddress addr = pfn_vkGetBufferDeviceAddress(device, &info);
            addrsInMemory[i] = addr;
        }

        VkDescriptorBufferInfo bufferDescriptor;
        bufferDescriptor.buffer = paramBuffer;
        bufferDescriptor.offset = 0;
        bufferDescriptor.range = bufferCreateInfo.size;

        VkWriteDescriptorSet writeDescriptorset = {
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            NULL,
            descriptorSet,
            0, // dstBinding,
            0, // dstArrayElement
            1, // descriptorCount
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            NULL,
            &bufferDescriptor,
            NULL,
        };

        vkUpdateDescriptorSets(device, 1, &writeDescriptorset, 0, NULL);

        // Skip initializing the allocations for non-correctness tests.
        if (correctness) {
            // Initialize input buffers to random values. These are relatively
            // small and have few mantissa bits set so we don't lose precision
            // in fp16 mode when running the correctness test.
            // Initialize the output buffer to an obvious value.
            for (uint32_t i = 0; i < NUM_ALLOC; ++i) {
                Allocation &a = allocations[i];
                uint32_t elementSize = (i < 2) ? inputElementSize : outputElementSize;
                for (uint32_t j = 0; j < a.bufferSize / elementSize; ++j) {
                    if (a.isFloatType()) {
                        a.setDataFloat(0, j, ((float)(rand() & 0x3) - 1.0f) / 4.0f);
                        if (i == 3) {
                            a.setDataFloat(0, j, outerproduct ? 0.0f : 1234.0f);
                        }
                    } else {
                        a.setDataInt(j, (rand() & 0xff) - 128);
                    }
                }
            }

            for (uint32_t m = 0; m < numNetworks; ++m) {
            for (uint32_t j = 0; j <= numLayers; ++j) {
                uint32_t rawOffset = m * networkMatrixStride + matOffsets[j];
                uint32_t optOffset = m * networkMatrixStride + matOffsetsOpt[j];

                uint32_t inSize = (j == 0) ? K : layerSize;
                uint32_t outSize = (j == numLayers) ? N : layerSize;

                size_t dstSize = allocations[ALLOC_B].bufferSize - optOffset;

                VkConvertCooperativeVectorMatrixInfoNV info =
                {
                    VK_STRUCTURE_TYPE_CONVERT_COOPERATIVE_VECTOR_MATRIX_INFO_NV,            // VkStructureType                       sType;
                    nullptr,                                                                // void const*                           pNext;
                    matStrides[j] * outSize,                                                // size_t                                srcSize;
                    (VkDeviceAddress)allocations[ALLOC_B].ptr + rawOffset,                  // VkDeviceOrHostAddressConstKHR         srcData;
                    &dstSize,                                                               // size_t*                               pDstSize;
                    (VkDeviceAddress)allocations[ALLOC_B].ptr + optOffset,                  // VkDeviceOrHostAddressKHR              dstData;
                    matrixType,                                                             // VkComponentTypeKHR                    srcComponentType;
                    matrixType,                                                             // VkComponentTypeKHR                    dstComponentType;
                    outSize,                                                                // uint32_t                              numRows;
                    inSize,                                                                 // uint32_t                              numColumns;
                    VK_COOPERATIVE_VECTOR_MATRIX_LAYOUT_ROW_MAJOR_NV,                       // VkCooperativeVectorMatrixLayoutNV     srcLayout;
                    matStrides[j],                                                          // size_t                                srcStride;
                    VK_COOPERATIVE_VECTOR_MATRIX_LAYOUT_INFERENCING_OPTIMAL_NV,             // VkCooperativeVectorMatrixLayoutNV     dstLayout;
                    0,                                                                      // size_t                                dstStride;
                };

                result = pfn_vkConvertCooperativeVectorMatrixNV(device, &info);
                CHECK_RESULT(result);
            }
            }
        }
        uint32_t resultShift = floorLog2(layerSize * 128);

        // Specialize the shader with the vector/matrix dimensions, strides, etc.
        const uint32_t specData[] = {
            N,
            K,
            numLayers,
            layerSize,
            workgroupSize,
            networkMatrixStride,
            networkBiasStride,
            (useOptimal ? matOffsetsOpt : matOffsets)[0],
            (useOptimal ? matOffsetsOpt : matOffsets)[1],
            (useOptimal ? matOffsetsOpt : matOffsets)[2],
            (useOptimal ? matOffsetsOpt : matOffsets)[3],
            biasOffsets[0],
            biasOffsets[1],
            biasOffsets[2],
            biasOffsets[3],
            (uint32_t)(matrixLayout),
            nonUniform,
            resultShift,
        };

#if 0
        for (int i = 0; i < ARRAY_LENGTH(specData); ++i) {
            printf("specdata[%d] = %d\n", i, specData[i]);
        }
#endif

        VkSpecializationMapEntry entries[] = {
            {0, sizeof(uint32_t) * 0, sizeof(uint32_t)},
            {1, sizeof(uint32_t) * 1, sizeof(uint32_t)},
            {2, sizeof(uint32_t) * 2, sizeof(uint32_t)},
            {3, sizeof(uint32_t) * 3, sizeof(uint32_t)},
            {4, sizeof(uint32_t) * 4, sizeof(uint32_t)},
            {5, sizeof(uint32_t) * 5, sizeof(uint32_t)},
            {6, sizeof(uint32_t) * 6, sizeof(uint32_t)},
            {7, sizeof(uint32_t) * 7, sizeof(uint32_t)},
            {8, sizeof(uint32_t) * 8, sizeof(uint32_t)},
            {9, sizeof(uint32_t) * 9, sizeof(uint32_t)},
            {10, sizeof(uint32_t) * 10, sizeof(uint32_t)},
            {11, sizeof(uint32_t) * 11, sizeof(uint32_t)},
            {12, sizeof(uint32_t) * 12, sizeof(uint32_t)},
            {13, sizeof(uint32_t) * 13, sizeof(uint32_t)},
            {14, sizeof(uint32_t) * 14, sizeof(uint32_t)},
            {15, sizeof(uint32_t) * 15, sizeof(uint32_t)},
            {16, sizeof(uint32_t) * 16, sizeof(uint32_t)},
            {17, sizeof(uint32_t) * 17, sizeof(uint32_t)},
        };

        VkSpecializationInfo specInfo =
        {
            ARRAY_LENGTH(specData),
            entries,
            sizeof(specData),
            specData,
        };

        VkPipelineShaderStageCreateInfo shaderCreateInfo[2] = {
            {
                VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                NULL,
                0,
                raygen ? VK_SHADER_STAGE_RAYGEN_BIT_KHR : frag ? VK_SHADER_STAGE_FRAGMENT_BIT : VK_SHADER_STAGE_COMPUTE_BIT,
                shaderModule,
                "main",
                &specInfo,
            },
            {
                VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                NULL,
                0,
                VK_SHADER_STAGE_VERTEX_BIT,
                vertShaderModule,
                "main",
                nullptr,
            }
        };

        VkRenderPass renderPass {};
        VkFramebuffer framebuffer {};

        VkPipeline pipeline {};

        VkStridedDeviceAddressRegionKHR raygenShaderBindingTableRegion {};
        VkStridedDeviceAddressRegionKHR missShaderBindingTableRegion {};
        VkStridedDeviceAddressRegionKHR hitShaderBindingTableRegion {};
        VkStridedDeviceAddressRegionKHR callableShaderBindingTableRegion {};

        if (raygen) {

            VkRayTracingShaderGroupCreateInfoKHR    shaderGroupCreateInfo    =
            {
                VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
                NULL,
                VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR,
                0,
                VK_SHADER_UNUSED_KHR,
                VK_SHADER_UNUSED_KHR,
                VK_SHADER_UNUSED_KHR,
                NULL,
            };

            VkRayTracingPipelineCreateInfoKHR pipelineCreateInfo = {
                VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                NULL,
                0,
                1,
                &shaderCreateInfo[0],
                1,
                &shaderGroupCreateInfo,
                1,
                NULL,
                NULL,
                NULL,
                pipelineLayout,
                VK_NULL_HANDLE,
                0
            };

            result = pfn_vkCreateRayTracingPipelinesKHR(device, VK_NULL_HANDLE, VK_NULL_HANDLE, 1, &pipelineCreateInfo, NULL, &pipeline);
            CHECK_RESULT(result);

            result = pfn_vkGetRayTracingShaderGroupHandlesKHR(device, pipeline, 0, 1, rayTracingProperties.shaderGroupHandleSize, sbtPtr);
            CHECK_RESULT(result);

            raygenShaderBindingTableRegion.deviceAddress = sbtDeviceAddress;
            raygenShaderBindingTableRegion.stride = rayTracingProperties.shaderGroupHandleSize;
            raygenShaderBindingTableRegion.size = rayTracingProperties.shaderGroupHandleSize;
        } else if (frag) {
            VkSubpassDescription sd = {};
            sd.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
            VkRenderPassCreateInfo rpci = { VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO };
            rpci.subpassCount = 1;
            rpci.pSubpasses = &sd;

            result = vkCreateRenderPass(device, &rpci, nullptr, &renderPass);
            CHECK_RESULT(result);

            VkFramebufferCreateInfo fci = { VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO };
            fci.width = workgroupSize;
            fci.height = numWorkgroups;
            fci.layers = 1;

            result = vkCreateFramebuffer(device, &fci, nullptr, &framebuffer);
            CHECK_RESULT(result);

            VkPipelineVertexInputStateCreateInfo visci = { VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };
            VkPipelineInputAssemblyStateCreateInfo iasci = { VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO };
            iasci.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

            VkPipelineRasterizationStateCreateInfo rsci = { VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO };
            VkPipelineMultisampleStateCreateInfo msci = { VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO };
            msci.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
            VkViewport viewport = { 0, 0, (float)fci.width, (float)fci.height, -1, 1 };
            VkRect2D scissor = { { 0, 0 }, { fci.width, fci.height } };

            VkPipelineViewportStateCreateInfo vsci = { VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO };
            vsci.viewportCount = vsci.scissorCount = 1;
            vsci.pViewports = &viewport;
            vsci.pScissors = &scissor;

            VkPipelineTessellationStateCreateInfo tsci = { VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_STATE_CREATE_INFO };

            VkGraphicsPipelineCreateInfo pipelineCreateInfo = {
                VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
                NULL,
                0,              // VkPipelineCreateFlags                            flags;
                2,              // uint32_t                                         stageCount;
                &shaderCreateInfo[0], // const VkPipelineShaderStageCreateInfo*           pStages;
                &visci,         // const VkPipelineVertexInputStateCreateInfo*      pVertexInputState;
                &iasci,         // const VkPipelineInputAssemblyStateCreateInfo*    pInputAssemblyState;
                &tsci,          // const VkPipelineTessellationStateCreateInfo*     pTessellationState;
                &vsci,          // const VkPipelineViewportStateCreateInfo*         pViewportState;
                &rsci,          // const VkPipelineRasterizationStateCreateInfo*    pRasterizationState;
                &msci,          // const VkPipelineMultisampleStateCreateInfo*      pMultisampleState;
                nullptr,        // const VkPipelineDepthStencilStateCreateInfo*     pDepthStencilState;
                nullptr,        // const VkPipelineColorBlendStateCreateInfo*       pColorBlendState;
                nullptr,        // const VkPipelineDynamicStateCreateInfo*          pDynamicState;
                pipelineLayout, // VkPipelineLayout                                 layout;
                renderPass,     // VkRenderPass                                     renderPass;
                0,              // uint32_t                                         subpass;
                VK_NULL_HANDLE, // VkPipeline                                       basePipelineHandle;
                0,              // int32_t                                          basePipelineIndex;
            };

            result = vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineCreateInfo, nullptr, &pipeline);
            CHECK_RESULT(result);
        } else {
            VkComputePipelineCreateInfo pipelineCreateInfo = {
                VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                NULL,
                0,
                shaderCreateInfo[0],
                pipelineLayout,
                VK_NULL_HANDLE,
                0
            };

            result = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineCreateInfo, NULL, &pipeline);
            CHECK_RESULT(result);
        }

        VkCommandBufferBeginInfo commandBufferBeginInfo = {
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            NULL,
            VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT,
            NULL,
        };

        VkSubmitInfo submitInfo = {
            VK_STRUCTURE_TYPE_SUBMIT_INFO,
            NULL,
            0,
            NULL,
            NULL,
            1,
            &commandBuffers[0],
            0,
            NULL,
        };

        if (correctness) {
            // Download input buffers to device memory.
            result = vkBeginCommandBuffer(commandBuffers[0], &commandBufferBeginInfo);
            CHECK_RESULT(result);

            for (uint32_t i = 0; i < 4; ++i) {
                Allocation &a = allocations[i];
                VkBufferCopy copy = { 0, 0, a.bufferSize };
                vkCmdCopyBuffer(commandBuffers[0], a.hostBuffer, a.deviceBuffer, 1, &copy);
            }

            result = vkEndCommandBuffer(commandBuffers[0]);
            CHECK_RESULT(result);

            submitInfo.pCommandBuffers = &commandBuffers[0];

            result = vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
            CHECK_RESULT(result);
            result = vkQueueWaitIdle(queue);
            CHECK_RESULT(result);
        }

        // Run the shader.
        result = vkBeginCommandBuffer(commandBuffers[1], &commandBufferBeginInfo);
        CHECK_RESULT(result);

        uint32_t pushConst = networkRepeatShift;
        vkCmdPushConstants(commandBuffers[1], pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pushConst), &pushConst);

        uint32_t repeatCount = correctness ? 1 : 500;

        if (raygen) {
            vkCmdBindDescriptorSets(commandBuffers[1], VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipelineLayout, 0u, 1, &descriptorSet, 0u, NULL);
            vkCmdBindPipeline(commandBuffers[1], VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipeline);
            for (uint32_t i = 0; i < repeatCount; ++i) {
                pfn_vkCmdTraceRaysKHR(commandBuffers[1],
                                      &raygenShaderBindingTableRegion,
                                      &missShaderBindingTableRegion,
                                      &hitShaderBindingTableRegion,
                                      &callableShaderBindingTableRegion,
                                      workgroupSize, numWorkgroups, 1);
            }
        } else if (frag) {
            vkCmdBindDescriptorSets(commandBuffers[1], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0u, 1, &descriptorSet, 0u, NULL);
            vkCmdBindPipeline(commandBuffers[1], VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

            VkRenderPassBeginInfo begin = { VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO };
            begin.renderPass = renderPass;
            begin.framebuffer = framebuffer;
            begin.renderArea = { {0, 0}, {workgroupSize, numWorkgroups} };

            vkCmdBeginRenderPass(commandBuffers[1], &begin, VK_SUBPASS_CONTENTS_INLINE);
            vkCmdDraw(commandBuffers[1], 3 * repeatCount, 1, 0, 0);
            vkCmdEndRenderPass(commandBuffers[1]);
        } else {
            vkCmdBindDescriptorSets(commandBuffers[1], VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0u, 1, &descriptorSet, 0u, NULL);
            vkCmdBindPipeline(commandBuffers[1], VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
            for (uint32_t i = 0; i < repeatCount; ++i) {
                vkCmdDispatch(commandBuffers[1], numWorkgroups, 1, 1);
            }
        }

        result = vkEndCommandBuffer(commandBuffers[1]);
        CHECK_RESULT(result);

        if (!correctness) {
            // warmup submits, to get the clocks up before we run the timing
            submitInfo.pCommandBuffers = &commandBuffers[1];
            int warmupCount = 1;
            for (int i = 0; i < warmupCount; ++i) {
                result = vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
                CHECK_RESULT(result);
                result = vkQueueWaitIdle(queue);
                CHECK_RESULT(result);
            }
        }

        uint32_t submitRepeatCount = correctness ? 1 : 10;

        // Time the submit/wait time for this command buffer.
        auto beginTime = std::chrono::high_resolution_clock::now();

        submitInfo.pCommandBuffers = &commandBuffers[1];
        for (uint32_t i = 0; i < submitRepeatCount; ++i) {
            result = vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
            CHECK_RESULT(result);
        }
        result = vkQueueWaitIdle(queue);
        CHECK_RESULT(result);

        auto endTime = std::chrono::high_resolution_clock::now();
        uint64_t elapsedUs = std::chrono::duration_cast<std::chrono::microseconds>(endTime - beginTime).count();
        uint64_t flops = 0;
        if (outerproduct) {
            flops = layerSize*layerSize;
            flops = flops * (numInvocations * (uint64_t)repeatCount * (uint64_t)submitRepeatCount);
        } else {
            flops += layerSize*K;
            for (uint32_t layer = 1; layer < numLayers; ++layer) {
                flops += layerSize*layerSize;
            }
            flops += N*layerSize;
            flops = flops * (2ULL * numInvocations * (uint64_t)repeatCount * (uint64_t)submitRepeatCount);
        }
        double tflops = (double)flops / (double)(elapsedUs / 1000000.0) / (1000.0*1000.0*1000.0*1000.0);

        if (!correctness) {
            printf("  %f TFlops\n", tflops);
        }

        if (correctness)
        {
            // Upload the result from device memory.
            result = vkBeginCommandBuffer(commandBuffers[2], &commandBufferBeginInfo);
            CHECK_RESULT(result);
            {
                Allocation &a = allocations[ALLOC_D];
                VkBufferCopy copy = { 0, 0, a.bufferSize };
                vkCmdCopyBuffer(commandBuffers[2], a.deviceBuffer, a.hostBuffer, 1, &copy);
            }
            result = vkEndCommandBuffer(commandBuffers[2]);
            CHECK_RESULT(result);

            submitInfo.pCommandBuffers = &commandBuffers[2];
            result = vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
            CHECK_RESULT(result);
            result = vkQueueWaitIdle(queue);
            CHECK_RESULT(result);

            const Allocation &alloc_a = allocations[ALLOC_A];
            const Allocation &alloc_b = allocations[ALLOC_B];
            const Allocation &alloc_c = allocations[ALLOC_C];
            const Allocation &alloc_d = allocations[ALLOC_D];
            bool pass = true;
            float maxRelativeError = 0;

            if (outerproduct) {
                vector<uint8_t> outputRowMajor(outputSize);
                size_t dstSize = outputSize;

                VkConvertCooperativeVectorMatrixInfoNV info =
                {
                    VK_STRUCTURE_TYPE_CONVERT_COOPERATIVE_VECTOR_MATRIX_INFO_NV,            // VkStructureType                       sType;
                    nullptr,                                                                // void const*                           pNext;
                    allocations[ALLOC_D].bufferSize,                                        // size_t                                srcSize;
                    (VkDeviceAddress)allocations[ALLOC_D].ptr,                              // VkDeviceOrHostAddressConstKHR         srcData;
                    &dstSize,                                                               // size_t*                               pDstSize;
                    (VkDeviceAddress)outputRowMajor.data(),                                 // VkDeviceOrHostAddressKHR              dstData;
                    matrixType,                                                             // VkComponentTypeKHR                    srcComponentType;
                    matrixType,                                                             // VkComponentTypeKHR                    dstComponentType;
                    layerSize,                                                              // uint32_t                              numRows;
                    layerSize,                                                              // uint32_t                              numColumns;
                    matrixLayout,                                                           // VkCooperativeVectorMatrixLayoutNV     srcLayout;
                    0,                                                                      // size_t                                srcStride;
                    VK_COOPERATIVE_VECTOR_MATRIX_LAYOUT_ROW_MAJOR_NV,                       // VkCooperativeVectorMatrixLayoutNV     dstLayout;
                    layerSize * matrixElementSize,                                          // size_t                                dstStride;
                };

                result = pfn_vkConvertCooperativeVectorMatrixNV(device, &info);
                CHECK_RESULT(result);

                for (uint32_t i = 0; i < layerSize; ++i) {
                    for (uint32_t j = 0; j < layerSize; ++j) {
                        float ref = 0;
                        for (uint32_t inv = 0; inv < numInvocations; ++inv)
                        {
                            if (nonUniform && ((inv % 32) == 31)) {
                                // nonuniform tests skip lane 31.
                                continue;
                            }

                            uint32_t inputAIndex = (inv % workgroupSize) * inputVectorPaddedElements;

                            float x = alloc_a.getDataFloat(0, inputAIndex + i);
                            float y = alloc_a.getDataFloat(0, inputAIndex + j);
                            ref += x * y;
                        }

                        float output = getDataFloat(outputRowMajor.data(), matrixType, 0, i * layerSize + j);
                        if (ref != output) {
                            printf("i=%d j=%d ref %f output %f\n", i, j, ref, output);
                            pass = false;
                        }
                    }
                }
            } else {
                for (uint32_t i = 0; i < numInvocations; ++i)
                {
                    if (nonUniform && ((i % 32) == 31)) {
                        // nonuniform tests skip lane 31.
                        continue;
                    }

                    uint32_t inputAIndex = (i % workgroupSize) * inputVectorPaddedElements;
                    uint32_t outputIndex = i * outputVectorPaddedElements;
                    uint32_t matrixIndex = i >> networkRepeatShift;
                    uint32_t biasIndex = matrixIndex;

                    if (!int8) {
                        assert(alloc_a.isFloatType());
                        vector<float> tempK(K);
                        vector<float> tempN(N);
                        vector<float> vec0(layerSize), vec1(layerSize);

                        for (uint32_t k = 0; k < K; ++k) {
                            tempK[k] = alloc_a.getDataFloat(0, inputAIndex + k);
                        }

                        auto const matmul = [&](uint32_t inDim, std::vector<float> const &inArray, uint32_t outDim, std::vector<float> &outArray, uint32_t mOffset, uint32_t layer)
                        {
                            bool columnMajor = false;
                            for (uint32_t o = 0; o < outDim; ++o)
                            {
                                float ref = 0;
                                for (uint32_t in = 0; in < inDim; ++in)
                                {
                                    float inputA = inArray[in];
                                    uint32_t offset = columnMajor ? (in*matStrides[layer]) : (o*matStrides[layer]); 
                                    uint32_t index = columnMajor ? o : in;
                                    float inputB = alloc_b.getDataFloat(offset + mOffset, index);
                                    ref += inputA * inputB;
                                }
                                outArray[o] = ref;
                            }
                        };
                        auto const addBias = [&](uint32_t outDim, std::vector<float> &outArray, uint32_t biasOffset)
                        {
                            for (uint32_t o = 0; o < outDim; ++o)
                            {
                                float inputC = alloc_c.getDataFloat(biasOffset, o);
                                outArray[o] += inputC;
                            }
                        };
                        auto const relu = [&](uint32_t outDim, std::vector<float> &outArray)
                        {
                            for (uint32_t o = 0; o < outDim; ++o)
                            {
                                outArray[o] = std::max(outArray[o], 0.f);
                            }
                        };
                        auto const quantize = [&](uint32_t dim, std::vector<float> &arr) {
                            if (fp8)
                            {
                                for (uint32_t o = 0; o < dim; ++o)
                                {
					                float before = arr[o];
                                    uint8_t temp;
                                    setDataFloat(&temp, matrixType, 0, 0, before);
                                    float after = getDataFloat(&temp, matrixType, 0, 0);
                                    arr[o] = after;
                                }
                            }
                        };

                        quantize(K, tempK);
                        matmul(K, tempK, layerSize, vec0, matrixIndex * networkMatrixStride + matOffsets[0], 0);
                        addBias(layerSize, vec0, biasIndex * networkBiasStride + biasOffsets[0]);
                        relu(layerSize, vec0);
                        for (uint32_t layer = 1; layer < numLayers; ++layer) {
                            quantize(layerSize, vec0);
                            matmul(layerSize, vec0, layerSize, vec1, matrixIndex * networkMatrixStride + matOffsets[layer], layer);
                            addBias(layerSize, vec1, biasIndex * networkBiasStride + biasOffsets[layer]);
                            relu(layerSize, vec1);
                            vec0 = vec1;
                        }
                        quantize(layerSize, vec0);
                        matmul(layerSize, vec0, N, tempN, matrixIndex * networkMatrixStride + matOffsets[numLayers], numLayers);
                        addBias(N, tempN, biasIndex * networkBiasStride + biasOffsets[numLayers]);

                        for (uint32_t n = 0; n < N; ++n)
                        {
                            float ref = tempN[n];
                            float output = alloc_d.getDataFloat(0, outputIndex + n);
                            if (output != ref) {
                                float denom = ref == 0.0f ? 1.0f : fabs(ref);
                                float err = fabs(output - ref) / denom;
                                float relativeLimit = numLayers > 2 ? 0.15f : 0.01f;

                                maxRelativeError = std::max(maxRelativeError, err);

                                if (err > relativeLimit) {
                                    printf("invocation=%d, n=%d ref %f output %f\n", i, n,ref, output);
                                    pass = false;
                                }
                            }
                        }
                    } else {
                        vector<int64_t> tempK(K);
                        vector<int64_t> tempN(N);
                        vector<int64_t> vec0(layerSize), vec1(layerSize);

                        for (uint32_t k = 0; k < K; ++k) {
                            tempK[k] = alloc_a.getDataInt(0, inputAIndex + k);
                        }

                        auto const matmul = [&](uint32_t inDim, std::vector<int64_t> const &inArray, uint32_t outDim, std::vector<int64_t> &outArray, uint32_t mOffset, uint32_t layer)
                        {
                            bool columnMajor = false;
                            for (uint32_t o = 0; o < outDim; ++o)
                            {
                                int64_t ref = 0;
                                for (uint32_t in = 0; in < inDim; ++in)
                                {
                                    int64_t inputA = inArray[in];
                                    uint32_t offset = columnMajor ? (in*matStrides[layer]) : (o*matStrides[layer]); 
                                    uint32_t index = columnMajor ? o : in;
                                    int64_t inputB = alloc_b.getDataInt(offset + mOffset, index);
                                    ref += inputA * inputB;
                                }
                                outArray[o] = ref;
                            }
                        };
                        auto const addBias = [&](uint32_t outDim, std::vector<int64_t> &outArray, uint32_t biasOffset)
                        {
                            for (uint32_t o = 0; o < outDim; ++o)
                            {
                                int64_t inputC = alloc_c.getDataInt(biasOffset, o);
                                outArray[o] += inputC;
                            }
                        };
                        auto const relu = [&](uint32_t outDim, std::vector<int64_t> &outArray)
                        {
                            for (uint32_t o = 0; o < outDim; ++o)
                            {
                                outArray[o] = std::max(outArray[o], int64_t{0});
                            }
                        };

                        auto const trunc = [&](uint32_t outDim, std::vector<int64_t> &outArray)
                        {
                            for (uint32_t o = 0; o < outDim; ++o) {
                                outArray[o] >>= resultShift;
                                outArray[o] = std::min(outArray[o], int64_t{255});
                            }
                        };

                        matmul(K, tempK, layerSize, vec0, matrixIndex * networkMatrixStride + matOffsets[0], 0);
                        addBias(layerSize, vec0, biasIndex * networkBiasStride + biasOffsets[0]);
                        relu(layerSize, vec0);
                        trunc(layerSize, vec0);

                        for (uint32_t layer = 1; layer < numLayers; ++layer) {
                            matmul(layerSize, vec0, layerSize, vec1, matrixIndex * networkMatrixStride + matOffsets[layer], layer);
                            addBias(layerSize, vec1, biasIndex * networkBiasStride + biasOffsets[layer]);
                            relu(layerSize, vec1);
                            vec0 = vec1;
                            trunc(layerSize, vec0);
                        }
                        matmul(layerSize, vec0, N, tempN, matrixIndex * networkMatrixStride + matOffsets[numLayers], numLayers);
                        addBias(N, tempN, biasIndex * networkBiasStride + biasOffsets[numLayers]);

                        for (uint32_t n = 0; n < N; ++n)
                        {
                            int64_t ref = tempN[n];
                            int64_t output = alloc_d.getDataInt(0, outputIndex + n);
                            if (output != ref) {
                                printf("invocation=%d, n=%d ref %u output %u\n", i, n, (int32_t)ref, (int32_t)output);
                                pass = false;
                            }
                        }
                    }
                }
            }
            printf("%s maxRelativeError=%f\n", pass ? "pass" : "fail", maxRelativeError);
        }

        // Free the memory/buffers/pipeline for this iteration.
        for (int i = 0; i < NUM_ALLOC; ++i) {
            destroyAllocation(device, allocations[i]);
        }
        vkDestroyFramebuffer(device, framebuffer, NULL);
        vkDestroyRenderPass(device, renderPass, NULL);
        vkDestroyPipeline(device, pipeline, NULL);
        vkDestroyShaderModule(device, shaderModule, NULL);
        vkDestroyBuffer(device, paramBuffer, NULL);
        vkFreeMemory(device, paramMemory, NULL);
    }

    vkDestroyBuffer(device, sbtBuffer, NULL);
    vkFreeMemory(device, sbtMemory, NULL);

    printf("\ndone\n");

    return 0;
}
