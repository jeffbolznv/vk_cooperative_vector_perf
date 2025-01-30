vk_cooperative_vector_perf is a sample/benchmark demonstrating performance of
using the VK_NV_cooperative_vector Vulkan extension, and the associated
GL_NV_cooperative_vector GLSL extension.

The benchmark runs a shader that computes an MLP with two or three layers,
32 or 64 elements per layer, and varying numbers of invocations sharing a
matrix, and reports the performance in teraflops.

Running this application requires an NVIDIA Turing or later GPU, and a recent
driver that supports the VK_NV_cooperative_vector extension (available from
https://developer.nvidia.com/vulkan-driver).

Modifying and rebuilding the shaders requires a glslangValidator.exe with
GL_NV_cooperative_vector support.
