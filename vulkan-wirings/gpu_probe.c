/*
 * gpu_probe.c - minimal Vulkan device lister for vgpu.
 *
 * We cannot rely on `vulkaninfo` (it segfaults against the Android Vulkan
 * loader on this device), so this tiny program enumerates Vulkan physical
 * devices directly and reports whether a *hardware* GPU is reachable.
 *
 * Exit codes:
 *   0  -> at least one hardware GPU (integrated / discrete / virtual) visible
 *   1  -> no devices, or only CPU/software rasterizers (SwiftShader, llvmpipe)
 *   2  -> Vulkan instance could not be created (loader/driver problem)
 *
 * Build (done automatically by the vgpu script if missing):
 *   clang gpu_probe.c -o gpu_probe -lvulkan -I"$PREFIX/include" -L"$PREFIX/lib" -rdynamic
 */
#define VK_NO_PROTO
#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *kind_str(VkPhysicalDeviceType t) {
    switch (t) {
        case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return "integrated-gpu";
        case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:   return "discrete-gpu";
        case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:    return "virtual-gpu";
        case VK_PHYSICAL_DEVICE_TYPE_CPU:            return "cpu";
        default:                                     return "other";
    }
}

static int is_hardware(VkPhysicalDeviceType t) {
    return t == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ||
           t == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU   ||
           t == VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU;
}

int main(void) {
    VkApplicationInfo ai = {0};
    ai.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    ai.apiVersion = VK_API_VERSION_1_0;

    VkInstanceCreateInfo ci = {0};
    ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ci.pApplicationInfo = &ai;

    VkInstance inst = VK_NULL_HANDLE;
    VkResult r = vkCreateInstance(&ci, NULL, &inst);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "vkCreateInstance failed: %d\n", (int)r);
        return 2;
    }

    uint32_t n = 0;
    vkEnumeratePhysicalDevices(inst, &n, NULL);
    VkPhysicalDevice *pds = NULL;
    if (n > 0) {
        pds = malloc(sizeof(VkPhysicalDevice) * n);
        vkEnumeratePhysicalDevices(inst, &n, pds);
    }

    int gpu_found = 0;
    for (uint32_t i = 0; i < n; i++) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(pds[i], &props);
        const char *kind = kind_str(props.deviceType);
        printf("GPU %u: %s [%s] vendor=0x%x device=0x%x\n",
               i, props.deviceName, kind, props.vendorID, props.deviceID);
        if (is_hardware(props.deviceType)) gpu_found = 1;
    }
    free(pds);

    if (n == 0) {
        printf("no Vulkan physical devices found\n");
        return 1;
    }
    return gpu_found ? 0 : 1;
}
