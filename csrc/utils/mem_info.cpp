#include <c10/xpu/XPUFunctions.h>
#include <level_zero/ze_api.h>
#include <sycl/sycl.hpp>

#include <iostream>

size_t getTotalMemory(ze_device_handle_t& device) {
  uint32_t memoryCount = 0;
  zeDeviceGetMemoryProperties(device, &memoryCount, nullptr);
  auto pMemoryProperties = new ze_device_memory_properties_t[memoryCount];
  for (uint32_t mem = 0; mem < memoryCount; ++mem) {
    pMemoryProperties[mem].stype = ZE_STRUCTURE_TYPE_DEVICE_MEMORY_PROPERTIES;
    pMemoryProperties[mem].pNext = nullptr;
  }
  zeDeviceGetMemoryProperties(device, &memoryCount, pMemoryProperties);
  size_t totalMemory = 0;
  for (uint32_t mem = 0; mem < memoryCount; ++mem) {
    totalMemory += pMemoryProperties[mem].totalSize;
  }
  delete[] pMemoryProperties;

  return totalMemory;
}

size_t getUsableMemory(ze_device_handle_t& device) {
#ifdef VLLM_XPU_HAS_ZE_USABLEMEM
  ze_device_properties_t deviceProperties{};
  ze_device_usablemem_size_ext_properties_t usableMemProps{};

  usableMemProps.stype = ZE_STRUCTURE_TYPE_DEVICE_USABLEMEM_SIZE_EXT_PROPERTIES;
  usableMemProps.pNext = nullptr;
  usableMemProps.currUsableMemSize = 0;
  deviceProperties.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
  deviceProperties.pNext = &usableMemProps;

  zeDeviceGetProperties(device, &deviceProperties);
  // currUsableMemSize stays 0 if the runtime driver does not implement the
  // extension even though the headers declare it; the caller then falls back.
  return usableMemProps.currUsableMemSize;
#else
  // Level Zero loader < 1.27.0: the device-usablemem-size extension is not
  // available in the headers. The caller falls back to total memory.
  (void)device;
  return 0;
#endif
}

std::tuple<int64_t, int64_t> getMemoryInfo(int64_t device_index) {
  const auto& device =
      c10::xpu::get_raw_device(static_cast<c10::DeviceIndex>(device_index));
  auto level_zero_device =
      sycl::get_native<sycl::backend::ext_oneapi_level_zero>(device);
  size_t free = getUsableMemory(level_zero_device);
  const size_t total = getTotalMemory(level_zero_device);

  // Older Level Zero loaders/drivers lack the device-usablemem-size extension,
  // so `free` comes back as 0. If build is with newer headers but runtime is 
  // older, free will also be 0. Fall back to the Intel SYCL free-memory query
  // (requires ZES_ENABLE_SYSMAN); if that is unavailable too, report total.
  if (free == 0) {
    if (device.has(sycl::aspect::ext_intel_free_memory)) {
      free = device.get_info<sycl::ext::intel::info::device::free_memory>();
    } else {
      free = total;
    }
  }

  if (total > static_cast<size_t>(std::numeric_limits<int64_t>::max()) ||
      free > static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
    std::cerr << "Memory size exceeds int64_t max value!" << std::endl;
    return {-1, -1};  // or handle this case as appropriate
  }
  return {static_cast<int64_t>(free), static_cast<int64_t>(total)};
}
