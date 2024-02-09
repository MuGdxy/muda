#include <catch2/catch.hpp>
#include <muda/muda.h>
#include <muda/container.h>
#include <example_common.h>
using namespace muda;

void device_query()
{
    example_desc("print device infos");
    cudaDeviceProp prop{};
    int            device;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);
    std::cout << "name: " << prop.name << "  |>"
              << "ASCII string identifying device" << std::endl;
    std::cout << "luidDeviceNodeMask: " << prop.luidDeviceNodeMask << "  |>"
              << "LUID device node mask. Value is undefined on TCC and non-Windows platforms"
              << std::endl;
    std::cout << "totalGlobalMem: " << prop.totalGlobalMem << "  |>"
              << "global memory available on device in bytes" << std::endl;
    std::cout << "sharedMemPerBlock: " << prop.sharedMemPerBlock << "  |>"
              << "Shared memory available per block in bytes" << std::endl;
    std::cout << "regsPerBlock: " << prop.regsPerBlock << "  |>"
              << "2-bit registers available per block" << std::endl;
    std::cout << "warpSize: " << prop.warpSize << "  |>"
              << "Warp size in threads" << std::endl;
    std::cout << "memPitch: " << prop.memPitch << "  |>"
              << "Maximum pitch in bytes allowed by memory copies" << std::endl;
    std::cout << "maxThreadsPerBlock: " << prop.maxThreadsPerBlock << "  |>"
              << "Maximum number of threads per block" << std::endl;
    std::cout << "maxThreadsDim: (" << prop.maxThreadsDim[0] << ","
              << prop.maxThreadsDim[1] << "," << prop.maxThreadsDim[2]
              << ")  |>Maximum size of each dimension of a block" << std::endl;
    std::cout << "maxGridSize: (" << prop.maxGridSize[0] << ","
              << prop.maxGridSize[1] << "," << prop.maxGridSize[2]
              << ")  |>Maximum size of each dimension of a grid" << std::endl;
    std::cout << "clockRate: " << prop.clockRate << "  |>"
              << "Clock frequency in kilohertz" << std::endl;
    std::cout << "totalConstMem: " << prop.totalConstMem << "  |>"
              << "Constant memory available on device in bytes" << std::endl;
    std::cout << "major: " << prop.major << "  |>"
              << "Major compute capability" << std::endl;
    std::cout << "minor: " << prop.minor << "  |>"
              << "Minor compute capability" << std::endl;
    std::cout << "textureAlignment: " << prop.textureAlignment << "  |>"
              << "Alignment requirement for textures" << std::endl;
    std::cout << "texturePitchAlignment: " << prop.texturePitchAlignment << "  |>"
              << "Pitch alignment requirement for texture references bound to pitched memory"
              << std::endl;
    std::cout << "deviceOverlap: " << prop.deviceOverlap << "  |>"
              << "Device can concurrently copy memory and execute a kernel. Deprecated. Use instead asyncEngineCount."
              << std::endl;
    std::cout << "multiProcessorCount: " << prop.multiProcessorCount << "  |>"
              << "Number of multiprocessors on device" << std::endl;
    std::cout << "kernelExecTimeoutEnabled: " << prop.kernelExecTimeoutEnabled << "  |>"
              << "Specified whether there is a run time limit on kernels" << std::endl;
    std::cout << "Integrated: " << prop.integrated << "  |>"
              << "Device is integrated as opposed to discrete" << std::endl;
    std::cout << "canMapHostMemory: " << prop.canMapHostMemory << "  |>"
              << "Device can map host memory with cudaHostAlloc/cudaHostGetDevicePointer"
              << std::endl;
    std::cout << "computeMode: " << prop.computeMode << "  |>"
              << "Compute mode (See ::cudaComputeMode)" << std::endl;
    std::cout << "maxTexture1D: " << prop.maxTexture1D << "  |>"
              << "Maximum 1D texture size" << std::endl;
    std::cout << "maxTexture1DMipmap: " << prop.maxTexture1DMipmap << "  |>"
              << "Maximum 1D mipmapped texture size" << std::endl;
    std::cout << "maxTexture1DLinear: " << prop.maxTexture1DLinear << "  |>"
              << "Deprecated, do not use. Use cudaDeviceGetTexture1DLinearMaxWidth() or cuDeviceGetTexture1DLinearMaxWidth() instead."
              << std::endl;
    std::cout << "maxTexture2D: (" << prop.maxTexture2D[0] << ","
              << prop.maxTexture2D[1] << ")  |>Maximum 2D texture dimensions"
              << std::endl;
    std::cout << "maxTexture2DMipmap: (" << prop.maxTexture2DMipmap[0] << ","
              << prop.maxTexture2DMipmap[1]
              << ")  |>Maximum 2D mipmapped texture dimensions" << std::endl;
    std::cout << "maxTexture2DLinear: (" << prop.maxTexture2DLinear[0] << ","
              << prop.maxTexture2DLinear[1] << "," << prop.maxTexture2DLinear[2]
              << ")  |>Maximum dimensions (width, height, pitch) for 2D textures bound to pitched memory"
              << std::endl;
    std::cout << "maxTexture2DGather: (" << prop.maxTexture2DGather[0] << ","
              << prop.maxTexture2DGather[1]
              << ")  |>Maximum 2D texture dimensions if texture gather operations have to be performed"
              << std::endl;
    std::cout << "maxTexture3D: (" << prop.maxTexture3D[0] << ","
              << prop.maxTexture3D[1] << "," << prop.maxTexture3D[2]
              << ")  |>Maximum 3D texture dimensions" << std::endl;
    std::cout << "maxTexture3DAlt: (" << prop.maxTexture3DAlt[0] << ","
              << prop.maxTexture3DAlt[1] << "," << prop.maxTexture3DAlt[2]
              << ")  |>Maximum alternate 3D texture dimensions" << std::endl;
    std::cout << "maxTextureCubemap: " << prop.maxTextureCubemap << "  |>"
              << "Maximum Cubemap texture dimensions" << std::endl;
    std::cout << "maxTexture1DLayered: (" << prop.maxTexture1DLayered[0] << ","
              << prop.maxTexture1DLayered[1]
              << ")  |>Maximum 1D layered texture dimensions" << std::endl;
    std::cout << "maxTexture2DLayered: (" << prop.maxTexture2DLayered[0] << ","
              << prop.maxTexture2DLayered[1] << "," << prop.maxTexture2DLayered[2]
              << ")  |>Maximum 2D layered texture dimensions" << std::endl;
    std::cout << "maxTextureCubemapLayered: (" << prop.maxTextureCubemapLayered[0]
              << "," << prop.maxTextureCubemapLayered[1]
              << ")  |>Maximum Cubemap layered texture dimensions" << std::endl;
    std::cout << "maxSurface1D: " << prop.maxSurface1D << "  |>"
              << "Maximum 1D surface size" << std::endl;
    std::cout << "maxSurface2D: (" << prop.maxSurface2D[0] << ","
              << prop.maxSurface2D[1] << ")  |>Maximum 2D surface dimensions" << std::endl;
    std::cout << "maxSurface3D: (" << prop.maxSurface3D[0] << ","
              << prop.maxSurface3D[1] << "," << prop.maxSurface3D[2]
              << ")  |>Maximum 3D surface dimensions" << std::endl;
    std::cout << "maxSurface1DLayered: (" << prop.maxSurface1DLayered[0] << ","
              << prop.maxSurface1DLayered[1]
              << ")  |>Maximum 1D layered surface dimensions" << std::endl;
    std::cout << "maxSurface2DLayered: (" << prop.maxSurface2DLayered[0] << ","
              << prop.maxSurface2DLayered[1] << "," << prop.maxSurface2DLayered[2]
              << ")  |>Maximum 2D layered surface dimensions" << std::endl;
    std::cout << "maxSurfaceCubemap: " << prop.maxSurfaceCubemap << "  |>"
              << "Maximum Cubemap surface dimensions" << std::endl;
    std::cout << "maxSurfaceCubemapLayered: (" << prop.maxSurfaceCubemapLayered[0]
              << "," << prop.maxSurfaceCubemapLayered[1]
              << ")  |>Maximum Cubemap layered surface dimensions" << std::endl;
    std::cout << "surfaceAlignment: " << prop.surfaceAlignment << "  |>"
              << "Alignment requirements for surfaces" << std::endl;
    std::cout << "concurrentKernels: " << prop.concurrentKernels << "  |>"
              << "Device can possibly execute multiple kernels concurrently"
              << std::endl;
    std::cout << "ECCEnabled: " << prop.ECCEnabled << "  |>"
              << "Device has ECC support enabled" << std::endl;
    std::cout << "pciBusID: " << prop.pciBusID << "  |>"
              << "PCI bus ID of the device" << std::endl;
    std::cout << "pciDeviceID: " << prop.pciDeviceID << "  |>"
              << "PCI device ID of the device" << std::endl;
    std::cout << "pciDomainID: " << prop.pciDomainID << "  |>"
              << "PCI domain ID of the device" << std::endl;
    std::cout << "tccDriver: " << prop.tccDriver << "  |>"
              << "1 if device is a Tesla device using TCC driver, 0 otherwise"
              << std::endl;
    std::cout << "asyncEngineCount: " << prop.asyncEngineCount << "  |>"
              << "Number of asynchronous engines" << std::endl;
    std::cout << "unifiedAddressing: " << prop.unifiedAddressing << "  |>"
              << "Device shares a unified address space with the host" << std::endl;
    std::cout << "memoryClockRate: " << prop.memoryClockRate << "  |>"
              << "Peak memory clock frequency in kilohertz" << std::endl;
    std::cout << "memoryBusWidth: " << prop.memoryBusWidth << "  |>"
              << "Global memory bus width in bits" << std::endl;
    std::cout << "l2CacheSize: " << prop.l2CacheSize << "  |>"
              << "Size of L2 cache in bytes" << std::endl;
    std::cout << "persistingL2CacheMaxSize: " << prop.persistingL2CacheMaxSize << "  |>"
              << "Device's maximum l2 persisting lines capacity setting in bytes"
              << std::endl;
    std::cout << "maxThreadsPerMultiProcessor: " << prop.maxThreadsPerMultiProcessor << "  |>"
              << "Maximum resident threads per multiprocessor" << std::endl;
    std::cout << "streamPrioritiesSupported: " << prop.streamPrioritiesSupported << "  |>"
              << "Device supports stream priorities" << std::endl;
    std::cout << "globalL1CacheSupported: " << prop.globalL1CacheSupported << "  |>"
              << "Device supports caching globals in L1" << std::endl;
    std::cout << "localL1CacheSupported: " << prop.localL1CacheSupported << "  |>"
              << "Device supports caching locals in L1" << std::endl;
    std::cout << "sharedMemPerMultiprocessor: " << prop.sharedMemPerMultiprocessor << "  |>"
              << "Shared memory available per multiprocessor in bytes" << std::endl;
    std::cout << "regsPerMultiprocessor: " << prop.regsPerMultiprocessor << "  |>"
              << "32-bit registers available per multiprocessor" << std::endl;
    std::cout << "managedMemory: " << prop.managedMemory << "  |>"
              << "Device supports allocating managed memory on this system"
              << std::endl;
    std::cout << "isMultiGpuBoard: " << prop.isMultiGpuBoard << "  |>"
              << "Device is on a multi-GPU board" << std::endl;
    std::cout << "multiGpuBoardGroupID: " << prop.multiGpuBoardGroupID << "  |>"
              << "Unique identifier for a group of devices on the same multi-GPU board"
              << std::endl;
    std::cout << "hostNativeAtomicSupported: " << prop.hostNativeAtomicSupported << "  |>"
              << "Link between the device and the host supports native atomic operations"
              << std::endl;
    std::cout << "singleToDoublePrecisionPerfRatio: " << prop.singleToDoublePrecisionPerfRatio
              << "  |>"
              << "Ratio of single precision performance (in floating-point operations per second) to double precision performance"
              << std::endl;
    std::cout << "pageableMemoryAccess: " << prop.pageableMemoryAccess << "  |>"
              << "Device supports coherently accessing pageable memory without calling cudaHostRegister on it"
              << std::endl;
    std::cout << "concurrentManagedAccess: " << prop.concurrentManagedAccess << "  |>"
              << "Device can coherently access managed memory concurrently with the CPU"
              << std::endl;
    std::cout << "computePreemptionSupported: " << prop.computePreemptionSupported << "  |>"
              << "Device supports Compute Preemption" << std::endl;
    std::cout << "canUseHostPointerForRegisteredMem: " << prop.canUseHostPointerForRegisteredMem
              << "  |>"
              << "Device can access host registered memory at the same virtual address as the CPU"
              << std::endl;
    std::cout << "cooperativeLaunch: " << prop.cooperativeLaunch << "  |>"
              << "Device supports launching cooperative kernels via ::cudaLaunchCooperativeKernel"
              << std::endl;
    std::cout << "cooperativeMultiDeviceLaunch: " << prop.cooperativeMultiDeviceLaunch
              << "  |>"
              << "Deprecated, cudaLaunchCooperativeKernelMultiDevice is deprecated."
              << std::endl;
    std::cout << "sharedMemPerBlockOptin: " << prop.sharedMemPerBlockOptin << "  |>"
              << "Per device maximum shared memory per block usable by special opt in"
              << std::endl;
    std::cout << "pageableMemoryAccessUsesHostPageTables: "
              << prop.pageableMemoryAccessUsesHostPageTables << "  |>"
              << "Device accesses pageable memory via the host's page tables"
              << std::endl;
    std::cout << "directManagedMemAccessFromHost: " << prop.directManagedMemAccessFromHost
              << "Host can directly access managed memory on the device without migration."
              << std::endl;
    std::cout << "maxBlocksPerMultiProcessor: " << prop.maxBlocksPerMultiProcessor << "  |>"
              << "Maximum number of resident blocks per multiprocessor" << std::endl;
    std::cout
        << "accessPolicyMaxWindowSize: " << prop.accessPolicyMaxWindowSize << "  |>"
              << "The maximum value of ::cudaAccessPolicyWindow::num_bytes."
              << std::endl;
    std::cout
        << "reservedSharedMemPerBlock: " << prop.reservedSharedMemPerBlock << "  |>"
              << "Shared memory reserved by CUDA driver per block in bytes"
              << std::endl;
}

TEST_CASE("device_query", "[query]")
{
    device_query();
}
