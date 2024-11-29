.. meta::
    :description: PyTorch compatibility
    :keywords: GPU, PyTorch compatibility

********************************************************************************
PyTorch compatibility
********************************************************************************

`PyTorch <https://pytorch.org/>`_ is an open-source tensor library designed for
deep learning. PyTorch on ROCm provides mixed-precision and large-scale training
using our `MIOpen <https://github.com/ROCm/MIOpen>`_ and
`RCCL <https://github.com/ROCm/rccl>`_ libraries.

Supported and unsupported features
================================================================================

The GPU accelerated PyTorch features are collected in the next sections and
grouped as supported or unsupported by ROCm.

Tensor data types
--------------------------------------------------------------------------------

The single data types of `torch.Tensor <https://pytorch.org/docs/stable/tensors.html>`_

Supported tensor data types
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
    :header-rows: 1

    * - Data Type
      - Description
      - Since PyTorch
      - Since ROCm
    * - torch.float8_e4m3fn
      - 8-bit floating point, e4m3
      - 2.3
      - 5.5
    * - torch.float8_e5m2
      - 8-bit floating point, e5m2
      - 2.3
      - 5.5
    * - torch.float16 or torch.half
      - 16-bit floating point
      - 0.1.6
      - 2.0
    * - torch.bfloat16
      - 16-bit floating point
      - 1.6
      - 2.6
    * - torch.float32 or torch.float
      - 32-bit floating point
      - 0.1.12_2
      - 2.0
    * - torch.float64 or torch.double
      - 64-bit floating point
      - 0.1.12_2
      - 2.0
    * - torch.complex32 or torch.chalf
      - PyTorch provides native support for 32-bit complex numbers
      - 1.6
      - 2.0
    * - torch.complex64 or torch.cfloat
      - PyTorch provides native support for 64-bit complex numbers
      - 1.6
      - 2.0
    * - torch.complex128 or torch.cdouble
      - PyTorch provides native support for 128-bit complex numbers
      - 1.6
      - 2.0
    * - torch.uint8
      - 8-bit integer (unsigned)
      - 0.1.12_2
      - 2.0
    * - torch.uint16
      - 16-bit integer (unsigned)
      - 2.3
      - 
    * - torch.uint32
      - 32-bit integer (unsigned)
      - 2.3
      - 
    * - torch.uint64
      - 32-bit integer (unsigned)
      - 2.3
      - 
    * - torch.int8
      - 8-bit integer (signed)
      - 
      - 
    * - torch.int16 or torch.short
      - 16-bit integer (signed)
      - 0.1.12_2
      - 2.0
    * - torch.int32 or torch.int
      - 32-bit integer (signed)
      - 0.1.12_2
      - 2.0
    * - torch.int64 or torch.long
      - 64-bit integer (signed)
      - 0.1.12_2
      - 2.0
    * - torch.bool
      - Boolean
      - 1.2
      - 2.0
    * - torch.quint8
      - quantized 8-bit integer (unsigned)
      - 1.8
      - 
    * - torch.qint8
      - quantized 8-bit integer (signed)
      - 1.8
      - 
    * - torch.qint32
      - quantized 32-bit integer (signed)
      - 1.8
      - 
    * - torch.quint4x2
      - quantized 4-bit integer (unsigned)
      - 1.8
      -

.. note::

  Unsigned types asides from uint8 are currently planned to only have limited
  support in eager mode (they primarily exist to assist usage with torch.compile);
  if you need eager support and the extra range is not needed, we recommend
  using their signed variants instead. See https://github.com/pytorch/pytorch/issues/58734 for more details.

Supported torch AMP
--------------------------------------------------------------------------------

.. list-table::
    :header-rows: 1

    * - Data Type
      - Description
      - Since PyTorch
      - Since ROCm
    * - AMP (Automatic Mixed Precision)
      - | PyTorch that automates the process of using both 16-bit
        | (half-precision, float16) and 32-bit (single-precision, float32)
        | floating-point types in model training and inference
      - 1.9
      - 2.5

Supported and unsupported CUDA torch backends
--------------------------------------------------------------------------------

Return whether PyTorch is built with CUDA support.

Note that this doesn’t necessarily mean CUDA is available; just that if this
PyTorch binary were run on a machine with working CUDA drivers and devices, we
would be able to use it.

Supported CUDA torch backends
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
    :header-rows: 1

    * - Data Type
      - Description
      - Since PyTorch
      - Since ROCm
    * - matmul.allow_fp16_reduced_precision_reduction
      - | Reduced precision reductions (e.g., with fp16 accumulation type)
        | are allowed with fp16 GEMMs
      - 
      - 
    * - matmul.allow_bf16_reduced_precision_reduction
      - Reduced precision reductions are allowed with bf16 GEMMs.
      - 
      - 


Unsupported CUDA torch backends
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
    :header-rows: 1

    * - Data Type
      - Description
      - Since PyTorch
    * - matmul.allow_tf32
      - | A bool that controls whether TensorFloat-32
        | tensor cores may be used in matrix
        | multiplications.
      - 1.7

Supported cuDNN torch backends
--------------------------------------------------------------------------------



Supported cuDNN torch backends
--------------------------------------------------------------------------------


Supported distributed library features
--------------------------------------------------------------------------------

The PyTorch distributed library includes a collective of parallelism modules, a
communications layer, and infrastructure for launching and debugging large
training jobs.


The Distributed Library feature in PyTorch provides tools and APIs for building
and running distributed machine learning workflows. It allows training models
across multiple processes, GPUs, or nodes in a cluster, enabling efficient use
of computational resources and scalability for large-scale tasks.

.. list-table::
    :header-rows: 1

    * - Features
      - Description
      - Since PyTorch
      - Since ROCm
    * - TensorPipe
      - | TensorPipe is a point-to-point communication library integrated 
        | into PyTorch for distributed training. It is designed to handle
        | tensor data transfers efficiently between different processes
        | or devices, including those on separate machines.
      - 1.8
      - 5.4
    * - RPC Device Map Passing
      - | RPC Device Map Passing in PyTorch refers to a feature of the
        | Remote Procedure Call (RPC) framework that enables developers
        | to control and specify how tensors are transferred between 
        | devices during remote operations. It allows fine-grained
        | management of device placement when sending tensors across
        | nodes in distributed training or execution scenarios.
      - 1.9
      - ?
    * - Gloo
      - | Gloo is designed for multi-machine and multi-GPU setups,
        | enabling efficient communication and synchronization between
        | processes. Gloo is one of the default backends for
        | PyTorch's Distributed Data Parallel (DDP) and RPC frameworks,
        | alongside other backends like NCCL and MPI.
      - 1.0
      - 2.0
    * - MPI
      - | MPI (Message Passing Interface) in PyTorch refers
        | to the use of the MPI backend for distributed communication
        | in the torch.distributed module. It enables inter-process
        | communication, primarily in distributed training settings,
        | using the widely adopted MPI standard.
      - 1.9
      - 
    * - TorchElastic
      - | TorchElastic is a PyTorch library that enables fault-tolerant
        | and elastic training in distributed environments. It is
        | designed to handle dynamically changing resources, such as
        | adding or removing nodes during training, which is especially
        | useful in cloud-based or preemptible environments.
      - 1.9
      - 

Unsupported PyTorch features
================================================================================

The GPU accelerated PyTorch features, which are not supported by ROCm collected
in the next sections.
 
.. list-table::
    :header-rows: 1

    * - Features
      - Description
      - Since PyTorch
      - Supported
      - Since ROCm
    * - Random Number Generator
      - Specialized RNG for generating random numbers directly on GPUs.
      - 
      - ✅
      - 
    * - Communication collectives
      - | A set of APIs that enable efficient communication between
        | multiple GPUs, allowing for distributed computing and data
        | parallelism.
      - 
      - ✅
      - 
    * - Streams and events
      - 
      - 
      - ✅
      -       
    * - Graphs (beta)
      - 
      - 
      - ✅
      - 
    * - Memory management
      - 
      - 
      - ✅
      - 
    * - Running process lists
      - | Return a human-readable printout of the running processes
        | and their GPU memory use for a given device.
      - 1.8
      - ✅
      - 
    * - CUDACachingAllocator bypass
      - | Allows to bypass PyTorch’s default CUDA memory allocator
        | (the CUDACachingAllocator) and directly allocate memory
        | on the GPU using native CUDA/HIP functions.
      - 1.1.0
      - ✅
      - 
    * - CUDA Fuser
      - Fusing multiple CUDA kernel operations into a single kernel
      - 1.8
      - ✅
      - 3.5
    * - Enable stream priorities
      - 
      - 
      - ✅
      - 
    * - Tensor scatter functions
      - | Functions are specialized tensor operations used for
        | manipulating tensors by "scattering" data to specific
        | indices.
      - 
      - ✅
      - 
    * - Capturable CUDAGeneratorImpl
      -
      - 
      - ✅
      - 
    * - CuDNN-based LSTM:Support
      -
      -
      - ✅
      - 
    * - Non-Deterministic Alert CUDA Operations
      -
      -
      - ✅
      - 
    * - TorchScript
      -
      -
      - ✅
      - 
    * - Custom Python Classes
      -
      -
      - ✅
      - 
    * - NVIDIA Tools Extension (NVTX)
      - 
      -
      - ✅
      - 
    * - Lazy loading NVRTC
      - 
      -
      - ✅
      - 
    * - Jiterator (beta)
      - Context-manager that selects a given stream.
      -
      - ✅
      - 

Distributed module features

.. list-table::
    :header-rows: 1

    * - Features
      - Description
      - Since PyTorch
      - Supported
      - Since ROCm
    * - TensorPipe
      - | TensorPipe is a point-to-point communication library integrated 
        | into PyTorch for distributed training. It is designed to handle
        | tensor data transfers efficiently between different processes
        | or devices, including those on separate machines.
      - 1.8
      - ✅
      - 
    * - RPC Device Map Passing
      - | RPC Device Map Passing in PyTorch refers to a feature of the
        | Remote Procedure Call (RPC) framework that enables developers
        | to control and specify how tensors are transferred between 
        | devices during remote operations. It allows fine-grained
        | management of device placement when sending tensors across
        | nodes in distributed training or execution scenarios.
      - 1.9
      - ✅
      - 
    * - Gloo
      - | Gloo is designed for multi-machine and multi-GPU setups,
        | enabling efficient communication and synchronization between
        | processes. Gloo is one of the default backends for
        | PyTorch's Distributed Data Parallel (DDP) and RPC frameworks,
        | alongside other backends like NCCL and MPI.
      - 1.0
      - ✅
      - 
    * - MPI
      - | MPI (Message Passing Interface) in PyTorch refers
        | to the use of the MPI backend for distributed communication
        | in the torch.distributed module. It enables inter-process
        | communication, primarily in distributed training settings,
        | using the widely adopted MPI standard.
      - 1.9
      - ✅
      - 
    * - TorchElastic
      - | TorchElastic is a PyTorch library that enables fault-tolerant
        | and elastic training in distributed environments. It is
        | designed to handle dynamically changing resources, such as
        | adding or removing nodes during training, which is especially
        | useful in cloud-based or preemptible environments.
      - 1.9
      - ✅
      - 

Torch compiler features on ROCm.

.. list-table::
    :header-rows: 1

    * - Features
      - Description
      - Supported
    * - torch.compiler (TorchDynamo)
      - | An internal API that uses a CPython feature called the Frame 
        | Evaluation API to safely capture PyTorch graphs. Methods that are 
        | available externally for PyTorch users are surfaced through the
        | torch.compiler namespace.
      - ❌
    * - torch.compiler (TorchInductor)
      - | The default torch.compile deep learning compiler that generates fast
        | code for multiple accelerators and backends. You need to use a backend
        | compiler to make speedups through torch.compile possible. For NVIDIA,
        | AMD and Intel GPUs, it leverages OpenAI Triton as the key building block.
      - ✅
    * - torch.compiler (AOT Autograd)
      - | Autograd captures not only the user-level code, but also
        | backpropagation, which results in capturing the backwards pass
        | “ahead-of-time”. This enables acceleration of both forwards and
        | backwards pass using TorchInductor.
      - ✅

Torch compiler features on ROCm.

    * - Features
      - Description
      - Since PyTorch
      - Since ROCm
    * - NHWC
      - The NHWC memory layout format for tensors.
      - 1.9
      - 
    * - FX:Conv/Batch Norm fuser
      - | Automatically fuses convolution (Conv) and
        | batch normalization (BatchNorm) layers during
        | model optimization.
      - 1.9
      - 
    * - Vision: Quantized Transfer Learning
      - | Enables transfer learning using quantized models,
        | which are optimized for efficiency by reducing the
        | numerical precision of model parameters and
        | operations (e.g., from 32-bit floating-point to 8-bit integers)
      - 1.9
      - 
    * - BERT: Dynamic Quantization
      - | Enables dynamic quantization techniques to the
        | BERT (Bidirectional Encoder Representations from Transformers)
        | model.
      - 1.9 
      - 







Feature



TourchScript

JIT Support

C++ API

CUDA Synchronize

Other

Elementwise Ops Backwards Compatabilty

Unit Test Parity: including c++ tests

CI Resources

Kineto

Conda Packaging

Torchlib Packaging

FX

Eager Mode

Multi tensor code paths

Modules

vision

random

bottleneck

nccl

Autograd

Torchbind

TorchVmap

RPC

TensorCUDA

TF32



PyTorch version	Python	C++	Stable CUDA	Experimental CUDA	Stable ROCm
2.5	>=3.9, <=3.12, (3.13 experimental)	C++17	CUDA 11.8, CUDA 12.1, CUDA 12.4, CUDNN 9.1.0.70	None	ROCm 6.2
2.4	>=3.8, <=3.12	C++17	CUDA 11.8, CUDA 12.1, CUDNN 9.1.0.70	CUDA 12.4, CUDNN 9.1.0.70	ROCm 6.1
2.3	>=3.8, <=3.11, (3.12 experimental)	C++17	CUDA 11.8, CUDNN 8.7.0.84	CUDA 12.1, CUDNN 8.9.2.26	ROCm 6.0
2.2	>=3.8, <=3.11, (3.12 experimental)	C++17	CUDA 11.8, CUDNN 8.7.0.84	CUDA 12.1, CUDNN 8.9.2.26	ROCm 5.7
2.1	>=3.8, <=3.11	C++17	CUDA 11.8, CUDNN 8.7.0.84	CUDA 12.1, CUDNN 8.9.2.26	ROCm 5.6
2.0	>=3.8, <=3.11	C++14	CUDA 11.7, CUDNN 8.5.0.96	CUDA 11.8, CUDNN 8.7.0.84	ROCm 5.4
1.13	>=3.7, <=3.10	C++14	CUDA 11.6, CUDNN 8.3.2.44	CUDA 11.7, CUDNN 8.5.0.96	ROCm 5.2
1.12	>=3.7, <=3.10	C++14	CUDA 11.3, CUDNN 8.3.2.44	CUDA 11.6, CUDNN 8.3.2.44	ROCm 5.0


Release notes
================================================================================

ROCm 6.2.1 - New support for FBGEMM (Facebook General Matrix Multiplication)
--------------------------------------------------------------------------------

FBGEMM is a low-precision, high-performance CPU kernel library for convolution
and matrix multiplication. It is used for server-side inference and as a back
end for PyTorch quantized operators. FBGEMM_GPU includes a collection of PyTorch
GPU operator libraries for training and inference. For more information, see the
ROCm `Model acceleration libraries guide <https://rocm.docs.amd.com/en/docs-6.2.1/how-to/llm-fine-tuning-optimization/model-acceleration-libraries.html>`
and `PyTorch's FBGEMM GitHub repository <https://github.com/pytorch/FBGEMM>`.

**rocAL** (2.0.0) - Added Pytorch iterator for audio.

ROCm 6.2.0
--------------------------------------------------------------------------------

ROCm 6.2.0 supports PyTorch versions 2.2 and 2.3 and TensorFlow version 2.16.

See [Installing PyTorch for ROCm](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-6.2.0/how-to/3rd-party/pytorch-install.html)
and [Installing TensorFlow for ROCm](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-6.2.0/how-to/3rd-party/tensorflow-install.html)
for installation instructions.

Refer to the
[Third-party support matrix](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-6.2.0/reference/3rd-party-support-matrix.html#deep-learning)
for a comprehensive list of third-party frameworks and libraries supported by ROCm.

Optimized framework support for OpenXLA
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

PyTorch for ROCm and TensorFlow for ROCm now provide native support for OpenXLA.
OpenXLA is an open-source ML compiler ecosystem that enables developers to
compile and optimize models from all leading ML frameworks. For more
information, see [Installing PyTorch for ROCm](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-6.2.0/how-to/3rd-party/pytorch-install.html)
and [Installing TensorFlow for ROCm](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-6.2.0/how-to/3rd-party/tensorflow-install.html).

PyTorch support for Autocast (automatic mixed precision)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

PyTorch now supports Autocast for recurrent neural networks (RNNs) on ROCm. This
can help to reduce computational workloads and improve performance. Based on the
information about the magnitude of values, Autocast can substitute the original
`float32` linear layers and convolutions with their `float16` or `bfloat16`
variants. For more information, see [Automatic mixed precision](https://rocm.docs.amd.com/en/docs-6.2.0/how-to/rocm-for-ai/train-a-model.html#automatic-mixed-precision-amp).

PyTorch TunableOp
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Improved optimization and tuning of GEMMs. It requires Docker with PyTorch 2.3
or later.

ROCm 6.1.0
--------------------------------------------------------------------------------

New Torch-MIGraphX driver
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This driver calls MIGraphX directly from PyTorch. It provides an ``mgx_module``
object that you can invoke like any other Torch module, but which utilizes the
MIGraphX inference engine internally. Torch-MIGraphX supports FP32, FP16, and
INT8 datatypes.

ROCm 6.0.0
--------------------------------------------------------------------------------

Upstream support is now available for popular AI frameworks like TensorFlow,
JAX, and PyTorch.

Added TorchMIGraphX
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We introduced a Dynamo backend for Torch, which allows PyTorch to use MIGraphX
directly without first requiring a model to be converted to the ONNX model
format. With a single line of code, PyTorch users can utilize the performance
and quantization benefits provided by MIGraphX.

Added support for the PyTorch kernel plugin
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We added awareness of `__HIP_NO_HALF_CONVERSIONS__` to support PyTorch users.







ROCm 4.5
--------------------------------------------------------------------------------

# AMD ROCm™ v4.5 Release Notes 

This document describes the features, fixed issues, and information about downloading and installing the AMD ROCm™ software. It also covers known issues and deprecations in this release.

- Supported Operating Environments and Documentation Updates
  * [Supported Operating Environments](#Supported-Operating-Environments)
  * [ROCm Installation Updates](#ROCm-Installation-Updates)
  * [AMD ROCm Documentation Updates](#AMD-ROCm-Documentation-Updates)

   
- What\'s New in This Release
  * [HIP Enhancements](#HIP-Enhancements)
  * [Unified Memory Support in ROCm](#Unified-Memory-Support-in-ROCm)
  * [System Management Interface](#System-Management-Interface) 
  * [ROCm Math and Communication Libraries](#ROCm-Math-and-Communication-Libraries)
  * [OpenMP Enhancements](#OpenMP-Enhancements)   

- Known Issues in This Release
  * [Known Issues in This Release](#Known-Issues-in-This-Release)

- Deprecations in This Release
  * [Deprecations](#Deprecations)

- [Hardware and Software Support](#Hardware-and-Software-Support)

- [Machine Learning and High Performance Computing Software Stack for AMD GPU](#Machine-Learning-and-High-Performance-Computing-Software-Stack-for-AMD-GPU)
  * [ROCm Binary Package Structure](#ROCm-Binary-Package-Structure)
  * [ROCm Platform Packages](#ROCm-Platform-Packages)
  

# What's New in This Release 

## HIP Enhancements

The ROCm v4.5 release consists of the following HIP enhancements:



  - Support for HIP Graph 

ROCm v4.5 extends support for HIP Graph. For details, refer to the HIP API Guide at

https://github.com/RadeonOpenCompute/ROCm/blob/master/AMD-HIP-API-4.5.pdf

### Enhanced *launch_bounds* Check Error Log Message 

When a kernel is launched with HIP APIs, for example, hipModuleLaunchKernel(), HIP validates to check that input kernel
dimension size is not larger than specified launch_bounds.

If exceeded, HIP returns launch failure if AMD_LOG_LEVEL is set with the proper value. Users can find more information in the error log message,
including launch parameters of kernel dim size, launch bounds, and the name of the faulting kernel. It is helpful to figure out the faulting
kernel. Besides, the kernel dim size and launch bounds values will also assist in debugging such failures.

For more details, refer to the HIP Programming Guide at

https://github.com/RadeonOpenCompute/ROCm/blob/master/AMD_HIP_Programming_Guide.pdf


- HIP Runtime Compilation

HIP now supports runtime compilation (hipRTC), the usage of which will provide the possibility of optimizations and performance improvement
compared with other APIs via regular offline static compilation. 

hipRTC APIs accept HIP source files in character string format as input parameters and create handles of programs by compiling the HIP source
files without spawning separate processes.

For more details on hipRTC APIs, refer to the HIP API Guide at

https://github.com/RadeonOpenCompute/ROCm/blob/master/AMD-HIP-API-4.5.pdf


### Planned HIP Enhancements and Fixes

#### Changes to hiprtc implementation to match nvrtc behavior

In this release, there are changes to the *hiprtc* implementation to
match the *nvrtc* behavior.

**Impact:** Applications can no longer explicitly include HIP runtime header files. Minor code changes are required to remove the HIP runtime
header files.

#### HIP device attribute enumeration

In a future release, there will be a breaking change in the HIP device attribute enumeration. Enum values are being rearranged to accommodate
future enhancements and additions.

**Impact:** This will require users to rebuild their applications. No code changes are required.

#### Changes to behavior of hipGetLastError() and hipPeekAtLastError() to match CUDA behavior available

In a later release, changes to behavior of hipGetLastError() and hipPeekAtLastError() to match CUDA behavior will be available.

**Impact:** Applications relying on the previous behavior will be impacted and may require some code changes.

## Unified Memory Support in ROCm

Unified memory allows applications to map and migrate data between CPU and GPU seamlessly without explicitly copying it between different
allocations. This enables a more complete implementation of *hipMallocManaged*, *hipMemAdvise*, *hipMemPrefetchAsync* and related
APIs. Without unified memory, these APIs only support system memory. With unified memory, the driver can automatically migrate such memory to
GPU memory for faster access.

### Supported Operating Systems and Versions

This feature is only supported on recent Linux kernels. Currently, it works on Ubuntu versions with 5.6 or newer kernels and the DKMS driver
from ROCm. Current releases of RHEL and SLES do not support this feature yet. Future releases of those distributions will add support for this.
The unified memory feature is also supported in the KFD driver included with upstream kernels starting from Linux 5.14.

Unified memory only works on GFXv9 and later GPUs, including Vega10 and MI100. Fiji, Polaris and older GPUs are not supported. To check whether
unified memory is enabled, look in the kernel log for this message:

```  
     \$ dmesg \| grep \"HMM registered"
```  

If unified memory is enabled, there should be a message like "HMM registered xyzMB device memory". If unified memory is not supported on
your GPU or kernel version, this message is missing.

### Unified Memory Support and XNACK

Unified memory support comes in two flavours, XNACK-enabled and XNACK-disabled. XNACK refers to the ability of the GPU to handle page
faults gracefully and retry a memory access. In XNACK-enabled mode, the GPU can handle retry after page-faults, which enables mapping and
migrating data on demand, as well as memory overcommitment. In XNACK-disabled mode, all memory must be resident and mapped in the GPU
page tables when the GPU is executing application code. Any migrations involve temporary preemption of the GPU queues by the driver. Both page
fault handling and preemptions, happen automatically and are transparent to the applications.

XNACK-enabled mode only has experimental support. XNACK-enabled mode requires compiling shader code differently. By default, the ROCm
compiler builds code that works in both modes. Code can be optimized for one specific mode with compiler options:

OpenCL:

```  
     clang \... -mcpu=gfx908:xnack+:sramecc- \... // xnack on, sramecc off\
     clang \... -mcpu=gfx908:xnack-:sramecc+ \... // xnack off, sramecc on
```  

HIP:
```  
     clang \... \--cuda-gpu-arch=gfx906:xnack+ \... // xnack on\
     clang \... \--cuda-gpu-arch=gfx906:xnack- \... // xnack off
```  
Not all the math libraries included in ROCm support XNACK-enabled mode on current hardware. Applications will fail to run if their shaders are
compiled in the incorrect mode.

On current hardware, the XNACK mode can be chosen at boot-time by a module parameter amdgpu.noretry. The default is XNACK-disabled
(amdgpu.noretry=1).



## ROCm Math and Communication Libraries

In this release, ROCm Math and Communication Libraries consists of the
following enhancements and fixes:

| Library   | Changes                                                  |
| ---       | ---                                                      |
| rocBLAS | **Optimizations** <ul><li>Improved performance of non-batched and batched syr for all sizes and data types</li><li>Improved performance of non-batched and batched hemv for all sizes and data types</li><li>Improved performance of non-batched and batched symv for all sizes and data types</li><li>Improved memory utilization in rocblas-bench, rocblas-test gemm functions, increasing possible runtime sizes.</li></ul>**Changes** <ul><li>Update from C++14 to C++17.</li>  <li>Packaging split into a runtime package (called rocblas) and a development package (called rocblas-dev for .deb packages, and rocblas-devel for .rpm packages). The development package depends on runtime. The runtime package suggests the development package for all supported OSes except CentOS 7 to aid in the transition. The 'suggest' feature in packaging is introduced as a deprecated feature and will be removed in a future ROCm release.</li></ul> **Fixed**<ul><li>For function geam avoid overflow in offset calculation.</li>  <li> For function syr avoid overflow in offset calculation.</li> <li>For function gemv (Transpose-case) avoid overflow in offset calculation.</li> <li>For functions ssyrk and dsyrk, allow conjugate-transpose case to match legacy BLAS. Behavior is the same as the transpose case.</li></ul> |
| hipBLAS| **Added**<ul><li>More support for hipblas-bench</li></ul>**Fixed**<ul><li>Avoid large offset overflow for gemv and hemv in hipblas-test</li></ul>**Changed**<ul><li>Packaging split into a runtime package called hipblas and a development package called hipblas-devel. The development package depends on runtime. The runtime package suggests the development package for all supported OSes except CentOS 7 to aid in the transition. The 'suggests' feature in packaging is a transitional feature and will be removed in a future rocm release.</li></ul> |
| rocFFT | **Optimizations**<ul><li>Optimized SBCC kernels of length 52, 60, 72, 80, 84, 96, 104, 108, 112, 160, 168, 208, 216, 224, 240 with new kernel generator.</li></ul>**Added**<ul><li>Split 2D device code into separate libraries.</li> </ul>**Changed**<ul><li>Packaging split into a runtime package called rocfft and a development package called rocfft-devel. The development package depends on runtime. The runtime package suggests the development package for all supported OSes except CentOS 7 to aid in the transition. The suggests feature in packaging is introduced as a deprecated feature and will be removed in a future rocm release.</li></ul>**Fixed**<ul><li>Fixed a few validation failures of even-length R2C inplace. 2D, 3D cubics sizes such as 100^2 (or ^3), 200^2 (or ^3), 256^2 (or ^3)...etc. We don't combine the three kernels (stockham-r2c-transpose). We only combine two kernels (r2c-transpose) instead.</li></ul> |
| hipFFT | **Changed**  <ul><li>Packaging split into a runtime package called hipfft and a development package called hipfft-devel. The development package depends on runtime. The runtime package suggests the development package for all supported OSes except CentOS 7 to aid in the transition. The 'suggests' feature in packaging is a transitional feature and will be removed in a future rocm release.</li></ul> |
| rocSPARSE | **Added** <ul><li>Triangular solve for multiple right-hand sides using BSR format</li> <li>SpMV for BSRX format</li> <li>SpMM in CSR format enhanced to work with transposed A</li> <li>Matrix coloring for CSR matrices </li><li>Added batched tridiagonal solve (gtsv_strided_batch)</li></ul> **Improved** <ul><li>Fixed a bug with gemvi on Navi21 </li><li>Optimization for pivot based gtsv</li></ul> |
| hipSPARSE | **Added** <ul><li>Triangular solve for multiple right-hand sides using BSR format</li> <li>SpMV for BSRX format</li> <li>SpMM in CSR format enhanced to work with transposed A</li> <li>Matrix coloring for CSR matrices </li>  <li>Added batched tridiagonal solve (gtsv_strided_batch)</li></ul> **Improved** <ul><li>Fixed a bug with gemvi on Navi21</li> <li>Optimization for pivot based gtsv</li></ul> |
| rocALUTION | **Changed** <ul><li>Packaging split into a runtime package called rocalution and a development package called rocalution-devel. The development package depends on runtime. The runtime package suggests the development package for all supported OSes except CentOS 7 to aid in the transition. The 'suggests' feature in packaging is a transitional feature and will be removed in a future rocm release.</li></ul> **Improved** <ul><li>(A)MG solving phase optimization</li></ul> |
| rocTHRUST | **Changed**  <ul><li>Packaging changed to a development package (called rocthrust-dev for .deb packages, and rocthrust-devel for .rpm packages). As rocThrust is a header-only library, there is no runtime package. To aid in the transition, the development package sets the "provides" field to provide the package rocthrust, so that existing packages depending on rocthrust can continue to work. The 'provides' feature is a transitional feature and will be removed in a future ROCm release.</li></ul> |
| rocSOLVER | **Added** <ul><li>RQ factorization routines:</li><li>GERQ2, GERQF (with batched and strided_batchedversions)</li>  <li>Linear solvers for general square systems:</li> <li>GESV (with batched and strided_batched versions)</li><li>Linear solvers for symmetric/hermitian positive definite systems:</li> <li>POTRS (with batched and strided_batched versions)</li> <li>POSV (with batched and strided_batched versions) </li> <li>Inverse of symmetric/hermitian positive definite matrices:</li><li>POTRI (with batched and strided_batched versions)</li> <li>General matrix inversion without pivoting:  </li>  <li>GETRI_NPVT (with batched and strided_batched versions)</li> <li>GETRI_NPVT_OUTOFPLACE (with batched and  strided_batched versions)</li></ul>**Optimized**<ul><li>Improved performance of LU factorization (especially for large matrix sizes)</li> <li>Changed</li>  <li>Raised reference LAPACK version used for rocSOLVER test and benchmark clients to v3.9.1</li>  <li>Minor CMake improvements for users building from source without install.sh:</li> <li>Removed fmt::fmt from rocsolver\'s public usage requirements</li> <li>Enabled small-size optimizations by default </li>  <li>Split packaging into a runtime package ('rocsolver') and a development package ('rocsolver-devel'). The development package depends on the runtime package. To aid in the transition, the runtime package suggests the development package (except on CentOS 7). This use of the 'suggests feature' is transitional and will be removed in a future ROCm release.</li></ul> **Fixed** <ul><li>Use of the GCC / Clang __attribute__((deprecated(...))) extension is now guarded by compiler detection macros. |
| hipSOLVER | The following functions were added in this release:<ul><li>gesv</li><ul><li>hipsolverSSgesv_bufferSize, hipsolverDDgesv_bufferSize, hipsolverCCgesv_bufferSize, hipsolverZZgesv_bufferSize</li><li>hipsolverSSgesv, hipsolverDDgesv, hipsolverCCgesv, hipsolverZZgesv</li></ul><li>potrs</li><ul><li>hipsolverSpotrs_bufferSize, hipsolverDpotrs_bufferSize, hipsolverCpotrs_bufferSize, hipsolverZpotrs_bufferSize</li><li>hipsolverSpotrs, hipsolverDpotrs, hipsolverCpotrs, hipsolverZpotrs</li></ul><li>potrsBatched</li><ul><li>hipsolverSpotrsBatched_bufferSize, hipsolverDpotrsBatched_bufferSize, hipsolverCpotrsBatched_bufferSize, hipsolverZpotrsBatched_bufferSize</li><li>hipsolverSpotrsBatched, hipsolverDpotrsBatched, hipsolverCpotrsBatched, hipsolverZpotrsBatched</li></ul><li>potri</li><ul><li>hipsolverSpotri_bufferSize, hipsolverDpotri_bufferSize, hipsolverCpotri_bufferSize, hipsolverZpotri_bufferSize</li><li>hipsolverSpotri, hipsolverDpotri, hipsolverCpotri, hipsolverZpotri</li></ul></ul></li></ul> |
| RCCL | **Added** <ul><li>Compatibility with NCCL 2.9.9 </li></ul>**Changed**  <ul><li>Packaging split into a runtime package called rccl and a development package called rccl-devel. The development package depends on runtime. The runtime package suggests the development package for all supported OSes except CentOS 7 to aid in the transition. The 'suggests' feature in packaging is a tranistional feature and will be removed in a future rocm release.</li></ul> |
| hipCUB | **Changed**  <ul><li>Packaging changed to a development package (called hipcub-dev for .deb packages, and hipcub-devel for .rpm packages). As hipCUB is a header-only library, there is no runtime package. To aid in the transition, the development package sets the "provides" field to provide the package hipcub, so that existing packages depending on hipcub can continue to work. This provides feature is introduced as a deprecated feature and will be removed in a future ROCm release.</li></ul> |
| rocPRIM| **Added** <ul><li>bfloat16 support added.</li></ul> **Changed**  <ul><li>Packaging split into a runtime package called rocprim and a development package called rocprim-devel. The development package depends on runtime. The runtime package suggests the development package for all supported OSes except CentOS 7 to aid in the transition. The suggests feature in packaging is introduced as a deprecated feature and will be removed in a future rocm release.</li> <li>As rocPRIM is a header-only library, the runtime package is an empty placeholder used to aid in the transition. This package is also a deprecated feature and will be removed in a future rocm release.</li></ul> **Deprecated** <ul><li>The warp_size() function is now deprecated; please switch to host_warp_size() and device_warp_size() for host and device references respectively.</li></ul> |
| rocRAND| **Changed**  <ul><li>Packaging split into a runtime package called rocrand and a development package called rocrand-devel. The development package depends on runtime. The runtime package suggests the development package for all supported OSes except CentOS 7 to aid in the transition. The suggests feature in packaging is introduced as a deprecated feature and will be removed in a future rocm release.</li></ul> **Fixed** <ul><li>Fix for mrg_uniform_distribution_double generating incorrect range of values</li> <li>Fix for order of state calls for log_normal, normal, and uniform</li></ul> **Known issues**  <ul><li>kernel_xorwow test is failing for certain GPU architectures.</li></ul> |

For more information about ROCm Libraries, refer to the documentation at

<https://rocmdocs.amd.com/en/latest/ROCm_Libraries/ROCm_Libraries.html>

# Known Issues in This Release 

The following are the known issues in this release.

## clinfo and rocminfo Do Not Display Marketing Name

clinfo and rocminfo display a blank field for Marketing Name. 

This is due to a missing package that is not yet available from ROCm. This package will be distributed in future ROCm releases.

## Compiler Support for Function Pointers and Virtual Functions

A known issue in the compiler support for function pointers and virtual functions on the GPU may cause undefined behavior due to register
corruption. 

A temporary workaround is to compile the affected application with this option:

```
     -mllvm -amdgpu-fixed-function-abi=1
     
 ```

**Note:** This is an internal compiler flag and may be removed without notice once the issue is addressed in a future release.


## Debugger Process Exit May Cause ROCgdb Internal Error

If the debugger process exits during debugging, ROCgdb may report internal errors. This issue occurs as it attempts to access the AMD GPU
state for the exited process. To recover, users must restart ROCgdb.
 
As a workaround, users can set breakpoints to prevent the debugged process from exiting. For example, users can set breakpoints at the last
statement of the main function and in the abort() and exit() functions. This temporary solution allows the application to be re-run without
restarting ROCgdb.

This issue is currently under investigation and will be fixed in a future release.

For more information, refer to the ROCgdb User Guide at,

https://github.com/RadeonOpenCompute/ROCm/blob/master/ROCDebugger_User_Guide.pdf

## Cache Issues with ROCProfiler

When the same kernel is launched back-to-back multiple times on a GPU, a cache flush is executed each time the kernel finishes when profiler data is collected. The cache flush is inserted by ROCprofiler for each kernel. This prevents kernel from being cached, instead it is being read each time it is launched. As a result the cache hit rate from rocprofiler is reported as 0% or very low.

This issue is under investigation and will be fixed in a future release. 

## Stability Issue on LAMMPS-KOKKOS Applications

On mGPU machines, lammps-kokkos applications experience a stability issue (AMD Instinct MI100™). 

As a workaround, perform a Translation LookAside Buffer (TLB) flush. 

The issue is under active investigation and will be resolved in a future release.


ROCm 4.3
--------------------------------------------------------------------------------

- HIP:

  - Support for Managed Memory Allocation: HIP now supports and automatically 
    manages Heterogeneous Memory Management (HMM) allocation. The HIP
    application performs a capability check before making the managed memory API
    call hipMallocManaged.

**Note**: The _managed_ keyword is unsupported currently. 

```
	int managed_memory = 0;
	HIPCHECK(hipDeviceGetAttribute(&managed_memory,
 	 hipDeviceAttributeManagedMemory,p_gpuDevice));
	if (!managed_memory ) {
  	printf ("info: managed memory access not supported on the device %d\n Skipped\n", p_gpuDevice);
	}
	else {
 	 HIPCHECK(hipSetDevice(p_gpuDevice));
  	HIPCHECK(hipMallocManaged(&Hmm, N * sizeof(T)));
	. . .
	}
```

### Kernel Enqueue Serialization

Developers can control kernel command serialization from the host using the following environment variable,
AMD_SERIALIZE_KERNEL
	
* AMD_SERIALIZE_KERNEL = 1, Wait for completion before enqueue,

* AMD_SERIALIZE_KERNEL = 2, Wait for completion after enqueue,

* AMD_SERIALIZE_KERNEL = 3, Both.

This environment variable setting enables HIP runtime to wait for GPU idle before/after any GPU command.


### NUMA-aware Host Memory Allocation
	
The Non-Uniform Memory Architecture (NUMA) policy determines how memory is allocated and selects a CPU closest to each GPU. 
	
NUMA also measures the distance between the GPU and CPU devices. By default, each GPU selects a Numa CPU node that has the least NUMA distance between them; the host memory is automatically allocated closest to the memory pool of the NUMA node of the current GPU device. 
	
Note, using the *hipSetDevice* API with a different GPU provides access to the host allocation. However, it may have a longer NUMA distance.


### New Atomic System Scope Atomic Operations
	
HIP now provides new APIs with _system as a suffix to support system scope atomic operations. For example,  atomicAnd atomic is dedicated to the GPU device, and atomicAnd_system allows developers to extend the atomic operation to system scope from the GPU device to other CPUs and GPU devices in the system.
	
For more information, refer to the HIP Programming Guide at,
	
https://github.com/RadeonOpenCompute/ROCm/blob/master/AMD_HIP_Programming_Guide_v4.3.pdf

### Indirect Function Call and C++ Virtual Functions 
	
While the new release of the ROCm compiler supports indirect function calls and C++ virtual functions on a device, there are some known limitations and issues. 
	
**Limitations**
	
* An address to a function is device specific.  Note, a function address taken on the host can not be used on a device, and a function address taken on a device can not be used on the host.  On a system with multiple devices, an address taken on one device can not be used on a different device.
	
* C++ virtual functions only work on the device where the object was constructed.
	
* Indirect call to a device function with function scope shared memory allocation is not supported. For example, LDS.
	
* Indirect call to a device function defined in a source file different than the calling function/kernel is only supported when compiling the entire program with -fgpu-rdc.
	
**Known Issues in This Release**
	
* Programs containing kernels with different launch bounds may crash when making an indirect function call.  This issue is due to a compiler issue miscalculating the register budget for the callee function.
	
* Programs may not work correctly when making an indirect call to a function that uses more resources. For example, scratch memory, shared memory, registers made available by the caller.
	
* Compiling a program with objects with pure or deleted virtual functions on the device will result in a linker error.  This issue is due to the missing implementation of some C++ runtime functions on the device.
	
* Constructing an object with virtual functions in private or shared memory may crash the program due to a compiler issue when generating code for the constructor.  


	
### Add 64-bit Energy Accumulator In-band
	
This feature provides an average value of energy consumed over time in a free-flowing RAPL counter, a 64-bit Energy Accumulator.
	
Sample output
	
```
	$ rocm_smi.py --showenergycounter
	=============================== Consumed Energy ================================
	GPU[0] : Energy counter: 2424868
	GPU[0] : Accumulated Energy (uJ): 0.0	

```	
	
### Support for Continuous Clocks Values
	
ROCm SMI will support continuous clock values instead of the previous discrete levels. Moving forward the updated sysfs file will consist of only MIN and MAX values and the user can set the clock value in the given range. 
	
Sample output:

```
	$ rocm_smi.py --setsrange 551 1270 
	Do you accept these terms? [y/N] y                                                                                    
	============================= Set Valid sclk Range=======
	GPU[0]          : Successfully set sclk from 551(MHz) to 1270(MHz)                                                     
	GPU[1]          : Successfully set sclk from 551(MHz) to 1270(MHz)                                                     
	=========================================================================
                       
	$ rocm_smi.py --showsclkrange                                                                                                                                                                    
	============================ Show Valid sclk Range======                     

	GPU[0]          : Valid sclk range: 551Mhz - 1270Mhz                                                                  
	GPU[1]          : Valid sclk range: 551Mhz - 1270Mhz             
```
	
### Memory Utilization Counters

This feature provides a counter display memory utilization information as shown below.

Sample output
	
```
       $ rocm_smi.py --showmemuse
	========================== Current Memory Use ==============================

	GPU[0] : GPU memory use (%): 0
	GPU[0] : Memory Activity: 0
```	

### Performance Determinism

ROCm SMI supports performance determinism as a unique mode of operation. Performance variations are minimal as this enhancement allows users to control the entry and exit to set a soft maximum (ceiling) for the GFX clock.
	
Sample output

```
	$ rocm_smi.py --setperfdeterminism 650
	cat pp_od_clk_voltage
	GFXCLK:                
	0: 500Mhz
	1: 650Mhz *
	2: 1200Mhz
	$ rocm_smi.py --resetperfdeterminism 	
```	
	
**Note**: The idle clock will not take up higher clock values if no workload is running. After enabling determinism, users can run a GFX workload to set performance determinism to the desired clock value in the valid range.

	* GFX clock could either be less than or equal to the max value set in this mode. GFX clock will be at the max clock set in this mode only when required by the running 	workload.
	
	* VDDGFX will be higher by an offset (75mv or so based on PPTable) in the determinism mode.
	
### HBM Temperature Metric Per Stack

This feature will enable ROCm SMI to report all HBM temperature values as shown below.

Sample output

```	
   	$ rocm_smi.py –showtemp
	================================= Temperature =================================
	GPU[0] : Temperature (Sensor edge) (C): 29.0
	GPU[0] : Temperature (Sensor junction) (C): 36.0
	GPU[0] : Temperature (Sensor memory) (C): 45.0
	GPU[0] : Temperature (Sensor HBM 0) (C): 43.0
	GPU[0] : Temperature (Sensor HBM 1) (C): 42.0
	GPU[0] : Temperature (Sensor HBM 2) (C): 44.0
	GPU[0] : Temperature (Sensor HBM 3) (C): 45.0
```	

	
## ROCm Math and Communication Libraries 

### rocBLAS

**Optimizations**

* Improved performance of non-batched and batched rocblas_Xgemv for gfx908 when m <= 15000 and n <= 15000
	
* Improved performance of non-batched and batched rocblas_sgemv and rocblas_dgemv for gfx906 when m <= 6000 and n <= 6000
	
* Improved the overall performance of non-batched and batched rocblas_cgemv for gfx906
	
* Improved the overall performance of rocblas_Xtrsv

For more information, refer to 

https://rocblas.readthedocs.io/en/master/


### rocRAND

**Enhancements**
	
* gfx90a support added
	
* gfx1030 support added

* gfx803 supported re-enabled

**Fixed**
	
* Memory leaks in Poisson tests has been fixed.
	
* Memory leaks when generator has been created but setting seed/offset/dimensions display an exception has been fixed.

For more information, refer to

https://rocrand.readthedocs.io/en/latest/


### rocSOLVER	

**Enhancements**
	
Linear solvers for general non-square systems:
	
* GELS now supports underdetermined and transposed cases
	
* Inverse of triangular matrices
	
* TRTRI (with batched and strided_batched versions)
	
* Out-of-place general matrix inversion
	
* GETRI_OUTOFPLACE (with batched and strided_batched versions)
	
* Argument names for the benchmark client now match argument names from the public API
	
**Fixed Issues**
	
* Known issues with Thin-SVD. The problem was identified in the test specification, not in the thin-SVD implementation or the rocBLAS gemm_batched routines.

* Benchmark client longer crashes as a result of leading dimension or stride arguments not being provided on the command line.

**Optimizations**
	
* Improved general performance of matrix inversion (GETRI)

For more information, refer to

https://rocsolver.readthedocs.io/en/latest/


### rocSPARSE	
	
**Enhancements**
	
* (batched) tridiagonal solver with and without pivoting
	
* dense matrix sparse vector multiplication (gemvi)
	
* support for gfx90a
	
* sampled dense-dense matrix multiplication (sddmm)
	
**Improvements**
	
* client matrix download mechanism
	
* boost dependency in clients removed


For more information, refer to

https://rocsparse.readthedocs.io/en/latest/usermanual.html#rocsparse-gebsrmv


### hipBLAS

**Enhancements**
	
* Added *hipblasStatusToString*
	
**Fixed**
	
* Added catch() blocks around API calls to prevent the leak of C++ exceptions
	

### rocFFT

**Changes**
	
* Re-split device code into single-precision, double-precision, and miscellaneous kernels.
	
**Fixed Issues**
	
* double-precision planar->planar transpose.
	
* 3D transforms with unusual strides, for SBCC-optimized sizes.
	
* Improved buffer placement logic.

For more information, refer to

https://rocfft.readthedocs.io/en/rocm-4.3.0/
	

### hipFFT	

**Fixed Issues**
	
* CMAKE updates
	
* Added callback API in hipfftXt.h header.


### rocALUTION
	
**Enhancements**
	
* Support for gfx90a target
	
* Support for gfx1030 target
	
**Improvements**
	
* Install script
	
For more information, refer to
	
### rocTHRUST	

**Enhancements**
	
* Updated to match upstream Thrust 1.11
	
* gfx90a support added
	
* gfx803 support re-enabled

hipCUB	

Enhancements

* DiscardOutputIterator to backend header
	

# Machine Learning and High Performance Computing Software Stack for AMD GPU

For an updated version of the software stack for AMD GPU, see

https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html#software-stack-for-amd-gpu


ROCm 4.1
--------------------------------------------------------------------------------



ROCm 4.0
--------------------------------------------------------------------------------

- Important features of the AMD Instinct™ MI100 accelerator include:

  - Extended matrix core engine with Matrix Fused Multiply-Add (MFMA) for
    mixed-precision arithmetic and operates on KxN matrices (FP32, FP16, BF16, Int8).

  - Added native support for the bfloat16 data type

  - 3 Infinity fabric connections per GPU enable a fully connected group of 4
    GPUs in a ``hive``.

- Matrix Core Engines and GFX908 Considerations: The AMD CDNA architecture
  builds on GCN’s foundation of scalars and vectors and adds matrices while
  simultaneously adding support for new numerical formats for machine learning
  and preserving backward compatibility for any software written for the GCN
  architecture. These Matrix Core Engines add a new family of wavefront-level
  instructions, the Matrix Fused MultiplyAdd or MFMA. The MFMA family performs
  mixed-precision arithmetic and operates on KxN matrices using four different
  types of input data: 8-bit integers (INT8), 16-bit half-precision FP (FP16),
  16-bit brain FP (bf16), and 32-bit single-precision (FP32). All MFMA
  instructions produce either a 32-bit integer (INT32) or FP32 output, which
  reduces the likelihood of overflowing during the final accumulation stages of
  matrix multiplication. On nodes with gfx908, MFMA instructions are available
  to substantially speed up matrix operations. This hardware feature is used
  only in matrix multiplications functions in rocBLAS and supports only three
  base types f16_r, bf16_r, and f32_r. 

  - For half precision (f16_r and bf16_r) GEMM, use the function rocblas_gemm_ex, and set the compute_type parameter to f32_r.

  - For single precision (f32_r) GEMM, use the function rocblas_sgemm.

  - For single precision complex (f32_c) GEMM, use the function rocblas_cgemm.

ROCm 3.10
--------------------------------------------------------------------------------

## ROCm DATA CENTER TOOL 

The following enhancements are made to the ROCm Data Center Tool.

### Prometheus Plugin for ROCm Data Center Tool

The ROCm Data Center (RDC) Tool now provides the Prometheus plugin, a Python client to collect the telemetry data of the GPU. 
The RDC uses Python binding for Prometheus and the collected plugin. The Python binding maps the RDC C APIs to Python using ctypes. The functions supported by C APIs can also be used in the Python binding.

For installation instructions, refer to the ROCm Data Center Tool User Guide at

https://github.com/RadeonOpenCompute/ROCm/blob/master/AMD_ROCm_DataCenter_Tool_User_Guide.pdf

### Python Binding

The ROCm Data Center (RDC) Tool now uses PyThon Binding for Prometheus and collectd plugins. PyThon binding maps the RDC C APIs to PyThon using ctypes. All the functions supported by C APIs can also be used in PyThon binding. A generic PyThon class RdcReader is created to simplify the usage of the RDC:

* Users can only specify the fields they want to monitor. RdcReader creates groups and fieldgroups, watches the fields, and fetches the fields. 

* RdcReader can support both the Embedded and Standalone mode. Standalone mode can be used with and without authentication.

* In the Standalone mode, the RdcReader can automatically reconnect to rdcd when connection is lost.When rdcd is restarted, the previously created group and fieldgroup may lose. The RdcReader can re-create them and watch the fields after a reconnect. 

* If the client is restarted, RdcReader can detect the groups and fieldgroups created previously, and, therefore, can avoid recreating them.

* Users can pass the unit converter if they do not want to use the RDC default unit.

See the following sample program to monitor the power and GPU utilization using the RdcReader:

```

from RdcReader import RdcReader
from RdcUtil import RdcUtil
from rdc_bootstrap import *
 
default_field_ids = [
        rdc_field_t.RDC_FI_POWER_USAGE,
        rdc_field_t.RDC_FI_GPU_UTIL
]
 
class SimpleRdcReader(RdcReader):
    def __init__(self):
        RdcReader.__init__(self,ip_port=None, field_ids = default_field_ids, update_freq=1000000)
    def handle_field(self, gpu_index, value):
        field_name = self.rdc_util.field_id_string(value.field_id).lower()
        print("%d %d:%s %d" % (value.ts, gpu_index, field_name, value.value.l_int))
 
if __name__ == '__main__':
    reader = SimpleRdcReader()
    while True:
        time.sleep(1)
        reader.process()
        
 ```

For more information about RDC Python binding and the Prometheus plugin integration, refer to the ROCm Data Center Tool User Guide at

https://github.com/RadeonOpenCompute/ROCm/blob/master/AMD_ROCm_DataCenter_Tool_User_Guide.pdf

- New rocSOLVER APIs: The following new rocSOLVER APIs are added in this release:

![Screenshot](https://github.com/Rmalavally/ROCm/blob/master/images/rocsolverAPI.PNG)

For more information, refer to 

https://rocsolver.readthedocs.io/en/latest/userguide_api.html

- RCCL: Alltoallv Support in PyTorch. The AMD ROCm v3.10 release includes a new
  API for ROCm Communication Collectives Library (RCCL). This API sends data
  from all to all ranks and each rank provides arrays of input/output data
  counts and offsets. 

ROCm 3.9
--------------------------------------------------------------------------------

- Improved GEMM Performance: Currently, rocblas_gemm_ext2() supports matrix
  multiplication D <= alpha * A * B + beta * C, where the A, B, C, and D
  matrices are single-precision float, column-major, and non-transposed, except
  that the row stride of C may equal 0.

- New Matrix Pruning Functions: In this release, the following new Matrix
  Pruning functions are introduced. 

- rocSOLVER General Matrix Singular Value Decomposition API: The rocSOLVER
  General Matrix Singular Value Decomposition (GESVD) API is now available in
  the AMD ROCm v3.9 release. 

- Fixed Defects:

  - Random Soft Hang Observed When Running ResNet-Based Models

  - MIGraphx -> test_gpu_ops_test FAILED

ROCm 3.7
--------------------------------------------------------------------------------

- ROCm COMMUNICATIONS COLLECTIVE LIBRARY

  - Compatibility with NVIDIA Communications Collective Library v2\.7 API

  - ROCm Communications Collective Library (RCCL) is now compatible with the
    NVIDIA Communications Collective Library (NCCL) v2.7 API.

  - RCCL (pronounced "Rickle") is a stand-alone library of standard collective
    communication routines for GPUs, implementing all-reduce, all-gather,
    reduce, broadcast, reduce-scatter, gather, scatter, and all-to-all. There is
    also initial support for direct GPU-to-GPU send and receive operations. It
    has been optimized to achieve high bandwidth on platforms using PCIe, xGMI
    as well as networking using InfiniBand Verbs or TCP/IP sockets. RCCL
    supports an arbitrary number of GPUs installed in a single node or multiple
    nodes, and can be used in either single- or multi-process (e.g., MPI)
    applications.

  - The collective operations are implemented using ring and tree algorithms and
    have been optimized for throughput and latency. For best performance, small
    operations can be either batched into larger operations or aggregated
    through the API.

  - For more information about RCCL APIs and compatibility with NCCL v2.7, see
    https://rccl.readthedocs.io/en/develop/index.html

- rocSolver: Singular Value Decomposition of Bi-diagonal Matrices.
  Rocsolver_bdsqr now computes the Singular Value Decomposition (SVD) of
  bi-diagonal matrices. It is an auxiliary function for the SVD of general
  matrices (function rocsolver_gesvd). 

- rocSPARSE: Operations for Sparse Matrices. This enhancement provides a dense
  matrix sparse matrix multiplication using the CSR storage format.

ROCm 3.5
--------------------------------------------------------------------------------

- ROCm Communications Collective Library: 

  - Re-enable target 0x803
  
  - Build time improvements for the HIP-Clang compiler

  - AMD RCCL is now compatible with NVIDIA Communications Collective Library
    (NCCL) v2.6.4 and provides the following features: 

    - Network interface improvements with API v3
    
    - Network topology detection 

    - Improved CPU type detection
    
    - Infiniband adaptive routing support

- MIOpen: Optional Kernel Package Installation

  - MIOpen provides an optional pre-compiled kernel package to reduce startup
    latency. 

  - NOTE: The installation of this package is optional. MIOpen will continue to
    function as expected even if you choose to not install the pre-compiled
    kernel package. This is because MIOpen compiles the kernels on the target
    machine once the kernel is run. However, the compilation step may
    significantly increase the startup time for different operations.

  - To install the kernel package for your GPU architecture, use the following
    command:

    *apt-get install miopen-kernels-<arch>-<num cu>*
 
    * <arch> is the GPU architecture. For example, gfx900, gfx906
    * <num cu> is the number of CUs available in the GPU. For example, 56 or 64 

- API for CPU Affinity:

  - A new API is introduced for aiding applications to select the appropriate
    memory node for a given accelerator(GPU). 

  - The API for CPU affinity has the following signature:

    - *rsmi_status_t rsmi_topo_numa_affinity_get(uint32_t dv_ind, uint32_t *numa_node);*

    - This API takes as input, device index (dv_ind), and returns the NUMA node
      (CPU affinity), stored at the location pointed by numa_node pointer,
      associated with the device.

    - Non-Uniform Memory Access (NUMA) is a computer memory design used in
      multiprocessing, where the memory access time depends on the memory
      location relative to the processor. 

ROCm 3.3
--------------------------------------------------------------------------------

- Support for 3D Pooling Layers: ROCm is enhanced to include support for 3D
  pooling layers. The implementation of 3D pooling layers now allows users to
  run 3D convolutional networks, such as ResNext3D, on AMD Radeon Instinct GPUs. 

- ONNX Enhancements

  - Open Neural Network eXchange (ONNX) is a widely-used neural net exchange
    format. The AMD model compiler & optimizer support the pre-trained models in
    ONNX, NNEF, & Caffe formats. Currently, ONNX versions 1.3 and below are
    supported. 

  - The AMD Neural Net Intermediate Representation (NNIR) is enhanced to handle
    the rapidly changing ONNX versions and its layers. 

ROCm 3.0
--------------------------------------------------------------------------------

- AOMP: 

  - Initial distribution of AOMP 0.7-5

  - The code base for this release of AOMP is the Clang/LLVM 9.0 sources as of
    October 8th, 2019. The LLVM-project branch used to build this release is
    AOMP-191008. It is now locked. With this release, an artifact tarball of the
    entire source tree is created. This tree includes a Makefile in the root
    directory used to build AOMP from the release tarball. You can use Spack to
    build AOMP from this source tarball or build manually without Spack.

ROCm 2.10.0
--------------------------------------------------------------------------------

- rocBLAS:

  - Support for Both single and double precision, CGEMM (Complex GEMM) and ZGEMM.

  - Support is extended to the General Matrix Multiply (GEMM) routine for
    multiple small matrices processed simultaneously for rocBLAS in AMD Radeon
    Instinct MI50.  in

ROCm 2.9.0
--------------------------------------------------------------------------------

- MIGraphX: Introduces support for fp16 and int8 quantization. For additional
  details, as well as other new MIGraphX features, see MIGraphX documentation.

- rocSparse: Add csrgemm, which enables the user to perform matrix-matrix
  multiplication with two sparse matrices in CSR format.

ROCm 2.8.0
--------------------------------------------------------------------------------

- RCCL: 

  - Support for NCCL2.4.8 API
  
  - Implements ncclCommAbort() and ncclCommGetAsyncError() to match the NCCL
    2.4.x API

ROCm 2.7
--------------------------------------------------------------------------------

- rocRand:

  - Add support for new datatypes: uchar, ushort, half.

  - Improved performance on "Vega 7nm" chips, such as on the Radeon Instinct
    MI50.

  - mtgp32 uniform double performance changes due generation algorithm
    standardization. Better quality random numbers now generated with 30%
    decrease in performance.

  - Up to 5% performance improvements for other algorithms.

ROCm 2.6
--------------------------------------------------------------------------------

- Thrust: The first official release of rocThrust and hipCUB. rocThrust is a
  port of thrust, a parallel algorithm library. hipCUB is a port of CUB, a
  reusable software component library. Thrust/CUB has been ported to the
  HIP/ROCm platform to use the rocPRIM library.

- MIGraphX: Add optimizer to read models frozen from Tensorflow framework.
  Further details and an example usage at https://github.com/ROCmSoftwarePlatform/AMDMIGraphX/wiki/Getting-started:-using-the-new-features-of-MIGraphX-0.3

- MIOpen: 
  
  - New features including an immediate mode for selecting convolutions,
    bfloat16 support, new layers, modes, and algorithms.

  - MIOpenDriver, a tool for benchmarking and developing kernels is now shipped
    with MIOpen. BFloat16 now supported in HIP requires an updated rocBLAS as a
    GEMM backend.

  - Immediate mode API now provides the ability to quickly obtain a convolution
    kernel.

  - MIOpen now contains HIP source kernels and implements the ImplicitGEMM
    kernels. This is a new feature and is currently disabled by default. Use the
    environmental variable "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM=1" to activation
    this feature. ImplicitGEMM requires an up to date HIP version of at least 1.5.9211.

  - A new "loss" catagory of layers has been added, of which, CTC loss is the
    first. See the API reference for more details. 2.0 is the last release of
    active support for gfx803 architectures. In future releases, MIOpen will not
    actively debug and develop new features specifically for gfx803.

  - System Find-Db in memory cache is disabled by default. Please see build
    instructions to enable this feature. Additional documentation can be found
    here: https://rocmsoftwareplatform.github.io/MIOpen/doc/html/

- RCCL2: Supports collectives intranode communication using PCIe, Infinity
  Fabric™, and pinned host memory, as well as internode communication using
  Ethernet (TCP/IP sockets) and Infiniband/RoCE (Infiniband Verbs). Note: For
  Infiniband/RoCE, RDMA is not currently supported.

ROCm 2.5.0
--------------------------------------------------------------------------------

- Multi-GPU support is enabled in PyTorch using Dataparallel path for versions
  of PyTorch built using the 06c8aa7a3bbd91cda2fd6255ec82aad21fa1c0d5 commit or
  later.

- Add support for BFloat16 on Radeon Instinct MI50, MI60.

- HIP API has been enhanced to allow independent kernels to run in parallel on
  the same stream.

- UCX: Support for UCX version 1.6 has been added.

- rocBlas:

  -  Add BFloat16 GEMM support in rocBLAS/Tensile.

  - Support mixed precision GEMM with BFloat16 input and output matrices, and all
    arithmetic in IEEE32 bit.

- rocSparse: Performance optimizations for csrsv routines.

- Thrust: Preview release for early adopters. rocThrust is a port of thrust, a
  parallel algorithm library. Thrust has been ported to the HIP/ROCm platform to
  use the rocPRIM library. The HIP ported library works on HIP/ROCm platforms.

- Add support to connect four Radeon Instinct MI60 or Radeon Instinct MI50
  boards in one hive via AMD Infinity Fabric™ Link GPU interconnect technology
  has been added.

ROCm 2.4.0
--------------------------------------------------------------------------------

- Add support to connect two Radeon Instinct MI60 or Radeon Instinct MI50 boards
  via AMD Infinity Fabric™ Link GPU interconnect technology.

ROCm 2.3.0
--------------------------------------------------------------------------------

- MIVisionX: ONNX parser changes to adjust to new file formats.

- MIGraphX:

  - Add support for additional ONNX operators and fixes that now enable a large
    set of Imagenet models.

  - Add support for RNN Operators.

  - Add support for multi-stream execution.

- Caffe2: Enabled multi-gpu support.

- rocBLAS:

  - Introduces support and performance optimizations for Int8 GEMM.

  - Add TRSV support.

  - Improvements and optimizations with tensile.

  - Functional implementation of BLAS L1/L2/L3 functions and prioritized them.

- MIOpen: Add full 3-D convolution support and int8 support for inference.
  Additionally, there are major updates in the performance database for major
  models including those found in Torchvision.

ROCm 2.2.0
--------------------------------------------------------------------------------

- rocSparse: Optimization on Vega20. Cache usage optimizations for csrsv (sparse
  triangular solve), coomv (SpMV in COO format) and ellmv (SpMV in ELL format)
  are available.

- DGEMM and DTRSM Optimization: Improved DGEMM performance for reduced matrix
  sizes k=384, k=256.

- Caffe2: Added support for multi-GPU training.

ROCm 2.1.0
--------------------------------------------------------------------------------

- DGEMM: Improved DGEMM performance for large square and reduced matrix sizes
  k=384, k=256.

ROCm 2.0.0
--------------------------------------------------------------------------------

- PyTorch/Caffe2:

  - Vega 7nm support.

  - fp16 support is enabled.

  - Several bug fixes and performance enhancements.

  - Breaking changes are introduced in ROCm 2.0 which are not addressed upstream
    yet. Meanwhile, please continue to use ROCm fork at https://github.com/ROCm/pytorch

- rocSPARSE & hipSPARSE: introduction.

- rocBLAS: Introduction with improved DGEMM efficiency on Vega 7nm.

- MIOpen:
  
  - This release contains general bug fixes and an updated performance database.
  
  - Group convolutions backwards weights performance has been improved.
  
  - RNNs now support fp16.
  
  - Tensorflow multi-gpu and Tensorf