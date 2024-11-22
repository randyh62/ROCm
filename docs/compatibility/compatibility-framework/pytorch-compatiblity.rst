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

ROCm 4.0
--------------------------------------------------------------------------------


### Key Features of AMD Instinct™ MI100 

Important features of the AMD Instinct™ MI100 accelerator include:

* Extended matrix core engine with Matrix Fused Multiply-Add (MFMA) for mixed-precision arithmetic and operates on KxN matrices (FP32, FP16, BF16, Int8) 

* Added native support for the bfloat16 data type

* 3 Infinity fabric connections per GPU enable a fully connected group of 4 GPUs in a ‘hive’ 

![Screenshot](https://github.com/Rmalavally/ROCm/blob/master/images/keyfeatures.PNG)


### Matrix Core Engines and GFX908 Considerations

The AMD CDNA architecture builds on GCN’s foundation of scalars and vectors and adds matrices while simultaneously adding support for new numerical formats for machine learning and preserving backward compatibility for any software written for the GCN architecture. These Matrix Core Engines add a new family of wavefront-level instructions, the Matrix Fused MultiplyAdd or MFMA. The MFMA family performs mixed-precision arithmetic and operates on KxN matrices using four different types of input data: 8-bit integers (INT8), 16-bit half-precision FP (FP16), 16-bit brain FP (bf16), and 32-bit single-precision (FP32). All MFMA instructions produce either a 32-bit integer (INT32) or FP32 output, which reduces the likelihood of overflowing during the final accumulation stages of matrix multiplication.

On nodes with gfx908, MFMA instructions are available to substantially speed up matrix operations. This hardware feature is used only in matrix multiplications functions in rocBLAS and supports only three base types f16_r, bf16_r, and f32_r. 

* For half precision (f16_r and bf16_r) GEMM, use the function rocblas_gemm_ex, and set the compute_type parameter to f32_r.

* For single precision (f32_r) GEMM, use the function rocblas_sgemm.

* For single precision complex (f32_c) GEMM, use the function rocblas_cgemm.


### References
* For more information about bfloat16, see 

https://rocblas.readthedocs.io/en/master/usermanual.html

* For more details about AMD Instinct™ MI100 accelerator key features, see 

https://www.amd.com/system/files/documents/instinct-mi100-brochure.pdf

* For more information about the AMD Instinct MI100 accelerator, refer to the following sources:

  - AMD CDNA whitepaper at https://www.amd.com/system/files/documents/amd-cdna-whitepaper.pdf
  
  - MI100 datasheet at https://www.amd.com/system/files/documents/instinct-mi100-brochure.pdf

* AMD Instinct MI100/CDNA1 Shader Instruction Set Architecture (Dec. 2020) – This document describes the current environment, organization, and program state of AMD CDNA “Instinct MI100” devices. It details the instruction set and the microcode formats native to this family of processors that are accessible to programmers and compilers.

https://developer.amd.com/wp-content/resources/CDNA1_Shader_ISA_14December2020.pdf


## RAS Enhancements

RAS (Reliability, Availability, and Accessibility) features provide help with data center GPU management. It is a method provided to users to track and manage data points via options implemented in the ROCm-SMI Command Line Interface (CLI) tool. 

For more information about rocm-smi, see 

https://github.com/RadeonOpenCompute/ROC-smi 

The command options are wrappers of the system calls into the device driver interface as described here:

https://dri.freedesktop.org/docs/drm/gpu/amdgpu.html#amdgpu-ras-support



## Using CMake with AMD ROCm

Most components in AMD ROCm support CMake 3.5 or higher out-of-the-box and do not require any special Find modules. A Find module is often used downstream to find the files by guessing locations of files with platform-specific hints. Typically, the Find module is required when the upstream is not built with CMake or the package configuration files are not available.

AMD ROCm provides the respective config-file packages, and this enables find_package to be used directly. AMD ROCm does not require any Find module as the config-file packages are shipped with the upstream projects.

For more information, see 

https://rocmdocs.amd.com/en/latest/Installation_Guide/Using-CMake-with-AMD-ROCm.html


## AMD ROCm and Mesa Multimedia

AMD ROCm extends support to Mesa Multimedia. Mesa is an open-source software implementation of OpenGL, Vulkan, and other graphics API specifications. Mesa translates these specifications to vendor-specific graphics hardware drivers.

For detailed installation instructions, refer to

https://rocmdocs.amd.com/en/latest/Installation_Guide/Mesa-Multimedia-Installation.html


## ROCm System Management Information 

The following enhancements are made to ROCm System Management Interface (SMI).

### Support for Printing PCle Information on AMD Instinct™100

AMD ROCm extends support for printing PCle information on AMD Instinct MI100. 

To check the pp_dpm_pcie file, use *"rocm-smi --showclocks"*.

*/opt/rocm-4.0.0-6132/bin/rocm_smi.py  --showclocks*

![Screenshot](https://github.com/Rmalavally/ROCm/blob/master/images/SMI.PNG)


### New API for xGMI 

Rocm_smi_lib now provides an API that exposes xGMI (inter-chip Global Memory Interconnect) throughput from one node to another. 

Refer to the rocm_smi_lib API documentation for more details. 

https://github.com/RadeonOpenCompute/ROCm/blob/master/ROCm_SMI_API_Guide_v4.0.pdf




## AMD GPU Debugger Enhancements

In this release, AMD GPU Debugger has the following enhancements:

* ROCm v4.0 ROCgdb is based on gdb 10.1

* Extended support for AMD Instinct™ MI100 


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


## ROCm SYSTEM MANAGEMENT INFORMATION 

### System DMA (SDMA) Utilization

Per-process, the SDMA usage is exposed via the ROCm SMI library. The structure rsmi_process_info_t is extended to include sdma_usage. sdma_usage is a 64-bit value that counts the duration (in microseconds) for which the SDMA engine was active during that process's lifetime. 

For example, see the rsmi_compute_process_info_by_pid_get() API below.

```

/**
* @brief This structure contains information specific to a process.
*/
  typedef struct {
      - - -,
      uint64_t sdma_usage; // SDMA usage in microseconds
  } rsmi_process_info_t;
  rsmi_status_t
      rsmi_compute_process_info_by_pid_get(uint32_t pid,
          rsmi_process_info_t *proc);

```

### ROCm-SMI Command Line Interface

The SDMA usage per-process is available using the following command,

```
$ rocm-smi –showpids

```

For more information, see the ROCm SMI API guide at,

https://github.com/RadeonOpenCompute/ROCm/blob/master/ROCm_SMI_API_Guide_v3.10.pdf


### Enhanced ROCm SMI Library for Events

ROCm-SMI library clients can now register to receive the following events: 

* GPU PRE RESET: This reset event is sent to the client just before a GPU is going to be RESET.

* GPU POST RESET: This reset event is sent to the client after a successful GPU RESET.

* GPU THERMAL THROTTLE: This Thermal throttling event is sent if GPU clocks are throttled.


For more information, refer to the ROCm SMI API Guide at:

https://github.com/RadeonOpenCompute/ROCm/blob/master/ROCm_SMI_API_Guide_v3.10.pdf


### ROCm SMI – Command Line Interface Hardware Topology

This feature provides a matrix representation of the GPUs present in a system by providing information of the manner in which the nodes are connected. This is represented in terms of weights, hops, and link types between two given GPUs. It also provides the numa node and the CPU affinity associated with every GPU.

![Screenshot](https://github.com/Rmalavally/ROCm/blob/master/images/CLI1.PNG)

![Screenshot](https://github.com/Rmalavally/ROCm/blob/master/images/CLI2.PNG)


## ROCm MATH and COMMUNICATION LIBRARIES

### New rocSOLVER APIs
The following new rocSOLVER APIs are added in this release:

![Screenshot](https://github.com/Rmalavally/ROCm/blob/master/images/rocsolverAPI.PNG)

For more information, refer to 

https://rocsolver.readthedocs.io/en/latest/userguide_api.html

### RCCL Alltoallv Support in PyTorch

The AMD ROCm v3.10 release includes a new API for ROCm Communication Collectives Library (RCCL). This API sends data from all to all ranks and each rank provides arrays of input/output data counts and offsets. 

For details about the functions and parameters, see 

https://rccl.readthedocs.io/en/master/allapi.html

## ROCm AOMP ENHANCEMENTS

### AOMP Release 11.11-0

The source code base for this release is the upstream LLVM 11 monorepo release/11.x sources with the hash value 

*176249bd6732a8044d457092ed932768724a6f06*

This release includes fixes to the internal Clang math headers:

* This set of changes applies to clang internal headers to support OpenMP C, C++, and FORTRAN and for HIP C. This establishes consistency between NVPTX and AMDGCN offloading and between OpenMP, HIP, and CUDA. OpenMP uses function variants and header overlays to define device versions of functions. This causes clang LLVM IR codegen to mangled names of variants in both the definition and callsites of functions defined in the internal clang headers. These changes apply to headers found in the installation subdirectory lib/clang/11.0.0/include.

* These changes temporarily eliminate the use of the libm bitcode libraries for C and C++. Although math functions are now defined with internal clang headers, a bitcode library of the C functions defined in the headers is still built for FORTRAN toolchain linking because FORTRAN cannot use c math headers. This bitcode library is installed in lib/libdevice/libm-.bc. The source build of this bitcode library is done with the aomp-extras repository and the component built script build_extras.sh. In the future, we will introduce across the board changes to eliminate massive header files for math libraries and replace them with linking to bitcode libraries.

* Added support for -gpubnames in Flang Driver

* Added an example category for Kokkos. The Kokkos example makefile detects if Kokkos is installed and, if not, it builds Kokkos from the Web. Refer to the script kokkos_build.sh in the bin directory on how to build Kokkos. Kokkos now builds cleanly with the OpenMP backend for simple test cases. 

* Fixed hostrpc cmake race condition in the build of openmp

* Add a fatal error if missing -Xopenmp-target or -march options when -fopenmp-targets is specified. However, we do forgive this requirement for offloading to host when there is only a single target and that target is the host.

* Fix a bug in InstructionSimplify pass where a comparison of two constants of different sizes found in the optimization pass. This fixes issue #182 which was causing kokkos build failure.

* Fix openmp error message output for no_rocm_device_lib, was asserting.

* Changed linkage on constant per-kernel symbols from external to weaklinkageonly to prevent duplicate symbols when building kokkos.



ROCm 3.9
--------------------------------------------------------------------------------

## ROCm Compiler Enhancements

The ROCm compiler support in the llvm-amdgpu-12.0.dev-amd64.deb package is enhanced to include support for OpenMP. To utilize this support, the additional package openmp-extras_12.9-0_amd64.deb is required. 

Note, by default, both packages are installed during the ROCm v3.9 installation. For information about ROCm installation, refer to the ROCm Installation Guide. 

AMD ROCm supports the following compilers:

* C++ compiler - Clang++ 
* C compiler - Clang  
* Flang - FORTRAN compiler (FORTRAN 2003 standard)

**NOTE** : All of the above-mentioned compilers support:

* OpenMP standard 4.5 and an evolving subset of the OpenMP 5.0 standard
* OpenMP computational offloading to the AMD GPUs

For more information about AMD ROCm compilers, see the Compiler Documentation section at,

https://rocmdocs.amd.com/en/latest/index.html

  
### Auxiliary Package Supporting OpenMP

The openmp-extras_12.9-0_amd64.deb auxiliary package supports OpenMP within the ROCm compiler. It contains OpenMP specific header files, which are installed in /opt/rocm/llvm/include as well as runtime libraries, fortran runtime libraries, and device bitcode files in /opt/rocm/llvm/lib. The auxiliary package also consists of examples in the /opt/rocm/llvm/examples folder.

**NOTE**: The optional AOMP package resides in /opt/rocm//aomp/bin/clang and the ROCm compiler, which supports OpenMP for AMDGPU, is located in /opt/rocm/llvm/bin/clang.

### AOMP Optional Package Deprecation

Before the AMD ROCm v3.9 release, the optional AOMP package provided support for OpenMP. While AOMP is available in this release, the optional package may be deprecated from ROCm in the future. It is recommended you transition to the ROCm compiler or AOMP standalone releases for OpenMP support. 

### Understanding ROCm Compiler OpenMP Support and AOMP OpenMP Support

The AOMP OpenMP support in ROCm v3.9 is based on the standalone AOMP v11.9-0, with LLVM v11 as the underlying system. However, the ROCm compiler's OpenMP support is based on LLVM v12 (upstream).

**NOTE**: Do not combine the object files from the two LLVM implementations. You must rebuild the application in its entirety using either the AOMP OpenMP or the ROCm OpenMP implementation.  

### Example – OpenMP Using the ROCm Compiler

```

$ cat helloworld.c
#include <stdio.h>
#include <omp.h>
 int main(void) {
  int isHost = 1; 
#pragma omp target map(tofrom: isHost)
  {
    isHost = omp_is_initial_device();
    printf("Hello world. %d\n", 100);
    for (int i =0; i<5; i++) {
      printf("Hello world. iteration %d\n", i);
    }
  }
   printf("Target region executed on the %s\n", isHost ? "host" : "device");
  return isHost;
}
$ /opt/rocm/llvm/bin/clang  -O3 -target x86_64-pc-linux-gnu -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx900 helloworld.c -o helloworld
$ export LIBOMPTARGET_KERNEL_TRACE=1
$ ./helloworld
DEVID: 0 SGN:1 ConstWGSize:256  args: 1 teamsXthrds:(   1X 256) reqd:(   1X   0) n:__omp_offloading_34_af0aaa_main_l7
Hello world. 100
Hello world. iteration 0
Hello world. iteration 1
Hello world. iteration 2
Hello world. iteration 3
Hello world. iteration 4
Target region executed on the device

```

For more examples, see */opt/rocm/llvm/examples*.


## ROCm SYSTEM MANAGEMENT INFORMATION

The AMD ROCm v3.9 release consists of the following ROCm System Management Information (SMI) enhancements:

* Shows the hardware topology

* The ROCm-SMI showpids option shows per-process Compute Unit (CU) Occupancy, VRAM usage, and SDMA usage

* Support for GPU Reset Event and Thermal Throttling Event in ROCm-SMI Library

### ROCm-SMI Hardware Topology

The ROCm-SMI Command Line Interface (CLI) is enhanced to include new options to denote GPU inter-connect topology in the system along with the relative distance between each other and the closest NUMA (CPU) node for each GPU.

![Screenshot](https://github.com/Rmalavally/ROCm/blob/master/images/ROCMCLI1.PNG)

### Compute Unit Occupancy

The AMD ROCm stack now supports a user process in querying Compute Unit (CU) occupancy at a particular moment. This service can be accessed to determine if a process P is using sufficient compute units.

A periodic collection is used to build the profile of a compute unit occupancy for a workload. 

![Screenshot](https://github.com/Rmalavally/ROCm/blob/master/images/ROCMCLI2.PNG)


ROCm supports this capability only on GFX9 devices. Users can access the functionality in two ways:

* indirectly from the SMI library 

* directly via Sysfs 

**NOTE**: On systems that have both GFX9 and non-GFX9 devices, users should interpret the compute unit (CU) occupancy value carefully as the service does not support non-GFX9 devices. 

### Accessing Compute Unit Occupancy Indirectly

The ROCm System Management Interface (SMI) library provides a convenient interface to determine the CU occupancy for a process. To get the CU occupancy of a process reported in percentage terms, invoke the SMI interface using rsmi_compute_process_info_by_pid_get(). The value is reported through the member field cu_occupancy of struct rsmi_process_info_t.

```
/**
   * @brief Encodes information about a process
   * @cu_occupancy Compute Unit usage in percent
   */
  typedef struct {
      - - -,
      uint32_t cu_occupancy;
  } rsmi_process_info_t;

  /**
   * API to get information about a process
  rsmi_status_t
      rsmi_compute_process_info_by_pid_get(uint32_t pid,
          rsmi_process_info_t *proc);
```


### Accessing Compute Unit Occupancy Directly Using SYSFS

Information provided by SMI library is built from sysfs. For every valid device, ROCm stack surfaces a file by the name cu_occupancy in Sysfs. Users can read this file to determine how that device is being used by a particular workload. The general structure of the file path is /proc/<pid>/stats_<gpuid>/cu_occupancy
 
```
/**
   * CU occupancy files for processes P1 and P2 on two devices with 
   * ids: 1008 and 112326
   */
  /sys/devices/virtual/kfd/kfd/proc/<Pid_1>/stats_1008/cu_occupancy
  /sys/devices/virtual/kfd/kfd/proc/<Pid_1>/stats_2326/cu_occupancy
  /sys/devices/virtual/kfd/kfd/proc/<Pid_2>/stats_1008/cu_occupancy
  /sys/devices/virtual/kfd/kfd/proc/<Pid_2>/stats_2326/cu_occupancy
  
// To get CU occupancy for a process P<i>
  for each valid-device from device-list {
    path-1 = Build path for cu_occupancy file;
    path-2 = Build path for file Gpu-Properties;
    cu_in_use += Open and Read the file path-1;
    cu_total_cnt += Open and Read the file path-2;
  }
  cu_percent = ((cu_in_use * 100) / cu_total_cnt);
  
```

### GPU Reset Event and Thermal Throttling Event

The ROCm-SMI library clients can now register for the following events:

![Screenshot](https://github.com/Rmalavally/ROCm/blob/master/images/ROCMCLI3.PNG)


## ROCm Math and Communication Libraries

### ‘rocfft_execution_info_set_stream’ API

rocFFT is a software library for computing Fast Fourier Transforms (FFT). It is part of AMD’s software ecosystem based on ROCm. In addition to AMD GPU devices, the library can be compiled with the CUDA compiler using HIP tools for running on Nvidia GPU devices.

The ‘rocfft_execution_info_set_stream’ API is a function to specify optional and additional information to control execution.  This API specifies the compute stream, which must be invoked before the call to rocfft_execute. Compute stream is the underlying device queue/stream where the library computations are inserted. 

#### PREREQUISITES

Using the compute stream API makes the following assumptions:

* This stream already exists in the program and assigns work to the stream

* The stream must be of type hipStream_t. Note, it is an error to pass the address of a hipStream_t object

#### PARAMETERS

Input

* info execution info handle
* stream underlying compute stream

### Improved GEMM Performance

Currently, rocblas_gemm_ext2() supports matrix multiplication D <= alpha * A * B + beta * C, where the A, B, C, and D matrices are single-precision float, column-major, and non-transposed, except that the row stride of C may equal 0. This means the first row of C is broadcast M times in C:

![Screenshot](https://github.com/Rmalavally/ROCm/blob/master/images/GEMM2.PNG)

If an optimized kernel solution for a particular problem is not available, a slow fallback algorithm is used, and the first time a fallback algorithm is used, the following message is printed to standard error:

*“Warning: Using slow on-host algorithm, because it is not implemented in Tensile yet.”

**NOTE**: ROCBLAS_LAYER controls the logging of the calls. It is recommended to use logging with the rocblas_gemm_ext2() feature, to identify the precise parameters which are passed to it.

* Setting the ROCBLAS_LAYER environment variable to 2 will print the problem parameters as they are being executed.
* Setting the ROCBLAS_LAYER environment variable to 4 will collect all of the sizes, and print them out at the end of program execution.

For more logging information, refer to https://rocblas.readthedocs.io/en/latest/logging.html.


### New Matrix Pruning Functions

In this release, the following new Matrix Pruning functions are introduced. 

![Screenshot](https://github.com/Rmalavally/ROCm/blob/master/images/matrix.png)


### rocSOLVER General Matrix Singular Value Decomposition API

The rocSOLVER General Matrix Singular Value Decomposition (GESVD) API is now available in the AMD ROCm v3.9 release. 

GESVD computes the Singular Values and, optionally, the Singular Vectors of a general m-by-n matrix A (Singular Value Decomposition).

The SVD of matrix A is given by:

```
A = U * S * V'

```

For more information, refer to 

https://rocsolver.readthedocs.io/en/latest/userguide_api.html 


## ROCm AOMP ENHANCEMENTS

### AOMP v11.9-0

The source code base for this release is the upstream LLVM 11 monorepo release/11.x sources as of August 18, 2020, with the hash value 

*1e6907f09030b636054b1c7b01de36f281a61fa2*

The llvm-project branch used to build this release is aomp11. In addition to completing the source tarball, the artifacts of this release include the file llvm-project.patch. This file shows the delta from the llvm-project upstream release/11.x. The size of this patch XXXX lines in XXX files. These changes include support for flang driver, OMPD support, and the hsa libomptarget plugin. The goal is to reduce this with continued upstreaming activity.

The changes for this release of AOMP are:

* Fix compiler warnings for build_project.sh and build_openmp.sh.

* Fix: [flang] The AOMP 11.7-1 Fortran compiler claims to support the -isystem flag, but ignores it.

* Fix: [flang] producing internal compiler error when a character is used with KIND.

* Fix: [flang] openmp map clause on complex allocatable expressions !$omp target data map( chunk%tiles(1)%field%density0).

* DeviceRTL memory footprint has been reduced from ~2.3GB to ~770MB for AMDGCN target.

* Workaround for red_bug_51 failing on gfx908.

* Switch to python3 for ompd and rocgdb.

* Now require cmake 3.13.4 to compile from source.

* Fix aompcc to accept file type cxx.


### AOMP v11.08-0

The source code base for this release is the upstream LLVM 11 monorepo release/11.x sources as of August 18, 2020 with the hash value 

*aabff0f7d564b22600b33731e0d78d2e70d060b4*

The amd-llvm-project branch used to build this release is amd-stg-openmp. In addition to complete source tarball, the artifacts of this release includes the file llvm-project.patch. This file shows the delta from the llvm-project upstream release/11.x which is currently at 32715 lines in 240 files. These changes include support for flang driver, OMPD support and the hsa libomptarget plugin. Our goal is to reduce this with continued upstreaming activity.

These are the major changes for this release of AOMP:

* Switch to the LLVM 11.x stable code base.

* OMPD updates for flang.

* To support debugging OpenMP, selected OpenMP runtime sources are included in lib-debug/src/openmp. The ROCgdb debugger will find these automatically.

* Threadsafe hsa plugin for libomptarget.

* Updates to support device libraries.

* Openmpi configure issue with real16 resolved.

* DeviceRTL memory use is now independent of number of openmp binaries.

* Startup latency on first kernel launch reduced by order of magnitude.

### AOMP v11.07-1

The source code base for this release is the upstream LLVM 11 monorepo development sources as July 10, 2020 with hash valued 979c5023d3f0656cf51bd645936f52acd62b0333 The amd-llvm-project branch used to build this release is amd-stg-openmp. In addition to complete source tarball, the artifacts of this release includes the file llvm-project.patch. This file shows the delta from the llvm-project upstream trunk which is currently at 34121 lines in 277 files. Our goal is to reduce this with continued upstreaming activity.

* Inclusion of OMPD support which is not yet upstream

* Build of ROCgdb

* Host runtime optimisation. GPU image information is now mostly read on the host instead of from the GPU.

* Fixed the source build scripts so that building from the source tarball does not fail because of missing test directories. This fixes issue #116.


# Fixed Defects

The following defects are fixed in this release:

* Random Soft Hang Observed When Running ResNet-Based Models
* (AOMP) ‘Undefined Hidden Symbol’ Linker Error Causes Compilation Failure in HIP
* MIGraphx -> test_gpu_ops_test FAILED
* Unable to install RDC on CentOS/RHEL 7.8/8.2 & SLES

ROCm 3.7
--------------------------------------------------------------------------------

ROCm COMMUNICATIONS COLLECTIVE LIBRARY
Compatibility with NVIDIA Communications Collective Library v2.7 API
ROCm Communications Collective Library (RCCL) is now compatible with the NVIDIA Communications Collective Library (NCCL) v2.7 API.

RCCL (pronounced "Rickle") is a stand-alone library of standard collective communication routines for GPUs, implementing all-reduce, all-gather, reduce, broadcast, reduce-scatter, gather, scatter, and all-to-all. There is also initial support for direct GPU-to-GPU send and receive operations. It has been optimized to achieve high bandwidth on platforms using PCIe, xGMI as well as networking using InfiniBand Verbs or TCP/IP sockets. RCCL supports an arbitrary number of GPUs installed in a single node or multiple nodes, and can be used in either single- or multi-process (e.g., MPI) applications.

The collective operations are implemented using ring and tree algorithms and have been optimized for throughput and latency. For best performance, small operations can be either batched into larger operations or aggregated through the API.

For more information about RCCL APIs and compatibility with NCCL v2.7, see https://rccl.readthedocs.io/en/develop/index.html

Singular Value Decomposition of Bi-diagonal Matrices
Rocsolver_bdsqr now computes the Singular Value Decomposition (SVD) of bi-diagonal matrices. It is an auxiliary function for the SVD of general matrices (function rocsolver_gesvd).

BDSQR computes the singular value decomposition (SVD) of a n-by-n bidiagonal matrix B.

The SVD of B has the following form:

B = Ub * S * Vb' where • S is the n-by-n diagonal matrix of singular values of B • the columns of Ub are the left singular vectors of B • the columns of Vb are its right singular vectors

The computation of the singular vectors is optional; this function accepts input matrices U (of size nu-by-n) and V (of size n-by-nv) that are overwritten with U*Ub and Vb’*V. If nu = 0 no left vectors are computed; if nv = 0 no right vectors are computed.

Optionally, this function can also compute Ub’*C for a given n-by-nc input matrix C.

PARAMETERS

• [in] handle: rocblas_handle.

• [in] uplo: rocblas_fill.

Specifies whether B is upper or lower bidiagonal.

• [in] n: rocblas_int. n >= 0.

The number of rows and columns of matrix B.

• [in] nv: rocblas_int. nv >= 0.

The number of columns of matrix V.

• [in] nu: rocblas_int. nu >= 0.

The number of rows of matrix U.

• [in] nc: rocblas_int. nu >= 0.

The number of columns of matrix C.

• [inout] D: pointer to real type. Array on the GPU of dimension n.

On entry, the diagonal elements of B. On exit, if info = 0, the singular values of B in decreasing order; if info > 0, the diagonal elements of a bidiagonal matrix orthogonally equivalent to B.

• [inout] E: pointer to real type. Array on the GPU of dimension n-1.

On entry, the off-diagonal elements of B. On exit, if info > 0, the off-diagonal elements of a bidiagonal matrix orthogonally equivalent to B (if info = 0 this matrix converges to zero).

• [inout] V: pointer to type. Array on the GPU of dimension ldv*nv.

On entry, the matrix V. On exit, it is overwritten with Vb’*V. (Not referenced if nv = 0).

• [in] ldv: rocblas_int. ldv >= n if nv > 0, or ldv >=1 if nv = 0.

Specifies the leading dimension of V.

• [inout] U: pointer to type. Array on the GPU of dimension ldu*n.

On entry, the matrix U. On exit, it is overwritten with U*Ub. (Not referenced if nu = 0).

• [in] ldu: rocblas_int. ldu >= nu.

Specifies the leading dimension of U.

• [inout] C: pointer to type. Array on the GPU of dimension ldc*nc.

On entry, the matrix C. On exit, it is overwritten with Ub’*C. (Not referenced if nc = 0).

• [in] ldc: rocblas_int. ldc >= n if nc > 0, or ldc >=1 if nc = 0.

Specifies the leading dimension of C.

• [out] info: pointer to a rocblas_int on the GPU.

If info = 0, successful exit. If info = i > 0, i elements of E have not converged to zero.

For more information, see https://rocsolver.readthedocs.io/en/latest/userguide_api.html#rocsolver-type-bdsqr

rocSPARSE_gemmi() Operations for Sparse Matrices
This enhancement provides a dense matrix sparse matrix multiplication using the CSR storage format. rocsparse_gemmi multiplies the scalar αα with a dense m×km×k matrix AA and the sparse k×nk×n matrix BB defined in the CSR storage format, and adds the result to the dense m×nm×n matrix CC that is multiplied by the scalar ββ, such that C:=α⋅op(A)⋅op(B)+β⋅CC:=α⋅op(A)⋅op(B)+β⋅C with

op(A)=⎧⎩⎨⎪⎪A,AT,AH,if trans_A == rocsparse_operation_noneif trans_A == rocsparse_operation_transposeif trans_A == rocsparse_operation_conjugate_transposeop(A)={A,if trans_A == rocsparse_operation_noneAT,if trans_A == rocsparse_operation_transposeAH,if trans_A == rocsparse_operation_conjugate_transpose

and

op(B)=⎧⎩⎨⎪⎪B,BT,BH,if trans_B == rocsparse_operation_noneif trans_B == rocsparse_operation_transposeif trans_B == rocsparse_operation_conjugate_transposeop(B)={B,if trans_B == rocsparse_operation_noneBT,if trans_B == rocsparse_operation_transposeBH,if trans_B == rocsparse_operation_conjugate_transpose Note: This function is non-blocking and executed asynchronously with the host. It may return before the actual computation has finished.

For more information and examples, see https://rocsparse.readthedocs.io/en/master/usermanual.html#rocsparse-gemmi  


ROCm 3.5
--------------------------------------------------------------------------------

ROCm Communications Collective Library
The ROCm Communications Collective Library (RCCL) consists of the following enhancements:

Re-enable target 0x803
Build time improvements for the HIP-Clang compiler
NVIDIA Communications Collective Library Version Compatibility
AMD RCCL is now compatible with NVIDIA Communications Collective Library (NCCL) v2.6.4 and provides the following features:

Network interface improvements with API v3
Network topology detection
Improved CPU type detection
Infiniband adaptive routing support
MIOpen Optional Kernel Package Installation
MIOpen provides an optional pre-compiled kernel package to reduce startup latency.

NOTE: The installation of this package is optional. MIOpen will continue to function as expected even if you choose to not install the pre-compiled kernel package. This is because MIOpen compiles the kernels on the target machine once the kernel is run. However, the compilation step may significantly increase the startup time for different operations.

To install the kernel package for your GPU architecture, use the following command:

apt-get install miopen-kernels--

is the GPU architecture. For example, gfx900, gfx906
is the number of CUs available in the GPU. For example, 56 or 64
New SMI Event Interface and Library
An SMI event interface is added to the kernel and ROCm SMI lib for system administrators to get notified when specific events occur. On the kernel side, AMDKFD_IOC_SMI_EVENTS input/output control is enhanced to allow notifications propagation to user mode through the event channel.

On the ROCm SMI lib side, APIs are added to set an event mask and receive event notifications with a timeout option. Further, ROCm SMI API details can be found in the PDF generated by Doxygen from source or by referring to the rocm_smi.h header file (see the rsmi_event_notification_* functions).

For the more details about ROCm SMI API, see

https://github.com/RadeonOpenCompute/ROCm/blob/master/ROCm_SMI_Manual.pdf

API for CPU Affinity
A new API is introduced for aiding applications to select the appropriate memory node for a given accelerator(GPU).

The API for CPU affinity has the following signature:

*rsmi_status_t rsmi_topo_numa_affinity_get(uint32_t dv_ind, uint32_t numa_node);

This API takes as input, device index (dv_ind), and returns the NUMA node (CPU affinity), stored at the location pointed by numa_node pointer, associated with the device.

Non-Uniform Memory Access (NUMA) is a computer memory design used in multiprocessing, where the memory access time depends on the memory location relative to the processor.

Radeon Performance Primitives Library
The new Radeon Performance Primitives (RPP) library is a comprehensive high-performance computer vision library for AMD (CPU and GPU) with the HIP and OpenCL backend. The target operating system is Linux.

ScreenShot

For more information about prerequisites and library functions, see

https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX/tree/master/docs

ROCm 3.2
--------------------------------------------------------------------------------

The AMD ROCm v3.2 release was not productized.

ROCm 3.1
--------------------------------------------------------------------------------

Change in ROCm Installation Directory Structure
A fresh installation of the ROCm toolkit installs the packages in the /opt/rocm- folder. Previously, ROCm toolkit packages were installed in the /opt/rocm folder.

Reliability, Accessibility, and Serviceability Support for Vega 7nm
The Reliability, Accessibility, and Serviceability (RAS) support for Vega7nm is now available.

SLURM Support for AMD GPU
SLURM (Simple Linux Utility for Resource Management) is an open source, fault-tolerant, and highly scalable cluster management and job scheduling system for large and small Linux clusters.

ROCm 3.0
--------------------------------------------------------------------------------

New features and enhancements in ROCm v3.0
Support for CentOS RHEL v7.7
Support is extended for CentOS/RHEL v7.7 in the ROCm v3.0 release. For more information about the CentOS/RHEL v7.7 release, see:

CentOS/RHEL

Initial distribution of AOMP 0.7-5 in ROCm v3.0
The code base for this release of AOMP is the Clang/LLVM 9.0 sources as of October 8th, 2019. The LLVM-project branch used to build this release is AOMP-191008. It is now locked. With this release, an artifact tarball of the entire source tree is created. This tree includes a Makefile in the root directory used to build AOMP from the release tarball. You can use Spack to build AOMP from this source tarball or build manually without Spack.

For more information about AOMP 0.7-5, see: AOMP

Fast Fourier Transform Updates
The Fast Fourier Transform (FFT) is an efficient algorithm for computing the Discrete Fourier Transform. Fast Fourier transforms are used in signal processing, image processing, and many other areas. The following real FFT performance change is made in the ROCm v3.0 release:

• Implement efficient real/complex 2D transforms for even lengths.

Other improvements:

• More 2D test coverage sizes.

• Fix buffer allocation error for large 1D transforms.

• C++ compatibility improvements.

MemCopy Enhancement for rocProf
In the v3.0 release, the rocProf tool is enhanced with an additional capability to dump asynchronous GPU memcopy information into a .csv file. You can use the '-hsa-trace' option to create the results_mcopy.csv file. Future enhancements will include column labels.

ROCm 2.10.0
--------------------------------------------------------------------------------

rocBLAS - Support for Complex GEMM
The rocBLAS library is a gpu-accelerated implementation of the standard Basic Linear Algebra
Subroutines (BLAS). rocBLAS is designed to enable you to develop algorithms, including high
performance computing, image analysis, and machine learning.
In the AMD ROCm release v2.10, support is extended to the General Matrix Multiply (GEMM)
routine for multiple small matrices processed simultaneously for rocBLAS in AMD Radeon
Instinct MI50. Both single and double precision, CGEMM and ZGEMM, are now supported in
rocBLAS.

ROCm 2.9.0
--------------------------------------------------------------------------------

Initial release for Radeon Augmentation Library(RALI)
The AMD Radeon Augmentation Library (RALI) is designed to efficiently decode and process images from a variety of storage formats and modify them through a processing graph programmable by the user. RALI currently provides C API.

Quantization in MIGraphX v0.4
MIGraphX 0.4 introduces support for fp16 and int8 quantization. For additional details, as well as other new MIGraphX features, see MIGraphX documentation.

rocSparse csrgemm
csrgemm enables the user to perform matrix-matrix multiplication with two sparse matrices in CSR format.

Singularity Support
ROCm 2.9 adds support for Singularity container version 2.5.2.

Initial release of rocTX
ROCm 2.9 introduces rocTX, which provides a C API for code markup for performance profiling. This initial release of rocTX supports annotation of code ranges and ASCII markers. For an example, see this code.

Added support for Ubuntu 18.04.3
Ubuntu 18.04.3 is now supported in ROCm 2.9.

ROCm 2.8.0
--------------------------------------------------------------------------------

Support for NCCL2.4.8 API
Implements ncclCommAbort() and ncclCommGetAsyncError() to match the NCCL 2.4.x API

ROCm 2.7.2
--------------------------------------------------------------------------------

This release is a hotfix for ROCm release 2.7.

Issues fixed in ROCm 2.7.2
A defect in upgrades from older ROCm releases has been fixed.
rocprofiler --hiptrace and --hsatrace fails to load roctracer library
In ROCm 2.7.2, rocprofiler --hiptrace and --hsatrace fails to load roctracer library defect has been fixed.
To generate traces, please provide directory path also using the parameter: -d <$directoryPath> for example:

/opt/rocm/bin/rocprof  --hsa-trace -d $PWD/traces /opt/rocm/hip/samples/0_Intro/bit_extract/bit_extract
All traces and results will be saved under $PWD/traces path

Upgrading from ROCm 2.7 to 2.7.2
To upgrade, please remove 2.7 completely as specified for ubuntu or for centos/rhel, and install 2.7.2 as per instructions install instructions

Other notes
To use rocprofiler features, the following steps need to be completed before using rocprofiler:

Step-1: Install roctracer
Ubuntu 16.04 or Ubuntu 18.04:
sudo apt install roctracer-dev
CentOS/RHEL 7.6:
sudo yum install roctracer-dev
Step-2: Add /opt/rocm/roctracer/lib to LD_LIBRARY_PATH
New features and enhancements in ROCm 2.7
[rocFFT] Real FFT Functional
Improved real/complex 1D even-length transforms of unit stride. Performance improvements of up to 4.5x are observed. Large problem sizes should see approximately 2x.

rocRand Enhancements and Optimizations
Added support for new datatypes: uchar, ushort, half.
Improved performance on "Vega 7nm" chips, such as on the Radeon Instinct MI50
mtgp32 uniform double performance changes due generation algorithm standardization. Better quality random numbers now generated with 30% decrease in performance
Up to 5% performance improvements for other algorithms
RAS
Added support for RAS on Radeon Instinct MI50, including:

Memory error detection
Memory error detection counter
ROCm-SMI enhancements
Added ROCm-SMI CLI and LIB support for FW version, compute running processes, utilization rates, utilization counter, link error counter, and unique ID.

ROCm 2.6.0
--------------------------------------------------------------------------------

Thrust - Functional Support on Vega20
ROCm2.6 contains the first official release of rocThrust and hipCUB. rocThrust is a port of thrust, a parallel algorithm library. hipCUB is a port of CUB, a reusable software component library. Thrust/CUB has been ported to the HIP/ROCm platform to use the rocPRIM library. The HIP ported library works on HIP/ROCm platforms.

Note: rocThrust and hipCUB library replaces https://github.com/ROCmSoftwarePlatform/thrust (hip-thrust), i.e. hip-thrust has been separated into two libraries, rocThrust and hipCUB. Existing hip-thrust users are encouraged to port their code to rocThrust and/or hipCUB. Hip-thrust will be removed from official distribution later this year.

MIGraphX v0.3
MIGraphX optimizer adds support to read models frozen from Tensorflow framework. Further details and an example usage at https://github.com/ROCmSoftwarePlatform/AMDMIGraphX/wiki/Getting-started:-using-the-new-features-of-MIGraphX-0.3

MIOpen 2.0
This release contains several new features including an immediate mode for selecting convolutions, bfloat16 support, new layers, modes, and algorithms.
MIOpenDriver, a tool for benchmarking and developing kernels is now shipped with MIOpen. BFloat16 now supported in HIP requires an updated rocBLAS as a GEMM backend.
Immediate mode API now provides the ability to quickly obtain a convolution kernel.
MIOpen now contains HIP source kernels and implements the ImplicitGEMM kernels. This is a new feature and is currently disabled by default. Use the environmental variable "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM=1" to activation this feature. ImplicitGEMM requires an up to date HIP version of at least 1.5.9211.
A new "loss" catagory of layers has been added, of which, CTC loss is the first. See the API reference for more details. 2.0 is the last release of active support for gfx803 architectures. In future releases, MIOpen will not actively debug and develop new features specifically for gfx803.
System Find-Db in memory cache is disabled by default. Please see build instructions to enable this feature. Additional documentation can be found here: https://rocmsoftwareplatform.github.io/MIOpen/doc/html/
Bloat16 software support in rocBLAS/Tensile
Added mixed precision bfloat16/IEEE f32 to gemm_ex. The input and output matrices are bfloat16. All arithmetic is in IEEE f32.

AMD Infinity Fabric™ Link enablement
The ability to connect four Radeon Instinct MI60 or Radeon Instinct MI50 boards in two hives or two Radeon Instinct MI60 or Radeon Instinct MI50 boards in four hives via AMD Infinity Fabric™ Link GPU interconnect technology has been added.

ROCm-smi features and bug fixes
mGPU & Vendor check
Fix clock printout if DPM is disabled
Fix finding marketing info on CentOS
Clarify some error messages
ROCm-smi-lib enhancements
Documentation updates
Improvements to *name_get functions
RCCL2 Enablement
RCCL2 supports collectives intranode communication using PCIe, Infinity Fabric™, and pinned host memory, as well as internode communication using Ethernet (TCP/IP sockets) and Infiniband/RoCE (Infiniband Verbs). Note: For Infiniband/RoCE, RDMA is not currently supported.

rocFFT enhancements
Added: Debian package with FFT test, benchmark, and sample programs
Improved: hipFFT interfaces
Improved: rocFFT CPU reference code, plan generation code and logging code

ROCm 2.5.0
--------------------------------------------------------------------------------

UCX 1.6 support
Support for UCX version 1.6 has been added.

BFloat16 GEMM in rocBLAS/Tensile
Software support for BFloat16 on Radeon Instinct MI50, MI60 has been added. This includes:

Mixed precision GEMM with BFloat16 input and output matrices, and all arithmetic in IEEE32 bit
Input matrix values are converted from BFloat16 to IEEE32 bit, all arithmetic and accumulation is IEEE32 bit. Output values are rounded from IEEE32 bit to BFloat16
Accuracy should be correct to 0.5 ULP
ROCm-SMI enhancements
CLI support for querying the memory size, driver version, and firmware version has been added to ROCm-smi.

[PyTorch] multi-GPU functional support (CPU aggregation/Data Parallel)
Multi-GPU support is enabled in PyTorch using Dataparallel path for versions of PyTorch built using the 06c8aa7a3bbd91cda2fd6255ec82aad21fa1c0d5 commit or later.

rocSparse optimization on Radeon Instinct MI50 and MI60
This release includes performance optimizations for csrsv routines in the rocSparse library.

[Thrust] Preview
Preview release for early adopters. rocThrust is a port of thrust, a parallel algorithm library. Thrust has been ported to the HIP/ROCm platform to use the rocPRIM library. The HIP ported library works on HIP/ROCm platforms.

Note: This library will replace https://github.com/ROCmSoftwarePlatform/thrust in a future release. The package for rocThrust (this library) currently conflicts with version 2.5 package of thrust. They should not be installed together.

Support overlapping kernel execution in same HIP stream
HIP API has been enhanced to allow independent kernels to run in parallel on the same stream.

AMD Infinity Fabric™ Link enablement
The ability to connect four Radeon Instinct MI60 or Radeon Instinct MI50 boards in one hive via AMD Infinity Fabric™ Link GPU interconnect technology has been added.



ROCm 2.4.0
--------------------------------------------------------------------------------

TensorFlow 2.0 support
ROCm 2.4 includes the enhanced compilation toolchain and a set of bug fixes to support TensorFlow 2.0 features natively

AMD Infinity Fabric™ Link enablement
ROCm 2.4 adds support to connect two Radeon Instinct MI60 or Radeon Instinct MI50 boards via AMD Infinity Fabric™ Link GPU interconnect technology.

ROCm 2.3.0
--------------------------------------------------------------------------------

Mem usage per GPU
Per GPU memory usage is added to rocm-smi. Display information regarding used/total bytes for VRAM, visible VRAM and GTT, via the --showmeminfo flag

MIVisionX, v1.1 - ONNX
ONNX parser changes to adjust to new file formats

MIGraphX, v0.2
MIGraphX 0.2 supports the following new features:

New Python API
Support for additional ONNX operators and fixes that now enable a large set of Imagenet models
Support for RNN Operators
Support for multi-stream Execution
[Experimental] Support for Tensorflow frozen protobuf files
See: Getting-started:-using-the-new-features-of-MIGraphX-0.2 for more details

MIOpen, v1.8 - 3d convolutions and int8
This release contains full 3-D convolution support and int8 support for inference.
Additionally, there are major updates in the performance database for major models including those found in Torchvision.
See: MIOpen releases

Caffe2 - mGPU support
Multi-gpu support is enabled for Caffe2.

rocTracer library, ROCm tracing API for collecting runtimes API and asynchronous GPU activity traces
HIP/HCC domains support is introduced in rocTracer library.

BLAS - Int8 GEMM performance, Int8 functional and performance
Introduces support and performance optimizations for Int8 GEMM, implements TRSV support, and includes improvements and optimizations with Tensile.

Prioritized L1/L2/L3 BLAS (functional)
Functional implementation of BLAS L1/L2/L3 functions

BLAS - tensile optimization
Improvements and optimizations with tensile

MIOpen Int8 support
Support for int8

ROCm 2.2.0
--------------------------------------------------------------------------------

rocSparse Optimization on Vega20
Cache usage optimizations for csrsv (sparse triangular solve), coomv (SpMV in COO format) and ellmv (SpMV in ELL format) are available.

DGEMM and DTRSM Optimization
Improved DGEMM performance for reduced matrix sizes (k=384, k=256)

Caffe2
Added support for multi-GPU training

ROCm 2.1.0
--------------------------------------------------------------------------------

DGEMM Optimizations -
Improved DGEMM performance for large square and reduced matrix sizes (k=384, k=256)


ROCm 2.0.0
--------------------------------------------------------------------------------

PyTorch/Caffe2 with Vega 7nm Support
fp16 support is enabled
Several bug fixes and performance enhancements
Known Issue: breaking changes are introduced in ROCm 2.0 which are not addressed upstream yet. Meanwhile, please continue to use ROCm fork at https://github.com/ROCmSoftwarePlatform/pytorch
