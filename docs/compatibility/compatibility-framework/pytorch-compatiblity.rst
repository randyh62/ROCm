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

Supported and unsupported tensor data types
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
      - 
      - 5.5
    * - torch.float8_e5m2
      - 8-bit floating point, e5m2
      - 
      - 5.5
    * - torch.float16 or torch.half
      - 16-bit floating point
      - 
      - 
    * - torch.bfloat16
      - 16-bit floating point
      - 1.2
      - 2.6
    * - torch.float32 or torch.float
      - 32-bit floating point
      - 
      - 
    * - torch.float64 or torch.double
      - 64-bit floating point
      - 
      - 
    * - torch.complex32 or torch.chalf
      - PyTorch provides native support for 32-bit complex numbers
      -
      -
    * - torch.complex64 or torch.cfloat
      - PyTorch provides native support for 64-bit complex numbers
      - 1.9
      - 2.0
    * - torch.complex128 or torch.cdouble
      - PyTorch provides native support for 128-bit complex numbers
      - 1.9
      - 2.0
    * - torch.uint8
      - 8-bit integer (unsigned)
      - 
      - 
    * - torch.uint16
      - 16-bit integer (unsigned)
      - 
      - 
    * - torch.uint32
      - 32-bit integer (unsigned)
      - 
      - 
    * - torch.uint64
      - 32-bit integer (unsigned)
      - 
      - 
    * - torch.int8
      - 8-bit integer (signed)
      - 
      - 
    * - torch.int16 or torch.short
      - 16-bit integer (signed)
      - 
      - 
    * - torch.int32 or torch.int
      - 32-bit integer (signed)
      - 
      - 
    * - torch.int64 or torch.long
      - 64-bit integer (signed)
      - 
      - 
    * - torch.bool
      - Boolean
      - 
      - 
    * - torch.quint8
      - quantized 8-bit integer (unsigned)
      - 
      - 
    * - torch.qint8
      - quantized 8-bit integer (signed)
      - 
      - 
    * - torch.qint32
      - quantized 32-bit integer (signed)
      - 
      - 
    * - torch.quint4x2
      - quantized 4-bit integer (unsigned)
      - 
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
