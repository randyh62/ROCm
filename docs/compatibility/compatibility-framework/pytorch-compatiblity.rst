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

Supported features
================================================================================

✅: full support

⚠️: partial support

❌: not supported

The supported data types in PyTorch are listed in the following table.

.. list-table::
    :header-rows: 1

    * - Features
      - Description
      - Supported
    * - Float8 support
      - | Float8 offers two types of formats, E4M3 and E5M2,
        | and has the potential to further reduce memory usage
        | and accelerate training and inference for large models,
        | especially on hardware designed to support it.
      - ❌
    * - Float16 support
      - | PyTorch provides a ``torch.float16`` data type, which can be
        | used for tensor creation, conversion, and storage. You can
        | create float16 tensors directly with 
        | ``torch.tensor(data, dtype=torch.float16)``, or convert an
        | existing tensor to float16 using ``.to(torch.float16)``.
      - ✅
    * - BFloat16 support
      - | PyTorch provides a ``torch.bfloat16`` data type, which can be
        | used for tensor creation, conversion, and storage. You can
        | create bfloat16 tensors directly with
        | ``torch.tensor(data, dtype=torch.bfloat16)``, or convert an
        | existing tensor to bfloat16 using ``.to(torch.bfloat16)``.
      - ✅
    * - TF32
      - 
      - ❌
    * - Complex support
      - | PyTorch provides native support for complex numbers with
        | two data types: ``torch.complex64`` and ``torch.complex128``.
      - ✅
    * - AMP (Automatic Mixed Precision)
      - 
      - ✅

The supported data types in PyTorch are listed in the following table.

.. list-table::
    :header-rows: 1

    * - Random Number Generator
      - 
      - ✅
    * - Communication collectives
      - 
      - ✅
    * - Streams and events
      - 
      - ✅
    * - Graphs (beta)
      - 
      - ✅
    * - Memory management
      - 
      - ✅
    * - Running process lists
      - | Return a human-readable printout of the running processes
        | and their GPU memory use for a given device.
      - ✅
    * - CUDACachingAllocator bypass
      -
      - ✅
    * - CUDA Fuser
      -
      - ❌
    * - Enable stream priorities
      - 
      - ✅
    * - Tensor scatter functions
      - | Functions are specialized tensor operations used for
        | manipulating tensors by "scattering" data to specific
        | indices.
      - ✅
    * - NVIDIA Tools Extension (NVTX)
      - 
      - ✅
    * - Lazy loading NVRTC
      - 
      - ✅
    * - Jiterator (beta)
      - Context-manager that selects a given stream.
      - ✅

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



Feature

CUDA Graphs





Capturable CUDAGeneratorImpl

CuDNN-based LSTM:Support

Non-Deterministic Alert CUDA Operations

TorchScript

Custom Python Classes

Distributed

TensorPipe

RPC Device Map Passing

Gloo

MPI

TorchElastic

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





BETA

NHWC

FX:Conv/Batch Norm fuser

Vision: Quantized Transfer Learning

BERT: Dynamic Quantization