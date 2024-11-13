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

Supported features
================================================================================

✅: full support

⚠️: partial support

❌: not supported


.. list-table::
    :header-rows: 1

    * - Features
      - Description
      - Supported
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
    * - NVIDIA Tools Extension (NVTX)
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
    * - torch.cuda.StreamContext
      - Context-manager that selects a given stream.
      - ✅
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
    