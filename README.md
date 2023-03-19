# Cformers

Fast inference of SoTA AI Models on your CPU.

## Introduction

We identify three pillers to enable fast inference of SoTA AI models on your CPU:
1. Fast C/C++ inference kernels utilizing SIMD - like GGML, LLaMa.cpp
2. Machine Learning Research & Exploration front - Compression through quantization, sparsification, training on more data, collecting data and training instruction & chat models - like Open-Assistant, Open-Chat-Kit, etc.
3. Easy to use API for fast AI inference in dynamically typed language like Python.

This repo is focussed on addressing the third one.

## Tentative Guiding Principles

- Moar Speed. Focus on inference, not training.
- Precompressed models.
- Minimal setup required - `pip install cformers` and you are good to go.
- Easily switch between models and quantization types.
- Support variety of prompts.

And most importantly:
- You, the users, get to decide which direction we take this project.

## Usage

```bash
git clone https://github.com/nolanoOrg/cformers.git
cd cformers/cpp && make && cd ..
python -c

```

## Coming Soon:

Features:
- [X] Switch between models
- [ ] Chat-mode (interactive mode)
- [ ] Support for more prompts

Code-base (for dev-sanity)
- Reuse the code wherever possible.

## Models

For now, we are focussing on AutoRegressive-style generative models.

- [x] GPT-J
- [x] BLOOM
- [ ] GPT-NeoX/Pythia/Open-Assistant/Open-Chat-Kit
- [ ] CodeGen
- [ ] LLaMa & Alpaca
- [ ] OPT & Galactica
- [ ] T5
- [ ] RWKV
- [ ] GPT-2
- And more (including multimodal)...

## Quantization types:
- [x] Int4 with fixed zero-offset 
- [ ] Int4 with variable zero-offset
- [ ] GPTQ-Int4 with fixed zero-offset
- [ ] GPTQ-Int4 with variable zero-offset
- Int3 quantization, proxy quantization and binning.

## Contributions

We encourage contributions from the community.

### Providing feedback:

- Let us know what features you want, what models you want to use.
- Reporting bugs, raising issues and sending Pull Requests.


### Easy first issues:
Following are some easy first issues ways in which you can help improve CTransformers:
- Pick an existing HF model, quantize it, upload to HF and add it to the mapping in `ctransformers/map_model_to_url.py`
- Add support for new models.
- Add support for new quantization types.
- Add pipelines (Prompts) for various tasks - sentiment, summarization, email writing, etc.

### Issues on Machine Learning exploration (low-key research) side:
- Try out GPTQ on these models and upload the resulting models to HF. [TODO: Link to Issue]
- Benchmark the quantized models. [TODO: Link to Issue]
- Can we merge Query and Key Matrices for GPT-J/LLaMa? [TODO: Link to Issue]
- Explore CALM (Confident Adaptive Language Modelling) with 4-bit precision models [TODO: Link to Issue]
- Saving Keys and Values in memory at lower precision (refer FlexGen) [TODO: Link to Issue]
- Try out other quantization techniques like proxy quantization, etc. [TODO: Link to Issue]
- Explore SparseGPT [TODO: Link to Issue]
- Explore Quantization of Multimodal Models [TODO: Link to Issue]

### Non-Python side
If you are allergic to Python, you can:
- Port support for fast loading here: https://github.com/ggerganov/llama.cpp/issues/91#issuecomment-1473271638

You can also contribute to LLaMa.cpp and we will port those niceties here.
- Add support for greater than 32 bin/group size int4 quantized weights with GGML/LLaMa.cpp (A potential pitfalls - the intermediate representation may not be losslessly grouppable to >32 bin size, only weight matrix may be grouppable to >32 bin size, etc.)
- Speed up quantized matrix multiplication in GGML/LLaMa.cpp
- Add Int3 and Int2 quantization support to GGML/LLaMa.cpp
- Add fast Ampere-sparse quantized matrix multiplication functions in GGML/LLaMa.cpp


## Known Limitations (We are fixing this.)

This is currently 3-4 times slower than the [both](https://github.com/NolanoOrg/llama-int4-quant) the [existing](https://github.com/ggerganov/llama.cpp) C++ implementations of LLaMa.

We are creating bindings over the C++ kernels and calling them from Python to speed this up.

We would love to hear from you various ways in which we can speed this up.

## License
MIT License

## Communication and Support

Discord: https://discord.gg/peBU7yWa
