# Cformers

SoTA Transformers with C-backend for fast inference on your CPU.

## Introduction

We identify three pillers to enable fast inference of SoTA AI models on your CPU:
1. Fast C/C++ LLM inference kernels for CPU.
2. Machine Learning Research & Exploration front - Compression through quantization, sparsification, training on more data, collecting data and training instruction & chat models.
3. Easy to use API for fast AI inference in dynamically typed language like Python.

This project aims to address the third using LLaMa.cpp and GGML.

## Guiding Principles

- Inference Speed! Focus on inference, not training.
- Precompressed models.
- Minimal setup required - `pip install cformers` should be good to get started.
- Easily switch between models and quantization types.
- Support variety of prompts.

And most importantly:
- You, the users, get to decide which direction we take this project.

## Usage

Setup
```bash
pip install cformers
```

Usage:
```python
from cformers import AutoInference as AI
ai = AI('EleutherAI/gpt-j-6B')
x = ai.generate('def parse_html(html_doc):', num_tokens_to_generate=500)
print(x['token_str'])
```

OR

```python
from interface import AutoInference as AI
ai = AI('OpenAssistant/oasst-sft-1-pythia-12b')
x = ai.generate("<|prompter|>What's the Earth total population<|endoftext|><|assistant|>", num_tokens_to_generate=100)
print(x['token_str'])
```
OR

```bash
python chat.py
```
chat.py accepts the following parameteres:

- ``` -t 100 ``` Number of tokens to generate 
- ```-p Tell me a joke``` for a single prompt interaction
- ```-m pythia``` to load one of the available (bloom, pythia or gptj )

Following Architectures are supported:
- GPT-J
- BLOOM
- GPT-NeoX/Pythia/Open-Assistant/Open-Chat-Kit
- CodeGen

Currently following huggingface models are supported:
- EleutherAI/gpt-j-6B
- bigscience/bloom-560m
- bigscience/bloom-1b1
- bigscience/bloom-1b7
- bigscience/bloom-3b
- BigScience/bloom-7b1
- OpenAssistant/oasst-sft-1-pythia-12b
- togethercomputer/GPT-NeoXT-Chat-Base-20B (thanks to @[HCBlackFox](https://github.com/HCBlackFox))
- Salesforce/codegen-350M-mono
- Salesforce/codegen-2B-mono
- Salesforce/codegen-6B-mono
- Salesforce/codegen-16B-mono
- gpt2 (thanks to @[kamalojasv181](https://github.com/kamalojasv181))

We need to quantize and upload remaining models based on the supported architectures on huggingface. We would appreciate your help in this regard.

## Coming Soon:

Features:
- [X] Switch between models
- [ ] Chat-mode (interactive mode)
- [ ] Various tools to support Prompt-engineering, chaining, saving and sharing.

Code-base restructuring:
- [ ] Switch to Pybind11 rather than Subprocess - expected speedup: 3-4x
- [ ] Restructure the codebase to reuse.
- [ ] Somehow create llama.cpp as a git-submodule/dependency.

## Models

For now, we are focussing on AutoRegressive-style generative models.

- [x] GPT-J
- [x] BLOOM
- [x] GPT-NeoX/Pythia/Open-Assistant/Open-Chat-Kit (Architecture supported, need to quantize and upload models.)
- [X] CodeGen
- [ ] LLaMa & Alpaca
- [ ] OPT & Galactica
- [ ] T5
- [ ] RWKV
- [x] GPT-2
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

### Issues on Machine Learning side (some are exploratory):
- Try out GPTQ on these models and upload the resulting models to HF.
- Benchmark the quantized models. [#2](https://github.com/NolanoOrg/cformers/issues/2)
- Can we merge Query and Key Matrices for GPT-J/LLaMa? [#3](https://github.com/NolanoOrg/cformers/issues/3)
- Explore CALM (Confident Adaptive Language Modelling) with 4-bit precision models [#4](https://github.com/NolanoOrg/cformers/issues/4)
- Saving Keys and Values in memory at lower precision (refer FlexGen) [#6](https://github.com/NolanoOrg/cformers/issues/6)
- Try out other quantization techniques like proxy quantization, etc.
- Explore SparseGPT [#5](https://github.com/NolanoOrg/cformers/issues/5)
- Explore Quantization of Multimodal Models

### Non-Python side
If you are allergic to Python, you can:
- Port support for fast loading here: https://github.com/ggerganov/llama.cpp/issues/91#issuecomment-1473271638

You can also contribute to LLaMa.cpp and we will port those niceties here.
- Add support for greater than 32 bin/group size int4 quantized weights with GGML/LLaMa.cpp (A potential pitfalls - the intermediate representation may not be losslessly grouppable to >32 bin size, only weight matrix may be grouppable to >32 bin size, etc.)
- Speed up quantized matrix multiplication in GGML/LLaMa.cpp
- Add Int3 and Int2 quantization support to GGML/LLaMa.cpp
- Add fast Ampere-sparse quantized matrix multiplication functions in GGML/LLaMa.cpp

## Misc. Notes

Our interface is still limited to generation. We are working to support other features:

- [ ] Allow stopping-generation midway
- [ ] Anti-prompt for conversation models
- [ ] Returning embeddings and/or logits.
- [ ] [Dev] Switch to pybindings over the C++ kernels and calling them from Python.
- [ ] [Dev] Re-use the code in main.cpp

We would love to hear from you various ways in which we can speed up and improve the interface.

## License
MIT License

## Communication and Support

Discord: https://discord.gg/HGujTPQtR6
