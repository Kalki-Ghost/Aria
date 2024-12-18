---
language:
- en
library_name: transformers
license: apache-2.0
pipeline_tag: image-text-to-text
tags:
- multimodal
- aria
base_model:
- rhymes-ai/Aria-Base-64K
---
<!-- <p align="center">
  <br>Aria</br>
</p>  -->


# Aria Model Card

[Dec 1, 2024] *We have released the base models (with native multimodal pre-training) for Aria ([Aria-Base-8K](https://huggingface.co/rhymes-ai/Aria-Base-8K) and [Aria-Base-64K](https://huggingface.co/rhymes-ai/Aria-Base-64K)) for research purposes and continue training.*
<!-- 
- Aria is the **first open multimodal native MoE** model, capable of seamlessly handling various input modalities within a MoE architecture.
- Aria performs **on par with GPT-4o mini and Gemini 1.5 Flash** across a range of multimodal tasks while maintaining strong performance on **text**-only tasks.
- Compared to similar or even larger models, Aria boasts **faster speeds** and **lower costs**. This high efficiency stems from its ability to activate only 3.9B parameters during inference – the **fewest** among models with comparable performance.
 -->
## Key features

- **SoTA Multimodal Native Performance**: Aria achieves strong performance on a wide range of multimodal, language, and coding tasks. It is superior in video and document understanding.
- **Lightweight and Fast**: Aria is a mixture-of-expert model with 3.9B activated parameters per token. It efficently encodes visual input of variable sizes and aspect ratios.  
- **Long Multimodal Context Window**: Aria supports multimodal input of up to 64K tokens. It can caption a 256-frame video in 10 seconds.

<p align="center">
🔗 <a href="https://rhymes.ai/" target="_blank"> Try Aria!</a> · 📖 <a href="https://www.rhymes.ai/blog-details/aria-first-open-multimodal-native-moe-model" target="_blank">Blog</a> · 📌 <a href="https://arxiv.org/pdf/2410.05993" target="_blank">Paper</a> 
 · ⭐ <a href="https://github.com/rhymes-ai/Aria" target="_blank">GitHub</a> · 🟣 <a href="https://discord.com/invite/u8HxU23myj" target="_blank"> Discord </a>
</p> 


<!-- # Model Info

| Model  | Download  | Parameter | Context Length |
| :---- | :------- | :------------ | :------ |
| Aria | < HF link - TBD> | • Activation: 3.9B (3.5B MoE + 0.4B Visual Encoder) <br> • Total: 25.3B | 64K           | -->

## Benchmark
| Category                            | Benchmark         |  Aria  | Pixtral 12B | Llama3.2 11B | GPT-4o mini | Gemini-1.5 Flash |
|:-------------------------------------|:-------------------|:--------:|:-------------:|:--------------:|:-------------:|:------------------:|
| **Knowledge (Multimodal)**          | MMMU              |  54.9  |    52.5     |    50.7      |    59.4     |      56.1        |
| **Math (Multimodal)**               | MathVista         |  66.1  |    58.0     |    51.5      |      -      |      58.4        |
| **Document**                        | DocQA             |  92.6  |    90.7     |    84.4      |      -      |      89.9        |
| **Chart**                           | ChartQA           |  86.4  |    81.8     |    83.4      |      -      |      85.4        |
| **Scene Text**                      | TextVQA           |  81.1  |      -      |      -       |      -      |      78.7        |
| **General Visual QA**               | MMBench-1.1       |  80.3  |      -      |      -       |    76.0     |        -         |
| **Video Understanding**             | LongVideoBench    |  65.3  |    47.4     |    45.7      |    58.8     |      62.4        |
| **Knowledge (Language)**            | MMLU (5-shot)     |  73.3  |    69.2     |    69.4      |      -      |      78.9        |
| **Math (Language)**                 | MATH              |  50.8  |    48.1     |    51.9      |    70.2     |        -         |
| **Reasoning (Language)**            | ARC Challenge     |  91.0  |      -      |    83.4      |    96.4     |        -         |
| **Coding**                          | HumanEval         |  73.2  |    72.0     |    72.6      |    87.2     |      74.3        |


## Quick Start
### Installation
```
# Install transformers from GitHub until the next release includes the Aria model
pip install git+https://github.com/huggingface/transformers.git

pip install accelerate sentencepiece torchvision requests torch Pillow
pip install flash-attn --no-build-isolation

# For better inference performance, you can install grouped-gemm, which may take 3-5 minutes to install
pip install grouped_gemm==0.1.6
```

### Inference

Aria has 25.3B total parameters, it can be loaded in one A100 (80GB) GPU with bfloat16 precision.

Here is a code snippet to show you how to use Aria.

```python
import requests
import torch
from PIL import Image

from transformers import AriaProcessor, AriaForConditionalGeneration


model_id_or_path = "rhymes-ai/Aria"
model = AriaForConditionalGeneration.from_pretrained(
    model_id_or_path, device_map="auto", torch_dtype=torch.bfloat16
)

processor = AriaProcessor.from_pretrained(model_id_or_path)

image = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"text": "what is the image?", "type": "text"},
        ],
    }
]

text = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=text, images=image, return_tensors="pt")
inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)
inputs.to(model.device)

output = model.generate(
    **inputs,
    max_new_tokens=15,
    stop_strings=["<|im_end|>"],
    tokenizer=processor.tokenizer,
    do_sample=True,
    temperature=0.9,
)
output_ids = output[0][inputs["input_ids"].shape[1]:]
response = processor.decode(output_ids, skip_special_tokens=True)
print(response)
```

### Advanced Inference and Fine-tuning
We provide a [codebase](https://github.com/rhymes-ai/Aria) for more advanced usage of Aria,
including vllm inference, cookbooks, and fine-tuning on custom datasets.



## Citation
If you find our work helpful, please consider citing.
```
@article{aria,
  title={Aria: An Open Multimodal Native Mixture-of-Experts Model}, 
  author={Dongxu Li and Yudong Liu and Haoning Wu and Yue Wang and Zhiqi Shen and Bowen Qu and Xinyao Niu and Guoyin Wang and Bei Chen and Junnan Li},
  year={2024},
  journal={arXiv preprint arXiv:2410.05993},
}
```