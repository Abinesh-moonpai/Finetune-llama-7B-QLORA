# Fine-Tuning Llama-7B-Uncensored with Vicuna 70k Dataset Using Quantised Low Rank Adapations (LoRA)

This repository provides resources to fine-tune the Llama-7B-Uncensored model using the Vicuna 70k dataset and Quantised Low Rank Adapations (LoRA).

## Dataset: Vicuna 70k

The Vicuna 70k dataset is a rich collection of user-generated conversations sourced from ShareGPT.com. It encompasses a wide array of topics, including but not limited to:

- Casual conversation
- Storytelling
- Problem-solving

## Technique: Quantised Low Rank Adapations (LoRA)

LoRA is a cutting-edge technique designed to reduce the size of large language models without sacrificing performance. It achieves this by:

1. **Quantizing the Parameters:** Reducing the numerical precision of the model's parameters.
2. **Applying Low-Rank Approximations:** Utilizing low-rank approximations on the quantized parameters.



## Guide to Fine-Tuning

This repository offers a comprehensive guide to fine-tuning the Llama-7B-Uncensored model using the Vicuna 70k dataset with LoRA. It also includes detailed instructions for evaluating the fine-tuned model.

### Running the Code

To train the model, simply run the following command:

```bash
python train.py configs/open_llama_7b_qlora_uncensored.yaml
```

## Contributing

Feel free to contribute to this project by submitting issues, pull requests, or reaching out with any questions or suggestions.
