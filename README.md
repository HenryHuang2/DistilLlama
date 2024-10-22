# Knowledge Distillation from LLaMA 2 (7B) to an 83M LLaMA Model

This codebase distills knowledge from Meta's LLaMA 2 (7B) model to a smaller 83M LLaMA model.

## File Descriptions

### `inference.py`
This script contains the inference code for generating outputs from the model.

- **Run it with:**
  ```bash
  python inference.py "Your First Prompt" "Your Second Prompt"...
  ```

### `data.py`
Provides PyTorch dataset classes and a collation function for preparing text data for language modeling tasks.

- **`ParquetDataset`:** Loads text from a Parquet file, processes and tokenizes each line using SentencePiece, and pads sequences to a specified maximum length.
- **`TextFileDataset`:** Similar to `ParquetDataset` but reads from a plain text file.
  - Both classes tokenize lines, truncate or pad sequences to a fixed length, and return PyTorch tensors.
- **Function - `llama_collate_fn`:** A collation function for batch processing that splits each tensor into input (`x`) and target (`y`) sequences, shifting tokens by one position.
  - For the sentence "I have an apple":
    - `x`: [<BOS>, "I", "have", "an"]
    - `y`: ["I", "have", "an", "apple", <EOS>]

### `train.py`
Contains training code for the model, using a teacher model. It employs a cosine scheduler with warmup and the AdamW optimizer.

### `model.py`
Defines the LLaMA 2 model structure in PyTorch.

### `download.sh`
Downloads the pretrained weights from Meta.

### `config.py`
Provides configuration for the model, training, data, and inference.

### `nparameters.py`
Calculates the number of model parameters.

### `main.py`
The main script for starting training. It assumes a SLURM environment but can run locally with adjustments:

```python
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
```

Set `world_size`, `rank`, and `gpus_per_node` based on your local environment. Also, adjust `num_workers` in the dataloader accordingly.

## Training Details
The model was trained on Spartan HPC at the University of Melbourne, using two A100 80GB GPUs. Here are some inference examples after 3 hours (30,000 steps) of training:

- **Given the prompt:** "City"
  - **Generated output:** "City Council personnel engaged Jewish settlement project"
- **Given the prompt:** "President"
  - **Generated output:** "President Donald Trumpt promises to support ..."

## Future Improvements

1. **Data Preprocessing:** Improve data preprocessing by cleaning and filtering text. Segment long sequences into smaller chunks.
2. **Teacher Model Selection:** Use a teacher model pretrained on the same dataset. For this project, since OpenWebText was used as the training dataset, GPT-2 would be a suitable teacher as it is trained on the same dataset and uses a matching tokenizer.
3. **DDP Optimization:** Currently, each GPU instance has a separate copy of the teacher model, which is inefficient. Consider optimizing the Distributed Data Parallel (DDP) setup.

## Acknowledgements
- University of Melbourne for providing the resources
- AGL Energy for initiating the project
- [LLaMA 2 from scratch](https://github.com/abdallah197/llama2-from-scratch) for codebase
- [Meta LLaMA](https://github.com/meta-llama/llama/tree/main) for pretrained weights

