from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelArgs:
    dim: int = 512
    n_layers: int = 16
    n_heads: int = 8
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    mode: str = 'train'
    batch_size: int = 64
    max_seq_length: int = 128
    pre_trained_model_path: Optional[str] = None
    pre_trained_tokenizer_path: Optional[str] = None

@dataclass
class TrainArgs(ModelArgs):
    n_epochs: int = 2
    log_interval: int = 3000
    lr: float = 3e-4
    warmup_steps: int = 4000
    accumulation_steps: int = 8
    load_model: bool = True
    temperature: float = 2
    alpha: float = 0.3
    n_random_sample: int = 5000
    save_dir: str = 'Baby-Llama-Checkpoints'

@dataclass
class DataArgs(ModelArgs):
    wikitext_path: str = 'wikitext/wikitext-103-raw-v1'
    openwebtext_path: str = 'openwebtext_dataset1.txt'
    tokenizer_model_path: str = 'tokenizer.model'
    train_ratio: float = 0.9
    
@dataclass
class InferenceArgs(ModelArgs):
    checkpoint_dir: str = 'Baby-Llama-Checkpoints'
    tokenizer_path: str = 'tokenizer.model'
    max_seq_len: int = 120
    temperature: float = 0.6
    top_p: float = 0.9