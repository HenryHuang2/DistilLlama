import torch
from torch.utils.data import Dataset
import pyarrow.parquet as pq
import sentencepiece as spm
from typing import List, Tuple, Callable

def llama_collate_fn(batch: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    x = [item[:-1] for item in batch]
    y = [item[1:] for item in batch]
    x_batch = torch.stack(x)
    y_batch = torch.stack(y)
    return x_batch, y_batch

class ParquetDataset(Dataset):
    def __init__(self, file_path: str, tokenizer: spm.SentencePieceProcessor, max_sequence_length: int, pad_token_id: int = 0, clean_up_fn: Callable = None) -> None:
        self.max_sequence_length = max_sequence_length
        self.pad_token_id = pad_token_id
        self.tokenizer = tokenizer
        
        table = pq.read_table(file_path)
        assert table.num_columns == 1, "Expected a single column in the Parquet file"
        column = table.column(0)
        self.raw_text_array = column.to_pylist()
        
        # TODO: use a better cleaning function
        if clean_up_fn is None:
            def clean_up_fn(line: str) -> str:
                proccessed_line = line.strip()
                if len(tokenizer.encode(proccessed_line)) < 0.5 * max_sequence_length:
                    return ""
                else:
                    return proccessed_line
                
        self.text_array = [clean_up_fn(text) for text in self.raw_text_array]
        self.text_array = [text for text in self.text_array if text]

    def __len__(self) -> int:
        return len(self.text_array)

    def __getitem__(self, index: int) -> torch.Tensor:
        text = self.text_array[index]
        tokenized_text = self.tokenizer.encode(text, out_type=int, add_bos=True, add_eos=True) # TODO: should we add bos and eos to the tokenizer?
        tokenized_text = tokenized_text[:self.max_sequence_length]
        padding_length = self.max_sequence_length - len(tokenized_text)
        if padding_length > 0:
            tokenized_text.extend([self.pad_token_id] * padding_length)
        assert len(tokenized_text) == self.max_sequence_length
        return torch.tensor(tokenized_text, dtype=torch.long)

class TextFileDataset(Dataset):
    def __init__(self, file_path: str, tokenizer: spm.SentencePieceProcessor, max_sequence_length: int, pad_token_id: int = 0, clean_up_fn: Callable = None) -> None:
        self.max_sequence_length = max_sequence_length
        self.pad_token_id = pad_token_id
        self.tokenizer = tokenizer
        
        if clean_up_fn is None:
            def clean_up_fn(line: str) -> str:
                proccessed_line = line.strip()
                if len(tokenizer.encode(proccessed_line)) < 0.5 * max_sequence_length:
                    return ""
                else:
                    return proccessed_line

        with open(file_path, 'r', encoding='utf-8') as file:
            self.lines = [clean_up_fn(line) for line in file]
            
        self.lines = [line for line in self.lines if line]

    def __len__(self) -> int:
        return len(self.lines)

    def __getitem__(self, index: int) -> torch.Tensor:
        line = self.lines[index]
        tokenized_line = self.tokenizer.encode(line, out_type=int, add_bos=True, add_eos=True)
        tokenized_line = tokenized_line[:self.max_sequence_length]
        padding_length = self.max_sequence_length - len(tokenized_line)
        if padding_length > 0:
            tokenized_line.extend([self.pad_token_id] * padding_length)
        assert len(tokenized_line) == self.max_sequence_length
        return torch.tensor(tokenized_line, dtype=torch.long)
