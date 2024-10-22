import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import sentencepiece as spm
from pathlib import Path
from torch.utils.data import ConcatDataset

from arguments import get_args
from config import ModelArgs, TrainArgs
from data import ParquetDataset, TextFileDataset, llama_collate_fn
from model import Transformer
from train import train

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def run_main():
    args = get_args()
    
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(args['tokenizer_model_path'])
    
    wiki_dataset = ParquetDataset(args['wikitext_path'], tokenizer, args['max_seq_length'], pad_token_id=0)
    openweb_dataset = TextFileDataset(args['openwebtext_path'], tokenizer, args['max_seq_length'], pad_token_id=0)
    
    dataset = ConcatDataset([wiki_dataset, openweb_dataset])
    
    dataset_size = len(dataset)
    train_size = int(dataset_size * args['train_ratio'])
    eval_size = dataset_size - train_size

    train_dataset, eval_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, eval_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    world_size = int(os.environ["WORLD_SIZE"]) # 
    rank = int(os.environ["SLURM_PROCID"]) # 
    gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"]) # 
    assert gpus_per_node == torch.cuda.device_count()
    print(f"Hello from rank {rank} of {world_size} where there are {gpus_per_node} allocated GPUs per node.", flush=True)

    setup(rank, world_size)
    local_rank = rank - gpus_per_node * (rank // gpus_per_node)
    torch.cuda.set_device(local_rank)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=42)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args['batch_size'],
        sampler=train_sampler,
        num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]),
        pin_memory=True,
        collate_fn=llama_collate_fn
    )

    eval_sampler = DistributedSampler(eval_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=args['batch_size'],
        sampler=eval_sampler,
        num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]),
        pin_memory=True,
        collate_fn=llama_collate_fn
    )

    student_model_args = ModelArgs()
    student_model_args.vocab_size = tokenizer.vocab_size()
    student_model = Transformer(device=local_rank, args=student_model_args).to(device=local_rank)
    student_model = DDP(student_model, device_ids=[local_rank])
    teacher_model_args = ModelArgs(
        dim=4096,
        n_layers=32,
        n_heads=32,
        vocab_size=student_model_args.vocab_size,
        multiple_of=256,
        norm_eps=1e-5,
        max_seq_length=args['max_seq_length'],
        pre_trained_model_path='llama-2-7b'
    )
    teacher_model = Transformer(device=local_rank, args=teacher_model_args).to(device=local_rank, dtype=torch.bfloat16)


    # Load teacher model weights
    checkpoints = sorted(Path(teacher_model_args.pre_trained_model_path).glob('*.pth'))
    assert len(checkpoints) > 0
    chk_path = checkpoints[0]
    checkpoint = torch.load(chk_path, map_location='cpu')  # Load to CPU first
    if 'rope.freqs' in checkpoint:
        del checkpoint['rope.freqs']
    teacher_model.load_state_dict(checkpoint)
    del checkpoint  # Free up memory
    torch.cuda.empty_cache()

    train_config = TrainArgs()
    
    train(
        rank=rank,
        device=local_rank,
        student_model=student_model,
        teacher_model=teacher_model,
        train_config=train_config,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        args=args
    )
    
    cleanup()

if __name__ == '__main__':
    run_main()
