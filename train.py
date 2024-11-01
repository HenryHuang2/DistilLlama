import logging
from pathlib import Path
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def distillation_loss(student_logits, teacher_logits, temperature=2.0):
    return F.kl_div(
        F.log_softmax(student_logits / temperature, dim=-1),
        F.softmax(teacher_logits / temperature, dim=-1),
        reduction='batchmean'
    ) * (temperature ** 2)

@torch.no_grad()
def evaluate(student_model, eval_dataloader, device, rank, n_random_sample):
    student_model.eval()
    total_loss = 0
    total_samples = 0
    
    eval_dataloader.sampler.set_epoch(rank)
    
    for batch in eval_dataloader:
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)
        
        _, loss = student_model(inputs, 0, targets)
        
        total_loss += loss.item() * inputs.size(0)
        total_samples += inputs.size(0)
        
        if total_samples >= n_random_sample:
            break
    
    # Gather losses from all processes
    all_losses = [torch.tensor(0.0).to(device) for _ in range(dist.get_world_size())]
    all_samples = [torch.tensor(0).to(device) for _ in range(dist.get_world_size())]
    
    dist.all_gather(all_losses, torch.tensor(total_loss).to(device))
    dist.all_gather(all_samples, torch.tensor(total_samples).to(device))
    
    # Compute the average loss across all processes
    total_loss = sum(all_losses)
    total_samples = sum(all_samples)
    
    student_model.train()
    
    return total_loss.item() / total_samples.item() if total_samples.item() > 0 else float('inf')

def train(rank, device, student_model, teacher_model, train_config, train_dataloader, eval_dataloader, args):
    student_model = student_model.to(device)
    teacher_model = teacher_model.to(device)
    teacher_model.eval()

    optimizer = AdamW(student_model.parameters(), lr=train_config.lr)

    # Total number of training steps
    total_steps = len(train_dataloader) * args['n_epochs']

    # Create the cosine scheduler with warmup
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args['warmup_steps'],
        num_training_steps=total_steps
    )

    start_epoch, start_step = 0, 0
    if args['load_model']:
        checkpoints = sorted(Path(args['checkpoint_dir']).glob('checkpoint_*.pt'))
        if len(checkpoints) > 0:
            chk_path = checkpoints[-1]  # Load the latest checkpoint
            print(f'Loading checkpoint {chk_path}')
            checkpoint = torch.load(chk_path, map_location='cpu')
            student_model.module.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            start_epoch = checkpoint['epoch']
            start_step = checkpoint['step'] + 1  # Start from the next step
            print(f'Resuming from epoch {start_epoch}, step {start_step}')
        else:
            print('No checkpoint found. Starting from scratch.')

    temperature = args['temperature']
    alpha = args['alpha']
    accumulation_steps = args['accumulation_steps']

    for epoch in range(start_epoch, args['n_epochs']):
        train_dataloader.sampler.set_epoch(epoch)
        student_model.train()
        
        optimizer.zero_grad()
        for step, (inputs, targets) in enumerate(train_dataloader, start=start_step):
            print(f"Rank {rank} Device {device} Epoch {epoch} Step {step}/{len(train_dataloader)}")
            
            inputs, targets = inputs.to(device), targets.to(device)
            
            student_logits, task_loss = student_model(inputs, 0, targets)
            with torch.no_grad():
                teacher_logits, _ = teacher_model(inputs, 0)

            distill_loss = distillation_loss(student_logits, teacher_logits, temperature)
            loss = (alpha * distill_loss + (1 - alpha) * task_loss)
            loss.backward() # one graph
            
            if (step + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            if (step + 1) % args['log_interval'] == 0:
                eval_loss = evaluate(student_model, eval_dataloader, device, rank, args['n_random_sample'])
                if rank == 0:
                    logging.info(
                        f'Epoch: {epoch}, Step: {step + 1}/{len(train_dataloader)}, '
                        f'Eval Loss: {eval_loss:.4f}'
                    )
                    save_checkpoint(student_model, optimizer, scheduler, epoch, step, eval_loss, args)
                    logging.info(f"Checkpoint saved at epoch {epoch}, step {step}")

        # Reset start_step to 0 after the first epoch when resuming
        if epoch == start_epoch:
            start_step = 0



def save_checkpoint(model, optimizer, scheduler, epoch, step, loss, args):
    checkpoint = {
        'model': model.module.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch,
        'step': step,
        'loss': loss,
    }
    checkpoint_path = f"{args['save_dir']}/checkpoint_epoch{epoch}_step{step}.pt"
    torch.save(checkpoint, checkpoint_path)
    logging.info(f"Checkpoint saved: {checkpoint_path}")
