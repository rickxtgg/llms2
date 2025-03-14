import os
import platform
import argparse
import time
import math
import warnings
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, DistributedSampler, Dataset, Subset
from torch.utils.data.dataset import ConcatDataset
from contextlib import nullcontext
from pathlib import Path
from transformers import AutoTokenizer

from model.model import MiniMindLM
from model.LMConfig import LMConfig
from model.dataset import PretrainDataset

warnings.filterwarnings('ignore')


class StreamingPretrainDataset(Dataset):
    """流式数据加载的预训练数据集，减少内存占用"""
    def __init__(self, data_path, tokenizer, max_length=512, buffer_size=1000, seed=42):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.buffer_size = buffer_size
        self.data_path = data_path
        self.random = random.Random(seed)
        self.buffer = []
        self.file_iter = None
        self._init_file_iter()
        self._fill_buffer()
    
    def _init_file_iter(self):
        self.file = open(self.data_path, 'r', encoding='utf-8')
        self.file_iter = iter(self.file)
    
    def _fill_buffer(self):
        try:
            while len(self.buffer) < self.buffer_size:
                line = next(self.file_iter)
                data = json.loads(line.strip())
                self.buffer.append(data)
        except StopIteration:
            # 如果文件读完了，重新开始读取
            self.file.close()
            self._init_file_iter()
            if not self.buffer:  # 如果缓冲区为空，再次尝试填充
                self._fill_buffer()
    
    def __len__(self):
        # 返回一个足够大的值，因为我们是流式读取
        return 1000000  # 假设数据集足够大
    
    def __getitem__(self, index):
        # 从缓冲区随机选择一个样本
        if not self.buffer:
            self._fill_buffer()
        
        idx = self.random.randint(0, len(self.buffer) - 1)
        sample = self.buffer[idx]
        
        # 从缓冲区移除这个样本并添加新样本
        self.buffer.pop(idx)
        try:
            line = next(self.file_iter)
            data = json.loads(line.strip())
            self.buffer.append(data)
        except StopIteration:
            self._init_file_iter()
            self._fill_buffer()
        
        # 构建输入文本
        text = f"{self.tokenizer.bos_token}{str(sample['text'])}{self.tokenizer.eos_token}"
        
        # 数据增强：随机截断
        if self.random.random() < 0.3 and len(text) > 100:
            max_cut = min(len(text) // 2, 100)
            cut_len = self.random.randint(10, max_cut)
            start_idx = self.random.randint(0, len(text) - cut_len - 1)
            text = text[:start_idx] + text[start_idx + cut_len:]
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoding.input_ids.squeeze()
        loss_mask = (input_ids != self.tokenizer.pad_token_id)

        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)
        return X, Y, loss_mask


class LabelSmoothingLoss(nn.Module):
    """标签平滑损失函数"""
    def __init__(self, smoothing=0.1, reduction='mean'):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction
        self.confidence = 1.0 - smoothing
    
    def forward(self, pred, target, mask=None):
        pred = pred.view(-1, pred.size(-1))
        target = target.view(-1)
        
        if mask is not None:
            mask = mask.view(-1)
            pred = pred[mask.bool()]
            target = target[mask.bool()]
        
        logprobs = torch.nn.functional.log_softmax(pred, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def Logger(content):
    if not ddp or dist.get_rank() == 0:
        print(content)


def get_lr_scheduler(optimizer, args, total_steps):
    """获取学习率调度器"""
    if args.lr_scheduler == 'cosine':
        return CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=args.min_lr)
    elif args.lr_scheduler == 'linear_warmup_cosine':
        warmup_scheduler = LinearLR(
            optimizer, 
            start_factor=0.1, 
            end_factor=1.0, 
            total_iters=args.warmup_steps
        )
        cosine_scheduler = CosineAnnealingLR(
            optimizer, 
            T_max=total_steps - args.warmup_steps, 
            eta_min=args.min_lr
        )
        return SequentialLR(
            optimizer, 
            schedulers=[warmup_scheduler, cosine_scheduler], 
            milestones=[args.warmup_steps]
        )
    elif args.lr_scheduler == 'cosine_warmup':
        warmup_scheduler = LinearLR(
            optimizer, 
            start_factor=0.1, 
            end_factor=1.0, 
            total_iters=args.warmup_steps
        )
        cosine_scheduler = CosineAnnealingLR(
            optimizer, 
            T_max=total_steps - args.warmup_steps, 
            eta_min=args.min_lr
        )
        return SequentialLR(
            optimizer, 
            schedulers=[warmup_scheduler, cosine_scheduler], 
            milestones=[args.warmup_steps]
        )
    else:
        # 默认使用自定义的余弦调度
        return None


def train_epoch(epoch, wandb, args, model, optimizer, train_loader, scaler, lr_scheduler=None):
    loss_fct = LabelSmoothingLoss(smoothing=args.label_smoothing) if args.label_smoothing > 0 else nn.CrossEntropyLoss(reduction='none')
    start_time = time.time()
    total_loss = 0
    total_samples = 0
    
    for step, (X, Y, loss_mask) in enumerate(train_loader):
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)

        # 使用学习率调度器
        if lr_scheduler is None:
            # 使用自定义的余弦调度
            lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch, args.learning_rate)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        else:
            lr = optimizer.param_groups[0]['lr']

        with ctx:
            res = model(X)
            if args.label_smoothing > 0:
                # 使用标签平滑损失
                loss = loss_fct(
                    res.logits.view(-1, res.logits.size(-1)),
                    Y.view(-1),
                    loss_mask.view(-1)
                )
            else:
                # 使用普通交叉熵损失
                loss = loss_fct(
                    res.logits.view(-1, res.logits.size(-1)),
                    Y.view(-1)
                ).view(Y.size())
                loss = (loss * loss_mask).sum() / loss_mask.sum()
            
            # 添加辅助损失
            loss += res.aux_loss
            loss = loss / args.accumulation_steps

        # 更新统计信息
        batch_size = X.size(0)
        total_loss += loss.item() * args.accumulation_steps * batch_size
        total_samples += batch_size

        scaler.scale(loss).backward()

        if (step + 1) % args.accumulation_steps == 0:
            # 梯度裁剪
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)
            
            # 更新学习率
            if lr_scheduler is not None:
                lr_scheduler.step()

        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            Logger(
                'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.8f} epoch_Time:{}min:'.format(
                    epoch + 1,
                    args.epochs,
                    step,
                    iter_per_epoch,
                    loss.item() * args.accumulation_steps,
                    lr,
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60))

            if (wandb is not None) and (not ddp or dist.get_rank() == 0):
                wandb.log({
                    "loss": loss.item() * args.accumulation_steps,
                    "lr": lr,
                    "epoch_Time": spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60
                })

        # 保存检查点
        if (step + 1) % args.save_interval == 0 and (not ddp or dist.get_rank() == 0):
            save_checkpoint(model, optimizer, epoch, step, args, scaler, lr_scheduler)
    
    # 计算平均损失
    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    return avg_loss


def get_lr(current_step, total_steps, lr):
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))


def save_checkpoint(model, optimizer, epoch, step, args, scaler=None, lr_scheduler=None):
    """保存检查点"""
    model.eval()
    moe_path = '_moe' if lm_config.use_moe else ''
    ckp_name = f'pretrain_{lm_config.dim}{moe_path}'
    
    # 保存最新的检查点
    ckp_path = f'{args.save_dir}/{ckp_name}.pth'
    
    # 每隔一定步数保存一个版本的检查点
    if args.save_version_interval > 0 and step % args.save_version_interval == 0:
        version_path = f'{args.save_dir}/{ckp_name}_e{epoch}_s{step}.pth'
    else:
        version_path = None
    
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    
    # 保存完整检查点（包括优化器状态等）
    checkpoint = {
        'model': state_dict,
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'step': step,
        'config': lm_config.__dict__,
        'args': vars(args)
    }
    
    if scaler is not None:
        checkpoint['scaler'] = scaler.state_dict()
    
    if lr_scheduler is not None:
        checkpoint['lr_scheduler'] = lr_scheduler.state_dict()
    
    # 保存最新检查点
    torch.save(checkpoint, ckp_path)
    
    # 保存版本检查点
    if version_path:
        torch.save(checkpoint, version_path)
    
    # 只保存模型权重（兼容旧版本）
    torch.save(state_dict, f'{args.save_dir}/{ckp_name}_weights_only.pth')
    
    model.train()


def load_checkpoint(model, optimizer, args, scaler=None, lr_scheduler=None):
    """加载检查点"""
    if not args.resume:
        return 0, 0
    
    checkpoint_path = args.resume
    if not os.path.exists(checkpoint_path):
        Logger(f"Checkpoint {checkpoint_path} not found, starting from scratch.")
        return 0, 0
    
    Logger(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=args.device)
    
    # 加载模型权重
    if 'model' in checkpoint:
        # 新版本检查点
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model.module.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint['model'])
        
        # 加载优化器状态
        if 'optimizer' in checkpoint and optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
        
        # 加载scaler状态
        if 'scaler' in checkpoint and scaler is not None:
            scaler.load_state_dict(checkpoint['scaler'])
        
        # 加载学习率调度器状态
        if 'lr_scheduler' in checkpoint and lr_scheduler is not None:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        
        # 返回恢复的轮次和步数
        return checkpoint.get('epoch', 0), checkpoint.get('step', 0)
    else:
        # 旧版本检查点（只有模型权重）
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model.module.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint)
        return 0, 0


def init_model(lm_config, args):
    """初始化模型和分词器"""
    tokenizer = AutoTokenizer.from_pretrained('./model/minimind_tokenizer')
    model = MiniMindLM(lm_config).to(args.device)
    
    # 初始化模型权重
    if args.init_method != 'default':
        Logger(f"Initializing model weights with {args.init_method} method")
        for name, p in model.named_parameters():
            if 'weight' in name:
                if args.init_method == 'normal':
                    nn.init.normal_(p, mean=0.0, std=0.02)
                elif args.init_method == 'xavier_uniform':
                    nn.init.xavier_uniform_(p)
                elif args.init_method == 'xavier_normal':
                    nn.init.xavier_normal_(p)
                elif args.init_method == 'kaiming_uniform':
                    nn.init.kaiming_uniform_(p)
                elif args.init_method == 'kaiming_normal':
                    nn.init.kaiming_normal_(p)
    
    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    Logger(f'LLM总参数量：{total_params / 1e6:.3f} 百万')
    
    # 验证模型结构
    if args.validate_model:
        Logger("Validating model structure...")
        with torch.no_grad():
            dummy_input = torch.randint(0, lm_config.vocab_size, (1, 10), device=args.device)
            try:
                output = model(dummy_input)
                Logger("Model structure validation passed!")
            except Exception as e:
                Logger(f"Model structure validation failed: {e}")
                raise e
    
    return model, tokenizer


def init_distributed_mode(args):
    """初始化分布式训练环境"""
    if not args.ddp:
        return
    
    global ddp_local_rank, DEVICE
    
    # 初始化进程组
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    
    # 获取分布式训练信息
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    DEVICE = f"cuda:{ddp_local_rank}" if torch.cuda.is_available() else "cpu"
    
    # 设置当前设备
    if torch.cuda.is_available():
        torch.cuda.set_device(DEVICE)
    
    # 打印分布式训练信息
    Logger(f"Initialized distributed training with rank {ddp_rank}, local_rank {ddp_local_rank}, world_size {ddp_world_size}")
    
    # 设置随机种子，确保每个进程使用不同的数据
    torch.manual_seed(args.seed + ddp_rank)
    np.random.seed(args.seed + ddp_rank)
    random.seed(args.seed + ddp_rank)


# torchrun --nproc_per_node 2 train_pretrain_enhanced.py
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Enhanced Pretraining")
    # 基本训练参数
    parser.add_argument("--out_dir", type=str, default="out")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--seed", type=int, default=1337)
    
    # 学习率调度参数
    parser.add_argument("--lr_scheduler", type=str, default="", choices=["", "cosine", "linear_warmup_cosine", "cosine_warmup"])
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--warmup_ratio", type=float, default=0.0, help="预热步数比例，优先级高于warmup_steps")
    parser.add_argument("--min_lr", type=float, default=1e-5)
    
    # 优化器参数
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--accumulation_steps", type=int, default=8)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    
    # 损失函数参数
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    
    # 数据加载参数
    parser.add_argument("--data_path", type=str, default="./dataset/pretrain_hq.jsonl")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--streaming", action="store_true", help="使用流式数据加载")
    parser.add_argument("--buffer_size", type=int, default=1000, help="流式数据加载的缓冲区大小")
    
    # 模型参数
    parser.add_argument('--dim', default=512, type=int)
    parser.add_argument('--n_layers', default=8, type=int)
    parser.add_argument('--max_seq_len', default=512, type=int)
    parser.add_argument('--use_moe', action="store_true")
    parser.add_argument('--init_method', type=str, default='default', 
                        choices=['default', 'normal', 'xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal'])
    parser.add_argument('--validate_model', action="store_true")
    
    # 分布式训练参数
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument('--local_rank', type=int, default=-1)
    
    # 日志和保存参数
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--save_version_interval", type=int, default=1000)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain")
    
    # 恢复训练参数
    parser.add_argument("--resume", type=str, default="", help="恢复训练的检查点路径")
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # 创建保存目录
    args.save_dir = os.path.join(args.out_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    
    # 设置设备类型
    device_type = "cuda" if "cuda" in args.device else "cpu"
    
    # 设置混合精度上下文
    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype={
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
        'float32': torch.float32
    }[args.dtype])
    
    # 初始化分布式环境
    ddp = int(os.environ.get("RANK", -1)) != -1  # 是否是分布式训练
    ddp_local_rank, DEVICE = 0, "cuda:0" if torch.cuda.is_available() else "cpu"
    
    if ddp:
        args.ddp = True
        init_distributed_mode(args)
        args.device = torch.device(DEVICE)
    
    # 初始化wandb
    args.wandb_run_name = f"MiniMind-Pretrain-Dim-{args.dim}-Layers-{args.n_layers}-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LR-{args.learning_rate}"
    
    if args.use_wandb and (not ddp or ddp_local_rank == 0):
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))
    else:
        wandb = None
    
    # 初始化模型配置
    lm_config = LMConfig(dim=args.dim, n_layers=args.n_layers, max_seq_len=args.max_seq_len, use_moe=args.use_moe)
    
    # 初始化模型和分词器
    model, tokenizer = init_model(lm_config, args)
    
    # 初始化数据集
    if args.streaming:
        train_ds = StreamingPretrainDataset(
            args.data_path, 
            tokenizer, 
            max_length=lm_config.max_seq_len,
            buffer_size=args.buffer_size
        )
    else:
        train_ds = PretrainDataset(args.data_path, tokenizer, max_length=lm_config.max_seq_len)
    
    # 初始化数据加载器
    train_sampler = DistributedSampler(train_ds) if ddp else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=False,
        shuffle=False if ddp else True,
        num_workers=args.num_workers,
        sampler=train_sampler
    )
    
    # 初始化优化器
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay
    )
    
    # 初始化混合精度训练
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16']))
    
    # 计算总训练步数
    iter_per_epoch = len(train_loader)
    total_steps = iter_per_epoch * args.epochs
    
    # 如果设置了warmup_ratio，则根据总步数计算warmup_steps
    if args.warmup_ratio > 0:
        args.warmup_steps = int(total_steps * args.warmup_ratio)
        Logger(f"根据warmup_ratio={args.warmup_ratio}计算得到warmup_steps={args.warmup_steps}")
    
    # 初始化学习率调度器
    lr_scheduler = get_lr_scheduler(optimizer, args, total_steps) if args.lr_scheduler else None
    
    # 分布式训练设置
    if ddp:
        model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank])
    
    # 恢复训练
    start_epoch, start_step = load_checkpoint(model, optimizer, args, scaler, lr_scheduler)
    
    # 训练循环
    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        avg_loss = train_epoch(epoch, wandb, args, model, optimizer, train_loader, scaler, lr_scheduler)
        
        # 每个epoch结束后保存检查点
        if not ddp or dist.get_rank() == 0:
            save_checkpoint(model, optimizer, epoch + 1, 0, args, scaler, lr_scheduler)
            Logger(f"Epoch {epoch + 1} completed with average loss: {avg_loss:.4f}")
    
    # 训练结束
    Logger("Training completed!")
    
    # 关闭wandb
    if wandb is not None and (not ddp or dist.get_rank() == 0):
        wandb.finish()