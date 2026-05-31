#!/usr/bin/env python3
"""Single-file autoresearch training script for action-conditioned MiniGrid.

The agent edits this file. setup.py owns fixed data prep, dataloading, and eval.
"""

from __future__ import annotations

import gc
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from setup import (
    DEFAULT_DATA_PATH,
    DEFAULT_NUM_ACTIONS,
    TIME_BUDGET,
    ensure_dataset,
    evaluate_world_model,
    make_dataloader,
)


# ---------------------------------------------------------------------------
# Autoresearch knobs
# ---------------------------------------------------------------------------

DEPTH = 3
DATA_PATH = Path(os.environ.get("TW_DATA_PATH", DEFAULT_DATA_PATH))
TIME_BUDGET_SECONDS = float(os.environ.get("TW_TIME_BUDGET", TIME_BUDGET))
SEED = 42


@dataclass
class WorldModelConfig:
    depth: int
    frame_size: int
    patch_size: int
    context_length: int
    num_actions: int
    embed_dim: int
    num_heads: int
    mlp_ratio: int
    dropout: float
    batch_size: int
    learning_rate: float
    weight_decay: float
    warmdown_ratio: float
    eval_batches: int


def round_multiple(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def build_config(depth: int) -> WorldModelConfig:
    depth = max(1, int(depth))
    patch_size = 8 if depth <= 4 else 4
    context_length = min(4, max(1, (depth + 1) // 2))
    embed_dim = round_multiple(max(32, depth * 24), 16)
    num_heads = max(2, min(8, embed_dim // 32))
    embed_dim = round_multiple(embed_dim, num_heads)
    batch_size = max(16, 192 // max(1, depth))
    learning_rate = 4e-4 * math.sqrt(64 / embed_dim)
    return WorldModelConfig(
        depth=depth,
        frame_size=64,
        patch_size=patch_size,
        context_length=context_length,
        num_actions=DEFAULT_NUM_ACTIONS,
        embed_dim=embed_dim,
        num_heads=num_heads,
        mlp_ratio=4,
        dropout=0.0,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.05,
        warmdown_ratio=0.5,
        eval_batches=16,
    )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class PatchWorldModel(nn.Module):
    def __init__(self, config: WorldModelConfig):
        super().__init__()
        self.config = config
        self.grid_size = config.frame_size // config.patch_size
        self.num_patches = self.grid_size * self.grid_size
        patch_dim = 3 * config.patch_size * config.patch_size

        self.patch_embed = nn.Conv2d(
            3,
            config.embed_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias=False,
        )
        self.spatial_pos = nn.Parameter(torch.zeros(1, 1, self.num_patches, config.embed_dim))
        self.temporal_pos = nn.Parameter(
            torch.zeros(1, config.context_length, 1, config.embed_dim)
        )
        self.action_embed = nn.Embedding(config.num_actions, config.embed_dim)
        self.action_pos = nn.Parameter(torch.zeros(1, 1, config.embed_dim))

        self.blocks = nn.ModuleList([Block(config) for _ in range(config.depth)])
        self.norm = nn.LayerNorm(config.embed_dim)
        self.patch_head = nn.Linear(config.embed_dim, patch_dim)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))

    def _patchify_context(self, context: torch.Tensor) -> torch.Tensor:
        bsz, steps, channels, height, width = context.shape
        x = context.reshape(bsz * steps, channels, height, width)
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        x = x.reshape(bsz, steps, self.num_patches, self.config.embed_dim)
        x = x + self.spatial_pos + self.temporal_pos[:, :steps]
        return x.reshape(bsz, steps * self.num_patches, self.config.embed_dim)

    def _unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        bsz = patches.shape[0]
        p = self.config.patch_size
        g = self.grid_size
        patches = patches.reshape(bsz, g, g, 3, p, p)
        return patches.permute(0, 3, 1, 4, 2, 5).contiguous().reshape(
            bsz, 3, self.config.frame_size, self.config.frame_size
        )

    def forward(self, context: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        tokens = self._patchify_context(context)
        action_token = self.action_embed(action).unsqueeze(1) + self.action_pos
        tokens = torch.cat([tokens, action_token], dim=1)
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)

        start = (self.config.context_length - 1) * self.num_patches
        end = start + self.num_patches
        next_patch_tokens = tokens[:, start:end]
        next_patches = self.patch_head(next_patch_tokens)
        return torch.sigmoid(self._unpatchify(next_patches))

    def predict_next(self, context: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self(context, action)


class SelfAttention(nn.Module):
    def __init__(self, config: WorldModelConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.embed_dim // config.num_heads
        self.qkv = nn.Linear(config.embed_dim, 3 * config.embed_dim, bias=False)
        self.proj = nn.Linear(config.embed_dim, config.embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, embed_dim = x.shape
        qkv = self.qkv(x).reshape(
            bsz, seq_len, 3, self.num_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().reshape(bsz, seq_len, embed_dim)
        return self.proj(y)


class Block(nn.Module):
    def __init__(self, config: WorldModelConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.embed_dim)
        self.attn = SelfAttention(config)
        self.norm2 = nn.LayerNorm(config.embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(config.embed_dim, config.mlp_ratio * config.embed_dim),
            nn.GELU(),
            nn.Linear(config.mlp_ratio * config.embed_dim, config.embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def get_device() -> torch.device:
    requested = os.environ.get("TW_DEVICE")
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_lr_multiplier(progress: float, warmdown_ratio: float) -> float:
    if progress < 1.0 - warmdown_ratio:
        return 1.0
    cooldown = (1.0 - progress) / warmdown_ratio
    return max(0.0, min(1.0, cooldown))


def main() -> int:
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.set_float32_matmul_precision("high")

    config = build_config(DEPTH)
    data_path = ensure_dataset(DATA_PATH)
    device = get_device()
    autocast_enabled = device.type == "cuda"
    autocast_ctx = torch.amp.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=autocast_enabled,
    )

    train_loader = make_dataloader(
        data_path,
        split="train",
        context_length=config.context_length,
        batch_size=config.batch_size,
    )
    val_loader = make_dataloader(
        data_path,
        split="val",
        context_length=config.context_length,
        batch_size=config.batch_size,
        shuffle=False,
    )
    train_iter = iter(train_loader)

    model = PatchWorldModel(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=config.weight_decay,
    )
    if device.type == "cuda":
        model = torch.compile(model)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Config: {asdict(config)}")
    print(f"Device: {device}")
    print(f"Data: {data_path}")
    print(f"Time budget: {TIME_BUDGET_SECONDS:.0f}s")
    print(f"num_params_M: {num_params / 1e6:.3f}")

    total_training_time = 0.0
    smooth_loss = 0.0
    step = 0
    t_start = time.time()

    while True:
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()

        try:
            context, action, target = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            context, action, target = next(train_iter)

        context = context.to(device, non_blocking=True)
        action = action.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        progress = min(total_training_time / TIME_BUDGET_SECONDS, 1.0)
        lr_multiplier = get_lr_multiplier(progress, config.warmdown_ratio)
        for group in optimizer.param_groups:
            group["lr"] = config.learning_rate * lr_multiplier

        model.train()
        optimizer.zero_grad(set_to_none=True)
        with autocast_ctx:
            pred = model(context, action)
            loss = F.mse_loss(pred.contiguous(), target.contiguous())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0
        if step > 3:
            total_training_time += dt

        loss_f = float(loss.detach().item())
        if math.isnan(loss_f) or loss_f > 10:
            print("FAIL")
            return 1

        ema_beta = 0.9
        smooth_loss = ema_beta * smooth_loss + (1 - ema_beta) * loss_f
        debiased_loss = smooth_loss / (1 - ema_beta ** (step + 1))
        remaining = max(0.0, TIME_BUDGET_SECONDS - total_training_time)
        print(
            f"\rstep {step:05d} ({100 * progress:.1f}%) | "
            f"loss: {debiased_loss:.6f} | lrm: {lr_multiplier:.2f} | "
            f"dt: {dt * 1000:.0f}ms | remaining: {remaining:.0f}s",
            end="",
            flush=True,
        )

        if step == 0:
            gc.collect()
        step += 1
        if step > 3 and total_training_time >= TIME_BUDGET_SECONDS:
            break

    print()
    metrics = evaluate_world_model(
        model,
        val_loader,
        device=device,
        num_actions=config.num_actions,
        max_batches=config.eval_batches,
    )
    total_seconds = time.time() - t_start
    print("---")
    for key, value in metrics.items():
        print(f"{key}: {value:.8f}")
    print(f"training_seconds: {total_training_time:.1f}")
    print(f"total_seconds: {total_seconds:.1f}")
    print(f"num_steps: {step}")
    print(f"num_params_M: {num_params / 1e6:.3f}")
    print(f"depth: {DEPTH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
