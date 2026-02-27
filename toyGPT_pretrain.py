import math
import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass

# ── Fixed seed for pre-training ───────────────────────────────────────────────

seed = 1

# ── Dataset ──────────────────────────────────────────────────────────────────

np.random.seed(0)

n_digit = 10
seq_len = 10  # total T = seq_len + 1
max_seq_len = 10  # max digit positions for pre-training

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_variable_length_dataset(n_sample, min_len=2, max_len=50):
    """
    Build digit-wise addition examples with variable lengths (min_len–max_len digits).
    All sequences are padded to max_len+1 with a binary mask for valid positions.
    """
    T = max_len + 1
    inputs  = torch.zeros(n_sample, T, 2, dtype=torch.long)
    targets = torch.zeros(n_sample, T, dtype=torch.long)
    mask    = torch.zeros(n_sample, T, dtype=torch.float32)

    lengths = torch.randint(min_len, max_len + 1, (n_sample,))

    for idx in range(n_sample):
        sl = lengths[idx].item()
        actual_T = sl + 1

        inp = torch.randint(0, n_digit, (actual_T, 2))
        inp[-1, :] = 0
        tgt = torch.zeros(actual_T, dtype=torch.long)
        carry = 0
        for i in range(sl):
            s = inp[i, 0] + inp[i, 1] + carry
            tgt[i] = s % n_digit
            carry = s // n_digit
        tgt[sl] = carry

        inputs[idx, :actual_T] = inp
        targets[idx, :actual_T] = tgt
        mask[idx, :actual_T] = 1.0

    return inputs.float().to(device), targets.to(device), mask.to(device)


# ── Model ─────────────────────────────────────────────────────────────────────

class LayerNorm(nn.Module):
    def __init__(self, ndim, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size),
        )

    def forward(self, x):
        B, T, C = x.size()
        hs = C // self.n_head
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)
        q = q.view(B, T, self.n_head, hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(hs))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(F.gelu(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = seq_len + 1  # 11 — supports up to 10-digit addition
    n_layer: int = 1
    n_head: int = 2
    n_embd: int = 4
    dropout: float = 0.0
    bias: bool = False

class AdditionGPT(nn.Module):
    def __init__(self, config, input_dim: int = 2, output_dim: int = n_digit):
        super().__init__()
        self.config = config
        self.input_proj = nn.Linear(input_dim, config.n_embd, bias=config.bias)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = LayerNorm(config.n_embd, bias=config.bias)
        self.output_head = nn.Linear(config.n_embd, output_dim, bias=False)

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        n = sum(p.numel() for p in self.parameters())
        print(f"AdditionGPT (toy_v2): {n} parameters")
        assert n < 300, f"expected <300 parameters, got {n}"

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, x):
        B, T, _ = x.shape
        pos = torch.arange(T, dtype=torch.long, device=x.device)
        h = self.drop(self.input_proj(x / (n_digit - 1) - 0.5) + self.pos_emb(pos))
        for block in self.blocks:
            h = block(h)
        h = self.ln_f(h)
        return self.output_head(h)

# ── Metrics ──────────────────────────────────────────────────────────────────

def masked_accuracy(logits, target, mask):
    correct = (logits.argmax(dim=-1) == target).float()
    return (correct * mask).sum() / mask.sum()


def masked_accuracy_seq(logits, target, mask):
    correct = (logits.argmax(dim=-1) == target).float()
    seq_correct = ((correct * mask).sum(dim=-1) == mask.sum(dim=-1)).float()
    return seq_correct.mean()


# ── Training ─────────────────────────────────────────────────────────────────

torch.manual_seed(seed)

config = GPTConfig()
model = AdditionGPT(config)
model.to(device)

max_lr = 8e-3
min_lr = max_lr / 10
warmup_steps = 1000


def get_lr(step: int, total_steps: int) -> float:
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (max_lr - min_lr) * cosine


os.makedirs("ckpt", exist_ok=True)

# ════════════════════════════════════════════════════════════════════════════
# Pre-training on variable-length addition (2–10 digits)
# ════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("Pre-training on 2–10 digit addition (seed=1)")
print("="*80 + "\n")

pretrain_train_inputs, pretrain_train_targets, pretrain_train_mask = \
    create_variable_length_dataset(10000, min_len=2, max_len=10)
pretrain_test_inputs, pretrain_test_targets, pretrain_test_mask = \
    create_variable_length_dataset(10000, min_len=2, max_len=10)

n_steps_pt = 100000
optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr)

ckpt_path_pt = "ckpt/toygpt_pt.pt"
best_test_acc_seq_pt = -1.0

for i in range(n_steps_pt):
    lr = get_lr(i, n_steps_pt)
    for pg in optimizer.param_groups:
        pg["lr"] = lr

    # ── train step ──
    model.train()
    train_logits = model(pretrain_train_inputs)

    logits_flat  = train_logits.reshape(-1, n_digit)
    targets_flat = pretrain_train_targets.reshape(-1)
    mask_flat    = pretrain_train_mask.reshape(-1)

    per_token_loss = F.cross_entropy(logits_flat, targets_flat, reduction='none')
    loss = (per_token_loss * mask_flat).sum() / mask_flat.sum()

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad()

    # ── eval ──
    if i % 100 == 0:
        model.eval()
        with torch.no_grad():
            train_acc     = masked_accuracy(train_logits.detach(), pretrain_train_targets, pretrain_train_mask)
            train_acc_seq = masked_accuracy_seq(train_logits.detach(), pretrain_train_targets, pretrain_train_mask)

            test_logits = model(pretrain_test_inputs)
            logits_flat_t  = test_logits.reshape(-1, n_digit)
            targets_flat_t = pretrain_test_targets.reshape(-1)
            mask_flat_t    = pretrain_test_mask.reshape(-1)
            per_token_loss_t = F.cross_entropy(logits_flat_t, targets_flat_t, reduction='none')
            test_loss = (per_token_loss_t * mask_flat_t).sum() / mask_flat_t.sum()

            test_acc     = masked_accuracy(test_logits, pretrain_test_targets, pretrain_test_mask)
            test_acc_seq = masked_accuracy_seq(test_logits, pretrain_test_targets, pretrain_test_mask)

        if test_acc_seq.item() > best_test_acc_seq_pt:
            best_test_acc_seq_pt = test_acc_seq.item()
            torch.save({"step": i, "model": model.state_dict(),
                         "test_acc_seq": best_test_acc_seq_pt, "phase": "pretrain"}, ckpt_path_pt)

        print(
            f"[PT] Step {i:5d}  lr {lr:.2e}  loss {loss.item():.4f}  "
            f"train acc {train_acc:.4f}  test acc {test_acc:.4f}  "
            f"train seq {train_acc_seq:.4f}  test seq {test_acc_seq:.4f}  "
            f"best seq {best_test_acc_seq_pt:.4f}"
        )

print(f"\nPre-training complete. Best test seq acc: {best_test_acc_seq_pt:.4f}")
print(f"Checkpoint saved to {ckpt_path_pt}")