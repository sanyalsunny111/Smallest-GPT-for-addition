import argparse
import math
import os
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
from collections import deque

# ── Args ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=1)
args = parser.parse_args()

seed = args.seed

# ── Dataset ──────────────────────────────────────────────────────────────────

np.random.seed(0)

n_digit = 10
seq_len = 10  # total T = seq_len + 1
max_seq_len = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_dataset(n_sample, sl=seq_len):
    T = sl + 1
    inputs = torch.randint(0, n_digit, (n_sample, T, 2))
    inputs[:, -1, :] = 0
    targets = torch.zeros(n_sample, T, dtype=torch.long)
    carry = 0
    for i in range(sl):
        s = inputs[:, i, 0] + inputs[:, i, 1] + carry
        targets[:, i] = s % n_digit
        carry = s // n_digit
    targets[:, sl] = carry
    return inputs.float().to(device), targets.to(device)


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
    block_size: int = seq_len + 1
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

def accuracy(logits, target):
    return (logits.argmax(dim=-1) == target).float().mean()


def accuracy_seq(logits, target):
    return (logits.argmax(dim=-1) == target).all(dim=-1).float().mean()


# ── Training ─────────────────────────────────────────────────────────────────

torch.manual_seed(seed)

config = GPTConfig()
model = AdditionGPT(config)
model.to(device)

n_params = sum(p.numel() for p in model.parameters())

# Load pre-trained checkpoint
ckpt_path_pt = "ckpt/toygpt_pt.pt"
if not os.path.exists(ckpt_path_pt):
    print(f"ERROR: Pre-trained checkpoint not found at {ckpt_path_pt}")
    print("Run pretrain_lawa4.py first.")
    exit(1)

ckpt_pt = torch.load(ckpt_path_pt, map_location=device, weights_only=True)
model.load_state_dict(ckpt_pt["model"])
print(f"Loaded pre-trained checkpoint from step {ckpt_pt['step']} (seq acc {ckpt_pt['test_acc_seq']:.4f})")

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
os.makedirs("logs", exist_ok=True)

# ── Logging setup ─────────────────────────────────────────────────────────────

log_path = f"logs/toygpt_lawa_seed={seed}.txt"
log_file = open(log_path, "w")

def log(msg):
    print(msg)
    log_file.write(msg + "\n")
    log_file.flush()


# ════════════════════════════════════════════════════════════════════════════
# Fine-tuning on 10-digit addition (with LAWA)
# ════════════════════════════════════════════════════════════════════════════

log("\n" + "="*80)
log(f"Fine-tuning on 10-digit addition (with LAWA) — seed={seed}")
log("="*80 + "\n")

train_inputs, train_targets = create_dataset(10000)
test_inputs, test_targets = create_dataset(10000)

n_steps_ft = 100000
optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr)

ckpt_path_ft = f"ckpt/toygpt_ft_seed={seed}.pt"
best_test_acc_seq = -1.0
best_lawa_acc_seq = -1.0

# ── LAWA: Latest Weight Averaging ────────────────────────────────────────────

lawa_k = 5
lawa_interval = 2000
lawa_buffer = deque(maxlen=lawa_k)
lawa_model = copy.deepcopy(model)


def update_lawa():
    if not lawa_buffer:
        return
    avg_sd = {}
    for key in lawa_buffer[0]:
        avg_sd[key] = torch.stack([sd[key].float() for sd in lawa_buffer]).mean(dim=0)
    lawa_model.load_state_dict(avg_sd)


for i in range(n_steps_ft):
    lr = get_lr(i, n_steps_ft)
    for pg in optimizer.param_groups:
        pg["lr"] = lr

    # ── train step ──
    model.train()
    train_logits = model(train_inputs)
    loss = F.cross_entropy(
        train_logits.reshape(-1, n_digit),
        train_targets.reshape(-1),
    )
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad()

    # ── LAWA: save snapshot every lawa_interval steps ──
    if i % lawa_interval == 0:
        lawa_buffer.append({k: v.clone() for k, v in model.state_dict().items()})
        update_lawa()

    # ── logging & eval ──
    if i % 100 == 0:
        model.eval()
        lawa_model.eval()
        with torch.no_grad():
            test_logits = model(test_inputs)
            test_acc = accuracy(test_logits, test_targets)
            test_acc_seq = accuracy_seq(test_logits, test_targets)

            train_acc = accuracy(train_logits.detach(), train_targets)
            train_acc_seq = accuracy_seq(train_logits.detach(), train_targets)

            lawa_logits = lawa_model(test_inputs)
            lawa_acc = accuracy(lawa_logits, test_targets)
            lawa_acc_seq = accuracy_seq(lawa_logits, test_targets)

        if test_acc_seq.item() > best_test_acc_seq:
            best_test_acc_seq = test_acc_seq.item()
            torch.save({"step": i, "model": model.state_dict(),
                        "test_acc_seq": best_test_acc_seq, "seed": seed}, ckpt_path_ft)

        if lawa_acc_seq.item() > best_lawa_acc_seq:
            best_lawa_acc_seq = lawa_acc_seq.item()

        log(
            f"[FT] Step {i:5d}  lr {lr:.2e}  loss {loss.item():.4f}  "
            f"train acc {train_acc:.4f}  test acc {test_acc:.4f}  "
            f"train seq {train_acc_seq:.4f}  test seq {test_acc_seq:.4f}  "
            f"best seq {best_test_acc_seq:.4f}  "
            f"lawa acc {lawa_acc:.4f}  lawa seq {lawa_acc_seq:.4f}  "
            f"best lawa seq {best_lawa_acc_seq:.4f} (k={len(lawa_buffer)})"
        )

# ── Final summary ─────────────────────────────────────────────────────────────

summary = (
    f"\n{'='*70}\n"
    f"Fine-tuning complete\n"
    f"  seed          : {seed}\n"
    f"  n_params      : {n_params}\n"
    f"  finetune steps: {n_steps_ft}\n"
    f"  max_lr        : {max_lr}\n"
    f"  warmup_steps  : {warmup_steps}\n"
    f"  lawa_k        : {lawa_k}\n"
    f"  lawa_interval : {lawa_interval}\n"
    f"  best test seq acc     : {best_test_acc_seq:.4f}\n"
    f"  best lawa seq acc     : {best_lawa_acc_seq:.4f}\n"
    f"{'='*70}"
)
log(summary)

log_file.close()