"""Educational reproduction of the Transformer on a deterministic copy task.

This is a small, runnable reproduction of the mechanism, not a WMT14 BLEU
reproduction of the original paper. It uses PyTorch's Transformer module and
keeps the target-side causal mask explicit for inspection.
"""

from __future__ import annotations

import argparse
import json
import random
import time

import torch
from torch import Tensor, nn


PAD = 0
BOS = 1
EOS = 2
FIRST_TOKEN = 3


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def causal_mask(size: int, device: torch.device) -> Tensor:
    """Return an upper-triangular boolean mask blocking future positions."""
    return torch.triu(torch.ones(size, size, dtype=torch.bool, device=device), diagonal=1)


def make_batch(batch_size: int, length: int, vocab_size: int, device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
    src = torch.randint(FIRST_TOKEN, vocab_size, (batch_size, length), device=device)
    target = torch.cat(
        [
            torch.full((batch_size, 1), BOS, dtype=torch.long, device=device),
            src,
            torch.full((batch_size, 1), EOS, dtype=torch.long, device=device),
        ],
        dim=1,
    )
    return src, target[:, :-1], target[:, 1:]


class CopyTransformer(nn.Module):
    def __init__(self, vocab_size: int, max_length: int, d_model: int, heads: int, layers: int) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_length, d_model)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=heads,
            num_encoder_layers=layers,
            num_decoder_layers=layers,
            dim_feedforward=d_model * 2,
            dropout=0.0,
            batch_first=True,
            norm_first=False,
        )
        self.output = nn.Linear(d_model, vocab_size)
        self.scale = d_model**0.5

    def embed(self, tokens: Tensor) -> Tensor:
        positions = torch.arange(tokens.size(1), device=tokens.device).unsqueeze(0)
        return self.token_embedding(tokens) * self.scale + self.position_embedding(positions)

    def forward(self, src: Tensor, target_input: Tensor) -> Tensor:
        return self.output(
            self.transformer(
                self.embed(src),
                self.embed(target_input),
                tgt_mask=causal_mask(target_input.size(1), target_input.device),
            )
        )


@torch.no_grad()
def greedy_decode(model: CopyTransformer, src: Tensor, max_steps: int) -> Tensor:
    generated = torch.full((src.size(0), 1), BOS, dtype=torch.long, device=src.device)
    for _ in range(max_steps):
        logits = model(src, generated)
        next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)
        if bool((next_token == EOS).all()):
            break
    return generated[:, 1:]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a small Transformer copy-task reproduction.")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求使用 CUDA，但当前虚拟环境未检测到可用 GPU。")
    device = torch.device(args.device)
    set_seed(args.seed)

    vocab_size = 32
    source_length = 8
    batch_size = 128
    model = CopyTransformer(vocab_size, source_length + 2, d_model=64, heads=4, layers=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()

    started = time.perf_counter()
    model.train()
    losses: list[float] = []
    for step in range(1, args.steps + 1):
        src, target_input, target_output = make_batch(batch_size, source_length, vocab_size, device)
        logits = model(src, target_input)
        loss = criterion(logits.reshape(-1, vocab_size), target_output.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    model.eval()
    eval_src, _, eval_target = make_batch(32, source_length, vocab_size, device)
    predictions = greedy_decode(model, eval_src, max_steps=source_length + 1)
    width = min(predictions.size(1), eval_target.size(1))
    token_accuracy = float((predictions[:, :width] == eval_target[:, :width]).float().mean().cpu())
    sequence_accuracy = float((predictions[:, :width] == eval_target[:, :width]).all(dim=1).float().mean().cpu())

    result = {
        "device": str(device),
        "torch_version": torch.__version__,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "seed": args.seed,
        "steps": args.steps,
        "model": {"d_model": 64, "heads": 4, "layers": 2, "parameters": sum(p.numel() for p in model.parameters())},
        "initial_loss": round(losses[0], 5),
        "final_loss": round(losses[-1], 5),
        "token_accuracy": round(token_accuracy, 4),
        "sequence_accuracy": round(sequence_accuracy, 4),
        "seconds": round(time.perf_counter() - started, 3),
        "example": {
            "source": eval_src[0].cpu().tolist(),
            "target": eval_target[0].cpu().tolist(),
            "prediction": predictions[0].cpu().tolist(),
        },
        "task": "copy source tokens and append EOS; educational reproduction, not WMT14 BLEU.",
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
