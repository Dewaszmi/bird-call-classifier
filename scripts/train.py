#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from datasets.batching import (
    BATCHING_STRATEGIES,
    build_dataloader,
    uses_lengths,
    uses_masked_pooling,
)
from datasets.spectrogram import (
    DEFAULT_FIXED_DURATION_SEC,
    SpectrogramDataset,
    discover_samples,
    duration_to_frames,
    train_val_test_split,
)
from models.vgg import BirdVGG

DATA_DIR = ROOT / "processed_data"
OUTPUT_DIR = ROOT / "checkpoints"
LOG_DIR = ROOT / "runs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BirdVGG on mel spectrograms.")
    parser.add_argument(
        "--batching-strategy",
        choices=BATCHING_STRATEGIES,
        default="none",
        help=(
            "Batching strategy: "
            "'none' = fixed 128x128 resize + regular GAP (benchmark); "
            "'no-batch' = batch size 1, no padding, gradient accumulation; "
            "'length-bucketing' = batch similar lengths + regular GAP; "
            "'masked-gap' = pad to fixed duration, batch together, masked GAP ignores padded silence"
        ),
    )
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--accum-steps",
        type=int,
        default=32,
        help="Gradient accumulation steps for --batching-strategy no-batch (default: 32)",
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument(
        "--lr-end-factor",
        type=float,
        default=0.1,
        help="Final LR as a fraction of the initial LR (linear decay, default: 0.1)",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default=(
            "mps"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            else ("cuda" if torch.cuda.is_available() else "cpu")
        ),
        help="Training device",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="TensorBoard run name (default: timestamp)",
    )
    parser.add_argument(
        "--fixed-duration",
        type=float,
        default=DEFAULT_FIXED_DURATION_SEC,
        help=(
            "Pad/truncate to this duration for --batching-strategy none benchmark "
            f"(default: {DEFAULT_FIXED_DURATION_SEC})"
        ),
    )
    parser.add_argument(
        "--no-tensorboard",
        action="store_true",
        help="Disable TensorBoard logging",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pooling_mode_for_strategy(batching_strategy: str) -> str:
    return "masked-gap" if uses_masked_pooling(batching_strategy) else "gap"


def macro_f1(preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> float:
    f1_scores: list[float] = []
    for class_idx in range(num_classes):
        pred_pos = preds == class_idx
        true_pos = targets == class_idx
        tp = (pred_pos & true_pos).sum().item()
        fp = (pred_pos & ~true_pos).sum().item()
        fn = (~pred_pos & true_pos).sum().item()

        if tp == 0 and fp == 0 and fn == 0:
            continue

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision + recall == 0:
            f1_scores.append(0.0)
        else:
            f1_scores.append(2 * precision * recall / (precision + recall))

    return sum(f1_scores) / len(f1_scores) if f1_scores else 0.0


def run_forward(
    model: BirdVGG,
    inputs: torch.Tensor,
    lengths: torch.Tensor | None,
) -> torch.Tensor:
    if model.pooling_mode == "masked-gap":
        if lengths is None:
            raise ValueError("lengths are required for masked-gap pooling")
        return model(inputs, lengths)
    return model(inputs)


def iter_batches(
    loader: DataLoader,
    batching_strategy: str,
):
    for batch in loader:
        if uses_lengths(batching_strategy):
            inputs, targets, lengths = batch
            yield inputs, targets, lengths
        else:
            inputs, targets = batch
            yield inputs, targets, None


@torch.no_grad()
def evaluate(
    model: BirdVGG,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    batching_strategy: str,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_preds: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []

    for inputs, targets, lengths in iter_batches(loader, batching_strategy):
        inputs = inputs.to(device)
        targets = targets.to(device)
        if lengths is not None:
            lengths = lengths.to(device)

        logits = run_forward(model, inputs, lengths)
        loss = criterion(logits, targets)
        preds = logits.argmax(dim=1)

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (preds == targets).sum().item()
        total_samples += batch_size
        all_preds.append(preds.cpu())
        all_targets.append(targets.cpu())

    preds_tensor = torch.cat(all_preds)
    targets_tensor = torch.cat(all_targets)

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
        "macro_f1": macro_f1(preds_tensor, targets_tensor, num_classes),
    }


def train_one_epoch(
    model: BirdVGG,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    batching_strategy: str,
    accum_steps: int = 1,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0

    optimizer.zero_grad()
    for step, (inputs, targets, lengths) in enumerate(
        iter_batches(loader, batching_strategy), start=1
    ):
        inputs = inputs.to(device)
        targets = targets.to(device)
        if lengths is not None:
            lengths = lengths.to(device)

        logits = run_forward(model, inputs, lengths)
        loss = criterion(logits, targets) / accum_steps
        loss.backward()

        if step % accum_steps == 0 or step == len(loader):
            optimizer.step()
            optimizer.zero_grad()

        batch_size = targets.size(0)
        total_loss += loss.item() * accum_steps * batch_size
        total_samples += batch_size

    return total_loss / total_samples


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    if args.batching_strategy == "no-batch" and args.batch_size != 1:
        print(
            f"Note: --batching-strategy no-batch forces batch size 1 "
            f"(effective batch size={args.accum_steps})."
        )

    samples, classes = discover_samples(DATA_DIR)
    train_samples, val_samples, test_samples = train_val_test_split(
        samples, val_ratio=args.val_ratio, test_ratio=args.test_ratio, seed=args.seed
    )

    uses_fixed_duration = args.batching_strategy in {"none", "masked-gap"}
    resize_to = (128, 128) if args.batching_strategy == "none" else None
    fixed_time_frames = (
        duration_to_frames(args.fixed_duration) if uses_fixed_duration else None
    )
    return_valid_lengths = args.batching_strategy == "masked-gap"

    train_dataset = SpectrogramDataset(
        train_samples,
        classes,
        resize_to=resize_to,
        fixed_time_frames=fixed_time_frames,
        return_valid_lengths=return_valid_lengths,
    )
    val_dataset = SpectrogramDataset(
        val_samples,
        classes,
        resize_to=resize_to,
        fixed_time_frames=fixed_time_frames,
        return_valid_lengths=return_valid_lengths,
    )
    test_dataset = SpectrogramDataset(
        test_samples,
        classes,
        resize_to=resize_to,
        fixed_time_frames=fixed_time_frames,
        return_valid_lengths=return_valid_lengths,
    )

    train_loader = build_dataloader(
        train_dataset,
        batching_strategy=args.batching_strategy,
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed,
    )
    val_loader = build_dataloader(
        val_dataset,
        batching_strategy=args.batching_strategy,
        batch_size=args.batch_size,
        shuffle=False,
        seed=args.seed,
    )
    test_loader = build_dataloader(
        test_dataset,
        batching_strategy=args.batching_strategy,
        batch_size=args.batch_size,
        shuffle=False,
        seed=args.seed,
    )

    pooling_mode = pooling_mode_for_strategy(args.batching_strategy)
    model = BirdVGG(num_classes=len(classes), pooling_mode=pooling_mode).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler: torch.optim.lr_scheduler.LinearLR | None = None
    if args.epochs > 1:
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=args.lr_end_factor,
            total_iters=args.epochs - 1,
        )

    accum_steps = args.accum_steps if args.batching_strategy == "no-batch" else 1
    if args.batching_strategy == "no-batch":
        print(
            f"Using gradient accumulation: {accum_steps} steps (effective batch size={accum_steps})"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    best_val_f1 = -1.0
    history: list[dict[str, float]] = []

    writer: SummaryWriter | None = None
    log_path: Path | None = None
    if not args.no_tensorboard:
        run_name = args.run_name or datetime.now().strftime("%Y%m%d-%H%M%S")
        log_path = LOG_DIR / run_name
        log_path.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=log_path)

    print(f"Device: {device}")
    print(f"Batching strategy: {args.batching_strategy} | Pooling: {pooling_mode}")
    print(
        f"Classes: {len(classes)} | Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}"
    )
    if writer is not None and log_path is not None:
        print(f"TensorBoard logs: {log_path}")

    for epoch in range(1, args.epochs + 1):
        if args.batching_strategy == "length-bucketing":
            train_loader.batch_sampler.set_epoch(epoch)  # type: ignore[attr-defined]

        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            args.batching_strategy,
            accum_steps=accum_steps,
        )
        val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            device,
            len(classes),
            args.batching_strategy,
        )

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
        }
        history.append(record)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_macro_f1={val_metrics['macro_f1']:.4f}"
        )

        if writer is not None:
            current_lr = optimizer.param_groups[0]["lr"]
            writer.add_scalar("loss/train", train_loss, epoch)
            writer.add_scalar("loss/val", val_metrics["loss"], epoch)
            writer.add_scalar("accuracy/val", val_metrics["accuracy"], epoch)
            writer.add_scalar("macro_f1/val", val_metrics["macro_f1"], epoch)
            writer.add_scalar("lr", current_lr, epoch)

        if scheduler is not None and epoch < args.epochs:
            scheduler.step()

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "classes": classes,
                "epoch": epoch,
                "val_macro_f1": best_val_f1,
                "val_accuracy": val_metrics["accuracy"],
                "batching_strategy": args.batching_strategy,
                "pooling_mode": pooling_mode,
                "fixed_duration": args.fixed_duration,
            }
            torch.save(checkpoint, OUTPUT_DIR / "best.pt")

    with (OUTPUT_DIR / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best val macro F1: {best_val_f1:.4f}")
    print(f"Checkpoint saved to {OUTPUT_DIR / 'best.pt'}")

    print("\nEvaluating best model on test set...")
    model.load_state_dict(
        torch.load(OUTPUT_DIR / "best.pt", weights_only=True)["model_state_dict"]
    )
    test_metrics = evaluate(
        model,
        test_loader,
        criterion,
        device,
        len(classes),
        args.batching_strategy,
    )
    print(
        f"Test Loss: {test_metrics['loss']:.4f} | "
        f"Test Accuracy: {test_metrics['accuracy']:.4f} | "
        f"Test Macro F1: {test_metrics['macro_f1']:.4f}"
    )

    if writer is not None:
        writer.add_hparams(
            {
                "lr": args.lr,
                "lr_end_factor": args.lr_end_factor,
                "batch_size": args.batch_size,
                "batching_strategy": args.batching_strategy,
                "accum_steps": accum_steps,
                "epochs": args.epochs,
                "weight_decay": args.weight_decay,
                "val_ratio": args.val_ratio,
                "test_ratio": args.test_ratio,
                "seed": args.seed,
                "num_classes": len(classes),
            },
            {
                "hparam/best_val_macro_f1": best_val_f1,
            },
        )
        writer.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
