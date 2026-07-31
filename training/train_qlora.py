#!/usr/bin/env python3
"""QLoRA fine-tune of Llama-3.1-Nemotron-Nano-8B-v1 on the generated finance corpus.

Runs **inside** `westpac-qlora` (see training/Dockerfile.qlora). The host python has no
torch, and a stock wheel would not carry sm_121 kernels — "no kernel image is available
for execution" is the symptom of running this outside the container.

    # smoke test: 10 steps, reports measured step time, writes no final adapter
    docker run --rm --gpus all -v $PWD:/workspace -w /workspace \
      -v $HOME/models:/models westpac-qlora:latest \
      'python3 training/train_qlora.py --smoke'

    # full run
    docker run --rm --gpus all -v $PWD:/workspace -w /workspace \
      -v $HOME/models:/models westpac-qlora:latest \
      'python3 training/train_qlora.py'

Two invariants this script refuses to run without:

* **The prompt contract matches the dataset.** ``prepare_data.py`` records a SHA-256 of
  the prompt bytes and both rendered message shapes in ``stats.json``. If ``src/prompts.py``
  has changed since generation, the adapter would be trained on one format and served
  another — the one failure mode that wastes the entire run. We fail loudly instead.
* **Loss is computed on assistant tokens only.** Without masking, the model learns to
  generate tool results as well as answers, which is the opposite of the goal. The
  Nemotron chat template has no ``{% generation %}`` marker, so TRL's
  ``assistant_only_loss`` cannot be used; masking is done explicitly here and asserted.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

IGNORE = -100

# The 291-shard weight-loading bar writes thousands of lines into a captured log for no
# benefit; step logging is what we actually want to read.
from transformers.utils import logging as hf_logging  # noqa: E402

hf_logging.disable_progress_bar()


# ======================================================================================
# dataset
# ======================================================================================


@dataclass
class Row:
    input_ids: list[int]
    labels: list[int]


class MaskedChatDataset(Dataset):
    """Pre-tokenised chat rows with the prompt masked out of the loss.

    The prompt is rendered with ``tokenize=False`` and tokenised as a string, because
    ``apply_chat_template(tokenize=True)`` returns a ``BatchEncoding`` in transformers 5.x
    whose ``len()`` is the number of fields, not the number of tokens — a silent way to
    mis-measure every sequence.
    """

    def __init__(self, path: Path, tok, max_seq_len: int) -> None:
        self.rows: list[Row] = []
        self.n_truncated = 0
        self.n_answer_truncated = 0

        for line in path.open(encoding="utf-8"):
            rec = json.loads(line)
            msgs = rec["messages"]
            prompt_text = tok.apply_chat_template(
                msgs[:-1], add_generation_prompt=True, tokenize=False
            )
            prompt_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
            answer_ids = tok(
                msgs[-1]["content"], add_special_tokens=False
            )["input_ids"]
            eot = tok.convert_tokens_to_ids("<|eot_id|>")
            answer_ids = answer_ids + [eot]

            ids = prompt_ids + answer_ids
            if len(ids) > max_seq_len:
                self.n_truncated += 1
                # Truncate from the PROMPT side so the answer survives intact. A
                # truncated answer teaches the model to emit nothing; a slightly
                # truncated evidence block is merely harder.
                keep = max_seq_len - len(answer_ids)
                if keep <= 0:
                    self.n_answer_truncated += 1
                    continue
                prompt_ids = prompt_ids[:keep]
                ids = prompt_ids + answer_ids

            labels = [IGNORE] * len(prompt_ids) + list(answer_ids)
            assert len(labels) == len(ids)
            self.rows.append(Row(ids, labels))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        r = self.rows[i]
        # group_by_length reads this key to bucket similar lengths together.
        return {"input_ids": r.input_ids, "labels": r.labels, "length": len(r.input_ids)}


class PadCollator:
    """Pads to the longest sequence **in the batch**, not to max_seq_len.

    This is the single largest throughput lever in the run: measured p50 is 370 tokens
    against an 832 cap, so padding to the cap would spend well over half of all FLOPs on
    padding. Combined with ``group_by_length`` it removes most of that waste.
    """

    def __init__(self, pad_id: int) -> None:
        self.pad_id = pad_id

    def __call__(self, features: list[dict]) -> dict:
        width = max(len(f["input_ids"]) for f in features)
        input_ids, labels, attn = [], [], []
        for f in features:
            n = len(f["input_ids"])
            pad = width - n
            input_ids.append(f["input_ids"] + [self.pad_id] * pad)
            labels.append(f["labels"] + [IGNORE] * pad)
            attn.append([1] * n + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }


# ======================================================================================
# guards
# ======================================================================================


def verify_contract(data_dir: Path) -> None:
    """Refuse to train if src/prompts.py changed after the dataset was generated."""
    stats_path = data_dir / "stats.json"
    if not stats_path.exists():
        raise SystemExit(f"missing {stats_path} — run training/prepare_data.py first")
    recorded = json.loads(stats_path.read_text()).get("contract_hash")

    sys.path.insert(0, str(REPO_ROOT / "training"))
    from prepare_data import contract_hash  # noqa: E402  (needs the path above)

    live = contract_hash()
    if recorded != live:
        raise SystemExit(
            "PROMPT CONTRACT DRIFT — refusing to train.\n"
            f"  dataset was generated against {recorded}\n"
            f"  src/prompts.py now hashes to  {live}\n"
            "The adapter would be trained on one prompt format and served another, which "
            "degrades it to noise. Re-run training/prepare_data.py, then retrain."
        )
    print(f"  contract hash OK: {live[:16]}…")


def assert_masking(ds: MaskedChatDataset, tok) -> None:
    """The loss must see the answer and nothing before it."""
    r = ds.rows[0]
    supervised = [i for i, l in enumerate(r.labels) if l != IGNORE]
    assert supervised, "no supervised tokens — masking is broken"
    first = supervised[0]
    assert all(l == IGNORE for l in r.labels[:first]), "prompt tokens leaked into the loss"
    decoded = tok.decode([r.input_ids[i] for i in supervised])
    print(f"  loss mask: {len(supervised)}/{len(r.labels)} tokens supervised")
    print(f"  supervised text begins: {decoded[:100]!r}")


# ======================================================================================
# callbacks
# ======================================================================================


class LengthGroupedTrainer(Trainer):
    """Trainer that batches similar-length rows together.

    transformers 5.x removed the ``group_by_length`` TrainingArguments flag but kept
    ``LengthGroupedSampler``, so the behaviour is restored here explicitly. This matters:
    with the dynamic-padding collator, grouping is what converts a p50 of 370 tokens into
    actual saved compute instead of one 778-token row inflating an entire batch.
    """

    def _get_train_sampler(self, *a, **kw):
        from transformers.trainer_pt_utils import LengthGroupedSampler

        return LengthGroupedSampler(
            self.args.train_batch_size,
            lengths=[len(r.input_ids) for r in self.train_dataset.rows],
        )


class StepTimer(TrainerCallback):
    """Measures the real step rate so max_steps is set from data, not from the handout's
    ~90 s/step guess."""

    def __init__(self) -> None:
        self.t0: float | None = None
        self.first_done = False
        self.times: list[float] = []

    def on_step_begin(self, args, state, control, **kw):
        self.t0 = time.perf_counter()

    def on_step_end(self, args, state, control, **kw):
        if self.t0 is None:
            return
        dt = time.perf_counter() - self.t0
        # The first step includes kernel autotune / allocator warmup; excluded.
        if not self.first_done:
            self.first_done = True
            print(f"  step 1 (incl. warmup): {dt:.2f}s")
            return
        self.times.append(dt)

    def report(self, total_steps: int) -> None:
        if not self.times:
            return
        s = sorted(self.times)
        med = s[len(s) // 2]
        print(f"\n  measured step time: median {med:.2f}s over {len(s)} steps")
        print(f"  => {total_steps} steps ≈ {med * total_steps / 60:.1f} min")


# ======================================================================================
# main
# ======================================================================================


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="training/config.yaml")
    ap.add_argument("--smoke", action="store_true",
                    help="10 steps, measure step time, do not save a final adapter")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="override; set from the smoke-test measurement")
    ap.add_argument("--gpu-fraction", type=float,
                    default=float(os.environ.get("GPU_MEM_FRACTION", "0") or 0),
                    help="cap this process at a fraction of total GPU memory; 0 = uncapped")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="override per_device_train_batch_size (memory pressure relief)")
    ap.add_argument("--grad-accum", type=int, default=None,
                    help="override gradient_accumulation_steps; pair with --batch-size to "
                         "hold the effective batch (and therefore the tuned LR) constant")
    ap.add_argument("--grad-checkpointing", dest="grad_ckpt",
                    action=argparse.BooleanOptionalAction, default=None,
                    help="override gradient_checkpointing. The config disables it to buy "
                         "throughput, which assumes the whole GB10 is available; under a "
                         "--gpu-fraction cap the ~36 GiB of stored activations is what "
                         "OOMs, so it must go back on.")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    tcfg = cfg["training"]
    data_dir = Path(cfg["data_dir"])

    if args.batch_size is not None:
        tcfg["per_device_train_batch_size"] = args.batch_size
    if args.grad_accum is not None:
        tcfg["gradient_accumulation_steps"] = args.grad_accum
    if args.grad_ckpt is not None:
        tcfg["gradient_checkpointing"] = args.grad_ckpt

    # The GB10's memory is unified: vllm-brain's ~50 GiB and the desktop session are
    # carved out of the same 121 GiB the trainer allocates from. Cap ourselves so an
    # allocator spike cannot evict the serving container another workstream depends on.
    if args.gpu_fraction > 0:
        torch.cuda.set_per_process_memory_fraction(args.gpu_fraction, 0)
        total = torch.cuda.mem_get_info()[1] / 2**30
        free = torch.cuda.mem_get_info()[0] / 2**30
        print(f"== gpu budget ==\n  cap {args.gpu_fraction:.0%} of {total:.1f} GiB "
              f"= {args.gpu_fraction * total:.1f} GiB | actually free now {free:.1f} GiB")

    print("== guards ==")
    verify_contract(data_dir)

    print("\n== tokenizer ==")
    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    print("\n== data ==")
    train_ds = MaskedChatDataset(data_dir / "train.jsonl", tok, cfg["max_seq_len"])
    val_ds = MaskedChatDataset(data_dir / "val.jsonl", tok, cfg["max_seq_len"])
    print(f"  train {len(train_ds)} rows | val {len(val_ds)} rows")
    print(f"  prompt-side truncations: {train_ds.n_truncated} "
          f"(answers dropped: {train_ds.n_answer_truncated})")
    assert_masking(train_ds, tok)

    print("\n== model (4-bit NF4) ==")
    q = cfg["quantization"]
    bnb = BitsAndBytesConfig(
        load_in_4bit=q["load_in_4bit"],
        bnb_4bit_quant_type=q["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=q["bnb_4bit_use_double_quant"],
        bnb_4bit_compute_dtype=getattr(torch, q["bnb_4bit_compute_dtype"]),
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"],
        quantization_config=bnb,
        dtype=torch.bfloat16,
        attn_implementation=tcfg["attn_implementation"],
        device_map={"": 0},
    )
    model.config.use_cache = False

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=tcfg["gradient_checkpointing"]
    )
    lcfg = cfg["lora"]
    model = get_peft_model(
        model,
        LoraConfig(
            r=lcfg["r"],
            lora_alpha=lcfg["alpha"],
            lora_dropout=lcfg["dropout"],
            target_modules=lcfg["target_modules"],
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    model.print_trainable_parameters()

    max_steps = args.max_steps if args.max_steps is not None else tcfg["max_steps"]
    if args.smoke:
        max_steps = 10

    print("\n== train ==")
    out_dir = Path(cfg["output_dir"]) / ("smoke" if args.smoke else "run")
    targs = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=tcfg["per_device_train_batch_size"],
        gradient_accumulation_steps=tcfg["gradient_accumulation_steps"],
        num_train_epochs=tcfg["num_train_epochs"],
        max_steps=max_steps,
        learning_rate=float(tcfg["learning_rate"]),
        lr_scheduler_type=tcfg["lr_scheduler_type"],
        warmup_ratio=tcfg["warmup_ratio"],
        weight_decay=tcfg["weight_decay"],
        max_grad_norm=tcfg["max_grad_norm"],
        logging_steps=tcfg["logging_steps"],
        save_steps=tcfg["save_steps"] if not args.smoke else 10_000,
        save_total_limit=tcfg["save_total_limit"],
        bf16=tcfg["bf16"],
        optim=tcfg["optim"],
        gradient_checkpointing=tcfg["gradient_checkpointing"],
        length_column_name="length",
        dataloader_num_workers=tcfg["dataloader_num_workers"],
        remove_unused_columns=False,
        report_to=[],
        seed=tcfg["seed"],
    )

    timer = StepTimer()
    trainer = LengthGroupedTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        data_collator=PadCollator(tok.pad_token_id),
        callbacks=[timer],
    )

    t0 = time.perf_counter()
    result = trainer.train()
    wall = time.perf_counter() - t0

    planned = max_steps if max_steps and max_steps > 0 else int(
        len(train_ds) / tcfg["per_device_train_batch_size"] * tcfg["num_train_epochs"]
    )
    timer.report(planned)
    print(f"\n  wall clock: {wall / 60:.1f} min for {result.global_step} steps")
    print(f"  final train loss: {result.training_loss:.4f}")
    print(f"  peak GPU alloc: {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")

    if args.smoke:
        print("\n  smoke test complete — no final adapter written by design")
        return

    final = Path(cfg["output_dir"]) / "final"
    trainer.model.save_pretrained(str(final))
    tok.save_pretrained(str(final))
    print(f"\n  adapter written to {final}")

    (Path(cfg["output_dir"]) / "train_summary.json").write_text(json.dumps({
        "steps": result.global_step,
        "train_loss": result.training_loss,
        "wall_minutes": round(wall / 60, 2),
        "median_step_s": round(sorted(timer.times)[len(timer.times) // 2], 3)
                          if timer.times else None,
        "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "config": cfg,
    }, indent=2))


if __name__ == "__main__":
    main()
