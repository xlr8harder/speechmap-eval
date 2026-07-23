#!/usr/bin/env python3
"""Convert Qwen3.5 adapter keys between Unsloth and local Transformers layouts."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from safetensors.torch import load_file, save_file


UNSLOTH_PREFIX = "base_model.model.model.language_model.layers."
PROJECT_PREFIX = "base_model.model.model.layers."


def convert_adapter_keys(source_dir: Path, target_dir: Path, *, reverse: bool = False) -> dict[str, int]:
    source_dir = source_dir.resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    source_weights = source_dir / "adapter_model.safetensors"
    if not source_weights.exists():
        raise FileNotFoundError(f"missing {source_weights}")

    source_prefix = PROJECT_PREFIX if reverse else UNSLOTH_PREFIX
    target_prefix = UNSLOTH_PREFIX if reverse else PROJECT_PREFIX
    tensors = load_file(str(source_weights))
    converted = {}
    renamed = 0
    unchanged = 0
    for key, tensor in tensors.items():
        if key.startswith(source_prefix):
            new_key = target_prefix + key[len(source_prefix) :]
            renamed += 1
        else:
            new_key = key
            unchanged += 1
        converted[new_key] = tensor
    save_file(converted, str(target_dir / "adapter_model.safetensors"))

    for path in source_dir.iterdir():
        if path.name == "adapter_model.safetensors" or path.is_dir():
            continue
        shutil.copy2(path, target_dir / path.name)

    return {"renamed": renamed, "unchanged": unchanged, "total": len(tensors)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("target_dir", type=Path)
    parser.add_argument(
        "--reverse",
        action="store_true",
        help="Convert local Transformers keys back to the Unsloth Qwen3.5 layout.",
    )
    args = parser.parse_args()
    stats = convert_adapter_keys(args.source_dir, args.target_dir, reverse=args.reverse)
    print(stats)


if __name__ == "__main__":
    main()
