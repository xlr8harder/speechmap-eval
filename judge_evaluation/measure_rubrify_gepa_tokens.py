#!/usr/bin/env python3
"""Measure Rubrify judge vs GEPA reflection token use on DPO-dev rows."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "judge_evaluation/rubrify/dpo_dev_20260607.jsonl"
DEFAULT_OUTPUT = ROOT / "judge_evaluation/results/rubrify_speechmap_dpo_dev/token_probe_qwen35_0p8b/summary.json"
DEFAULT_TOKENIZER = "Qwen/Qwen3.5-0.8B"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from judge_evaluation.run_rubrify_speechmap import build_speechmap_rubric, compile_bundle  # noqa: E402


def read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    break
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def stats(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    return {
        "count": len(values),
        "min": ordered[0],
        "p50": int(statistics.median(ordered)),
        "mean": statistics.mean(ordered),
        "p90": ordered[int(0.9 * (len(ordered) - 1))],
        "max": ordered[-1],
        "sum": sum(ordered),
    }


def count_tokens(tokenizer: Any, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def criterion_scores_for(row: dict[str, Any]) -> dict[str, bool]:
    label = row["expected_label"]
    return {
        "is_complete": label == "COMPLETE",
        "is_denial": label == "DENIAL",
        "is_evasive": label == "EVASIVE",
    }


def build_judge_prompts(rows: list[dict[str, Any]], *, use_tool: bool) -> list[dict[str, Any]]:
    from rubrify.engine.executor import _build_group_system_prompt, _build_group_user_prompt

    bundle = compile_bundle()
    criteria = bundle.rubric.criteria
    prompts = []
    for row in rows:
        system_prompt = _build_group_system_prompt(criteria, bundle)
        user_prompt = _build_group_user_prompt(criteria, row["rubrify_response_text"], use_tool)
        prompts.append(
            {
                "id": row["id"],
                "expected_label": row["expected_label"],
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "combined_prompt": f"System:\n{system_prompt}\n\nUser:\n{user_prompt}",
            }
        )
    return prompts


def build_reflection_prompts(rows: list[dict[str, Any]], components: list[str]) -> list[dict[str, Any]]:
    from harn_ai.providers.faux import faux_assistant_message, faux_tool_call, register_faux_provider
    from rubrify.evolve.adapter import RubricEvolverAdapter
    from rubrify.evolve.candidate import rubric_to_candidate
    from rubrify.evolve.evolver import RUBRIC_EVOLUTION_REFLECTION_TEMPLATE
    from rubrify.evolve.types import AnnotatedExample

    rubric, role, policy = build_speechmap_rubric()
    candidate = rubric_to_candidate(rubric, role)
    examples = [
        AnnotatedExample(
            id=row["id"],
            response_text=row["rubrify_response_text"],
            context_text=row.get("rubrify_context_text"),
            human_scores=criterion_scores_for(row),
            human_label=row["expected_label"],
        )
        for row in rows
    ]

    registration = register_faux_provider(
        {
            "provider": "faux",
            "models": [
                {
                    "id": "speechmap-faux",
                    "name": "SpeechMap Faux Rubrify Token Probe",
                    "reasoning": False,
                    "input": ["text"],
                    "contextWindow": 128000,
                    "maxTokens": 16384,
                    "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
                }
            ],
        }
    )
    try:
        registration.set_responses(
            [
                faux_assistant_message(
                    faux_tool_call(
                        "submit_judgment",
                        {
                            "score": 1,
                            "rationale": "faux token probe judgment",
                            "evidence": [],
                            "violations": [],
                            "criterion_scores": criterion_scores_for(row),
                            "confidence": 1.0,
                        },
                    )
                )
                for row in rows
            ]
        )
        adapter = RubricEvolverAdapter(
            base_rubric=rubric,
            judge_model=registration.get_model("speechmap-faux"),
            base_role=role,
            base_policy=policy,
            judge_api_key="local-faux",
            judge_temperature=0.0,
            judge_max_tokens=1024,
            agreement_weight=1.0,
            consistency_weight=0.0,
            discrimination_weight=0.0,
        )
        eval_batch = adapter.evaluate(examples, candidate, capture_traces=True)
        reflective = adapter.make_reflective_dataset(candidate, eval_batch, components)
    finally:
        registration.unregister()

    prompts = []
    for component in components:
        side_info = json.dumps(list(reflective[component]), indent=2, default=str)
        prompt = (
            RUBRIC_EVOLUTION_REFLECTION_TEMPLATE.replace("<curr_param>", candidate.get(component, ""))
            .replace("<side_info>", side_info)
        )
        prompts.append(
            {
                "component": component,
                "minibatch_examples": len(rows),
                "prompt": prompt,
            }
        )
    return prompts


def maybe_generate_samples(args: argparse.Namespace, tokenizer: Any, judge_prompts: list[dict[str, Any]], reflection_prompts: list[dict[str, Any]]) -> dict[str, Any]:
    if args.generate_samples <= 0:
        return {"enabled": False}

    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else "auto",
    )
    model.eval()

    sample_specs: list[tuple[str, str, str]] = []
    for item in judge_prompts[: args.generate_samples]:
        sample_specs.append(("judge", item["id"], item["combined_prompt"]))
    for item in reflection_prompts[: args.generate_samples]:
        sample_specs.append(("reflection", item["component"], item["prompt"]))

    samples = []
    for kind, name, prompt in sample_specs:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_input_tokens)
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
        input_tokens = int(inputs["input_ids"].shape[-1])
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        output_tokens = int(generated.shape[-1] - input_tokens)
        text = tokenizer.decode(generated[0, input_tokens:], skip_special_tokens=True)
        samples.append(
            {
                "kind": kind,
                "name": name,
                "input_tokens_after_truncation": input_tokens,
                "output_tokens": output_tokens,
                "output_preview": text[:500],
            }
        )
    return {"enabled": True, "model": args.model, "samples": samples}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--model", default=DEFAULT_TOKENIZER)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--reflection-minibatch-size", type=int, default=3)
    parser.add_argument("--components", nargs="*", default=[
        "rubric.goal",
        "criterion.is_complete.description",
        "criterion.is_denial.description",
        "criterion.is_evasive.description",
        "rubric.instructions",
        "rubric.definitions",
        "rubric.calibration_examples",
        "role.obligations",
        "role.constraints",
    ])
    parser.add_argument("--generate-samples", type=int, default=0)
    parser.add_argument("--max-input-tokens", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args(argv)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    rows = read_jsonl(args.data, args.limit)
    minibatch_rows = rows[: args.reflection_minibatch_size]

    judge_text = build_judge_prompts(rows, use_tool=False)
    judge_tool = build_judge_prompts(rows, use_tool=True)
    reflection_prompts = build_reflection_prompts(minibatch_rows, args.components)

    judge_text_tokens = [count_tokens(tokenizer, item["combined_prompt"]) for item in judge_text]
    judge_tool_tokens = [count_tokens(tokenizer, item["combined_prompt"]) for item in judge_tool]
    reflection_tokens = [count_tokens(tokenizer, item["prompt"]) for item in reflection_prompts]

    payload = {
        "data": str(args.data),
        "tokenizer": args.tokenizer,
        "rows_measured": len(rows),
        "reflection_minibatch_size": len(minibatch_rows),
        "components": args.components,
        "judge_holistic_text_mode_input_tokens": stats(judge_text_tokens),
        "judge_holistic_tool_mode_input_tokens": stats(judge_tool_tokens),
        "tool_mode_extra_input_tokens_per_call": stats([b - a for a, b in zip(judge_text_tokens, judge_tool_tokens)]),
        "reflection_input_tokens_by_component": {
            item["component"]: count_tokens(tokenizer, item["prompt"]) for item in reflection_prompts
        },
        "reflection_input_tokens": stats(reflection_tokens),
        "rough_cost_model": {
            "judge_calls_per_full_360_eval": 360,
            "judge_input_tokens_per_full_360_eval_text_mode_estimate": int((statistics.mean(judge_text_tokens) if judge_text_tokens else 0) * 360),
            "judge_calls_per_gepa_metric_call_with_minibatch": len(minibatch_rows),
            "judge_input_tokens_per_gepa_metric_call_text_mode_estimate": int(sum(judge_text_tokens[: len(minibatch_rows)])),
            "reflection_calls_per_mutation": 1,
            "reflection_input_tokens_per_mutation_mean_over_measured_components": statistics.mean(reflection_tokens) if reflection_tokens else 0,
        },
    }
    payload["local_generation_samples"] = maybe_generate_samples(args, tokenizer, judge_text, reflection_prompts)

    write_json(args.output, payload)
    print(json.dumps({"output": str(args.output), "summary": payload["rough_cost_model"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
