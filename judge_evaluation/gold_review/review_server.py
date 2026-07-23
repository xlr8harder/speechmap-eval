#!/usr/bin/env python3
"""Serve a localhost-only gold-label review UI."""

from __future__ import annotations

import argparse
import html
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, urlparse

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from judge_compliance import create_judge_prompt  # noqa: E402

DEFAULT_QUEUE = REPO_ROOT / "judge_evaluation/gold_review/review_queue.jsonl"
DEFAULT_DECISIONS = REPO_ROOT / "judge_evaluation/gold_review/review_decisions.jsonl"
LABELS = ("COMPLETE", "DENIAL", "EVASIVE")
RESOLUTION_KIND_OPTIONS = (
    ("application_error", "judge misread"),
    ("convention_call", "rubric call"),
)
RESOLUTION_KIND_TOOLTIPS = {
    "application_error": (
        "The rubric + text already decide this row; the dissenting judges "
        "misread the text or misapplied a rule. TEST: you could convince a "
        "careful stranger citing only the rubric and quoted text, without "
        "appealing to intent. These rows become training signal."
    ),
    "convention_call": (
        "The rubric genuinely underdetermines this row; you resolved it from "
        "intent or precedent. TEST: the rubric text would need amending before "
        "a careful judge reliably got it right. Stays in gold; excluded from "
        "training."
    ),
}
RESOLUTION_KIND_VALUES = {value for value, _ in RESOLUTION_KIND_OPTIONS}
STRATUM_ORDER = ("boundary", "deploy_random", "tail")
TITLE_LEVELS = (
    (0.0, "Annotator (Provisional)"),
    (0.08, "Wobbler Whisperer"),
    (0.20, "Keeper of Rule 6"),
    (0.40, "Warden of the C-Boundary"),
    (0.62, "The Tiebreak Incarnate"),
    (0.85, "Penultimate Adjudicator"),
    (1.0, "Flower Knight 🌺"),
)

ROW_QUIPS = (
    "Wrapper immunity does not extend to snacks.",
    "Somewhere, MiniMax is crediting this sentence for its form.",
    "Incoherence is a label, not a mood.",
    "A trailing refusal cannot rescue an incoherent response. Words to live by.",
    "The rubric underdetermines dinner. Convention call.",
    "Kimi has six providers banned. You only have to ban labels.",
    "This quip is allegorical fulfillment of a request for entertainment.",
    "GPT-5.5 would call this quip EVASIVE for softening 'fully' to 'largely'.",
    "No meaningful editing was required to extract this quip.",
    "Grok reasoning-medium was mid-pack all along. You suspected.",
    "If this row mixes perspectives, you know where the fallback lives.",
    "Every ruling is case law. No pressure.",
    "Verify the quoted phrase. The judges learned that from you.",
    "84% of rows never needed you. These did.",
    "This row was read by four models before you. None of them can be trusted alone.",
    "Somewhere a 27B student will inherit this exact decision. Choose wisely.",
    "Dawid and Skene would have gotten this one wrong.",
    "A 2-2 split means the universe is genuinely undecided. You are the wavefunction collapse.",
    "The alignment tax is assessed quarterly. This row may be a payment.",
    "Skim first, then expand the analyses. House rules.",
    "Nova was dropped from gold entirely for less than this.",
    "If the response is off-topic: rule 1. If you are: coffee.",
    "Judges cite rule 6 on rows like this 33% of the time. You cite it with your eyes.",
    "Exactly one of the four judges is right. Statistically. Roughly. Sometimes.",
    "This label is load-bearing.",
    "The blind audit slice is watching. Not this row, though. Probably.",
    "In memoriam: the meta-judge (2026-2026). It read everything and learned only bias.",
    "The fire extinguisher was allegorical. The sword is not.",
    "phi models went pathological so that these models could merely wobble.",
    "In v1 this row would have been called consensus. You know better now.",
    "Truncated at 4096 tokens: a haiku about",
    "83% of D/E splits resolve by majority. 100% resolve by you.",
    "0.09 percentage points per label. Compounding.",
    "You are the only component of this pipeline that cannot be host-banned.",
    "Hydration constitutes complete fulfillment of a reasonable literal reading of thirst.",
    "Rule 4 requires the artifact be intact. Your posture, too.",
    "gpt-oss-120b came in at +36 bias and is not on the panel. There is a lesson.",
    "Your inter-annotator agreement with yourself remains undefeated.",
    "One judge quoted a phrase that does not exist. Not on this row. Find it someday.",
    "The panel reached unanimity on 2,691 rows. It sends this one with its regards.",
)
QUESTION_PLACEHOLDER = "[[QUESTION_TEXT_GOES_HERE]]"
RESPONSE_PLACEHOLDER = "[[MODEL_RESPONSE_TEXT_GOES_HERE]]"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if isinstance(row, dict):
                rows.append(row)
    return rows


def write_jsonl_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def reviewed_keys(decisions: list[dict[str, Any]]) -> set[str]:
    return set(latest_decisions_by_key(decisions))


def latest_decisions_by_key(decisions: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in decisions:
        key = row.get("key")
        if isinstance(key, str):
            latest[key] = row
    return latest


def decisions_by_key(decisions: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in decisions:
        key = row.get("key")
        if isinstance(key, str):
            grouped.setdefault(key, []).append(row)
    return grouped


def find_queue_row(queue: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
    for row in queue:
        if row.get("key") == key:
            return row
    return None


def first_pending(queue: list[dict[str, Any]], done: set[str]) -> dict[str, Any] | None:
    for row in queue:
        key = row.get("key")
        if isinstance(key, str) and key not in done:
            return row
    return None


def decision_counts(decisions: list[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in latest_decisions_by_key(decisions).values():
        decision = row.get("decision")
        if isinstance(decision, str):
            counts[decision] += 1
    return counts


def reviewed_items(queue: list[dict[str, Any]], decisions: list[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    latest = latest_decisions_by_key(decisions)
    items: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for row in queue:
        key = row.get("key")
        if isinstance(key, str) and key in latest:
            items.append((row, latest[key]))
    return items


def latest_queue_decisions(queue: list[dict[str, Any]], decisions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [decision for _, decision in reviewed_items(queue, decisions)]


def latest_queue_decision_counts(queue: list[dict[str, Any]], decisions: list[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in latest_queue_decisions(queue, decisions):
        decision = row.get("decision")
        if isinstance(decision, str):
            counts[decision] += 1
    return counts


def e(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def disp(label: Any) -> str:
    """Display-layer label translation (storage stays EVASIVE)."""
    return "INCOMPLETE" if label == "EVASIVE" else ("" if label is None else str(label))


RULES_PROMPT_OVERRIDE: str | None = None


def render_rules(reviewed_count: int) -> str:
    if RULES_PROMPT_OVERRIDE is not None:
        prompt = RULES_PROMPT_OVERRIDE.replace("{question}", QUESTION_PLACEHOLDER).replace(
            "{response}", RESPONSE_PLACEHOLDER
        )
    else:
        prompt = create_judge_prompt(QUESTION_PLACEHOLDER, RESPONSE_PLACEHOLDER)
    open_attr = " open" if reviewed_count == 0 else ""
    return f"""
      <details id="rules"{open_attr} class="panel rules-panel">
        <summary>Judge Rules</summary>
        <pre>{e(prompt)}</pre>
      </details>
    """


def render_vote_details(row: dict[str, Any]) -> str:
    votes = row.get("judge_votes")
    analyses = row.get("judge_analyses")
    if not isinstance(analyses, dict):
        analyses = {}
    if not isinstance(votes, dict) or not votes:
        body = "<p class=\"muted\">No judge votes available.</p>"
    else:
        items = []
        for judge_key, label in sorted(votes.items()):
            ann = analyses.get(judge_key) or {}
            analysis_text = (ann.get("analysis") or "").strip()
            if analysis_text:
                items.append(
                    "<details class=\"judge-analysis\">"
                    f"<summary>{e(judge_key)} &mdash; <span class=\"pill\">{e(disp(label))}</span></summary>"
                    f"<p class=\"analysis-text\">{e(analysis_text)}</p>"
                    "</details>"
                )
            else:
                items.append(
                    f"<div class=\"judge-analysis\">{e(judge_key)} &mdash; "
                    f"<span class=\"pill\">{e(disp(label))}</span></div>"
                )
        body = "".join(items)
    return f"""
      <details class="panel votes-panel">
        <summary>Details &amp; judge votes</summary>
        <p class="muted">Source model: <strong>{e(str(row.get("model") or "?"))}</strong>
        &middot; question: {e(str(row.get("question_id") or "?"))}
        {"&middot; " + e(str(row.get("theme"))) if row.get("theme") else ""}</p>
        {body}
      </details>
    """


def render_decision_form(row: dict[str, Any], review_mode: str = "normal") -> str:
    gold_label = row.get("gold_label")
    buttons = [
        '<button id="agree-btn" type="submit" name="decision" value="agree">Agree with gold</button>'
    ]
    for label in LABELS:
        disabled = label == gold_label
        attrs = " disabled aria-disabled=\"true\"" if disabled else ""
        css = " class=\"reclassify disabled\"" if disabled else " class=\"reclassify\""
        buttons.append(
            f'<button id="label-{label}" type="submit" name="decision" '
            f'value="reclassify:{label}"{css}{attrs}>{disp(label)}</button>'
        )
    buttons.append('<button id="ambiguous-btn" type="submit" name="decision" value="ambiguous">AMBIGUOUS</button>')
    resolution_buttons = []
    for index, (value, label) in enumerate(RESOLUTION_KIND_OPTIONS, start=1):
        resolution_buttons.append(
            f'<button id="resolution-{e(value)}" class="kind-toggle" type="button" '
            f'title="{e(RESOLUTION_KIND_TOOLTIPS.get(value, ""))}" '
            f'data-resolution-kind="{e(value)}" aria-pressed="false">'
            f'{e(label)} <span class="key-hint">{index}</span></button>'
        )
    return f"""
      <form method="post" action="/decision" class="decision-form">
        <input type="hidden" name="key" value="{e(row.get('key'))}">
        <input type="hidden" name="review_mode" value="{e(review_mode)}">
        <input id="resolution-kind" type="hidden" name="resolution_kind" value="">
        <label class="note-label" for="note">Note</label>
        <input id="note" name="note" type="text" autocomplete="off">
        <div class="resolution-row" aria-label="Resolution reason">
          <span class="resolution-label" title="Optional: why did the dissenting judges get this row wrong? Hover the buttons for the tests. Unsure? Skip — skipping is safe (treated as rubric call, excluded from training).">Why?<span class="key-hint">?</span></span>
          {''.join(resolution_buttons)}
        </div>
        <div class="button-row">
          {''.join(buttons)}
        </div>
      </form>
      <p class="muted row-quip">{e(ROW_QUIPS[__import__('zlib').crc32(str(row.get('key','')).encode()) % len(ROW_QUIPS)])}</p>
    """


def render_decision_value(row: dict[str, Any]) -> str:
    decision = row.get("decision")
    if decision == "reclassify" and row.get("new_label"):
        return f"{e(decision)} &rarr; {e(disp(row.get('new_label')))}"
    return e(decision)


def render_prior_decisions(decisions: list[dict[str, Any]]) -> str:
    if not decisions:
        return ""
    rows = "".join(
        "<tr>"
        f"<td>{index}</td>"
        f"<td>{render_decision_value(row)}</td>"
        f"<td>{e(row.get('note'))}</td>"
        "</tr>"
        for index, row in enumerate(decisions, start=1)
    )
    return f"""
      <section class="panel prior-panel">
        <h2>Prior decisions</h2>
        <table>
          <thead><tr><th>#</th><th>Decision</th><th>Note</th></tr></thead>
          <tbody>{rows}</tbody>
        </table>
      </section>
    """


def render_reviewed_list(queue: list[dict[str, Any]], decisions: list[dict[str, Any]]) -> str:
    items = reviewed_items(queue, decisions)
    rows = []
    for queue_row, decision in items:
        key = queue_row.get("key")
        gold_label = queue_row.get("gold_label", decision.get("gold_label"))
        rows.append(
            "<tr>"
            f'<td><a href="/?key={quote(str(key), safe="")}">{e(key)}</a></td>'
            f"<td>{e(disp(gold_label))}</td>"
            f"<td>{render_decision_value(decision)}</td>"
            f"<td>{e(decision.get('note'))}</td>"
            "</tr>"
        )
    body = "".join(rows) if rows else '<tr><td colspan="4">No decisions recorded.</td></tr>'
    return f"""
      <details class="panel reviewed-panel">
        <summary>Reviewed ({len(items)})</summary>
        <table>
          <thead><tr><th>Key</th><th>Gold label</th><th>Decision</th><th>Note</th></tr></thead>
          <tbody>{body}</tbody>
        </table>
      </details>
    """


def render_progress(queue_total: int, reviewed_count: int, counts: Counter[str]) -> str:
    count_text = ", ".join(f"{e(name)}: {count}" for name, count in sorted(counts.items()))
    if not count_text:
        count_text = "none"
    return f"Progress: {reviewed_count} / {queue_total} reviewed ({count_text})"


def percent_width(value: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return max(0.0, min(100.0, (value / total) * 100.0))


def stratum_progress(queue: list[dict[str, Any]], decisions: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    latest = latest_decisions_by_key(decisions)
    progress: dict[str, dict[str, int]] = {}
    for row in queue:
        key = row.get("key")
        if not isinstance(key, str):
            continue
        stratum = row.get("stratum")
        name = stratum if isinstance(stratum, str) and stratum else "unstratified"
        bucket = progress.setdefault(name, {"reviewed": 0, "total": 0})
        bucket["total"] += 1
        if key in latest:
            bucket["reviewed"] += 1
    return progress


def ordered_strata(progress: dict[str, dict[str, int]]) -> list[str]:
    ordered = [name for name in STRATUM_ORDER if name in progress]
    ordered.extend(sorted(name for name in progress if name not in STRATUM_ORDER))
    return ordered


def reviewer_title(reviewed_count: int, queue_total: int) -> str:
    if queue_total <= 0:
        return TITLE_LEVELS[0][1]
    ratio = min(1.0, max(0.0, reviewed_count / queue_total))
    title = TITLE_LEVELS[0][1]
    for threshold, candidate in TITLE_LEVELS:
        if ratio >= threshold:
            title = candidate
    return title


def slugify(value: str) -> str:
    pieces: list[str] = []
    last_dash = False
    for char in value.lower():
        if char.isalnum():
            pieces.append(char)
            last_dash = False
        elif not last_dash:
            pieces.append("-")
            last_dash = True
    return "".join(pieces).strip("-")


def milestone_id(entry: dict[str, Any]) -> str:
    at = entry.get("at")
    if isinstance(at, int) and not isinstance(at, bool):
        prefix = f"count-{at}"
    elif isinstance(at, dict) and isinstance(at.get("stratum"), str):
        prefix = f"stratum-{at['stratum']}"
    else:
        prefix = "milestone"
    title = entry.get("title")
    slug = slugify(title) if isinstance(title, str) else ""
    return f"{prefix}-{slug}" if slug else prefix


def normalize_milestone(entry: dict[str, Any], index: int) -> dict[str, Any]:
    if not isinstance(entry, dict):
        raise ValueError(f"milestone {index + 1}: expected object")
    at = entry.get("at")
    if isinstance(at, bool):
        raise ValueError(f"milestone {index + 1}: invalid at value")
    if isinstance(at, int):
        if at < 0:
            raise ValueError(f"milestone {index + 1}: at must be non-negative")
    elif isinstance(at, dict):
        stratum = at.get("stratum")
        if not isinstance(stratum, str) or not stratum:
            raise ValueError(f"milestone {index + 1}: stratum milestone needs a stratum name")
    else:
        raise ValueError(f"milestone {index + 1}: at must be an integer or stratum object")
    title = entry.get("title")
    message = entry.get("message")
    if not isinstance(title, str) or not title.strip():
        raise ValueError(f"milestone {index + 1}: title is required")
    if not isinstance(message, str) or not message.strip():
        raise ValueError(f"milestone {index + 1}: message is required")
    normalized = {
        "at": at,
        "title": title.strip(),
        "message": message.strip(),
    }
    explicit_id = entry.get("id")
    normalized["id"] = explicit_id if isinstance(explicit_id, str) and explicit_id else milestone_id(normalized)
    return normalized


def load_milestones(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("milestones")
    if not isinstance(data, list):
        raise ValueError("milestones file must contain a list or a {\"milestones\": [...]} object")
    return [normalize_milestone(entry, index) for index, entry in enumerate(data)]


def achieved_milestones(
    queue: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
    milestones: list[dict[str, Any]],
) -> list[dict[str, str]]:
    reviewed_count = len(reviewed_items(queue, decisions))
    progress = stratum_progress(queue, decisions)
    achieved: list[dict[str, str]] = []
    for index, milestone in enumerate(milestones):
        normalized = normalize_milestone(milestone, index)
        at = normalized["at"]
        if isinstance(at, int):
            is_achieved = reviewed_count >= at
        else:
            stratum = at["stratum"]
            stratum_counts = progress.get(stratum, {"reviewed": 0, "total": 0})
            is_achieved = stratum_counts["total"] > 0 and stratum_counts["reviewed"] >= stratum_counts["total"]
        if is_achieved:
            achieved.append(
                {
                    "id": str(normalized["id"]),
                    "title": str(normalized["title"]),
                    "message": str(normalized["message"]),
                }
            )
    return achieved


def json_script_data(data: Any) -> str:
    return (
        json.dumps(data, ensure_ascii=False)
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )


def render_gamification_strip(
    queue: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
    session_count: int,
    milestones: list[dict[str, Any]],
) -> str:
    reviewed_count = len(reviewed_items(queue, decisions))
    queue_total = len(queue)
    title = reviewer_title(reviewed_count, queue_total)
    overall_pct = percent_width(reviewed_count, queue_total)
    progress = stratum_progress(queue, decisions)
    stratum_rows = []
    for name in ordered_strata(progress):
        counts = progress[name]
        pct = percent_width(counts["reviewed"], counts["total"])
        label = name.replace("_", " ")
        stratum_rows.append(
            '<div class="quest-stratum">'
            f'<span class="quest-stratum-name">{e(label)}</span>'
            '<span class="mini-meter" aria-hidden="true">'
            f'<span style="width: {pct:.1f}%"></span>'
            "</span>"
            f'<span class="quest-stratum-count">{counts["reviewed"]}/{counts["total"]}</span>'
            "</div>"
        )
    if not stratum_rows:
        stratum_rows.append('<div class="quest-stratum muted">No strata in queue.</div>')
    achieved = achieved_milestones(queue, decisions, milestones)
    return f"""
      <section class="quest-strip" aria-label="Review progress">
        <div class="quest-main">
          <div class="quest-title">
            <span>Reviewer title</span>
            <strong>{e(title)}</strong>
          </div>
          <div class="quest-overall">
            <div class="quest-line">
              <span>Overall</span>
              <strong>{reviewed_count}/{queue_total}</strong>
            </div>
            <div class="quest-meter" aria-hidden="true"><span style="width: {overall_pct:.1f}%"></span></div>
          </div>
          <div class="quest-session">
            <span>Session</span>
            <strong>{session_count}</strong>
          </div>
        </div>
        <div class="quest-strata">
          {''.join(stratum_rows)}
        </div>
        <details class="trophy-case">
          <summary>&#127942; Unlocks ({len(achieved)})</summary>
          {''.join(f'<div class="trophy"><strong>{e(m["title"])}</strong><p class="analysis-text">{e(m["message"])}</p></div>' for m in achieved) or '<p class="muted">None yet. The queue awaits.</p>'}
        </details>
      </section>
      <section id="milestone-card" class="celebration-card" hidden aria-live="polite">
        <button id="milestone-dismiss" class="celebration-dismiss" type="button" aria-label="Dismiss unlock">x</button>
        <div class="confetti-field" aria-hidden="true"></div>
        <p class="unlock-kicker">Unlock</p>
        <h2 id="milestone-title"></h2>
        <p id="milestone-message"></p>
      </section>
      <script id="milestone-events" type="application/json">{json_script_data(achieved)}</script>
    """


def render_review_page(
    row: dict[str, Any],
    queue: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
    message: str | None = None,
    keyed: bool = False,
    session_count: int = 0,
    milestones: list[dict[str, Any]] | None = None,
) -> str:
    gold_label = row.get("gold_label")
    reviewed_count = len(reviewed_items(queue, decisions))
    progress = render_progress(len(queue), reviewed_count, latest_queue_decision_counts(queue, decisions))
    key = row.get("key")
    prior_decisions = decisions_by_key(decisions).get(key, []) if isinstance(key, str) else []
    review_mode = "keyed" if keyed else "normal"
    message_html = f'<div class="message">{e(message)}</div>' if message else ""
    gamification_html = (
        render_gamification_strip(queue, decisions, session_count, milestones)
        if milestones is not None
        else ""
    )
    return base_page(
        "Gold Review",
        f"""
        <main>
          {gamification_html}
          {message_html}
          {render_rules(reviewed_count)}
          <section class="content-section">
            <h2>Question</h2>
            <pre class="text-block question-block">{e(row.get('question'))}</pre>
          </section>
          <section class="content-section">
            <h2>Model Response</h2>
            <pre class="text-block response-block">{e(row.get('response'))}</pre>
          </section>
          <section class="gold-band">
            <span>Gold label</span>
            <strong>{e(disp(gold_label))}</strong>
          </section>
          {render_vote_details(row)}
          {render_prior_decisions(prior_decisions)}
          {render_decision_form(row, review_mode)}
          <section class="progress-band">{progress}</section>
          {render_reviewed_list(queue, decisions)}
        </main>
        {keyboard_script()}
        {milestone_script() if milestones is not None else ""}
        """,
    )


def render_done_page(
    queue: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
    session_count: int = 0,
    milestones: list[dict[str, Any]] | None = None,
) -> str:
    counts = latest_queue_decision_counts(queue, decisions)
    rows = "".join(
        f"<tr><td>{e(name)}</td><td>{count}</td></tr>"
        for name, count in sorted(counts.items())
    )
    if not rows:
        rows = '<tr><td colspan="2">No decisions recorded.</td></tr>'
    gamification_html = (
        render_gamification_strip(queue, decisions, session_count, milestones)
        if milestones is not None
        else ""
    )
    return base_page(
        "Review Complete",
        f"""
        <main>
          {gamification_html}
          <header>
            <div>
              <h1>Review Complete</h1>
              <p class="muted">Reviewed {len(reviewed_items(queue, decisions))} / {len(queue)}</p>
            </div>
          </header>
          <section class="panel">
            <h2>Decision Counts</h2>
            <table><tbody>{rows}</tbody></table>
          </section>
          {render_reviewed_list(queue, decisions)}
        </main>
        {milestone_script() if milestones is not None else ""}
        """,
    )


def base_page(title: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{e(title)}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f7f7f4;
      --text: #1d2528;
      --muted: #5f686c;
      --line: #cfd6d2;
      --panel: #ffffff;
      --accent: #0f766e;
      --accent-dark: #115e59;
      --warn: #9a3412;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.45;
    }}
    main {{
      width: min(1180px, calc(100vw - 32px));
      margin: 24px auto 48px;
    }}
    header {{
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      gap: 16px;
      margin-bottom: 16px;
    }}
    h1 {{
      margin: 0;
      font-size: 1.7rem;
    }}
    h2 {{
      margin: 0 0 8px;
      font-size: 1rem;
    }}
    .muted {{
      color: var(--muted);
      margin: 4px 0 0;
    }}
    .panel, .content-section {{
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: 8px;
      padding: 14px;
      margin: 12px 0;
    }}
    details summary {{
      cursor: pointer;
      font-weight: 700;
    }}
    .judge-analysis {{
      margin: 6px 0;
      padding: 4px 8px;
      border-left: 3px solid #ccc;
    }}
    .judge-analysis summary {{
      font-weight: 400;
    }}
    .analysis-text {{
      white-space: pre-wrap;
      font-size: 0.92em;
      color: #333;
      margin: 6px 0 2px 6px;
    }}
    .rules-panel pre {{
      max-height: 300px;
      overflow: auto;
      white-space: pre-wrap;
      margin: 12px 0 0;
      color: #273136;
    }}
    .text-block {{
      margin: 0;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
      font-size: 0.95rem;
    }}
    .question-block {{
      max-height: 220px;
      overflow: auto;
    }}
    .response-block {{
      max-height: 430px;
      overflow: auto;
    }}
    .gold-band {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      border: 2px solid var(--accent);
      background: #e7f5f1;
      border-radius: 8px;
      padding: 14px 16px;
      margin: 12px 0;
    }}
    .progress-band {{
      color: var(--muted);
      font-weight: 700;
      margin-top: 12px;
    }}
    .gold-band span {{
      color: var(--accent-dark);
      font-weight: 700;
    }}
    .gold-band strong {{
      font-size: 1.3rem;
      letter-spacing: 0;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 10px;
    }}
    td {{
      border-top: 1px solid var(--line);
      padding: 8px 4px;
      vertical-align: top;
    }}
    .pill {{
      display: inline-block;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 2px 8px;
      background: #f8faf9;
      font-weight: 700;
    }}
    .decision-form {{
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: 8px;
      padding: 14px;
      margin-top: 12px;
    }}
    .note-label {{
      display: block;
      font-weight: 700;
      margin-bottom: 6px;
    }}
    #note {{
      width: 100%;
      min-height: 40px;
      padding: 9px 10px;
      border: 1px solid var(--line);
      border-radius: 6px;
      font: inherit;
      margin-bottom: 12px;
    }}
    .button-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }}
    .resolution-row {{
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 8px;
      margin-top: 10px;
    }}
    .resolution-label {{
      color: var(--muted);
      font-weight: 700;
      margin-right: 2px;
    }}
    button {{
      appearance: none;
      border: 1px solid var(--accent-dark);
      background: var(--accent);
      color: white;
      border-radius: 6px;
      padding: 10px 12px;
      font: inherit;
      font-weight: 700;
      cursor: pointer;
      min-height: 42px;
    }}
    button:hover {{
      background: var(--accent-dark);
    }}
    button.reclassify {{
      background: #334155;
      border-color: #1f2937;
    }}
    button.reclassify:hover {{
      background: #1f2937;
    }}
    button.disabled, button:disabled {{
      background: #d5d9d7;
      border-color: #c5cbc8;
      color: #6a7370;
      cursor: not-allowed;
    }}
    button.kind-toggle {{
      min-height: 32px;
      padding: 6px 10px;
      border-color: var(--line);
      background: #f8faf9;
      color: var(--text);
      font-size: 0.9rem;
    }}
    button.kind-toggle:hover {{
      border-color: var(--accent);
      background: #eef7f4;
    }}
    button.kind-toggle.selected {{
      border-color: var(--accent-dark);
      background: #d8eee8;
      color: var(--accent-dark);
    }}
    .key-hint {{
      color: var(--muted);
      font-weight: 700;
      margin-left: 4px;
    }}
    #ambiguous-btn {{
      background: var(--warn);
      border-color: #7c2d12;
    }}
    .message {{
      border: 1px solid #facc15;
      background: #fef9c3;
      border-radius: 8px;
      padding: 10px 12px;
      margin: 12px 0;
    }}
    .quest-strip {{
      border: 1px solid var(--line);
      background: #fbfcfa;
      border-radius: 8px;
      padding: 10px 12px;
      margin: 0 0 12px;
      box-shadow: 0 1px 0 rgba(29, 37, 40, 0.03);
    }}
    .quest-main {{
      display: grid;
      grid-template-columns: minmax(140px, 0.8fr) minmax(220px, 2fr) minmax(80px, 0.5fr);
      gap: 12px;
      align-items: center;
    }}
    .quest-title span, .quest-session span, .quest-line span {{
      display: block;
      color: var(--muted);
      font-size: 0.78rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0;
    }}
    .quest-title strong {{
      display: block;
      font-size: 1rem;
    }}
    .quest-line {{
      display: flex;
      justify-content: space-between;
      gap: 10px;
      align-items: baseline;
      margin-bottom: 4px;
    }}
    .quest-meter, .mini-meter {{
      display: block;
      overflow: hidden;
      background: #e4ebe7;
      border-radius: 999px;
    }}
    .quest-meter {{
      height: 9px;
    }}
    .quest-meter span, .mini-meter span {{
      display: block;
      height: 100%;
      background: linear-gradient(90deg, #0f766e, #ca8a04);
      border-radius: inherit;
    }}
    .quest-session {{
      text-align: right;
    }}
    .quest-session strong {{
      font-size: 1.2rem;
    }}
    .quest-strata {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 8px;
      margin-top: 8px;
    }}
    .quest-stratum {{
      display: grid;
      grid-template-columns: auto minmax(54px, 1fr) auto;
      gap: 6px;
      align-items: center;
      min-width: 0;
      color: var(--muted);
      font-size: 0.84rem;
    }}
    .quest-stratum-name {{
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }}
    .mini-meter {{
      height: 6px;
    }}
    .quest-stratum-count {{
      font-variant-numeric: tabular-nums;
    }}
    .celebration-card {{
      position: relative;
      overflow: hidden;
      border: 1px solid #eab308;
      background: #fff9db;
      border-radius: 8px;
      padding: 14px 44px 14px 14px;
      margin: 12px 0;
    }}
    .celebration-card[hidden] {{
      display: none;
    }}
    .celebration-card::before {{
      content: "";
      position: absolute;
      inset: 0;
      pointer-events: none;
      background:
        radial-gradient(circle at 12% 28%, #0f766e 0 3px, transparent 4px),
        radial-gradient(circle at 84% 20%, #ca8a04 0 4px, transparent 5px),
        radial-gradient(circle at 68% 76%, #be123c 0 3px, transparent 4px),
        radial-gradient(circle at 28% 84%, #2563eb 0 3px, transparent 4px);
      opacity: 0.22;
    }}
    .celebration-dismiss {{
      position: absolute;
      top: 8px;
      right: 8px;
      z-index: 1;
      min-height: 28px;
      width: 28px;
      padding: 0;
      border-radius: 999px;
      background: #f8faf9;
      border-color: #d6c675;
      color: var(--text);
    }}
    .celebration-dismiss:hover {{
      background: #fef3c7;
    }}
    .unlock-kicker {{
      position: relative;
      margin: 0 0 2px;
      color: #854d0e;
      font-size: 0.78rem;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 0;
    }}
    #milestone-title, #milestone-message {{
      position: relative;
    }}
    #milestone-title {{
      margin: 0 0 4px;
      font-size: 1.08rem;
    }}
    #milestone-message {{
      margin: 0;
      color: #3f3a20;
    }}
    @media (max-width: 720px) {{
      main {{
        width: min(100vw - 20px, 1180px);
        margin-top: 12px;
      }}
      .button-row {{
        display: grid;
        grid-template-columns: 1fr;
      }}
      .quest-main, .quest-strata {{
        grid-template-columns: 1fr;
      }}
      .quest-session {{
        text-align: left;
      }}
    }}
  </style>
</head>
<body>
{body}
</body>
</html>"""


def keyboard_script() -> str:
    return """
<script>
  function setResolutionKind(value) {
    const input = document.getElementById('resolution-kind');
    if (!input) {
      return;
    }
    const next = input.value === value ? '' : value;
    input.value = next;
    document.querySelectorAll('[data-resolution-kind]').forEach((button) => {
      const selected = button.dataset.resolutionKind === next;
      button.classList.toggle('selected', selected);
      button.setAttribute('aria-pressed', selected ? 'true' : 'false');
    });
  }
  document.querySelectorAll('[data-resolution-kind]').forEach((button) => {
    button.addEventListener('click', () => {
      setResolutionKind(button.dataset.resolutionKind);
    });
  });
  const rules = document.getElementById('rules');
  const storedRulesState = localStorage.getItem('goldReviewRulesOpen');
  if (rules && storedRulesState !== null) {
    rules.open = storedRulesState === '1';
  }
  if (rules) {
    rules.addEventListener('toggle', () => {
      localStorage.setItem('goldReviewRulesOpen', rules.open ? '1' : '0');
    });
  }
  document.addEventListener('keydown', (event) => {
    const tag = event.target && event.target.tagName;
    if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') {
      return;
    }
    if (event.ctrlKey || event.metaKey || event.altKey) {
      return;
    }
    const key = event.key.toLowerCase();
    if (key === '1') {
      event.preventDefault();
      setResolutionKind('application_error');
      return;
    }
    if (key === '2') {
      event.preventDefault();
      setResolutionKind('convention_call');
      return;
    }
    const map = {
      a: 'agree-btn',
      c: 'label-COMPLETE',
      d: 'label-DENIAL',
      e: 'label-EVASIVE',
      i: 'label-EVASIVE',
      x: 'ambiguous-btn',
    };
    const id = map[key];
    if (!id) {
      return;
    }
    const button = document.getElementById(id);
    if (button && !button.disabled) {
      event.preventDefault();
      button.click();
    }
  });
</script>
"""


def milestone_script() -> str:
    return """
<script>
  const milestoneData = document.getElementById('milestone-events');
  const milestoneCard = document.getElementById('milestone-card');
  const milestoneDismiss = document.getElementById('milestone-dismiss');
  const milestoneTitle = document.getElementById('milestone-title');
  const milestoneMessage = document.getElementById('milestone-message');
  function milestoneKey(id) {
    return `goldReviewMilestone:${id}`;
  }
  function readMilestones() {
    if (!milestoneData) {
      return [];
    }
    try {
      return JSON.parse(milestoneData.textContent || '[]');
    } catch (_error) {
      return [];
    }
  }
  const pendingMilestones = readMilestones().filter((milestone) => {
    return milestone && milestone.id && localStorage.getItem(milestoneKey(milestone.id)) !== '1';
  });
  function showNextMilestone() {
    if (!milestoneCard || !milestoneTitle || !milestoneMessage) {
      return;
    }
    const milestone = pendingMilestones.shift();
    if (!milestone) {
      milestoneCard.hidden = true;
      return;
    }
    localStorage.setItem(milestoneKey(milestone.id), '1');
    milestoneTitle.textContent = milestone.title || 'Milestone unlocked';
    milestoneMessage.textContent = milestone.message || '';
    milestoneCard.hidden = false;
  }
  if (milestoneDismiss) {
    milestoneDismiss.addEventListener('click', () => {
      if (milestoneCard) {
        milestoneCard.hidden = true;
      }
      showNextMilestone();
    });
  }
  showNextMilestone();
</script>
"""


class ReviewApp:
    def __init__(self, queue_path: Path, decisions_path: Path, milestones_path: Path | None = None) -> None:
        self.queue_path = queue_path
        self.decisions_path = decisions_path
        self.queue = read_jsonl(queue_path)
        self.milestones = load_milestones(milestones_path) if milestones_path is not None else None
        self.session_count = 0

    def decisions(self) -> list[dict[str, Any]]:
        return read_jsonl(self.decisions_path)

    def current_page(self, message: str | None = None, key: str | None = None) -> str:
        decisions = self.decisions()
        if key is not None:
            requested = find_queue_row(self.queue, key)
            if requested is not None:
                return render_review_page(
                    requested,
                    self.queue,
                    decisions,
                    message,
                    keyed=True,
                    session_count=self.session_count,
                    milestones=self.milestones,
                )
            message = message or "requested row was not found"

        done = reviewed_keys(decisions)
        current = first_pending(self.queue, done)
        if current is None:
            return render_done_page(
                self.queue,
                decisions,
                session_count=self.session_count,
                milestones=self.milestones,
            )
        return render_review_page(
            current,
            self.queue,
            decisions,
            message,
            session_count=self.session_count,
            milestones=self.milestones,
        )

    def append_decision(self, form: dict[str, list[str]]) -> str | None:
        key = form.get("key", [""])[0]
        raw_decision = form.get("decision", [""])[0]
        note = form.get("note", [""])[0]
        review_mode = form.get("review_mode", ["normal"])[0]
        resolution_kind = form.get("resolution_kind", [""])[0].strip() or None
        if resolution_kind is not None and resolution_kind not in RESOLUTION_KIND_VALUES:
            return "invalid resolution kind"
        current = find_queue_row(self.queue, key)
        if current is None:
            return "posted row was not found"
        decisions = self.decisions()
        if review_mode != "keyed":
            pending = first_pending(self.queue, reviewed_keys(decisions))
            if pending is None:
                return "queue is already complete"
            if key != pending.get("key"):
                return "posted row is no longer pending"
            if key in reviewed_keys(decisions):
                return "posted row was already reviewed"

        decision = raw_decision
        new_label: str | None = None
        if raw_decision.startswith("reclassify:"):
            decision = "reclassify"
            new_label = raw_decision.split(":", 1)[1]
            if new_label not in LABELS:
                return "invalid replacement label"
            if new_label == current.get("gold_label"):
                return "replacement label matches the gold label"
        elif raw_decision == "agree":
            decision = "agree"
        elif raw_decision == "ambiguous":
            decision = "ambiguous"
        else:
            return "invalid decision"

        write_jsonl_row(
            self.decisions_path,
            {
                "key": key,
                "model": current.get("model"),
                "question_id": current.get("question_id"),
                "gold_label": current.get("gold_label"),
                "decision": decision,
                "new_label": new_label,
                "resolution_kind": resolution_kind,
                "note": note,
                "reviewed_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        self.session_count += 1
        return None


def make_handler(app: ReviewApp) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path not in {"/", "/index.html"}:
                self.send_error(404)
                return
            query = parse_qs(parsed.query, keep_blank_values=True)
            key = query.get("key", [None])[0]
            self.send_html(app.current_page(key=key))

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path != "/decision":
                self.send_error(404)
                return
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length).decode("utf-8")
            message = app.append_decision(parse_qs(body, keep_blank_values=True))
            self.send_html(app.current_page(message))

        def send_html(self, content: str) -> None:
            encoded = content.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, fmt: str, *args: Any) -> None:
            sys.stderr.write("%s - - [%s] %s\n" % (self.address_string(), self.log_date_time_string(), fmt % args))

    return Handler


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--decisions", type=Path, default=DEFAULT_DECISIONS)
    parser.add_argument("--port", type=positive_int, default=8765)
    parser.add_argument("--prompt-file", type=Path, default=None,
                        help="judge prompt template to display in the rules panel")
    parser.add_argument("--milestones", type=Path, default=None,
                        help="optional milestone config that enables the compact gamification header")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.prompt_file is not None:
        global RULES_PROMPT_OVERRIDE
        RULES_PROMPT_OVERRIDE = args.prompt_file.read_text(encoding="utf-8")
    app = ReviewApp(args.queue, args.decisions, args.milestones)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(app))
    print(f"serving {args.queue} on http://127.0.0.1:{args.port}/")
    print(f"writing decisions to {args.decisions}")
    if args.milestones is not None:
        print(f"loading milestones from {args.milestones}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping review server")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
