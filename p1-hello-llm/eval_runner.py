"""
P1 · Hello, LLM — eval-instrumented multi-model comparison.

Final version: full gold-set sweep with summary stats.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from anthropic import Anthropic
from tenacity import retry, stop_after_attempt, wait_exponential

# ---------- Setup ----------
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
anthropic_client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

# Pricing as of late 2025 — USD per 1M tokens.
# Update from https://platform.openai.com/docs/pricing and
# https://docs.anthropic.com/en/docs/about-claude/pricing if it changes.
PRICING = {
    "gpt-4o-mini":               {"input": 0.150, "output": 0.600},
    "gpt-4o":                    {"input": 2.500, "output": 10.000},
    "claude-haiku-4-5-20251001": {"input": 1.000, "output": 5.000},
}

MODELS = ["gpt-4o-mini", "gpt-4o", "claude-haiku-4-5-20251001"]

SYSTEM = (
    "You are a fixed-income operations assistant. Answer concisely. "
    "If you do not know an answer or it would be financial advice, refuse politely. "
    "When the question asks for JSON, return ONLY valid JSON with no surrounding text."
)


# ---------- LLM call wrappers ----------
@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=2, max=20))
def call_openai(model: str, question: str) -> dict:
    t0 = time.perf_counter()
    resp = openai_client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": question}],
        temperature=0, seed=42, max_tokens=400,
    )
    return {
        "response": resp.choices[0].message.content or "",
        "prompt_tokens": resp.usage.prompt_tokens,
        "completion_tokens": resp.usage.completion_tokens,
        "latency_ms": int((time.perf_counter() - t0) * 1000),
    }


@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=2, max=20))
def call_anthropic(model: str, question: str) -> dict:
    t0 = time.perf_counter()
    resp = anthropic_client.messages.create(
        model=model, system=SYSTEM,
        messages=[{"role": "user", "content": question}],
        temperature=0, max_tokens=400,
    )
    return {
        "response": resp.content[0].text,
        "prompt_tokens": resp.usage.input_tokens,
        "completion_tokens": resp.usage.output_tokens,
        "latency_ms": int((time.perf_counter() - t0) * 1000),
    }


def call_llm(model: str, question: str) -> dict:
    if model.startswith("gpt-"):
        return call_openai(model, question)
    if model.startswith("claude-"):
        return call_anthropic(model, question)
    raise ValueError(f"Unknown model family: {model}")


# ---------- Eval functions ----------
REFUSAL_HINTS = (
    "cannot", "do not know", "i don't know", "not financial advice",
    "i'm not able", "as an ai", "refuse", "sorry",
)


def keyword_eval(response: str, expected_keywords: str, must_not_contain: str) -> int:
    if pd.isna(expected_keywords) or not expected_keywords:
        return 1
    text = response.lower()
    expected = [k.strip().lower() for k in str(expected_keywords).split(";") if k.strip()]
    forbidden = [] if pd.isna(must_not_contain) else [k.strip().lower() for k in str(must_not_contain).split(";") if k.strip()]    
    if any(f in text for f in forbidden):
        return 0
    return int(any(k in text for k in expected))


def refusal_eval(response: str, category: str) -> int:
    if category != "refusal":
        return 1
    return int(any(h in response.lower() for h in REFUSAL_HINTS))


def json_validity_eval(response: str, category: str) -> int:
    if category not in ("format", "adversarial"):
        return 1
    cleaned = re.sub(r"^```(?:json)?|```$", "", response.strip(), flags=re.MULTILINE).strip()
    try:
        json.loads(cleaned)
        return 1
    except Exception:
        return 0


# ---------- Cost calculation ----------
def usd_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Compute the USD cost of one request given token counts."""
    p = PRICING[model]
    # Pricing is per 1M tokens. So we divide actual tokens by 1,000,000 and multiply by the rate.
    return (prompt_tokens / 1_000_000) * p["input"] + (completion_tokens / 1_000_000) * p["output"]


# ---------- Main loop ----------
def main():
    here = Path(__file__).parent
    gold = pd.read_csv(here / "gold_set.csv")
    print(f"Loaded {len(gold)} questions across {gold['category'].nunique()} categories.")

    rows = []  # we'll collect one row per (question × model) here, then convert to DataFrame
    for _, q in gold.iterrows():
        # iterrows() yields (index, Series) tuples. We don't care about the index, so use _.
        # Each q is a Series — q["question"], q["category"], etc. work like dict access.
        for model in MODELS:
            print(f"  {model[:18]:<18}  {q['id']}  {q['question'][:60]}...")
            try:
                r = call_llm(model, q["question"])
            except Exception as e:
                print(f"    ! error: {e}")
                continue  # skip this row, keep going
            kw = keyword_eval(r["response"], q["expected_keywords"], q["must_not_contain"])
            rf = refusal_eval(r["response"], q["category"])
            jv = json_validity_eval(r["response"], q["category"])
            cost = usd_cost(model, r["prompt_tokens"], r["completion_tokens"])
            rows.append({
                "qid": q["id"],
                "category": q["category"],
                "model": model,
                "response": r["response"][:300],   # truncate so the CSV stays readable
                "prompt_tokens": r["prompt_tokens"],
                "completion_tokens": r["completion_tokens"],
                "latency_ms": r["latency_ms"],
                "cost_usd": round(cost, 6),
                "keyword_eval": kw,
                "refusal_eval": rf,
                "json_eval": jv,
                "passed_all": int(kw and rf and jv),  # all 3 must pass for an overall pass
            })

    df = pd.DataFrame(rows)
    df.to_csv(here / "runs.csv", index=False)
    print(f"\nWrote {len(df)} rows to runs.csv")

    # ---------- Per-model summary ----------
    summary = df.groupby("model").agg(
        accuracy=("passed_all", "mean"),
        avg_cost_usd=("cost_usd", "mean"),
        cost_per_1k=("cost_usd", lambda s: round(s.mean() * 1000, 4)),
        p50_latency_ms=("latency_ms", lambda s: int(s.quantile(0.50))),
        p95_latency_ms=("latency_ms", lambda s: int(s.quantile(0.95))),
        keyword_pass=("keyword_eval", "mean"),
        refusal_pass=("refusal_eval", "mean"),
        json_pass=("json_eval", "mean"),
    ).reset_index()
    # Convert fractions to percentages for readability
    summary["accuracy"] = (summary["accuracy"] * 100).round(1)
    summary[["keyword_pass", "refusal_pass", "json_pass"]] = (
        summary[["keyword_pass", "refusal_pass", "json_pass"]] * 100
    ).round(1)
    summary.to_csv(here / "summary.csv", index=False)
    print("\nPer-model summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()