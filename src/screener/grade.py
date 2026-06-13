"""Stage 3: vision grading via LiteLLM. Returns one JSON verdict per chart."""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import litellm

from .config import Settings, get_settings

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent / "prompt.txt"


@lru_cache(maxsize=1)
def system_prompt() -> str:
    """Read the system prompt from prompt.txt (cached). Edit that file to iterate."""
    return _PROMPT_PATH.read_text()


_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _encode_image(path: Path) -> str:
    data = path.read_bytes()
    return f"data:image/png;base64,{base64.b64encode(data).decode('ascii')}"


def _load_examples(examples_dir: Path) -> list[dict[str, Any]]:
    """Build a fixed few-shot block from examples/{vcp,not_vcp}/*.png."""
    parts: list[dict[str, Any]] = []
    pos_dir = examples_dir / "vcp"
    neg_dir = examples_dir / "not_vcp"

    positives = sorted(pos_dir.glob("*.png"))[:4] if pos_dir.exists() else []
    negatives = sorted(neg_dir.glob("*.png"))[:2] if neg_dir.exists() else []

    if positives:
        parts.append({"type": "text", "text": "REFERENCE: the following are VALID VCP setups."})
        for p in positives:
            parts.append({"type": "image_url", "image_url": {"url": _encode_image(p)}})
    if negatives:
        parts.append({"type": "text", "text": "REFERENCE: the following are NOT valid VCP setups."})
        for p in negatives:
            parts.append({"type": "image_url", "image_url": {"url": _encode_image(p)}})
    return parts


def _strip_fences(text: str) -> str:
    return _JSON_FENCE_RE.sub("", text).strip()


def _parse_response(text: str) -> dict[str, Any]:
    cleaned = _strip_fences(text)
    # Some models leak a leading "json" word or trailing prose; isolate the outermost {...}.
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"No JSON object found in response: {text[:200]!r}")
    return json.loads(cleaned[start : end + 1])


@dataclass
class GradeRequest:
    symbol: str
    chart_path: Path
    close: float
    high_52w: float
    tightness_10d: float | None = None  # (high10 - low10) / close
    vol_10d_avg: float | None = None
    vol_50d_avg: float | None = None


def _format_features(req: GradeRequest) -> str:
    """Render the precomputed features the prompt promises ('trust these numbers')."""
    if req.close <= 0 and req.high_52w <= 0:
        return ""  # calibration mode: no DB row, skip the block entirely.
    lines = [
        "PRECOMPUTED FEATURES (measured from raw OHLCV — trust these over visual estimates):",
        f"  latest_close: {req.close:.2f}",
        f"  high_52w (pivot proxy): {req.high_52w:.2f}",
        f"  pct_below_52w_high: {(1 - req.close / req.high_52w) * 100:.2f}%",
    ]
    if req.tightness_10d is not None:
        lines.append(f"  range_10d_pct (high10-low10)/close: {req.tightness_10d * 100:.2f}%")
    if req.vol_10d_avg is not None and req.vol_50d_avg:
        lines.append(
            f"  volume_10d_avg / volume_50d_avg: {req.vol_10d_avg / req.vol_50d_avg:.2f} "
            f"(<1.0 means volume contracting)"
        )
    return "\n".join(lines)


async def _grade_one(
    req: GradeRequest,
    *,
    settings: Settings,
    few_shot: list[dict[str, Any]],
    failures_log: Path,
) -> dict[str, Any] | None:
    user_parts: list[dict[str, Any]] = list(few_shot)
    feature_block = _format_features(req)
    header = f"Grade this chart. Symbol: {req.symbol}."
    user_text = f"{header}\n\n{feature_block}\n\nRespond with the JSON object only." if feature_block else f"{header}\n\nRespond with the JSON object only."
    user_parts.append({"type": "text", "text": user_text})
    user_parts.append({"type": "image_url", "image_url": {"url": _encode_image(req.chart_path)}})

    messages = [
        {"role": "system", "content": system_prompt()},
        {"role": "user", "content": user_parts},
    ]

    last_err: Exception | None = None
    for attempt in range(2):
        try:
            response = await asyncio.wait_for(
                litellm.acompletion(model=settings.vcp_model, messages=messages, temperature=0),
                timeout=settings.grade_timeout_s,
            )
            text = response["choices"][0]["message"]["content"]
            parsed = _parse_response(text)
            parsed.setdefault("symbol", req.symbol)
            return parsed
        except (json.JSONDecodeError, ValueError) as e:
            last_err = e
            # Reinforce the JSON-only instruction on the retry.
            messages[0] = {"role": "system", "content": system_prompt() + "\n\nReturn JSON only. No prose."}
            continue
        except Exception as e:  # network, timeout, provider error
            last_err = e
            break

    failures_log.parent.mkdir(parents=True, exist_ok=True)
    with failures_log.open("a") as fh:
        fh.write(f"{req.symbol}\t{type(last_err).__name__}\t{last_err}\n")
    logger.warning("grade failed for %s: %s", req.symbol, last_err)
    return None


async def grade_all(
    requests: list[GradeRequest],
    *,
    out_dir: Path,
    date_str: str,
    settings: Settings | None = None,
    include_few_shot: bool = True,
) -> list[dict[str, Any]]:
    """Grade every chart concurrently (bounded by settings.concurrency)."""
    settings = settings or get_settings()
    failures_log = out_dir / f"grade_failures_{date_str}.log"
    few_shot = _load_examples(settings.examples_dir) if include_few_shot else []
    if include_few_shot and not few_shot:
        logger.warning("no reference images in %s — grading without few-shot context", settings.examples_dir)

    sem = asyncio.Semaphore(settings.concurrency)

    async def _bounded(req: GradeRequest) -> dict[str, Any] | None:
        async with sem:
            return await _grade_one(req, settings=settings, few_shot=few_shot, failures_log=failures_log)

    results = await asyncio.gather(*[_bounded(r) for r in requests])
    return [r for r in results if r is not None]


def grade_all_sync(requests: list[GradeRequest], **kwargs: Any) -> list[dict[str, Any]]:
    return asyncio.run(grade_all(requests, **kwargs))
