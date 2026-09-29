from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from app.v12_paper_engine import PaperPosition, PaperPositionEngine


STATE_VERSION = 1


def save_engine_state(
    engine: PaperPositionEngine,
    path: str | Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "version": STATE_VERSION,
        "equity": float(engine.equity),
        "peak_equity": float(engine.peak_equity),
        "day": engine.day,
        "day_start_equity": float(engine.day_start_equity),
        "trades": engine.trades,
        "position": (
            asdict(engine.position)
            if engine.position is not None
            else None
        ),
    }

    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, default=str),
        encoding="utf-8",
    )
    tmp.replace(path)


def load_engine_state(
    engine: PaperPositionEngine,
    path: str | Path,
) -> bool:
    path = Path(path)

    if not path.exists():
        return False

    payload = json.loads(path.read_text(encoding="utf-8"))

    if payload.get("version") != STATE_VERSION:
        raise RuntimeError(
            f"Unsupported paper state version: "
            f"{payload.get('version')}"
        )

    engine.equity = float(payload["equity"])
    engine.peak_equity = float(payload["peak_equity"])
    engine.day = payload.get("day")
    engine.day_start_equity = float(payload["day_start_equity"])
    engine.trades = list(payload.get("trades", []))

    position = payload.get("position")

    if position is None:
        engine.position = None
    else:
        engine.position = PaperPosition(**position)

    engine._write_trades()
    return True


def save_shadow_taken(
    shadow_taken: dict[float, set[tuple[str, str]]],
    path: str | Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        str(threshold): sorted(
            [list(item) for item in entries]
        )
        for threshold, entries in shadow_taken.items()
    }

    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    tmp.replace(path)


def load_shadow_taken(
    path: str | Path,
    thresholds: list[float],
) -> dict[float, set[tuple[str, str]]]:
    path = Path(path)

    result = {
        float(threshold): set()
        for threshold in thresholds
    }

    if not path.exists():
        return result

    payload = json.loads(path.read_text(encoding="utf-8"))

    for threshold in thresholds:
        values = payload.get(str(threshold), [])
        result[float(threshold)] = {
            (str(item[0]), str(item[1]))
            for item in values
            if isinstance(item, list) and len(item) == 2
        }

    return result
