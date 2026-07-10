"""Atomic artifact persistence for order generation."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from quantpits.order.opinions import ModelOpinionsResult, to_json_payload
from quantpits.runtime import OutputRef
from quantpits.utils.workspace import WorkspaceContext


@dataclass(frozen=True)
class OrderArtifactPaths:
    opinion_csv: Path
    opinion_json: Path
    sell_csv: Path
    buy_csv: Path


@dataclass(frozen=True)
class OrderArtifactLedger:
    opinion_csv: str | None = None
    opinion_json: str | None = None
    sell_csv: str | None = None
    buy_csv: str | None = None
    outputs: tuple[OutputRef, ...] = ()


@dataclass(frozen=True)
class OrderPersistenceRequest:
    ctx: WorkspaceContext
    output_dir: Path
    trade_date: str
    source_label: str
    sell_orders: tuple[dict, ...]
    buy_orders: tuple[dict, ...]
    opinions: ModelOpinionsResult | None


class OrderPersistenceError(RuntimeError):
    def __init__(self, message: str, *, committed_outputs: tuple[OutputRef, ...] = ()):
        super().__init__(message)
        self.committed_outputs = committed_outputs


def build_order_artifact_paths(output_dir: str | Path, trade_date: str, source_label: str) -> OrderArtifactPaths:
    root = Path(output_dir)
    return OrderArtifactPaths(
        opinion_csv=root / f"model_opinions_{trade_date}.csv",
        opinion_json=root / f"model_opinions_{trade_date}.json",
        sell_csv=root / f"sell_suggestion_{source_label}_{trade_date}.csv",
        buy_csv=root / f"buy_suggestion_{source_label}_{trade_date}.csv",
    )


def display_path(ctx: WorkspaceContext, path: Path) -> str:
    try:
        return path.resolve().relative_to(ctx.root).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _atomic_write(path: Path, writer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temp = Path(temp_name)
    try:
        writer(temp)
        with temp.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def atomic_write_csv(dataframe: Any, path: Path, *, index: bool) -> None:
    _atomic_write(path, lambda temp: dataframe.to_csv(temp, index=index))


def atomic_write_json(payload: dict, path: Path) -> None:
    def writer(temp: Path) -> None:
        with temp.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=4, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

    _atomic_write(path, writer)


def persist_order_artifacts(request: OrderPersistenceRequest) -> OrderArtifactLedger:
    import pandas as pd

    paths = build_order_artifact_paths(request.output_dir, request.trade_date, request.source_label)
    committed: list[OutputRef] = []
    values: dict[str, str | None] = {"opinion_csv": None, "opinion_json": None, "sell_csv": None, "buy_csv": None}

    def commit(field: str, path: Path, *, kind: str = "report") -> None:
        shown = display_path(request.ctx, path)
        values[field] = shown
        committed.append(OutputRef(shown, kind=kind, overwrite=True))

    try:
        if request.opinions is not None:
            atomic_write_csv(request.opinions.dataframe, paths.opinion_csv, index=True)
            commit("opinion_csv", paths.opinion_csv)
            atomic_write_json(to_json_payload(request.opinions, trade_date=request.trade_date), paths.opinion_json)
            commit("opinion_json", paths.opinion_json)
        if request.sell_orders:
            atomic_write_csv(pd.DataFrame(request.sell_orders), paths.sell_csv, index=False)
            commit("sell_csv", paths.sell_csv)
        if request.buy_orders:
            atomic_write_csv(pd.DataFrame(request.buy_orders), paths.buy_csv, index=False)
            commit("buy_csv", paths.buy_csv)
    except Exception as exc:
        raise OrderPersistenceError(str(exc), committed_outputs=tuple(committed)) from exc
    return OrderArtifactLedger(outputs=tuple(committed), **values)
