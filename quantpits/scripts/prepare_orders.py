"""Prepare quantities from saved candidates and explicit cash-open inputs."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from quantpits.order.preparation import prepare_orders
from quantpits.order.persistence import write_derived_json
from quantpits.utils.workspace import WorkspaceContext


def _read(ctx, value):
    path = Path(value).expanduser()
    path = (path if path.is_absolute() else ctx.root / path).resolve()
    try:
        relative = path.relative_to(ctx.root)
    except ValueError as exc:
        raise ValueError("preparation input must stay inside workspace") from exc
    raw = path.read_bytes()
    import hashlib
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("preparation input JSON root must be an object: %s" % relative)
    return value, dict(path=relative.as_posix(), sha256=hashlib.sha256(raw).hexdigest())


def main(argv=None):
    parser = argparse.ArgumentParser(description="从完整候选和当前现金／开盘观察准备数量，不执行交易")
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--observation", required=True)
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--output-dir", help="workspace/output 内的新结果目录，写 order_preparation.json")
    output.add_argument("--preview", action="store_true", help="只输出 JSON，零写入（默认）")
    args = parser.parse_args(argv)
    ctx = WorkspaceContext.from_root(args.workspace)
    try:
        candidates, candidate_ref = _read(ctx, args.candidates)
        observation, observation_ref = _read(ctx, args.observation)
        if candidates.get("workspace") != str(ctx.root):
            raise ValueError("candidate workspace differs from explicit workspace")
        result = prepare_orders(candidates, observation)
        result["inputs"] = dict(candidates=candidate_ref, observation=observation_ref)
        if args.output_dir is not None:
            status = write_derived_json(ctx, Path(args.output_dir) / "order_preparation.json", result)
            print(json.dumps(dict(output_status=status, result=result), ensure_ascii=False, indent=2))
        else:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print("prepare_orders: %s" % exc, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
