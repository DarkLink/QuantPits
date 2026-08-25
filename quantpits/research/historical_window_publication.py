"""Create-new publication for one complete retrospective shadow window."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from quantpits.research.historical_cycle import (
    ARM_IDS,
    EVIDENCE_CLASS,
    _canonical_json,
    _deterministic_id,
    _directory_descriptor_identity,
    _revalidate_ranking,
    _root_identity,
    _try_root_identity,
)
from quantpits.research.historical_window import (
    HistoricalShadowWindowReplay,
    HistoricalWindowResult,
    compact_window_summary,
    revalidate_historical_window_result,
)
from quantpits.research.replay import revalidate_replay_result


BUNDLE_KIND = "B3B_RETROSPECTIVE_WINDOW_V1"
_PUBLICATION_AUTHORITY = object()


class HistoricalWindowPublicationContractError(ValueError):
    """The requested publication representation is invalid."""


class HistoricalWindowPublicationInputError(RuntimeError):
    """The publication source or namespace could not be observed exactly."""


def _raw_digest(data: bytes) -> Dict[str, str]:
    return {"algorithm": "sha256", "value": hashlib.sha256(data).hexdigest()}


def _full_digest(data: bytes) -> Dict[str, Any]:
    return {
        "algorithm": "sha256", "domain": "raw_bytes",
        "value": hashlib.sha256(data).hexdigest(), "size_bytes": len(data),
    }


def _valid_digest(value: Any, domain: str) -> bool:
    return bool(
        isinstance(value, Mapping)
        and set(value) == {"algorithm", "domain", "value", "size_bytes"}
        and value.get("algorithm") == "sha256"
        and value.get("domain") == domain
        and isinstance(value.get("value"), str)
        and len(value["value"]) == 64
        and all(character in "0123456789abcdef" for character in value["value"])
        and not isinstance(value.get("size_bytes"), bool)
        and isinstance(value.get("size_bytes"), int)
        and value["size_bytes"] >= 0
    )


@dataclass(frozen=True, init=False)
class WindowPublicationReceipt:
    operation_id: str
    status: str
    reason_code: str
    did_write: bool
    result_digest: Mapping[str, Any]
    manifest_digest: Optional[Mapping[str, Any]]
    member_count: int
    output_root_identity: Optional[Tuple[int, int, int, int, int]]
    root_parent_identity_before: Tuple[int, int, int, int, int]
    root_parent_identity_after: Optional[Tuple[int, int, int, int, int]]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if args or kwargs.pop("_authority", None) is not _PUBLICATION_AUTHORITY:
            raise HistoricalWindowPublicationContractError(
                "window publication receipts are writer-owned"
            )
        expected = {
            "operation_id", "status", "reason_code", "did_write", "result_digest",
            "manifest_digest", "member_count", "output_root_identity",
            "root_parent_identity_before", "root_parent_identity_after",
        }
        if set(kwargs) != expected:
            raise HistoricalWindowPublicationContractError("receipt fields are not exact")
        for key, value in kwargs.items():
            object.__setattr__(self, key, value)
        if (
            not isinstance(self.operation_id, str) or len(self.operation_id) != 64
            or any(character not in "0123456789abcdef" for character in self.operation_id)
            or not isinstance(self.did_write, bool)
            or not _valid_digest(self.result_digest, "canonical_json")
        ):
            raise HistoricalWindowPublicationContractError("receipt identity is invalid")
        for name in ("root_parent_identity_before", "root_parent_identity_after", "output_root_identity"):
            identity = getattr(self, name)
            if identity is not None and (
                not isinstance(identity, tuple) or len(identity) != 5
                or any(isinstance(item, bool) or not isinstance(item, int) for item in identity)
            ):
                raise HistoricalWindowPublicationContractError("receipt filesystem identity is invalid")
        if self.root_parent_identity_before is None:
            raise HistoricalWindowPublicationContractError("receipt parent identity is required")
        if self.manifest_digest is not None and not _valid_digest(self.manifest_digest, "raw_bytes"):
            raise HistoricalWindowPublicationContractError("receipt manifest digest is invalid")
        if self.status not in ("COMMITTED", "CONFLICT", "UNCERTAIN"):
            raise HistoricalWindowPublicationContractError("receipt status is invalid")
        required_reason = {
            "COMMITTED": "COMMITTED",
            "CONFLICT": "OUTPUT_ALREADY_EXISTS",
            "UNCERTAIN": "PUBLICATION_POSTCONDITION_UNCERTAIN",
        }[self.status]
        if self.reason_code != required_reason:
            raise HistoricalWindowPublicationContractError("receipt reason is inconsistent")
        if isinstance(self.member_count, bool) or not isinstance(self.member_count, int) or self.member_count < 0:
            raise HistoricalWindowPublicationContractError("receipt member count is invalid")
        if self.status == "COMMITTED":
            requested = self.member_count in (24, 29, 34)
            valid = (
                self.did_write is True and requested and self.manifest_digest is not None
                and self.output_root_identity is not None
                and self.root_parent_identity_after == self.root_parent_identity_before
            )
        elif self.status == "CONFLICT":
            valid = (
                self.did_write is False and self.manifest_digest is None
                and self.member_count == 0 and self.output_root_identity is None
                and self.root_parent_identity_after == self.root_parent_identity_before
            )
        else:
            valid = (
                self.manifest_digest is None
                and self.member_count <= 34
                and not (
                    self.did_write is False
                    and (self.member_count != 0 or self.output_root_identity is not None)
                )
            )
        if not valid:
            raise HistoricalWindowPublicationContractError("receipt cross-fields are inconsistent")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "operation_id": self.operation_id,
            "status": self.status,
            "reason_code": self.reason_code,
            "did_write": self.did_write,
            "result_digest": dict(self.result_digest),
            "manifest_digest": None if self.manifest_digest is None else dict(self.manifest_digest),
            "member_count": self.member_count,
            "output_root_identity": (
                None if self.output_root_identity is None else list(self.output_root_identity)
            ),
            "root_parent_identity_before": list(self.root_parent_identity_before),
            "root_parent_identity_after": (
                None if self.root_parent_identity_after is None
                else list(self.root_parent_identity_after)
            ),
        }


def _metrics_csv(summary: Mapping[str, Any]) -> bytes:
    fields = (
        "arm_id", "terminal_normalized_nav", "max_drawdown",
        "cumulative_cost_rate", "mean_holding_overlap_ratio_with_champion",
    )
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for item in summary["arms"]:
        writer.writerow({field: item[field] for field in fields})
    return stream.getvalue().encode("utf-8")


def _report(summary: Mapping[str, Any]) -> bytes:
    lines = ["# Stage B3B Retrospective Window Bundle", ""]
    lines.extend("- %s" % warning for warning in summary["warnings"])
    lines.extend([
        "", "- Status: `%s`" % summary["status"].upper(),
        "- Requested / terminal cycles: `%d` / `%d`" % (
            summary["requested_cycle_count"], summary["terminal_cycle_count"],
        ),
        "- Complete / blocked arm-cycles: `%d` / `%d`" % (
            summary["complete_cycle_arm_count"], summary["blocked_cycle_arm_count"],
        ),
        "- Chain continuity checked: `%s`" % str(summary["chain_continuity_checked"]).lower(),
        "- Metric capability: `%s`" % str(summary["metric_capability"]).lower(),
        "- Result digest: `%s`" % summary["window_digest"]["value"], "",
        "| arm_id | terminal_normalized_nav | max_drawdown | cumulative_cost_rate | mean_holding_overlap_ratio_with_champion |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    for item in summary["arms"]:
        lines.append("| %s | %s | %s | %s | %s |" % tuple(
            "" if item[field] is None else item[field] for field in (
                "arm_id", "terminal_normalized_nav", "max_drawdown",
                "cumulative_cost_rate", "mean_holding_overlap_ratio_with_champion",
            )
        ))
    lines.extend(["", "No arm was selected; no promotion capability.", ""])
    return "\n".join(lines).encode("utf-8")


def build_historical_window_artifacts(
    runner: HistoricalShadowWindowReplay, result: HistoricalWindowResult,
) -> Mapping[str, bytes]:
    """Build and validate the exact pre-manifest member map without writing."""
    if not isinstance(runner, HistoricalShadowWindowReplay):
        raise HistoricalWindowPublicationContractError("publication runner is foreign")
    try:
        snapshot = revalidate_historical_window_result(result)
        stage_a = revalidate_replay_result(runner._stage_a)
    except (KeyboardInterrupt, SystemExit, GeneratorExit):
        raise
    except Exception as exc:
        raise HistoricalWindowPublicationContractError("publication source revalidation failed") from exc
    if (
        snapshot["status"] != "COMPLETE"
        or snapshot["metric_capability"] is not True
        or snapshot["chain_continuity_checked"] is not True
        or snapshot["prospective_claim"] is not False
        or snapshot["promotion_capability"] is not False
    ):
        raise HistoricalWindowPublicationContractError("window has no publication capability")
    anchors = tuple(snapshot["requested_anchor_ids"])
    if (
        anchors != tuple(runner._anchors)
        or tuple(snapshot["requested_arm_ids"]) != ARM_IDS
        or snapshot["stage_a_result_digest"] != stage_a["result_digest"]
        or snapshot["window_identity"] != dict(runner._window_identity)
        or snapshot["engine_source_fingerprint"] != dict(runner._source)
        or snapshot["environment_fingerprint"] != dict(runner._environment)
        or snapshot["profile"] != runner._profile.public_payload()
        or snapshot["source_models"] != stage_a["source_models"]
        or snapshot["universe_identity"] != stage_a["universe_identity"]
        or snapshot["calendar_identity"] != stage_a["calendar_identity"]
        or snapshot["source_to_materialization_relation"] != stage_a["source_to_materialization_relation"]
    ):
        raise HistoricalWindowPublicationContractError("runner and result identities are foreign")
    summary = compact_window_summary(snapshot)
    members: Dict[str, bytes] = {
        "result.json": snapshot.to_canonical_json_bytes(),
        "summary.json": _canonical_json(summary),
        "metrics.csv": _metrics_csv(summary),
        "report.md": _report(summary),
    }
    for anchor, terminal_cycle in zip(anchors, snapshot["terminal_cycles"]):
        cycle = runner._cycle(anchor)
        terminal = {item["arm_id"]: item for item in terminal_cycle["terminal_arms"]}
        current_rankings = stage_a["rankings"].get(anchor)
        if (
            tuple(terminal) != ARM_IDS or tuple(cycle._rankings) != ARM_IDS
            or not isinstance(current_rankings, Mapping) or tuple(current_rankings) != ARM_IDS
        ):
            raise HistoricalWindowPublicationContractError("ranking member set is not exact")
        for arm in ARM_IDS:
            try:
                current = _revalidate_ranking(current_rankings[arm])
                carried = _revalidate_ranking(cycle._rankings[arm])
            except (KeyboardInterrupt, SystemExit, GeneratorExit):
                raise
            except Exception as exc:
                raise HistoricalWindowPublicationContractError("ranking revalidation failed") from exc
            ranking_bytes = current.to_csv_bytes()
            if carried.to_csv_bytes() != ranking_bytes:
                raise HistoricalWindowPublicationContractError("runner ranking snapshot drifted")
            if hashlib.sha256(ranking_bytes).hexdigest() != terminal[arm]["ranking_digest"]:
                raise HistoricalWindowPublicationContractError("ranking digest join failed")
            logical = "rankings/%s/%s.csv" % (anchor, arm)
            if Path(logical).as_posix() != logical or logical in members:
                raise HistoricalWindowPublicationContractError("artifact path is not canonical and unique")
            members[logical] = ranking_bytes
    expected = 4 + len(anchors) * len(ARM_IDS)
    if len(members) != expected:
        raise HistoricalWindowPublicationContractError("artifact count is not exact")
    return MappingProxyType({key: members[key] for key in sorted(members)})


def _write_file(directory_fd: int, name: str, data: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(name, flags, 0o600, dir_fd=directory_fd)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(data)
            handle.flush()
            os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_exact_regular(root: Path, logical: str) -> bytes:
    current = root
    parts = Path(logical).parts
    for part in parts:
        current = current / part
        info = os.lstat(str(current))
        if stat.S_ISLNK(info.st_mode):
            raise HistoricalWindowPublicationInputError("publication path became a symlink")
    before = os.lstat(str(current))
    if (
        not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
        or stat.S_IMODE(before.st_mode) != 0o600
    ):
        raise HistoricalWindowPublicationInputError("publication member is not private regular data")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(str(current), flags)
    try:
        opened = os.fstat(descriptor)
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after_open = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after = os.lstat(str(current))
    identity = lambda value: (value.st_dev, value.st_ino, value.st_mode, value.st_size, value.st_mtime_ns)
    if len({identity(before), identity(opened), identity(after_open), identity(after)}) != 1:
        raise HistoricalWindowPublicationInputError("publication member identity changed")
    return b"".join(chunks)


def _inventory(descriptor: int, expected: Sequence[str], logical: str) -> None:
    if tuple(sorted(os.listdir(descriptor))) != tuple(sorted(expected)):
        raise HistoricalWindowPublicationInputError("publication inventory is not exact: %s" % logical)


def write_historical_window_output(
    runner: HistoricalShadowWindowReplay,
    result: HistoricalWindowResult,
    output_root: Path,
) -> WindowPublicationReceipt:
    """Publish one exact bundle, granting COMMITTED only after final reobservation."""
    members = build_historical_window_artifacts(runner, result)
    snapshot = revalidate_historical_window_result(result)
    if members.get("result.json") != snapshot.to_canonical_json_bytes():
        raise HistoricalWindowPublicationContractError(
            "publication result changed after artifact freeze"
        )
    try:
        root = Path(output_root)
    except (TypeError, ValueError) as exc:
        raise HistoricalWindowPublicationContractError("output root is not a path") from exc
    parent = Path("/tmp").resolve(strict=True)
    if (
        not root.is_absolute() or root.parent != parent or root != parent / root.name
        or not root.name or root.name in (".", "..") or "\0" in root.name
    ):
        raise HistoricalWindowPublicationContractError(
            "output root must be one direct child of physical /tmp"
        )
    operation_id = _deterministic_id("B3B_PUBLICATION_V1", {
        "result_digest": snapshot["result_digest"]["value"], "output_name": root.name,
    })
    parent_before = _root_identity(parent)
    descriptors = []
    directory_fds: Dict[str, int] = {}
    directory_identities: Dict[str, Tuple[int, int, int, int, int]] = {}
    did_write = False
    observed_members = 0
    root_identity = None
    try:
        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        parent_fd = os.open(str(parent), directory_flags)
        descriptors.append(parent_fd)
        if _directory_descriptor_identity(parent_fd) != parent_before or _root_identity(parent) != parent_before:
            raise HistoricalWindowPublicationInputError("publication parent identity drifted")
        try:
            os.mkdir(root.name, 0o700, dir_fd=parent_fd)
        except FileExistsError:
            parent_after = _try_root_identity(parent)
            if parent_after == parent_before and _directory_descriptor_identity(parent_fd) == parent_before:
                return WindowPublicationReceipt(
                    _authority=_PUBLICATION_AUTHORITY, operation_id=operation_id,
                    status="CONFLICT", reason_code="OUTPUT_ALREADY_EXISTS", did_write=False,
                    result_digest=snapshot["result_digest"], manifest_digest=None, member_count=0,
                    output_root_identity=None, root_parent_identity_before=parent_before,
                    root_parent_identity_after=parent_after,
                )
            raise HistoricalWindowPublicationInputError("publication parent identity drifted at create")
        did_write = True
        root_fd = os.open(root.name, directory_flags, dir_fd=parent_fd)
        descriptors.append(root_fd)
        os.fchmod(root_fd, 0o700)
        root_identity = _directory_descriptor_identity(root_fd)
        if _root_identity(root) != root_identity:
            raise HistoricalWindowPublicationInputError("publication root identity drifted after create")
        directory_fds["."] = root_fd
        directory_identities["."] = root_identity
        os.mkdir("rankings", 0o700, dir_fd=root_fd)
        rankings_fd = os.open("rankings", directory_flags, dir_fd=root_fd)
        descriptors.append(rankings_fd)
        os.fchmod(rankings_fd, 0o700)
        directory_fds["rankings"] = rankings_fd
        directory_identities["rankings"] = _directory_descriptor_identity(rankings_fd)
        for anchor in snapshot["requested_anchor_ids"]:
            os.mkdir(anchor, 0o700, dir_fd=rankings_fd)
            anchor_fd = os.open(anchor, directory_flags, dir_fd=rankings_fd)
            descriptors.append(anchor_fd)
            os.fchmod(anchor_fd, 0o700)
            logical = "rankings/" + anchor
            directory_fds[logical] = anchor_fd
            directory_identities[logical] = _directory_descriptor_identity(anchor_fd)
        manifest_rows = []
        for logical, data in members.items():
            parts = Path(logical).parts
            if len(parts) == 1:
                target_fd = root_fd
            elif len(parts) == 3 and "rankings/%s" % parts[1] in directory_fds:
                target_fd = directory_fds["rankings/%s" % parts[1]]
            else:
                raise HistoricalWindowPublicationContractError("member escaped frozen layout")
            _write_file(target_fd, parts[-1], data)
            if _read_exact_regular(root, logical) != data:
                raise HistoricalWindowPublicationInputError("publication member bytes changed")
            observed_members += 1
            manifest_rows.append({
                "logical_path": logical, "size_bytes": len(data), "digest": _raw_digest(data),
            })
        for descriptor in reversed(descriptors[1:]):
            os.fsync(descriptor)
        manifest_payload = {
            "schema_version": 1, "bundle_kind": BUNDLE_KIND,
            "evidence_class": EVIDENCE_CLASS, "prospective_claim": False,
            "promotion_capability": False,
            "window_result_digest": snapshot["result_digest"],
            "requested_anchor_ids": snapshot["requested_anchor_ids"],
            "requested_arm_ids": list(ARM_IDS), "member_count": len(manifest_rows),
            "members": manifest_rows,
        }
        manifest = _canonical_json(manifest_payload)
        _write_file(root_fd, "output_manifest.json", manifest)
        if _read_exact_regular(root, "output_manifest.json") != manifest:
            raise HistoricalWindowPublicationInputError("manifest bytes changed")
        for item in manifest_rows:
            data = _read_exact_regular(root, item["logical_path"])
            if len(data) != item["size_bytes"] or _raw_digest(data) != item["digest"]:
                raise HistoricalWindowPublicationInputError("final member digest changed")
        root_files = [Path(name).name for name in members if len(Path(name).parts) == 1]
        _inventory(root_fd, tuple(root_files) + ("rankings", "output_manifest.json"), ".")
        anchors = tuple(snapshot["requested_anchor_ids"])
        _inventory(rankings_fd, anchors, "rankings")
        for anchor in anchors:
            _inventory(directory_fds["rankings/" + anchor], tuple(arm + ".csv" for arm in ARM_IDS), "rankings/" + anchor)
        for logical, identity in directory_identities.items():
            path = root if logical == "." else root.joinpath(*Path(logical).parts)
            if _root_identity(path) != identity or stat.S_IMODE(os.lstat(str(path)).st_mode) != 0o700:
                raise HistoricalWindowPublicationInputError("publication directory identity changed")
        if (
            _directory_descriptor_identity(parent_fd) != parent_before
            or _root_identity(parent) != parent_before
            or _directory_descriptor_identity(root_fd) != root_identity
            or _root_identity(root) != root_identity
        ):
            raise HistoricalWindowPublicationInputError("publication namespace continuity failed")
        os.fsync(root_fd)
        os.fsync(parent_fd)
        if _root_identity(root) != root_identity or _root_identity(parent) != parent_before:
            raise HistoricalWindowPublicationInputError("publication final namespace changed")
        return WindowPublicationReceipt(
            _authority=_PUBLICATION_AUTHORITY, operation_id=operation_id,
            status="COMMITTED", reason_code="COMMITTED", did_write=True,
            result_digest=snapshot["result_digest"], manifest_digest=_full_digest(manifest),
            member_count=len(manifest_rows), output_root_identity=root_identity,
            root_parent_identity_before=parent_before,
            root_parent_identity_after=_try_root_identity(parent),
        )
    except (KeyboardInterrupt, SystemExit, GeneratorExit):
        raise
    except Exception:
        current = _try_root_identity(root)
        return WindowPublicationReceipt(
            _authority=_PUBLICATION_AUTHORITY, operation_id=operation_id,
            status="UNCERTAIN", reason_code="PUBLICATION_POSTCONDITION_UNCERTAIN",
            did_write=did_write, result_digest=snapshot["result_digest"],
            manifest_digest=None, member_count=observed_members,
            output_root_identity=root_identity if current == root_identity else None,
            root_parent_identity_before=parent_before,
            root_parent_identity_after=_try_root_identity(parent),
        )
    finally:
        for descriptor in reversed(descriptors):
            try:
                os.close(descriptor)
            except OSError:
                pass


__all__ = [
    "BUNDLE_KIND", "HistoricalWindowPublicationContractError",
    "HistoricalWindowPublicationInputError", "WindowPublicationReceipt",
    "build_historical_window_artifacts", "write_historical_window_output",
]
