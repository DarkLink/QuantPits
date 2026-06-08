"""
RunContext — 统一管理 Ensemble 搜索流水线的输出路径。

每次运行按 {script_name}_{anchor_date}/ 创建独立子目录，
内部分为 is/ 和 oos/ 两层，确保产出物结构清晰、可整目录归档。

使用方式:
    ctx = RunContext(base_dir="output/ensemble_runs",
                     script_name="brute_force",
                     anchor_date="2026-04-03")
    ctx.ensure_dirs()
    df.to_csv(ctx.is_path("results.csv"))
"""

import os
from dataclasses import dataclass, field


@dataclass
class RunContext:
    """管理一次 Ensemble 搜索运行的目录结构。"""

    base_dir: str       # e.g. "output/ensemble_runs"
    script_name: str    # e.g. "brute_force", "brute_force_fast", "minentropy"
    anchor_date: str    # e.g. "2026-04-03"
    run_label: str = "" # optional label to disambiguate same-date runs

    # ------------------------------------------------------------------
    # 核心路径属性
    # ------------------------------------------------------------------
    @property
    def _dir_name(self) -> str:
        """目录名: {script_name}_{anchor_date}[_{run_label}]"""
        base = f"{self.script_name}_{self.anchor_date}"
        if self.run_label:
            base += f"_{self.run_label}"
        return base

    @property
    def run_dir(self) -> str:
        """运行根目录: base_dir/{_dir_name}/"""
        return os.path.join(self.base_dir, self._dir_name)

    @property
    def is_dir(self) -> str:
        """IS (In-Sample) 产出目录"""
        return os.path.join(self.run_dir, "is")

    @property
    def oos_dir(self) -> str:
        """OOS (Out-of-Sample) 产出目录"""
        return os.path.join(self.run_dir, "oos")

    # ------------------------------------------------------------------
    # 便捷路径生成
    # ------------------------------------------------------------------
    def is_path(self, filename: str) -> str:
        """返回 IS 目录下的文件路径"""
        return os.path.join(self.is_dir, filename)

    def oos_path(self, filename: str) -> str:
        """返回 OOS 目录下的文件路径"""
        return os.path.join(self.oos_dir, filename)

    def run_path(self, filename: str) -> str:
        """返回 run 根目录下的文件路径"""
        return os.path.join(self.run_dir, filename)

    # ------------------------------------------------------------------
    # 目录创建
    # ------------------------------------------------------------------
    def ensure_dirs(self):
        """创建所有必要的目录 (幂等)"""
        for d in (self.run_dir, self.is_dir, self.oos_dir):
            os.makedirs(d, exist_ok=True)

    # ------------------------------------------------------------------
    # 从 metadata JSON 反推 RunContext
    # ------------------------------------------------------------------
    @classmethod
    def from_metadata(cls, metadata_path: str, base_dir: str = None) -> "RunContext":
        """
        从 metadata JSON 文件路径反推 RunContext。

        支持两种目录结构:
        1. 新结构: base_dir/{script}_{date}/run_metadata.json
           → 直接从目录名解析
        2. 旧结构: output/brute_force/run_metadata_{date}.json
           → 兼容模式，fallback 到 metadata 所在目录
        """
        import json
        with open(metadata_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        anchor_date = meta["anchor_date"]
        script_name = meta.get("script_used", "brute_force_ensemble")
        # 标准化 script_name (去掉 _ensemble 后缀)
        script_name = _normalize_script_name(script_name)

        metadata_dir = os.path.dirname(os.path.abspath(metadata_path))
        dir_basename = os.path.basename(metadata_dir)

        # 检测是新结构 (目录名 = script_date[_label]) 还是旧结构
        run_label = ""
        expected_dir_name = f"{script_name}_{anchor_date}"
        if dir_basename == expected_dir_name or dir_basename.startswith(expected_dir_name + "_"):
            # 新结构（含 --run-label 变体）：parent of metadata_dir 就是 base_dir
            resolved_base = os.path.dirname(metadata_dir)
            # 从目录名反推 run_label
            if dir_basename != expected_dir_name:
                run_label = dir_basename[len(expected_dir_name) + 1:]
            else:
                run_label = ""
        else:
            # 旧结构兼容：用 metadata 所在目录做 run_dir
            # 此时不再创建 is/oos 子目录，回退到平铺模式
            if base_dir:
                resolved_base = base_dir
            else:
                resolved_base = metadata_dir

        ctx = cls(
            base_dir=resolved_base,
            script_name=script_name,
            anchor_date=anchor_date,
            run_label=run_label,
        )

        # 如果旧结构，检查实际 run_dir 是否就是 metadata 所在目录
        if not os.path.isdir(ctx.run_dir):
            # 旧结构 fallback: 直接使用 metadata_dir 作为 run_dir
            # 创建一个 legacy 模式的 context
            ctx = _LegacyRunContext(
                base_dir=resolved_base,
                script_name=script_name,
                anchor_date=anchor_date,
                legacy_dir=metadata_dir,
            )

        return ctx

    def __repr__(self):
        return (
            f"RunContext(run_dir={self.run_dir!r}, "
            f"script={self.script_name!r}, date={self.anchor_date!r})"
        )


@dataclass
class _LegacyRunContext(RunContext):
    """旧目录结构的兼容模式 — 所有文件平铺在同一目录。"""

    legacy_dir: str = ""

    @property
    def run_dir(self) -> str:
        return self.legacy_dir

    @property
    def is_dir(self) -> str:
        return self.legacy_dir

    @property
    def oos_dir(self) -> str:
        return self.legacy_dir

    def ensure_dirs(self):
        os.makedirs(self.legacy_dir, exist_ok=True)


def _normalize_script_name(raw: str) -> str:
    """将脚本标识标准化为简短名称。

    Examples:
        "brute_force_ensemble" -> "brute_force"
        "brute_force_fast"     -> "brute_force_fast"
        "minentropy"           -> "minentropy"
    """
    mapping = {
        "brute_force_ensemble": "brute_force",
        "brute_force_fast": "brute_force_fast",
        "minentropy": "minentropy",
    }
    return mapping.get(raw, raw)
