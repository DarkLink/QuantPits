import os
import json
import uuid
import hashlib
from datetime import datetime

class OperatorLog:
    """
    Operator Log — 操作日志工具
    
    记录核心脚本的每次运行到 data/operator_log.jsonl。
    格式: 一行一条 JSON，追加写入，永不覆盖。
    """
    def __init__(self, script_name: str, args: list = None,
                 tags: list = None, notes: str = "",
                 log_file: str = None,
                 run_id: str = None,
                 manifest_path: str = None,
                 plan_fingerprint: str = None,
                 transaction_id: str = None):
        """
        script_name: 脚本名，如 "static_train", "brute_force_fast"
        args:        CLI 参数列表（不含脚本路径）
        tags:        标签列表，如 ["test"], ["experiment", "playground"]
        notes:       自由文本备注
        log_file:    日志文件路径（默认从 env.ROOT_DIR 自动解析）
        """
        self.script_name = script_name
        self.args = args or []
        self.tags = tags or []
        self.notes = notes
        self.timestamp_start = None
        self.result_summary = {}
        self.source = "human"
        self.action_item_id = None
        self.run_id = run_id
        self.manifest_path = manifest_path
        self.plan_fingerprint = plan_fingerprint
        self.transaction_id = transaction_id
        self._committed = False
        self._committed_fingerprint = None
        # Resolve log_file lazily so module is importable without active workspace
        if log_file is not None:
            self.log_file = log_file
        else:
            from quantpits.utils.env import ROOT_DIR
            self.log_file = os.path.join(ROOT_DIR, 'data', 'operator_log.jsonl')

    def set_result(self, summary: dict):
        """设置运行结果摘要（可在运行中途更新）"""
        self.result_summary.update(summary)

    def set_source(self, source: str):
        """设置操作来源: 'human' | 'llm_critic' | 'scheduled'"""
        self.source = source

    def set_action_item_id(self, action_item_id: str):
        """关联到触发此次运行的 ActionItem ID（LLM 操作时使用）"""
        self.action_item_id = action_item_id

    def set_run_manifest(self, *, run_id: str, manifest_path: str):
        """关联运行 manifest，路径建议使用 workspace-relative path。"""
        self.run_id = run_id
        self.manifest_path = manifest_path

    def set_plan_fingerprint(self, fingerprint: str):
        """记录 dry-run plan 的稳定指纹。"""
        self.plan_fingerprint = fingerprint

    def __enter__(self) -> 'OperatorLog':
        """记录开始时间"""
        self.timestamp_start = datetime.now()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """记录结束时间、耗时、异常信息，写入 JSONL"""
        if self._committed:
            return False
        try:
            exception_info = None if not exc_type else {
                "type": exc_type.__name__, "value": str(exc_val),
            }
            self.commit(exception_info=exception_info, adopt_existing=False)
        except Exception as e:
            # 日志系统本身的错误只 print warning，不影响主流程
            print(f"⚠️  OperatorLog Warning: Could not write log: {e}")
        
        # __exit__ 返回 False 表示异常不被吞掉，继续向上传递
        return False

    def _build_entry(self, exception_info=None):
        timestamp_end = datetime.now()
        duration = (timestamp_end - self.timestamp_start).total_seconds()
        log_id = f"{self.timestamp_start.strftime('%Y%m%d_%H%M%S')}_{self.script_name}_{uuid.uuid4().hex[:4]}"
        return {
            "log_id": log_id,
            "timestamp_start": self.timestamp_start.isoformat(),
            "timestamp_end": timestamp_end.isoformat(),
            "duration_seconds": float(duration),
            "script": self.script_name,
            "args": self.args,
            "source": self.source,
            "tags": self.tags,
            "notes": self.notes,
            "action_item_id": self.action_item_id,
            "run_id": self.run_id,
            "manifest_path": self.manifest_path,
            "plan_fingerprint": self.plan_fingerprint,
            "transaction_id": self.transaction_id,
            "result_summary": self.result_summary,
            "exception": exception_info,
        }

    @staticmethod
    def _entry_fingerprint(entry):
        line = json.dumps(entry, sort_keys=True, separators=(",", ":")) + "\n"
        return hashlib.sha256(line.encode("utf-8")).hexdigest()

    def _find_durable_link(self):
        if not self.run_id or not self.transaction_id or not os.path.isfile(self.log_file):
            return None
        expected = (
            self.script_name, self.run_id, self.transaction_id, self.manifest_path,
            self.plan_fingerprint, self.result_summary.get("status"),
        )
        with open(self.log_file, "r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    entry = json.loads(line)
                except (TypeError, ValueError):
                    continue
                actual = (
                    entry.get("script"), entry.get("run_id"), entry.get("transaction_id"),
                    entry.get("manifest_path"), entry.get("plan_fingerprint"),
                    entry.get("result_summary", {}).get("status"),
                )
                if actual == expected and entry.get("exception") is None:
                    return self._entry_fingerprint(entry)
        return None

    def commit(self, *, exception_info=None, adopt_existing=True):
        """Durably append the current entry and expose failures to the caller.

        Training closure uses this explicit operation before acknowledging its
        operator-log step. Generic callers retain the best-effort context-exit
        behavior.
        """
        if self._committed:
            return self._committed_fingerprint
        if self.timestamp_start is None:
            raise RuntimeError("OperatorLog must be entered before commit")
        if adopt_existing:
            fingerprint = self._find_durable_link()
            if fingerprint:
                self._committed = True
                self._committed_fingerprint = fingerprint
                return fingerprint
        entry = self._build_entry(exception_info=exception_info)
        fingerprint = self._write_entry(entry)
        self._committed = True
        self._committed_fingerprint = fingerprint
        return fingerprint

    def _write_entry(self, entry: dict):
        """写入日志项，尝试保证写入的原子性（仅限单条记录）"""
        parent = os.path.dirname(self.log_file)
        existed = os.path.exists(self.log_file)
        os.makedirs(parent, exist_ok=True)
        
        # 对于 JSONL，我们直接追加。为了尽量减少部分写入的可能性，我们先生成完整的行
        line = json.dumps(entry) + "\n"
        with open(self.log_file, 'a', encoding="utf-8") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
        if not existed:
            directory_fd = os.open(parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        return self._entry_fingerprint(entry)
