import os
import json
import uuid
import time
from datetime import datetime

class OperatorLog:
    """
    Operator Log — 操作日志工具
    
    记录核心脚本的每次运行到 data/operator_log.jsonl。
    格式: 一行一条 JSON，追加写入，永不覆盖。
    """
    def __init__(self, script_name: str, args: list = None, 
                 tags: list = None, notes: str = "",
                 log_file: str = None):
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

    def __enter__(self) -> 'OperatorLog':
        """记录开始时间"""
        self.timestamp_start = datetime.now()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """记录结束时间、耗时、异常信息，写入 JSONL"""
        try:
            timestamp_end = datetime.now()
            duration = (timestamp_end - self.timestamp_start).total_seconds()
            
            exception_info = None
            if exc_type:
                exception_info = {
                    "type": exc_type.__name__,
                    "value": str(exc_val)
                }

            log_id = f"{self.timestamp_start.strftime('%Y%m%d_%H%M%S')}_{self.script_name}_{uuid.uuid4().hex[:4]}"
            
            entry = {
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
                "result_summary": self.result_summary,
                "exception": exception_info
            }

            self._write_entry(entry)
        except Exception as e:
            # 日志系统本身的错误只 print warning，不影响主流程
            print(f"⚠️  OperatorLog Warning: Could not write log: {e}")
        
        # __exit__ 返回 False 表示异常不被吞掉，继续向上传递
        return False

    def _write_entry(self, entry: dict):
        """写入日志项，尝试保证写入的原子性（仅限单条记录）"""
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        
        # 对于 JSONL，我们直接追加。为了尽量减少部分写入的可能性，我们先生成完整的行
        line = json.dumps(entry) + "\n"
        with open(self.log_file, 'a') as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
