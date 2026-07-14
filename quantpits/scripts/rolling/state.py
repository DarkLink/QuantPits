"""
Rolling 训练状态管理模块

RollingState 管理训练进度的持久化，支持断点恢复 (--resume)。
状态以 JSON 格式保存，结构为 {window_idx: {model_name: record_id}}。
"""

import os
import json
from datetime import datetime


class RollingState:
    """Rolling 训练状态管理，支持断点恢复"""

    def __init__(self, state_file=None, readonly=False):
        from quantpits.utils.train_utils import ROLLING_STATE_FILE
        self.state_file = state_file or ROLLING_STATE_FILE
        self._state = self._load_without_lock() if readonly else self._load()

    def _load_without_lock(self):
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    content = f.read().strip()
                    if content:
                        return json.loads(content)
            except (json.JSONDecodeError, IOError):
                pass
        return self._empty()

    def _load(self):
        from quantpits.utils.train_utils import file_lock
        with file_lock(self.state_file):
            return self._load_without_lock()

    def _empty(self):
        return {
            'started_at': None,
            'rolling_config': {},
            'anchor_date': None,
            'training_method': 'slide',  # 'slide' or 'cpcv'
            'completed_windows': {},  # {window_idx_str: {model_name: record_id}}
            'current_window_idx': None,
            'current_model': None,
            'total_windows': 0,
        }

    def _save_without_lock(self):
        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
        with open(self.state_file, 'w') as f:
            json.dump(self._state, f, indent=4)

    def save(self):
        from quantpits.utils.train_utils import file_lock
        with file_lock(self.state_file):
            self._save_without_lock()

    def init_run(self, rolling_config, anchor_date, total_windows, training_method='slide'):
        from quantpits.utils.train_utils import file_lock
        with file_lock(self.state_file):
            old_method = self._state.get('training_method')
            self._state = self._empty()
            self._state['started_at'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self._state['rolling_config'] = rolling_config
            self._state['anchor_date'] = anchor_date
            self._state['total_windows'] = total_windows
            self._state['training_method'] = training_method
            self._save_without_lock()
            if old_method and old_method != training_method:
                print(f"  ℹ️  Training method changed: {old_method} → {training_method}")

    def is_window_model_done(self, window_idx, model_name):
        key = str(window_idx)
        return key in self._state['completed_windows'] and \
               model_name in self._state['completed_windows'][key]

    def mark_window_model_done(self, window_idx, model_name, record_id):
        from quantpits.utils.train_utils import file_lock
        with file_lock(self.state_file):
            # Reload to merge with any concurrent window updates
            self._state = self._load_without_lock()
            key = str(window_idx)
            if key not in self._state['completed_windows']:
                self._state['completed_windows'][key] = {}
            self._state['completed_windows'][key][model_name] = record_id
            self._state['current_window_idx'] = window_idx
            self._state['current_model'] = model_name
            self._save_without_lock()

    def get_completed_record_ids(self, model_name):
        """获取某模型在所有已完成 windows 的 record_ids (按 window_idx 排序)"""
        result = []
        for key in sorted(self._state['completed_windows'].keys(), key=int):
            models = self._state['completed_windows'][key]
            if model_name in models:
                result.append({
                    'window_idx': int(key),
                    'record_id': models[model_name],
                })
        return result

    def get_all_completed_windows(self):
        return self._state['completed_windows']

    @property
    def anchor_date(self):
        return self._state.get('anchor_date')

    def get_last_completed_window_idx(self):
        """获取最后一个已完成的 window index"""
        completed = self._state.get('completed_windows', {})
        if not completed:
            return None
        return max(int(k) for k in completed.keys())

    def remove_model(self, model_name):
        """删除指定模型在所有 window 中的训练记录（供 --retrain-models 使用）"""
        from quantpits.utils.train_utils import file_lock
        with file_lock(self.state_file):
            self._state = self._load_without_lock()
            count = 0
            for key in list(self._state['completed_windows'].keys()):
                if model_name in self._state['completed_windows'][key]:
                    del self._state['completed_windows'][key][model_name]
                    count += 1
            # 清理空 window
            empty = [k for k, v in self._state['completed_windows'].items() if not v]
            for k in empty:
                del self._state['completed_windows'][k]
            if count > 0:
                self._save_without_lock()
            return count

    def remove_window(self, window_idx):
        """删除指定 window 的所有训练记录（供 --retrain-last 使用）"""
        from quantpits.utils.train_utils import file_lock
        with file_lock(self.state_file):
            self._state = self._load_without_lock()
            key = str(window_idx)
            if key in self._state['completed_windows']:
                del self._state['completed_windows'][key]
                self._save_without_lock()
                return True
            return False

    def clear(self):
        from quantpits.utils.train_utils import file_lock, backup_file_with_date
        with file_lock(self.state_file):
            if os.path.exists(self.state_file):
                backup_file_with_date(self.state_file, prefix="rolling_state")
                os.remove(self.state_file)
                print("🗑️  Rolling 状态已清除")

    def show(self):
        if not self._state.get('started_at'):
            print("ℹ️  没有找到 Rolling 运行状态")
            return

        s = self._state
        print("\n📋 Rolling 运行状态:")
        print(f"  开始时间: {s.get('started_at', 'N/A')}")
        print(f"  锚点日期: {s.get('anchor_date', 'N/A')}")
        print(f"  总 Windows: {s.get('total_windows', 0)}")

        completed = s.get('completed_windows', {})
        n_completed_windows = len(completed)
        total_models = sum(len(m) for m in completed.values())
        print(f"  已完成 Windows: {n_completed_windows}")
        print(f"  已完成 模型×Window: {total_models}")

        if completed:
            for widx in sorted(completed.keys(), key=int):
                models = completed[widx]
                print(f"    Window {widx}: {list(models.keys())}")

        cw = s.get('current_window_idx')
        cm = s.get('current_model')
        if cw is not None:
            print(f"  最后完成: Window {cw}, 模型 {cm}")
