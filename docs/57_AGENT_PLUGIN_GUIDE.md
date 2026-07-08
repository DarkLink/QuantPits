# 代理插件开发指南 (Workspace-Local Agent Plugins)

在 Deep Analysis (MAS) 系统的 7 阶段流水线架构中，Agent 扮演着分析数据的核心角色。系统支持通过 **Agent Manifest** 提供完全可插拔、基于工作区的本地代理插件。

这种设计的优势在于：**保持工作区隔离，不污染全局环境**。开发者可以在特定的 Playgound 工作区下测试自定义代理，而无需修改系统的全局代码或环境变量 (如 `PYTHONPATH`)。

---

## 1. 原理与加载机制

系统通过 `quantpits.scripts.deep_analysis.agents.__init__.py` 中的 `load_manifest_agents()` 方法动态加载代理。

1. **读取 Manifest**: 从 `--manifest` 参数指定的文件 (通常位于 `workspaces/<name>/config/agent_manifest.json`) 读取代理定义。
2. **环境隔离 (sys.path 注入)**: 在加载代理类之前，系统会将当前执行上下文（工作区根目录）动态注入到 Python 的 `sys.path` 的首位。
3. **类加载**: 使用 `importlib.import_module` 根据 Manifest 中的 `class_path` 导入代理类。
4. **清理**: 加载完成后，系统会自动移除注入的路径，确保后续操作不受污染。

---

## 2. 开发自定义代理

以下步骤演示如何在工作区 `Demo_Workspace` 下创建一个本地代理。

### 步骤 A: 编写代理代码

在工作区目录中创建一个 Python 文件，例如 `workspaces/Demo_Workspace/custom_agent.py`:

```python
from quantpits.scripts.deep_analysis.base_agent import BaseAgent, AgentFindings, AnalysisContext

class WorkspaceLocalAgent(BaseAgent):
    name = "workspace_local_plugin"
    description = "A clean plugin agent defined within this workspace."

    def analyze(self, ctx: AnalysisContext) -> AgentFindings:
        # 执行自定义分析逻辑
        findings = [
            self._make_finding(
                severity="info",
                title="Workspace Local Rule Fired",
                detail="Successfully loaded and executed workspace-isolated custom agent."
            )
        ]
        return AgentFindings(self.name, ctx.window_label, findings, [], {})
```

### 步骤 B: 注册 Manifest

在工作区配置目录下创建 `agent_manifest.json`，例如 `workspaces/Demo_Workspace/config/agent_manifest.json`:

```json
{
  "agents": [
    {
      "name": "workspace_local_plugin",
      "class_path": "custom_agent.WorkspaceLocalAgent",
      "enabled": true
    }
  ]
}
```
**注意**: `class_path` 的包路径是相对工作区根目录的。由于我们将 `custom_agent.py` 放在了工作区根目录，这里的路径就是 `custom_agent.WorkspaceLocalAgent`。

---

## 3. 运行流水线

使用 `--manifest` 指定配置文件，并通过 `--agents` 指定要运行的代理：

```bash
# 进入量化系统根目录
cd /path/to/QuantPits_Release

# 执行深度分析
python quantpits/scripts/run_deep_analysis.py \
    --stage agents \
    --agents workspace_local_plugin \
    --manifest workspaces/Demo_Workspace/config/agent_manifest.json \
    --no-snapshot
```

运行输出中将包含由 `WorkspaceLocalAgent` 产生的信息，证明本地插件已成功隔离加载并执行完毕。

---

## 4. 阶段插件 (Pipeline Stage Plugins)

除了 Agent 插件，系统同样支持通过工作区本地清单注册**自定义流水线阶段**。机制与 Agent 插件完全一致：

```json
// config/pipeline_manifest.json
{
    "stages": [
        {
            "name": "custom_liquidity_check",
            "depends_on": ["signals"],
            "provides": ["liquidity_report"],
            "class_path": "custom_stages.liquidity.run_stage",
            "insert_after": "signals",
            "enabled": true
        }
    ]
}
```

自定义阶段函数使用 `@register_stage` 装饰器自声明依赖关系和产出字段：

```python
# custom_stages/liquidity.py
from quantpits.scripts.deep_analysis.stage_runner import register_stage

@register_stage(
    name='custom_liquidity_check',
    depends_on=['signals'],
    provides=['liquidity_report'],
)
def run_stage(state, **kwargs):
    # state.signals — 由上游 signals 阶段产出
    ...
    state.liquidity_report = {...}
    return state
```

加载方式与 Agent 插件共用 `--manifest` 参数（系统自动识别 `"agents"` 和 `"stages"` 键），或分别指定不同清单文件。详见 [50 — 深度分析系统使用指南](50_DEEP_ANALYSIS_GUIDE.md)。

---

## 5. 训练模式感知 (Training Mode Awareness)

自 Phase 2b 起，`AnalysisContext` 提供了 `training_context` 字段（类型：`Optional[TrainingModeContext]`），使自定义 agent 可以感知当前工作区的训练状态。

### 5.1 可用 API

```python
from quantpits.scripts.deep_analysis.base_agent import BaseAgent, AgentFindings, AnalysisContext

class MyModeAwareAgent(BaseAgent):
    name = "my_mode_aware_agent"
    description = "Demonstrates training mode awareness in a custom agent."

    def analyze(self, ctx: AnalysisContext) -> AgentFindings:
        findings = []
        tc = ctx.training_context

        if not tc:
            # 工作区无训练记录（全新工作区或数据缺失）
            findings.append(self._make_finding(
                'info', 'No training context',
                'Workspace has no training records yet.'
            ))
            return AgentFindings(self.name, ctx.window_label, findings, [], {})

        # 1. 检查是否为仅预测周期
        if tc.is_predict_only_cycle:
            findings.append(self._make_finding(
                'info', 'Predict-only cycle',
                'Current cycle is predict-only. Models were not retrained.'
            ))

        # 2. 查询模型的最新操作
        last_op = tc.get_last_operation("lstm_Alpha158")
        if last_op:
            findings.append(self._make_finding(
                'info', f'Last op for lstm_Alpha158: {last_op["type"]}',
                f"Mode: {last_op['mode']}, Date: {last_op['date'][:10]}"
            ))

        # 3. 获取多模式训练的模型
        cross_mode = tc.get_cross_mode_models()
        if cross_mode:
            findings.append(self._make_finding(
                'info', f'{len(cross_mode)} cross-mode models',
                f"Models trained in multiple modes: {', '.join(cross_mode[:5])}..."
            ))

        # 4. 按训练模式筛选
        rolling_models = tc.get_models_with_mode("rolling")
        static_models = tc.get_models_with_mode("static")

        # 5. 检查滚动管道延迟
        gap = tc.get_rolling_gap_days("rolling")
        if gap is not None and gap > 90:
            findings.append(self._make_finding(
                'warning', 'Rolling pipeline stale',
                f"Rolling pipeline has not run in {gap} days."
            ))

        return AgentFindings(self.name, ctx.window_label, findings, [], {})
```

### 5.2 关键数据源

`TrainingModeContext` 从工作区文件系统聚合训练状态，无需自定义 agent 自行解析：

| 属性/方法 | 返回类型 | 数据来源 |
|-----------|----------|----------|
| `tc.available_modes` | `List[str]` | `latest_train_records.json` 的 @suffix |
| `tc.models_by_name` | `Dict[str, Dict[str, str]]` | `latest_train_records.json` models 键 |
| `tc.is_predict_only_cycle` | `bool` | `training_history.jsonl` + `prediction_history.jsonl` |
| `tc.last_train` | `Dict[str, dict]` | `training_history.jsonl`（最近训练） |
| `tc.last_prediction` | `Dict[str, dict]` | `prediction_history.jsonl`（最近预测） |
| `tc.get_last_operation(name)` | `dict\|None` | 综合 last_train + last_prediction |
| `tc.get_rolling_gap_days(mode)` | `int\|None` | `rolling_state.json` vs anchor_date |
| `tc.rolling_states` | `Dict[str, dict]` | `data/rolling_state*.json` |

### 5.3 Predict-Only 周期校准

自定义 agent 应遵循与内置 agent 相同的约定：**在仅预测周期中抑制基于"未重训"的警告**。如果模型有训练历史（`tc.get_last_operation()` 返回 `type=train`），仅因本周期未重训不应标记为"陈旧"或"缺失"。只有从未训练过的模型才需要标记。
