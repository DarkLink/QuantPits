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
