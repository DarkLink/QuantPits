# 深度分析系统 (MAS) — LLM 观测与追踪指南 (Observability & Tracing)

为了实现高透明度的 AI 推理过程分析和事后审计，MAS 深度分析系统内置了完整的 LLM 追踪 (Trace) 能力。本指南说明了如何开启该功能、了解日志层级结构以及如何利用预留的接口扩展至多模型研讨（Multi-LLM Debate）模式。

## 1. 概览与启用

系统会在工作区执行期间，将所有 LLM API 的 Request 和 Response（包括推理模型的 `reasoning_content`，即 `<think>` 内容）**100% 完整记录**到 JSON 文件中，不进行任何截断。

### 开启追踪

可以在工作区的 `config/llm_config.json` 文件中配置 `trace` 块。默认情况下该功能应处于启用状态：

```json
{
    "critic_model": "deepseek-v4-pro",
    "summary_model": "gpt-4o",
    "trace": {
        "enabled": true,
        "output_dir": "output/deep_analysis/llm_traces",
        "langfuse": {
            "enabled": false,
            "host": "http://localhost:3000",
            "public_key_env": "LANGFUSE_PUBLIC_KEY",
            "secret_key_env": "LANGFUSE_SECRET_KEY"
        }
    }
}
```

* `enabled`: 设置为 `false` 可全局彻底关闭追踪，系统回退到 No-op 模式，无任何性能损耗。
* `output_dir`: JSON 文件的存储基准路径，相对于工作区目录。
* `langfuse`: 软依赖适配器。如果安装了 `langfuse` 库并且启用了此处配置，系统将自动将 traces 推送到 Langfuse 服务器，同时本地 JSON 记录照常保留。

## 2. 存储与目录层级设计

存储结构采用了 **Run → Session → Trace** 的层级体系。
执行 `run_deep_analysis.py` 后，会在 `output_dir` 下生成带有唯一后缀和日期的运行目录：

```text
output/deep_analysis/llm_traces/2026-06-05_run_abc12345/
├── manifest.json                        ← 全局索引与聚合统计
├── triage/
│   └── 001_triage_deepseek-v4-pro.json  ← 具体的 LLM API Call 记录
├── per_model/
│   ├── alstm_Alpha158/
│   │   └── 002_alstm_Alpha158_v4-pro.json
│   └── lightgbm_Alpha158/
│       └── 003_lightgbm_Alpha158_v4-pro.json
├── per_combo/
│   └── ...
├── execution_risk/
│   └── ...
├── synthesizer/
│   └── ...
└── summary/
    └── 007_summary_gpt-4o.json
```

### 概念定义

1. **Run (运行级)**: 一次 `run_deep_analysis.py` 的执行周期。
2. **Session (会话级)**: 系统的一个逻辑阶段（如 `triage`, `per_model_critic(alstm)`, `synthesizer`）。由于系统的某些阶段（如 per-model）是并发执行的，Session 实现了线程安全的日志隔离。
3. **Round & Participant (轮次与参与者)**: 这是为**多 LLM 模式预留的扩展维度**。在单模型模式下，默认 `round=1`, `participant=primary`。
4. **Trace (请求级)**: 单次大模型 API HTTP 请求与响应。

## 3. JSON Trace 数据结构

每一个 `.json` 文件都是一个 `LLMTraceRecord` 数据类的严格序列化表达，主要包含以下字段：

```json
{
  "trace_id": "a1b2c3d4-...",
  "run_id": "abc12345",
  "session_id": "...",
  "session_type": "per_model_critic",
  
  "round_number": 1,
  "round_phase": "sole",
  "participant_id": "primary",
  "participant_role": "sole",
  "input_trace_ids": [],
  
  "label": "Per-Model(alstm_Alpha158)",
  "timestamp": "2026-06-05T10:35:00.123456",
  "duration_ms": 32400,
  
  "model_requested": "deepseek-v4-pro",
  "model_responded": "deepseek-v4-pro",
  "temperature": 0.3,
  "max_tokens": 32768,
  
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."}
  ],
  "system_prompt_hash": "ab12cd34",
  
  "response_content": "{\"action\": \"retry\", ...}",
  "reasoning_content": "<think>\n市场波动率上升，... \n</think>",
  "finish_reason": "stop",
  
  "prompt_tokens": 8200,
  "completion_tokens": 1450,
  "total_tokens": 9650,
  "reasoning_tokens": 15200,
  "cached_tokens": 0,
  
  "success": true,
  "error": ""
}
```

> **注意**: `messages`, `response_content`, 和 `reasoning_content` 会完整保留原文本，不对长度做任何截断。

## 4. 多 LLM 研讨模式（Multi-LLM Debate）扩展预留

考虑到系统未来将从单 Critic 模型演进至**多 LLM 交叉盘问与研讨**模式（例如：Challenger 提出异议，Arbiter 进行裁决），我们对 `LLMTraceRecord` 和 `SessionContext` 进行了前瞻性设计。

### 扩展字段说明

* `round_number` (int): 对话轮次。
* `round_phase` (str): 阶段标识。预留值：`sole` (单模型), `independent_opinion` (独立意见期), `debate` (辩论期), `rebuttal` (反驳期), `consensus` (共识期), `arbitration` (裁决期)。
* `participant_id` (str): 发言模型的唯一标识。
* `participant_role` (str): 角色。预留值：`critic`, `challenger`, `arbiter`。
* `input_trace_ids` (List[str]): 图结构追踪 (DAG Edges)。记录当前 Prompt 拼接了哪些前置 LLM 的输出。

### 在代码中扩展多模型对话

现有的 `SessionContext` 是线程安全的跨轮次管理器。未来实现多模型时，只需在同一个 session 内调用流转方法，即可保持追踪聚合，无需修改底层写逻辑：

```python
with trace_logger.session("per_model_critic", target=model_name) as sess:
    
    # 1. 轮次 1: 独立意见
    sess.new_round(phase="independent_opinion")
    
    sess.set_participant("llm_A", role="critic")
    res_A = self._call_llm(..., session_ctx=sess)
    
    sess.set_participant("llm_B", role="challenger")
    res_B = self._call_llm(..., session_ctx=sess)
    
    # 2. 轮次 2: 辩论
    sess.new_round(phase="debate")
    sess.set_participant("llm_A", role="critic")
    # 显式传入 input_trace_ids 建立图关系
    res_A_rebuttal = self._call_llm(
        ..., 
        input_trace_ids=[res_B.trace_id], 
        session_ctx=sess
    )
```

## 5. Langfuse 适配器说明

模块 `quantpits/scripts/deep_analysis/langfuse_adapter.py` 提供了一个软依赖适配器：

* **层级映射**: Run -> Trace (Trace ID = run_id), Session -> Span, 单次请求 -> Generation。
* **思维链可见性**: `reasoning_content` 会被注入到 Generation 的 metadata / output 中，确保在 Langfuse 的可视化 UI 中可供审查。
* **静默降级**: 如果未安装 `langfuse` 包，该适配器会自动捕获 `ImportError` 并转为全 No-op，不影响基础的本地 JSON 写入功能。

## 6. 事后分析脚本示例

在日常 Debug 和 OOM-RL 策略分析中，可以使用 `jq` 快速查询本地 trace。

**查看本次运行的调用汇总统计**
```bash
cat output/deep_analysis/llm_traces/2026-06-05_run_*/manifest.json | jq '.summary'
```

**查看各个阶段的 Tokens 消耗**
```bash
cat output/deep_analysis/llm_traces/2026-06-05_run_*/manifest.json | jq '.sessions[] | {type: .session_type, target: .target, prompt: .tokens.prompt, reasoning: .tokens.reasoning}'
```

**快速提取深度思考内容 (Reasoning)**
```bash
cat output/deep_analysis/llm_traces/2026-06-05_run_*/per_model/alstm_Alpha158/*.json | jq -r '.reasoning_content'
```
