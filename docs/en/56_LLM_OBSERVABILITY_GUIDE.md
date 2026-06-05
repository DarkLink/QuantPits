# Deep Analysis System (MAS) — LLM Observability & Tracing Guide

To enable high transparency for AI reasoning processes and post-mortem auditing, the MAS Deep Analysis System features a comprehensive LLM Tracing capability out of the box. This guide explains how to enable this feature, understand the log hierarchy, and utilize the reserved extension points for Multi-LLM Debate modes.

## 1. Overview & Enabling

During the execution of a workspace, the system captures **100% of all LLM API Requests and Responses verbatim** (including the `reasoning_content` from reasoning models, e.g., `<think>` tags) and records them into JSON files without any truncation.

### Enabling Tracing

You can configure the `trace` block in the workspace's `config/llm_config.json` file. This feature is enabled by default:

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

* `enabled`: Setting this to `false` disables tracing globally. The system will fall back to a No-op mode with zero performance overhead.
* `output_dir`: The base path for storing JSON traces, relative to the workspace directory.
* `langfuse`: A soft-dependency adapter. If the `langfuse` library is installed and this is enabled, the system will automatically push traces to a Langfuse server while continuing to retain the local JSON records.

## 2. Storage & Directory Hierarchy

The storage structure follows a **Run → Session → Trace** hierarchy.
After executing `run_deep_analysis.py`, a unique dated run directory is created under the `output_dir`:

```text
output/deep_analysis/llm_traces/2026-06-05_run_abc12345/
├── manifest.json                        ← Global index and aggregated statistics
├── triage/
│   └── 001_triage_deepseek-v4-pro.json  ← Concrete LLM API Call record
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

### Concept Definitions

1. **Run**: A single execution lifecycle of `run_deep_analysis.py`.
2. **Session**: A logical phase of the system (e.g., `triage`, `per_model_critic(alstm)`, `synthesizer`). Because some phases (like per-model) are executed concurrently, Sessions ensure thread-safe log isolation.
3. **Round & Participant**: Reserved expansion dimensions for **Multi-LLM modes**. In single-LLM mode, they default to `round=1`, `participant=primary`.
4. **Trace**: A single HTTP request and response to/from the Large Language Model API.

## 3. JSON Trace Data Structure

Each `.json` file is a strictly serialized representation of the `LLMTraceRecord` dataclass, containing the following primary fields:

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
  "reasoning_content": "<think>\nMarket volatility rising, ... \n</think>",
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

> **Note**: The `messages`, `response_content`, and `reasoning_content` fields retain the full original text without any length truncation.

## 4. Multi-LLM Debate Mode Expansion (Reserved)

Anticipating the system's evolution from a single Critic model to a **Multi-LLM cross-examination and debate** mode (e.g., a Challenger raising objections, an Arbiter making a ruling), we have forward-engineered the `LLMTraceRecord` and `SessionContext`.

### Expansion Fields

* `round_number` (int): The current dialogue round.
* `round_phase` (str): Phase identifier. Reserved values: `sole` (single model), `independent_opinion`, `debate`, `rebuttal`, `consensus`, `arbitration`.
* `participant_id` (str): Unique identifier for the speaking model.
* `participant_role` (str): Role. Reserved values: `critic`, `challenger`, `arbiter`.
* `input_trace_ids` (List[str]): Graph-based tracing (DAG Edges). Records which preceding LLM outputs were concatenated into the current Prompt.

### Extending Multi-Model Dialogue in Code

The existing `SessionContext` is a thread-safe, cross-round manager. To implement multi-model operations in the future, simply invoke its transition methods within the same session to maintain trace aggregation without altering underlying write logic:

```python
with trace_logger.session("per_model_critic", target=model_name) as sess:
    
    # 1. Round 1: Independent Opinion
    sess.new_round(phase="independent_opinion")
    
    sess.set_participant("llm_A", role="critic")
    res_A = self._call_llm(..., session_ctx=sess)
    
    sess.set_participant("llm_B", role="challenger")
    res_B = self._call_llm(..., session_ctx=sess)
    
    # 2. Round 2: Debate
    sess.new_round(phase="debate")
    sess.set_participant("llm_A", role="critic")
    # Explicitly pass input_trace_ids to establish graph relations
    res_A_rebuttal = self._call_llm(
        ..., 
        input_trace_ids=[res_B.trace_id], 
        session_ctx=sess
    )
```

## 5. Langfuse Adapter Notes

The module `quantpits/scripts/deep_analysis/langfuse_adapter.py` provides a soft-dependency adapter:

* **Hierarchy Mapping**: Run -> Trace (Trace ID = run_id), Session -> Span, Single Request -> Generation.
* **Chain-of-Thought Visibility**: The `reasoning_content` is injected into the Generation's metadata/output, ensuring it remains visible and auditable within Langfuse's visualization UI.
* **Silent Degradation**: If the `langfuse` package is not installed, the adapter automatically catches the `ImportError` and converts all calls to No-op, leaving the foundational local JSON writing unaffected.

## 6. Post-Mortem Analysis Script Examples

During routine debugging and OOM-RL strategy analysis, you can use `jq` to rapidly query local traces.

**View aggregated statistics for the current run**
```bash
cat output/deep_analysis/llm_traces/2026-06-05_run_*/manifest.json | jq '.summary'
```

**View Token consumption across phases**
```bash
cat output/deep_analysis/llm_traces/2026-06-05_run_*/manifest.json | jq '.sessions[] | {type: .session_type, target: .target, prompt: .tokens.prompt, reasoning: .tokens.reasoning}'
```

**Quickly extract Deep Thinking (Reasoning) content**
```bash
cat output/deep_analysis/llm_traces/2026-06-05_run_*/per_model/alstm_Alpha158/*.json | jq -r '.reasoning_content'
```
