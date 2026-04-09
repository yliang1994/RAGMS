# MCP Usage

当前公开工具：

- `query_knowledge_hub`
- `list_collections`
- `get_document_summary`
- `ingest_documents`
- `get_trace_detail`
- `evaluate_collection`

协议流程固定为：

1. `initialize`
2. `notifications/initialized`
3. `tools/list`
4. `tools/call`

`evaluate_collection` 最小返回字段：

- `run_id`
- `trace_id`
- `collection`
- `dataset_name`
- `dataset_version`
- `backend_set`
- `aggregate_metrics`
- `quality_gate_status`
- `baseline_delta`
- `failed_samples_count`
- `result_path`
- `errors`

错误语义：

- 参数问题：返回负错误码和结构化 `error`
- 领域错误：保持结构化失败，不拖垮 MCP server 主进程
- 未知工具 / 未知方法：返回协议级错误
