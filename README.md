# RagServer

This repository follows `DEV_SPEC.md` as the primary source of truth.

## Current status

Stage A1 establishes the minimal Python project scaffold:

- `src/ragms` package
- `scripts/` entrypoint directory
- `tests/` baseline with isolated test environment defaults
- `data/` and `logs/` local runtime directories

## Local bootstrap

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install pydantic pyyaml python-dotenv pytest pytest-mock pytest-check pytest-bdd
python -m pip install -e .
python scripts/run_mcp_server.py
pytest --collect-only
```

## Notes

- Activate `.venv` before running any Python command.
- `tests/conftest.py` sets isolated runtime paths for test runs.

## Real Test Scripts

`tests/real_test/` 下有 3 个面向真实数据和真实服务的端到端脚本：

- `run_real_ingestion_test.py`：验证真实摄取链路，覆盖首次建库、增量跳过、强制重建。
- `run_real_query_test.py`：验证直连查询链路，调用 `scripts/query_cli.py`。
- `run_real_mcp_flow_test.py`：验证完整 MCP stdio 流程，覆盖 `initialize`、`tools/list`、`ingest_documents`、`get_document_summary`、`query_knowledge_hub`。

### 前置条件

- 使用仓库虚拟环境：`.venv`
- 已安装项目依赖，并且 `.venv` 中可导入 `ragms`
- `settings.yaml` 中的真实模型和向量库配置可用
- 可访问外部模型服务
  - `run_real_query_test.py` 和 `run_real_mcp_flow_test.py` 会触发真实 embedding / LLM 请求
  - 在受限沙箱里运行时，常见失败现象是 `ConnectError` 或 `Operation not permitted`
- `data/src_raw_data` 下存在待测试 PDF
- `query_test.txt` 中存在非空查询文本

### 1. 真实摄取测试

用途：检查真实 PDF 摄取、分块、存储、稀疏索引和元数据产物是否完整。

当前工作区里，`data/src_raw_data` 共有 `4` 个 PDF，因此建议显式传入 `--expected-count 4`。

```bash
.venv/bin/python tests/real_test/run_real_ingestion_test.py --expected-count 4
```

常用参数：

- `--settings`：指定配置文件，默认 `settings.yaml`
- `--source-dir`：指定 PDF 目录，默认 `data/src_raw_data`
- `--collection`：指定测试集合名，默认 `real_c_ingestion_test`
- `--expected-count`：要求目录内 PDF 数量与该值一致

脚本会顺序执行三轮：

- `force-index`
- `incremental-skip`
- `force-rebuild`

成功时会打印 `=== Real Ingestion Test Summary ===`，并汇总每轮状态与产物检查结果。

### 2. 真实查询测试

用途：检查直连查询链路是否能返回答案、引用和 top chunks。

```bash
.venv/bin/python tests/real_test/run_real_query_test.py
```

常用参数：

- `--settings`：指定配置文件
- `--query-file`：指定查询文本文件，默认 `query_test.txt`
- `--collection`：指定查询集合，默认 `real_c_ingestion_test`
- `--top-k`：检索 chunk 数，默认 `3`
- `--print-top-chunks`：输出前 N 个 chunk，默认 `3`
- `--filters`：透传 JSON 过滤条件
- `--return-debug`：输出调试信息

示例：

```bash
.venv/bin/python tests/real_test/run_real_query_test.py \
  --collection real_c_ingestion_test \
  --top-k 5 \
  --print-top-chunks 5
```

成功时会打印 `=== Real Query Test Summary ===`，其中包含：

- `retrieved_chunks`
- `printed_top_chunks`
- CLI 的完整 `stdout` / `stderr`

### 3. 真实 MCP 流程测试

用途：检查 MCP server 的完整 stdio 调用流程是否跑通。

```bash
.venv/bin/python tests/real_test/run_real_mcp_flow_test.py
```

常用参数：

- `--settings`：指定配置文件
- `--source-dir`：指定摄取目录
- `--query-file`：指定查询文本文件
- `--collection`：指定摄取和查询集合，默认 `real_c_ingestion_test`
- `--top-k`：传给 `query_knowledge_hub` 的 top-k，默认 `5`
- `--force-rebuild`：摄取时强制重建
- `--return-debug`：查询时返回 debug 信息
- `--timeout-seconds`：单次 MCP 响应超时，默认 `600`
- `--log-level`：MCP server 日志级别，默认 `INFO`

示例：

```bash
.venv/bin/python tests/real_test/run_real_mcp_flow_test.py \
  --collection real_c_ingestion_test \
  --top-k 5 \
  --log-level INFO
```

成功时会打印 `=== Real MCP Flow Test Summary ===`，并输出每一步的 request/response：

- `initialize`
- `tools/list`
- `tools/call:list_collections`
- `tools/call:ingest_documents`
- `tools/call:get_document_summary`
- `tools/call:query_knowledge_hub`

如果你只想快速验证当前修复是否生效，推荐按下面顺序跑：

```bash
.venv/bin/python tests/real_test/run_real_query_test.py
.venv/bin/python tests/real_test/run_real_mcp_flow_test.py
```
