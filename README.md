# RagServer

当前冻结交付版本：`1.0.0`

RagServer 是一个本地优先的 RAG 系统，包含：

- 文档摄取与元数据/图片持久化
- Hybrid Query Engine
- MCP Server
- Trace / Dashboard
- Evaluation / Baseline / Acceptance

项目以 [DEV_SPEC.md](/home/yliang/cv_project/RagServer/DEV_SPEC.md) 为实现计划与交付记录来源。

## 1. 环境准备

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m pip install pytest pytest-mock pytest-check pytest-bdd pytest-cov
```

所有 `python`、`pytest`、脚本执行都应在 `.venv` 中完成。

## 2. 配置

默认配置文件是 `settings.yaml`。最重要的字段：

- `vector_store.collection`
- `storage.sqlite.path`
- `observability.log_file`
- `dashboard.port`
- `evaluation.backends`

如果你要接真实模型，需要在 `llm`、`embedding`、`vision_llm` 中填入对应 provider 的 `api_key` / `base_url`。

## 3. 常用命令

摄取：

```bash
source .venv/bin/activate
python scripts/ingest_documents.py --settings settings.yaml --path data/src_raw_data
```

查询：

```bash
source .venv/bin/activate
python scripts/query_cli.py --settings settings.yaml "what is rag"
```

MCP Server：

```bash
source .venv/bin/activate
python scripts/run_mcp_server.py --settings settings.yaml
```

Dashboard：

```bash
source .venv/bin/activate
python scripts/run_dashboard.py --settings settings.yaml --serve
```

评估：

当前真实评估入口已接入 Dashboard 和 MCP `evaluate_collection`；本地验收推荐直接跑：

```bash
source .venv/bin/activate
python scripts/run_acceptance.py
```

## 4. MCP 客户端配置示例

GitHub Copilot `mcp.json`：

```json
{
  "servers": {
    "ragms": {
      "command": "/abs/path/to/RagServer/.venv/bin/python",
      "args": [
        "/abs/path/to/RagServer/scripts/run_mcp_server.py",
        "--settings",
        "/abs/path/to/RagServer/settings.yaml"
      ]
    }
  }
}
```

Claude Desktop `claude_desktop_config.json`：

```json
{
  "mcpServers": {
    "ragms": {
      "command": "/abs/path/to/RagServer/.venv/bin/python",
      "args": [
        "/abs/path/to/RagServer/scripts/run_mcp_server.py",
        "--settings",
        "/abs/path/to/RagServer/settings.yaml"
      ]
    }
  }
}
```

可直接按你的本地路径替换上面的绝对路径。

## 5. Dashboard 页面

当前 Dashboard 包含六页：

- `system_overview`
- `data_browser`
- `ingestion_management`
- `ingestion_trace`
- `query_trace`
- `evaluation_panel`

`evaluation_panel` 已支持：

- 查看历史报告
- 启动真实评估入口
- baseline 设置与对比
- 失败样本查看
- 配置快照与趋势查看

## 6. Trace 与报告位置

- Trace JSONL：`logs/traces.jsonl`
- SQLite 元数据：`data/metadata/ragms.db`
- 评估报告：`data/evaluation/reports/*.json`
- 评估运行记录：`data/evaluation/runs/*.json`

## 7. 一键验收

```bash
source .venv/bin/activate
python scripts/run_acceptance.py
```

脚本会输出结构化 JSON 摘要，至少包含：

- `status`
- `failed_steps`
- `artifact_paths`
- `trace_ids`
- `run_ids`
- 三个核心场景映射

## 8. 测试命令

MCP 协议级 E2E：

```bash
source .venv/bin/activate
pytest tests/e2e/test_mcp_client_simulation.py
```

Dashboard 最终回归：

```bash
source .venv/bin/activate
pytest tests/e2e/test_dashboard_smoke.py tests/e2e/test_dashboard_navigation_regression.py tests/e2e/test_evaluation_visible_in_dashboard.py
```

全链路验收：

```bash
source .venv/bin/activate
pytest tests/e2e/test_full_chain_acceptance.py
```

最终覆盖率：

```bash
source .venv/bin/activate
pytest --cov=src/ragms --cov-report=term-missing tests/unit tests/integration tests/e2e
```

## 9. 常见问题

`ModuleNotFoundError`：

使用 `.venv`，并确认已经执行 `python -m pip install -e .`。

`streamlit` 未安装：

```bash
source .venv/bin/activate
python -m pip install streamlit
```

`index.lock`：

当前仓库在自动提交过程中偶发暂态 `index.lock`，直接重试非破坏性 `git add` / `git commit` 即可。

真实模型调用失败：

- 检查 `settings.yaml` 的 provider / model / api_key
- 检查网络出口和 base URL
- 对 `ragas` / `deepeval` 缺依赖时，当前实现会稳定返回 skip / 结构化失败，而不会直接拖垮流程

## 10. 限制说明

- 默认测试与验收流程偏向本地可重复、无真实网络依赖
- 真实 provider 评估效果依赖外部模型与数据集质量
- Dashboard 当前是 Streamlit 壳，不是多用户部署方案
- Acceptance 脚本输出的是交付前本地摘要，不是长期任务编排器

## 11. 交付清单

正式交付检查以 `python scripts/run_acceptance.py` 的 JSON 输出为准，输出里会包含 `release_checklist`。当前冻结 baseline：

- `run_id`: `baseline-run`
- `collection`: `dashboard-demo`
- `dataset_version`: `v1`
- `backend_set`: `["custom_metrics"]`
- `generated_at`: `2026-04-09`

发布前需要至少确认：

- 版本号：`1.0.0`
- 一键验收：`python scripts/run_acceptance.py`
- 覆盖率命令：`pytest --cov=src/ragms --cov-report=term-missing tests/unit tests/integration tests/e2e`
- 关键能力：摄取、查询、MCP、Trace、Dashboard、Evaluation、baseline 对比、acceptance summary
- 限制说明：见第 10 节

覆盖率结果以最终 `pytest --cov` 输出为准；关键集成路径和三条核心 E2E 场景要求都已纳入最终回归清单。

本次正式冻结的覆盖率结果：

- 命令：`pytest --cov=src/ragms --cov-report=term-missing tests/unit tests/integration tests/e2e`
- 结果：`433 passed`
- `TOTAL` 行覆盖率：`88%`
- 核心门槛结论：核心单元逻辑 `>= 80%`、关键集成路径 `100%`、三条核心 E2E 场景 `100%`
- 非阻塞 warning：`jieba/pkg_resources` 与若干 `sqlite3 ResourceWarning`
