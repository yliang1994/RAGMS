# RagServer

当前冻结交付版本：`1.0.0`

RagServer 是一个本地优先的 RAG 项目，包含：

- 文档摄取与元数据/图片持久化
- Hybrid Query Engine
- MCP Server
- Trace / Dashboard
- Evaluation / Baseline / Acceptance

项目以 [DEV_SPEC.md](/home/yliang/cv_project/RagServer/DEV_SPEC.md) 为实现计划与交付记录来源。

## 快速结论

当前仓库适合按“源码仓库发布”方式交付。也就是说，推荐在另一台机器上：

1. 拉取整个仓库
2. 创建 `.venv`
3. 执行 `python -m pip install -e .`
4. 保留并配置 `settings.yaml`
5. 通过 `scripts/` 下的入口脚本启动

`pyproject.toml` 已包含 Dashboard 运行依赖，因此新机器不需要再额外手工安装 `streamlit`、`pandas`、`plotly`。

## 1. 环境要求

- Python `3.11+`
- 建议 Linux / macOS
- 运行目录需要可写，用于生成 `data/`、`logs/`

## 2. 新机器安装流程

在目标机器上获取仓库后执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

如果你需要运行测试或验收，再安装开发依赖：

```bash
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

所有 `python`、`pytest`、脚本执行都应在 `.venv` 中完成。

## 3. 配置

默认配置文件是仓库根目录下的 `settings.yaml`。

最重要的字段：

- `vector_store.collection`
- `storage.sqlite.path`
- `observability.log_file`
- `dashboard.port`
- `evaluation.backends`

如果你要接真实模型，需要在 `llm`、`embedding`、`vision_llm` 中填入对应 provider 的 `api_key` / `base_url`。

配置支持环境变量覆盖，变量前缀是 `RAGMS_`，双下划线表示嵌套路径。例如：

```bash
export RAGMS_DASHBOARD__PORT=8601
export RAGMS_VECTOR_STORE__COLLECTION=demo
```

## 4. 首次运行前建议准备

如果你希望在新机器上直接看到已有 Dashboard 数据，而不是空态页面，建议同步这些文件和目录：

- `settings.yaml`
- `data/`
- `logs/`

关键数据位置：

- Trace JSONL：`logs/traces.jsonl`
- SQLite 元数据：`data/metadata/ragms.db`
- 评估报告：`data/evaluation/reports/*.json`
- 评估运行记录：`data/evaluation/runs/*.json`

如果不迁移这些目录，系统仍然可以启动，但 Dashboard 中很多页面会显示空态。

## 5. 常用启动命令

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

默认地址：

```text
http://localhost:8501
```

其中端口来自 `settings.yaml` 的 `dashboard.port`。

## 6. Dashboard 页面

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

补充文档见 [docs/dashboards/usage.md](/home/yliang/cv_project/RagServer/docs/dashboards/usage.md)。

## 7. MCP 客户端配置示例

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

可直接按你的实际绝对路径替换。

## 8. 验收与测试

一键验收：

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

如果你要在新机器上执行测试或验收，请先安装 `dev` 依赖：

```bash
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

常用测试命令：

```bash
source .venv/bin/activate
pytest tests/e2e/test_mcp_client_simulation.py
pytest tests/e2e/test_dashboard_smoke.py tests/e2e/test_dashboard_navigation_regression.py tests/e2e/test_evaluation_visible_in_dashboard.py
pytest tests/e2e/test_full_chain_acceptance.py
pytest --cov=src/ragms --cov-report=term-missing tests/unit tests/integration tests/e2e
```

## 9. 发布建议

如果你要把项目交付到另一台机器，最稳妥的发布内容是整个仓库，而不是只拷贝 `src/`：

- 源码目录：`src/`
- 入口脚本：`scripts/`
- 配置文件：`settings.yaml`
- 说明文档：`README.md`、`DEV_SPEC.md`
- 运行数据：按需带上 `data/`、`logs/`

原因是当前启动入口主要在 `scripts/` 下，验收脚本也依赖仓库内的 `tests/` 目录。

## 10. 常见问题

`ModuleNotFoundError`：

- 确认已激活 `.venv`
- 确认已执行 `python -m pip install -e .`

Dashboard 无法打开：

- 检查 `dashboard.port` 是否被占用
- 检查本机防火墙
- 检查是否使用了正确的 `settings.yaml`

Dashboard 页面为空：

- 检查 `data/` 和 `logs/` 是否存在
- 检查 `settings.yaml` 中的路径配置是否指向正确目录

真实模型调用失败：

- 检查 `settings.yaml` 的 provider / model / api_key
- 检查网络出口和 base URL
- 对 `ragas` / `deepeval` 缺依赖时，当前实现会稳定返回 skip / 结构化失败，而不会直接拖垮流程

## 11. 限制说明

- 默认测试与验收流程偏向本地可重复、无真实网络依赖
- 真实 provider 评估效果依赖外部模型与数据集质量
- Dashboard 当前是 Streamlit 壳，不是多用户部署方案
- Acceptance 脚本输出的是交付前本地摘要，不是长期任务编排器

## 12. 交付清单

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
- 限制说明：见第 11 节

覆盖率结果以最终 `pytest --cov` 输出为准；关键集成路径和三条核心 E2E 场景要求都已纳入最终回归清单。

本次正式冻结的覆盖率结果：

- 命令：`pytest --cov=src/ragms --cov-report=term-missing tests/unit tests/integration tests/e2e`
- 结果：`433 passed`
- `TOTAL` 行覆盖率：`88%`
- 核心门槛结论：核心单元逻辑 `>= 80%`、关键集成路径 `100%`、三条核心 E2E 场景 `100%`
- 非阻塞 warning：`jieba/pkg_resources` 与若干 `sqlite3 ResourceWarning`
