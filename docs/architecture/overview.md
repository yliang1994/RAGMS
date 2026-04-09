# Architecture Overview

RagServer 由六个主路径组成：

- Ingestion Pipeline：文档加载、切分、transform、embedding、storage、lifecycle
- Query Engine：query processing、hybrid retrieval、rerank、response build、answer generation
- MCP Server：对外工具协议层
- Observability：trace manager、trace repository、dashboard
- Evaluation：dataset loader、evaluator stack、runner、report service、baseline
- Acceptance：E2E 回归与一键验收

关键持久化：

- SQLite：文档、图片、评估运行摘要
- JSONL：trace
- 文件系统：图片、稀疏索引、评估报告

关键交互关系：

- Ingestion 和 Query 都写 Trace
- Evaluation 运行会同时写 report、SQLite run summary、evaluation trace
- Dashboard 读 DataService / TraceService / ReportService
- MCP Tool 通过服务层和 runner 暴露结构化能力
