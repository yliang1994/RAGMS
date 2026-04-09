# Dashboard Usage

启动：

```bash
source .venv/bin/activate
python scripts/run_dashboard.py --settings settings.yaml --serve
```

默认地址：

```text
http://localhost:<dashboard.port>
```

六个页面：

- 系统总览：最近 trace、组件配置摘要、评估入口
- 数据浏览器：集合、文档、chunk、trace 跳转
- Ingestion 管理：上传、删除、重建
- Ingestion 追踪：摄取阶段时间线和回跳
- Query 追踪：查询阶段时间线、对比
- 评估面板：运行、报告、baseline、失败样本、趋势

评估面板关键状态：

- 无报告空态
- 运行中
- 运行成功
- 运行失败可排障

常见跳转：

- 文档 -> 数据浏览
- Trace -> Ingestion / Query Trace
- 报告 -> Evaluation Panel
- 报告 / Trace -> 数据浏览
