## 阶段 C：Ingestion Pipeline MVP（目标：能把 PDF 样例摄取到本地存储）

> 注：本阶段严格按 5.4.1 的离线数据流落地，并优先实现“增量跳过（SHA256）”。

### C1：定义核心数据类型/契约（Document/Chunk/ChunkRecord）
- **目标**：定义全链路（ingestion → retrieval → mcp tools）共用的数据结构/契约，避免散落在各子模块内导致的耦合与重复。
- **修改文件**：
  - `src/core/types.py`
  - `src/core/__init__.py`（可选：统一 re-export 以简化导入路径）
  - `tests/unit/test_core_types.py`
- **实现类/函数**（建议）：
  - `Document(id, text, metadata)`
  - `Chunk(id, text, metadata, start_offset, end_offset, source_ref?)`
  - `ChunkRecord(id, text, metadata, dense_vector?, sparse_vector?)`（用于存储/检索载体；字段按后续 C8~C12 演进）
- **验收标准**：
  - 类型可序列化（dict/json）且字段稳定（单元测试断言）。
  - `metadata` 约定最少包含 `source_path`，其余字段允许增量扩展但不得破坏兼容。
  - **`metadata.images` 字段规范**（用于多模态支持）：
    - 结构：`List[{"id": str, "path": str, "page": int, "text_offset": int, "text_length": int, "position": dict}]`
    - `id`：全局唯一图片标识符（建议格式：`{doc_hash}_{page}_{seq}`）
    - `path`：图片文件存储路径（约定：`data/images/{collection}/{image_id}.png`）
    - `page`：图片在原文档中的页码（可选，适用于PDF等分页文档）
    - `text_offset`：占位符在 `Document.text` 中的起始字符位置（从0开始计数）
    - `text_length`：占位符的字符长度（通常为 `len("[IMAGE: {image_id}]")`）
    - `position`：图片在原文档中的物理位置信息（可选，如PDF坐标、像素位置、尺寸等）
    - 说明：通过 `text_offset` 和 `text_length` 可精确定位图片在文本中的位置，支持同一图片多次出现的场景
  - **文本中图片占位符规范**：在 `Document.text` 中，图片位置使用 `[IMAGE: {image_id}]` 格式标记。
- **测试方法**：`pytest -q tests/unit/test_core_types.py`。

### C2：文件完整性检查（SHA256）
- **目标**：在Libs中实现 `file_integrity.py`：计算文件 hash，并提供“是否跳过”的判定接口（使用 SQLite 作为默认存储，支持后续替换为 Redis/PostgreSQL）。
- **修改文件**：
  - `src/libs/loader/file_integrity.py`
  - `tests/unit/test_file_integrity.py`
  - 数据库文件：`data/db/ingestion_history.db`（自动创建）
- **实现类/函数**：
  - `FileIntegrityChecker` 类（抽象接口）
  - `SQLiteIntegrityChecker(FileIntegrityChecker)` 类（默认实现）
    - `compute_sha256(path: str) -> str`
    - `should_skip(file_hash: str) -> bool`
    - `mark_success(file_hash: str, file_path: str, ...)`
    - `mark_failed(file_hash: str, error_msg: str)`
- **验收标准**：
  - 同一文件多次计算hash结果一致
  - 标记 success 后，`should_skip` 返回 `True`
  - 数据库文件正确创建在 `data/db/ingestion_history.db`
  - 支持并发写入（SQLite WAL模式）
- **测试方法**：`pytest -q tests/unit/test_file_integrity.py`。

### C3：Loader 抽象基类与 PDF Loader 壳子
- **目标**：在Libs中定义 `BaseLoader`，并实现 `PdfLoader` 的最小行为。
- **修改文件**：
  - `src/libs/loader/base_loader.py`
  - `src/libs/loader/pdf_loader.py`
  - `tests/unit/test_loader_pdf_contract.py`
- **实现类/函数**：
  - `BaseLoader.load(path) -> Document`
  - `PdfLoader.load(path)`
- **验收标准**：
  - **基础要求**：对 sample PDF（fixtures）能产出 Document，metadata 至少含 `source_path`。
  - **图片处理要求**（遵循 C1 定义的契约）：
    - 若 PDF 包含图片，应提取图片并保存到 `data/images/{doc_hash}/` 目录
    - 在 `Document.text` 中，图片位置插入占位符：`[IMAGE: {image_id}]`
    - 在 `metadata.images` 中记录图片信息（格式见 C1 规范）
    - 若 PDF 无图片，`metadata.images` 可为空列表或省略该字段
  - **降级行为**：图片提取失败不应阻塞文本解析，可在日志中记录警告。
- **测试方法**：`pytest -q tests/unit/test_loader_pdf_contract.py`。
- **测试建议**：
  - 准备两个测试文件：`simple.pdf`（纯文本）和 `with_images.pdf`（包含图片）
  - 验证纯文本PDF能正常解析
  - 验证带图片PDF能提取图片并正确插入占位符

### C4：Splitter 集成（调用 Libs）
- **目标**：实现 Chunking 模块作为 `libs.splitter` 和 Ingestion Pipeline 之间的**适配器层**，完成 Document→Chunks 的业务对象转换。
- **核心职责（DocumentChunker 相比 libs.splitter 的增值）**：
  - **职责边界说明**：
    - `libs.splitter`：纯文本切分工具（`str → List[str]`），不涉及业务对象
    - `DocumentChunker`：业务适配器（`Document对象 → List[Chunk对象]`），添加业务逻辑
  - **6 个增值功能**：
    1. **Chunk ID 生成**：为每个文本片段生成唯一且确定性的 ID（格式：`{doc_id}_{index:04d}_{hash_8chars}`）
    2. **元数据继承**：将 Document.metadata 复制到每个 Chunk.metadata（source_path, doc_type, title 等）
    3. **添加 chunk_index**：记录 chunk 在文档中的序号（从 0 开始），用于排序和定位
    4. **建立 source_ref**：记录 Chunk.source_ref 指向父 Document.id，支持溯源
    5. **图片引用按需分发**：扫描每个 chunk 文本中的 `[IMAGE: {id}]` 占位符，从 `Document.metadata["images"]` 中提取该 chunk 实际引用的 ImageRef，写入 `chunk.metadata["images"]`（仅含该 chunk 引用的子集）和 `chunk.metadata["image_refs"]`（image_id 列表）。无占位符的 chunk 不含 `images` 字段。⚠️ 不可简单整体继承或丢弃文档级 `images`，否则下游 C7 ImageCaptioner 将无法定位图片路径。
    6. **类型转换**：将 libs.splitter 的 `List[str]` 转换为符合 core.types 契约的 `List[Chunk]` 对象
- **修改文件**：
  - `src/ingestion/chunking/document_chunker.py`
  - `src/ingestion/chunking/__init__.py`
  - `tests/unit/test_document_chunker.py`
- **实现类/函数**：
  - `DocumentChunker` 类
  - `__init__(settings: Settings)`：通过 SplitterFactory 获取配置的 splitter 实例
  - `split_document(document: Document) -> List[Chunk]`：完整的转换流程
  - `_generate_chunk_id(doc_id: str, index: int, text: str) -> str`：生成稳定 Chunk ID
  - `_inherit_metadata(document: Document, chunk_index: int, chunk_text: str) -> dict`：元数据继承 + 图片引用按需分发逻辑（需要 chunk_text 来扫描 `[IMAGE: id]` 占位符）
- **验收标准**：
  - **配置驱动**：通过修改 settings.yaml 中的 splitter 配置（如 chunk_size），产出的 chunk 数量和长度发生相应变化
  - **ID 唯一性**：每个 Chunk 的 ID 在整个文档中唯一
  - **ID 确定性**：同一 Document 对象重复切分产生相同的 Chunk ID 序列
  - **元数据完整性**：Chunk.metadata 包含所有 Document.metadata 字段 + chunk_index 字段
  - **图片分发正确性**：含 `[IMAGE: id]` 占位符的 chunk 其 `metadata["images"]` 仅包含该 chunk 引用的图片子集；不含占位符的 chunk 无 `images` 字段；`metadata["image_refs"]` 列表与占位符一致
  - **溯源链接**：所有 Chunk.source_ref 正确指向父 Document.id
  - **类型契约**：输出的 Chunk 对象符合 `core/types.py` 中的 Chunk 定义（可序列化、字段完整）
- **测试方法**：`pytest -q tests/unit/test_document_chunker.py`（使用 FakeSplitter 隔离测试，无需真实 LLM/外部依赖）。

### C5：Transform 抽象基类 + ChunkRefiner（规则去噪 + LLM 增强）
- **目标**：定义 `BaseTransform`；实现 `ChunkRefiner`：先做规则去噪，再通过LLM进行智能增强，并提供失败降级机制（LLM异常时回退到规则结果，不阻塞 ingestion）。
- **前置条件**（必须准备）：
  - **必须配置LLM**：在 `config/settings.yaml` 中配置可用的LLM（provider/model/api_key）
  - **环境变量**：设置对应的API key环境变量（`OPENAI_API_KEY`/`OLLAMA_BASE_URL`等）
  - **验证目的**：通过真实LLM测试验证配置正确性和refinement效果
- **修改文件**：
  - `src/ingestion/transform/base_transform.py`（新增）
  - `src/ingestion/transform/chunk_refiner.py`（新增）
  - `src/core/trace/trace_context.py`（新增：最小实现，Phase F 完善）
  - `config/prompts/chunk_refinement.txt`（已存在，需验证内容并补充 {text} 占位符）
  - `tests/fixtures/noisy_chunks.json`（新增：8个典型噪声场景）
  - `tests/unit/test_chunk_refiner.py`（新增：27个单元测试）
  - `tests/integration/test_chunk_refiner_llm.py`（新增：真实LLM集成测试）
- **实现类/函数**：
  - `BaseTransform.transform(chunks, trace) -> List[Chunk]`
  - `ChunkRefiner.__init__(settings, llm?, prompt_path?)`
  - `ChunkRefiner.transform(chunks, trace) -> List[Chunk]`
  - `ChunkRefiner._rule_based_refine(text) -> str`（去空白/页眉页脚/格式标记/HTML注释）
  - `ChunkRefiner._llm_refine(text, trace) -> str | None`（可选 LLM 重写，失败返回 None）
  - `ChunkRefiner._load_prompt(prompt_path?)`（从文件加载prompt模板，支持默认fallback）
- **实现流程建议**：
  1. 先创建 `tests/fixtures/noisy_chunks.json`，包含8个典型噪声场景：
     - typical_noise_scenario: 综合噪声（页眉/页脚/空白）
     - ocr_errors: OCR错误文本
     - page_header_footer: 页眉页脚模式
     - excessive_whitespace: 多余空白
     - format_markers: HTML/Markdown标记
     - clean_text: 干净文本（验证不过度清理）
     - code_blocks: 代码块（验证保留内部格式）
     - mixed_noise: 真实混合场景
  2. 创建 `TraceContext` 占位实现（uuid生成trace_id，record_stage存储阶段数据）
  3. 实现 `BaseTransform` 抽象接口
  4. 实现 `ChunkRefiner._rule_based_refine` 规则去噪逻辑（正则匹配+分段处理）
  5. 编写规则模式单元测试（使用 fixtures 断言清洗效果）
  6. 实现 `_llm_refine` 可选增强（读取 prompt、调用 LLM、错误处理）
  7. 编写 LLM 模式单元测试（mock LLM 断言调用与输出）
  8. 编写降级场景测试（LLM 失败时回退到规则结果，标记 metadata）
  9. **编写真实LLM集成测试并执行验证**（必须执行，验证LLM配置）
- **验收标准**：
  - **单元测试（快速反馈循环）**：
    - 规则模式：对 fixtures 噪声样例能正确去噪（连续空白/页眉页脚/格式标记/分隔线）
    - 保留能力：代码块内部格式不被破坏，Markdown结构完整保留
    - LLM 模式：mock LLM 时能正确调用并返回重写结果，metadata 标记 `refined_by: "llm"`
    - 降级行为：LLM 失败时回退到规则结果，metadata 标记 `refined_by: "rule"` 和 fallback 原因
    - 配置开关：通过 `settings.yaml` 的 `ingestion.chunk_refiner.use_llm` 控制行为
    - 异常处理：单个chunk处理异常不影响其他chunk，保留原文
  - **集成测试（验收必须项）**：
    - ✅ **必须验证真实LLM调用成功**：使用前置条件中配置的LLM进行真实refinement
    - ✅ **必须验证输出质量**：LLM refined文本确实更干净（噪声减少、内容保留）
    - ✅ **必须验证降级机制**：无效模型名称时优雅降级到rule-based，不崩溃
    - 说明：这是验证"前置条件中准备的LLM配置是否正确"的必要步骤
- **测试方法**：
  - **阶段1-单元测试（开发中快速迭代）**：
    ```bash
    pytest tests/unit/test_chunk_refiner.py -v
    # ✅ 27个测试全部通过，使用Mock隔离，无需真实API
    ```
  - **阶段2-集成测试（验收必须执行）**：
    ```bash
    # 1. 运行真实LLM集成测试（必须）
    pytest tests/integration/test_chunk_refiner_llm.py -v -s
    # ✅ 验证LLM配置正确，refinement效果符合预期
    # ⚠️ 会产生真实API调用与费用
    
    # 2. Review打印输出，确认精炼质量
    # - 噪声是否被有效去除？
    # - 有效内容是否完整保留？
    # - 降级机制是否正常工作？
    ```
  - **测试分层逻辑**：
    - 单元测试：验证代码逻辑正确
    - 集成测试：验证系统可用性
    - 两者互补，缺一不可

### C6：MetadataEnricher（规则增强 + 可选 LLM 增强 + 降级）
- **目标**：实现元数据增强模块：提供规则增强的默认实现，并重点支持 LLM 增强（配置已就绪，LLM 开关打开）。利用 LLM 对 chunk 进行高质量的 title 生成、summary 摘要和 tags 提取。同时保留失败降级机制，确保不阻塞 ingestion。
- **修改文件**：
  - `src/ingestion/transform/metadata_enricher.py`
  - `tests/unit/test_metadata_enricher_contract.py`
- **验收标准**：
  - 规则模式：作为兜底逻辑，输出 metadata 必须包含 `title/summary/tags`（至少非空）。
  - **LLM 模式（核心）**：在 LLM 打开的情况下，确保真实调用 LLM（或高质量 Mock）并生成语义丰富的 metadata。需验证在有真实 LLM 配置下的连通性与效果。
  - 降级行为：LLM 调用失败时回退到规则模式结果（可在 metadata 标记降级原因，但不抛出致命异常）。
- **测试方法**：`pytest -q tests/unit/test_metadata_enricher_contract.py`，并确保包含开启 LLM 的集成测试用例。

### C7：ImageCaptioner（可选生成 caption + 降级不阻塞）
- **目标**：实现 `image_captioner.py`：当启用 Vision LLM 且存在 image_refs 时生成 caption 并写回 chunk metadata；当禁用/不可用/异常时走降级路径，不阻塞 ingestion。
- **修改文件**：
  - `src/ingestion/transform/image_captioner.py`
  - `config/prompts/image_captioning.txt`（作为默认 prompt 来源；可在测试中注入替代文本）
  - `tests/unit/test_image_captioner_fallback.py`
- **验收标准**：
  - 启用模式：存在 image_refs 时会生成 caption 并写入 metadata（测试中用 mock Vision LLM 断言调用与输出）。
  - 降级模式：当配置禁用或异常时，chunk 保留 image_refs，但不生成 caption 且标记 `has_unprocessed_images`。
- **测试方法**：`pytest -q tests/unit/test_image_captioner_fallback.py`。

### C8：DenseEncoder（依赖 libs.embedding）
- **目标**：实现 `dense_encoder.py`，把 chunks.text 批量送入 `BaseEmbedding`。
- **修改文件**：
  - `src/ingestion/embedding/dense_encoder.py`
  - `tests/unit/test_dense_encoder.py`
- **验收标准**：encoder 输出向量数量与 chunks 数量一致，维度一致。
- **测试方法**：`pytest -q tests/unit/test_dense_encoder.py`。

### C9：SparseEncoder（BM25 统计与输出契约）
- **目标**：实现 `sparse_encoder.py`：对 chunks 建立 BM25 所需统计（可先仅输出 term weights 结构，索引落地下一步做）。
- **修改文件**：
  - `src/ingestion/embedding/sparse_encoder.py`
  - `tests/unit/test_sparse_encoder.py`
- **验收标准**：输出结构可用于 bm25_indexer；对空文本有明确行为。
- **测试方法**：`pytest -q tests/unit/test_sparse_encoder.py`。

### C10：BatchProcessor（批处理编排）
- **目标**：实现 `batch_processor.py`：将 chunks 分 batch，驱动 dense/sparse 编码，记录批次耗时（为 trace 预留）。
- **修改文件**：
  - `src/ingestion/embedding/batch_processor.py`
  - `tests/unit/test_batch_processor.py`
- **验收标准**：batch_size=2 时对 5 chunks 分成 3 批，且顺序稳定。
- **测试方法**：`pytest -q tests/unit/test_batch_processor.py`。

---

**━━━━ 存储阶段分界线：以下任务负责将编码结果持久化 ━━━━**

> **说明**：C8-C10完成了Dense和Sparse的编码工作，C11-C13负责将编码结果存储到不同的后端。
> - **C11 (BM25Indexer)**：处理Sparse编码结果 → 构建倒排索引 → 存储到文件系统
> - **C12 (VectorUpserter)**：处理Dense编码结果 → 生成稳定ID → 存储到向量数据库
> - **C13 (ImageStorage)**：处理图片数据 → 文件存储 + 索引映射

---

### C11：BM25Indexer（倒排索引构建与持久化）
- **目标**：实现 `bm25_indexer.py`：接收 SparseEncoder 的term statistics输出，计算IDF，构建倒排索引，并持久化到 `data/db/bm25/`。
- **核心功能**：
  - 计算 IDF (Inverse Document Frequency)：`IDF(term) = log((N - df + 0.5) / (df + 0.5))`
  - 构建倒排索引结构：`{term: {idf, postings: [{chunk_id, tf, doc_length}]}}`
  - 索引序列化与加载（支持增量更新与重建）
- **修改文件**：
  - `src/ingestion/storage/bm25_indexer.py`
  - `tests/unit/test_bm25_indexer_roundtrip.py`
- **验收标准**：
  - build 后能 load 并对同一语料查询返回稳定 top ids
  - IDF计算准确（可用已知语料对比验证）
  - 支持索引重建与增量更新
- **测试方法**：`pytest -q tests/unit/test_bm25_indexer_roundtrip.py`。
- **备注**：本任务完成Sparse路径的最后一环，为D3 (SparseRetriever) 提供可查询的BM25索引。

### C12：VectorUpserter（向量存储与幂等性保证）
- **目标**：实现 `vector_upserter.py`：接收 DenseEncoder 的向量输出，生成稳定的 `chunk_id`，并调用 VectorStore 进行幂等写入。
- **核心功能**：
  - 生成确定性 chunk_id：`hash(source_path + chunk_index + content_hash[:8])`
  - 调用 `BaseVectorStore.upsert()` 写入向量数据库
  - 保证幂等性：同一内容重复写入不产生重复记录
- **修改文件**：
  - `src/ingestion/storage/vector_upserter.py`
  - `tests/unit/test_vector_upserter_idempotency.py`
- **验收标准**：
  - 同一 chunk 两次 upsert 产生相同 id
  - 内容变更时 id 变更
  - 支持批量 upsert 且保持顺序
- **测试方法**：`pytest -q tests/unit/test_vector_upserter_idempotency.py`。
- **备注**：本任务完成Dense路径的最后一环，为D2 (DenseRetriever) 提供可查询的向量数据库。

### C13：ImageStorage（图片文件存储与索引表契约）
- **目标**：实现 `image_storage.py`：保存图片到 `data/images/{collection}/`，并使用 **SQLite** 记录 image_id→path 映射。
- **修改文件**：
  - `src/ingestion/storage/image_storage.py`
  - `tests/unit/test_image_storage.py`
- **验收标准**：保存后文件存在；查找 image_id 返回正确路径；映射关系持久化在 `data/db/image_index.db`。
- **技术方案**：
  - 复用项目已有的 SQLite 架构模式（参考 `file_integrity.py` 的 `SQLiteIntegrityChecker`）
  - 数据库表结构：
    ```sql
    CREATE TABLE image_index (
        image_id TEXT PRIMARY KEY,
        file_path TEXT NOT NULL,
        collection TEXT,
        doc_hash TEXT,
        page_num INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    CREATE INDEX idx_collection ON image_index(collection);
    CREATE INDEX idx_doc_hash ON image_index(doc_hash);
    ```
  - 提供并发安全访问（WAL 模式）
  - 支持按 collection 批量查询
- **测试方法**：`pytest -q tests/unit/test_image_storage.py`。

### C14：Pipeline 编排（MVP 串起来）
- **目标**：实现 `pipeline.py`：串行执行（integrity→load→split→transform→encode→store），并对失败步骤做清晰异常。
- **修改文件**：
  - `src/ingestion/pipeline.py`
  - `tests/integration/test_ingestion_pipeline.py`
- **测试数据**：
  - **主测试文档**：`tests/fixtures/sample_documents/complex_technical_doc.pdf`
    - 8章节技术文档（~21KB）
    - 包含3张嵌入图片（需测试图片提取和描述）
    - 包含5个表格（测试表格内容解析）
    - 多页多段落（测试完整分块流程）
  - **辅助测试**：`tests/fixtures/sample_documents/simple.pdf`（简单场景回归）
- **验收标准**：
  - 对 `complex_technical_doc.pdf` 跑完整 pipeline，成功输出：
    - 向量索引文件到 ChromaDB
    - BM25 索引文件到 `data/db/bm25/`
    - 提取的图片到 `data/images/` (SHA256命名)
  - Pipeline 日志清晰展示各阶段进度
  - 失败步骤抛出明确异常信息
- **测试方法**：`pytest -v tests/integration/test_ingestion_pipeline.py`。

### C15：脚本入口 ingest.py（离线可用）
- **目标**：实现 `scripts/ingest.py`，支持 `--collection`、`--path`、`--force`，并调用 pipeline。
- **修改文件**：
  - `scripts/ingest.py`
  - `tests/e2e/test_data_ingestion.py`
- **验收标准**：命令行可运行并在 `data/db` 产生产物；重复运行在未变更时跳过。
- **测试方法**：`pytest -q tests/e2e/test_data_ingestion.py`（尽量用临时目录）。