# DEV_SPEC

Version: `0.0.3`

## 目录

- [1. 项目概述](#1-项目概述)
  - [1.1 设计理念](#11-设计理念)
  - [1.2 项目定位](#12-项目定位)
- [2. 核心特点](#2-核心特点)
  - [2.1 RAG 策略与设计要点](#21-rag-策略与设计要点)
  - [2.2 全链路可插拔架构](#22-全链路可插拔架构)
  - [2.3 MCP 原生集成](#23-mcp-原生集成)
  - [2.4 多模态图像处理](#24-多模态图像处理)
  - [2.5 可观测性、可视化管理与评估体系](#25-可观测性可视化管理与评估体系)
- [3. 技术选型](#3-技术选型)
  - [3.1 RAG 核心流水线设计](#31-rag-核心流水线设计)
  - [3.2 MCP 服务设计](#32-mcp-服务设计)
  - [3.3 可插拔架构设计](#33-可插拔架构设计)
  - [3.4 可观测性与 Dashboard 设计](#34-可观测性与-dashboard-设计)
  - [3.5 多模态图片处理设计](#35-多模态图片处理设计)
- [4. 测试方案](#4-测试方案)
  - [4.1 TDD 原则](#41-tdd-原则)
  - [4.2 分层测试策略](#42-分层测试策略)
  - [4.3 RAG 质量评估](#43-rag-质量评估)
- [5. 系统架构与模块设计](#5-系统架构与模块设计)
  - [5.1 整体架构图](#51-整体架构图)
  - [5.2 完整目录结构树](#52-完整目录结构树)
  - [5.3 模块职责说明表](#53-模块职责说明表)
  - [5.4 数据流说明](#54-数据流说明)
  - [5.5 配置驱动设计示例](#55-配置驱动设计示例)
- [6. 项目排期](#6-项目排期)
  - [6.1 阶段原则](#61-阶段原则)
  - [6.2 分阶段任务清单](#62-分阶段任务清单)
  - [6.3 进度跟踪表](#63-进度跟踪表)
- [7. 开发规范补充约束](#7-开发规范补充约束)
  - [7.1 编码规范](#71-编码规范)
  - [7.2 异常处理规范](#72-异常处理规范)
  - [7.3 日志规范](#73-日志规范)
  - [7.4 文档规范](#74-文档规范)
  - [7.5 完成标准](#75-完成标准)

## 1. 项目概述

### 1.1 设计理念

本项目旨在构建一个模块化、可插拔、可观测、可评估的 Python RAG 系统，并将其封装为基于 STDIO Transport 的 MCP Server，供本地 Agent 以子进程方式直接调用知识库能力。

核心设计原则：

- 本地优先：默认单机运行，不依赖外部编排平台，不做 HTTP 部署。
- 开箱即用：最小可运行链路清晰，安装后即可完成摄取、检索、问答、评估。
- 模块可插拔：LLM、Embedding、Reranker、VectorStore、Splitter、Evaluator 均通过抽象接口和配置切换。
- 配置驱动：行为由 `settings.yaml` 决定，避免在业务代码中写死供应商与参数。
- 可观测：Ingestion 与 Query 两条链路都必须具备 Trace、结构化日志、Dashboard 可视化。
- 可测试：以 pytest + TDD 为主，要求每个阶段都有明确验收标准与测试闭环。

### 1.2 项目定位

该项目不是一个“只做问答 Demo”的脚手架，而是一个面向长期演进的知识库基础设施，具备以下定位：

- 面向 Agent：通过 MCP Tools 暴露标准能力，便于智能体按工具方式调用。
- 面向工程落地：强调目录清晰、边界明确、便于扩展和维护。
- 面向 RAG 质量：从分块、召回、融合、重排、回答生成到评估形成完整闭环。
- 面向本地部署：单机 SQLite + ChromaDB + Streamlit Dashboard 即可完成全流程。

## 2. 核心特点

### 2.1 RAG 策略与设计要点

#### 分块策略

本项目不采用简单定长切分，而是以“尽量保留语义完整性”为首要原则。首期落地以 `RecursiveCharacterTextSplitter` 为基础，但切分配置必须围绕文档结构进行调优，例如优先按标题、段落、列表、表格邻接区域和中文标点边界切分，而不是机械地按固定字符数截断。

设计目标如下：

- 优先保持一个 chunk 内主题单一、语义连续。
- 尽量避免将图文说明、标题与正文、表格标题与表格内容拆散。
- 通过 `chunk_size`、`chunk_overlap`、`separators` 配置实现可调优切分。
- 为后续升级到结构感知分块、语义边界分块、章节感知分块预留统一接口。

这种分块策略的价值在于：它直接决定召回质量上限。若 chunk 边界错误，后续混合检索和重排即使设计再强，也只能在低质量候选集合上工作。

#### 混合检索

检索层采用混合检索路线，而不是只依赖向量召回。系统默认同时执行两条检索路径：

- BM25 稀疏检索：适合命中关键词、专有名词、编号、公式符号、局部精确表达。
- Dense Embedding 稠密检索：适合理解语义近似、同义改写、跨表述检索。

两路结果通过 RRF（Reciprocal Rank Fusion）融合，而不是简单按分数拼接。这样做的原因是不同检索器的原始分数不可直接比较，RRF 可以更稳定地聚合来自不同召回通道的排序信号。

混合检索的收益包括：

- 降低单一路径漏召回风险。
- 对“关键词明确”与“语义表达模糊”的问题都更鲁棒。
- 便于后续扩展更多召回器，例如 query expansion、metadata retrieval、领域规则检索。

#### 重排策略

系统采用“两段式检索排序”设计：

1. 第一阶段做广覆盖粗排，目标是尽量召回足够多的候选 chunk。
2. 第二阶段做高精度精排，目标是将最适合回答问题的上下文提到前面。

精排层支持两类方案：

- Cross-Encoder：适合本地或特定模型下的高质量相关性判别。
- LLM Rerank：适合需要更强语义理解、复杂推理或面向长文本场景的重排。

该设计不是为了“堆模块”，而是为了明确粗排与精排的职责边界：

- 粗排解决“不要漏掉”
- 精排解决“不要排错”

### 2.2 全链路可插拔架构

项目采用“抽象基类 + 工厂模式 + 配置驱动”的统一机制，使所有关键能力都能通过 `settings.yaml` 切换，而无需改动业务代码。

#### LLM 调用层插拔

LLM 层必须通过统一接口封装，首批支持：

- OpenAI
- Qwen
- DeepSeek

调用层插拔的意义在于：

- 可以按成本、速度、回答风格切换回答模型。
- 可以将生成模型与重写模型、图片描述模型拆分配置。
- 可以在不改代码的情况下快速比较不同模型对回答质量与延迟的影响。

#### Embedding 模型插拔

Embedding 层必须独立于 LLM 层配置，避免把“生成模型选择”和“检索模型选择”绑死在一起。

设计要求：

- Embedding provider 独立抽象。
- 支持未来扩展 query/document 双塔差异化配置。
- 支持零代码切换不同 embedding 模型做召回效果对比。

这能保证开发者在做检索优化时，只需修改配置即可完成向量模型 A/B 测试，而无需侵入 Query Pipeline。

#### 重排模型插拔

重排器同样独立配置，至少支持：

- Cross-Encoder Reranker
- LLM Reranker

其设计目标是让开发者能够低成本回答以下问题：

- 是否需要 rerank
- rerank 用哪种模型
- `candidate_top_n` 和 `final_top_n` 应该如何设定

这些都应通过配置切换完成，而不是改动检索主链路代码。

#### Rag 组件插拔

除模型层外，RAG 编排本身也必须模块化。

可插拔范围包括：

- Loader
- 切分
- Transformation（元数据增强、文档重写、图文融合、图片描述注入）

设计原则：

- 每个组件只负责单一职责。
- Pipeline 负责编排，不负责具体 provider 细节。
- 新增组件时只需实现接口、注册工厂、补充配置。

这样可以确保后续从 PDF 扩展到 Markdown、HTML、DOCX，或从简单文本增强扩展到领域知识注入时，不需要重写整个摄取链路。

#### 检索策略插拔

检索层不仅是 provider 可插拔，连“检索策略”本身也必须可切换。至少支持：

- 混合检索
- 纯向量检索
- 纯稀疏检索

这意味着系统能够在同一套文档、同一套评估数据集上，直接通过配置对比不同检索策略的效果、成本和响应时间。

#### 评估系统插拔

评估层需要支持多种 evaluator 并行存在，至少包括：

- Ragas
- 自定义指标体系
- 未来预留其他第三方评估框架

评估输出必须结构化存档，便于横向比较：

- 不同检索策略
- 不同 embedding 模型
- 不同 reranker
- 不同回答模型

这种设计确保开发者可以零代码修改即可进行 A/B 测试。系统应支持通过修改 `settings.yaml` 直接切换实验组合，然后将结果沉淀到统一评估记录中。

### 2.3 MCP 原生集成

项目从一开始就以 MCP Server 作为一等公民能力来设计，而不是在 RAG 完成后再额外包一层接口。

核心思路：

- 基于 Python 官方 MCP SDK 实现。
- 采用 STDIO Transport，本地子进程通信，不做 HTTP 暴露。
- 对外暴露标准化 tools，而不是暴露内部实现细节。

这种设计带来的价值：

- Agent 可以直接把知识库能力当作工具调用。
- RAG 系统与 Agent 之间采用更稳定的协议边界，而不是脆弱的自定义 CLI/HTTP 封装。
- 本地运行链路更轻，更符合“零外部服务依赖”的项目目标。

### 2.4 多模态图像处理

本项目的多模态能力采用 Image-to-Text 路线，而不是 CLIP 或多模态向量检索路线。

设计重点：

- 使用 Vision LLM 为图片生成可检索描述。
- 将图片描述缝合进 chunk 文本与元数据中。
- 让图片内容自然进入 BM25 与 Dense Retrieval 的统一检索语料。

这种方案的优点是：

- 架构简单，易于纳入现有文本 RAG 主链路。
- 对本地轻量部署更友好，不需要额外维护图像向量检索链路。
- 对“图表说明”“流程图摘要”“截图文字提取后问答”类场景足够实用。

### 2.5 可观测性、可视化管理与评估体系

系统不仅要“能跑”，还要“可追踪、可解释、可诊断、可比较”。

可观测性设计包括：

- Ingestion Trace
- Query Trace
- 结构化 JSON Lines 日志
- 各阶段耗时、输入输出摘要、错误信息和关键指标

可视化管理包括本地 Streamlit Dashboard，多页面展示：

- 系统总览
- 数据浏览
- Ingestion 管理
- Trace 查看
- 评估面板

评估体系则负责把“主观觉得效果不错”转化为“可重复比较的实验结果”，至少支持：

- hit_rate
- MRR
- Ragas 指标
- 自定义评估报表

三者合在一起的意义是：

- 发现问题时可以定位到具体链路阶段。
- 优化策略时可以量化前后收益。
- 做模型、检索器、重排器替换时可以形成可追踪的实验闭环。

## 3. 技术选型

### 3.1 RAG 核心流水线设计

#### 3.1.1 Ingestion Pipeline

目标：将原始文档稳定转化为可检索、可解释、可回溯、可增量更新的知识单元，并为 Dashboard 管理面板提供文档级生命周期管理能力。

该链路不采用“脚本堆叠”的一次性实现，而是采用自定义抽象接口驱动的可插拔架构。核心接口统一定义为：

- `BaseLoader`
- `BaseSplitter`
- `BaseTransform`
- `BaseEmbedding`
- `BaseVectorStore`

设计原则如下：

- Pipeline 只负责编排 `Loader -> Splitter -> Transform -> Embedding -> Upsert`，不感知具体 provider 细节。
- 各层通过明确的输入输出契约衔接，便于替换实现、插入 trace、做阶段级指标采集与失败重试。
- 所有阶段都必须支持可观测性，至少记录开始时间、结束时间、输入摘要、输出摘要、错误信息、重试次数与耗时。
- 向量存储在架构层统一收敛到 ChromaDB 官方 Python 包，避免首期同时维护多个 VectorStore 语义差异；但接口层仍通过 `BaseVectorStore` 保持可替换性。
- Embedding provider 需要与主流供应商良好适配，例如 OpenAI、Qwen、BGE、Jina、Voyage 等；具体接入方式由 `BaseEmbedding` 实现层负责屏蔽差异。

标准流程：

1. 文件发现与任务登记
2. 原始文件 SHA256 指纹计算
3. 基于 `ingestion_history` 的前置去重与增量判断
4. Loader 解析并输出 canonical Markdown `Document`
5. Splitter 将 Markdown `Document` 切分为稳定可定位的 `Chunk`
6. Transform 对 `Chunk` 做结构转换、语义增强与多模态补全，生成 `Smart Chunk`
7. Embedding 对新增内容执行双路向量化
8. Upsert 将向量与 payload 原子写入 Chroma
9. `ingestion_history`、文档状态与 trace 同步更新

##### 架构与职责边界

Ingestion Pipeline 以“阶段职责清晰、状态显式持久化、失败可恢复”为核心约束。

- Loader 负责“把原始文件变成标准文档表示”。
- Splitter 负责“把标准文档表示切成稳定边界的基础块”。
- Transform 负责“把基础块提升为检索友好的智能切片”。
- Embedding 负责“把智能切片转成可用于混合检索的向量表示”。
- Upsert 负责“把索引数据与业务 payload 幂等写入存储，并维护生命周期状态”。

其中每层都必须保持单一职责，避免将去重、切分、增强、向量写入混杂在同一类中，导致后续难以测试和难以做增量重试。

##### Loader

Loader 的输入是原始文件路径与基础任务上下文，输出是 canonical Markdown `Document`。当前首期仅实现 `pdf -> canonical Markdown subset` 转换，但接口必须为未来扩展 Markdown、HTML、DOCX、图片 OCR 等来源预留能力。

Loader 层关键要求：

- 在任何解析动作之前，先计算原始文件字节级 SHA256 哈希指纹。
- 基于该哈希检索 `ingestion_history` 表，默认采用 SQLite，可插拔存储路径为 `data/db/ingestion_history.db`。
- 若哈希已存在且未要求强制重建，则直接跳过后续解析与向量化流程。
- 若仅文件名或路径变化而内容哈希不变，则视为同一内容版本，更新来源映射关系，不重复解析和索引。
- Loader 需要在解析阶段同步抽取或补齐基础 metadata，至少包括：
  - `source_path`
  - `doc_type=pdf`
  - `page`
  - `title`
  - `heading_outline`
  - `images`
  - `source_sha256`
  - `document_id`
- 这些 metadata 不只是“附带字段”，而是后续切分定位、Transform 语义补全、Dashboard 回溯和删除操作的基础锚点。

Markdown 作为中间标准表示的意义在于：它能把 PDF 解析结果压缩为更稳定、更便于调试和重放的文本结构层。首期可使用 MarkItDown 作为默认 PDF 解析/转换引擎，例如标题、段落、列表、表格占位、图片引用等；但无论底层转换器如何更换，对外都统一输出 canonical Markdown `Document`。

##### Splitter

Splitter 输入必须是 Loader 输出的 Markdown `Document`，不得直接对原始 PDF 或任意字符串做隐式切分。首期统一采用 LangChain 的 `RecursiveCharacterTextSplitter`，但必须结合 Markdown 结构边界进行调优，而不是简单按固定字符数截断。

Splitter 输出为若干 `Chunk` 或 Document-like chunks，每个 chunk 都必须具备稳定、可重复计算的定位信息与来源信息，至少包含：

- `source`
- `chunk_index`
- `start_offset`
- `end_offset`

若实现使用等价定位字段，也必须满足以下能力：

- 能唯一回溯 chunk 在原始 Markdown 表示中的位置。
- 能在后续重跑时为相同内容生成稳定 `chunk_id`。
- 能支持 Dashboard 中的 chunk 浏览、定位与文档内跳转。

Splitter 的职责是生成“边界正确的基础块”，而不是做复杂语义增强。因此它应该尽可能保持确定性：

- 相同输入与相同配置必须产生相同切分结果。
- 配置变更应视为一次新的切分版本，以便触发受控重建。
- Chunk 元数据中必须保留来源文档标识、页码映射、标题路径等基础上下文，为后续 Transform 提供结构锚点。

##### Transform & Enrichment

Transform 层负责将 Splitter 产生的非结构化文本块，转化为结构化、富语义、面向检索优化的 `Smart Chunk`。这一层不是“简单加几个 metadata 字段”，而是从基础文本块到智能切片的真正语义提升层。

该层至少承担三类能力：

- 结构转换：将原始的 `String` 类型数据转化为强类型的 `Record/Object`，为下游检索提供字段级支持。
- 语义元数据注入：为 chunk 注入章节路径、主题标签、实体、关键词、摘要、适用场景、术语别名等检索增强信号。
- 多模态增强：把图片说明、图表摘要、OCR 结果或 Vision 模型生成的简述，按可控策略融合进 Smart Chunk 的文本与 metadata。

**核心增强策略**：
1. **智能重组 (Smart Chunking & Refinement)**：
  - 策略：利用 LLM 的语义理解能力，对上一阶段“粗切分”的片段进行二次加工。
  - 动作：合并在逻辑上紧密相关但被物理切断的段落，剔除无意义的页眉页脚或乱码（去噪），确保每个 Chunk 是自包含（Self-contained）的语义单元。
2. **语义元数据注入 (Semantic Metadata Enrichment)**：
  - 策略：在基础元数据（路径、页码）之上，利用 LLM 提取高维语义特征。
  - 产出：为每个 Chunk 自动生成 `Title`（精准小标题）、`Summary`（内容摘要）和 `Tags`（主题标签），并将其注入到 Metadata 字段中，支持后续的混合检索与精确过滤。
3. **多模态增强 (Multimodal Enrichment / Image Captioning)**：
  - 策略：扫描文档片段中的图像引用，调用 Vision LLM（如 GPT-4o）进行视觉理解。
  - 动作：生成高保真的文本描述（Caption），描述图表逻辑或提取截图文字。
  - 存储：将 Caption 文本“缝合”进 Chunk 的正文或 Metadata 中，打通模态隔阂，实现“搜文出图”。
**工程特性**：Transform 步骤设计为原子化与幂等操作，支持针对特定 Chunk 的独立重试与增量更新，避免因 LLM 调用失败导致整个文档处理中断。

##### Embedding

Embedding 层负责对 `Smart Chunk` 进行双路向量化，并通过批处理优化吞吐与成本。双路向量化包括：

- Dense Embeddings：语义向量，用于语义相似度检索。
- Sparse Embeddings：稀疏向量，用于关键词、术语、编号与局部精确匹配增强。

架构要求：

- `BaseEmbedding` 必须允许 Dense 与 Sparse 采用不同 provider / model。
- Embedding 批次大小、并发度、重试策略、限流策略需可配置。
- Embedding 输入应统一基于 `Smart Chunk`，而不是散落的自由文本，避免不同向量通道使用不一致语料。

Embedding 层必须做内容级去重，只对数据库中不存在的新内容哈希执行向量化：

- 以 `content_hash` 作为核心去重键，而不是文件名或路径。
- 当文件名变化但 chunk 内容未变时，直接复用已有向量与 payload。
- 当文档局部修改时，仅对受影响的新 chunk 重新向量化。

该策略的价值非常直接：能显著降低外部 Embedding API 调用成本，缩短重复摄取耗时，并保证增量更新时索引行为更可预测。

##### Upsert & Storage

Upsert 层负责将索引数据与业务数据统一写入向量数据库，默认且统一使用 Chroma 作为存储引擎。其写入对象不是“只有向量”，而是 All in One 的完整存储单元，同时包含：

1. **Index Data**：用于相似度计算的 Dense Vector 和 Sparse Vector。
2. **Payload Data**：完整 chunk 原始文本 `content` 及其 `metadata`。

`BaseVectorStore` 需要提供幂等 upsert 接口，并保证以下行为：

- 相同 `chunk_id` + 相同版本重复写入时不会生成重复索引。
- 相同 `content_hash` 的 chunk 在复用场景下可直接关联已有向量。
- 文档更新时可按 `document_id`、`source_sha256` 或版本号执行受控替换。

幂等性设计要求：

- 每个文档、chunk、内容版本都必须有稳定标识。
- 重复执行同一批 ingestion 任务不会导致重复索引、重复 payload、重复 trace 记录。
- Upsert 前应先做存在性检查或利用底层存储的覆盖语义，保证重复请求安全。

原子性要求：

- 以 Batch 为单位进行事务性写入。
- 只有当一个 batch 的向量与 payload 都成功持久化后，才更新该 batch 的 ingestion 状态。
- 任一 batch 失败时必须能回滚或标记为未完成，避免出现 metadata 已写入但向量缺失，或向量存在但文档状态未提交的不一致情况。

##### 文档生命周期管理

Ingestion 层不只是“写入索引”，还必须承担完整的文档生命周期管理能力，以支撑 Dashboard 中的文档浏览、状态查看、删除、重建和版本追踪。

最少需要支持以下能力：

- 文档注册：记录文档来源、哈希、状态、创建时间、最近摄取时间、版本号。
- 文档浏览：按文档查看基础信息、chunk 数、页数、最近状态、失败原因、处理耗时。
- 文档删除：根据 `document_id` 级联删除 Chroma 中的向量与 payload，以及 `ingestion_history` 中的映射记录，还有该文档关联的图片文件。
- 文档重建：支持指定文档强制重新执行 Loader/Splitter/Transform/Embedding/Upsert。
- 版本追踪：保留内容版本、配置版本和处理版本之间的关联，便于回溯“当前索引由哪次规则生成”。

建议维护的状态至少包括：

- `pending`
- `processing`
- `indexed`
- `skipped`
- `failed`
- `deleted`

##### 核心数据与实现约束

关键设计要求：

- 每个文档必须生成稳定的 `document_id`，每个 chunk 必须生成稳定的 `chunk_id`。
- 去重与增量判断必须前置到 Loader 之前，而不是在向量写入后再补做。
- Chroma 中存储的 payload 必须足以独立支持结果展示、引用回溯与 Dashboard 检视。
- Trace 必须覆盖从文件发现到最终 upsert 的所有阶段，并支持文档级与 chunk 级排障。
- 所有 provider 选择、阈值、批大小、切分参数、增强策略、向量化通道开关都必须配置化。

#### 3.1.2 Retrieval Pipeline

目标：给定已经完成上下文补全与指代消歧的用户 Query，稳定召回高相关上下文，并在检索质量、系统可用性、结果可解释性之间取得工程上可控的平衡。

Retrieval Pipeline 同样采用可插拔、可观测、可回退的架构。该链路的核心不是“堆更多检索器”，而是以受控的 Query 预处理、双路混合召回、稳健的过滤策略和可降级的精排机制，持续产出稳定候选集。

标准流程：

1. Query 预处理与请求校验
2. Query Trace 初始化
3. 关键词提取与稀疏查询表达式构建
4. 同义词 / 别名 / 缩写扩展
5. Dense / Sparse 双路并行召回
6. Metadata Pre-filter / Post-filter
7. RRF 融合与候选截断
8. 可选精排重排
9. Context 组装
10. Answer Generation
11. Citation 生成
12. Trace 落盘

##### 架构与职责边界

Retrieval Pipeline 以“召回覆盖优先、排序稳定、过滤可解释、失败可回退”为核心约束。

- Query 预处理负责生成适合进入稀疏与稠密路线的查询表示。
- Dense / Sparse Retriever 负责并行召回，不在此阶段混入回答生成逻辑。
- Fusion 负责统一聚合异构召回信号，不依赖原始分数绝对值。
- Reranker 负责候选精排，但不能成为系统可用性的单点依赖。
- Context Builder 与 Answer Generator 只消费最终候选，不反向耦合底层检索细节。

这意味着 Retrieval 层必须显式区分“召回”“过滤”“融合”“精排”“回答”几个阶段，避免将 query 改写、metadata 过滤和 rerank 逻辑散落在多个组件中，导致结果难以解释。

##### Query 预处理

系统默认假设输入 Query 在进入 Retrieval Pipeline 前，已经完成基础上下文补全与指代消歧。因此本节不再把“多轮会话改写”作为检索主链路的一部分，而是聚焦于为混合检索生成稳定的查询表示。

Query 预处理至少包含：

- 空白归一化与字符标准化
- `top_k`、`filters`、`collection` 等请求参数校验
- 可选语言识别与领域术语规范化
- 关键词提取
- 受控 query 扩展

关键词提取要求使用 NLP 工具从 Query 中提取关键实体、术语与动词，并进行停用词过滤，最终生成面向稀疏检索的 token 列表。这里的目标不是生成“更长的 query”，而是提取真正能够驱动 BM25 命中的核心信号，例如实体名、产品名、错误码、动作词、章节名、缩写等。

query 扩展采用受控策略，而不是无界发散。系统支持 Synonym / Alias Expansion，即同义词、别名、缩写扩展，但默认策略为：

- 扩展只融入稀疏检索表达式。
- 稠密检索保持单次执行。
- 默认不为每个同义词额外发起独立的向量检索请求。

这一策略的理由是明确的：Sparse Route 更适合吸收同义词、别名和缩写带来的词面增益；Dense Route 本身已经具备一定语义泛化能力，若为每个扩展词额外做 embedding 检索，成本会迅速上升，且容易引入排序噪声。

##### Sparse Route

Sparse Route 默认采用 BM25 作为关键词召回路径。它接收“原始关键词 + 同义词 / 别名 / 缩写扩展”的统一查询表达式，逻辑上按 `OR` 扩展，但应允许为原始关键词赋予更高权重，以降低语义漂移风险。

关键要求：

- 稀疏检索只执行一次，而不是针对每个扩展词分别发起检索。
- 检索对象粒度为 chunk 或 Smart Chunk。
- 支持与 metadata filter 协同工作。
- 对高置信关键词、原始实体词、错误码、编号等应具备更高权重或等价保护机制。

这种设计的目标是：在保持关键词覆盖面的同时，尽量不因为扩展词过多而把候选集引向错误主题。

##### Dense Route

Dense Route 使用原始 Query，或轻度改写后的语义 Query，生成一次 embedding，并只执行一次稠密检索。默认不为每个同义词、别名或缩写重复触发额外向量请求。

Dense Route 标准过程：

1. 生成 Query Embedding
2. 在向量库中执行相似度检索，默认采用 Cosine Similarity
3. 返回 Top-N 语义候选

关键要求：

- Dense 检索使用与索引侧一致或兼容的 embedding 配置。
- 支持 collection 限定与可前置的 metadata filter。
- 必须记录查询向量生成耗时、召回条数、过滤前后候选数量等指标。

Dense Route 的职责是弥补词面检索的覆盖不足，而不是替代整个检索层。因此它应尽量保持语义召回的简洁性和稳定性。

##### 双路混合检索与结果融合

Retrieval Pipeline 默认采用双路混合检索，并要求 Dense Route 与 Sparse Route 并行执行，以降低尾延迟并提高总召回覆盖。

并行召回定义如下：

- Dense Route：计算 Query Embedding -> 检索向量库（Cosine Similarity）-> 返回 Top-N 语义候选。
- Sparse Route：使用 BM25 算法 -> 检索倒排索引 -> 返回 Top-N 关键词候选。

结果融合固定采用 RRF（Reciprocal Rank Fusion），而不是直接比较 Dense / Sparse 的原始分数。原因在于不同检索路径的分数空间并不天然可比，直接加权求和往往会引入不稳定行为。

RRF 融合策略：

- 基于排名而不是分数绝对值做融合。
- 对同时被两路命中的候选给予自然提升。
- 对单路强命中候选提供兜底，降低单一模态缺陷导致的漏召回。

公式策略：

`Score = 1 / (k + Rank_Dense) + 1 / (k + Rank_Sparse)`

其中 `k` 为可配置平滑参数。该策略的价值不在于“数学更复杂”，而在于工程上更稳健，能在不同 provider、不同语料与不同检索通道质量波动下保持融合行为可预测。

##### Metadata Filtering Strategy

Metadata Filtering 的设计原则是：尽量坐在索引之前，无法前置则后置兜底。

若底层索引支持且过滤条件属于硬约束（Hard Filter），则应优先在 Dense / Sparse 检索阶段做 Pre-filter，以缩小候选集、降低检索成本并减少无关结果进入后续排序链路。例如：

- `collection`
- `document_id`
- `doc_type`
- 明确的租户隔离字段
- 已建立稳定索引的高质量结构化字段

但并非所有过滤都适合前置。对于底层索引不支持、字段缺失率高、质量不稳定或语义上不宜过早剪枝的过滤条件，应在 Rerank 前统一做 Post-filter，作为 safety net。

Post-filter 规则要求：

- 对缺失字段默认采取“宽松包含”策略，即 `missing -> include`，避免误杀潜在高价值召回。
- 过滤逻辑必须可解释，并在 trace 中记录命中过滤条件与淘汰原因。
- Post-filter 发生后，应重新统计剩余候选数量，为后续 rerank 与 context 组装提供输入。

##### Rerank

精排层必须通过抽象接口 `BaseReranker` 实现可插拔设计，以便独立替换重排后端，而不影响前面的混合召回主链路。

`BaseReranker` 至少需要支持以下模式：

- `cross_encoder`
- `llm_reranker`
- `disabled`

设计要求：

- 粗排阶段默认取 RRF 融合后的 `candidate_top_n`。
- 若配置 `cross_encoder`，则使用交叉编码器做高精度相关性重排。
- 若配置 `llm_reranker`，则允许使用更强语义理解能力做精排，但必须受并发、超时和成本控制。
- 若关闭重排，则直接使用 RRF Top-K 进入 context 组装。

Rerank 的职责是“在高质量候选集合上做更精细排序”，而不是补救前面完全失败的召回。因此候选集质量仍以前面的双路召回和过滤策略为基础。

##### 可用性回退与失败处理

精排层绝不能成为 Retrieval Pipeline 的单点故障源。当重排不可用、超时或执行失败时，系统必须自动回退到融合阶段的排序结果，也就是直接使用 RRF Top-K，确保系统可用性与结果稳定性。

最低保障要求：

- reranker 初始化失败时，系统仍可使用混合召回链路提供结果。
- reranker 单次调用超时或报错时，请求级自动降级到 RRF 结果。
- 回退行为必须在 trace 和日志中明确记录，便于后续排障与容量规划。

这类降级设计非常关键。对于问答系统而言，“稍弱但稳定的结果”通常优于“理论更强但经常失败的精排”。

##### 回答生成与返回结构

回答生成阶段必须消费最终排序后的候选 chunk，并显式保留来源信息。默认采用“基于证据回答，不编造”的回答模板，并提供无答案兜底。

返回结果至少包括：

- `answer`
- `citations`
- `retrieved_chunks`
- `trace_id`
- `debug_info`

其中 `debug_info` 应按配置裁剪，但在调试模式下建议至少包含 query 预处理结果、Dense / Sparse 召回摘要、filter 命中情况、融合前后排名与是否发生 rerank 回退。

##### 核心数据与实现约束

关键设计要求：

- Query 输入默认已经完成上下文补全与指代消歧，检索链路不重复承担会话改写主职责。
- 稀疏检索对“关键词 + 扩展词”只执行一次查询；稠密检索默认只执行一次 embedding 检索。
- 融合策略固定为 RRF，`k` 必须配置化。
- Metadata Filtering 必须优先前置，无法前置时通过 Post-filter 做安全兜底。
- Reranker 必须通过 `BaseReranker` 实现可插拔，并支持关闭。
- 当精排不可用时必须稳定回退到 RRF Top-K。
- Trace 必须覆盖 query 预处理、双路召回、过滤、融合、精排、回退和回答生成全阶段。

### 3.2 MCP 服务设计

MCP Server 采用 Python 官方 MCP SDK，通信方式固定为 STDIO Transport，本地以子进程方式运行，不提供 HTTP 服务。

设计要求：

- 入口为单独 server 进程，例如 `python -m src.mcp_server.server`
- Server 生命周期清晰：
  - 启动配置加载
  - 组件装配
  - tools 注册
  - STDIO 监听
- tool 逻辑与底层业务逻辑解耦，tool 层仅做参数解析、调用和结果包装
- 返回协议需要兼顾机器可解析性与人类可读性，尤其是引用信息必须做到透明、稳定、可展示。

#### 3.2.1 引用透明

MCP 对外返回结果时，应将“引用”视为一等公民能力，而不是回答文本的附属信息。推荐在返回的 `structuredContent` 中采用统一的 Citation 数据格式，同时在 `content` 数组中提供一份人类可读的 Markdown 回答，并以 `[1]`、`[2]` 这类编号标注引用。

这样做的目标是实现引用透明：

- 对支持结构化解析的 Client，`structuredContent` 可直接提供 citation 列表、chunk_id、document_id、source_path、页码范围、片段摘要等字段。
- 对不解析结构化内容、只展示 `content` 文本的 Client，仍能通过 Markdown 回答中的 `[1]` 标注向用户暴露清晰引用。
- 编号体系必须在 `structuredContent` 与 `content` 之间保持一致，避免人类看到的 `[1]` 与结构化 citation 中的第一条记录不对应。
- Citation 必须稳定可回溯，至少能够定位到来源文档、chunk 或等价证据单元。

推荐的 `structuredContent.citations` 至少包含：

- `index`
- `document_id`
- `chunk_id`
- `source_path`
- `page_range`
- `section_title`
- `snippet`

`content` 中的 Markdown 回答建议遵循以下原则：

- 正文内联使用 `[1]`、`[2]` 等引用标记。
- 回答末尾附引用列表或来源摘要。
- 当无足够证据回答时，应显式说明，并避免伪造引用。

#### 3.2.2 对外工具

首批 MCP Tools 与扩展工具统一采用表格化描述，便于后续维护、评审与 Dashboard / Client 对齐：

| 工具名称 | 功能描述 | 典型输入参数 | 输出特点 |
| --- | --- | --- | --- |
| `query_knowledge_hub` | 执行知识库检索、融合、重排与回答生成，是面向 Agent 的核心查询入口。 | `query`, `collection`, `top_k`, `filters`, `return_debug` | 返回带引用的回答、`structuredContent.citations`、`trace_id`、检索摘要；`content` 中同时提供带 `[1]` 标注的 Markdown 回答。 |
| `list_collections` | 列出当前可用知识库集合及其基础统计信息，供 Agent 做集合发现与选择。 | 可选过滤条件、分页参数 | 返回 collection 列表、文档数、chunk 数、更新时间等摘要信息，适合做浏览和预检查。 |
| `get_document_summary` | 查看指定文档的摘要、结构概览与最新摄取状态。 | `document_id` | 返回文档摘要、章节概览、关键元数据、最新 ingestion 状态，便于定位和回溯。 |
| `ingest_documents` | 触发文档摄取任务，支持新文档入库、增量跳过或强制重建。 | `paths`, `collection`, `force_rebuild`, `options` | 返回任务受理结果、文档级状态、跳过/失败摘要与 `trace_id`，适合编排批量摄取。 |
| `get_trace_detail` | 查询指定 trace 的详细执行过程，用于排障和性能分析。 | `trace_id` | 返回阶段耗时、输入输出摘要、错误信息、回退记录等可观测数据。 |
| `evaluate_collection` | 对指定集合执行检索或问答评估，产出结构化评估结果。 | `collection`, `dataset`, `metrics`, `eval_options` | 返回评估指标、实验配置、样本级结果摘要，可用于 A/B 对比与回归分析。 |

#### 3.2.3 MCP 分层建议

MCP 分层建议：

- `server.py`：Server 初始化与 tools 注册
- `tool_registry.py`：tool 声明与注册中心
- `protocol_handler.py`：参数验证与统一响应包装
- `tools/`：各 tool 适配器

### 3.3 可插拔架构设计

#### 3.3.1 抽象组件

项目采用“抽象基类 + 工厂模式 + 配置驱动”的统一设计，使各核心组件都能在不改动上层业务代码的前提下完成替换、扩展与回退。抽象层的目标不是单纯做一层包装，而是收敛输入输出契约、统一错误语义，并为 Trace、测试和 A/B 实验提供稳定边界。

以下组件必须定义抽象基类或协议接口：

- `BaseLoader`
- `BaseLLM`
- `BaseEmbedding`
- `BaseReranker`
- `BaseVectorStore`
- `BaseSplitter`
- `BaseEvaluator`

统一要求：

- 构造函数只接收结构化配置对象，不直接依赖全局变量。
- 对外暴露稳定的方法签名，避免上层编排逻辑感知底层实现差异。
- 错误抛出统一使用项目内异常体系，便于日志归一化和调用方处理。
- 通过统一配置文件（如 `settings.yaml`）指定各组件的具体后端，做到“改配置不改代码”。
- 使用工厂函数根据配置动态实例化对应实现类，实现统一装配入口。
- 抽象层需要为可观测性预留挂点，支持记录 provider、耗时、重试、降级和失败原因。

通用结构示意：

```text
业务代码
  │
  ▼
<Component>Factory.get_xxx()  ← 读取 settings.yaml
  │
  ├─→ ImplementationA()
  ├─→ ImplementationB()
  └─→ ImplementationC()
      │
      ▼
   统一实现抽象接口
```

#### 3.3.2 LLM 与 Embedding 抽象

LLM 与 Embedding 需要独立抽象，避免“生成模型选择”和“检索模型选择”耦合在一起。无论底层 provider 是 OpenAI、Qwen、DeepSeek、BGE、Jina、Voyage 还是其他实现，上层调用代码都应保持一致。

关键抽象如下：

- `LLMClient`：暴露 `chat(messages) -> response` 方法，屏蔽不同 Provider 的认证方式、请求结构、参数命名和响应格式差异。
- `EmbeddingClient`：暴露 `embed(texts) -> vectors` 方法，统一处理批量请求、并发控制、重试逻辑与向量维度归一化。

实现层采用 `BaseLLM` / `BaseEmbedding` 抽象基类，配合工厂模式统一装配：

- `llm_factory.py`：根据 `settings.yaml` 中的 `provider`、`model`、超参数和认证配置，返回对应的 `BaseLLM` 实现。
- `embedding_factory.py`：根据配置返回 Dense、Sparse 或双路 Embedding 实现，保证上层调用入口一致。

该设计的直接收益是：

- Query Pipeline、Rerank、Transform 等上层模块不需要感知具体模型供应商。
- 模型切换、限流策略调整、认证方式变化只影响实现层，不影响业务编排。
- 便于在同一套业务代码下做不同模型的效果、成本与延迟对比。

#### 3.3.3 检索策略抽象

检索链路中的向量数据库、Embedding、分块器与召回策略都应通过统一接口进行建模。项目为 `BaseVectorStore`、`BaseEmbedding`、`BaseSplitter` 等核心组件定义抽象基类，不同实现只需遵循同一接口即可无缝替换。

每个抽象层都配套工厂函数，例如 `embedding_factory.py`、`splitter_factory.py`、`vector_store_factory.py`，根据 `settings.yaml` 中的配置字段自动实例化对应实现，实现“改配置不改代码”的切换体验。

逐项设计如下：

- splitter：默认采用 LangChain 的 `RecursiveCharacterTextSplitter` 进行切分。如需切换为 `SentenceSplitter`、`SemanticSplitter` 或自定义切分器，只需实现 `BaseSplitter` 接口并在配置中指定即可。
- 向量数据库：定义统一的 `BaseVectorStore` 抽象接口，暴露 `.add()`、`.query()`、`.delete()` 等方法。所有向量数据库后端只需实现该接口即可插拔替换，并由 `VectorStoreFactory` 根据配置自动选择具体实现。项目首选 ChromaDB 作为默认向量数据库。
- 向量编码策略：定义 `BaseEmbedding` 抽象接口，支持不同 Embedding 模型的可插拔替换。项目当前采用双路编码（Dense + Sparse）策略。存储时，Dense Vector 和 Sparse Vector 与 Chunk 原文、Metadata 一起原子化写入向量数据库，确保检索时可同时利用两种向量。
- 召回策略：默认采用混合召回 + 精排（Hybrid + Rerank）策略，同时支持关闭精排，或切换为纯稠密召回、纯稀疏召回等模式。所有策略均通过配置切换，而不是通过条件分支散落在业务代码中。

该层的目标是让系统不仅“后端可替换”，连检索方法论本身也能被统一编排、统一评估和统一观测。

#### 3.3.4 评估框架抽象

评估层定义统一的 `BaseEvaluator` 接口，暴露 `evaluate(query, retrieved_chunks, generated_answer, ground_truth) -> metrics` 方法。所有评估框架都实现该接口，并输出标准化的指标字典，避免上层处理逻辑绑定某一评估工具的原始返回格式。

可选评估后端包括：

- `ragas`
- `DeepEval`
- 自定义指标实现

评估模块采用组合模式设计，可同时挂载多个 Evaluator，对同一批样本并行执行评估并生成综合报告。典型配置示例如下：

```yaml
evaluation:
  backends: [ragas, custom_metrics]
```

这种抽象方式可以保证：

- 新增评估框架时只需实现 `BaseEvaluator`，无需修改评估主流程。
- 不同评估结果能够沉淀为统一结构，便于横向比较与长期追踪。
- 检索、生成、重排策略的实验结果可以在同一套报告体系中汇总分析。

#### 3.3.5 配置管理与切换流程

- **配置文件结构示例** (`config/settings.yaml`)：
	```yaml
	llm:
	  provider: azure  # azure | openai | qwen | deepseek
	  model: gpt-5  # gpt-5 | gpt-4 | qwen
	  # provider-specific configs...
	
	embedding:
	  provider: openai
	  model: text-embedding-3-small
	
	vector_store:
	  backend: chroma  # chroma | qdrant | pinecone
	
	retrieval:
	  sparse_backend: bm25  # bm25 | elasticsearch
	  fusion_algorithm: rrf  # rrf | weighted_sum
	  rerank_backend: cross_encoder  # none | cross_encoder | llm
	
	evaluation:
	  backends: [ragas, custom_metrics]
	
	dashboard:
	  enabled: true
	  port: 8501
	  traces_dir: ./logs
	```

- **切换流程**：

	1. 修改 `settings.yaml` 中对应组件的 `backend` / `provider` 字段。
	2. 确保新后端的依赖已安装、凭据已配置。
	3. 重启服务，工厂函数自动加载新实现，无需修改业务代码。

### 3.4 可观测性与 Dashboard 设计

#### 3.4.1 设计理念

可观测性层的目标不是“额外补几条日志”，而是为系统建立可回放、可分析、可比较的运行事实记录。设计上遵循以下原则：

- 双链路全覆盖追踪：同时覆盖 Ingestion 与 Query 两条主链路，避免系统只能看到“问答结果”，却无法解释“文档如何进入索引”。
- 透明可回溯：每次请求都必须具备唯一 `trace_id`，能回溯到阶段级耗时、输入输出摘要、调用组件、失败原因与关键指标。
- 与业务解耦：追踪逻辑不能污染核心 Pipeline 代码，应通过统一 Trace Manager、装饰器、中间件或事件钩子插入，避免业务代码里散落日志拼接逻辑。
- 结构化日志 + 本地 Dashboard：底层以 JSON Lines 记录结构化 Trace，前端以本地 Dashboard 做可视化展示，不依赖外部托管平台。
- 自动根据可插拔组件渲染：由于系统中的 Loader、Splitter、Embedding、Reranker、VectorStore、Evaluator 都是可插拔的，Dashboard 不应写死页面字段，而应根据 Trace 中记录的组件名称、阶段类型与指标动态渲染。

该设计确保系统不仅能“知道失败了”，还能回答以下问题：

- 失败发生在哪个阶段、哪个 provider、哪次配置版本。
- 同一查询在不同检索策略下耗时和效果有何差异。
- 某次文档摄取为何被跳过、重建或部分失败。

#### 3.4.2 追踪数据结构

系统定义两类 Trace 记录，分别覆盖查询与摄取两条链路：

- `QueryTrace`：记录一次查询从 Query Rewrite、Hybrid Retrieval、Rerank、Prompt Assemble 到 Answer Generation 的完整过程。
- `IngestionTrace`：记录一次摄取从文件发现、Loader、Splitter、Transform、Embedding、Upsert 到状态提交的完整过程。

两类 Trace 都必须共享统一的顶层结构，以便 Dashboard 统一读取和过滤。Trace 的持久化粒度为“单次请求一条完整记录”，阶段详情以内嵌 `stages` 列表方式保存，而不是将每个阶段拆成独立日志行。

**A. Query Trace（查询追踪）**

每次查询请求生成唯一的 `trace_id`，记录从 Query 输入到 Response 输出的全过程：

**基础信息**：
- `trace_id`：请求唯一标识
- `trace_type`：`"query"`
- `start_time` / `end_time` / `duration_ms`：请求开始时间、结束时间与总耗时
- `user_query`：用户原始查询
- `collection`：检索的知识库集合
- `status`：本次请求状态

**各阶段详情 (Stages)**：

| 阶段 | 记录内容 |
|-----|---------|
| **Query Processing** | 原始 Query、改写后 Query（若有）、提取的关键词、method、耗时 |
| **Dense Retrieval** | 返回的 Top-N 候选及相似度分数、provider、耗时 |
| **Sparse Retrieval** | 返回的 Top-N 候选及 BM25 分数、method、耗时 |
| **Fusion** | 融合后的统一排名、algorithm、耗时 |
| **Rerank** | 重排后的最终排名及分数、backend、是否触发 Fallback、耗时 |
| **Generation（可选）** | Prompt 摘要、LLM provider、模型名称、首 token 耗时、答案摘要、总耗时 |

**汇总指标**：
- `duration_ms`：端到端总耗时
- `top_k_results`：最终返回的 Top-K 文档 ID
- `error`：异常信息（若有）

**评估指标 (Evaluation Metrics)**：
- `context_relevance`：召回文档与 Query 的相关性分数
- `answer_faithfulness`：生成答案与召回文档的一致性分数（若有生成环节）

**B. Ingestion Trace（摄取追踪）**

每次文档摄取生成唯一的 `trace_id`，记录从文件加载到存储完成的全过程：

**基础信息**：
- `trace_id`：摄取唯一标识
- `trace_type`：`"ingestion"`
- `start_time` / `end_time` / `duration_ms`：摄取开始时间、结束时间与总耗时
- `source_path`：源文件路径
- `collection`：目标集合名称
- `status`：本次摄取状态

**各阶段详情 (Stages)**：

| 阶段 | 记录内容 |
|-----|---------|
| **Load** | 文件大小、解析器（method: markitdown）、提取的图片数、耗时 |
| **Split** | splitter 类型（method）、产出 chunk 数、平均 chunk 长度、耗时 |
| **Transform** | 各 transform 名称与处理详情（refined/enriched/captioned 数量）、LLM provider、耗时 |
| **Embed** | embedding provider、batch 数、向量维度、dense + sparse 编码耗时 |
| **Upsert** | 存储后端（method: chroma）、upsert 数量、BM25 索引更新、图片存储、耗时 |

**汇总指标**：
- `duration_ms`：端到端总耗时
- `total_chunks`：最终存储的 chunk 数量
- `total_images`：处理的图片数量
- `skipped`：跳过的文件/chunk 数（已存在、未变更等）
- `error`：异常信息（若有）

#### 3.4.3 技术方案：结构化日志 + 本地 Web Dashboard

可观测性方案采用“结构化日志存储 + 本地可视化 UI”两层设计。

实现架构：

```text
RAG Pipeline / MCP Tools
        │
        ▼
Trace Collector (装饰器 / 回调 / Hook)
        │
        ▼
JSON Lines Trace Logs
logs/traces/query-YYYY-MM-DD.jsonl
logs/traces/ingestion-YYYY-MM-DD.jsonl
        │
        ▼
本地 Web Dashboard (Streamlit)
        │
        ▼
按 trace_id 查看单次完整链路与性能指标
```

底层日志方案：

- 基于 Python `logging` + JSON Formatter，将每次请求的 Trace 数据以 JSON Lines 格式追加写入本地文件。
- 每行对应一条完整请求记录，而不是零散文本片段；阶段级细节统一收敛在该记录的 `stages` 字段中。
- 文件建议按日期或链路类型滚动，例如：
  - `logs/traces/query-YYYY-MM-DD.jsonl`
  - `logs/traces/ingestion-YYYY-MM-DD.jsonl`
  - `logs/app/app-YYYY-MM-DD.jsonl`
- 结构化日志优先服务于机器读取与本地分析，因此禁止把关键信息只写在自由文本 message 中。

本地 Dashboard 方案：

- 基于 Streamlit 构建轻量级 Web UI，本地运行，不依赖 LangSmith 等外部 SaaS 平台。
- Dashboard 直接读取 JSONL、SQLite 与 Chroma 派生统计数据，提供交互式查询、筛选、聚合与可视化。
- 核心能力是按 `trace_id` 检索并展示单次请求的完整追踪链路，帮助开发者做性能定位、错误排查和策略对比；若某次 Query 未进入生成阶段，则页面应按实际 `stages` 动态渲染，而不是强制展示空白模块。

该技术路线的优势在于：

- 部署轻量，适合本地优先项目。
- 数据格式简单透明，便于调试、回放和离线分析。
- 与可插拔架构天然兼容，新增阶段或组件时只需扩展 Trace 字段与渲染规则。

#### 3.4.4 追踪机制实现

追踪机制建议通过统一的 Trace 基础设施实现，而不是让每个模块手工拼装日志。推荐实现方式如下：

- 在 Pipeline 入口生成 `trace_id`，并创建当前请求的 Trace 上下文。
- 通过 `TraceManager` 或等价组件统一管理阶段开始、阶段结束、异常捕获、指标聚合和最终落盘。
- 在 Loader、Splitter、Transform、Embedding、VectorStore、Retriever、Reranker、LLM、Evaluator 等抽象层边界自动打点，而不是在每个具体实现里重复写日志逻辑。
- 对每个阶段记录标准化的输入摘要、输出摘要、耗时、provider、重试次数和异常信息。
- 在请求结束时聚合阶段级信息，生成一条完整 Trace 记录并写入 JSONL。

推荐的实现职责拆分：

- `trace_manager.py`：Trace 上下文创建、阶段注册、结束收敛、落盘。
- `trace_models.py`：`QueryTrace`、`IngestionTrace`、`StageTrace` 等数据结构定义。
- `trace_logger.py`：`logging` 与 JSON Formatter 封装。
- `trace_utils.py`：输入输出摘要裁剪、异常序列化、时间统计、敏感字段脱敏。

指标记录建议至少覆盖：

- Ingestion：
  - 文档数
  - chunk 数
  - 平均 chunk 长度
  - 图片数
  - 跳过文档数
  - 各阶段耗时
- Retrieval / Query：
  - 稀疏召回数量
  - 稠密召回数量
  - 融合后数量
  - rerank 耗时
  - 首 token 耗时
  - 总响应耗时
- Evaluation：
  - `hit_rate`
  - `MRR`
  - `context_precision`
  - `answer_relevancy`

为避免追踪机制反向污染业务层，需要额外满足以下约束：

- 业务组件只暴露必要的摘要信息，不负责决定日志格式。
- Trace 写入失败不能拖垮主链路，最多降级为告警或 fallback 到简化日志。
- 所有敏感字段必须在落盘前脱敏，例如密钥、认证头、完整原始提示词中的敏感内容。

#### 3.4.5 Dashboard 功能设计

Dashboard 分为六大页面：

- `系统总览`
- `数据浏览器`
- `Ingestion管理`
- `Ingestion追踪`
- `Query追踪`
- `评估面板`

各页面职责建议如下：

- `系统总览`：展示集合数量、文档总量、chunk 总量、最近 Trace、组件配置摘要、最近异常与核心性能指标趋势。
- `数据浏览器`：浏览 collection、document、chunk 与 metadata，支持过滤、检索、定位和回溯到源文档。
- `Ingestion管理`：查看摄取任务、增量跳过情况、失败原因、重建入口、文档状态和版本信息。
- `Ingestion追踪`：按 `trace_id` 查看单次摄取链路的阶段时间线、阶段输入输出摘要、组件信息和失败节点。
- `Query追踪`：查看查询重写、Dense/Sparse 召回、融合、Rerank、生成等阶段细节，并支持不同 `trace_id` 间对比。
- `评估面板`：展示数据集维度的评估结果、指标趋势、provider 对比和 case-by-case 失败分析。

技术架构图：

```text
RAG Pipeline / MCP Tools
          │
          ▼
     TraceManager
          │
          ├─→ JSONL Trace Logs
          ├─→ SQLite Metadata
          └─→ Chroma Derived Stats
                   │
                   ▼
          Streamlit Dashboard
          ├─→ 系统总览
          ├─→ 数据浏览器
          ├─→ Ingestion管理
          ├─→ Ingestion追踪
          ├─→ Query追踪
          └─→ 评估面板
```

Dashboard 与 Trace 的数据关系如下：

- Dashboard 的 `Ingestion追踪` 页面直接消费 `IngestionTrace`，按 `trace_id` 渲染每个阶段的状态、耗时与错误信息。
- Dashboard 的 `Query追踪` 页面直接消费 `QueryTrace`，展示召回数量、融合结果、rerank 耗时、答案摘要等核心字段。
- `系统总览` 与 `评估面板` 对 Trace 数据做聚合统计，形成趋势图、分布图和策略对比图。
- `数据浏览器` 与 `Ingestion管理` 会将 Trace 与 SQLite / Chroma 中的文档实体关联起来，实现“从文档跳到追踪”与“从追踪回到文档”的双向回溯。

Dashboard 设计要求：

- 页面逻辑与底层服务分层，禁止在页面中直接写复杂业务逻辑。
- 优先读取 JSONL、SQLite 和 Chroma 的派生统计，不直接改写核心数据。
- 页面渲染应基于 Trace 中的组件元信息动态展示，避免写死 provider 或阶段名称。
- 所有图表和检索能力必须支持无外部联网环境下运行。

#### 3.4.6 配置示例

```yaml
observability:
  enabled: true
  
  # 日志配置
  logging:
    log_file: logs/traces.jsonl  # JSON Lines 格式日志文件
    log_level: INFO  # DEBUG | INFO | WARNING
  
  # 追踪粒度控制
  detail_level: standard  # minimal | standard | verbose

# Dashboard 管理平台配置
dashboard:
  enabled: true
  port: 8501                     # Streamlit 服务端口
  traces_dir: ./logs             # Trace 日志文件目录
  auto_refresh: true             # 是否自动刷新（轮询新 trace）
  refresh_interval: 5            # 自动刷新间隔（秒）
```

### 3.5 多模态图片处理设计
**目标：** 设计一套完整的图片处理方案，使 RAG 系统能够理解、索引并检索文档中的图片内容，实现"用自然语言搜索图片"的能力，同时保持架构的简洁性与可扩展性。

#### 3.5.1 设计理念与策略选型

多模态 RAG 的核心挑战在于：**如何让纯文本的检索系统"看懂"图片**。业界主要有两种技术路线：

| 策略 | 核心思路 | 优势 | 劣势 |
|-----|---------|------|------|
| **Image-to-Text (图转文)** | 利用 Vision LLM 将图片转化为文本描述，复用纯文本 RAG 链路 | 架构统一、实现简单、成本可控 | 描述质量依赖 LLM 能力，可能丢失视觉细节 |
| **Multi-Embedding (多模态向量)** | 使用 CLIP 等模型将图文统一映射到同一向量空间 | 保留原始视觉特征，支持图搜图 | 需引入额外向量库，架构复杂度高 |

**本项目选型：Image-to-Text（图转文）策略**

选型理由：
- **架构统一**：无需引入 CLIP 等多模态 Embedding 模型，无需维护独立的图像向量库，完全复用现有的文本 RAG 链路（Ingestion → Hybrid Search → Rerank）。
- **语义对齐**：通过 LLM 将图片的视觉信息转化为自然语言描述，天然与用户的文本查询在同一语义空间，检索效果可预期。
- **成本可控**：仅在数据摄取阶段一次性调用 Vision LLM，检索阶段无额外成本。
- **渐进增强**：未来如需支持"图搜图"等高级能力，可在此基础上叠加 CLIP Embedding，无需重构核心链路。

#### 3.5.2 图片处理全流程设计

图片处理贯穿 Ingestion Pipeline 的多个阶段，整体流程如下：

```
原始文档 (PDF/PPT/Markdown)
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  Loader 阶段：图片提取与引用收集                           │
│  - 解析文档，识别并提取嵌入的图片资源                        │
│  - 为每张图片生成唯一标识 (image_id)                       │
│  - 在文档文本中插入图片占位符/引用标记                       │
│  - 输出：Document (text + metadata.images[])             │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  Splitter 阶段：保持图文关联                               │
│  - 切分时保留图片引用标记在对应 Chunk 中                     │
│  - 确保图片与其上下文段落保持关联                            │
│  - 输出：Chunks (各自携带关联的 image_refs)                │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  Transform 阶段：图片理解与描述生成                         │
│  - 调用 Vision LLM 对每张图片生成结构化描述                  │
│  - 将描述文本注入到关联 Chunk 的正文或 Metadata 中           │
│  - 输出：Enriched Chunks (含图片语义信息)                  │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  Storage 阶段：双轨存储                                    │
│  - 向量库：存储增强后的 Chunk (含图片描述) 用于检索           │
│  - 文件系统/Blob：存储原始图片文件用于返回展示                │
└─────────────────────────────────────────────────────────┘
```

#### 3.5.3 各阶段技术要点

**1. Loader 阶段：图片提取与引用收集**

- **提取策略**：
  - 解析文档时识别嵌入的图片资源（PDF 中的 XObject、PPT 中的媒体文件、Markdown 中的 `![]()` 引用）。
  - 为每张图片生成全局唯一的 `image_id`（建议格式：`{doc_hash}_{page}_{seq}`）。
  - 将图片二进制数据提取并暂存，记录其在原文档中的位置信息。

- **引用标记**：
  - 在转换后的 Markdown 文本中，于图片原始位置插入占位符（如 `[IMAGE: {image_id}]`）。
  - 在 Document 的 Metadata 中维护 `images` 列表，记录每张图片的 `image_id`、原始路径、页码、尺寸等基础信息。

- **存储原始图片**：
  - 将提取的图片保存至本地文件系统的约定目录（如 `data/images/{collection}/{image_id}.png`）。
  - 仅保存需要的图片格式（推荐统一转换为 PNG/JPEG），控制存储体积。

**2. Splitter 阶段：保持图文关联**

- **关联保持原则**：
  - 图片引用标记应与其说明性文字（Caption、前后段落）尽量保持在同一 Chunk 中。
  - 若图片出现在章节开头或结尾，切分时应将其归入语义上最相关的 Chunk。

- **Chunk Metadata 扩展**：
  - 每个 Chunk 的 Metadata 中增加 `image_refs: List[image_id]` 字段，记录该 Chunk 关联的图片列表。
  - 此字段用于后续 Transform 阶段定位需要处理的图片，以及检索命中后定位需要返回的图片。

**3. Transform 阶段：图片理解与描述生成**

这是多模态处理的核心环节，负责将视觉信息转化为可检索的文本语义。

**双模型选型策略（推荐）**：

本项目采用**国内 + 国外双模型**方案，通过配置切换，兼顾不同部署环境和文档类型：

| 部署环境 | 主选模型 | 备选模型 | 说明 |
|---------|---------|---------|------|
| **国际化 / Azure 环境** | GPT-4o (Azure) | Qwen-VL-Max | 英文文档优先用 GPT-4o，中文文档可切换 Qwen-VL |
| **国内部署 / 纯中文场景** | Qwen-VL-Max | GPT-4o | 中文图表理解用 Qwen-VL，特殊需求可切换 GPT-4o |
| **成本敏感 / 大批量** | Qwen-VL-Plus | Gemini Pro Vision | 牺牲部分质量换取速度和成本 |

- **描述生成策略**：
  - **结构化 Prompt**：设计专用的图片理解 Prompt，引导 LLM 输出结构化描述，而非自由发挥。
  - **上下文感知**：将图片的前后文本段落一并传入 Vision LLM，帮助其理解图片在文档中的语境与作用。
  - **分类型处理**：针对不同类型的图片采用差异化的理解策略：

| 图片类型 | 理解重点 | Prompt 引导方向 |
|---------|---------|----------------|
| **流程图/架构图** | 节点、连接关系、流程逻辑 | "描述这张图的结构和流程步骤" |
| **数据图表** | 数据趋势、关键数值、对比关系 | "提取图表中的关键数据和结论" |
| **截图/UI** | 界面元素、操作指引、状态信息 | "描述截图中的界面内容和关键信息" |
| **照片/插图** | 主体对象、场景、视觉特征 | "描述图片中的主要内容" |

- **描述注入方式**：
  - **推荐：注入正文**：将生成的描述直接替换或追加到 Chunk 正文中的图片占位符位置，格式如 `[图片描述: {caption}]`。这样描述会被 Embedding 覆盖，可被直接检索。
  - **备选：注入 Metadata**：将描述存入 `chunk.metadata.image_captions` 字段。需确保检索时该字段也被索引。

- **幂等与增量处理**：
  - 为每张图片的描述计算内容哈希，存入 `processing_cache` 表。
  - 重复处理时，若图片内容未变且 Prompt 版本一致，直接复用缓存的描述，避免重复调用 Vision LLM。

**4. Storage 阶段：双轨存储**

- **向量库存储（用于检索）**：
  - 存储增强后的 Chunk，其正文已包含图片描述，Metadata 包含 `image_refs` 列表。
  - 检索时通过文本相似度即可命中包含相关图片描述的 Chunk。

- **原始图片存储（用于返回）**：
  - 图片文件存储于本地文件系统，路径记录在独立的 `images` 索引表中。
  - 索引表字段：`image_id`, `file_path`, `source_doc`, `page`, `width`, `height`, `mime_type`。
  - 检索命中后，根据 Chunk 的 `image_refs` 查询索引表，获取图片文件路径用于返回。

#### 3.5.4 检索与返回流程

当用户查询命中包含图片的 Chunk 时，系统需要将图片与文本一并返回：

```
用户查询: "系统架构是什么样的？"
    │
    ▼
Hybrid Search 命中 Chunk（正文含 "[图片描述: 系统采用三层架构...]"）
    │
    ▼
从 Chunk.metadata.image_refs 获取关联的 image_id 列表
    │
    ▼
查询 images 索引表，获取图片文件路径
    │
    ▼
读取图片文件，编码为 Base64
    │
    ▼
构造 MCP 响应，包含 TextContent + ImageContent
```

**MCP 响应格式**：

```json
{
  "content": [
    {
      "type": "text",
      "text": "根据文档，系统架构如下：...\n\n[1] 来源: architecture.pdf, 第5页"
    },
    {
      "type": "image",
      "data": "<base64-encoded-image>",
      "mimeType": "image/png"
    }
  ]
}
```

#### 3.5.5 质量保障与边界处理

- **描述质量检测**：
  - 对生成的描述进行基础质量检查（长度、是否包含关键信息）。
  - 若描述过短或 LLM 返回"无法识别"，标记该图片为 `low_quality`，可选择人工复核或跳过索引。

- **大尺寸/特殊图片处理**：
  - 超大图片在传入 Vision LLM 前进行压缩（保持宽高比，限制最大边长）。
  - 对于纯装饰性图片（如分隔线、背景图），可通过尺寸或位置规则过滤，不进入描述生成流程。

- **批量处理优化**：
  - 图片描述生成支持批量异步调用，提高吞吐量。
  - 单个文档处理失败时，记录失败的图片 ID，不影响其他图片的处理进度。

- **降级策略**：
  - 当 Vision LLM 不可用时，系统回退到"仅保留图片占位符"模式，图片不参与检索但不阻塞 Ingestion 流程。
  - 在 Chunk 中标记 `has_unprocessed_images: true`，后续可增量补充描述。

## 4. 测试方案

### 4.1 TDD 原则

开发默认遵循 TDD 或近似 TDD：

1. 先定义接口与验收行为
2. 先写失败测试
3. 实现最小功能
4. 重构并保持测试通过

要求：

- 新增核心模块必须先补单元测试
- 跨模块编排必须有集成测试
- 面向真实使用链路必须有 E2E 测试

### 4.2 分层测试策略

#### 单元测试

目标：验证单一组件行为。

覆盖对象：

- 配置加载与校验
- 工厂实例化
- chunk 切分逻辑
- RRF 融合
- rerank 结果排序
- SHA256 增量判断
- citation 生成
- MCP 参数校验

要求：

- 不依赖网络
- 外部模型调用必须 mock
- pytest 运行时间应尽量控制在秒级

#### 集成测试

目标：验证多个模块协作行为。

覆盖对象：

- PDF -> Markdown -> chunk -> embedding -> Chroma upsert
- sparse + dense + RRF + rerank
- MCP tool -> query engine
- trace 写入与 Dashboard 数据读取

要求：

- 使用临时目录和临时 SQLite/Chroma 数据
- 可用 fake provider 或 mock provider 替代真实大模型
- 输出需校验结构完整性

#### E2E 测试

目标：验证端到端用户场景。

核心场景：

- 导入一个 PDF 集合并完成问答
- 对已摄取文档执行增量跳过
- 使用 MCP tool 查询并返回引用
- 执行一轮评估并在 Dashboard 中可见

要求：

- 使用最小样例数据
- 可在 CI 中运行
- 避免依赖真实付费 API

### 4.3 RAG 质量评估

评估框架必须可插拔，支持：

- Ragas
- 自定义指标

最低指标集：

- `hit_rate`
- `mrr`
- `context_precision`
- `context_recall`
- `answer_relevancy`
- `faithfulness`（可在模型成本允许时启用）

评估要求：

- 评估数据集格式统一
- 支持离线批量评估
- 评估结果写入 SQLite 和 JSON 文件
- 每次评估记录 provider、参数、数据集版本、时间戳

## 5. 系统架构与模块设计

### 5.1 整体架构图

```text
                           +----------------------+
                           |   Agent / Client     |
                           +----------+-----------+
                                      |
                               MCP STDIO Tools
                                      |
                     +----------------v----------------+
                     |         MCP Server              |
                     | server / registry / tools       |
                     +----------------+----------------+
                                      |
                        +-------------v-------------+
                        |       Query Engine        |
                        | rewrite / retrieve /      |
                        | fuse / rerank / answer    |
                        +------+------+-------------+
                               |      |
               +---------------+      +-------------------+
               |                                      |
      +--------v--------+                    +--------v--------+
      | Sparse Retriever|                    | Dense Retriever |
      | BM25 Index      |                    | Embedding +     |
      |                 |                    | Chroma Search   |
      +--------+--------+                    +--------+--------+
               |                                      |
               +----------------+  +------------------+
                                |  |
                          +-----v--v------+
                          |   RRF +        |
                          |   Reranker     |
                          +-----+----------+
                                |
                          +-----v----------+
                          | Context Builder|
                          +-----+----------+
                                |
                          +-----v----------+
                          | LLM Answering  |
                          +----------------+


  +--------------------- Ingestion Pipeline ----------------------+
  | source docs -> MarkItDown -> image caption -> smart split -> |
  | LLM enrich -> embedding -> Chroma upsert -> SQLite metadata  |
  +---------------------------------------------------------------+

  +----------------------+     +------------------+     +------------------+
  | SQLite Metadata DB   |     | Chroma Vector DB |     | JSONL Trace Logs |
  +----------------------+     +------------------+     +------------------+
                    \                   |                          /
                     \                  |                         /
                      +-----------------v------------------------+
                      |       Streamlit Local Dashboard          |
                      +------------------------------------------+
```

### 5.2 完整目录结构树

```text
RagServer/
├── DEV_SPEC.md
├── README.md
├── pyproject.toml
├── pytest.ini
├── settings.yaml
├── .env.example
├── scripts/
│   ├── ingest.py
│   ├── query_cli.py
│   ├── run_mcp_server.py
│   ├── run_dashboard.py
│   └── evaluate.py
├── data/
│   ├── raw/
│   ├── processed/
│   ├── chroma/
│   ├── sqlite/
│   ├── eval/
│   └── cache/
├── logs/
│   ├── app/
│   └── traces/
├── src/
│   ├── __init__.py
│   ├── app/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── container.py
│   │   ├── exceptions.py
│   │   └── settings_models.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── interfaces/
│   │   │   ├── llm.py
│   │   │   ├── embedding.py
│   │   │   ├── reranker.py
│   │   │   ├── vector_store.py
│   │   │   ├── splitter.py
│   │   │   ├── evaluator.py
│   │   │   ├── retriever.py
│   │   │   ├── converter.py
│   │   │   └── image_captioner.py
│   │   ├── models/
│   │   │   ├── document.py
│   │   │   ├── chunk.py
│   │   │   ├── retrieval.py
│   │   │   ├── trace.py
│   │   │   └── evaluation.py
│   │   ├── factories/
│   │   │   ├── llm_factory.py
│   │   │   ├── embedding_factory.py
│   │   │   ├── reranker_factory.py
│   │   │   ├── vector_store_factory.py
│   │   │   ├── splitter_factory.py
│   │   │   ├── evaluator_factory.py
│   │   │   ├── converter_factory.py
│   │   │   └── image_caption_factory.py
│   │   ├── trace/
│   │   │   ├── trace_context.py
│   │   │   ├── trace_collector.py
│   │   │   └── events.py
│   │   └── utils/
│   │       ├── hashing.py
│   │       ├── ids.py
│   │       ├── time.py
│   │       └── jsonl.py
│   ├── providers/
│   │   ├── llm/
│   │   │   ├── openai_llm.py
│   │   │   ├── qwen_llm.py
│   │   │   └── deepseek_llm.py
│   │   ├── embedding/
│   │   │   ├── openai_embedding.py
│   │   │   ├── qwen_embedding.py
│   │   │   └── deepseek_embedding.py
│   │   ├── reranker/
│   │   │   ├── cross_encoder_reranker.py
│   │   │   └── llm_reranker.py
│   │   ├── splitter/
│   │   │   └── recursive_splitter.py
│   │   ├── vector_store/
│   │   │   └── chroma_store.py
│   │   ├── converter/
│   │   │   └── markitdown_converter.py
│   │   └── image_caption/
│   │       └── vision_llm_captioner.py
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── pipeline.py
│   │   ├── document_registry.py
│   │   ├── markdown_parser.py
│   │   ├── chunk_enricher.py
│   │   ├── image_extractor.py
│   │   └── services/
│   │       ├── fingerprint_service.py
│   │       ├── chunking_service.py
│   │       ├── embedding_service.py
│   │       └── upsert_service.py
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── pipeline.py
│   │   ├── query_rewriter.py
│   │   ├── bm25_retriever.py
│   │   ├── dense_retriever.py
│   │   ├── fusion.py
│   │   ├── rerank_pipeline.py
│   │   ├── context_builder.py
│   │   └── answer_generator.py
│   ├── storage/
│   │   ├── __init__.py
│   │   ├── sqlite/
│   │   │   ├── connection.py
│   │   │   ├── schema.py
│   │   │   ├── repositories.py
│   │   │   └── migrations.py
│   │   └── bm25/
│   │       ├── index_store.py
│   │       └── tokenizer.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── runner.py
│   │   ├── dataset_loader.py
│   │   ├── metrics.py
│   │   ├── ragas_evaluator.py
│   │   └── report_writer.py
│   ├── observability/
│   │   ├── __init__.py
│   │   ├── logger.py
│   │   ├── metrics.py
│   │   └── trace_service.py
│   ├── mcp_server/
│   │   ├── __init__.py
│   │   ├── server.py
│   │   ├── protocol_handler.py
│   │   ├── tool_registry.py
│   │   └── tools/
│   │       ├── __init__.py
│   │       ├── query_knowledge_hub.py
│   │       ├── list_collections.py
│   │       └── get_document_summary.py
│   └── dashboard/
│       ├── __init__.py
│       ├── app.py
│       ├── data_access.py
│       ├── components.py
│       └── pages/
│           ├── overview.py
│           ├── data_browser.py
│           ├── ingestion_admin.py
│           ├── trace_viewer.py
│           └── evaluation_panel.py
└── tests/
    ├── unit/
    │   ├── test_config.py
    │   ├── test_factories.py
    │   ├── test_hashing.py
    │   ├── test_fusion.py
    │   ├── test_chunking_service.py
    │   ├── test_rerank_pipeline.py
    │   ├── test_protocol_handler.py
    │   └── test_trace_context.py
    ├── integration/
    │   ├── test_ingestion_pipeline.py
    │   ├── test_query_pipeline.py
    │   ├── test_mcp_server.py
    │   ├── test_dashboard_data_access.py
    │   └── test_evaluation_runner.py
    ├── e2e/
    │   ├── test_ingest_and_query.py
    │   ├── test_incremental_ingestion.py
    │   └── test_mcp_stdio_runtime.py
    └── fixtures/
        ├── sample_pdfs/
        ├── sample_markdown/
        ├── sample_images/
        └── eval_datasets/
```

### 5.3 模块职责说明表

| 模块 | 主要职责 | 关键对象 |
|---|---|---|
| `app` | 配置加载、依赖装配、全局异常 | `AppConfig`, `Container` |
| `core.interfaces` | 抽象边界定义 | `BaseLLM`, `BaseVectorStore` |
| `core.factories` | provider 实例化 | `LLMFactory`, `VectorStoreFactory` |
| `core.models` | 域模型 | `Chunk`, `RetrievalResult`, `TraceEvent` |
| `providers` | 第三方能力适配 | `OpenAILLM`, `ChromaStore` |
| `ingestion` | 文档摄取编排 | `IngestionPipeline` |
| `retrieval` | 问答检索编排 | `QueryPipeline` |
| `storage.sqlite` | 元数据持久化 | repositories, schema |
| `storage.bm25` | 稀疏索引存储与查询 | `BM25IndexStore` |
| `evaluation` | RAG 评估执行与报表 | `EvaluationRunner` |
| `observability` | 日志、指标、trace | `TraceService`, `JsonlLogger` |
| `mcp_server` | MCP 服务与 tools | `MCPServer`, tool handlers |
| `dashboard` | 本地可视化面板 | Streamlit pages |
| `scripts` | CLI 入口 | ingest/query/evaluate/server |

### 5.4 数据流说明

#### 5.4.1 Ingestion Flow

```text
PDF
 -> MarkItDown 转 Markdown
 -> Markdown 解析
 -> 图片提取
 -> Vision LLM 图片描述
 -> 智能分块
 -> LLM 重写/元数据注入
 -> 生成 chunk 与 metadata
 -> 稠密 Embedding
 -> Chroma Upsert
 -> BM25 索引更新
 -> SQLite 文档与任务状态落库
 -> JSONL Trace 写入
```

#### 5.4.2 Query Flow

```text
User Query
 -> Query 预处理
 -> Query 改写
 -> BM25 检索
 -> Dense 检索
 -> RRF 融合
 -> 粗排截断
 -> Cross-Encoder / LLM Rerank
 -> Context 组装
 -> LLM 生成答案
 -> Citation 生成
 -> MCP Tool 返回结果
 -> JSONL Trace 写入
```

### 5.5 配置驱动设计示例

```yaml
app:
  name: rag-server
  env: local
  log_level: INFO

storage:
  sqlite_path: data/sqlite/rag.db
  chroma_path: data/chroma
  bm25_index_path: data/cache/bm25

models:
  llm:
    provider: openai
    model: gpt-4.1-mini
    api_key_env: OPENAI_API_KEY
  vision_llm:
    provider: openai
    model: gpt-4.1-mini
    api_key_env: OPENAI_API_KEY
  embedding:
    provider: openai
    model: text-embedding-3-large
    api_key_env: OPENAI_API_KEY
  reranker:
    type: cross_encoder
    model: BAAI/bge-reranker-base

rag:
  default_collection: knowledge_hub
  citation_enabled: true
  no_answer_policy: grounded_only

ingestion:
  converter:
    type: markitdown
  splitter:
    provider: langchain_recursive
    chunk_size: 900
    chunk_overlap: 150
    separators: ["\n## ", "\n### ", "\n\n", "\n", "。", " ", ""]
  enhancement:
    rewrite_for_retrieval: true
    inject_metadata_text: true
    image_captioning: true
  incremental:
    enabled: true
    strategy: sha256_skip

retrieval:
  sparse:
    provider: bm25
    top_k: 30
  dense:
    top_k: 30
  fusion:
    type: rrf
    rrf_k: 60
  rerank:
    enabled: true
    candidate_top_n: 20
    final_top_n: 8
  answer:
    max_context_chunks: 8

observability:
  trace_enabled: true
  trace_dir: logs/traces
  app_log_dir: logs/app
  jsonl_enabled: true

evaluation:
  default_dataset_dir: data/eval
  evaluators:
    - type: ragas
    - type: custom
      metrics: [hit_rate, mrr]

mcp:
  transport: stdio
  server_name: rag-mcp-server
  tools:
    - query_knowledge_hub
    - list_collections
    - get_document_summary

dashboard:
  enabled: true
  title: RAG Local Dashboard
```

## 6. 项目排期

### 6.1 阶段原则

- 总体按 A 到 I 九个阶段推进。
- 每个阶段设计为约 1 小时一个可验收增量。
- 每个阶段结束都必须满足“代码可运行 + 测试可验证 + 文档已同步”。

### 6.2 分阶段任务清单

#### 阶段 A：项目骨架与配置系统

目的：建立可扩展项目骨架、配置模型与依赖装配基础。

修改文件列表：

- `pyproject.toml`
- `settings.yaml`
- `src/app/config.py`
- `src/app/settings_models.py`
- `src/app/container.py`
- `tests/unit/test_config.py`

实现的类/函数：

- `load_settings()`
- `AppSettings`
- `build_container()`

验收标准：

- 能从 `settings.yaml` 成功加载配置
- 支持环境变量覆盖 API Key
- 容器可以根据配置创建基础组件实例

测试方法：

- `pytest tests/unit/test_config.py`

#### 阶段 B：核心抽象与工厂体系

目的：固定所有可插拔组件边界，避免后续 provider 实现耦合。

修改文件列表：

- `src/core/interfaces/*.py`
- `src/core/factories/*.py`
- `tests/unit/test_factories.py`

实现的类/函数：

- `BaseLLM`
- `BaseEmbedding`
- `BaseReranker`
- `BaseVectorStore`
- `LLMFactory.create()`
- `EmbeddingFactory.create()`

验收标准：

- 所有核心组件都有统一抽象
- 工厂可按 provider/type 正确实例化
- 未知 provider 会抛出明确异常

测试方法：

- `pytest tests/unit/test_factories.py`

#### 阶段 C：存储层与可观测基础

目的：建立 SQLite、Chroma、JSONL Trace 三类基础设施。

修改文件列表：

- `src/storage/sqlite/*`
- `src/providers/vector_store/chroma_store.py`
- `src/core/trace/*`
- `src/observability/logger.py`
- `tests/unit/test_trace_context.py`

实现的类/函数：

- `SQLiteConnectionManager`
- `DocumentRepository`
- `TraceContext`
- `TraceCollector`
- `JsonlLogger`
- `ChromaStore`

验收标准：

- SQLite 初始化成功
- Chroma collection 可创建与查询
- Trace 事件可写入 JSONL

测试方法：

- `pytest tests/unit/test_trace_context.py`
- `pytest tests/integration/test_dashboard_data_access.py`

#### 阶段 D：Ingestion Pipeline 最小闭环

目的：实现 PDF 到向量入库的最小可运行链路。

修改文件列表：

- `src/providers/converter/markitdown_converter.py`
- `src/ingestion/pipeline.py`
- `src/ingestion/markdown_parser.py`
- `src/ingestion/services/fingerprint_service.py`
- `src/ingestion/services/chunking_service.py`
- `src/ingestion/services/upsert_service.py`
- `scripts/ingest.py`
- `tests/integration/test_ingestion_pipeline.py`

实现的类/函数：

- `MarkItDownConverter.convert()`
- `FingerprintService.compute_sha256()`
- `ChunkingService.split_document()`
- `IngestionPipeline.run()`

验收标准：

- 可以 ingest 一个 PDF 并完成 chunk 入库
- 已处理文档再次 ingest 时可被 SHA256 跳过
- trace 中可看到完整 ingestion stages

测试方法：

- `pytest tests/integration/test_ingestion_pipeline.py`
- `pytest tests/e2e/test_incremental_ingestion.py`

#### 阶段 E：检索、融合与重排

目的：实现 Query Pipeline 主干能力。

修改文件列表：

- `src/retrieval/bm25_retriever.py`
- `src/retrieval/dense_retriever.py`
- `src/retrieval/fusion.py`
- `src/retrieval/rerank_pipeline.py`
- `src/retrieval/context_builder.py`
- `src/retrieval/pipeline.py`
- `tests/unit/test_fusion.py`
- `tests/unit/test_rerank_pipeline.py`
- `tests/integration/test_query_pipeline.py`

实现的类/函数：

- `BM25Retriever.retrieve()`
- `DenseRetriever.retrieve()`
- `reciprocal_rank_fusion()`
- `RerankPipeline.run()`
- `QueryPipeline.run()`

验收标准：

- 稀疏与稠密检索均可独立运行
- RRF 融合结果顺序符合预期
- rerank 可切换 `cross_encoder` 或 `llm_rerank`

测试方法：

- `pytest tests/unit/test_fusion.py`
- `pytest tests/unit/test_rerank_pipeline.py`
- `pytest tests/integration/test_query_pipeline.py`

#### 阶段 F：回答生成与多模态增强

目的：补齐最终回答生成、引用构建、图片描述写回链路。

修改文件列表：

- `src/providers/image_caption/vision_llm_captioner.py`
- `src/ingestion/image_extractor.py`
- `src/ingestion/chunk_enricher.py`
- `src/retrieval/answer_generator.py`
- `src/core/models/chunk.py`
- `tests/integration/test_ingestion_pipeline.py`
- `tests/e2e/test_ingest_and_query.py`

实现的类/函数：

- `VisionLLMCaptioner.describe()`
- `ChunkEnricher.enrich()`
- `AnswerGenerator.generate()`
- `CitationGenerator.build()`

验收标准：

- 图片描述会被注入 chunk 文本
- 回答结果包含引用来源
- 无图片或图片失败不会阻塞整体流程

测试方法：

- `pytest tests/integration/test_ingestion_pipeline.py`
- `pytest tests/e2e/test_ingest_and_query.py`

#### 阶段 G：MCP Server 集成

目的：通过 STDIO 暴露可调用工具。

修改文件列表：

- `src/mcp_server/server.py`
- `src/mcp_server/protocol_handler.py`
- `src/mcp_server/tool_registry.py`
- `src/mcp_server/tools/query_knowledge_hub.py`
- `src/mcp_server/tools/list_collections.py`
- `src/mcp_server/tools/get_document_summary.py`
- `scripts/run_mcp_server.py`
- `tests/unit/test_protocol_handler.py`
- `tests/integration/test_mcp_server.py`
- `tests/e2e/test_mcp_stdio_runtime.py`

实现的类/函数：

- `create_mcp_server()`
- `register_tools()`
- `handle_query_knowledge_hub()`
- `handle_list_collections()`
- `handle_get_document_summary()`

验收标准：

- MCP Server 可通过 STDIO 正常启动
- 三个首批 tools 可成功返回结构化结果
- tool 层不直接耦合 provider 细节

测试方法：

- `pytest tests/unit/test_protocol_handler.py`
- `pytest tests/integration/test_mcp_server.py`
- `pytest tests/e2e/test_mcp_stdio_runtime.py`

#### 阶段 H：评估体系

目的：建立 RAG 质量评估与结果存档能力。

修改文件列表：

- `src/evaluation/runner.py`
- `src/evaluation/dataset_loader.py`
- `src/evaluation/metrics.py`
- `src/evaluation/ragas_evaluator.py`
- `src/evaluation/report_writer.py`
- `scripts/evaluate.py`
- `tests/integration/test_evaluation_runner.py`

实现的类/函数：

- `EvaluationRunner.run()`
- `compute_hit_rate()`
- `compute_mrr()`
- `RagasEvaluator.evaluate()`

验收标准：

- 支持自定义指标跑通
- 支持接入 Ragas
- 结果可写入 SQLite 和文件报告

测试方法：

- `pytest tests/integration/test_evaluation_runner.py`

#### 阶段 I：Dashboard 与交付收口

目的：完成本地 Dashboard、文档与交付质量收口。

修改文件列表：

- `src/dashboard/app.py`
- `src/dashboard/data_access.py`
- `src/dashboard/components.py`
- `src/dashboard/pages/*.py`
- `README.md`
- `DEV_SPEC.md`
- `tests/integration/test_dashboard_data_access.py`

实现的类/函数：

- `load_overview_metrics()`
- `load_trace_timeline()`
- `load_collection_summary()`
- `render_evaluation_panel()`

验收标准：

- Dashboard 五个页面可正常显示
- 数据来自本地 SQLite / JSONL / Chroma 派生统计
- README 能指导本地运行 ingest、query、MCP、dashboard、evaluate

测试方法：

- `pytest tests/integration/test_dashboard_data_access.py`

### 6.3 进度跟踪表

| 阶段 | 目标 | 预计时长 | 状态 | 完成定义 |
|---|---|---:|---|---|
| A | 项目骨架与配置系统 | 1h | TODO | 配置可加载、容器可初始化 |
| B | 核心抽象与工厂体系 | 1h | TODO | 接口稳定、工厂可实例化 |
| C | 存储层与可观测基础 | 1h | TODO | SQLite/Chroma/Trace 可运行 |
| D | Ingestion Pipeline 最小闭环 | 1h | TODO | PDF 可入库且支持增量跳过 |
| E | 检索、融合与重排 | 1h | TODO | Query Pipeline 可返回候选结果 |
| F | 回答生成与多模态增强 | 1h | TODO | 回答带引用，图片描述入块 |
| G | MCP Server 集成 | 1h | TODO | STDIO tools 可调用 |
| H | 评估体系 | 1h | TODO | 指标可运行并落库 |
| I | Dashboard 与交付收口 | 1h | TODO | 面板可视化与文档齐备 |

## 7. 开发规范补充约束

### 7.1 编码规范

- Python 版本建议为 3.11+
- 统一使用类型标注
- 对外接口必须写 docstring
- 禁止在业务层直接拼接 provider SDK 调用，必须通过适配层封装
- 单文件职责清晰，避免出现超大“万能模块”

### 7.2 异常处理规范

- 项目内定义统一异常基类
- provider 调用异常需转换为领域异常
- 可恢复错误必须记录 trace warning
- 不可恢复错误必须记录 trace error 并携带上下文

### 7.3 日志规范

- 运行日志与 trace 分离
- 默认不打印大段原始文档内容
- 敏感信息必须脱敏，尤其是 API Key 和完整用户隐私数据

### 7.4 文档规范

- 每个核心模块应有简短模块说明
- 新增 provider 时必须同步更新：
  - `settings.yaml`
  - 工厂注册
  - 测试样例
  - README 或开发文档

### 7.5 完成标准

满足以下条件方可视为“项目第一版可交付”：

- 可 ingest PDF 并完成增量处理
- 可执行混合检索 + RRF + 两段式重排
- 可通过 MCP STDIO 暴露核心 tools
- 可在本地 Dashboard 查看系统状态与 trace
- 可执行至少一套自定义评估指标
- 单元、集成、E2E 测试均具备基础覆盖
