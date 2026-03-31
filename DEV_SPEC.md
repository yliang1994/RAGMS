# DEV_SPEC

Version: `1.0.0`

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
  - [6.2 总体进度表](#62-总体进度表)
  - [6.3 分阶段任务清单与进度跟踪](#63-分阶段任务清单与进度跟踪)
  - [6.4 执行建议](#64-执行建议)
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
- 基于该哈希检索 `ingestion_history` 表，默认采用统一的 SQLite 元数据库，可插拔存储路径为 `data/metadata/ragms.db`。
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

- 入口为单独 server 进程，例如 `PYTHONPATH=src python -m ragms.mcp_server.server`
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
- `embedding_factory.py`：根据配置返回 Dense Embedding 实现，保证上层调用入口一致。

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
- 向量编码策略：定义 `BaseEmbedding` 抽象接口，支持不同 Dense Embedding 模型的可插拔替换。项目当前采用 Hybrid Retrieval 策略，其中 Dense 路径通过 `BaseEmbedding` 生成向量，Sparse 路径通过 `BM25Indexer`/`BM25Tokenizer` 构建倒排索引。两条路径在编排层统一汇合，但不共享同一个 provider 抽象。
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
	  traces_file: ./logs/traces.jsonl
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
- 与业务解耦：追踪逻辑不能污染核心 Pipeline 代码，应通过统一 `TraceManager`、装饰器、中间件或事件钩子插入，避免业务代码里散落日志拼接逻辑。
- 结构化日志 + 本地 Dashboard：底层以 JSON Lines 记录结构化 Trace，前端以本地 Dashboard 做可视化展示，不依赖外部托管平台。
- 自动根据可插拔组件渲染：由于系统中的 Loader、Splitter、Embedding、Reranker、VectorStore、Evaluator 都是可插拔的，Dashboard 不应写死页面字段，而应根据 Trace 中记录的组件名称、阶段类型与指标动态渲染。

该设计确保系统不仅能“知道失败了”，还能回答以下问题：

- 失败发生在哪个阶段、哪个 provider、哪次配置版本。
- 同一查询在不同检索策略下耗时和效果有何差异。
- 某次文档摄取为何被跳过、重建或部分失败。

#### 3.4.2 追踪数据结构

系统定义三类 Trace 记录，分别覆盖查询、摄取与评估三条主链路：

- `QueryTrace`：记录一次查询从 Query Rewrite、Hybrid Retrieval、Rerank、Prompt Assemble 到 Answer Generation 的完整过程。
- `IngestionTrace`：记录一次摄取从文件发现、Loader、Splitter、Transform、Embedding、Upsert 到状态提交的完整过程。
- `EvaluationTrace`：记录一次评估从数据集加载、样本装配、Evaluator 执行、指标聚合到报告落盘的完整过程。

三类 Trace 都必须共享统一的顶层结构，以便 Dashboard 统一读取和过滤。Trace 的持久化粒度为“单次请求一条完整记录”，阶段详情以内嵌 `stages` 列表方式保存，而不是将每个阶段拆成独立日志行。

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

**C. Evaluation Trace（评估追踪）**

每次评估运行生成唯一的 `trace_id`，并与最终的 `run_id` 关联，记录从数据集加载到评估报告产出的全过程：

**基础信息**：
- `trace_id`：评估请求唯一标识
- `trace_type`：`"evaluation"`
- `run_id`：评估运行标识
- `start_time` / `end_time` / `duration_ms`：评估开始时间、结束时间与总耗时
- `collection`：被评估的知识库集合
- `dataset_version`：评估数据集版本
- `backends`：启用的 evaluator 列表
- `status`：本次评估状态

**各阶段详情 (Stages)**：

| 阶段 | 记录内容 |
|-----|---------|
| **Dataset Load** | 数据集名称、版本、样本数、过滤标签、耗时 |
| **Sample Build** | query / ground_truth / citations / trace 关联装配情况、耗时 |
| **Evaluator Execute** | 启用的 evaluator、成功/失败后端、降级信息、耗时 |
| **Metrics Aggregate** | retrieval / answer 指标摘要、baseline 差异、quality gate 判定、耗时 |
| **Report Persist** | 报告路径、SQLite 写入状态、`run_id`、耗时 |

**汇总指标**：
- `metrics_summary`：本次运行的关键指标摘要
- `quality_gate_status`：质量门槛通过/失败状态
- `baseline_delta`：相对 baseline 的变化摘要
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
JSON Lines Trace Log
logs/traces.jsonl
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
- Trace 默认统一写入 `logs/traces.jsonl`；如后续需要按体量归档，可在不改变读取接口的前提下追加滚动策略。
- 应用通用日志单独写入 `logs/app/` 目录。
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
- `trace_schema.py`：`QueryTrace`、`IngestionTrace`、`StageTrace` 等数据结构定义。
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
          ├─→ logs/traces.jsonl
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
- `系统总览` 与 `评估面板` 对 Query / Ingestion / Evaluation Trace 以及评估报告做聚合统计，形成趋势图、分布图和策略对比图。
- `数据浏览器` 与 `Ingestion管理` 会将 Trace 与 SQLite / Chroma 中的文档实体关联起来，实现“从文档跳到追踪”与“从追踪回到文档”的双向回溯。
- `评估面板` 会将 `EvaluationTrace` 与报告明细关联起来，实现“从评估报告回看 trace / config_snapshot / baseline 差异”的闭环。

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
  traces_file: ./logs/traces.jsonl  # Trace 日志文件路径
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

- 新增核心模块必须先补单元测试，单元测试应在秒级完成，支持开发者高频执行，立即发现引入的问题
- 跨模块编排必须有集成测试
- 面向真实使用链路必须有 E2E 测试
- 大量快速的单元测试作为基座，少量关键路径的集成测试作为保障，极少数端到端测试验证完整流程

### 4.2 分层测试策略

#### 4.2.1 单元测试

**目标**

单元测试的目标是验证单一模块、单一函数或单一类在明确输入下的行为正确性，尽早发现接口契约、边界条件和异常处理上的问题。该层测试必须聚焦“局部正确性”，而不是承担跨模块编排验证职责。

单元测试需要重点保证以下几点：

- 核心抽象接口在不同实现下都满足相同契约。
- 纯逻辑模块在正常路径、边界输入和异常路径下行为稳定。
- 配置驱动与工厂装配逻辑不会因新增 provider 或参数调整而 silently break。
- 开发者能够在本地高频执行，快速得到反馈，因此运行时间应尽量控制在秒级。

**覆盖范围**

| 测试对象 | 关注点 | 典型断言 |
|-----|------|------|
| 配置加载与校验 | 配置解析、默认值补齐、非法字段拦截 | 缺省配置是否生效、非法 provider 是否抛出预期异常 |
| 工厂实例化 | `Factory` 是否按配置返回正确实现 | `provider/type` 到实现类的映射是否正确，未知配置是否报错 |
| 抽象基类契约 | `BaseLLM`、`BaseEmbedding`、`BaseVectorStore` 等接口一致性 | 方法签名、返回结构、异常语义是否符合约定 |
| chunk 切分逻辑 | 切分边界、重叠策略、元数据保留 | chunk 数量、offset、标题路径、`chunk_id` 稳定性 |
| RRF 融合 | 多路召回结果融合排序是否正确 | 融合后排名、重复文档去重、不同输入顺序下结果稳定性 |
| rerank 结果排序 | 重排分数与最终排序逻辑 | Top-N 选择、分数降序、fallback 后结果结构 |
| SHA256 增量判断 | 去重与增量更新规则 | 内容不变是否跳过、内容变化是否触发重建 |
| citation 生成 | 引用格式、来源映射、位置定位 | 返回的文档 ID、chunk 引用、页码或来源路径是否正确 |
| MCP 参数校验 | tool 入参与错误处理 | 必填字段缺失、类型错误、非法枚举值是否被拦截 |
| Trace 辅助逻辑 | 摘要裁剪、时间统计、敏感字段脱敏 | 输出字段是否完整、敏感信息是否被过滤 |

**技术选型**

单元测试层统一采用以下技术方案：

- `pytest`：作为主测试框架，负责测试发现、参数化、fixture 管理与断言组织。
- `unittest.mock` / `pytest-mock`：用于替换外部依赖与副作用调用，例如 LLM API、Embedding API、文件系统访问、时间函数和数据库写入。
- `pytest-check` 或等价方案：用于支持多断言不中断执行，在同一个测试用例中尽可能收集多个失败点，提升问题定位效率。

选型约束如下：

- 单元测试默认不依赖网络，不调用真实外部模型、真实远程数据库或付费 API。
- 所有外部模型调用必须 mock，且优先 mock 在抽象边界处，而不是深入第三方 SDK 内部。
- 涉及文件、SQLite、Chroma 等本地依赖时，应优先使用临时目录、内存数据库或 fake 实现，避免污染真实开发数据。
- 测试应优先覆盖确定性逻辑，避免引入高随机性断言。

#### 4.2.2 集成测试

**目标**

集成测试的目标是验证多个模块在真实编排关系下能否正确协作，重点检查接口衔接、数据结构传递、组件装配和局部链路行为是否符合预期。该层测试不追求完整用户视角，而是验证“若干关键模块串起来之后是否还能稳定工作”。

集成测试需要重点覆盖以下风险：

- 单元测试通过，但模块之间的输入输出契约不兼容。
- 配置驱动下的工厂装配正确，但实际串联执行时上下游数据结构不匹配。
- 本地存储、日志、Trace、MCP 调用等基础设施在联动时出现状态不一致。
- 某个 provider 的 fake/mock 行为与真实编排路径偏差过大，导致上线前缺少链路级验证。

**覆盖范围**

| 测试对象 | 关注点 | 典型断言 |
|-----|------|------|
| PDF -> Markdown -> chunk -> embedding -> Chroma upsert | Ingestion 主链路中的模块衔接是否正确 | 文档是否被成功切分、向量是否生成、payload 是否完成 upsert |
| sparse + dense + RRF + rerank | 检索链路中多策略组合是否协同工作 | 稀疏/稠密召回结果是否被正确融合、rerank 后结果结构是否完整 |
| MCP tool -> query engine | MCP 接口层与内部查询引擎的集成是否正常 | 参数是否正确传递、响应结构是否符合 tool 协议、错误是否被正确包装 |
| trace 写入与 Dashboard 数据读取 | 可观测性链路是否闭环 | Trace 是否成功落盘、Dashboard 读取后字段是否完整、`trace_id` 是否可检索 |
| 工厂装配 + 配置切换 | 配置驱动的组件切换是否真正生效 | 切换不同 provider/backend 后，链路是否使用预期实现 |
| SQLite / Chroma / JSONL 协同 | 本地基础设施之间的数据一致性 | 文档状态、向量写入、Trace 记录之间是否能相互对应 |

**技术选型**

集成测试层统一采用以下技术方案：

- `pytest`：作为集成测试主框架，负责编排测试用例、fixture 生命周期和临时环境管理。
- `pytest-mock` / `unittest.mock`：用于替换外部 API 调用，但保留系统内部模块之间的真实交互。
- `tempfile` / pytest 临时目录 fixture：用于创建临时文件系统、临时 SQLite 数据库、临时 Chroma 数据目录。
- `pytest-check` 或等价方案：在单条集成链路测试中支持多断言不中断执行，尽可能一次性暴露结构、状态和输出上的多个问题。

选型约束如下：

- 集成测试可以使用 fake provider、stub provider 或 mock provider 替代真实大模型，但不应把所有内部模块都 mock 掉，否则会退化成单元测试。
- 输出校验必须覆盖结构完整性、关键字段、状态变化和副作用结果，而不仅仅是“函数成功返回”。
- 测试数据应尽量小而完整，既能覆盖关键链路，又能在 CI 中稳定运行。
- 涉及本地存储时必须使用隔离的临时环境，禁止读写开发者真实数据目录。

#### 4.2.3 E2E 测试

**目标**

E2E 测试的目标是从用户视角验证系统的端到端可用性，确认从输入样例数据到最终用户可见结果的完整链路能够在接近真实使用方式的条件下跑通。该层测试重点不是细抠单个模块细节，而是验证“系统作为一个整体是否可交付”。

E2E 测试需要重点回答以下问题：

- 用户是否能够完成一次完整的文档摄取、检索、问答和评估流程。
- 关键用户路径在默认配置下是否可运行、可观察、可回溯。
- 系统在最小真实场景中是否存在单元测试和集成测试无法暴露的编排问题。

**核心场景**：

**场景 1：数据准备（离线摄取）**
- **测试目标**：验证文档摄取流程的完整性与正确性
- **测试步骤**：
  - 准备测试文档（PDF 文件，包含文本、图片、表格等多种元素）
  - 执行离线摄取脚本，将文档导入知识库
  - 验证摄取结果：检查生成的 Chunk 数量、元数据完整性、图片描述生成
  - 验证存储状态：确认向量库和 BM25 索引正确创建
  - 验证幂等性：重复摄取同一文档，确保不产生重复数据
- **验证要点**：
  - Chunk 的切分质量（语义完整性、上下文保留）
  - 元数据字段完整性（source、page、title、tags 等）
  - 图片处理结果（Caption 生成、Base64 编码存储）
  - 向量与稀疏索引的正确性

**场景 2：召回测试**
- **测试目标**：验证检索系统的召回精度与排序质量
- **测试步骤**：
  - 基于已摄取的知识库，准备一组测试查询（包含不同难度与类型）
  - 执行混合检索（Dense + Sparse + Rerank）
  - 验证召回结果：检查 Top-K 文档是否包含预期来源
  - 对比不同检索策略的效果（纯 Dense、纯 Sparse、Hybrid）
  - 验证 Rerank 的影响：对比重排前后的结果变化
- **验证要点**：
  - Hit Rate@K：Top-K 结果命中率是否达标
  - 排序质量：正确答案是否排在前列（MRR、NDCG）
  - 边界情况处理：空查询、无结果查询、超长查询
  - 多模态召回：包含图片的文档是否能通过文本查询召回

**场景 3：MCP Client 功能测试**
- **测试目标**：验证 MCP Server 与 Client（如 GitHub Copilot）的协议兼容性与功能完整性
- **测试步骤**：
  - 启动 MCP Server（Stdio Transport 模式）
  - 模拟 MCP Client 发送各类 JSON-RPC 请求
  - 测试工具调用：`query_knowledge_hub`、`list_collections` 等
  - 验证返回格式：符合 MCP 协议规范（content 数组、structuredContent）
  - 测试引用透明性：返回结果包含完整的 Citation 信息
  - 测试多模态返回：包含图片的响应正确编码为 Base64
- **验证要点**：
  - 协议合规性：JSON-RPC 2.0 格式、错误码映射
  - 工具注册：`tools/list` 返回所有可用工具及其 Schema
  - 响应格式：TextContent 与 ImageContent 的正确组合
  - 错误处理：无效参数、超时、服务不可用等异常场景
  - 性能指标：单次请求的端到端延迟（含检索、重排、格式化）

**测试工具**：
- **BDD 框架**：`behave` 或 `pytest-bdd`（以 Gherkin 语法描述场景）
- **环境准备**：
  - 临时测试向量库（独立于生产数据）
  - 预置的标准测试文档集
  - 本地 MCP Server 进程（Stdio Transport）

### 4.3 RAG 质量评估

**目标**：验证已设计的评估体系（见 3.3.4 评估框架抽象）是否正确实现，并能有效评估 RAG 系统的召回与生成质量。

**测试要点**：

1. **黄金测试集准备**
   - 构建标准的"问题-答案-来源文档"测试集（JSON 格式）
   - 初期人工标注核心场景，后期持续积累坏 Case

2. **评估框架实现验证**
   - 验证 Ragas/DeepEval 等评估框架的正确集成
   - 确认评估接口能输出标准化的指标字典
   - 测试多评估器并行执行与结果汇总

3. **关键指标达标验证**
   - 检索指标：Hit Rate@K ≥ 90%、MRR ≥ 0.8、NDCG@K ≥ 0.85
   - 生成指标：Faithfulness ≥ 0.9、Answer Relevancy ≥ 0.85
   - 定期运行评估，监控指标是否回归

**说明**：本节重点是验证评估体系的工程实现，而非重新设计评估方法。

### 4.4 测试工具链与 CI/CD 集成

**本地开发工作流**：
- **快速验证**：仅运行单元测试，秒级反馈
- **完整验证**：单元测试 + 集成测试，生成覆盖率报告
- **质量评估**：定期执行 RAG 质量测试，监控指标变化

**CI/CD Pipeline 设计**（可选）：
> **说明**：本地项目不强制要求 CI/CD，但配置自动化测试流程有助于代码质量保障与持续集成实践。

- **单元测试阶段**：每次提交自动触发，验证基础功能，生成覆盖率报告
- **集成测试阶段**：单元测试通过后执行，验证模块协作
- **质量评估阶段**：PR 触发，运行完整的 RAG 质量测试，发布评估报告

**测试覆盖率目标**：
- **单元测试**：核心逻辑覆盖率 ≥ 80%
- **集成测试**：关键路径覆盖率 100%（如 Ingestion、Hybrid Search）
- **E2E 测试**：核心用户场景覆盖率 100%（至少 3 个关键流程）

## 5. 系统架构与模块设计

### 5.1 整体架构图

```text
+----------------------------------------------------------------------------------------------------------------------+
| Clients 外部调用层                                                                                                    |
|----------------------------------------------------------------------------------------------------------------------|
| MCP Client / Claude Desktop / Copilot / Local CLI / Streamlit Dashboard                                              |
+-------------------------------------------------------------+--------------------------------------------------------+
                                                              |
                                                      STDIO / Tool Call / Local Service
                                                              |
+----------------------------------------------------------------------------------------------------------------------+
| MCP Server 层                                                                                                         |
|----------------------------------------------------------------------------------------------------------------------|
| server.py | protocol_handler.py | tool_registry.py | tools/query.py | tools/ingest.py | tools/collections.py | tools/documents.py | tools/traces.py | tools/evaluation.py |
+-------------------------------------------------------------+--------------------------------------------------------+
                                                              |
                                  +---------------------------+----------------------------+
                                  |                                                        |
                                  v                                                        v
+----------------------------------------------------------------------------------------------------------------------+
| Core 层 (核心业务逻辑)                                                                                                |
|----------------------------------------------------------------------------------------------------------------------|
| Query Flow                                                                                                            |
| - Query Processor -> Hybrid Search -> Reranker(Optional) -> Response Builder                                          |
| - 输出 MCP Response(TextContent + ImageContent)                                                                        |
|                                                                                                                       |
| Ingestion Pipeline (离线数据摄取)                                                                                      |
| - File Integrity(SHA256) -> Loader(MarkItDown) -> Splitter(Recursive) -> Transform(Enrichment)                      |
| - Dense Embedding + BM25 Indexing -> Upsert(Storage)                                                                  |
|                                                                                                                       |
| Management Flow                                                                                                       |
| - DataService -> DocumentAdminService -> TraceService -> ReportService(Read Only)                                     |
|                                                                                                                       |
| Trace Collector                                                                                                       |
| - trace context -> stage recorder -> metrics aggregator -> error capture -> trace flush                              |
+-------------------------------------------------------------+--------------------------------------------------------+
                                                              |
                                  +---------------------------+----------------------------+
                                  |                                                        |
                                  v                                                        v
+----------------------------------------------------------------------------------------------------------------------+
| Libs 层 (可插拔抽象层)                                                                                                |
|----------------------------------------------------------------------------------------------------------------------|
| Abstractions: BaseLoader / BaseSplitter / BaseTransform / BaseLLM / BaseVisionLLM / BaseEmbedding / BaseReranker   |
|               / BaseVectorStore / BaseEvaluator                                                                       |
| Factories:    loader_factory.py / splitter_factory.py / llm_factory.py / vision_llm_factory.py / embedding_factory.py |
|               / reranker_factory.py / vector_store_factory.py / evaluator_factory.py                                  |
| Providers:    MarkItDown / RecursiveCharacterTextSplitter / OpenAI / Qwen / DeepSeek / GPT-4o Vision / Qwen-VL    |
|               / Chroma / Ragas / DeepEval                                                                            |
+-------------------------------------------------------------+--------------------------------------------------------+
                                                              |
                                                              v
+----------------------------------------------------------------------------------------------------------------------+
| Storage 层 (存储层)                                                                                                   |
|----------------------------------------------------------------------------------------------------------------------|
| Chroma Vector DB + Payload Store                                                                                      |
| Dense Retrieval Index | Sparse / BM25 Index | SQLite Metadata DB | Image Storage | logs/traces.jsonl | app logs     |
+----------------------------------------------------------------------------------------------------------------------+

+----------------------------------------------------------------------------------------------------------------------+
| Observability 层 (可观测性)                                                                                           |
|----------------------------------------------------------------------------------------------------------------------|
| TraceService | JSONL Logger | Metrics Aggregation | Streamlit Dashboard                                               |
| Pages: 系统总览 / 数据浏览 / Ingestion 管理 / Ingestion 追踪 / Query 追踪 / 评估面板                                  |
+----------------------------------------------------------------------------------------------------------------------+
```

### 5.2 完整目录结构树

```text
RAGMS/                                                         # 项目根目录
├── DEV_SPEC.md                                                # 开发规范与设计说明文档
├── README.md                                                  # 项目介绍与使用说明
├── pyproject.toml                                             # Python 项目依赖与构建配置
├── pytest.ini                                                 # pytest 测试框架配置
├── settings.yaml                                              # 主配置文件，驱动组件切换
├── .env.example                                               # 环境变量示例模板
├── .gitignore                                                 # Git 忽略规则
├── scripts/                                                   # 命令行脚本与本地启动入口
│   ├── run_mcp_server.py                                      # 启动 MCP Server 的脚本
│   ├── run_dashboard.py                                       # 启动本地 Dashboard 的脚本
│   ├── ingest_documents.py                                    # 批量执行文档摄取的脚本
│   ├── query_cli.py                                           # 本地命令行查询入口
│   ├── run_evaluation.py                                      # 执行评估任务的脚本
│   └── run_acceptance.py                                      # 执行一键全链路验收与交付前检查的脚本
├── data/                                                      # 本地数据目录
│   ├── raw/                                                   # 原始输入数据
│   │   ├── documents/                                         # 待摄取原始文档
│   │   └── datasets/                                          # 原始评估数据集
│   ├── staging/                                               # 中间产物暂存区
│   │   ├── markdown/                                          # Loader 输出的 Markdown 中间结果
│   │   ├── chunks/                                            # Splitter / Transform 产出的 chunk 中间结果
│   │   └── captions/                                          # 图片描述与多模态增强中间结果
│   ├── vector_store/                                          # 向量数据库本地存储目录
│   │   └── chroma/                                            # ChromaDB 数据文件
│   ├── indexes/                                               # 辅助索引与缓存目录
│   │   ├── sparse/                                            # 稀疏检索 / BM25 索引数据
│   │   └── cache/                                             # 检索与处理缓存
│   ├── metadata/                                              # 元数据存储目录
│   │   └── ragms.db                                           # SQLite 元数据库
│   ├── images/                                                # 提取后的图片与关联资源
│   ├── evaluation/                                            # 评估相关数据目录
│   │   ├── datasets/                                          # 标准化评估数据集
│   │   ├── runs/                                              # 每轮评估运行产物
│   │   └── reports/                                           # 评估报告输出
│   └── tmp/                                                   # 临时文件目录
├── logs/                                                      # 日志输出目录
│   ├── app/                                                   # 应用通用日志
│   ├── traces.jsonl                                           # Query / Ingestion / Evaluation 统一 Trace 日志文件
│   └── dashboard/                                             # Dashboard 运行日志
├── src/                                                       # 源代码主目录
│   └── ragms/                                                 # 项目主 Python 包
│       ├── __init__.py                                        # 主包初始化文件
│       ├── runtime/                                           # 运行时装配层
│       │   ├── __init__.py                                    # runtime 子包初始化文件
│       │   ├── config.py                                      # 配置读取与合并逻辑
│       │   ├── settings_models.py                             # 配置模型与校验定义
│       │   ├── container.py                                   # 依赖注入与对象装配容器
│       │   └── exceptions.py                                  # 全局异常定义
│       ├── mcp_server/                                        # MCP Server 层实现
│       │   ├── __init__.py                                    # MCP Server 子包初始化文件
│       │   ├── server.py                                      # MCP Server 启动与注册入口
│       │   ├── protocol_handler.py                            # MCP 协议封装与参数处理
│       │   ├── tool_registry.py                               # MCP tools 注册中心
│       │   ├── schemas.py                                     # MCP 输入输出数据结构定义
│       │   └── tools/                                         # 对外暴露的 MCP 工具实现
│       │       ├── __init__.py                                # tools 子包初始化文件
│       │       ├── query.py                                   # `query_knowledge_hub` 工具适配器
│       │       ├── ingest.py                                  # `ingest_documents` 工具适配器
│       │       ├── collections.py                             # `list_collections` 工具适配器
│       │       ├── documents.py                               # `get_document_summary` 工具适配器
│       │       ├── traces.py                                  # `get_trace_detail` 工具适配器
│       │       └── evaluation.py                              # `evaluate_collection` 工具适配器
│       ├── core/                                              # 核心业务逻辑层
│       │   ├── __init__.py                                    # core 子包初始化文件
│       │   ├── models/                                        # 核心领域模型
│       │   │   ├── document.py                                # 文档模型定义
│       │   │   ├── chunk.py                                   # chunk 模型定义
│       │   │   ├── retrieval.py                               # 检索结果模型定义
│       │   │   ├── evaluation.py                              # 评估结果模型定义
│       │   │   └── response.py                                # 响应对象模型定义
│       │   ├── query_engine/                                  # 在线查询引擎
│       │   │   ├── __init__.py                                # query_engine 子包初始化文件
│       │   │   ├── engine.py                                  # 查询主编排入口
│       │   │   ├── query_processor.py                         # Query Processor：关键词提取与过滤解析
│       │   │   ├── hybrid_search.py                           # Hybrid Search：Dense + Sparse + RRF
│       │   │   ├── retrievers/                                # 检索器集合
│       │   │   │   ├── dense_retriever.py                     # 稠密向量检索器
│       │   │   │   └── sparse_retriever.py                    # 稀疏 / BM25 检索器
│       │   │   ├── reranker.py                                # 可选精排器封装
│       │   │   ├── response_builder.py                        # Response Builder：引用、图片内容与结构化响应拼装
│       │   │   ├── citation_builder.py                        # 引用构建器
│       │   │   └── answer_generator.py                        # 回答生成器
│       │   ├── management/                                    # 管理操作服务层
│       │   │   ├── __init__.py                                # management 子包初始化文件
│       │   │   ├── data_service.py                            # Dashboard 数据浏览服务
│       │   │   ├── document_admin_service.py                  # 文档级管理服务
│       │   │   └── trace_service.py                           # Trace 查询与聚合服务
│       │   ├── evaluation/                                    # 评估编排模块
│       │   │   ├── __init__.py                                # evaluation 子包初始化文件
│       │   │   ├── runner.py                                  # 评估运行入口
│       │   │   ├── dataset_loader.py                          # 评估数据集加载器
│       │   │   ├── report_service.py                          # 评估报告生成服务
│       │   │   └── metrics/                                   # 评估指标实现
│       │   │       ├── retrieval_metrics.py                   # 检索指标实现
│       │   │       └── answer_metrics.py                      # 回答质量指标实现
│       │   └── trace_collector/                               # Trace 收集与管理模块
│       │       ├── __init__.py                                # trace_collector 子包初始化文件
│       │       ├── trace_manager.py                           # Trace 管理器
│       │       ├── trace_schema.py                            # Trace 数据结构定义
│       │       ├── stage_recorder.py                          # 阶段记录器
│       │       └── trace_utils.py                             # Trace 辅助工具
│       ├── ingestion_pipeline/                                # 离线数据摄取流水线
│       │   ├── __init__.py                                    # ingestion_pipeline 子包初始化文件
│       │   ├── pipeline.py                                    # 摄取流水线主入口，支持阶段回调
│       │   ├── callbacks.py                                   # Pipeline 回调协议与钩子实现
│       │   ├── file_integrity.py                              # File Integrity：SHA256 去重与增量判断
│       │   ├── lifecycle/                                     # 文档生命周期与状态管理
│       │   │   ├── document_registry.py                       # 文档注册表与状态映射
│       │   │   └── lifecycle_manager.py                       # 文档删除、重建、状态管理
│       │   ├── chunking/                                      # Chunking 模块：文档切分
│       │   │   └── split.py                                   # 文本切分阶段
│       │   ├── transform/                                     # Transform 模块：增强处理
│       │   │   ├── __init__.py                                # transform 子模块初始化文件
│       │   │   ├── pipeline.py                                # Transform 主编排
│       │   │   ├── smart_chunk_builder.py                     # Chunk 智能重组 / 去噪
│       │   │   ├── metadata_injection.py                      # 语义元数据注入（Title/Summary/Tags）
│       │   │   ├── image_captioning.py                        # 图片描述生成（Vision LLM）
│       │   │   └── services/
│       │   │       └── metadata_service.py                    # 元数据增强辅助服务
│       │   ├── embedding/                                     # Embedding 模块
│       │   │   ├── pipeline.py                                # Embedding 主编排
│       │   │   ├── dense_encoder.py                           # 稠密向量编码
│       │   │   ├── sparse_encoder.py                          # 稀疏向量编码（BM25）
│       │   │   └── optimization.py                            # 批处理、缓存、并发等处理优化
│       │   └── storage/                                       # Storage 模块
│       │       ├── pipeline.py                                # Storage 主编排
│       │       ├── vector_upsert.py                           # 向量库 Upsert
│       │       ├── bm25_indexing.py                           # BM25 索引构建
│       │       └── image_persistence.py                       # 图片文件落盘编排
│       ├── libs/                                              # 可插拔抽象层
│       │   ├── __init__.py                                    # libs 子包初始化文件
│       │   ├── abstractions/                                  # 抽象基类定义
│       │   │   ├── base_loader.py                             # Loader 抽象基类
│       │   │   ├── base_splitter.py                           # Splitter 抽象基类
│       │   │   ├── base_transform.py                          # Transform 抽象基类
│       │   │   ├── base_llm.py                                # LLM 抽象基类
│       │   │   ├── base_vision_llm.py                         # Vision LLM 抽象基类
│       │   │   ├── base_embedding.py                          # Embedding 抽象基类
│       │   │   ├── base_reranker.py                           # Reranker 抽象基类
│       │   │   ├── base_vector_store.py                       # VectorStore 抽象基类
│       │   │   └── base_evaluator.py                          # Evaluator 抽象基类
│       │   ├── factories/                                     # 工厂模式实现
│       │   │   ├── loader_factory.py                          # Loader 工厂
│       │   │   ├── splitter_factory.py                        # Splitter 工厂
│       │   │   ├── llm_factory.py                             # LLM 工厂
│       │   │   ├── vision_llm_factory.py                      # Vision LLM 工厂
│       │   │   ├── embedding_factory.py                       # Embedding 工厂
│       │   │   ├── reranker_factory.py                        # Reranker 工厂
│       │   │   ├── vector_store_factory.py                    # VectorStore 工厂
│       │   │   └── evaluator_factory.py                       # Evaluator 工厂
│       │   └── providers/                                     # 具体 provider 实现
│       │       ├── loaders/                                   # Loader 实现集合
│       │       │   └── markitdown_loader.py                   # MarkItDown 文档加载器
│       │       ├── splitters/                                 # Splitter 实现集合
│       │       │   └── recursive_character_splitter.py        # 递归字符切分器
│       │       ├── llm/                                       # LLM provider 实现
│       │       │   ├── openai_llm.py                          # OpenAI LLM 实现
│       │       │   ├── qwen_llm.py                            # Qwen LLM 实现
│       │       │   └── deepseek_llm.py                        # DeepSeek LLM 实现
│       │       ├── vision_llms/                               # Vision LLM provider 实现
│       │       │   ├── gpt4o_vision_llm.py                    # GPT-4o Vision 实现
│       │       │   └── qwen_vl_llm.py                         # Qwen-VL 实现
│       │       ├── embeddings/                                # Embedding provider 实现
│       │       │   ├── openai_embedding.py                    # OpenAI Embedding 实现
│       │       │   ├── bge_embedding.py                       # BGE Embedding 实现
│       │       │   └── jina_embedding.py                      # Jina Embedding 实现
│       │       ├── rerankers/                                 # Reranker 实现集合
│       │       │   ├── cross_encoder_reranker.py              # Cross-Encoder 重排器
│       │       │   └── llm_reranker.py                        # 基于 LLM 的重排器
│       │       ├── vector_stores/                             # 向量数据库实现集合
│       │       │   └── chroma_store.py                        # Chroma 向量库实现
│       │       └── evaluators/                                # 评估器实现集合
│       │           ├── ragas_evaluator.py                     # Ragas 评估器
│       │           ├── deepeval_evaluator.py                  # DeepEval 评估器
│       │           └── custom_metrics_evaluator.py            # 自定义指标评估器
│       ├── storage/                                           # 存储层实现
│       │   ├── __init__.py                                    # storage 子包初始化文件
│       │   ├── sqlite/                                        # SQLite 元数据存储
│       │   │   ├── connection.py                              # SQLite 连接管理
│       │   │   ├── schema.py                                  # SQLite 表结构定义
│       │   │   ├── repositories/                              # SQLite 仓储实现
│       │   │   │   ├── documents.py                           # 文档仓储
│       │   │   │   ├── ingestion_history.py                   # 摄取历史仓储
│       │   │   │   ├── images.py                              # 图片索引仓储
│       │   │   │   ├── processing_cache.py                    # 图片描述/增强缓存仓储
│       │   │   │   ├── evaluations.py                         # 评估结果仓储
│       │   │   │   └── traces.py                              # Trace 元数据仓储
│       │   │   └── migrations/                                # SQLite 迁移脚本目录
│       │   ├── vector_store/                                  # 向量存储适配层
│       │   │   └── chroma_client.py                           # Chroma 客户端封装
│       │   ├── indexes/                                       # 检索索引实现
│       │   │   ├── dense_index.py                             # 稠密向量索引封装
│       │   │   ├── bm25_indexer.py                            # BM25 索引器与删除逻辑
│       │   │   └── bm25_tokenizer.py                          # BM25 分词器实现
│       │   ├── images/                                        # 图片资源存储层
│       │   │   └── image_storage.py                           # 图片存储与读取封装
│       │   └── traces/                                        # Trace 落盘存储层
│       │       ├── jsonl_writer.py                            # `logs/traces.jsonl` 写入器
│       │       └── trace_repository.py                        # Trace 仓储封装
│       ├── observability/                                     # 可观测性实现层
│       │   ├── __init__.py                                    # observability 子包初始化文件
│       │   ├── logging/                                       # 日志模块
│       │   │   ├── logger.py                                  # 日志器封装
│       │   │   └── json_formatter.py                          # JSON 日志格式化器
│       │   ├── metrics/                                       # 指标采集模块
│       │   │   ├── ingestion_metrics.py                       # 摄取指标定义
│       │   │   ├── query_metrics.py                           # 查询指标定义
│       │   │   └── evaluation_metrics.py                      # 评估指标定义
│       │   └── dashboard/                                     # 本地 Dashboard 实现
│       │       ├── __init__.py                                # dashboard 子包初始化文件
│       │       ├── app.py                                     # Streamlit 应用入口
│       │       ├── components/                                # 通用 UI 组件
│       │       │   ├── tables.py                              # 表格组件
│       │       │   ├── charts.py                              # 图表组件
│       │       │   └── trace_timeline.py                      # Trace 时间线组件
│       │       └── pages/                                     # Dashboard 页面集合
│       │           ├── system_overview.py                     # 系统总览页
│       │           ├── data_browser.py                        # 数据浏览器页面
│       │           ├── ingestion_management.py                # Ingestion 管理页面
│       │           ├── ingestion_trace.py                     # Ingestion 追踪页面
│       │           ├── query_trace.py                         # Query 追踪页面
│       │           └── evaluation_panel.py                    # 评估面板页面
│       └── shared/                                            # 通用共享模块
│           ├── __init__.py                                    # shared 子包初始化文件
│           ├── enums.py                                       # 通用枚举定义
│           ├── constants.py                                   # 通用常量定义
│           └── utils/                                         # 通用工具函数
│               ├── hashing.py                                 # 哈希计算工具
│               ├── ids.py                                     # ID 生成工具
│               ├── clock.py                                   # 时间处理工具
│               ├── io.py                                      # 文件与 IO 工具
│               └── masking.py                                 # 敏感信息脱敏工具
├── tests/                                                     # 测试代码主目录
│   ├── unit/                                                  # 单元测试
│   │   ├── runtime/                                           # runtime 层单元测试
│   │   │   ├── test_config.py                                 # 配置模块测试
│   │   │   ├── test_settings_models.py                        # 配置模型校验测试
│   │   │   └── test_container.py                              # 容器装配测试
│   │   ├── libs/                                              # 抽象层与工厂层单元测试
│   │   │   ├── test_abstractions.py                           # 抽象基类契约测试
│   │   │   ├── test_loader_factory.py                         # Loader 工厂测试
│   │   │   ├── test_llm_factory.py                            # LLM 工厂测试
│   │   │   ├── test_vision_llm_factory.py                     # Vision LLM 工厂测试
│   │   │   ├── test_embedding_factory.py                      # Embedding 工厂测试
│   │   │   ├── test_vector_store_factory.py                   # VectorStore 工厂测试
│   │   │   ├── test_evaluator_factory.py                      # Evaluator 工厂测试
│   │   │   ├── test_ragas_evaluator.py                        # Ragas 评估器测试
│   │   │   ├── test_deepeval_evaluator.py                     # DeepEval 评估器测试
│   │   │   ├── test_custom_metrics_evaluator.py               # 自定义指标评估器测试
│   │   │   ├── providers/                                     # provider 层单元测试
│   │   │   │   ├── test_markitdown_loader.py                  # MarkItDown Loader 测试
│   │   │   │   ├── test_recursive_character_splitter.py       # Recursive Splitter 测试
│   │   │   │   ├── test_chroma_store.py                       # Chroma Store 测试
│   │   │   │   ├── test_openai_llm.py                         # OpenAI LLM provider 测试
│   │   │   │   ├── test_qwen_llm.py                           # Qwen LLM provider 测试
│   │   │   │   ├── test_gpt4o_vision_llm.py                   # GPT-4o Vision provider 测试
│   │   │   │   ├── test_qwen_vl_llm.py                        # Qwen-VL provider 测试
│   │   │   │   ├── test_openai_embedding.py                   # OpenAI Embedding provider 测试
│   │   │   │   ├── test_cross_encoder_reranker.py             # Cross-Encoder Reranker 测试
│   │   │   │   └── test_llm_reranker.py                       # LLM Reranker 测试
│   │   │   └── test_provider_wiring_smoke.py                  # provider 装配冒烟测试
│   │   ├── core/                                              # 核心业务层单元测试
│   │   │   ├── models/                                        # 核心领域模型单元测试
│   │   │   │   └── test_retrieval_models.py                   # 检索结果模型测试
│   │   │   ├── query_engine/                                  # 查询引擎单元测试
│   │   │   │   ├── test_query_processor.py                    # Query Processor 测试
│   │   │   │   ├── test_dense_retriever.py                    # Dense Retriever 测试
│   │   │   │   ├── test_sparse_retriever.py                   # Sparse Retriever 测试
│   │   │   │   ├── test_rrf.py                                # RRF 融合测试
│   │   │   │   ├── test_hybrid_search.py                      # Hybrid Search 编排测试
│   │   │   │   ├── test_reranker.py                           # Reranker 测试
│   │   │   │   ├── test_response_builder.py                   # Response Builder 测试
│   │   │   │   ├── test_citation_builder.py                   # Citation Builder 测试
│   │   │   │   └── test_answer_generator.py                   # Answer Generator 测试
│   │   │   ├── evaluation/                                    # 评估模块单元测试
│   │   │   │   ├── test_dataset_loader.py                     # 评估数据集加载测试
│   │   │   │   ├── test_composite_evaluator.py                # 组合评估器测试
│   │   │   │   ├── test_retrieval_metrics.py                  # 检索指标计算测试
│   │   │   │   └── test_answer_metrics.py                     # 回答指标计算测试
│   │   │   └── trace_collector/                               # Trace 模块单元测试
│   │   │       ├── test_trace_manager.py                      # Trace 管理器测试
│   │   │       └── test_trace_utils.py                        # Trace 工具测试
│   │   ├── ingestion_pipeline/                                # 摄取流水线单元测试
│   │   │   ├── test_file_integrity.py                         # 文件完整性与增量判断测试
│   │   │   ├── test_pipeline_callbacks.py                     # Pipeline 回调测试
│   │   │   ├── lifecycle/                                     # 文档生命周期测试
│   │   │   │   ├── test_document_registry.py                  # 文档注册表测试
│   │   │   │   └── test_lifecycle_manager.py                  # 文档生命周期管理测试
│   │   │   ├── chunking/                                      # Chunking 模块测试
│   │   │   │   └── test_split.py                              # 文本切分测试
│   │   │   ├── transform/                                     # Transform 模块测试
│   │   │   │   ├── test_pipeline.py                           # Transform 主编排测试
│   │   │   │   ├── test_smart_chunk_builder.py                # Chunk 智能重组/去噪测试
│   │   │   │   ├── test_metadata_injection.py                 # 语义元数据注入测试
│   │   │   │   └── test_image_captioning.py                   # 图片描述生成测试
│   │   │   ├── embedding/                                     # Embedding 模块测试
│   │   │   │   ├── test_pipeline.py                           # Embedding 主编排测试
│   │   │   │   ├── test_dense_encoder.py                      # 稠密向量编码测试
│   │   │   │   ├── test_sparse_encoder.py                     # 稀疏向量编码测试
│   │   │   │   └── test_optimization.py                       # 处理优化测试
│   │   │   └── storage/                                       # Storage 模块测试
│   │   │       ├── test_pipeline.py                           # Storage 主编排测试
│   │   │       ├── test_vector_upsert.py                      # 向量库 Upsert 测试
│   │   │       ├── test_bm25_indexing.py                      # BM25 索引构建测试
│   │   │       └── test_image_persistence.py                  # 图片文件存储测试
│   │   ├── mcp_server/                                        # MCP Server 单元测试
│   │   │   ├── test_server_bootstrap.py                       # MCP Server 启动与生命周期测试
│   │   │   ├── test_tool_registry.py                          # tool 注册表测试
│   │   │   ├── test_schemas.py                                # MCP schema 与参数校验测试
│   │   │   ├── test_protocol_handler.py                       # 协议处理器测试
│   │   │   └── tools/                                         # MCP tools 单元测试
│   │   │       ├── test_query_tool.py                         # 查询 tool 测试
│   │   │       ├── test_ingest_tool.py                        # 摄取 tool 测试
│   │   │       ├── test_collections_tool.py                   # 集合浏览 tool 测试
│   │   │       ├── test_documents_tool.py                     # 文档摘要 tool 测试
│   │   │       ├── test_traces_tool.py                        # Trace 查询 tool 测试
│   │   │       └── test_evaluation_tool.py                    # 评估 tool 测试
│   │   └── observability/                                     # 可观测性模块单元测试
│   │       ├── test_json_formatter.py                         # JSON 格式化器测试
│   │       └── test_traces_reader.py                          # Trace 读取器测试
│   ├── integration/                                           # 集成测试
│   │   ├── test_bootstrap_smoke.py                            # 工程骨架冒烟测试
│   │   ├── test_factory_wiring.py                             # 工厂装配集成测试
│   │   ├── test_ingestion_metadata_bootstrap.py               # SQLite 元数据初始化集成测试
│   │   ├── test_document_registry_persistence.py              # 文档注册表持久化集成测试
│   │   ├── test_document_lifecycle_integration.py             # 文档生命周期联动集成测试
│   │   ├── test_ingestion_pipeline.py                         # 摄取流水线集成测试
│   │   ├── test_ingestion_pipeline_storage.py                 # 摄取写入存储集成测试
│   │   ├── test_query_engine.py                               # 查询引擎集成测试
│   │   ├── test_mcp_server.py                                 # MCP Server 初始化与工具发现集成测试
│   │   ├── test_mcp_server_query.py                           # MCP Query tool 集成测试
│   │   ├── test_mcp_server_ingest.py                          # MCP Ingest tool 集成测试
│   │   ├── test_mcp_server_documents.py                       # MCP Collections / Documents 集成测试
│   │   ├── test_mcp_server_trace.py                           # MCP Trace tool 集成测试
│   │   ├── test_mcp_server_evaluation.py                      # MCP Evaluation tool 集成测试
│   │   ├── test_trace_write_and_read.py                       # Trace 写入与读取集成测试
│   │   ├── test_ingestion_trace_logging.py                    # Ingestion Trace 打点集成测试
│   │   ├── test_query_trace_logging.py                        # Query Trace 打点集成测试
│   │   ├── test_evaluation_trace_logging.py                   # Evaluation Trace 打点集成测试
│   │   ├── test_dashboard_shell.py                            # Dashboard 应用壳集成测试
│   │   ├── test_dashboard_system_overview.py                  # Dashboard 系统总览页测试
│   │   ├── test_dashboard_data_access.py                      # Dashboard 数据访问集成测试
│   │   ├── test_dashboard_ingestion_management.py             # Dashboard Ingestion 管理页测试
│   │   ├── test_dashboard_ingestion_trace.py                  # Dashboard Ingestion 追踪页测试
│   │   ├── test_dashboard_query_trace.py                      # Dashboard Query 追踪页测试
│   │   ├── test_dashboard_navigation.py                       # Dashboard 页面跳转与联动测试
│   │   ├── test_dashboard_trace_compare.py                    # Dashboard Query Trace 对比测试
│   │   └── test_evaluation_runner.py                          # 评估运行器集成测试
│   ├── e2e/                                                   # 端到端测试
│   │   ├── test_mcp_stdio_flow.py                             # MCP STDIO 调用链路 E2E 测试
│   │   ├── test_mcp_tool_contracts.py                         # MCP tool 响应契约 E2E 测试
│   │   ├── test_mcp_client_simulation.py                      # MCP Client 全工具 E2E 测试
│   │   ├── test_dashboard_smoke.py                            # Dashboard 最终回归 E2E 测试
│   │   ├── test_evaluation_visible_in_dashboard.py            # 真实评估结果在 Dashboard 可见性测试
│   │   └── test_full_chain_acceptance.py                      # 一键全链路验收 E2E 测试
│   ├── fixtures/                                              # 测试样例数据
│   │   ├── documents/                                         # 文档样例
│   │   ├── markdown/                                          # Markdown 样例
│   │   ├── images/                                            # 图片样例
│   │   ├── traces/                                            # Trace 样例
│   │   └── evaluation/                                        # 评估数据样例
│   ├── fakes/                                                 # fake provider 与假实现
│   │   ├── fake_llm.py                                        # 假 LLM 实现
│   │   ├── fake_vision_llm.py                                 # 假 Vision LLM 实现
│   │   ├── fake_embedding.py                                  # 假 Embedding 实现
│   │   ├── fake_reranker.py                                   # 假 Reranker 实现
│   │   ├── fake_vector_store.py                               # 假 VectorStore 实现
│   │   └── fake_evaluator.py                                  # 假 Evaluator 实现
│   └── conftest.py                                            # pytest 全局 fixtures 与钩子
└── docs/                                                      # 补充文档目录
    ├── architecture/                                          # 架构设计与实现对齐文档
    ├── api/                                                   # API / MCP tool 与验收说明文档
    └── dashboards/                                            # Dashboard 使用与验收说明文档
```

### 5.3 模块职责说明表

#### 5.3.1 `runtime` 层

| 模块 | 职责 | 关键技术点 |
|---|---|---|
| `runtime/config.py` | 读取 `settings.yaml`、环境变量与默认配置，生成统一配置对象。 | 配置分层合并、环境变量覆盖、路径标准化。 |
| `runtime/settings_models.py` | 定义配置模型与字段校验规则，约束 provider、backend、参数结构。 | 强类型配置模型、默认值注入、非法配置快速失败。 |
| `runtime/container.py` | 根据配置装配 `MCP Server`、`Query Engine`、`Ingestion Pipeline`、`TraceManager` 等核心对象。 | 依赖注入、工厂装配、运行时对象生命周期管理。 |
| `runtime/exceptions.py` | 定义跨层共享的异常类型，统一错误语义。 | 异常分层、统一错误码/错误类别、便于日志与 Trace 收敛。 |

#### 5.3.2 `mcp_server` 层

| 模块 | 职责 | 关键技术点 |
|---|---|---|
| `mcp_server/server.py` | 初始化 MCP Server 并挂载所有工具入口。 | 基于 Python MCP SDK、STDIO Transport、本地工具暴露。 |
| `mcp_server/protocol_handler.py` | 处理工具调用的参数解析、校验、响应包装与异常转换。 | 协议解耦、统一响应结构、错误拦截。 |
| `mcp_server/tool_registry.py` | 维护工具注册表，集中定义哪些能力对外可见。 | 工具声明集中管理、便于扩展和测试。 |
| `mcp_server/schemas.py` | 定义 MCP 工具入参与出参结构。 | 请求/响应 schema、类型约束、兼容性控制。 |
| `mcp_server/tools/query.py` | 暴露 `query_knowledge_hub`，调用 `Query Engine` 并将通用结构化结果适配为 MCP 输出。 | MCP 入参解析、Query Flow 透传、TextContent/ImageContent 适配输出。 |
| `mcp_server/tools/ingest.py` | 暴露 `ingest_documents`，调用 `Ingestion Pipeline`。 | 批量文档受理、增量跳过、强制重建。 |
| `mcp_server/tools/collections.py` | 暴露 `list_collections`。 | 集合枚举、状态汇总、基础管理接口。 |
| `mcp_server/tools/documents.py` | 暴露 `get_document_summary`。 | 文档级回溯、与 SQLite 元数据联动。 |
| `mcp_server/tools/traces.py` | 暴露 `get_trace_detail`。 | 按 `trace_id` 查询、链路可视化数据输出。 |
| `mcp_server/tools/evaluation.py` | 暴露 `evaluate_collection`。 | 评估编排入口、结果结构化输出。 |

#### 5.3.3 `core` 层

| 模块 | 职责 | 关键技术点 |
|---|---|---|
| `core/models/` | 定义文档、chunk、检索结果、评估结果、响应对象等领域模型。 | 强类型领域建模、跨模块统一数据契约。 |
| `core/query_engine/engine.py` | 查询链路主编排器，串联 Query Processor、Hybrid Search、Reranker 与 Response Builder。 | 在线 Query Flow 编排、模块解耦、统一返回结构。 |
| `core/query_engine/query_processor.py` | 处理用户查询的关键词提取、同义词扩展与 metadata filter 解析。 | Query 预处理、过滤条件解析、输入归一化。 |
| `core/query_engine/hybrid_search.py` | 统一编排 Dense Retrieval、Sparse Retrieval 与 RRF Fusion。 | 双路检索并行、RRF 融合、Top-M 候选生成。 |
| `core/query_engine/retrievers/` | 实现 Dense Retrieval 与 Sparse Retrieval。 | 向量检索、BM25 检索、检索结果标准化。 |
| `core/query_engine/reranker.py` | 对候选结果执行可选重排。 | CrossEncoder / LLM / None 三种模式切换。 |
| `core/query_engine/response_builder.py` | 生成 CLI / MCP 共用的通用结构化响应，附带引用和图片内容。 | 引用拼装、图片 Base64 编码、结果结构标准化。 |
| `core/query_engine/citation_builder.py` | 从检索结果构建引用信息。 | chunk 溯源、路径/页码映射、引用规范化。 |
| `core/query_engine/answer_generator.py` | 调用 LLM 生成最终回答。 | 统一 LLM 接口、答案生成、上下文使用控制。 |
| `core/management/data_service.py` | 为 Dashboard 提供文档列表、Chunk 详情、图片预览等数据读取能力。 | Chroma 元数据查询、图片列表聚合、只读数据服务。 |
| `core/management/document_admin_service.py` | 负责文档删除、重建和管理操作。 | Chroma 删除、BM25 索引移除、图片删除、完整性记录清理。 |
| `core/management/trace_service.py` | 提供 Trace 日志读取、分类和详情查询能力。 | 读取 `logs/traces.jsonl`、按 `trace_type` 分类 `query / ingestion / evaluation`、返回链路详情。 |
| `core/evaluation/report_service.py` | 管理评估报告的读取、对比与写入能力。 | 报告落盘、只读摘要查询、供 Dashboard 评估面板复用。 |
| `core/evaluation/runner.py` | 执行检索/问答评估任务。 | 多评估后端组合、批量运行、结果汇总。 |
| `core/evaluation/dataset_loader.py` | 加载并标准化评估数据集。 | 数据集 schema 统一、离线评估输入收敛。 |
| `core/evaluation/metrics/` | 实现检索指标与回答质量指标。 | `hit_rate`、`MRR`、`context_precision`、`answer_relevancy` 等。 |
| `core/trace_collector/trace_manager.py` | 管理一次请求的 Trace 生命周期。 | `trace_id` 生成、阶段注册、异常捕获、最终落盘。 |
| `core/trace_collector/trace_schema.py` | 定义 `QueryTrace`、`IngestionTrace`、`EvaluationTrace`、`StageTrace` 等结构。 | 统一 Trace 数据契约、结构化日志模型。 |
| `core/trace_collector/stage_recorder.py` | 记录阶段耗时、输入输出摘要和指标。 | 阶段级打点、指标聚合、失败节点定位。 |
| `core/trace_collector/trace_utils.py` | 提供摘要裁剪、异常序列化、敏感字段脱敏等能力。 | Trace 安全落盘、日志可读性优化。 |

#### 5.3.4 `ingestion_pipeline` 层

| 模块 | 职责 | 关键技术点 |
|---|---|---|
| `ingestion_pipeline/pipeline.py` | 离线摄取主编排器，统一编排 Loader、Chunking、Transform、Embedding、Storage 五段流程。 | Ingestion Flow 编排、阶段调度、回调机制、错误恢复。 |
| `ingestion_pipeline/callbacks.py` | 定义 Pipeline 回调协议与默认钩子。 | before/after stage 回调、进度通知、Trace/日志联动。 |
| `ingestion_pipeline/file_integrity.py` | 负责 SHA256 指纹计算与“未变更则跳过”逻辑。 | 文件完整性校验、增量摄取、跳过策略。 |
| `ingestion_pipeline/lifecycle/document_registry.py` | 维护文档登记、状态变更与来源映射。 | 文档注册、状态流转、来源追踪。 |
| `ingestion_pipeline/lifecycle/lifecycle_manager.py` | 负责摄取侧文档删除、重建与生命周期管理。 | 生命周期编排、重建控制、与存储层级联协作。 |
| `ingestion_pipeline/chunking/split.py` | 承担文档切分与 Chunk 构建。 | Recursive Splitter 接入、`image_refs` 保留、chunk 边界稳定性。 |
| `ingestion_pipeline/transform/pipeline.py` | 编排 Transform 模块内部各增强步骤。 | 多步骤增强串联、失败隔离、阶段结果落盘。 |
| `ingestion_pipeline/transform/smart_chunk_builder.py` | 对 Chunk 做智能重组、清洗与去噪。 | 邻近块合并、噪声剔除、上下文修复。 |
| `ingestion_pipeline/transform/metadata_injection.py` | 为 Chunk 注入语义元数据。 | Title / Summary / Tags 生成、结构化 metadata 写入。 |
| `ingestion_pipeline/transform/image_captioning.py` | 为图片生成 caption 并注入到 Chunk。 | Vision LLM 调用、图文上下文拼接、caption 融合。 |
| `ingestion_pipeline/transform/services/metadata_service.py` | 为 Transform 阶段提供元数据增强辅助能力。 | 元数据归一化、规则封装、阶段间服务复用。 |
| `ingestion_pipeline/embedding/pipeline.py` | 编排 Dense / Sparse 双路径编码。 | 双路结果聚合、批次切分、异常降级。 |
| `ingestion_pipeline/embedding/dense_encoder.py` | 负责稠密向量编码。 | Dense Embedding 调用、向量归一化、批处理。 |
| `ingestion_pipeline/embedding/sparse_encoder.py` | 负责稀疏向量编码。 | BM25 token/term 生成、稀疏表示构建。 |
| `ingestion_pipeline/embedding/optimization.py` | 提供编码处理优化能力。 | 批处理、缓存、并发、重试与限流控制。 |
| `ingestion_pipeline/storage/pipeline.py` | 编排摄取结果写入存储。 | 写入顺序控制、失败回滚、幂等保证。 |
| `ingestion_pipeline/storage/vector_upsert.py` | 负责向量库 Upsert。 | Chroma Upsert、payload 组装、重复写入覆盖。 |
| `ingestion_pipeline/storage/bm25_indexing.py` | 负责 BM25 索引构建。 | 倒排索引更新、文档级重建、删除同步。 |
| `ingestion_pipeline/storage/image_persistence.py` | 负责编排图片文件落盘与索引关联。 | 调用底层图片存储适配器、与文档生命周期联动。 |

#### 5.3.5 `libs` 层

| 模块 | 职责 | 关键技术点 |
|---|---|---|
| `libs/abstractions/` | 定义 Loader、Splitter、Transform、LLM、Vision LLM、Embedding、Reranker、VectorStore、Evaluator 抽象基类。 | 为 Chunking/Transform/Embedding/Storage 模块提供统一接口、实现可插拔、上层调用解耦。 |
| `libs/factories/loader_factory.py` | 根据配置实例化 Loader。 | 配置驱动、provider 切换、默认实现选择。 |
| `libs/factories/splitter_factory.py` | 根据配置实例化 Splitter。 | 统一切分接口、供 Chunking 模块切换不同切分策略。 |
| `libs/factories/llm_factory.py` | 根据配置实例化 LLM。 | 多 provider 屏蔽、认证与请求格式差异收敛。 |
| `libs/factories/vision_llm_factory.py` | 根据配置实例化 Vision LLM。 | 图像输入封装、上下文拼接、供 Transform 模块切换 GPT-4o / Qwen-VL。 |
| `libs/factories/embedding_factory.py` | 根据配置实例化 Dense Embedding。 | Dense provider 切换、供 Embedding 模块统一调用、批量请求统一。 |
| `libs/factories/reranker_factory.py` | 根据配置实例化 Reranker。 | Cross-Encoder / LLM Rerank 插拔切换。 |
| `libs/factories/vector_store_factory.py` | 根据配置实例化向量存储后端。 | Chroma 默认实现、接口保持可替换。 |
| `libs/factories/evaluator_factory.py` | 根据配置实例化评估器。 | ragas / DeepEval / 自定义评估器统一装配。 |
| `libs/providers/loaders/markitdown_loader.py` | 默认文档加载器，实现 PDF 到 Markdown 的标准化输出。 | MarkItDown 接入、canonical Markdown 输出。 |
| `libs/providers/splitters/recursive_character_splitter.py` | 默认文本切分器实现。 | LangChain `RecursiveCharacterTextSplitter`。 |
| `libs/providers/llm/*.py` | 不同 LLM provider 的实现层。 | OpenAI / Qwen / DeepSeek 适配、统一 `chat()` 接口。 |
| `libs/providers/vision_llms/*.py` | 不同 Vision LLM provider 的实现层。 | GPT-4o Vision / Qwen-VL 适配、统一 `caption()` 接口、支持图文上下文输入。 |
| `libs/providers/embeddings/*.py` | 不同 Dense Embedding provider 的实现层。 | 统一 `embed()` 接口、维度归一化、批处理。 |
| `libs/providers/rerankers/*.py` | 不同重排器的实现层。 | 精排模型封装、结果分数统一。 |
| `libs/providers/vector_stores/chroma_store.py` | 默认向量存储实现。 | Chroma 封装、payload + vector 原子写入。 |
| `libs/providers/evaluators/*.py` | 各类评估后端实现。 | 标准化 `evaluate()` 接口、指标字典输出。 |

#### 5.3.6 `storage` 层

| 模块 | 职责 | 关键技术点 |
|---|---|---|
| `storage/sqlite/connection.py` | 管理 SQLite 连接与事务。 | 本地优先、轻量持久化、事务控制。 |
| `storage/sqlite/schema.py` | 定义文档、摄取历史、图片索引、处理缓存、评估、Trace 等表结构。 | 元数据建模、可迁移 schema、统一元数据收敛。 |
| `storage/sqlite/repositories/` | 封装文档、摄取历史、图片索引、处理缓存、评估结果、Trace 元数据的读写。 | Repository 模式、业务存储解耦。 |
| `storage/sqlite/migrations/` | 存放 SQLite 迁移脚本。 | 版本化 schema 管理。 |
| `storage/vector_store/chroma_client.py` | 封装 Chroma 客户端访问。 | 向量写入、相似度查询、collection 管理。 |
| `storage/indexes/dense_index.py` | 稠密检索索引访问层。 | Dense Vector Search 封装。 |
| `storage/indexes/bm25_indexer.py` | 稀疏索引访问层与文档移除能力。 | BM25 建索引、删索引、稀疏检索支持。 |
| `storage/indexes/bm25_tokenizer.py` | 为 BM25 检索提供分词与标准化。 | 中文/混合文本分词、稀疏索引预处理。 |
| `storage/images/image_storage.py` | 存储和读取图片资源。 | 图片落盘、路径管理、文档删除级联清理。 |
| `storage/traces/jsonl_writer.py` | 将 Trace 以 JSONL 形式写入本地文件。 | 结构化日志、统一写入 `logs/traces.jsonl`。 |
| `storage/traces/trace_repository.py` | 封装 Trace 的查询与存储接口。 | Trace 落盘抽象、供 Dashboard / tools 复用。 |

#### 5.3.7 `observability` 层

| 模块 | 职责 | 关键技术点 |
|---|---|---|
| `observability/logging/logger.py` | 提供统一日志器入口。 | Python `logging` 封装、模块化日志管理。 |
| `observability/logging/json_formatter.py` | 将日志和 Trace 格式化为 JSON。 | JSON Lines、结构化字段输出。 |
| `observability/metrics/ingestion_metrics.py` | 定义和汇总摄取链路指标。 | 文档数、chunk 数、阶段耗时统计。 |
| `observability/metrics/query_metrics.py` | 定义和汇总查询链路指标。 | 召回数量、rerank 耗时、总响应耗时。 |
| `observability/metrics/evaluation_metrics.py` | 定义和汇总评估链路指标。 | 评估结果聚合、baseline 差异、quality gate 指标。 |
| `observability/dashboard/app.py` | 本地 Streamlit Dashboard 入口。 | 本地 Web UI、离线可运行、以 `core/management` 为主要业务服务层，并通过 `core/evaluation/report_service.py` 读取只读评估报告。 |
| `observability/dashboard/components/` | 封装通用表格、图表和 Trace 时间线组件。 | 可复用 UI 组件、动态渲染。 |
| `observability/dashboard/pages/system_overview.py` | 展示系统总览与核心指标。 | 汇总视图、趋势展示。 |
| `observability/dashboard/pages/data_browser.py` | 浏览集合、文档、chunk 与元数据。 | 数据检索、回溯与过滤。 |
| `observability/dashboard/pages/ingestion_management.py` | 管理摄取任务与文档状态。 | 任务列表、跳过/失败/重建视图。 |
| `observability/dashboard/pages/ingestion_trace.py` | 展示 Ingestion Trace 详情。 | 按 `trace_id` 查看摄取阶段链路、状态流转与失败节点。 |
| `observability/dashboard/pages/query_trace.py` | 展示 Query Trace 详情。 | 查看 Query 预处理、召回、融合、重排与生成链路，并支持 trace 对比。 |
| `observability/dashboard/pages/evaluation_panel.py` | 展示预置或已落盘的评估结果与实验对比。 | 空态处理、报告摘要渲染、Evaluation Trace 关联、case-by-case 分析。 |

#### 5.3.8 `shared` 层

| 模块 | 职责 | 关键技术点 |
|---|---|---|
| `shared/enums.py` | 定义跨层共享的枚举常量。 | 状态统一、减少魔法字符串。 |
| `shared/constants.py` | 定义项目级共享常量。 | 配置键、默认值、目录约定集中管理。 |
| `shared/utils/hashing.py` | 提供哈希计算能力。 | SHA256 指纹、内容去重。 |
| `shared/utils/ids.py` | 提供稳定 ID 生成能力。 | `document_id`、`chunk_id`、`trace_id` 生成。 |
| `shared/utils/clock.py` | 提供时间处理工具。 | 时间戳生成、耗时计算、统一时间口径。 |
| `shared/utils/io.py` | 提供文件与 IO 通用封装。 | 路径管理、文件读写、目录初始化。 |
| `shared/utils/masking.py` | 提供敏感字段脱敏工具。 | 日志与 Trace 安全输出。 |

### 5.4 数据流说明

#### 5.4.1 离线数据摄取流 (Ingestion Flow)

```
原始文档 (PDF)
      │
      ▼
┌─────────────────┐     未变更则跳过
│ File Integrity  │───────────────────────────► 结束
│   (SHA256)      │
└────────┬────────┘
         │ 新文件/已变更
         ▼
┌─────────────────┐
│     Loader      │  PDF → Markdown + 图片提取 + 元数据收集
│   (MarkItDown)  │
└────────┬────────┘
         │ Document (text + metadata.images)
         ▼
┌─────────────────┐
│    Chunking     │  文档切分，保留图片引用并构建稳定 Chunk
│ (Split Module)  │
└────────┬────────┘
         │ Chunks[] (with image_refs)
         ▼
┌─────────────────┐
│   Transform     │  智能重组/去噪 + 元数据注入 + Vision Caption
│   (Enhance)     │
└────────┬────────┘
         │ Enriched Chunks[] (with captions in text)
         ▼
┌─────────────────┐
│   Embedding     │  Dense 编码 + Sparse(BM25) 编码 + 处理优化
│   (Dual Encode) │
└────────┬────────┘
         │ Vectors + Chunks + Metadata
         ▼
┌─────────────────┐
│    Storage      │  Vector Upsert + BM25 Index + 图片文件存储
│  (Persist)      │
└────────┬────────┘
         │ 持久化成功
         ▼
┌─────────────────┐
│ Lifecycle Commit│  ingestion_history + 文档状态 + Trace 提交
│   (Finalize)    │
└─────────────────┘
```

#### 5.4.2 在线查询流 (Query Flow)

```
用户查询 (via MCP Client)
      │
      ▼
┌─────────────────┐
│  MCP Server     │  JSON-RPC 解析，工具路由
│ (Stdio Transport)│
└────────┬────────┘
         │ query + params
         ▼
┌─────────────────┐
│ Query Processor │  关键词提取 + 同义词扩展 + Metadata 解析
│                 │
└────────┬────────┘
         │ processed_query + filters
         ▼
┌─────────────────────────────────────────────┐
│              Hybrid Search                  │
│  ┌─────────────┐          ┌─────────────┐   │
│  │Dense Retrieval│  并行   │Sparse Retrieval│   │
│  │ (Embedding)  │◄───────►│  (BM25)     │   │
│  └──────┬──────┘          └──────┬──────┘   │
│         │                        │          │
│         └────────┬───────────────┘          │
│                  ▼                          │
│         ┌─────────────┐                     │
│         │   Fusion    │  RRF 融合           │
│         │   (RRF)     │                     │
│         └──────┬──────┘                     │
└────────────────┼────────────────────────────┘
                 │ Top-M 候选
                 ▼
┌─────────────────┐
│    Reranker     │  CrossEncoder / LLM / None
│   (Optional)    │
└────────┬────────┘
         │ Top-K 精排结果
         ▼
┌─────────────────┐
│ Response Builder│  引用生成 + 图片 Base64 编码 + 结构化响应构建
│                 │
└────────┬────────┘
         │ MCP Response (TextContent + ImageContent)
         ▼
返回给 MCP Client (Copilot / Claude Desktop)
```

#### 5.4.3 管理操作流 (Management Flow)

```
Dashboard (Streamlit UI)
      │
      ├─── 数据浏览 ──────────────────────────────────────────┐
      │                                                       │
      │    DataService                                        │
      │    ├── ChromaStore.get_by_metadata(source=...)        │
      │    ├── ImageStorage.list_images(collection, doc_hash) │
      │    └── 返回文档列表 / Chunk 详情 / 图片预览            │
      │                                                       │
      ├─── Ingestion 管理 ────────────────────────────────────┤
      │                                                       │
      │    DocumentAdminService                               │
      │    ├── ingest_documents(path, collection,             │
      │    │                   on_progress=callback)          │
      │    ├── rebuild_document(document_id)                  │
      │    ├── delete_document(document_id, collection)       │
      │    └── 返回文档状态 / 进度 / 管理结果                  │
      │                                                       │
      ├─── Trace 查看 ────────────────────────────────────────┤
      │                                                       │
      │    TraceService                                       │
      │    ├── 读取 logs/traces.jsonl                         │
      │    ├── 按 trace_type 分类 (query / ingestion / evaluation) │
      │    ├── compare_traces(trace_ids)                      │
      │    └── 返回 Trace 列表 / 详情 / 对比结果               │
      │                                                       │
      └─── 评估结果查看 ──────────────────────────────────────┘
                                                           │
                                                           ReportService
                                                           ├── list_reports()
                                                           ├── load_report(report_id)
                                                           ├── compare_runs(run_ids)
                                                           └── 返回报告摘要 / 详情 / 对比视图
```


### 5.5 配置驱动设计示例

```yaml
app:
  name: ragms
  env: local
  log_level: INFO
  default_collection: knowledge_hub

runtime:
  settings_file: settings.yaml
  env_file: .env
  fail_fast_on_invalid_config: true

mcp:
  transport: stdio
  server_name: ragms-mcp-server
  tools:
    - query_knowledge_hub
    - ingest_documents
    - list_collections
    - get_document_summary
    - get_trace_detail
    - evaluate_collection

models:
  llm:
    provider: openai
    model: gpt-4.1-mini
    api_key_env: OPENAI_API_KEY
    temperature: 0.1
    max_tokens: 2048
  transform_llm:
    provider: openai
    model: gpt-4.1-mini
    api_key_env: OPENAI_API_KEY
  vision_llm:
    provider: openai
    model: gpt-4.1-mini
    api_key_env: OPENAI_API_KEY
  embedding:
    dense:
      provider: openai
      model: text-embedding-3-large
      api_key_env: OPENAI_API_KEY
      batch_size: 64
    sparse:
      provider: bm25
      tokenizer: default
  reranker:
    mode: cross_encoder   # cross_encoder | llm | none
    model: BAAI/bge-reranker-base
    enabled: true

ingestion:
  file_integrity:
    enabled: true
    algorithm: sha256
    skip_unchanged: true
  loader:
    provider: markitdown
    extract_images: true
    output_format: markdown
  splitter:
    provider: recursive_character
    chunk_size: 900
    chunk_overlap: 150
    separators: ["\n## ", "\n### ", "\n\n", "\n", "。", " ", ""]
  transform:
    rewrite_for_retrieval: true
    inject_metadata_text: true
    image_captioning: true
    enrich_metadata: true
  embedding:
    dual_path: true
    dense_enabled: true
    sparse_enabled: true
  upsert:
    vector_backend: chroma
    bm25_enabled: true
    image_storage_enabled: true
    idempotent: true

query:
  processor:
    keyword_extraction: true
    synonym_expansion: true
    parse_metadata_filters: true
  hybrid_search:
    dense_top_k: 30
    sparse_top_k: 30
    fusion:
      type: rrf
      rrf_k: 60
    candidate_top_m: 20
  reranker:
    enabled: true
    backend: cross_encoder
    final_top_k: 8
  response_builder:
    include_citations: true
    include_images: true
    image_encoding: base64
    mcp_content_types: [text, image]

storage:
  sqlite:
    path: data/metadata/ragms.db
  chroma:
    path: data/vector_store/chroma
    collection_prefix: ragms_
  bm25:
    index_dir: data/indexes/sparse
  images:
    dir: data/images
  traces:
    file: logs/traces.jsonl
  app_logs:
    dir: logs/app

observability:
  trace_enabled: true
  trace_service:
    classify_by: trace_type
    supported_types: [query, ingestion]
  logging:
    jsonl_enabled: true
    formatter: json
  metrics:
    ingestion_enabled: true
    query_enabled: true
    evaluation_enabled: true

evaluation:
  dataset_dir: data/evaluation/datasets
  reports_dir: data/evaluation/reports
  backends:
    - type: ragas
    - type: custom_metrics
      metrics: [hit_rate, mrr, context_precision, answer_relevancy]

dashboard:
  enabled: true
  title: RAGMS Local Dashboard
  auto_refresh: true
  pages:
    - system_overview
    - data_browser
    - ingestion_management
    - ingestion_trace
    - query_trace
    - evaluation_panel
```

配置设计说明：

- `ingestion.file_integrity` 对应 5.4.1 中的 `File Integrity (SHA256)` 节点，控制是否跳过未变更文件。
- `ingestion.loader / splitter / transform / embedding / upsert` 直接映射离线摄取流的五个核心阶段。
- `storage.sqlite.path` 是统一元数据数据库，承载 `documents`、`ingestion_history`、`images`、`processing_cache`、`evaluations` 与 `traces` 等结构化状态。
- `query.processor / hybrid_search / reranker / response_builder` 直接映射 5.4.2 中的在线查询链路。
- `storage.traces.file` 统一使用 `logs/traces.jsonl`，与 5.4.3 中 `TraceService` 的读取路径保持一致。
- Dashboard 页面以 `core/management` 作为主要业务服务层；评估面板通过 `core/evaluation/report_service.py` 读取只读报告数据，避免在 UI 层重复定义第二套服务逻辑。
- 所有 provider 与 backend 均通过配置切换，业务代码不直接写死具体实现。

## 6. 项目排期

### 6.1 阶段原则

 - 总体按 A 到 I 九个阶段推进，每个阶段都必须形成独立可验收增量。
 - 每个阶段再拆分为多个小阶段，编号形如 `A1`、`A2`、`B1`、`B2`。
 - 每个小阶段必须具备明确的目标、修改文件、实现类/函数、验收标准和测试方法。
 - 默认遵循 TDD 或近似 TDD：先定义接口与测试，再补实现，最后做文档同步。
 - 每个阶段完成时必须满足“代码可运行 + 测试可验证 + 文档已同步”。

### 6.2 总体进度表

| 阶段 | 总任务数 | 已完成 | 进度 |
|------|---------|--------|------|
| 阶段 A | 5 | 0 | 0% |
| 阶段 B | 15 | 0 | 0% |
| 阶段 C | 14 | 0 | 0% |
| 阶段 D | 8 | 0 | 0% |
| 阶段 E | 9 | 0 | 0% |
| 阶段 F | 6 | 0 | 0% |
| 阶段 G | 8 | 0 | 0% |
| 阶段 H | 9 | 0 | 0% |
| 阶段 I | 5 | 0 | 0% |
| **总计** | **79** | **0** | **0%** |

**状态说明**：`[ ]` 未开始 | `[~]` 进行中 | `[x]` 已完成

### 6.3 分阶段任务清单与进度跟踪

#### 阶段 A：实现工程骨架与测试基座

目标：建立可运行、可配置、可测试的工程骨架；后续所有模块都能以 TDD 方式落地。

| 任务编号 | 任务名称 | 状态 | 完成日期 | 备注 |
|---------|---------|------|---------|------|
| A1 | 初始化目录树与最小可运行入口 | [x] | 2026-03-31 | 工程骨架已建立，启动脚本与测试发现通过 |
| A2 | 实现配置加载与配置模型 | [x] | 2026-03-31 | 支持 YAML + .env + 环境变量覆盖，配置单测通过 |
| A3 | 实现运行时容器与依赖装配入口 | [x] | 2026-03-31 | 容器可返回基础占位实例，异常统一封装，单测通过 |
| A4 | 建立 pytest 测试基座与 fake/fixture 机制 | [x] | 2026-03-31 | Fake provider 与公共 fixture 已建立，unit 测试通过 |
| A5 | 建立本地启动脚本与最小冒烟测试 | [x] | 2026-03-31 | 四类入口脚本可调用，bootstrap 集成测试通过 |

##### A1 初始化目录树与最小可运行入口

- 目标：落地 `src/ragms`、`scripts`、`tests`、`data`、`logs` 的最小工程骨架。
- 修改文件：`pyproject.toml`、`README.md`、`src/ragms/__init__.py`、`scripts/run_mcp_server.py`、`tests/conftest.py`
- 实现类/函数：`main()`、`get_project_root()`
- 验收标准：项目可安装；脚本可被 Python 正常执行；测试框架可被发现。
- 测试方法：`pytest --collect-only`

##### A2 实现配置加载与配置模型

- 目标：让系统可以从 `settings.yaml` 与环境变量加载配置。
- 修改文件：`settings.yaml`、`src/ragms/runtime/config.py`、`src/ragms/runtime/settings_models.py`、`tests/unit/runtime/test_config.py`
- 实现类/函数：`load_settings()`、`AppSettings`
- 验收标准：配置可成功解析；环境变量可覆盖密钥与路径；非法配置会快速报错。
- 测试方法：`pytest tests/unit/runtime/test_config.py`

##### A3 实现运行时容器与依赖装配入口

- 目标：建立 `runtime/container.py`，为后续 Core/Libs/MCP 装配依赖。
- 修改文件：`src/ragms/runtime/container.py`、`src/ragms/runtime/exceptions.py`、`tests/unit/runtime/test_container.py`
- 实现类/函数：`build_container()`、`ServiceContainer`
- 验收标准：容器可根据配置返回基础组件占位实例；依赖装配失败会抛出统一异常。
- 测试方法：`pytest tests/unit/runtime/test_container.py`

##### A4 建立 pytest 测试基座与 fake/fixture 机制

- 目标：建立单元/集成/E2E 三层测试目录、公共 fixture 与 fake provider。
- 修改文件：`pytest.ini`、`tests/conftest.py`、`tests/fakes/fake_llm.py`、`tests/fakes/fake_vision_llm.py`、`tests/fakes/fake_embedding.py`、`tests/fakes/fake_vector_store.py`、`tests/fakes/fake_reranker.py`、`tests/fakes/fake_evaluator.py`
- 实现类/函数：`FakeLLM`、`FakeVisionLLM`、`FakeEmbedding`、`FakeVectorStore`、`FakeReranker`、`FakeEvaluator`
- 验收标准：测试可使用 fake provider 脱离真实外部依赖运行；后续 Reranker / Evaluator / Vision LLM 相关模块也可在不依赖真实模型的前提下执行 TDD。
- 测试方法：`pytest tests/unit -q`

##### A5 建立本地启动脚本与最小冒烟测试

- 目标：提供最小可运行的本地入口，验证工程骨架可执行，并为后续离线摄取 CLI 预留稳定入口。
- 修改文件：`scripts/query_cli.py`、`scripts/run_dashboard.py`、`scripts/run_mcp_server.py`、`scripts/ingest_documents.py`、`tests/integration/test_bootstrap_smoke.py`
- 实现类/函数：`run_cli()`、`run_dashboard()`、`run_mcp_server_main()`、`ingest_documents_main()`
- 验收标准：CLI、Dashboard、MCP Server、Ingestion CLI 启动脚本均可被调用；`test_bootstrap_smoke.py` 对四类入口均完成最小启动验证。
- 测试方法：`pytest tests/integration/test_bootstrap_smoke.py`

#### 阶段 B：实现核心抽象与工厂体系

目标：实现 `libs` 可插拔层与工厂装配体系，确保默认 provider 与运行时容器可在真实环境完成实例化，并为后续 Core / Ingestion 提供稳定依赖基础。

| 任务编号 | 任务名称 | 状态 | 完成日期 | 备注 |
|---------|---------|------|---------|------|
| B1 | 定义抽象基类集合 | [x] | 2026-03-31 | 9 个抽象接口已建立，契约测试通过 |
| B2 | 实现 Loader / Splitter / VectorStore 工厂 | [x] | 2026-03-31 | 默认工厂与占位 provider 已接通，工厂单测通过 |
| B3 | 实现 LLM / Vision LLM / Embedding / Reranker 工厂 | [x] | 2026-03-31 | 四类模型工厂已接通，支持 provider 切换与配置校验 |
| B4.1 | 落地 MarkItDown Loader provider | [x] | 2026-03-31 | 支持文件读取、统一文档对象输出与异常输入校验 |
| B4.2 | 落地 Recursive Character Splitter provider | [x] | 2026-03-31 | 支持稳定窗口切分、overlap 控制与边界元数据输出 |
| B4.3 | 落地 OpenAI LLM provider | [x] | 2026-03-31 | 支持同步生成、流式输出与配置/输入异常校验 |
| B4.4 | 落地 Qwen / DeepSeek LLM providers | [x] | 2026-03-31 | 支持同步/流式输出、网关参数透传与失败场景校验 |
| B4.5 | 落地 GPT-4o Vision provider | [ ] |  |  |
| B4.6 | 落地 Qwen-VL provider | [ ] |  |  |
| B4.7 | 落地 OpenAI Embedding provider | [ ] |  |  |
| B4.8 | 落地 Cross-Encoder Reranker provider | [ ] |  |  |
| B4.9 | 落地 LLM Reranker provider | [ ] |  |  |
| B4.10 | 落地 Chroma VectorStore provider | [ ] |  |  |
| B5 | 落地 Evaluator 抽象与工厂 | [ ] |  |  |
| B6 | 完成工厂装配集成冒烟 | [ ] |  |  |

##### B1 定义抽象基类集合

- 目标：固定所有可插拔组件的接口边界。
- 修改文件：`src/ragms/libs/abstractions/base_loader.py`、`base_splitter.py`、`base_transform.py`、`base_llm.py`、`base_vision_llm.py`、`base_embedding.py`、`base_reranker.py`、`base_vector_store.py`、`base_evaluator.py`、`tests/unit/libs/test_abstractions.py`
- 实现类/函数：`BaseLoader`、`BaseSplitter`、`BaseTransform`、`BaseLLM`、`BaseVisionLLM`、`BaseEmbedding`、`BaseReranker`、`BaseVectorStore`、`BaseEvaluator`
- 验收标准：所有抽象基类接口完整；方法签名稳定；测试可验证接口契约。
- 测试方法：`pytest tests/unit/libs/test_abstractions.py`

##### B2 实现 Loader / Splitter / VectorStore 工厂

- 目标：实现文档加载、切分、向量存储三类工厂。
- 修改文件：`src/ragms/libs/factories/loader_factory.py`、`src/ragms/libs/factories/splitter_factory.py`、`src/ragms/libs/factories/vector_store_factory.py`、`tests/unit/libs/test_loader_factory.py`、`tests/unit/libs/test_splitter_factory.py`、`tests/unit/libs/test_vector_store_factory.py`
- 实现类/函数：`LoaderFactory.create()`、`SplitterFactory.create()`、`VectorStoreFactory.create()`
- 验收标准：可根据配置正确返回默认实现；未知 provider 抛出明确异常。
- 测试方法：`pytest tests/unit/libs/test_loader_factory.py tests/unit/libs/test_splitter_factory.py tests/unit/libs/test_vector_store_factory.py`

##### B3 实现 LLM / Vision LLM / Embedding / Reranker 工厂

- 目标：实现模型侧四类工厂并支持配置切换。
- 修改文件：`src/ragms/libs/factories/llm_factory.py`、`src/ragms/libs/factories/vision_llm_factory.py`、`src/ragms/libs/factories/embedding_factory.py`、`src/ragms/libs/factories/reranker_factory.py`、`tests/unit/libs/test_llm_factory.py`、`tests/unit/libs/test_vision_llm_factory.py`、`tests/unit/libs/test_embedding_factory.py`、`tests/unit/libs/test_reranker_factory.py`
- 实现类/函数：`LLMFactory.create()`、`VisionLLMFactory.create()`、`EmbeddingFactory.create()`、`RerankerFactory.create()`
- 验收标准：模型 provider 可按配置实例化；Vision LLM 可按文档语言或部署环境切换；配置缺失与未知类型可被正确拦截。
- 测试方法：`pytest tests/unit/libs/test_llm_factory.py tests/unit/libs/test_vision_llm_factory.py tests/unit/libs/test_embedding_factory.py tests/unit/libs/test_reranker_factory.py`

##### B4.1 落地 MarkItDown Loader provider

- 目标：提供默认文档加载器实现，支持后续摄取链路读取 Markdown / Office/PDF 等基础文档输入。
- 修改文件：`src/ragms/libs/providers/loaders/markitdown_loader.py`、`tests/unit/libs/providers/test_markitdown_loader.py`
- 实现类/函数：`MarkItDownLoader.load()`
- 验收标准：可将样例文档加载为统一文档对象；异常文件类型与空输入可被正确处理。
- 测试方法：`pytest tests/unit/libs/providers/test_markitdown_loader.py`

##### B4.2 落地 Recursive Character Splitter provider

- 目标：提供默认文本切分器实现，确保加载后的长文本可按统一策略切分为 chunk。
- 修改文件：`src/ragms/libs/providers/splitters/recursive_character_splitter.py`、`tests/unit/libs/providers/test_recursive_character_splitter.py`
- 实现类/函数：`RecursiveCharacterSplitter.split()`
- 验收标准：样例文本可稳定切分；chunk 大小、重叠长度与边界行为符合配置预期。
- 测试方法：`pytest tests/unit/libs/providers/test_recursive_character_splitter.py`

##### B4.3 落地 OpenAI LLM provider

- 目标：提供 OpenAI 文本大模型实现，支撑 Query 与 Transform 阶段的统一文本生成能力。
- 修改文件：`src/ragms/libs/providers/llm/openai_llm.py`、`tests/unit/libs/providers/test_openai_llm.py`
- 实现类/函数：`OpenAILLM.generate()`、`OpenAILLM.stream()`
- 验收标准：可根据配置完成同步生成与流式输出；鉴权缺失、模型名非法与上游响应异常可被正确处理。
- 测试方法：`pytest tests/unit/libs/providers/test_openai_llm.py`

##### B4.4 落地 Qwen / DeepSeek LLM providers

- 目标：提供 Qwen 与 DeepSeek 文本大模型实现，支撑国内部署、中文优先和兼容 OpenAI 风格网关的文本生成能力。
- 修改文件：`src/ragms/libs/providers/llm/qwen_llm.py`、`src/ragms/libs/providers/llm/deepseek_llm.py`、`tests/unit/libs/providers/test_qwen_llm.py`、`tests/unit/libs/providers/test_deepseek_llm.py`
- 实现类/函数：`QwenLLM.generate()`、`QwenLLM.stream()`、`DeepSeekLLM.generate()`、`DeepSeekLLM.stream()`
- 验收标准：两类 provider 都可根据配置完成同步生成与流式输出；兼容 OpenAI 风格网关参数；请求失败与限流场景行为明确。
- 测试方法：`pytest tests/unit/libs/providers/test_qwen_llm.py tests/unit/libs/providers/test_deepseek_llm.py`

##### B4.5 落地 GPT-4o Vision provider

- 目标：提供 GPT-4o Vision 图像理解实现，支撑图片 caption 与多模态增强链路。
- 修改文件：`src/ragms/libs/providers/vision_llms/gpt4o_vision_llm.py`、`tests/unit/libs/providers/test_gpt4o_vision_llm.py`
- 实现类/函数：`GPT4oVisionLLM.caption()`、`GPT4oVisionLLM.caption_batch()`
- 验收标准：可对单张图片和批量图片生成结构化描述；支持图文上下文输入；图像编码错误与模型响应异常可被正确处理。
- 测试方法：`pytest tests/unit/libs/providers/test_gpt4o_vision_llm.py`

##### B4.6 落地 Qwen-VL provider

- 目标：提供 Qwen-VL 图像理解实现，支撑中文图表与国内部署场景下的图片 caption 能力。
- 修改文件：`src/ragms/libs/providers/vision_llms/qwen_vl_llm.py`、`tests/unit/libs/providers/test_qwen_vl_llm.py`
- 实现类/函数：`QwenVLLLM.caption()`、`QwenVLLLM.caption_batch()`
- 验收标准：可对单张图片和批量图片生成结构化描述；支持图文上下文输入；模型不可用与返回质量不足场景行为明确。
- 测试方法：`pytest tests/unit/libs/providers/test_qwen_vl_llm.py`

##### B4.7 落地 OpenAI Embedding provider

- 目标：提供 OpenAI 向量化实现，支撑摄取链路与检索链路生成统一维度的文本向量。
- 修改文件：`src/ragms/libs/providers/embeddings/openai_embedding.py`、`tests/unit/libs/providers/test_openai_embedding.py`
- 实现类/函数：`OpenAIEmbedding.embed_documents()`、`OpenAIEmbedding.embed_query()`
- 验收标准：文档与查询文本可输出稳定向量结果；批量输入、空输入与维度不一致场景处理明确。
- 测试方法：`pytest tests/unit/libs/providers/test_openai_embedding.py`

##### B4.8 落地 Cross-Encoder Reranker provider

- 目标：提供 Cross-Encoder 重排模型实现，支撑多路召回结果按相关性重新排序。
- 修改文件：`src/ragms/libs/providers/rerankers/cross_encoder_reranker.py`、`tests/unit/libs/providers/test_cross_encoder_reranker.py`
- 实现类/函数：`CrossEncoderReranker.rerank()`
- 验收标准：候选文档可按相关性得分稳定排序；空候选集、超长候选集与 score 缺失场景行为明确。
- 测试方法：`pytest tests/unit/libs/providers/test_cross_encoder_reranker.py`

##### B4.9 落地 LLM Reranker provider

- 目标：提供基于 LLM 的重排实现，支撑规则不足时的语义精排能力。
- 修改文件：`src/ragms/libs/providers/rerankers/llm_reranker.py`、`tests/unit/libs/providers/test_llm_reranker.py`
- 实现类/函数：`LLMReranker.rerank()`
- 验收标准：可对候选文档进行语义重排；排序输出稳定；上游 LLM 超时、空候选集与格式化失败场景行为明确。
- 测试方法：`pytest tests/unit/libs/providers/test_llm_reranker.py`

##### B4.10 落地 Chroma VectorStore provider

- 目标：提供 Chroma 向量存储实现，支撑后续检索链路完成向量写入、查询与删除。
- 修改文件：`src/ragms/libs/providers/vector_stores/chroma_store.py`、`tests/unit/libs/providers/test_chroma_store.py`
- 实现类/函数：`ChromaStore.add()`、`ChromaStore.query()`、`ChromaStore.delete()`
- 验收标准：可对样例向量执行写入、相似度查询与删除；空集合与不存在 ID 的行为明确。
- 测试方法：`pytest tests/unit/libs/providers/test_chroma_store.py`

##### B5 落地 Evaluator 抽象与工厂

- 目标：建立评估器抽象、默认评估器实现和统一装配入口。
- 修改文件：`src/ragms/libs/abstractions/base_evaluator.py`、`src/ragms/libs/factories/evaluator_factory.py`、`tests/unit/libs/test_evaluator_factory.py`
- 实现类/函数：`BaseEvaluator.evaluate()`、`EvaluatorFactory.create()`
- 验收标准：评估器可按配置切换；评估输出格式统一为指标字典。
- 测试方法：`pytest tests/unit/libs/test_evaluator_factory.py`

##### B6 完成工厂装配集成冒烟

- 目标：验证 `runtime/container.py` 能通过真实工厂创建阶段 B 已落地的默认组件，并完成基础依赖装配校验。
- 修改文件：`src/ragms/runtime/container.py`、`tests/integration/test_factory_wiring.py`
- 实现类/函数：`build_container()` 集成路径
- 验收标准：默认配置下可成功装配 Loader、Splitter、LLM、VisionLLM、Embedding、Reranker、VectorStore、Evaluator 等默认组件；容器能返回对应实例并对缺失配置或未知 provider 做统一异常收敛。
- 测试方法：`pytest tests/integration/test_factory_wiring.py`

#### 阶段 C：Ingestion Pipeline

目标：离线摄取链路跑通，能把样例文档写入向量库/BM25 索引并支持增量。

| 任务编号 | 任务名称 | 状态 | 完成日期 | 备注 |
|---------|---------|------|---------|------|
| C1 | 实现 Pipeline 主流程与回调协议 | [ ] |  |  |
| C2 | 实现 SQLite 元数据底座、File Integrity 与摄取历史存储 | [ ] |  |  |
| C3 | 实现文档注册表与状态流转 | [ ] |  |  |
| C4 | 实现文档生命周期管理 | [ ] |  |  |
| C5 | 实现 Chunking 切分主流程 | [ ] |  |  |
| C6 | 实现 Transform 主编排 | [ ] |  |  |
| C7 | 实现 Chunk 智能重组与去噪 | [ ] |  |  |
| C8 | 实现语义元数据注入 | [ ] |  |  |
| C9 | 实现图片描述生成 | [ ] |  |  |
| C10 | 实现 Dense Embedding 编码 | [ ] |  |  |
| C11 | 实现 Sparse(BM25) 编码与处理优化 | [ ] |  |  |
| C12 | 实现 Storage 主编排与向量库 Upsert | [ ] |  |  |
| C13 | 实现 BM25 索引构建与图片文件存储 | [ ] |  |  |
| C14 | 打通 Ingestion Pipeline 与 CLI 集成链路 | [ ] |  |  |

##### C1 实现 Pipeline 主流程与回调协议

- 目标：建立摄取主流程骨架，支持阶段前后回调、进度通知与错误收敛。
- 修改文件：`src/ragms/ingestion_pipeline/pipeline.py`、`src/ragms/ingestion_pipeline/callbacks.py`、`tests/unit/ingestion_pipeline/test_pipeline_callbacks.py`
- 实现类/函数：`IngestionPipeline.run()`、`PipelineCallback.on_stage_start()`、`PipelineCallback.on_stage_end()`
- 验收标准：主流程可串联各子模块；回调在阶段边界被触发；失败可返回统一错误上下文。
- 测试方法：`pytest tests/unit/ingestion_pipeline/test_pipeline_callbacks.py`

##### C2 实现 SQLite 元数据底座、File Integrity 与摄取历史存储

- 目标：建立摄取链路所需的 SQLite 元数据底座，并支持 SHA256 增量判断和未变更跳过。
- 修改文件：`src/ragms/storage/sqlite/connection.py`、`src/ragms/storage/sqlite/schema.py`、`src/ragms/storage/sqlite/migrations/`、`src/ragms/storage/sqlite/repositories/ingestion_history.py`、`src/ragms/ingestion_pipeline/file_integrity.py`、`tests/unit/ingestion_pipeline/test_file_integrity.py`、`tests/integration/test_ingestion_metadata_bootstrap.py`
- 实现类/函数：`create_sqlite_connection()`、`initialize_metadata_schema()`、`run_sqlite_migrations()`、`FileIntegrity.compute_sha256()`、`FileIntegrity.should_skip()`、`IngestionHistoryRepository`
- 验收标准：默认配置下可初始化 `data/metadata/ragms.db` 及其基础 schema；`ingestion_history` 可读写；相同文件内容重复摄取会被跳过；内容变化会触发重新处理。
- 测试方法：`pytest tests/unit/ingestion_pipeline/test_file_integrity.py tests/integration/test_ingestion_metadata_bootstrap.py`

##### C3 实现文档注册表与状态流转

- 目标：建立以 SQLite `documents` 元数据表为落点的摄取文档注册表，记录状态、来源、哈希、时间戳与当前处理阶段。
- 修改文件：`src/ragms/ingestion_pipeline/lifecycle/document_registry.py`、`src/ragms/storage/sqlite/repositories/documents.py`、`tests/unit/ingestion_pipeline/lifecycle/test_document_registry.py`、`tests/integration/test_document_registry_persistence.py`
- 实现类/函数：`DocumentRegistry.register()`、`DocumentRegistry.update_status()`、`DocumentsRepository`
- 验收标准：文档可完成注册、状态迁移与来源查询；注册信息会持久化到 `documents` 表；非法状态跳转会被拦截；重新启动后可恢复最近状态。
- 测试方法：`pytest tests/unit/ingestion_pipeline/lifecycle/test_document_registry.py tests/integration/test_document_registry_persistence.py`

##### C4 实现文档生命周期管理

- 目标：实现摄取侧文档重建、删除与状态恢复编排，并与 `documents` / `ingestion_history` / 图片索引等持久化状态联动。
- 修改文件：`src/ragms/ingestion_pipeline/lifecycle/lifecycle_manager.py`、`tests/unit/ingestion_pipeline/lifecycle/test_lifecycle_manager.py`、`tests/integration/test_document_lifecycle_integration.py`
- 实现类/函数：`IngestionDocumentManager.rebuild()`、`IngestionDocumentManager.delete()`
- 验收标准：文档可触发重建与删除；生命周期操作能驱动注册表、`documents` 元数据、`ingestion_history` 与存储层联动，保证删除/重建后状态一致。
- 测试方法：`pytest tests/unit/ingestion_pipeline/lifecycle/test_lifecycle_manager.py tests/integration/test_document_lifecycle_integration.py`

##### C5 实现 Chunking 切分主流程

- 目标：将文档稳定切分为带定位与图片引用的 chunk。
- 修改文件：`src/ragms/core/models/chunk.py`、`src/ragms/ingestion_pipeline/chunking/split.py`、`tests/unit/ingestion_pipeline/chunking/test_split.py`
- 实现类/函数：`ChunkingPipeline.run()`、`Chunk.build_id()`
- 验收标准：chunk 数量、offset、`image_refs`、`chunk_id` 稳定性符合预期。
- 测试方法：`pytest tests/unit/ingestion_pipeline/chunking/test_split.py`

##### C6 实现 Transform 主编排

- 目标：编排 Transform 阶段内部增强步骤，形成统一的 Smart Chunk 输出。
- 修改文件：`src/ragms/ingestion_pipeline/transform/pipeline.py`、`tests/unit/ingestion_pipeline/transform/test_pipeline.py`
- 实现类/函数：`TransformPipeline.run()`
- 验收标准：Transform 可按顺序执行重组、元数据注入、图片描述等步骤；单个步骤失败时具备可控降级行为。
- 测试方法：`pytest tests/unit/ingestion_pipeline/transform/test_pipeline.py`

##### C7 实现 Chunk 智能重组与去噪

- 目标：对切分后的 chunk 进行智能拼接、去噪与上下文修复。
- 修改文件：`src/ragms/ingestion_pipeline/transform/smart_chunk_builder.py`、`tests/unit/ingestion_pipeline/transform/test_smart_chunk_builder.py`
- 实现类/函数：`SmartChunkBuilder.rewrite()`、`SmartChunkBuilder.denoise()`
- 验收标准：噪声段落会被清理；邻近上下文可按规则合并；输出结构保持稳定。
- 测试方法：`pytest tests/unit/ingestion_pipeline/transform/test_smart_chunk_builder.py`

##### C8 实现语义元数据注入

- 目标：为 chunk 注入 Title、Summary、Tags 等语义元数据。
- 修改文件：`src/ragms/ingestion_pipeline/transform/metadata_injection.py`、`src/ragms/ingestion_pipeline/transform/services/metadata_service.py`、`tests/unit/ingestion_pipeline/transform/test_metadata_injection.py`
- 实现类/函数：`inject_semantic_metadata()`、`MetadataService.enrich()`
- 验收标准：元数据可被稳定注入到 chunk；无标题或短文本场景处理明确。
- 测试方法：`pytest tests/unit/ingestion_pipeline/transform/test_metadata_injection.py`

##### C9 实现图片描述生成

- 目标：调用 Vision LLM 为图片生成 caption 并注入到关联 chunk，同时支持基于 `processing_cache` 的幂等复用。
- 修改文件：`src/ragms/ingestion_pipeline/transform/image_captioning.py`、`src/ragms/storage/sqlite/repositories/processing_cache.py`、`tests/unit/ingestion_pipeline/transform/test_image_captioning.py`
- 实现类/函数：`inject_image_caption()`、`generate_image_caption()`、`ProcessingCacheRepository`
- 验收标准：单图、多图、无图场景均可处理；相同图片内容且 Prompt 版本未变化时可复用缓存描述；Vision LLM 不可用时具备明确降级策略。
- 测试方法：`pytest tests/unit/ingestion_pipeline/transform/test_image_captioning.py`

##### C10 实现 Dense Embedding 编码

- 目标：完成稠密向量编码与批处理编排。
- 修改文件：`src/ragms/ingestion_pipeline/embedding/dense_encoder.py`、`tests/unit/ingestion_pipeline/embedding/test_dense_encoder.py`
- 实现类/函数：`DenseEncoder.encode_documents()`、`DenseEncoder.encode_query()`
- 验收标准：文本可批量编码为稳定向量；空输入与异常响应场景可被正确处理。
- 测试方法：`pytest tests/unit/ingestion_pipeline/embedding/test_dense_encoder.py`

##### C11 实现 Sparse(BM25) 编码与处理优化

- 目标：完成 BM25 稀疏编码、批处理优化、缓存与并发控制。
- 修改文件：`src/ragms/ingestion_pipeline/embedding/sparse_encoder.py`、`src/ragms/ingestion_pipeline/embedding/optimization.py`、`tests/unit/ingestion_pipeline/embedding/test_sparse_encoder.py`、`tests/unit/ingestion_pipeline/embedding/test_optimization.py`
- 实现类/函数：`SparseEncoder.encode()`、`optimize_embedding_batches()`
- 验收标准：BM25 token/term 结果稳定；大批量输入可按批次处理；优化逻辑不改变编码结果语义。
- 测试方法：`pytest tests/unit/ingestion_pipeline/embedding/test_sparse_encoder.py tests/unit/ingestion_pipeline/embedding/test_optimization.py`

##### C12 实现 Storage 主编排与向量库 Upsert

- 目标：编排摄取结果写入流程，并完成向量库写入。
- 修改文件：`src/ragms/ingestion_pipeline/storage/pipeline.py`、`src/ragms/ingestion_pipeline/storage/vector_upsert.py`、`tests/unit/ingestion_pipeline/storage/test_pipeline.py`、`tests/unit/ingestion_pipeline/storage/test_vector_upsert.py`
- 实现类/函数：`StoragePipeline.run()`、`VectorUpsert.write()`
- 验收标准：Dense 向量、chunk、metadata 可统一写入；重复写入保持幂等。
- 测试方法：`pytest tests/unit/ingestion_pipeline/storage/test_pipeline.py tests/unit/ingestion_pipeline/storage/test_vector_upsert.py`

##### C13 实现 BM25 索引构建与图片文件存储

- 目标：完成 BM25 索引更新、图片文件落盘与 `images` 索引表写入。
- 修改文件：`src/ragms/ingestion_pipeline/storage/bm25_indexing.py`、`src/ragms/ingestion_pipeline/storage/image_persistence.py`、`src/ragms/storage/indexes/bm25_indexer.py`、`src/ragms/storage/images/image_storage.py`、`src/ragms/storage/sqlite/repositories/images.py`、`tests/unit/ingestion_pipeline/storage/test_bm25_indexing.py`、`tests/unit/ingestion_pipeline/storage/test_image_persistence.py`
- 实现类/函数：`BM25StorageWriter.index()`、`ImageStorageWriter.save_all()`、`BM25Indexer.index_document()`、`ImagesRepository`
- 验收标准：BM25 索引可增量更新；图片可稳定落盘、写入 `images` 索引表并与文档/chunk 建立查询关联。
- 测试方法：`pytest tests/unit/ingestion_pipeline/storage/test_bm25_indexing.py tests/unit/ingestion_pipeline/storage/test_image_persistence.py`

##### C14 打通 Ingestion Pipeline 与 CLI 集成链路

- 目标：完成离线摄取主编排器、CLI 入口与存储集成测试。
- 修改文件：`src/ragms/ingestion_pipeline/pipeline.py`、`scripts/ingest_documents.py`、`tests/integration/test_ingestion_pipeline.py`、`tests/integration/test_ingestion_pipeline_storage.py`
- 实现类/函数：`IngestionPipeline.run()`、`ingest_documents_main()`
- 验收标准：样例 PDF 能从 CLI 完整入库；增量跳过生效；Chroma/BM25/图片存储集成通过。
- 测试方法：`pytest tests/integration/test_ingestion_pipeline.py tests/integration/test_ingestion_pipeline_storage.py`

#### 阶段 D：Retrieval（Query Processor + Dense/Sparse + RRF + Rerank）

目标：在线查询链路跑通，能够完成 Query 预处理、Dense/Sparse 双路召回、RRF 融合、带 fallback 的可选精排，并通过本地查询 CLI 返回答案、引用与结构化结果。

| 任务编号 | 任务名称 | 状态 | 完成日期 | 备注 |
|---------|---------|------|---------|------|
| D1 | 实现 Query Processor 关键词提取与 Filter 解析 | [ ] |  |  |
| D2 | 定义 Retrieval 结果模型 | [ ] |  |  |
| D3 | 实现 DenseRetriever（调用 `BaseVectorStore.query`） | [ ] |  |  |
| D4 | 实现 SparseRetriever（BM25 查询） | [ ] |  |  |
| D5 | 实现 RRF 融合 | [ ] |  |  |
| D6 | 实现 HybridSearch 编排 | [ ] |  |  |
| D7 | 实现 Reranker（Core 编排 + Fallback） | [ ] |  |  |
| D8 | 打通 Query Engine 与查询 CLI | [ ] |  |  |

##### D1 实现 Query Processor 关键词提取与 Filter 解析

- 目标：完成 Query 归一化、关键词提取、受控扩展与 metadata filter 解析，为 Dense/Sparse 两条路线生成统一输入。
- 修改文件：`src/ragms/core/query_engine/query_processor.py`、`tests/unit/core/query_engine/test_query_processor.py`
- 实现类/函数：`QueryProcessor.extract_keywords()`、`QueryProcessor.parse_filters()`、`QueryProcessor.process()`
- 验收标准：输入 Query 可完成空白归一化、关键词提取与停用词过滤；`collection`、`top_k`、`filters` 等请求参数可被正确校验与解析；可输出 Dense 路线使用的标准化 query、Sparse 路线使用的关键词表达式以及可前置/后置过滤信息。
- 测试方法：`pytest tests/unit/core/query_engine/test_query_processor.py`

##### D2 定义 Retrieval 结果模型

- 目标：在 `core/models/retrieval.py` 中定义检索链路统一使用的候选与结果模型，作为 Dense / Sparse / RRF / Reranker 的共享数据契约。
- 修改文件：`src/ragms/core/models/retrieval.py`、`tests/unit/core/models/test_retrieval_models.py`
- 实现类/函数：`RetrievalCandidate`、`HybridSearchResult`
- 验收标准：检索结果模型可统一表达 `chunk_id`、`document_id`、content、metadata、score、source_route` 等核心字段，并兼容 `dense_rank`、`sparse_rank`、`rrf_score`、`rerank_score`、fallback 标记等阶段补充信息；序列化与比较行为稳定。
- 测试方法：`pytest tests/unit/core/models/test_retrieval_models.py`

##### D3 实现 DenseRetriever（调用 `BaseVectorStore.query`）

- 目标：实现稠密检索路径，基于单次 Query Embedding 调用向量库查询并返回 `RetrievalCandidate`。
- 修改文件：`src/ragms/core/models/retrieval.py`、`src/ragms/core/query_engine/retrievers/dense_retriever.py`、`tests/unit/core/query_engine/test_dense_retriever.py`
- 实现类/函数：`DenseRetriever.retrieve()`
- 验收标准：DenseRetriever 会基于标准化 query 只生成一次查询向量，并调用 `BaseVectorStore.query()` 执行检索；支持 `collection` 与可前置 metadata filter；返回结果统一为 `RetrievalCandidate` 列表，至少包含 `chunk_id`、`document_id`、score、metadata、source_route=`dense` 等标准字段。
- 测试方法：`pytest tests/unit/core/query_engine/test_dense_retriever.py`

##### D4 实现 SparseRetriever（BM25 查询）

- 目标：实现稀疏检索路径，基于关键词表达式执行 BM25 查询并返回 `RetrievalCandidate`。
- 修改文件：`src/ragms/core/models/retrieval.py`、`src/ragms/core/query_engine/retrievers/sparse_retriever.py`、`src/ragms/storage/indexes/bm25_indexer.py`、`tests/unit/core/query_engine/test_sparse_retriever.py`
- 实现类/函数：`SparseRetriever.retrieve()`、`BM25Indexer.search()`
- 验收标准：SparseRetriever 会对“原始关键词 + 受控扩展词”执行一次 BM25 查询，而不是对每个扩展词分别检索；支持 `collection` 与可前置 metadata filter；返回结果统一为 `RetrievalCandidate` 列表，至少包含 `chunk_id`、`document_id`、score、metadata、source_route=`sparse` 等标准字段。
- 测试方法：`pytest tests/unit/core/query_engine/test_sparse_retriever.py`

##### D5 实现 RRF 融合

- 目标：实现基于排名的 Dense/Sparse 结果融合，保证混合召回排序稳定且可解释。
- 修改文件：`src/ragms/core/query_engine/hybrid_search.py`、`tests/unit/core/query_engine/test_rrf.py`
- 实现类/函数：`reciprocal_rank_fusion()`
- 验收标准：RRF 仅基于排名而非原始分数做融合；重复命中文档会被去重并自然提升；单路强命中结果可被保留；不同输入顺序下融合结果稳定，`k` 平滑参数可配置。
- 测试方法：`pytest tests/unit/core/query_engine/test_rrf.py`

##### D6 实现 HybridSearch 编排

- 目标：编排 DenseRetriever、SparseRetriever、Pre/Post-filter 与 RRF 融合，形成统一候选集输出。
- 修改文件：`src/ragms/core/query_engine/hybrid_search.py`、`tests/unit/core/query_engine/test_hybrid_search.py`
- 实现类/函数：`HybridSearch.search()`
- 验收标准：HybridSearch 可并行调度 Dense / Sparse 两条召回路径；前置过滤条件会传递给检索器，后置过滤条件会在融合前后按策略执行；输出结果会按 `candidate_top_n` 截断，并保留 route 来源、融合分数、过滤统计等编排信息。
- 测试方法：`pytest tests/unit/core/query_engine/test_hybrid_search.py`

##### D7 实现 Reranker（Core 编排 + Fallback）

- 目标：在 Core 层实现 `cross_encoder / llm_reranker / disabled` 三种精排模式，并在不可用时自动回退到 RRF 结果。
- 修改文件：`src/ragms/core/query_engine/reranker.py`、`tests/unit/core/query_engine/test_reranker.py`
- 实现类/函数：`Reranker.run()`、`Reranker.run_with_fallback()`
- 验收标准：Reranker 可根据配置切换 `cross_encoder`、`llm_reranker` 与 `disabled` 三种模式；重排输入固定取 HybridSearch 输出的 `candidate_top_n`；当 reranker 初始化失败、调用超时或执行报错时，请求级自动回退到 RRF 排序结果，并保留统一的 `RetrievalCandidate` / `HybridSearchResult` 结构与 fallback 标记。
- 测试方法：`pytest tests/unit/core/query_engine/test_reranker.py`

##### D8 打通 Query Engine 与查询 CLI

- 目标：形成 `Query Processor -> Hybrid Search -> Reranker -> Response Builder -> Answer Generator` 的完整在线链路，并对外提供稳定的本地 CLI 入口，同时为后续 Query Trace 接入预留稳定扩展点。
- 修改文件：`src/ragms/core/query_engine/engine.py`、`src/ragms/core/query_engine/response_builder.py`、`src/ragms/core/query_engine/citation_builder.py`、`src/ragms/core/query_engine/answer_generator.py`、`scripts/query_cli.py`、`tests/unit/core/query_engine/test_response_builder.py`、`tests/unit/core/query_engine/test_citation_builder.py`、`tests/unit/core/query_engine/test_answer_generator.py`、`tests/integration/test_query_engine.py`
- 实现类/函数：`QueryEngine.run()`、`ResponseBuilder.build()`、`CitationBuilder.build()`、`AnswerGenerator.generate()`、`run_cli()`
- 验收标准：样例 Query 可从 CLI 触发完整在线查询链路；返回结果包含答案正文、引用信息与结构化候选摘要；Query Engine 对 Trace 上下文透传和阶段打点预留稳定扩展点；无结果查询、filter 生效、reranker fallback 等关键路径均可被集成测试覆盖。
- 测试方法：`pytest tests/unit/core/query_engine/test_response_builder.py tests/unit/core/query_engine/test_citation_builder.py tests/unit/core/query_engine/test_answer_generator.py tests/integration/test_query_engine.py`

#### 阶段 E：MCP Server 层与 Tools 落地

目标：完成基于 STDIO Transport 的 MCP Server 落地，稳定暴露 `query_knowledge_hub`、`ingest_documents`、`list_collections`、`get_document_summary` 四类核心工具，完成 tool schema、参数校验、统一错误语义、引用透明和结构化响应契约；同时为后续阶段接入 `get_trace_detail` 与 `evaluate_collection` 预留稳定扩展点。

| 任务编号 | 任务名称 | 状态 | 完成日期 | 备注 |
|---------|---------|------|---------|------|
| E1 | 实现 MCP Server 启动入口与生命周期 | [ ] |  |  |
| E2 | 实现 Tool Registry 与 Tool 元数据声明 | [ ] |  |  |
| E3 | 实现 Schema 定义与参数校验 | [ ] |  |  |
| E4 | 实现统一协议响应包装与错误收敛 | [ ] |  |  |
| E5 | 实现 `query_knowledge_hub` Tool | [ ] |  |  |
| E6 | 实现 `ingest_documents` Tool | [ ] |  |  |
| E7 | 实现 `list_collections` Tool | [ ] |  |  |
| E8 | 实现 `get_document_summary` Tool | [ ] |  |  |
| E9 | 完成核心工具 MCP 集成测试与 STDIO E2E 验收 | [ ] |  |  |

##### E1 实现 MCP Server 启动入口与生命周期

- 目标：建立 MCP Server 主进程骨架，固定“配置加载 -> 容器装配 -> Server 创建 -> Tool 注册 -> STDIO 监听 -> 优雅退出”的生命周期。
- 修改文件：`src/ragms/mcp_server/server.py`、`src/ragms/runtime/container.py`、`scripts/run_mcp_server.py`、`tests/unit/mcp_server/test_server_bootstrap.py`
- 实现类/函数：`create_server()`、`run_mcp_server_main()`、`bootstrap_mcp_runtime()`
- 验收标准：Server 可通过脚本正常启动；启动阶段可完成依赖装配并进入监听状态；初始化失败时会返回统一异常并安全退出，不留下半初始化状态。
- 测试方法：`pytest tests/unit/mcp_server/test_server_bootstrap.py`

##### E2 实现 Tool Registry 与 Tool 元数据声明

- 目标：集中定义所有对外 MCP tools 的名称、描述、输入输出 schema 绑定关系和处理函数映射，避免 tool 声明散落在多个模块。
- 修改文件：`src/ragms/mcp_server/tool_registry.py`、`src/ragms/mcp_server/server.py`、`tests/unit/mcp_server/test_tool_registry.py`
- 实现类/函数：`build_tool_registry()`、`register_tools()`、`ToolDefinition`
- 验收标准：当前阶段交付的四类核心工具均可在注册表中集中声明，并为后续 `get_trace_detail`、`evaluate_collection` 接入预留扩展点；tool 名称、描述、schema 和 handler 绑定关系一致；重复注册、未知 handler 或缺失 schema 会被快速拦截。
- 测试方法：`pytest tests/unit/mcp_server/test_tool_registry.py`

##### E3 实现 Schema 定义与参数校验

- 目标：为阶段 E 交付的四类核心工具建立独立的请求/响应 schema，并在协议层完成统一的参数校验、默认值注入和类型错误提示，同时为后续 Trace / Evaluation tool 扩展保留兼容空间。
- 修改文件：`src/ragms/mcp_server/schemas.py`、`src/ragms/mcp_server/protocol_handler.py`、`tests/unit/mcp_server/test_schemas.py`
- 实现类/函数：`QueryToolRequest`、`IngestToolRequest`、`CollectionToolRequest`、`DocumentSummaryRequest`、`ProtocolHandler.validate_arguments()`
- 验收标准：各 tool 的必填参数、默认参数和类型约束明确；非法输入会在进入业务逻辑前被拦截；校验错误信息可直接暴露给 MCP Client 使用。
- 测试方法：`pytest tests/unit/mcp_server/test_schemas.py`

##### E4 实现统一协议响应包装与错误收敛

- 目标：实现 MCP 响应包装器，统一生成 `content`、`structuredContent`、错误响应和可选调试字段，确保 tool 层不重复处理协议细节。
- 修改文件：`src/ragms/mcp_server/protocol_handler.py`、`src/ragms/mcp_server/schemas.py`、`tests/unit/mcp_server/test_protocol_handler.py`
- 实现类/函数：`ProtocolHandler.handle()`、`ProtocolHandler.build_success_response()`、`ProtocolHandler.build_error_response()`、`ProtocolHandler.serialize_exception()`
- 验收标准：成功响应可统一输出人类可读内容和结构化内容；错误响应可稳定收敛参数错误、业务错误和内部异常；未知 tool、非法参数、底层异常都不会泄露原始栈信息。
- 测试方法：`pytest tests/unit/mcp_server/test_protocol_handler.py`

##### E5 实现 `query_knowledge_hub` Tool

- 目标：落地查询工具，完成 Query Engine 结果到 MCP 输出的适配，并保证引用透明和 `trace_id` 透传。
- 修改文件：`src/ragms/mcp_server/tools/query.py`、`src/ragms/core/query_engine/response_builder.py`、`tests/unit/mcp_server/tools/test_query_tool.py`、`tests/integration/test_mcp_server_query.py`
- 实现类/函数：`handle_query_knowledge_hub()`、`build_query_structured_content()`、`build_query_markdown_content()`
- 验收标准：工具可接收 `query`、`collection`、`top_k`、`filters`、`return_debug` 等参数；返回结果包含回答、`structuredContent.citations`、候选摘要和 `trace_id`；Markdown 中的 `[1]`、`[2]` 与结构化引用编号一一对应；无结果和 fallback 场景说明明确。
- 测试方法：`pytest tests/unit/mcp_server/tools/test_query_tool.py tests/integration/test_mcp_server_query.py`

##### E6 实现 `ingest_documents` Tool

- 目标：落地摄取工具，完成批量路径受理、增量跳过、强制重建等参数适配，并将摄取结果标准化为 MCP 输出。
- 修改文件：`src/ragms/mcp_server/tools/ingest.py`、`scripts/ingest_documents.py`、`tests/unit/mcp_server/tools/test_ingest_tool.py`、`tests/integration/test_mcp_server_ingest.py`
- 实现类/函数：`handle_ingest_documents()`、`normalize_ingest_request()`、`serialize_ingestion_result()`
- 验收标准：工具支持单路径和多路径摄取；支持 `collection`、`force_rebuild`、`options` 等参数；返回结果包含文档级状态、跳过/失败摘要、受理统计和 `trace_id`；部分文档失败时整体响应结构仍稳定。
- 测试方法：`pytest tests/unit/mcp_server/tools/test_ingest_tool.py tests/integration/test_mcp_server_ingest.py`

##### E7 实现 `list_collections` Tool

- 目标：落地集合浏览工具，返回可供 Agent 做预检查和集合选择的基础统计信息。
- 修改文件：`src/ragms/mcp_server/tools/collections.py`、`src/ragms/core/management/data_service.py`、`tests/unit/mcp_server/tools/test_collections_tool.py`
- 实现类/函数：`handle_list_collections()`、`serialize_collection_summary()`、`DataService.list_collections()`
- 验收标准：工具可返回集合名、文档数、chunk 数、更新时间等摘要字段；支持空集合和无结果场景；输出结构稳定，便于客户端直接消费。
- 测试方法：`pytest tests/unit/mcp_server/tools/test_collections_tool.py`

##### E8 实现 `get_document_summary` Tool

- 目标：落地文档摘要工具，支持按 `document_id` 回溯文档摘要、结构概览和最新摄取状态。
- 修改文件：`src/ragms/mcp_server/tools/documents.py`、`src/ragms/core/management/data_service.py`、`src/ragms/storage/sqlite/repositories/documents.py`、`tests/unit/mcp_server/tools/test_documents_tool.py`
- 实现类/函数：`handle_get_document_summary()`、`serialize_document_summary()`、`DataService.get_document_summary()`
- 验收标准：工具可返回文档摘要、章节概览、来源路径、关键 metadata、摄取状态及必要的图片/页码摘要；文档不存在或状态不完整时返回语义清晰的错误或空结果。
- 测试方法：`pytest tests/unit/mcp_server/tools/test_documents_tool.py`

##### E9 完成核心工具 MCP 集成测试与 STDIO E2E 验收

- 目标：完成进程内集成测试和真实子进程 STDIO E2E 测试，验证初始化、工具发现、四类核心工具调用、异常返回与进程退出行为。
- 修改文件：`tests/integration/test_mcp_server.py`、`tests/integration/test_mcp_server_query.py`、`tests/integration/test_mcp_server_ingest.py`、`tests/integration/test_mcp_server_documents.py`、`tests/e2e/test_mcp_stdio_flow.py`、`tests/e2e/test_mcp_tool_contracts.py`
- 实现类/函数：`start_stdio_server_process()`、`call_mcp_tool()`、`assert_mcp_response_contract()`
- 验收标准：模拟 MCP Client 可完成初始化、列出工具并调用四类核心工具；`content` / `structuredContent` 契约在真实 STDIO 边界下成立；异常请求不会导致服务崩溃；阶段 E 完成后系统已具备可被本地 Agent 直接接入的稳定 MCP 核心能力。
- 测试方法：`pytest tests/integration/test_mcp_server.py tests/integration/test_mcp_server_query.py tests/integration/test_mcp_server_ingest.py tests/integration/test_mcp_server_documents.py tests/e2e/test_mcp_stdio_flow.py tests/e2e/test_mcp_tool_contracts.py`

#### 阶段 F：Trace 基础设施与打点

目标：增强 `TraceManager`，实现结构化日志持久化，在 Ingestion + Query 双链路打点，添加 Pipeline 进度回调，并在底层能力稳定后接入 `get_trace_detail` MCP Tool。

| 任务编号 | 任务名称 | 状态 | 完成日期 | 备注 |
|---------|---------|------|---------|------|
| F1 | 实现 Trace 模型与 TraceManager | [ ] |  |  |
| F2 | 实现 JSONL 日志持久化 | [ ] |  |  |
| F3 | 为 Ingestion 打点 | [ ] |  |  |
| F4 | 为 Query 打点 | [ ] |  |  |
| F5 | 增加进度回调与双链路集成测试 | [ ] |  |  |
| F6 | 接入 `get_trace_detail` MCP Tool | [ ] |  |  |

##### F1 实现 Trace 模型与 TraceManager

- 目标：定义 QueryTrace / IngestionTrace / StageTrace 以及 TraceManager。
- 修改文件：`src/ragms/core/trace_collector/trace_schema.py`、`src/ragms/core/trace_collector/trace_manager.py`
- 实现类/函数：`TraceManager.start_trace()`、`TraceManager.finish_trace()`
- 验收标准：Trace 生命周期完整；阶段可注册与收敛。
- 测试方法：`pytest tests/unit/core/trace_collector/test_trace_manager.py`

##### F2 实现 JSONL 日志持久化

- 目标：将 Trace 结构化写入 `logs/traces.jsonl`。
- 修改文件：`src/ragms/storage/traces/jsonl_writer.py`、`src/ragms/observability/logging/json_formatter.py`
- 实现类/函数：`JsonlTraceWriter.write()`、`JsonFormatter.format()`
- 验收标准：单次请求可落为一条完整 JSONL 记录。
- 测试方法：`pytest tests/unit/observability/test_json_formatter.py`

##### F3 为 Ingestion 打点

- 目标：在 File Integrity、Loader、Splitter、Transform、Embedding、Upsert 等阶段统一打点。
- 修改文件：`src/ragms/ingestion_pipeline/pipeline.py`、`src/ragms/ingestion_pipeline/chunking/split.py`、`src/ragms/ingestion_pipeline/transform/pipeline.py`、`src/ragms/ingestion_pipeline/embedding/pipeline.py`、`src/ragms/ingestion_pipeline/storage/pipeline.py`
- 实现类/函数：`record_ingestion_stage()`
- 验收标准：Ingestion Trace 可完整显示各阶段耗时、状态和关键摘要。
- 测试方法：`pytest tests/integration/test_ingestion_trace_logging.py`

##### F4 为 Query 打点

- 目标：在 Query Processor、Hybrid Search、Reranker、Response Builder 等阶段统一打点。
- 修改文件：`src/ragms/core/query_engine/engine.py`、`src/ragms/core/query_engine/query_processor.py`、`src/ragms/core/query_engine/hybrid_search.py`、`src/ragms/core/query_engine/reranker.py`、`src/ragms/core/query_engine/response_builder.py`
- 实现类/函数：`record_query_stage()`
- 验收标准：Query Trace 可完整显示检索、融合、精排、生成细节。
- 测试方法：`pytest tests/integration/test_query_trace_logging.py`

##### F5 增加进度回调与双链路集成测试

- 目标：为 Ingestion / Dashboard 增加进度回调，并完成双链路 Trace 集成验证。
- 修改文件：`src/ragms/ingestion_pipeline/pipeline.py`、`src/ragms/core/management/trace_service.py`
- 实现类/函数：`on_progress`、`TraceService.list_traces()`
- 验收标准：摄取过程可实时回传进度；Dashboard 可按类型读取 Trace。
- 测试方法：`pytest tests/integration/test_trace_write_and_read.py`

##### F6 接入 `get_trace_detail` MCP Tool

- 目标：在 TraceManager、JSONL 持久化和 TraceService 能力稳定后，对外暴露 `get_trace_detail` MCP Tool。
- 修改文件：`src/ragms/mcp_server/tools/traces.py`、`src/ragms/mcp_server/schemas.py`、`src/ragms/mcp_server/tool_registry.py`、`src/ragms/core/management/trace_service.py`、`tests/unit/mcp_server/tools/test_traces_tool.py`、`tests/integration/test_mcp_server_trace.py`
- 实现类/函数：`handle_get_trace_detail()`、`serialize_trace_detail()`、`TraceService.get_trace_detail()`
- 验收标准：工具可按 `trace_id` 返回 `trace_type`、阶段列表、耗时、状态、输入输出摘要、错误信息和 fallback 记录；缺失 trace、损坏日志和非法 `trace_id` 会被统一收敛；MCP Client 可通过 tool 直接查询真实 Trace 记录。
- 测试方法：`pytest tests/unit/mcp_server/tools/test_traces_tool.py tests/integration/test_mcp_server_trace.py`

#### 阶段 G：可视化管理平台 Dashboard

目标：基于第 3.4 节的 Dashboard 设计、第 4 章的测试约束和第 5 章的模块分层，完成一个本地优先、离线可运行、以 `core/management` 为主要业务服务层、以 `core/evaluation/report_service.py` 提供只读评估报告接口的 Streamlit Dashboard。阶段 G 完成后，系统必须已经具备六页面可视化管理能力：可查看系统总览、浏览 collection/document/chunk/image、执行文档级管理操作、按 `trace_id` 追踪 Ingestion / Query 链路，并在无评估报告时显示明确空态、在存在预置样例或已落盘报告时展示评估摘要；页面之间支持从文档跳转到 Trace、从 Trace 回溯到文档或 chunk、从评估结果回看对应运行配置与指标明细，满足第 1 章“本地优先、可观测、可测试”的既定目标。

| 任务编号 | 任务名称 | 状态 | 完成日期 | 备注 |
|---------|---------|------|---------|------|
| G1 | 建立 Dashboard 应用壳、页面注册与运行配置 | [ ] |  |  |
| G2 | 完成 Dashboard 共享数据聚合接口与通用组件 | [ ] |  |  |
| G3 | 实现系统总览页 | [ ] |  |  |
| G4 | 实现数据浏览器页 | [ ] |  |  |
| G5 | 实现 Ingestion 管理页 | [ ] |  |  |
| G6 | 实现 Ingestion 追踪页 | [ ] |  |  |
| G7 | 实现 Query 追踪页 | [ ] |  |  |
| G8 | 实现评估面板、页面联动与 Dashboard 验收 | [ ] |  |  |

##### G1 建立 Dashboard 应用壳、页面注册与运行配置

- 目标：搭建 Streamlit 入口、统一导航、页面注册、运行时依赖装配和自动刷新机制，保证 Dashboard 在本地离线环境下可直接启动。
- 修改文件：`scripts/run_dashboard.py`、`src/ragms/observability/dashboard/app.py`、`src/ragms/runtime/container.py`、`src/ragms/runtime/config.py`
- 实现类/函数：`run_dashboard_main()`、`render_app_shell()`、`build_dashboard_context()`
- 验收标准：`scripts/run_dashboard.py` 可启动 Dashboard；六个页面可被统一注册与切换；自动刷新、日志路径、数据目录、端口等行为由配置驱动；无网络连接时页面仍可读取本地 JSONL / SQLite / Chroma 派生数据完成渲染。
- 测试方法：`pytest tests/integration/test_dashboard_shell.py`

##### G2 完成 Dashboard 共享数据聚合接口与通用组件

- 目标：收敛 Dashboard 侧所有查询与展示契约，避免页面直接读写底层存储；补齐表格、图表、Trace 时间线等共享组件，形成统一展示基座，并为评估面板建立只读报告装载契约。
- 修改文件：`src/ragms/core/management/data_service.py`、`src/ragms/core/management/trace_service.py`、`src/ragms/core/evaluation/report_service.py`、`src/ragms/observability/dashboard/components/tables.py`、`src/ragms/observability/dashboard/components/charts.py`、`src/ragms/observability/dashboard/components/trace_timeline.py`
- 实现类/函数：`DataService.get_system_overview_metrics()`、`TraceService.list_trace_summaries()`、`ReportService.list_evaluation_runs()`、`render_table()`、`render_chart()`、`render_trace_timeline()`
- 验收标准：页面层不直接拼接 SQL、JSONL 解析或 Chroma 查询；共享组件可稳定渲染空态、错误态和正常态；Trace 与 provider 信息采用动态字段渲染，不写死阶段名或模型名；评估面板只消费只读报告接口，不直接扫描页面层私有目录。
- 测试方法：`pytest tests/integration/test_dashboard_data_access.py tests/integration/test_dashboard_system_overview.py`

##### G3 实现系统总览页

- 目标：提供系统整体运行面板，汇总知识库规模、组件配置摘要、最近 Trace、异常概览和关键趋势指标。
- 修改文件：`src/ragms/observability/dashboard/pages/system_overview.py`、`src/ragms/core/management/data_service.py`、`src/ragms/core/management/trace_service.py`
- 实现类/函数：`render_system_overview()`、`DataService.get_collection_statistics()`、`TraceService.get_recent_failures()`
- 验收标准：页面可展示集合数、文档数、chunk 数、图片数、最近摄取/查询记录、失败统计、耗时趋势和当前组件配置摘要；当 Trace 或集合为空时能给出明确空态提示，不出现空白页。
- 测试方法：`pytest tests/integration/test_dashboard_system_overview.py`

##### G4 实现数据浏览器页

- 目标：提供 collection -> document -> chunk -> image 的逐层浏览和回溯能力，支撑第 3.1 节与第 3.5 节定义的文档生命周期和多模态检视需求。
- 修改文件：`src/ragms/observability/dashboard/pages/data_browser.py`、`src/ragms/core/management/data_service.py`
- 实现类/函数：`render_data_browser()`、`DataService.list_documents()`、`DataService.get_document_detail()`、`DataService.get_chunk_detail()`
- 验收标准：支持按 collection、文档状态、页码、标签、关键词等条件过滤；可查看 chunk 正文、metadata、引用来源、图片预览与关联 `trace_id`；支持从文档详情跳转到相关 Ingestion Trace。
- 测试方法：`pytest tests/integration/test_dashboard_data_access.py`

##### G5 实现 Ingestion 管理页

- 目标：把摄取侧文档管理能力完整暴露到 UI，支持触发摄取、查看进度、识别增量跳过、删除文档和重建文档。
- 修改文件：`src/ragms/observability/dashboard/pages/ingestion_management.py`、`src/ragms/core/management/document_admin_service.py`、`src/ragms/ingestion_pipeline/pipeline.py`
- 实现类/函数：`render_ingestion_management()`、`DocumentAdminService.ingest_documents()`、`DocumentAdminService.delete_document()`、`DocumentAdminService.rebuild_document()`
- 验收标准：页面可展示文档状态、最近摄取时间、失败原因、跳过原因和版本信息；触发删除或重建后状态可刷新并与 SQLite / Chroma / 图片存储保持一致；摄取执行过程中可显示阶段进度与当前状态。
- 测试方法：`pytest tests/integration/test_dashboard_ingestion_management.py`

##### G6 实现 Ingestion 追踪页

- 目标：为摄取链路提供按 `trace_id` 的完整回放能力，支持阶段时间线、摘要、错误和上下文定位。
- 修改文件：`src/ragms/observability/dashboard/pages/ingestion_trace.py`、`src/ragms/core/management/trace_service.py`、`src/ragms/observability/dashboard/components/trace_timeline.py`
- 实现类/函数：`render_ingestion_trace()`、`TraceService.list_traces(trace_type=\"ingestion\")`、`TraceService.get_trace_detail()`
- 验收标准：支持按时间、状态、collection、`trace_id` 过滤；页面可展示 Load、Split、Transform、Embed、Upsert 等阶段的耗时、输入输出摘要、provider、错误信息和进度；可从 trace 回跳到对应文档或源文件。
- 测试方法：`pytest tests/integration/test_dashboard_ingestion_trace.py`

##### G7 实现 Query 追踪页

- 目标：为查询链路提供细粒度排障视图，覆盖 Query Processing、Dense / Sparse Retrieval、Fusion、Rerank、Generation 等阶段，并支持关键 trace 对比。
- 修改文件：`src/ragms/observability/dashboard/pages/query_trace.py`、`src/ragms/core/management/trace_service.py`、`src/ragms/observability/dashboard/components/charts.py`
- 实现类/函数：`render_query_trace()`、`TraceService.compare_traces()`、`render_query_trace_comparison()`
- 验收标准：页面可展示改写后查询、关键词、召回数量、融合结果、重排摘要、生成耗时、fallback 信息和最终引用摘要；支持选择两个 `trace_id` 做指标与阶段耗时对比；若查询未进入生成阶段，页面也能按实际阶段动态渲染。
- 测试方法：`pytest tests/integration/test_dashboard_query_trace.py tests/integration/test_dashboard_trace_compare.py`

##### G8 实现评估面板、页面联动与 Dashboard 验收

- 目标：完成评估面板页面壳、跨页面跳转与整站验收，使 Dashboard 在阶段 G 结束时已形成完整的管理闭环。该任务以“支持无报告空态、预置样例报告和本地已落盘报告的只读展示”为范围，不要求系统在本阶段已经完成真实评估执行；真实评估运行、golden baseline 建立和 `test_evaluation_visible_in_dashboard.py` 的验收继续在阶段 H 完成。
- 修改文件：`src/ragms/observability/dashboard/pages/evaluation_panel.py`、`src/ragms/core/evaluation/report_service.py`、`tests/integration/test_dashboard_navigation.py`、`tests/e2e/test_dashboard_smoke.py`
- 实现类/函数：`render_evaluation_panel()`、`ReportService.load_report_detail()`、`resolve_dashboard_navigation_target()`
- 验收标准：评估面板在无报告时可展示明确空态，在存在预置样例或本地已落盘报告时可展示摘要、指标概览和详情入口；系统总览、数据浏览、Ingestion 管理、Ingestion 追踪、Query 追踪、评估面板六页均可正常打开并完成关键交互；文档 -> Trace、Trace -> 文档 / chunk、评估结果 -> collection / 配置摘要等关键跳转成立；Dashboard 冒烟测试通过。
- 测试方法：`pytest tests/integration/test_dashboard_navigation.py tests/e2e/test_dashboard_smoke.py`

阶段 G 完成判定：

- 本地执行 `scripts/run_dashboard.py` 后可稳定打开六个页面，且无外部网络依赖。
- 所有页面仅通过 `core/management` 与 `core/evaluation/report_service.py` 访问业务数据，不在 UI 层直接操作底层存储。
- 文档、chunk、图片、Ingestion Trace、Query Trace、评估报告之间形成可点击的双向回溯链路。
- Query Trace 对比能力和跨页面跳转能力有明确测试覆盖，不依赖人工点选验证。
- 删除、重建、摄取进度查看、Trace 过滤与对比、评估结果筛选等核心交互均可用。
- `pytest tests/integration/test_dashboard_shell.py tests/integration/test_dashboard_system_overview.py tests/integration/test_dashboard_data_access.py tests/integration/test_dashboard_ingestion_management.py tests/integration/test_dashboard_ingestion_trace.py tests/integration/test_dashboard_query_trace.py tests/integration/test_dashboard_navigation.py tests/integration/test_dashboard_trace_compare.py tests/e2e/test_dashboard_smoke.py` 全部通过。

#### 阶段 H：评估体系

目标：基于第 3.3.4 节评估框架抽象、第 4.3 节质量评估目标、第 5 章模块边界以及阶段 A-G 已交付的摄取、检索、MCP、Trace 与 Dashboard 能力，完成“可运行、可比较、可回归、可收口”的评估体系。阶段 H 不是单独补一个 evaluator，而是要把 `dataset -> evaluator -> runner -> report -> dashboard -> MCP tool -> regression gate` 整条质量闭环打通；阶段 H 完成后，系统必须能够对指定 collection 执行真实评估、生成结构化报告、在 Dashboard 中查看和比较实验结果、通过 MCP Tool 远程触发评估，并且补齐所有此前阶段遗留但会影响评估闭环与质量达标的未完成部分，使系统在进入阶段 I 前不再存在 A-G 的前置阻塞项。

| 任务编号 | 任务名称 | 状态 | 完成日期 | 备注 |
|---------|---------|------|---------|------|
| H1 | 收敛评估输入契约并补齐 A-G 前置遗留项 | [ ] |  |  |
| H2 | 建立 golden test set、坏 case 集与数据集加载器 | [ ] |  |  |
| H3 | 实现检索指标与自定义指标评估器 | [ ] |  |  |
| H4 | 实现 `RagasEvaluator` 与评估后端适配层 | [ ] |  |  |
| H5 | 实现 `CompositeEvaluator`、工厂装配与配置切换 | [ ] |  |  |
| H6 | 实现 `EvalRunner`、评估报告落盘与实验对比元数据 | [ ] |  |  |
| H7 | 打通 Dashboard 真实评估结果、趋势展示与 baseline 管理 | [ ] |  |  |
| H8 | 接入 `evaluate_collection` MCP Tool | [ ] |  |  |
| H9 | 基于评估结果完成质量门槛达标与前序阶段收口 | [ ] |  |  |

##### H1 收敛评估输入契约并补齐 A-G 前置遗留项

- 目标：先收敛评估链路所依赖的输入契约，把阶段 A-G 中所有会阻塞评估闭环的遗留问题在本任务内补齐，而不是留到后续阶段被动修补。
- 修改文件：`src/ragms/core/models/retrieval.py`、`src/ragms/core/models/response.py`、`src/ragms/core/query_engine/response_builder.py`、`src/ragms/core/query_engine/citation_builder.py`、`src/ragms/core/management/trace_service.py`、`src/ragms/runtime/settings_models.py`
- 实现类/函数：`normalize_evaluation_sample()`、`build_evaluation_input()`、`snapshot_runtime_config()`
- 验收标准：评估输入统一包含 `query`、`retrieved_chunks`、`generated_answer`、`ground_truth`、`citations`、`trace_id`、`collection`、`config_snapshot`、`dataset_version` 等核心字段；Query / MCP / Dashboard / Report 之间对样本 ID、引用、配置快照和 trace 关联的口径一致；凡是 A-G 中阻塞评估闭环的未完成部分，必须在本任务内补齐并同步更新相关测试。
- 测试方法：`pytest tests/integration/test_query_engine.py tests/integration/test_mcp_server_query.py tests/integration/test_trace_write_and_read.py`

##### H2 建立 golden test set、坏 case 集与数据集加载器

- 目标：建立标准化评估数据资产，使评估既能覆盖正常样本，也能覆盖已知坏 case、图片相关样本和过滤查询样本。
- 修改文件：`data/evaluation/datasets/*`、`src/ragms/core/evaluation/dataset_loader.py`、`src/ragms/core/models/evaluation.py`
- 实现类/函数：`DatasetLoader.load()`、`DatasetLoader.validate_manifest()`、`DatasetLoader.list_dataset_versions()`
- 验收标准：golden test set、回归坏 case 集和样本 manifest 可被统一读取；样本结构可映射为标准评估输入；支持数据集版本号、collection 绑定关系、ground truth 来源和样本标签管理。
- 测试方法：`pytest tests/unit/core/evaluation/test_dataset_loader.py`

##### H3 实现检索指标与自定义指标评估器

- 目标：先补齐不依赖外部评估框架的确定性指标，作为回归门槛和质量闸门的基础。
- 修改文件：`src/ragms/core/evaluation/metrics/retrieval_metrics.py`、`src/ragms/core/evaluation/metrics/answer_metrics.py`、`src/ragms/libs/providers/evaluators/custom_metrics_evaluator.py`
- 实现类/函数：`compute_hit_rate_at_k()`、`compute_mrr()`、`compute_ndcg_at_k()`、`compute_citation_coverage()`、`CustomMetricsEvaluator.evaluate()`
- 验收标准：至少可稳定输出 `hit_rate@k`、`MRR`、`NDCG@K`、citation 覆盖率和必要的答案结构指标；指标字典与 `BaseEvaluator` 契约一致，可被 CompositeEvaluator 和报告服务直接消费。
- 测试方法：`pytest tests/unit/core/evaluation/test_retrieval_metrics.py tests/unit/core/evaluation/test_answer_metrics.py tests/unit/libs/test_custom_metrics_evaluator.py`

##### H4 实现 `RagasEvaluator` 与评估后端适配层

- 目标：对接 ragas，并补齐评估后端适配层，保证未来接入 DeepEval 或其他后端时不需要重写主流程。
- 修改文件：`src/ragms/libs/providers/evaluators/ragas_evaluator.py`、`src/ragms/libs/providers/evaluators/deepeval_evaluator.py`、`src/ragms/libs/abstractions/base_evaluator.py`
- 实现类/函数：`RagasEvaluator.evaluate()`、`DeepEvalEvaluator.evaluate()`、`normalize_backend_metrics()`
- 验收标准：ragas 可对标准样本输出 `context_precision`、`answer_relevancy`、`faithfulness` 等标准化指标；可选后端通过同一接口被装配；后端缺失依赖或不可用时能明确降级而不破坏评估主链路。
- 测试方法：`pytest tests/unit/libs/test_ragas_evaluator.py tests/unit/libs/test_deepeval_evaluator.py`

##### H5 实现 `CompositeEvaluator`、工厂装配与配置切换

- 目标：将 `custom_metrics`、`ragas`、`deepeval` 等后端纳入统一组合执行框架，并允许通过配置切换实验组合。
- 修改文件：`src/ragms/core/evaluation/runner.py`、`src/ragms/libs/factories/evaluator_factory.py`、`src/ragms/runtime/settings_models.py`、`settings.yaml`
- 实现类/函数：`CompositeEvaluator.evaluate()`、`build_evaluator_stack()`、`resolve_evaluation_backends()`
- 验收标准：可按配置组合多个 evaluator 并输出统一结果；不同后端的失败、跳过和部分成功会被结构化收敛；运行时配置快照可随评估结果一起持久化。
- 测试方法：`pytest tests/unit/core/evaluation/test_composite_evaluator.py tests/unit/runtime/test_settings_models.py`

##### H6 实现 `EvalRunner`、Evaluation Trace、评估报告落盘与实验对比元数据

- 目标：完成评估主编排、Evaluation Trace 打点、报告持久化、运行标识与实验对比元数据落盘，使每次评估都可被长期追踪、可观测和可回放。
- 修改文件：`src/ragms/core/evaluation/runner.py`、`src/ragms/core/evaluation/report_service.py`、`src/ragms/core/trace_collector/trace_manager.py`、`src/ragms/observability/metrics/evaluation_metrics.py`、`src/ragms/storage/sqlite/repositories/evaluations.py`、`src/ragms/storage/sqlite/schema.py`
- 实现类/函数：`EvalRunner.run()`、`record_evaluation_trace()`、`aggregate_evaluation_metrics()`、`ReportService.write_report()`、`ReportService.compare_runs()`、`EvaluationRepository.save_run()`
- 验收标准：每次评估运行都具备稳定 `run_id` 和 `trace_id`，并落盘 `collection`、`dataset_version`、`backend_set`、`config_snapshot`、指标摘要、样本级结果和失败样本明细；`TraceService` 可读取 `evaluation` 类型 trace；报告既可供 Dashboard 读取，也可供 MCP Tool 返回结果摘要。
- 测试方法：`pytest tests/integration/test_evaluation_runner.py tests/integration/test_evaluation_trace_logging.py`

##### H7 打通 Dashboard 真实评估结果、趋势展示与 baseline 管理

- 目标：将阶段 G 的评估面板从“页面壳 + 只读占位”升级为“真实报告消费端”，并建立 baseline 对比能力。
- 修改文件：`src/ragms/observability/dashboard/pages/evaluation_panel.py`、`src/ragms/core/evaluation/report_service.py`、`tests/e2e/test_evaluation_visible_in_dashboard.py`
- 实现类/函数：`load_latest_evaluation()`、`load_baseline_comparison()`、`ReportService.resolve_latest_run()`
- 验收标准：评估面板可读取 H6 真实落盘的评估结果，展示趋势、provider 对比、case-by-case 失败样本、baseline 差异和配置快照；无报告时显示明确空态；存在多次运行时支持按 `run_id` 做对比。
- 测试方法：`pytest tests/e2e/test_evaluation_visible_in_dashboard.py`

##### H8 接入 `evaluate_collection` MCP Tool

- 目标：在数据集、评估器、Runner 和报告服务稳定后，对外暴露可远程触发的评估工具，并保证协议契约清晰稳定。
- 修改文件：`src/ragms/mcp_server/tools/evaluation.py`、`src/ragms/mcp_server/schemas.py`、`src/ragms/mcp_server/tool_registry.py`、`src/ragms/core/evaluation/runner.py`、`src/ragms/core/evaluation/report_service.py`、`tests/unit/mcp_server/tools/test_evaluation_tool.py`、`tests/integration/test_mcp_server_evaluation.py`
- 实现类/函数：`handle_evaluate_collection()`、`normalize_evaluation_request()`、`serialize_evaluation_result()`
- 验收标准：工具可接收 `collection`、`dataset`、`metrics`、`eval_options` 等参数；可触发真实评估并返回 `run_id`、指标摘要、样本统计、结果路径与失败信息；评估失败不会拖垮 MCP Server 主进程；输出结构适合 Agent 做 A/B 比较与回归分析。
- 测试方法：`pytest tests/unit/mcp_server/tools/test_evaluation_tool.py tests/integration/test_mcp_server_evaluation.py`

##### H9 基于评估结果完成质量门槛达标与前序阶段收口

- 目标：以 H1-H8 产出的评估报告为依据，对阶段 A-G 中仍未达标或未收口的部分做最后一轮补齐和调优，直到系统满足第 4.3 节定义的质量门槛后，才允许结束阶段 H。
- 修改文件：`src/ragms/core/query_engine/*`、`src/ragms/ingestion_pipeline/*`、`src/ragms/mcp_server/*`、`src/ragms/observability/dashboard/*`、`settings.yaml`、`DEV_SPEC.md`
- 实现类/函数：`ReportService.assert_quality_gate()`、`collect_open_stage_gaps()`、`apply_regression_fixes()`
- 验收标准：默认配置在 golden test set 上至少满足 `Hit Rate@K >= 90%`、`MRR >= 0.8`、`NDCG@K >= 0.85`、`Faithfulness >= 0.9`、`Answer Relevancy >= 0.85`；若任一指标不达标，则必须在本任务内回补对应的摄取、检索、重排、引用、Trace、Dashboard 或 MCP 问题，直至指标达标；阶段 H 完成时，A-G 不再存在阻塞 I 阶段的未完成项。
- 测试方法：`pytest tests/integration/test_query_engine.py tests/integration/test_evaluation_runner.py tests/integration/test_mcp_server_evaluation.py tests/e2e/test_evaluation_visible_in_dashboard.py`

阶段 H 完成判定：

- 可以基于指定 `collection` 和 `dataset_version` 执行真实评估，并生成稳定 `run_id` 的结构化报告。
- `custom_metrics`、`ragas` 以及配置中启用的其他 evaluator 可通过统一接口运行并输出标准化指标。
- Dashboard 评估面板可读取真实评估结果、展示趋势与 baseline 对比，并能回看配置快照与坏 case 明细。
- `TraceService` 可读取 `evaluation` 类型 trace，系统总览与评估面板可消费评估指标聚合结果。
- `evaluate_collection` MCP Tool 可稳定触发评估并返回适合 Agent 消费的摘要结果。
- 第 4.3 节定义的关键质量门槛全部满足；任何由评估暴露出的 A-G 遗留问题都已在 H 阶段内补齐。
- `pytest tests/unit/core/evaluation tests/unit/libs/test_ragas_evaluator.py tests/unit/libs/test_deepeval_evaluator.py tests/unit/libs/test_custom_metrics_evaluator.py tests/integration/test_evaluation_runner.py tests/integration/test_evaluation_trace_logging.py tests/integration/test_mcp_server_evaluation.py tests/e2e/test_evaluation_visible_in_dashboard.py` 全部通过。

#### 阶段 I：端到端验收与文档收口

目标：在阶段 H 已经完成质量闭环与前序阶段收口的前提下，完成最终端到端回归、交付文档补齐、一键验收编排和版本冻结，使项目达到“可安装、可运行、可验收、可交付”的既定目标。阶段 I 不再新增核心业务能力，而是要把已经实现的摄取、检索、MCP、Trace、Dashboard、Evaluation 串成正式交付路径；阶段 I 完成后，新成员应可按照 README 和文档在本地完成安装、摄取、查询、评估、Dashboard 查看和一键验收，项目也具备可发布的版本与交付说明。

| 任务编号 | 任务名称 | 状态 | 完成日期 | 备注 |
|---------|---------|------|---------|------|
| I1 | 实现 MCP Client 全工具最终 E2E 回归 | [ ] |  |  |
| I2 | 实现 Dashboard 最终回归 E2E | [ ] |  |  |
| I3 | 完成一键全链路验收脚本与摘要输出 | [ ] |  |  |
| I4 | 完善 README、API、Dashboard 与架构文档 | [ ] |  |  |
| I5 | 完成最终验收、交付清单与版本收口 | [ ] |  |  |

##### I1 实现 MCP Client 全工具最终 E2E 回归

- 目标：从最终用户视角模拟真实 MCP Client，对阶段 E/F/H 已交付的全部关键工具做正式端到端回归，而不再只覆盖部分调用。
- 修改文件：`tests/e2e/test_mcp_client_simulation.py`
- 实现类/函数：`simulate_mcp_client_roundtrip()`
- 验收标准：可通过模拟客户端完成 `query_knowledge_hub`、`ingest_documents`、`list_collections`、`get_document_summary`、`get_trace_detail`、`evaluate_collection` 全链路调用；响应结构、错误语义、引用透明和关键标识（如 `trace_id`、`run_id`）均符合规范。
- 测试方法：`pytest tests/e2e/test_mcp_client_simulation.py`

##### I2 实现 Dashboard 最终回归 E2E

- 目标：对最终 Dashboard 做交付前回归，验证页面加载、关键跳转、真实数据展示和核心交互在最终配置下稳定可用。
- 修改文件：`tests/e2e/test_dashboard_smoke.py`、`tests/e2e/test_evaluation_visible_in_dashboard.py`
- 实现类/函数：`dashboard_smoke_check()`、`assert_dashboard_regression_flow()`
- 验收标准：系统总览、数据浏览、Ingestion 管理、Ingestion 追踪、Query 追踪、评估面板六页均可正常打开；关键导航、真实评估结果展示、文档/Trace/评估报告跳转和必要的空态渲染均通过最终回归。
- 测试方法：`pytest tests/e2e/test_dashboard_smoke.py tests/e2e/test_evaluation_visible_in_dashboard.py`

##### I3 完成一键全链路验收脚本与摘要输出

- 目标：形成正式的“安装后可直接执行”的全链路验收入口，把摄取、查询、MCP、Trace、Dashboard、Evaluation 串成一条可重复运行的验收脚本，并显式映射第 4.2.3 节定义的三个核心 E2E 场景。
- 修改文件：`scripts/run_acceptance.py`、`tests/e2e/test_full_chain_acceptance.py`
- 实现类/函数：`run_full_acceptance()`、`render_acceptance_summary()`
- 验收标准：一条命令可执行“摄取 -> 查询 -> MCP 调用 -> Trace 校验 -> Dashboard 校验 -> Evaluation 校验”的完整路径，并输出结构化摘要、失败项和建议排查入口；验收摘要必须显式覆盖第 4.2.3 节的场景 1（数据准备）、场景 2（召回/质量评估）、场景 3（MCP Client 功能测试）。
- 测试方法：`pytest tests/e2e/test_full_chain_acceptance.py`

##### I4 完善 README、API、Dashboard 与架构文档

- 目标：把项目使用说明从“开发者知道怎么跑”提升到“新成员按文档即可完成完整流程”。
- 修改文件：`README.md`、`DEV_SPEC.md`、`docs/architecture/*`、`docs/api/*`、`docs/dashboards/*`
- 实现类/函数：文档补充为主，无新增核心函数
- 验收标准：文档完整覆盖安装、配置、摄取、查询、MCP tools、Trace 查看、Dashboard 使用、评估执行、一键验收和常见问题排查；第 5 章目录树与实际交付文件一致，文档中的命令、路径和页面名称与实现一致。
- 测试方法：人工走查 + README / 文档流程冒烟验证

##### I5 完成最终验收、交付清单与版本收口

- 目标：在所有功能、测试和文档完成后，形成正式交付物并冻结发布版本。
- 修改文件：`README.md`、`DEV_SPEC.md`、`pyproject.toml`、`scripts/run_acceptance.py`
- 实现类/函数：`render_acceptance_summary()`、`build_release_checklist()`
- 验收标准：所有核心测试通过；一键验收脚本输出通过摘要；交付清单包含版本号、关键命令、支持能力、限制说明和验收结果；文档、目录结构、测试与实现保持一致；覆盖率报告与第 4.4 节保持一致，其中单元测试核心逻辑覆盖率应达到 `>= 80%`、关键集成路径覆盖率达到 `100%`、第 4.2.3 节定义的三个核心 E2E 场景覆盖率达到 `100%`，项目方可进入发布或归档阶段。
- 测试方法：`pytest --cov=src/ragms --cov-report=term-missing tests/unit tests/integration tests/e2e`

阶段 I 完成判定：

- 新成员仅依赖 `README.md` 与 `docs/` 下文档，即可在本地完成安装、摄取、查询、MCP 调用、Trace 查看、Dashboard 使用、评估执行和一键验收。
- `tests/e2e/test_mcp_client_simulation.py`、`tests/e2e/test_dashboard_smoke.py`、`tests/e2e/test_evaluation_visible_in_dashboard.py`、`tests/e2e/test_full_chain_acceptance.py` 全部通过。
- `scripts/run_acceptance.py` 可输出结构化验收摘要，作为最终交付检查入口。
- 第 4.2.3 节定义的三个核心 E2E 场景在验收摘要中都有明确映射；第 4.4 节覆盖率目标已形成可检查报告。
- `README.md`、`DEV_SPEC.md`、`docs/architecture/*`、`docs/api/*`、`docs/dashboards/*` 与第 5 章文件结构保持一致。
- 版本号、交付清单和验收结论全部收口，项目达到可交付状态。

### 6.4 执行建议

- 建议严格按 A → I 顺序推进，避免在工程骨架和抽象层未稳定前提前进入 Dashboard 或评估实现。
- 阶段 C、D、F、G、H 是核心交付阶段，任何一个阶段开始前都应确保前一阶段的测试基线稳定。
- 若需要压缩周期，优先并行的是阶段 G 与阶段 H 的页面开发，但前提是阶段 F 的 Trace 基础设施已经稳定。

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
