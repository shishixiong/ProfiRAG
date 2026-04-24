# ProfiRAG 设计文档

## 架构概览

ProfiRAG 是一个模块化的 RAG 框架，在共享检索基础设施上构建了三种执行模式。

```
┌─────────────────────────────────────────────────────────┐
│                        入口点                             │
│                   main.py (交互式 CLI)                    │
├─────────────────────────────────────────────────────────┤
│                   RAGPipeline (编排层)                    │
│  ┌──────────────┬──────────────┬──────────────────────┐ │
│  │ Pipeline 模式 │ ReAct Agent  │    PlanAgent         │ │
│  │  (固定流程)   │ (思考→行动)  │ (规划→执行→总结)     │ │
│  └──────────────┴──────────────┴──────────────────────┘ │
├─────────────────────────────────────────────────────────┤
│                     共享基础设施                          │
│  ┌──────────┬──────────┬──────────┬──────────────────┐  │
│  │ 检索器    │ 重排序器  │ 生成器    │ 查询转换器       │  │
│  │ (混合)   │ (3种类型) │ (4种模式) │(HyDE+改写+多查)  │  │
│  └──────────┴──────────┴──────────┴──────────────────┘  │
├─────────────────────────────────────────────────────────┤
│                      存储层                               │
│  ┌──────────┬──────────┬──────────────────────────────┐ │
│  │  Qdrant  │PostgreSQL│         本地存储              │ │
│  └──────────┴──────────┴──────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

## 模块设计

### 1. 配置 (`config/settings.py`)

基于 Pydantic 的分层配置系统：
- `EnvSettings` — 从 `.env` 读取原始环境变量
- `RAGConfig` — 结构化配置（存储、LLM、嵌入、检索、重排序、Agent）
- `RAGConfig.from_env()` 工厂方法构建完整配置树

**设计决策**：将扁平的环境变量与结构化配置分离，以支持 `.env` 和 YAML 两种配置方式。

### 2. 存储 (`storage/`)

通过 `StorageRegistry` 实现抽象工厂模式：
```python
StorageRegistry.get_store(type, config) → BaseVectorStore
```

三种后端：
- **Qdrant** — 原生 BM25 + 向量，生产级
- **PostgreSQL/pgvector** — SQL 原生，适合混合工作负载
- **Local** — 基于文件，零依赖，用于测试

**设计决策**：避免供应商锁定。存储后端是实现了 `BaseVectorStore` 的插件。

### 3. 摄入 (`ingestion/`)

- `DocumentLoader` — 读取 PDF、Markdown 等
- `TextSplitter` — 句子、Token、语义分割器
- `ChineseTextSplitter` — 基于 jieba 的中文分割
- `ASTSplitter` — 代码感知分割（Python、Java、C++、Go）
- `MarkdownSplitter` — 标题感知的 Markdown 分割
- `ImageProcessor` — 从 PDF 中提取和描述图片

### 4. 检索 (`retrieval/`)

#### HybridRetriever
结合稠密（向量）和稀疏（BM25）检索。使用 Qdrant 时，委托给向量存储的原生混合搜索。使用 RRF（倒数排名融合）进行分数合并。

**设计决策**：与其自己实现 BM25，我们利用 Qdrant 的原生 BM25 支持。`retrieve_mode` 参数允许在查询时切换 hybrid/sparse/vector 模式。

#### QueryTransform 流水线
```python
PreRetrievalPipeline → List[QueryBundle]
  ├── HyDEQueryTransform (可选)
  ├── QueryRewriter (可选)
  └── MultiQueryGenerator (可选)
```

#### Reranker
三种具有一致接口的提供商：
- `CrossEncoderReranker` — 本地交叉编码器模型
- `CohereReranker` — Cohere API 格式
- `DashScopeReranker` — 阿里云灵积

### 5. 生成 (`generation/`)

#### ResponseSynthesizer
封装 LlamaIndex 合成器，支持自定义提示词：
- `synthesize()` — 标准生成
- `synthesize_streaming()` — 流式生成
- `synthesize_custom()` — 自定义提示词模板

#### PromptTemplates
四种中文响应模式：
- `simple` — 简洁，<100字，无引用
- `default` — 标准，带引用
- `professional` — 详细、结构化、含代码示例
- `technical` — 严格遵循文档，含版本信息

### 6. Agent (`agent/`)

#### 工具架构 (`tools.py`)

```
RAGTools
├── 检索工具 (产生 _last_retrieved_nodes)
│   ├── vector_search(query, top_k)
│   ├── keyword_search(query, top_k)
│   ├── multi_query_search(query)
│   └── hyde_search(query)
├── 优化工具 (消费 + 修改 _last_retrieved_nodes)
│   ├── rewrite_query(query)
│   ├── rerank_results(query, top_n)     ← 需要 reranker 配置
│   └── filter_results(source_file, min_score, max_score, top_k)
├── 生成工具 (消费 _last_retrieved_nodes)
│   ├── generate_answer(question, mode, top_k)
│   └── retrieve_and_answer(question, mode, top_k)
└── 特殊工具
    └── table_lookup(table_reference)    ← 需要 markdown_base_path
```

**核心设计模式**：`_last_retrieved_nodes` 是工具间共享的可变状态。检索工具写入它；优化工具读取+写入；生成工具消费它。这形成了自然的流水线：搜索 → 优化 → 回答。

**工具可用性**：
| 工具 | 始终可用 | 条件可用 |
|------|:-:|:-:|
| vector_search | x | |
| keyword_search | x | |
| multi_query_search | x | |
| hyde_search | x | |
| rewrite_query | x | (LLM 回退) |
| filter_results | x | |
| rerank_results | | 配置了 reranker |
| table_lookup | | 设置了 markdown_base_path |

#### ReActAgent (`react_agent.py`)
- 封装 LlamaIndex `ReActAgent`，使用自定义系统提示词
- 系统提示词指导工具选择（何时使用每种检索/优化工具）
- 通过 `AgentFactory` 实现工厂模式

#### PlanAgent (`plan_agent.py`)
四阶段执行：
1. **Plan（规划）**：LLM 生成结构化计划及复杂度分析
2. **Approve（审批）**：用户审核/批准计划（可通过 `auto_approve` 跳过）
3. **Execute（执行）**：顺序执行步骤，处理失败情况
4. **Finalize（总结）**：从收集的结果生成答案

失败处理：最多自动重规划 `max_replan_attempts` 次，重规划提示词会建议替代策略。

### 7. Pipeline (`pipeline/rag_pipeline.py`)

编排层，将所有组件连接在一起：
- 从配置创建所有组件
- 标准查询流程：transform → retrieve → rerank → synthesize
- Agent 模式通过 `query_with_agent()` 路由
- 延迟初始化 Agent（仅在首次使用时创建）

## 数据流

### 标准 Pipeline
```
Query → PreRetrievalPipeline.transform()
     → HybridRetriever.retrieve()  [每个变体]
     → deduplicate_nodes()
     → Reranker.rerank()
     → ResponseSynthesizer.synthesize_custom()
     → Answer
```

### ReAct Agent
```
Query → ReActAgent.run()
     → [循环] 思考 → 选择工具 → 执行 → 观察
     → 最终答案
```

### PlanAgent
```
Query → PlanGenerator.generate_plan()        [LLM 分析复杂度]
     → 用户审批 (如果复杂)
     → PlanExecutor.execute()
     → [失败时] PlanGenerator.replan()   [最多 N 次]
     → _finalize_answer()                     [从结果合成]
     → Answer
```

## 配置设计

### 为什么用 Pydantic？
- 类型安全的配置与验证
- 环境变量自动映射
- 嵌套配置模型（清晰的关注点分离）
- 可序列化，支持检查点/可复现性

### 关键环境变量

| 变量 | 默认值 | 用途 |
|------|--------|------|
| `PROFIRAG_STORAGE_TYPE` | qdrant | 存储后端选择 |
| `PROFIRAG_INDEX_MODE` | hybrid | hybrid（稠密+BM25）或仅向量 |
| `PROFIRAG_RETRIEVE_INDEX_MODE` | hybrid | 查询检索模式 |
| `PROFIRAG_RERANK_ENABLED` | true | 启用检索后重排序 |
| `PROFIRAG_AGENT_ENABLED` | false | 启用 Agent 模式（对比 Pipeline） |
| `PROFIRAG_AGENT_MODE` | react | react、plan 或 pipeline |
| `PROFIRAG_AGENT_MARKDOWN_BASE_PATH` | - | table_lookup 工具所需 |

## 扩展点

1. **新工具**：在 `RAGTools` 中添加 `create_*_tool()` 方法，加入 `create_all_tools()`，更新提示词
2. **新存储后端**：实现 `BaseVectorStore`，在 `StorageRegistry` 中注册
3. **新 Reranker 提供商**：实现 `BaseReranker`，添加到 `Reranker._create_impl()`
4. **新 Agent 模式**：实现新的 Agent 类，添加到 `AgentFactory` 和 `RAGPipeline.query_with_agent()`
5. **新响应模式**：在 `PromptTemplates.MODE_TEMPLATES_ZH` 中添加提示词模板