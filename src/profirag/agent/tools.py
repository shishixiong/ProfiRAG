"""RAG Agent tools for ReAct Agent"""

import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from llama_index.core.tools import FunctionTool
from llama_index.core.schema import NodeWithScore


# 表格索引链接正则模式
# 匹配格式: 表 X-X 标题 → [查看表格](tables/xxx.md)
TABLE_INDEX_PATTERN = re.compile(
    r'表\s*(\d+[-\.\d]*)\s*(.*?)\s*→\s*\[查看表格\]\((tables/[^)]+\.md)\)'
)


class RAGTools:
    """RAG系统工具集 - 为ReAct Agent提供检索和生成工具"""

    def __init__(
        self,
        retriever: Any,
        synthesizer: Any,
        llm: Any,
        pre_retrieval: Any = None,
        markdown_base_path: Optional[str] = None,
    ):
        """初始化工具集

        Args:
            retriever: HybridRetriever实例
            synthesizer: ResponseSynthesizer实例
            llm: LLM实例
            pre_retrieval: PreRetrievalPipeline实例（可选）
            markdown_base_path: Markdown文件目录路径（用于表格索引解析）
        """
        self.retriever = retriever
        self.synthesizer = synthesizer
        self.llm = llm
        self.pre_retrieval = pre_retrieval
        self.markdown_base_path = Path(markdown_base_path) if markdown_base_path else None

        # 存储最近检索的结果（供answer工具使用）
        self._last_retrieved_nodes: List[NodeWithScore] = []

    def create_retrieval_tool(self) -> FunctionTool:
        """创建向量检索工具"""
        def vector_search(query: str, top_k: int = 5) -> str:
            """使用向量相似度搜索文档库

            Args:
                query: 搜索查询字符串
                top_k: 返回结果数量

            Returns:
                格式化的检索结果
            """
            nodes = self.retriever.retrieve(query, top_k=top_k, retrieve_mode='vector')
            self._last_retrieved_nodes = nodes  # 保存供后续使用
            return self._format_nodes(nodes)

        return FunctionTool.from_defaults(
            fn=vector_search,
            name="vector_search",
            description="使用向量相似度搜索文档库，返回相关文档片段。适合语义搜索。"
        )

    def create_bm25_tool(self) -> FunctionTool:
        """创建BM25关键词搜索工具"""
        # 从环境变量判断是否启用 BM25
        index_mode = os.environ.get("PROFIRAG_INDEX_MODE", "hybrid")

        def keyword_search(query: str, top_k: int = 5) -> str:
            """使用BM25关键词搜索

            Args:
                query: 搜索查询字符串
                top_k: 返回结果数量

            Returns:
                格式化的检索结果
            """
            # 如果不是 vector 模式，就可以使用 BM25 (sparse 检索)
            if index_mode != "vector":
                nodes = self.retriever.retrieve(query, top_k=top_k, retrieve_mode='sparse')
                self._last_retrieved_nodes = nodes
                return self._format_nodes(nodes)
            return "BM25索引未启用（当前为vector模式），请使用vector_search工具"

        return FunctionTool.from_defaults(
            fn=keyword_search,
            name="keyword_search",
            description="使用BM25关键词搜索，适合精确匹配和关键词查询。"
        )

    def create_multi_query_tool(self) -> FunctionTool:
        """创建多Query检索工具"""
        def multi_query_search(query: str) -> str:
            """生成多个查询变体并检索

            Args:
                query: 原始查询字符串

            Returns:
                合并后的检索结果
            """
            # 生成查询变体
            variants = self._generate_variants(query)
            all_nodes = []

            # 对每个变体进行检索
            for v in variants:
                nodes = self.retriever.retrieve(v, top_k=3)
                all_nodes.extend(nodes)

            # 去重并格式化
            unique_nodes = self._deduplicate(all_nodes)
            self._last_retrieved_nodes = unique_nodes[:10]
            return self._format_nodes(unique_nodes[:10])

        return FunctionTool.from_defaults(
            fn=multi_query_search,
            name="multi_query_search",
            description="生成多个查询变体并检索，扩大检索范围，适合复杂问题。"
        )

    def create_hyde_tool(self) -> FunctionTool:
        """创建HyDE检索工具"""
        def hyde_search(query: str) -> str:
            """生成假设文档并检索

            Args:
                query: 查询字符串

            Returns:
                HyDE检索结果
            """
            # 生成假设文档
            hypothetical_doc = self._generate_hypothetical(query)
            # 使用假设文档检索
            nodes = self.retriever.retrieve(hypothetical_doc, top_k=5)
            self._last_retrieved_nodes = nodes
            return self._format_nodes(nodes)

        return FunctionTool.from_defaults(
            fn=hyde_search,
            name="hyde_search",
            description="生成假设文档进行检索，适合问题表述不清晰的情况。"
        )

    def create_final_answer_tool(self) -> FunctionTool:
        """创建最终回答生成工具"""
        def generate_answer(question: str, mode: str = "default", top_k: int = 5) -> str:
            """基于检索到的上下文生成最终回答

            Args:
                question: 用户的问题
                mode: 回答模式，可选值：
                    - "simple": 简洁回答，直接回复不超过100字，无引用标注
                    - "default": 默认回答，有引用标注和基本结构（推荐）
                    - "professional": 专业回答，详细结构化，完整代码示例，参数说明
                    - "technical": 技术规范回答，严格按照文档，包含版本差异说明
                top_k: 使用前N个检索结果（默认5）

            Returns:
                生成的回答
            """
            if not self._last_retrieved_nodes:
                return "错误：请先使用检索工具（vector_search/keyword_search）获取相关文档"

            # 获取指定数量的检索结果
            nodes = self._last_retrieved_nodes[:top_k]

            # 根据模式选择 prompt template
            from ..generation.prompts import PromptTemplates
            template = PromptTemplates.get_template_by_mode(mode)

            # 使用自定义 prompt 生成回答
            response = self.synthesizer.synthesize_custom(
                question,
                nodes,
                custom_prompt=template
            )

            # 添加来源信息（仅在非 simple 模式）
            if mode != "simple":
                sources_summary = self._format_sources_summary(nodes)
                return f"{response}\n\n---\n**参考来源**: {sources_summary}"

            return response

        return FunctionTool.from_defaults(
            fn=generate_answer,
            name="generate_answer",
            description="""基于检索到的文档生成最终回答。

回答模式说明：
- simple: 简洁回答，适合快速查询，直接回复无引用
- default: 标准回答，有引用标注和基本结构（推荐大多数场景）
- professional: 专业回答，详细结构化，适合技术文档深度问答
- technical: 技术规范回答，严格按文档表述，包含版本差异和适用范围

必须先使用检索工具获取文档。"""
        )

    def create_retrieve_with_context_tool(self) -> FunctionTool:
        """创建带上下文信息的检索工具"""
        def retrieve_for_answer(question: str, mode: str = "default", top_k: int = 5) -> str:
            """检索文档并直接生成回答（一步完成）

            Args:
                question: 用户问题
                mode: 回答模式，可选值：
                    - "simple": 简洁回答，直接回复不超过100字，无引用标注
                    - "default": 默认回答，有引用标注和基本结构（推荐）
                    - "professional": 专业回答，详细结构化，完整代码示例，参数说明
                    - "technical": 技术规范回答，严格按照文档，包含版本差异说明
                top_k: 检索数量

            Returns:
                检索结果和生成的回答
            """
            # 检索
            nodes = self.retriever.retrieve(question, top_k=top_k)
            self._last_retrieved_nodes = nodes

            # 根据模式选择 prompt template
            from ..generation.prompts import PromptTemplates
            template = PromptTemplates.get_template_by_mode(mode)

            # 使用自定义 prompt 生成回答
            response = self.synthesizer.synthesize_custom(
                question,
                nodes[:top_k],
                custom_prompt=template
            )

            # 添加来源信息（仅在非 simple 模式）
            if mode != "simple" and nodes:
                sources_summary = self._format_sources_summary(nodes[:top_k])
                return f"{response}\n\n---\n**参考来源**: {sources_summary}"

            return response

        return FunctionTool.from_defaults(
            fn=retrieve_for_answer,
            name="retrieve_and_answer",
            description="""一步完成检索和回答生成，适合简单问题。

回答模式说明：
- simple: 简洁回答，适合快速查询，直接回复无引用
- default: 标准回答，有引用标注和基本结构（推荐大多数场景）
- professional: 专业回答，详细结构化，适合技术文档深度问答
- technical: 技术规范回答，严格按文档表述，包含版本差异和适用范围

适用场景：问题明确、单次检索即可获取足够信息。"""
        )

    def create_table_lookup_tool(self) -> FunctionTool:
        """创建表格内容查询工具"""
        def table_lookup(table_reference: str) -> str:
            """读取表格索引链接对应的表格内容

            Args:
                table_reference: 表格索引链接，支持以下格式：
                    - "tables/xxx.md" - 直接的表格文件路径
                    - "表 X-X 标题 → [查看表格](tables/xxx.md)" - 完整的索引链接

            Returns:
                表格的完整 Markdown 内容，如果找不到则返回错误信息
            """
            if not self.markdown_base_path:
                return "错误：未配置 markdown_base_path，无法解析表格索引"

            # 从完整链接中提取路径
            match = TABLE_INDEX_PATTERN.search(table_reference)
            if match:
                table_path = match.group(3)  # tables/xxx.md
                table_num = match.group(1)   # X-X
                table_title = match.group(2) # 标题
            else:
                # 直接路径格式
                table_path = table_reference
                table_num = ""
                table_title = ""

            # 构建完整路径并读取文件
            full_path = self.markdown_base_path / table_path
            if not full_path.exists():
                return f"错误：表格文件不存在: {table_path}"

            try:
                content = full_path.read_text(encoding="utf-8")
                # 返回表格内容，带标题信息
                header = f"表格 {table_num}: {table_title}" if table_num else "表格内容"
                return f"{header}\n\n{content}"
            except Exception as e:
                return f"错误：读取表格文件失败: {str(e)}"

        return FunctionTool.from_defaults(
            fn=table_lookup,
            name="table_lookup",
            description="读取表格索引链接指向的表格详细内容。当检索结果中包含表格索引（如 '表 X-X → [查看表格](tables/xxx.md)'）时，使用此工具获取完整表格数据。"
        )

    def create_all_tools(self) -> List[FunctionTool]:
        """创建所有工具列表"""
        tools = [
            self.create_retrieval_tool(),
            self.create_bm25_tool(),
            self.create_multi_query_tool(),
            self.create_hyde_tool(),
            self.create_final_answer_tool(),
            self.create_retrieve_with_context_tool(),
        ]
        # 添加表格查询工具（仅当配置了 markdown_base_path）
        if self.markdown_base_path:
            tools.append(self.create_table_lookup_tool())
        return tools

    def _format_nodes(self, nodes: List[NodeWithScore]) -> str:
        """格式化检索结果

        Args:
            nodes: 检索结果列表

        Returns:
            格式化的字符串
        """
        if not nodes:
            return "未找到相关文档"

        formatted = []
        for i, n in enumerate(nodes):
            score = n.score if hasattr(n, 'score') else 0
            text = n.node.text if hasattr(n, 'node') else str(n)
            metadata = n.node.metadata if hasattr(n, 'node') else {}

            # 截断长文本
            text_preview = text[:300] + "..." if len(text) > 300 else text
            source = metadata.get('source_file', metadata.get('source_path', '未知来源'))

            formatted.append(f"[文档{i+1}] 相关度: {score:.3f}")
            formatted.append(f"来源: {source}")
            formatted.append(f"内容: {text_preview}")

            # 检测并标注表格索引链接
            if self.markdown_base_path and TABLE_INDEX_PATTERN.search(text):
                formatted.append("⚠️ [包含表格索引，可使用 table_lookup 工具查看详情]")

            formatted.append("")  # 空行分隔

        return "\n".join(formatted)

    def _generate_variants(self, query: str) -> List[str]:
        """生成查询变体

        Args:
            query: 原始查询

        Returns:
            查询变体列表
        """
        prompt = f"""请生成3个查询变体，用于在技术文档库中检索相关信息。

要求：
1. 保持原问题的核心意图不变
2. 使用不同的措辞和关键词
3. 可以从不同角度表述（如：功能描述、使用场景、技术参数等）
4. 变体应适合关键词搜索，使用文档中可能出现的术语

原问题: {query}

输出格式（每行一个变体，不需要编号）:
变体1
变体2
变体3"""

        try:
            response = self.llm.complete(prompt)
            variants = []
            for line in response.text.strip().split("\n"):
                line = line.strip()
                # 跳过编号前缀和说明性文字
                if line and not line.startswith(("原问题", "变体", "Query", "要求", "输出")):
                    # 移除数字前缀如 "1. " 或 "1: " 或 "变体1: "
                    if len(line) > 2 and line[0].isdigit() and line[1] in ".: ":
                        line = line[2:].strip()
                    elif line.startswith("变体") and ":" in line:
                        line = line.split(":")[1].strip()
                    if line:
                        variants.append(line)

            # 确保至少有原查询
            if not variants:
                variants = [query]

            return variants[:3]
        except Exception as e:
            # 如果LLM调用失败，返回原查询
            return [query]

    def _generate_hypothetical(self, query: str) -> str:
        """生成假设文档

        Args:
            query: 查询字符串

        Returns:
            假设文档内容
        """
        prompt = f"""请撰写一段假设性的技术文档内容，这段内容应该能够完整回答以下问题。

要求：
1. 内容风格应类似技术手册（如产品文档、API说明）
2. 包含问题中涉及的关键概念、参数或操作步骤
3. 使用专业术语和规范表述
4. 如果涉及代码或命令，给出具体示例
5. 内容长度约200-500字

问题: {query}

假设文档内容:"""

        try:
            response = self.llm.complete(prompt)
            return response.text.strip()
        except Exception:
            return query  # 失败时返回原查询

    def _deduplicate(self, nodes: List[NodeWithScore]) -> List[NodeWithScore]:
        """去重节点

        Args:
            nodes: 节点列表

        Returns:
            去重后的列表
        """
        seen_ids = set()
        unique = []
        for n in nodes:
            node_id = n.node.node_id if hasattr(n, 'node') else str(n)
            if node_id not in seen_ids:
                seen_ids.add(node_id)
                unique.append(n)
        return unique

    def _format_sources_summary(self, nodes: List[NodeWithScore]) -> str:
        """格式化来源摘要（用于回答末尾）

        Args:
            nodes: 检索结果列表

        Returns:
            简洁的来源摘要字符串
        """
        sources = []
        for i, n in enumerate(nodes[:5]):
            metadata = n.node.metadata if hasattr(n, 'node') else {}
            source_file = metadata.get('source_file', metadata.get('source_path', '未知'))
            # 简化来源名称
            if '/' in source_file:
                source_file = source_file.split('/')[-1]
            sources.append(f"[{i+1}] {source_file}")

        return ", ".join(sources)


class ToolResultFormatter:
    """工具结果格式化器"""

    @staticmethod
    def format_for_display(result: str) -> str:
        """格式化工具结果用于显示

        Args:
            result: 工具返回的原始结果

        Returns:
            格式化后的结果
        """
        # 添加分隔线和标题
        lines = result.split("\n")
        formatted = ["─" * 40]
        formatted.extend(lines)
        formatted.append("─" * 40)
        return "\n".join(formatted)

    @staticmethod
    def extract_sources(result: str) -> List[Dict[str, str]]:
        """从结果中提取来源信息

        Args:
            result: 工具返回的结果

        Returns:
            来源信息列表
        """
        sources = []
        lines = result.split("\n")

        current_source = {}
        for line in lines:
            if line.startswith("[文档"):
                if current_source:
                    sources.append(current_source)
                current_source = {"index": line}
            elif line.startswith("来源:"):
                current_source["source"] = line.split(":", 1)[1].strip()
            elif line.startswith("内容:"):
                current_source["content"] = line.split(":", 1)[1].strip()

        if current_source:
            sources.append(current_source)

        return sources