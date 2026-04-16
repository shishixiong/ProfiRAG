"""RAG Agent tools for ReAct Agent"""

from typing import List, Dict, Any, Optional
from llama_index.core.tools import FunctionTool
from llama_index.core.schema import NodeWithScore


class RAGTools:
    """RAG系统工具集 - 为ReAct Agent提供检索和生成工具"""

    def __init__(
        self,
        retriever: Any,
        synthesizer: Any,
        llm: Any,
        pre_retrieval: Any = None,
    ):
        """初始化工具集

        Args:
            retriever: HybridRetriever实例
            synthesizer: ResponseSynthesizer实例
            llm: LLM实例
            pre_retrieval: PreRetrievalPipeline实例（可选）
        """
        self.retriever = retriever
        self.synthesizer = synthesizer
        self.llm = llm
        self.pre_retrieval = pre_retrieval

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
            nodes = self.retriever.retrieve(query, top_k=top_k)
            self._last_retrieved_nodes = nodes  # 保存供后续使用
            return self._format_nodes(nodes)

        return FunctionTool.from_defaults(
            fn=vector_search,
            name="vector_search",
            description="使用向量相似度搜索文档库，返回相关文档片段。适合语义搜索。"
        )

    def create_bm25_tool(self) -> FunctionTool:
        """创建BM25关键词搜索工具"""
        def keyword_search(query: str, top_k: int = 5) -> str:
            """使用BM25关键词搜索

            Args:
                query: 搜索查询字符串
                top_k: 返回结果数量

            Returns:
                格式化的检索结果
            """
            if self.retriever.bm25_index:
                nodes = self.retriever.bm25_index.retrieve(query, top_k=top_k)
                self._last_retrieved_nodes = nodes
                return self._format_nodes(nodes)
            return "BM25索引未启用，请使用vector_search工具"

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
        def generate_answer(question: str) -> str:
            """基于检索到的上下文生成最终回答

            Args:
                question: 用户的问题

            Returns:
                生成的回答
            """
            if not self._last_retrieved_nodes:
                return "错误：请先使用检索工具（vector_search/keyword_search）获取相关文档"

            # 使用最近的检索结果生成回答
            response = self.synthesizer.synthesize(
                question,
                self._last_retrieved_nodes[:5]
            )
            return response

        return FunctionTool.from_defaults(
            fn=generate_answer,
            name="generate_answer",
            description="基于检索到的文档生成最终回答。必须先使用检索工具获取文档。"
        )

    def create_retrieve_with_context_tool(self) -> FunctionTool:
        """创建带上下文信息的检索工具"""
        def retrieve_for_answer(question: str, top_k: int = 5) -> str:
            """检索文档并直接生成回答

            Args:
                question: 用户问题
                top_k: 检索数量

            Returns:
                检索结果和生成的回答
            """
            # 检索
            nodes = self.retriever.retrieve(question, top_k=top_k)
            self._last_retrieved_nodes = nodes

            # 格式化检索结果
            context = self._format_nodes(nodes)

            # 生成回答
            response = self.synthesizer.synthesize(question, nodes[:top_k])

            return f"检索结果:\n{context}\n\n生成的回答:\n{response}"

        return FunctionTool.from_defaults(
            fn=retrieve_for_answer,
            name="retrieve_and_answer",
            description="一步完成检索和回答生成，适合简单问题。"
        )

    def create_all_tools(self) -> List[FunctionTool]:
        """创建所有工具列表"""
        return [
            self.create_retrieval_tool(),
            self.create_bm25_tool(),
            self.create_multi_query_tool(),
            self.create_hyde_tool(),
            self.create_final_answer_tool(),
            self.create_retrieve_with_context_tool(),
        ]

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
            formatted.append("")  # 空行分隔

        return "\n".join(formatted)

    def _generate_variants(self, query: str) -> List[str]:
        """生成查询变体

        Args:
            query: 原始查询

        Returns:
            查询变体列表
        """
        prompt = f"""请生成3个不同的查询变体，用于搜索文档库。每个变体应该表达相同意图但使用不同措辞。

原问题: {query}

请直接输出3个变体，每行一个:"""

        try:
            response = self.llm.complete(prompt)
            variants = []
            for line in response.text.strip().split("\n"):
                line = line.strip()
                # 跳过编号前缀
                if line and not line.startswith(("原问题", "变体", "Query")):
                    # 移除数字前缀如 "1. " 或 "1: "
                    if len(line) > 2 and line[0].isdigit() and line[1] in ".: ":
                        line = line[2:].strip()
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
        prompt = f"""请写一段假设性的文档内容，这段内容应该能够完美回答以下问题。

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