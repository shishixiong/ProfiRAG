"""ReAct Agent for RAG system"""

import asyncio
from typing import List, Dict, Any, Optional
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import BaseTool

from .tools import RAGTools


def run_async(coro):
    """Run async coroutine in sync context."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is None:
        return asyncio.run(coro)
    else:
        # Already in async context, create new thread
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, coro)
            return future.result()


class RAGReActAgent:
    """基于ReAct模式的RAG问答Agent

    ReAct循环: Think → Act → Observe → Think...

    Agent可以:
    - 动态决定是否需要检索
    - 选择合适的检索工具
    - 进行多轮检索
    - 根据结果质量决定是否继续
    """

    def __init__(
        self,
        tools: RAGTools,
        llm: Any,
        max_iterations: int = 10,
        verbose: bool = True,
        system_prompt: Optional[str] = None,
    ):
        """初始化ReAct Agent

        Args:
            tools: RAGTools工具集实例
            llm: LLM实例
            max_iterations: 最大迭代次数
            verbose: 是否显示详细日志
            system_prompt: 自定义系统提示词
        """
        self.tools = tools
        self.llm = llm
        self.max_iterations = max_iterations
        self.verbose = verbose

        # 创建工具列表
        self._tools_list = tools.create_all_tools()

        # 默认系统提示词
        self._system_prompt = system_prompt or self._default_system_prompt()

        # 创建ReAct Agent
        self._agent = ReActAgent(
            tools=self._tools_list,
            llm=llm,
            max_iterations=max_iterations,
            verbose=verbose,
        )

        # 记录执行历史
        self._execution_history: List[Dict[str, Any]] = []

    def _default_system_prompt(self) -> str:
        """默认系统提示词"""
        return """你是一个专业的RAG问答助手，帮助用户回答关于文档库的问题。

你的工作流程:
1. 分析用户问题，理解需求
2. 决定是否需要检索文档
3. 选择合适的检索工具:
   - vector_search: 语义相似度搜索，适合概念性问题
   - keyword_search: BM25关键词搜索，适合精确匹配
   - multi_query_search: 多查询变体搜索，适合复杂问题
   - hyde_search: 假设文档搜索，适合表述不清的问题
   - retrieve_and_answer: 一步完成检索和回答
4. 观察检索结果，评估是否足够
5. 如需更多信息，继续检索
6. 使用generate_answer生成最终回答

表格处理:
- 检索结果中如果包含表格索引（如 "表 X-X → [查看表格](tables/xxx.md)"），
  使用 table_lookup 工具获取完整表格数据
- table_lookup 的参数可以是完整索引链接或表格文件路径

注意事项:
- 简单问题可以直接用retrieve_and_answer
- 复杂问题可能需要多轮检索
- 检索结果不足时可以换用不同工具
- 回答要基于检索到的文档和表格内容，不要编造信息"""

    def query(self, question: str) -> Dict[str, Any]:
        """执行Agent问答

        Args:
            question: 用户问题

        Returns:
            包含回答、来源、工具调用记录的结果字典
        """
        # 清空执行历史
        self._execution_history = []

        try:
            # 执行ReAct循环 - ReActAgent.run是异步方法
            async def _run():
                return await self._agent.run(question)

            response = run_async(_run())

            # 提取结果 - response可能是AgentOutput类型
            response_text = str(response) if response else "无回答"

            result = {
                "response": response_text,
                "question": question,
                "sources": self._extract_sources(response),
                "tool_calls": self._extract_tool_calls(response),
                "iterations": self._count_iterations(response),
                "mode": "react",
            }

            return result

        except Exception as e:
            return {
                "response": f"Agent执行出错: {str(e)}",
                "question": question,
                "sources": [],
                "tool_calls": [],
                "iterations": 0,
                "mode": "react",
                "error": str(e),
            }

    def query_stream(self, question: str):
        """流式执行Agent问答

        Args:
            question: 用户问题

        Yields:
            响应片段
        """
        try:
            # ReActAgent的run是异步的，使用runner
            from llama_index.core.agent.runner import AgentRunner
            runner = AgentRunner(self._agent)
            for chunk in runner.run_stream(question):
                yield str(chunk)
        except Exception as e:
            yield f"Agent执行出错: {str(e)}"

    def _extract_sources(self, response: Any) -> List[Dict[str, Any]]:
        """从响应中提取来源信息

        Args:
            response: Agent响应对象

        Returns:
            来源信息列表
        """
        sources = []

        # 尝试从工具调用结果中提取
        if hasattr(response, 'sources'):
            for src in response.sources:
                if hasattr(src, 'node'):
                    sources.append({
                        "text": src.node.text[:300],
                        "score": src.score if hasattr(src, 'score') else 0,
                        "source_file": src.node.metadata.get('source_file', ''),
                        "node_id": src.node.node_id,
                    })

        # 也可以从保存的检索结果中提取
        if not sources and self.tools._last_retrieved_nodes:
            for n in self.tools._last_retrieved_nodes:
                sources.append({
                    "text": n.node.text[:300],
                    "score": n.score,
                    "source_file": n.node.metadata.get('source_file', ''),
                    "node_id": n.node.node_id,
                })

        return sources

    def _extract_tool_calls(self, response: Any) -> List[Dict[str, Any]]:
        """提取工具调用记录

        Args:
            response: Agent响应对象

        Returns:
            工具调用记录列表
        """
        tool_calls = []

        # ReActAgent会在响应中记录工具调用
        # 具体实现取决于LlamaIndex版本
        if hasattr(response, 'tool_calls'):
            for tc in response.tool_calls:
                tool_calls.append({
                    "tool": tc.tool_name if hasattr(tc, 'tool_name') else str(tc),
                    "input": tc.tool_input if hasattr(tc, 'tool_input') else {},
                })

        return tool_calls

    def _count_iterations(self, response: Any) -> int:
        """统计迭代次数

        Args:
            response: Agent响应对象

        Returns:
            迭代次数
        """
        # 从工具调用数量估算
        tool_calls = self._extract_tool_calls(response)
        return len(tool_calls)

    def reset(self) -> None:
        """重置Agent状态"""
        self._execution_history = []
        self.tools._last_retrieved_nodes = []
        # ReActAgent的重置方法
        if hasattr(self._agent, 'reset'):
            self._agent.reset()

    def set_verbose(self, verbose: bool) -> None:
        """设置是否显示详细日志

        Args:
            verbose: 是否显示详细日志
        """
        self.verbose = verbose
        # ReActAgent的verbose设置
        if hasattr(self._agent, 'verbose'):
            self._agent.verbose = verbose


class AgentFactory:
    """Agent工厂类"""

    @staticmethod
    def create_react_agent(
        retriever: Any,
        synthesizer: Any,
        llm: Any,
        max_iterations: int = 10,
        verbose: bool = True,
        markdown_base_path: Optional[str] = None,
        pre_retrieval: Any = None,
    ) -> RAGReActAgent:
        """创建ReAct Agent

        Args:
            retriever: 检索器实例
            synthesizer: 合成器实例
            llm: LLM实例
            max_iterations: 最大迭代次数
            verbose: 是否显示详细日志
            markdown_base_path: Markdown文件目录路径（用于表格索引解析）
            pre_retrieval: PreRetrievalPipeline实例（可选）

        Returns:
            RAGReActAgent实例
        """
        tools = RAGTools(
            retriever=retriever,
            synthesizer=synthesizer,
            llm=llm,
            markdown_base_path=markdown_base_path,
            pre_retrieval=pre_retrieval,
        )
        return RAGReActAgent(
            tools=tools,
            llm=llm,
            max_iterations=max_iterations,
            verbose=verbose,
        )