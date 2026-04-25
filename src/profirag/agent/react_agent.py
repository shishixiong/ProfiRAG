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

        # 创建ReAct Agent（传递系统提示词）
        self._agent = ReActAgent(
            tools=self._tools_list,
            llm=llm,
            system_prompt=self._system_prompt,
            max_iterations=max_iterations,
            verbose=verbose,
        )

        # 记录执行历史
        self._execution_history: List[Dict[str, Any]] = []

    def _default_system_prompt(self) -> str:
        """默认系统提示词"""
        return """你是一个专业的技术文档问答助手，基于文档库回答用户问题。

## 回答工具选择

有两种回答生成方式，根据场景选择：

| 场景 | 推荐工具 | 说明 |
|-----|---------|------|
| 简单问题、一步完成 | retrieve_and_answer | 检索+回答一步完成，效率高 |
| 复杂问题、多轮检索 | vector_search → generate_answer | 先检索评估，再生成回答 |
| 需要特定回答模式 | retrieve_and_answer(mode=...) | 指定 professional/technical 模式 |

回答模式说明（两个工具都支持）：
- simple: 简洁回答，<100字，无引用标注
- default: 默认回答，有引用标注和基本结构（推荐）
- professional: 专业回答，详细结构化，代码示例，参数说明
- technical: 技术规范回答，严格按文档，包含版本差异

## 检索工具选择

| 问题类型 | 推荐工具 | 原因 |
|---------|---------|------|
| 精确术语/命令 | keyword_search | BM25关键词匹配更精准 |
| 语义描述/模糊问题 | vector_search | 语义相似度更灵活 |
| 复杂/多角度问题 | multi_query_search | 多变体扩大覆盖 |
| 表述不清的问题 | hyde_search 或 rewrite_query | 重写查询或假设文档补充语义 |
| 代码/参数问题 | keyword_search | 关键词匹配代码片段 |

## 结果优化工具

检索后可使用以下工具优化结果：

| 工具 | 适用场景 | 说明 |
|-----|---------|------|
| rerank_results | 检索结果相关性低 | 对结果重排序，提高相关性（需先检索） |
| filter_results | 结果过多/需聚焦特定文档 | 按来源文件或分数过滤（需先检索） |

**使用时机**：
- 检索后发现结果相关度分数普遍较低（<0.5）→ 使用 rerank_results(query, top_n=5)
- 用户指定了特定文档或需要高置信度结果 → 使用 filter_results(source_file="xxx.md", min_score=0.3)
- 过滤后结果太少 → 可重新检索或放宽过滤条件

## 工作流程

1. **分析问题**：判断问题类型（概念、操作、参数、代码等）
2. **预处理（可选）**：
   - 问题表述模糊 → 先用 rewrite_query 重写查询
3. **选择策略**：
   - 简单问题 → 直接用 retrieve_and_answer
   - 复杂问题 → 先用检索工具，评估结果，再用 generate_answer
4. **执行检索**：调用工具获取文档片段
5. **评估结果**：
   - 相关度分数 > 0.5：结果较好
   - 相关度分数偏低：使用 rerank_results 优化
   - 结果过多或需聚焦：使用 filter_results 过滤
   - 结果包含表格索引：使用 table_lookup 获取完整表格
   - 结果与问题不相关：换用其他工具重新检索
6. **判断终止**：
   - 已获得足够信息 → 生成回答
   - 尝试3种工具仍无结果 → 说明信息不足
   - 已达 max_iterations → 说明超时
7. **生成回答**：选择合适的回答模式生成最终回答

## 表格处理

当检索结果包含表格索引时（格式："表 X-X 标题 → [查看表格](tables/xxx.md)"）：
- 必须使用 table_lookup 工具获取完整表格内容
- 参数可以是完整链接或直接路径（如 "tables/xxx.md"）
- 表格数据应完整呈现给用户，不要截断

## 失败处理

当检索结果不足时：
- **查询问题**：先用 rewrite_query 重写，再重新检索
- **结果质量问题**：使用 rerank_results 重排序优化
- **范围问题**：使用 filter_results 过滤无关结果
- 换用其他检索工具（vector_search → keyword_search → multi_query_search）
- 调整 top_k 参数（默认5，可尝试10）
- 如果所有工具都无结果，如实说明"文档库中未找到相关信息"
- 尝试提供部分信息，不要直接放弃

## 回答质量标准

1. **引用标注**：使用 [文档N] 标注信息来源
2. **结构清晰**：复杂问题用分段或列表组织
3. **代码完整**：代码片段用代码块格式呈现
4. **如实说明**：信息不足时明确说明，提供已知部分
5. **不编造**：只使用检索到的信息，不添加外部知识

## 禁止行为

- 禁止编造文档库中不存在的信息
- 禁止跳过表格索引而不查看完整表格
- 禁止在没有检索结果的情况下直接回答
- 禁止过度迭代（超过5轮仍未终止）
- 禁止在未检索的情况下使用 rerank_results 或 filter_results"""

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
        reranker: Any = None,
        query_rewriter: Any = None,
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
            reranker: Reranker实例（可选，用于结果重排序）
            query_rewriter: QueryRewriter实例（可选，用于查询重写）

        Returns:
            RAGReActAgent实例
        """
        tools = RAGTools(
            retriever=retriever,
            synthesizer=synthesizer,
            llm=llm,
            markdown_base_path=markdown_base_path,
            pre_retrieval=pre_retrieval,
            reranker=reranker,
            query_rewriter=query_rewriter,
        )
        return RAGReActAgent(
            tools=tools,
            llm=llm,
            max_iterations=max_iterations,
            verbose=verbose,
        )

    @staticmethod
    def create_plan_agent(
        retriever: Any,
        synthesizer: Any,
        llm: Any,
        verbose: bool = True,
        show_plan: bool = True,
        require_approval: bool = True,
        max_replan_attempts: int = 3,
        markdown_base_path: Optional[str] = None,
        pre_retrieval: Any = None,
        approval_callback: Optional[Any] = None,
        reranker: Any = None,
        query_rewriter: Any = None,
    ):
        """创建Plan-based Agent

        Args:
            retriever: 检索器实例
            synthesizer: 合成器实例
            llm: LLM实例
            verbose: 是否显示详细日志
            show_plan: 是否显示执行计划
            require_approval: 是否需要用户确认计划
            max_replan_attempts: 失败重规划最大次数
            markdown_base_path: Markdown文件目录路径（用于表格索引解析）
            pre_retrieval: PreRetrievalPipeline实例（可选）
            approval_callback: 计划批准回调函数（可选）
            reranker: Reranker实例（可选，用于结果重排序）
            query_rewriter: QueryRewriter实例（可选，用于查询重写）

        Returns:
            RAGPlanAgent实例
        """
        from .plan_agent import RAGPlanAgent

        tools = RAGTools(
            retriever=retriever,
            synthesizer=synthesizer,
            llm=llm,
            markdown_base_path=markdown_base_path,
            pre_retrieval=pre_retrieval,
            reranker=reranker,
            query_rewriter=query_rewriter,
        )
        return RAGPlanAgent(
            tools=tools,
            llm=llm,
            verbose=verbose,
            show_plan=show_plan,
            require_approval=require_approval,
            max_replan_attempts=max_replan_attempts,
            approval_callback=approval_callback,
        )

    @staticmethod
    def create_conversation_agent(
        agent_type: str,  # "react" or "plan"
        retriever: Any,
        synthesizer: Any,
        llm: Any,
        max_history_turns: int = 6,
        keep_recent_turns: int = 2,
        enable_auto_context: bool = True,
        verbose: bool = False,
        **kwargs
    ):
        """Create ConversationManager wrapping specified agent type.

        Args:
            agent_type: "react" or "plan"
            retriever: Retriever instance
            synthesizer: Synthesizer instance
            llm: LLM instance
            max_history_turns: Max turns before summarization
            keep_recent_turns: Turns kept verbatim
            enable_auto_context: Enable LLM context decision
            verbose: Print enrichment details
            **kwargs: Additional args for underlying agent

        Returns:
            ConversationManager instance
        """
        from .conversation import ConversationManager

        if agent_type == "plan":
            agent = AgentFactory.create_plan_agent(
                retriever=retriever,
                synthesizer=synthesizer,
                llm=llm,
                **kwargs
            )
        else:
            agent = AgentFactory.create_react_agent(
                retriever=retriever,
                synthesizer=synthesizer,
                llm=llm,
                **kwargs
            )

        return ConversationManager(
            agent=agent,
            llm=llm,
            max_history_turns=max_history_turns,
            keep_recent_turns=keep_recent_turns,
            enable_auto_context=enable_auto_context,
            verbose=verbose,
        )