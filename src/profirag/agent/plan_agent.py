"""Plan-based RAG Agent for intelligent query execution"""

import json
import re
import time
from enum import Enum
from typing import List, Dict, Any, Optional, Callable

from pydantic import BaseModel, Field
from llama_index.core.tools import FunctionTool

from .tools import RAGTools


# Prompts for plan generation
COMPLEXITY_ANALYSIS_PROMPT = """分析用户问题复杂度，制定最优检索和回答计划。

问题复杂度判断标准：
- **简单**: 单概念查询、一步可完成、无歧义 → 直接用 retrieve_and_answer
- **中等**: 需1-2步检索、可能需要评估结果质量、有明确范围 → vector_search → generate_answer
- **复杂**: 多角度问题、涉及多个概念、需要多轮检索、可能涉及表格 → multi_query_search → table_lookup? → generate_answer

可用工具及其参数：
- vector_search(query, top_k=5): 向量相似度检索
- keyword_search(query, top_k=5): BM25关键词检索
- multi_query_search(query): 多变体检索扩大覆盖
- hyde_search(query): 假设文档检索
- generate_answer(question, mode="default", top_k=5): 基于检索结果生成回答，mode可选simple/default/professional/technical
- retrieve_and_answer(question, mode="default", top_k=5): 检索+回答一步完成，mode可选simple/default/professional/technical
- table_lookup(table_reference): 查看表格完整内容

输出格式（JSON）：
{{"complexity": "simple|medium|complex", "reasoning": "复杂度判断原因", "requires_approval": true或false, "steps": [{{"tool_name": "工具名", "parameters": {{}}, "expected_output": "预期输出描述"}}]}}

用户问题: {query_str}

请分析复杂度并制定执行计划（输出JSON）："""


REPLAN_PROMPT = """执行计划第{failed_step_index}步失败，请调整计划。

原计划:
{original_plan_json}

失败步骤: {failed_tool_name}
错误信息: {error_message}
已执行步骤结果摘要: {context_summary}

请生成新的执行计划（JSON格式），跳过或替换失败步骤。如果检索结果不足，可以尝试其他检索工具。"""


MODIFY_PLAN_PROMPT = """根据用户反馈修改执行计划。

原计划:
{plan_json}

用户修改指令: {modification}

请输出修改后的计划（JSON格式）："""


class StepStatus(str, Enum):
    """Execution step status"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"


class PlanComplexity(str, Enum):
    """Plan complexity level"""
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"


class PlanStep(BaseModel):
    """Single step in execution plan"""
    tool_name: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    expected_output: str = ""
    depends_on: Optional[int] = None
    status: StepStatus = StepStatus.PENDING
    result: Optional[str] = None
    error: Optional[str] = None


class ExecutionPlan(BaseModel):
    """Complete execution plan"""
    steps: List[PlanStep]
    reasoning: str
    complexity: PlanComplexity = PlanComplexity.MEDIUM
    requires_approval: bool = True


class StepResult(BaseModel):
    """Result of a single step execution"""
    step_index: int
    tool_name: str
    input_params: Dict[str, Any]
    output: str
    success: bool
    error: Optional[str] = None
    duration_ms: Optional[int] = None


class PlanExecutionResult(BaseModel):
    """Complete execution result"""
    plan: ExecutionPlan
    step_results: List[StepResult]
    replan_count: int = 0
    final_answer: str = ""
    success: bool = True
    error: Optional[str] = None


class PlanApproval(BaseModel):
    """User approval response"""
    approved: bool
    modified_plan: Optional[ExecutionPlan] = None
    feedback: Optional[str] = None


class PlanGenerator:
    """Generate execution plan using LLM with complexity analysis"""

    def __init__(self, llm: Any):
        self.llm = llm

    def generate_plan(self, query_str: str, available_tools: List[str]) -> ExecutionPlan:
        """Generate plan with complexity analysis"""
        prompt = COMPLEXITY_ANALYSIS_PROMPT.format(query_str=query_str)
        response = self.llm.complete(prompt)
        plan_dict = self._parse_plan_response(response.text, available_tools)
        return ExecutionPlan(**plan_dict)

    def _parse_plan_response(self, response_text: str, available_tools: List[str]) -> Dict:
        """Parse JSON from LLM response"""
        # Extract JSON from response
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            try:
                plan_dict = json.loads(json_match.group())
                # Normalize field names
                if "steps" in plan_dict:
                    for step in plan_dict["steps"]:
                        # Ensure tool_name field
                        if "tool" in step and "tool_name" not in step:
                            step["tool_name"] = step.pop("tool")
                        # Ensure parameters field
                        if "params" in step and "parameters" not in step:
                            step["parameters"] = step.pop("params")
                        if "parameters" not in step:
                            step["parameters"] = {}
                        # Validate tool name
                        if step["tool_name"] not in available_tools:
                            step["tool_name"] = "retrieve_and_answer"
                return plan_dict
            except json.JSONDecodeError:
                pass

        # Fallback: simple plan
        return {
            "steps": [{"tool_name": "retrieve_and_answer", "parameters": {"mode": "default"}}],
            "reasoning": "解析失败，使用默认计划",
            "complexity": "medium",
            "requires_approval": False
        }

    def replan_after_failure(
        self,
        original_plan: ExecutionPlan,
        failed_step_index: int,
        error: str,
        context: Dict[str, Any],
        available_tools: List[str]
    ) -> ExecutionPlan:
        """Generate new plan after step failure"""
        context_summary = {}
        for key, value in context.items():
            if isinstance(value, str):
                context_summary[key] = value[:200] + "..." if len(value) > 200 else value
            else:
                context_summary[key] = str(value)[:200]

        prompt = REPLAN_PROMPT.format(
            failed_step_index=failed_step_index + 1,
            original_plan_json=original_plan.model_dump_json(),
            failed_tool_name=original_plan.steps[failed_step_index].tool_name,
            error_message=error,
            context_summary=json.dumps(context_summary, ensure_ascii=False)
        )

        response = self.llm.complete(prompt)
        plan_dict = self._parse_plan_response(response.text, available_tools)
        return ExecutionPlan(**plan_dict)

    def modify_plan(self, plan: ExecutionPlan, modification: str, available_tools: List[str]) -> ExecutionPlan:
        """Modify plan based on user feedback"""
        prompt = MODIFY_PLAN_PROMPT.format(
            plan_json=plan.model_dump_json(),
            modification=modification
        )

        response = self.llm.complete(prompt)
        plan_dict = self._parse_plan_response(response.text, available_tools)
        return ExecutionPlan(**plan_dict)


class PlanExecutor:
    """Execute plan steps with detailed logging and failure handling"""

    def __init__(
        self,
        tools: Dict[str, FunctionTool],
        plan_generator: PlanGenerator,
        available_tools: List[str],
        verbose: bool = True,
        logger: Optional[Callable] = None
    ):
        self.tools = tools
        self.plan_generator = plan_generator
        self.available_tools = available_tools
        self.verbose = verbose
        self.logger = logger or print

    def execute(
        self,
        plan: ExecutionPlan,
        max_replan: int = 3
    ) -> PlanExecutionResult:
        """Execute plan with failure handling and replanning"""
        replan_count = 0
        current_plan = plan
        all_results: List[StepResult] = []
        context: Dict[str, Any] = {}

        while replan_count <= max_replan:
            # Execute steps
            step_results, failed_step, context = self._execute_plan_steps(current_plan, context)

            if not failed_step:
                # All steps succeeded
                all_results.extend(step_results)
                return PlanExecutionResult(
                    plan=current_plan,
                    step_results=all_results,
                    replan_count=replan_count,
                    success=True
                )

            # Handle failure
            all_results.extend(step_results[:failed_step.step_index + 1])

            if replan_count < max_replan:
                self.logger(f"\n⚠️ 步骤 {failed_step.step_index + 1} 失败，正在重新规划...")
                current_plan = self.plan_generator.replan_after_failure(
                    current_plan,
                    failed_step.step_index,
                    failed_step.error or "Unknown error",
                    context,
                    self.available_tools
                )
                replan_count += 1
                self.logger(f"🔄 新计划 (重规划 #{replan_count}):")
                self._display_plan(current_plan)
                # Reset context for new plan execution
                context = {}
            else:
                return PlanExecutionResult(
                    plan=current_plan,
                    step_results=all_results,
                    replan_count=replan_count,
                    success=False,
                    error="Max replan attempts reached"
                )

        return PlanExecutionResult(
            plan=current_plan,
            step_results=all_results,
            replan_count=replan_count,
            success=True
        )

    def _execute_plan_steps(
        self,
        plan: ExecutionPlan,
        initial_context: Dict[str, Any]
    ) -> tuple[List[StepResult], Optional[StepResult], Dict[str, Any]]:
        """Execute all steps in plan, returns (results, failed_step, context)"""
        results = []
        context = initial_context.copy()
        failed_step = None

        for i, step in enumerate(plan.steps):
            self.logger(f"\n📌 执行步骤 {i + 1}: {step.tool_name}")

            start_time = time.time()
            try:
                tool = self.tools.get(step.tool_name)
                if not tool:
                    raise ValueError(f"Tool '{step.tool_name}' not found")

                # Get the function from FunctionTool
                tool_fn = tool.fn if hasattr(tool, 'fn') else tool
                params = self._resolve_params(step.parameters, context)

                if self.verbose:
                    self.logger(f"   参数: {params}")

                result = tool_fn(**params)
                duration = int((time.time() - start_time) * 1000)

                step_result = StepResult(
                    step_index=i,
                    tool_name=step.tool_name,
                    input_params=params,
                    output=str(result) if result else "",
                    success=True,
                    duration_ms=duration
                )

                self.logger(f"   ✅ 成功 ({duration}ms)")
                if self.verbose and result:
                    output_preview = str(result)[:200] + "..." if len(str(result)) > 200 else str(result)
                    self.logger(f"   结果: {output_preview}")

                context[f"step_{i}"] = result
                results.append(step_result)

            except Exception as e:
                duration = int((time.time() - start_time) * 1000)
                step_result = StepResult(
                    step_index=i,
                    tool_name=step.tool_name,
                    input_params=step.parameters,
                    output="",
                    success=False,
                    error=str(e),
                    duration_ms=duration
                )
                self.logger(f"   ❌ 失败: {str(e)}")
                results.append(step_result)
                failed_step = step_result
                break  # Stop execution on failure

        return results, failed_step, context

    def _resolve_params(
        self,
        params: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Resolve parameter references from context"""
        resolved = {}
        for key, value in params.items():
            if isinstance(value, str) and value.startswith("$step_"):
                # Reference to previous step result
                step_ref = value[1:]  # Remove $
                if step_ref in context:
                    resolved[key] = context[step_ref]
                else:
                    resolved[key] = value
            else:
                resolved[key] = value
        return resolved

    def _display_plan(self, plan: ExecutionPlan) -> None:
        """Display plan steps"""
        self.logger("\n📌 新执行计划:")
        for i, step in enumerate(plan.steps):
            params_str = ", ".join(f"{k}={v}" for k, v in step.parameters.items())
            self.logger(f"  {i + 1}. {step.tool_name}({params_str})")


class RAGPlanAgent:
    """Plan-based RAG Agent: Plan → Approve → Execute → Answer"""

    def __init__(
        self,
        tools: RAGTools,
        llm: Any,
        verbose: bool = True,
        show_plan: bool = True,
        require_approval: bool = True,
        max_replan_attempts: int = 3,
        approval_callback: Optional[Callable[[ExecutionPlan], PlanApproval]] = None,
    ):
        """Initialize PlanAgent.

        Args:
            tools: RAGTools instance
            llm: LLM instance for planning and answer generation
            verbose: Show detailed logs
            show_plan: Show execution plan
            require_approval: Require user approval for complex plans
            max_replan_attempts: Maximum replanning attempts on failure
            approval_callback: Optional callback for approval (bypasses interactive prompt)
        """
        self.tools = tools
        self.llm = llm
        self.verbose = verbose
        self.show_plan = show_plan
        self.require_approval = require_approval
        self.max_replan_attempts = max_replan_attempts
        self.approval_callback = approval_callback

        self._plan_generator = PlanGenerator(llm)
        self._tools_list = tools.create_all_tools()
        self._tools_dict = {t.metadata.name: t for t in self._tools_list}
        self._executor = PlanExecutor(
            self._tools_dict,
            self._plan_generator,
            list(self._tools_dict.keys()),
            verbose,
            self._log
        )

    def query(self, question: str, auto_approve: bool = False) -> Dict[str, Any]:
        """Execute query with planning.

        Args:
            question: User question
            auto_approve: Auto approve plan (skip approval prompt)

        Returns:
            Result dictionary with response, plan, execution results
        """
        self._log(f"\n{'=' * 50}")
        self._log(f"🤖 PlanAgent 处理问题: {question}")
        self._log(f"{'=' * 50}\n")

        # Phase 1: Generate Plan
        plan = self._plan_generator.generate_plan(
            question,
            list(self._tools_dict.keys())
        )

        self._log(f"📋 计划复杂度: {plan.complexity}")
        self._log(f"📝 计划原因: {plan.reasoning}")

        if self.show_plan:
            self._display_plan(plan)

        # Phase 2: Approval (计划确认)
        if self.require_approval and plan.requires_approval and not auto_approve:
            approval = self._get_approval(plan)
            if not approval.approved:
                return {
                    "response": "用户拒绝了执行计划",
                    "plan": plan,
                    "mode": "plan",
                    "approved": False,
                }
            if approval.modified_plan:
                plan = approval.modified_plan
                self._log("\n📝 使用修改后的计划:")
                self._display_plan(plan)

        # Phase 3: Execute with Replanning (失败重规划)
        execution_result = self._executor.execute(plan, self.max_replan_attempts)

        if not execution_result.success:
            return {
                "response": f"执行失败: {execution_result.error or '未知错误'}",
                "plan": plan,
                "execution_result": execution_result,
                "mode": "plan",
                "error": execution_result.error,
            }

        # Phase 4: Generate Final Answer
        final_answer = self._finalize_answer(question, execution_result)

        return {
            "response": final_answer,
            "plan": plan,
            "execution_result": execution_result,
            "step_results": execution_result.step_results,
            "replan_count": execution_result.replan_count,
            "mode": "plan",
        }

    def _get_approval(self, plan: ExecutionPlan) -> PlanApproval:
        """Get user approval for plan"""
        if self.approval_callback:
            return self.approval_callback(plan)

        # Default: interactive approval
        self._log("\n" + "-" * 40)
        self._log("是否批准此计划？")
        self._log("  y - 执行计划")
        self._log("  n - 拒绝执行")
        self._log("  m - 修改计划")
        self._log("-" * 40)

        try:
            response = input("请选择 (y/n/m): ").strip().lower()
        except EOFError:
            # Non-interactive environment, auto approve
            return PlanApproval(approved=True)

        if response == "y":
            return PlanApproval(approved=True)
        elif response == "n":
            return PlanApproval(approved=False)
        elif response == "m":
            self._log("\n请输入修改指令（如：'添加 keyword_search 步骤'）:")
            try:
                modification = input("修改: ").strip()
            except EOFError:
                modification = ""
            if modification:
                modified_plan = self._plan_generator.modify_plan(
                    plan,
                    modification,
                    list(self._tools_dict.keys())
                )
                return PlanApproval(approved=True, modified_plan=modified_plan, feedback=modification)
            return PlanApproval(approved=True)

        return PlanApproval(approved=False)

    def _finalize_answer(self, question: str, execution_result: PlanExecutionResult) -> str:
        """Generate final answer from execution results"""
        # Check if last step already generated an answer
        last_result = execution_result.step_results[-1] if execution_result.step_results else None

        if last_result and last_result.tool_name in ("generate_answer", "retrieve_and_answer"):
            # Use the answer from the last step
            return last_result.output

        # Otherwise, generate answer from collected results
        context_parts = []
        for result in execution_result.step_results:
            if result.success and result.output:
                context_parts.append(f"[步骤{result.step_index + 1}] {result.tool_name}: {result.output}")

        if not context_parts:
            return "无法生成回答，执行结果为空"

        context_str = "\n\n".join(context_parts)

        prompt = f"""基于以下执行结果生成最终回答。

问题: {question}

执行结果:
{context_str}

请生成完整、准确的回答（使用中文）："""

        response = self.llm.complete(prompt)
        return response.text

    def _log(self, message: str) -> None:
        """Log message if verbose"""
        if self.verbose:
            print(message)

    def _display_plan(self, plan: ExecutionPlan) -> None:
        """Display plan steps"""
        self._log("\n📌 执行计划:")
        for i, step in enumerate(plan.steps):
            params_str = ", ".join(f"{k}={v}" for k, v in step.parameters.items())
            self._log(f"  {i + 1}. {step.tool_name}({params_str})")
            if step.expected_output:
                self._log(f"     预期: {step.expected_output}")
        self._log("")