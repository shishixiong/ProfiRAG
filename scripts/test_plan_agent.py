"""Test PlanAgent functionality"""

import sys
sys.path.insert(0, "src")

import argparse
from profirag.pipeline import RAGPipeline
from profirag.agent import AgentFactory, PlanGenerator, RAGPlanAgent


def test_complexity_analysis():
    """Test plan complexity analysis"""
    print("\n" + "="*50)
    print("测试: 复杂度分析")
    print("="*50)

    pipeline = RAGPipeline.from_env()
    generator = PlanGenerator(pipeline._llm)

    queries = [
        ("GaussDB是什么?", "simple"),
        ("GaussDB的连接参数有哪些?", "medium"),
        ("如何配置GaussDB的SSL连接，涉及哪些参数和步骤?", "complex"),
        ("GaussDB支持哪些数据类型，请给出详细说明和示例", "complex"),
    ]

    available_tools = ["vector_search", "keyword_search", "retrieve_and_answer", "generate_answer", "multi_query_search"]

    for query, expected_complexity in queries:
        print(f"\n问题: {query}")
        print(f"预期复杂度: {expected_complexity}")
        plan = generator.generate_plan(query, available_tools)
        print(f"实际复杂度: {plan.complexity}")
        print(f"计划原因: {plan.reasoning}")
        print(f"步骤数: {len(plan.steps)}")
        for i, step in enumerate(plan.steps):
            print(f"  {i+1}. {step.tool_name}({step.parameters})")


def test_plan_agent_query():
    """Test full PlanAgent workflow"""
    print("\n" + "="*50)
    print("测试: PlanAgent 查询")
    print("="*50)

    pipeline = RAGPipeline.from_env()

    # Test with auto_approve to avoid interactive prompts
    query = "GaussDB支持哪些数据类型?"
    print(f"\n问题: {query}")

    result = pipeline.query_with_agent(
        query,
        mode="plan",
        auto_approve=True
    )

    print(f"\n回答: {result['response'][:500]}...")
    print(f"\n计划步骤数: {len(result['plan'].steps)}")
    print(f"重规划次数: {result.get('replan_count', 0)}")
    print(f"模式: {result['mode']}")

    # Show step results
    if result.get('step_results'):
        print("\n步骤执行结果:")
        for sr in result['step_results']:
            status = "✅" if sr.success else "❌"
            print(f"  {status} 步骤{sr.step_index+1}: {sr.tool_name} ({sr.duration_ms}ms)")


def test_plan_agent_with_approval():
    """Test PlanAgent with approval process (interactive)"""
    print("\n" + "="*50)
    print("测试: PlanAgent 计划确认 (交互式)")
    print("="*50)

    pipeline = RAGPipeline.from_env()

    query = "GaussDB的连接参数有哪些，如何配置?"
    print(f"\n问题: {query}")
    print("\n注意: 这将需要你确认计划，请输入 'y' 执行")

    result = pipeline.query_with_agent(
        query,
        mode="plan",
        auto_approve=False
    )

    print(f"\n回答: {result['response'][:500]}...")
    print(f"计划是否批准: {result.get('approved', True)}")


def test_failure_handling():
    """Test replanning on failure"""
    print("\n" + "="*50)
    print("测试: 失败重规划")
    print("="*50)

    pipeline = RAGPipeline.from_env()

    # Simulate a query that might return empty results
    query = "查询一个不存在的话题 xyz123abc"
    print(f"\n问题: {query}")

    result = pipeline.query_with_agent(
        query,
        mode="plan",
        auto_approve=True
    )

    print(f"\n回答: {result['response']}")
    print(f"成功: {result.get('success', True)}")
    if result.get('error'):
        print(f"错误: {result['error']}")
    print(f"重规划次数: {result.get('replan_count', 0)}")


def compare_react_vs_plan():
    """Compare ReAct and Plan agents on same query"""
    print("\n" + "="*50)
    print("测试: ReAct vs Plan Agent 对比")
    print("="*50)

    pipeline = RAGPipeline.from_env()

    query = "GaussDB的连接参数有哪些?"
    print(f"\n问题: {query}")

    # ReAct mode
    print("\n--- ReAct Agent ---")
    try:
        r1 = pipeline.query_with_agent(query, mode="react")
        print(f"回答: {r1['response'][:300]}...")
        print(f"迭代次数: {r1.get('iterations', 'N/A')}")
    except Exception as e:
        print(f"ReAct Agent 错误: {e}")

    # Plan mode
    print("\n--- Plan Agent ---")
    r2 = pipeline.query_with_agent(query, mode="plan", auto_approve=True)
    print(f"回答: {r2['response'][:300]}...")
    print(f"计划步骤: {len(r2['plan'].steps)}")
    print(f"重规划次数: {r2.get('replan_count', 0)}")


def interactive_query():
    """Interactive query with PlanAgent"""
    print("\n" + "="*50)
    print("交互式查询模式")
    print("="*50)

    pipeline = RAGPipeline.from_env()

    while True:
        print("\n输入问题 (输入 'quit' 退出):")
        try:
            query = input("> ").strip()
        except EOFError:
            break

        if query.lower() == 'quit':
            break

        if not query:
            continue

        print("\n选择模式:")
        print("  1 - Plan Agent (auto approve)")
        print("  2 - Plan Agent (需要确认)")
        print("  3 - ReAct Agent")
        print("  4 - Pipeline")

        try:
            choice = input("选择 (1-4): ").strip()
        except EOFError:
            choice = "1"

        if choice == "1":
            result = pipeline.query_with_agent(query, mode="plan", auto_approve=True)
        elif choice == "2":
            result = pipeline.query_with_agent(query, mode="plan", auto_approve=False)
        elif choice == "3":
            result = pipeline.query_with_agent(query, mode="react")
        else:
            result = pipeline.query(query)

        print(f"\n回答:\n{result['response']}")


def main():
    parser = argparse.ArgumentParser(description="Test PlanAgent functionality")
    parser.add_argument("--test", choices=[
        "complexity", "query", "approval", "failure", "compare", "interactive"
    ], default="complexity", help="Test to run")
    parser.add_argument("--query", type=str, help="Query string for interactive test")

    args = parser.parse_args()

    if args.test == "complexity":
        test_complexity_analysis()
    elif args.test == "query":
        test_plan_agent_query()
    elif args.test == "approval":
        test_plan_agent_with_approval()
    elif args.test == "failure":
        test_failure_handling()
    elif args.test == "compare":
        compare_react_vs_plan()
    elif args.test == "interactive":
        interactive_query()


if __name__ == "__main__":
    main()