#!/usr/bin/env python3
"""ProfiRAG Interactive Q&A System - CLI entry point.

An interactive command-line interface for querying the RAG system.
Supports text responses with source citations and associated images.

Usage:
    profirag                    # Start interactive session
    profirag --query "问题"     # Single query mode
    profirag --help             # Show help

Commands in interactive mode:
    /help        - Show available commands
    /stats       - Show system statistics
    /images on   - Enable image retrieval
    /images off  - Disable image retrieval
    /clear       - Clear screen
    /quit        - Exit session
"""

import sys
import argparse
from pathlib import Path

from profirag.config.settings import load_config
from profirag.pipeline.rag_pipeline import RAGPipeline
from dotenv import load_dotenv


class InteractiveSession:
    """Interactive Q&A session manager."""

    def __init__(self, config, show_images: bool = True, query_mode: str = "pipeline"):
        print("=" * 60)
        print("  ProfiRAG Interactive Q&A System")
        print("=" * 60)
        print()

        self.config = config

        print("正在初始化 RAG 系统...")
        self.pipeline = RAGPipeline(self.config)

        self.show_images = show_images
        self.query_count = 0
        self.query_mode = query_mode

        stats = self.pipeline.get_stats()
        print()
        print("系统状态:")
        print(f"  - 向量数据库: {stats['vector_store']['count']} 条记录")
        print(f"  - LLM模型: {stats['llm']['model']}")
        print(f"  - 嵌入模型: {stats['embedding']['model']}")
        print()
        print("系统已就绪，请输入问题开始对话。")
        print("输入 /help 查看可用命令，/quit 退出。")
        print("-" * 60)

    def process_query(self, query: str) -> None:
        self.query_count += 1
        print()
        print(f"[问题 #{self.query_count}] {query}")
        print("-" * 60)

        try:
            if self.query_mode == "plan":
                result = self.pipeline.query_with_agent(query, mode="plan", auto_approve=True)
                self._display_plan_agent_result(result)
            elif self.query_mode == "agent":
                result = self.pipeline.query_with_agent(query, mode="agent", timeout=120)
                self._display_agent_result(result)
            elif self.show_images:
                result = self.pipeline.query_with_images(query, top_k=5)
                self._display_result_with_images(result)
            else:
                result = self.pipeline.query(query, top_k=5)
                self._display_result(result)

        except KeyboardInterrupt:
            print("\n⚠️  操作被用户中断 (Ctrl-C)")
            print("提示: 如需退出程序，请在主提示符处输入 /quit")
        except Exception as e:
            print(f"处理问题时出错: {e}")
            import traceback
            traceback.print_exc()

        print("-" * 60)

    def _format_source_display(self, source: dict) -> str:
        source_file = source.get("source_file", "未知")
        header_path = source.get("header_path", "")

        if '/' in source_file:
            source_file = source_file.split('/')[-1]

        if header_path and header_path != '/':
            clean_path = header_path.strip('/').replace('/', ' > ')
            return f"{source_file}#{clean_path}"
        else:
            return source_file

    def _display_result(self, result: dict) -> None:
        print()
        print("【回答】")
        print(result.get("response", "无回答"))
        print()

        sources = result.get("sources", [])
        if sources:
            print("【参考来源】")
            for i, source in enumerate(sources[:3], 1):
                score = source.get("score", 0)
                text = source.get("text", "")[:200]
                source_display = self._format_source_display(source)
                print(f"  {i}. [{score:.2f}] {source_display}")
                if text:
                    print(f"     {text}...")
            print()

    def _display_result_with_images(self, result: dict) -> None:
        print()
        print("【回答】")
        print(result.get("response", "无回答"))
        print()

        sources = result.get("sources", [])
        if sources:
            print("【参考来源】")
            for i, source in enumerate(sources[:3], 1):
                score = source.get("score", 0)
                text = source.get("text", "")[:150]
                source_display = self._format_source_display(source)
                print(f"  {i}. [{score:.2f}] {source_display}")
                if text:
                    print(f"     {text}...")
            print()

        images = result.get("images", [])
        if images:
            print("【相关图片】")
            for i, img in enumerate(images, 1):
                path = img.get("path", "")
                desc = img.get("description", "")
                score = img.get("score", 0)
                exists = Path(path).exists() if path else False
                status = "✓" if exists else "✗"
                print(f"  {i}. [{score:.2f}] {status} {path}")
                if desc:
                    print(f"     描述: {desc[:100]}...")
            print()
        else:
            print("【相关图片】 无")
            print()

    def _display_agent_result(self, result: dict) -> None:
        print()
        print("【回答】")
        response = result.get("response", "无回答")
        print(response)
        print()

        mode = result.get("mode", "unknown")
        iterations = result.get("iterations", 0)
        print(f"【Agent信息】 模式: {mode}, 迭代次数: {iterations}")
        print()

        sources = result.get("sources", [])
        if sources:
            has_sources_in_response = "**参考来源**" in str(response) or "参考来源:" in str(response)
            if not has_sources_in_response:
                print("【参考来源】")
                for i, source in enumerate(sources[:3], 1):
                    score = source.get("score", 0)
                    text = source.get("text", "")[:150]
                    source_display = self._format_source_display(source)
                    print(f"  {i}. [{score:.2f}] {source_display}")
                    if text:
                        print(f"     {text}...")
                print()

        tool_calls = result.get("tool_calls", [])
        if tool_calls:
            print("【工具调用】")
            for i, tc in enumerate(tool_calls, 1):
                tool_name = tc.get("tool", "unknown")
                print(f"  {i}. {tool_name}")
            print()

    def _display_plan_agent_result(self, result: dict) -> None:
        print()
        print("【回答】")
        response = result.get("response", "无回答")
        print(response)
        print()

        plan = result.get("plan")
        if plan:
            complexity = plan.complexity if hasattr(plan, 'complexity') else "unknown"
            reasoning = plan.reasoning if hasattr(plan, 'reasoning') else ""
            step_count = len(plan.steps) if hasattr(plan, 'steps') else 0
            print(f"【Plan信息】")
            print(f"  复杂度: {complexity}")
            print(f"  步骤数: {step_count}")
            print(f"  重规划次数: {result.get('replan_count', 0)}")
            if reasoning:
                print(f"  计划原因: {reasoning[:100]}...")
            print()

        step_results = result.get("step_results", [])
        if step_results:
            print("【执行步骤】")
            for i, sr in enumerate(step_results):
                status = "✅" if sr.success else "❌"
                duration = sr.duration_ms or 0
                print(f"  {i+1}. {status} {sr.tool_name} ({duration}ms)")
            print()

        sources = result.get("sources", [])
        if sources:
            last_step = step_results[-1] if step_results else None
            if last_step and last_step.tool_name in ("generate_answer", "retrieve_and_answer"):
                pass
            else:
                print("【参考来源】")
                for i, source in enumerate(sources[:3], 1):
                    score = source.get("score", 0)
                    text = source.get("text", "")[:150]
                    source_display = self._format_source_display(source)
                    print(f"  {i}. [{score:.2f}] {source_display}")
                    if text:
                        print(f"     {text}...")
                print()

    def handle_command(self, command: str) -> bool:
        cmd = command.lower().strip()

        if cmd in ("/quit", "/exit", "/q"):
            print("再见!")
            return False

        elif cmd == "/help":
            print()
            print("可用命令:")
            print("  /help        - 显示帮助信息")
            print("  /stats       - 显示系统统计")
            print("  /mode pipeline - 使用Pipeline模式（固定流程）")
            print("  /mode agent    - 使用Agent模式（ReAct动态决策）")
            print("  /mode plan     - 使用PlanAgent模式（先规划后执行）")
            print("  /images on   - 启用图片检索")
            print("  /images off  - 禁用图片检索")
            print("  /clear       - 清屏")
            print("  /quit        - 退出程序")
            print()

        elif cmd == "/stats":
            stats = self.pipeline.get_stats()
            print()
            print("系统统计:")
            print(f"  - 向量数据库: {stats['vector_store']['count']} 条")
            print(f"  - 查询次数: {self.query_count}")
            print(f"  - 查询模式: {self.query_mode}")
            print(f"  - 图片检索: {'启用' if self.show_images else '禁用'}")
            print()

        elif cmd.startswith("/mode"):
            parts = cmd.split()
            if len(parts) == 2:
                if parts[1] == "pipeline":
                    self.query_mode = "pipeline"
                    print("已切换到Pipeline模式")
                elif parts[1] == "agent":
                    self.query_mode = "agent"
                    print("已切换到ReAct Agent模式")
                elif parts[1] == "plan":
                    self.query_mode = "plan"
                    print("已切换到Plan Agent模式")
                else:
                    print("用法: /mode pipeline | /mode agent | /mode plan")
            else:
                print(f"当前查询模式: {self.query_mode}")

        elif cmd.startswith("/images"):
            parts = cmd.split()
            if len(parts) == 2:
                if parts[1] == "on":
                    self.show_images = True
                    print("图片检索已启用")
                elif parts[1] == "off":
                    self.show_images = False
                    print("图片检索已禁用")
                else:
                    print("用法: /images on 或 /images off")
            else:
                print(f"当前图片检索状态: {'启用' if self.show_images else '禁用'}")

        elif cmd == "/clear":
            print("\033[2J\033[H")

        else:
            print(f"未知命令: {command}")
            print("输入 /help 查看可用命令")

        return True

    def run(self) -> None:
        while True:
            try:
                user_input = input("\n请输入问题: ").strip()

                if not user_input:
                    continue

                if user_input.startswith("/"):
                    if not self.handle_command(user_input):
                        break
                    continue

                self.process_query(user_input)

            except KeyboardInterrupt:
                print("\n再见!")
                break
            except EOFError:
                print("\n再见!")
                break


def single_query(query: str, config, show_images: bool = True, query_mode: str = "pipeline") -> None:
    pipeline = RAGPipeline(config)

    if query_mode == "plan":
        result = pipeline.query_with_agent(query, mode="plan", auto_approve=True)
        if "plan" in result and hasattr(result["plan"], "model_dump"):
            result["plan"] = result["plan"].model_dump()
        if "execution_result" in result and hasattr(result["execution_result"], "model_dump"):
            result["execution_result"] = result["execution_result"].model_dump()
        if "step_results" in result:
            result["step_results"] = [
                sr.model_dump() if hasattr(sr, "model_dump") else sr
                for sr in result["step_results"]
            ]
    elif query_mode == "agent":
        result = pipeline.query_with_agent(query, mode="agent")
    elif show_images:
        result = pipeline.query_with_images(query, top_k=5)
    else:
        result = pipeline.query(query, top_k=5)

    import json
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(
        description="ProfiRAG Interactive Q&A System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--env", "-e",
        type=str,
        default=".env",
        help="Configuration file path (default: .env)",
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        help="Execute single query and exit",
    )
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="Disable image retrieval",
    )
    parser.add_argument(
        "--mode", "-m",
        type=str,
        choices=["pipeline", "agent", "plan"],
        default=None,
        help="Query mode: pipeline (fixed flow), agent (ReAct dynamic), or plan (PlanAgent)",
    )
    parser.add_argument(
        "--markdown-base-path",
        type=str,
        default=None,
        help="Markdown files directory path for table index resolution (used by Agent)",
    )

    args = parser.parse_args()

    show_images = not args.no_images
    query_mode = args.mode or "pipeline"

    env_path = Path(args.env)
    if not env_path.is_absolute():
        env_path = Path.cwd() / env_path
    load_dotenv(env_path)

    config = load_config(str(env_path))
    if args.markdown_base_path:
        config.agent.markdown_base_path = args.markdown_base_path

    if args.query:
        single_query(args.query, config, show_images, query_mode)
    else:
        session = InteractiveSession(config, show_images, query_mode)
        session.run()


if __name__ == "__main__":
    main()
