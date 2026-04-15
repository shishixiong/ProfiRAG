#!/usr/bin/env python3
"""
ProfiRAG Interactive Q&A System

An interactive command-line interface for querying the RAG system.
Supports text responses with source citations and associated images.

Usage:
    python main.py                    # Start interactive session
    python main.py --query "问题"     # Single query mode
    python main.py --help             # Show help

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

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from profirag.config.settings import load_config
from profirag.pipeline.rag_pipeline import RAGPipeline


class InteractiveSession:
    """Interactive Q&A session manager."""

    def __init__(self, env_file: str = ".env", show_images: bool = True):
        """Initialize interactive session.

        Args:
            env_file: Path to configuration file
            show_images: Whether to show images in responses
        """
        print("=" * 60)
        print("  ProfiRAG Interactive Q&A System")
        print("=" * 60)
        print()

        print("正在加载配置...")
        self.config = load_config(env_file)

        print("正在初始化 RAG 系统...")
        self.pipeline = RAGPipeline(self.config)

        self.show_images = show_images
        self.query_count = 0

        # Show system stats
        stats = self.pipeline.get_stats()
        print()
        print("系统状态:")
        print(f"  - 向量数据库: {stats['vector_store']['count']} 条记录")
        print(f"  - BM25索引: {stats['bm25_index']['count']} 条记录")
        print(f"  - LLM模型: {stats['llm']['model']}")
        print(f"  - 嵌入模型: {stats['embedding']['model']}")
        print()
        print("系统已就绪，请输入问题开始对话。")
        print("输入 /help 查看可用命令，/quit 退出。")
        print("-" * 60)

    def process_query(self, query: str) -> None:
        """Process a single query and display results.

        Args:
            query: User's query string
        """
        self.query_count += 1
        print()
        print(f"[问题 #{self.query_count}] {query}")
        print("-" * 60)

        try:
            if self.show_images:
                result = self.pipeline.query_with_images(query, top_k=5)
                self._display_result_with_images(result)
            else:
                result = self.pipeline.query(query, top_k=5)
                self._display_result(result)

        except Exception as e:
            print(f"处理问题时出错: {e}")
            import traceback
            traceback.print_exc()

        print("-" * 60)

    def _display_result(self, result: dict) -> None:
        """Display query result without images.

        Args:
            result: Query result dictionary
        """
        print()
        print("【回答】")
        print(result.get("response", "无回答"))
        print()

        # Show sources
        sources = result.get("sources", [])
        if sources:
            print("【参考来源】")
            for i, source in enumerate(sources[:3], 1):
                score = source.get("score", 0)
                text = source.get("text", "")[:200]
                source_file = source.get("source_file", "未知")
                print(f"  {i}. [{score:.2f}] {source_file}")
                print(f"     {text}...")
            print()

    def _display_result_with_images(self, result: dict) -> None:
        """Display query result with images.

        Args:
            result: Query result dictionary
        """
        print()
        print("【回答】")
        print(result.get("response", "无回答"))
        print()

        # Show sources
        sources = result.get("sources", [])
        if sources:
            print("【参考来源】")
            for i, source in enumerate(sources[:3], 1):
                score = source.get("score", 0)
                text = source.get("text", "")[:150]
                source_file = source.get("source_file", "未知")
                print(f"  {i}. [{score:.2f}] {source_file}")
                if text:
                    print(f"     {text}...")
            print()

        # Show images
        images = result.get("images", [])
        if images:
            print("【相关图片】")
            for i, img in enumerate(images, 1):
                path = img.get("path", "")
                desc = img.get("description", "")
                score = img.get("score", 0)
                # Check if file exists
                exists = Path(path).exists() if path else False
                status = "✓" if exists else "✗"
                print(f"  {i}. [{score:.2f}] {status} {path}")
                if desc:
                    print(f"     描述: {desc[:100]}...")
            print()
        else:
            print("【相关图片】 无")
            print()

    def handle_command(self, command: str) -> bool:
        """Handle special commands.

        Args:
            command: Command string (starts with /)

        Returns:
            True to continue, False to quit
        """
        cmd = command.lower().strip()

        if cmd in ("/quit", "/exit", "/q"):
            print("再见!")
            return False

        elif cmd == "/help":
            print()
            print("可用命令:")
            print("  /help        - 显示帮助信息")
            print("  /stats       - 显示系统统计")
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
            print(f"  - BM25索引: {stats['bm25_index']['count']} 条")
            print(f"  - 查询次数: {self.query_count}")
            print(f"  - 图片检索: {'启用' if self.show_images else '禁用'}")
            print()

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
            print("\033[2J\033[H")  # ANSI clear screen

        else:
            print(f"未知命令: {command}")
            print("输入 /help 查看可用命令")

        return True

    def run(self) -> None:
        """Run interactive session loop."""
        while True:
            try:
                # Get user input
                user_input = input("\n请输入问题: ").strip()

                if not user_input:
                    continue

                # Handle commands
                if user_input.startswith("/"):
                    if not self.handle_command(user_input):
                        break
                    continue

                # Process query
                self.process_query(user_input)

            except KeyboardInterrupt:
                print("\n再见!")
                break
            except EOFError:
                print("\n再见!")
                break


def single_query(query: str, env_file: str = ".env", show_images: bool = True) -> None:
    """Execute a single query and exit.

    Args:
        query: Query string
        env_file: Configuration file path
        show_images: Whether to include images
    """
    config = load_config(env_file)
    pipeline = RAGPipeline(config)

    if show_images:
        result = pipeline.query_with_images(query, top_k=5)
    else:
        result = pipeline.query(query, top_k=5)

    # Output as JSON for easy parsing
    import json
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main():
    """Main entry point."""
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

    args = parser.parse_args()

    show_images = not args.no_images

    if args.query:
        # Single query mode
        single_query(args.query, args.env, show_images)
    else:
        # Interactive mode
        session = InteractiveSession(args.env, show_images)
        session.run()


if __name__ == "__main__":
    main()