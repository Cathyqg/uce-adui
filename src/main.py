"""
uce-adui - 主入口
使用 LangGraph 执行完整迁移流程
"""
from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Dict, List, Optional

from langgraph.checkpoint.memory import MemorySaver

from src.core.graph import compile_graph
from src.core.graph_hybrid import compile_hybrid_graph
from src.core.state import MigrationGraphState, ReviewDecision, create_initial_state


class MigrationEngine:
    """迁移引擎 - 管理整个迁移流程"""
    
    def __init__(
        self,
        checkpointer=None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        初始化迁移引擎
        
        Args:
            checkpointer: LangGraph checkpointer (默认使用 MemorySaver)
            config: 全局配置
        """
        self.checkpointer = checkpointer or MemorySaver()
        self.config = config or {}
        graph_mode = (self.config.get("graph_mode") or os.getenv("MIGRATION_GRAPH_MODE", "hybrid")).lower()
        graph_debug = self.config.get("graph_debug")
        if graph_mode == "hybrid":
            self.graph = (
                compile_hybrid_graph(self.checkpointer)
                if graph_debug is None
                else compile_hybrid_graph(self.checkpointer, debug=bool(graph_debug))
            )
        else:
            self.graph = (
                compile_graph(self.checkpointer)
                if graph_debug is None
                else compile_graph(self.checkpointer, debug=bool(graph_debug))
            )
        self._current_thread_id = None
    
    async def start_migration(
        self,
        source_path: str,
        aem_page_json_paths: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None,
        thread_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        启动迁移流程
        
        Args:
            source_path: AEM 组件源码路径
            aem_page_json_paths: AEM 页面 JSON 文件路径列表
            config: 迁移配置
            thread_id: 会话 ID (用于恢复)
        
        Returns:
            迁移状态或中断信息
        """
        # 合并配置
        migration_config = {**self.config, **(config or {})}
        
        # 创建初始状态
        initial_state = create_initial_state(
            source_path=source_path,
            aem_page_json_paths=aem_page_json_paths,
            config=migration_config
        )
        
        # 设置线程 ID
        self._current_thread_id = thread_id or initial_state["session_id"]
        
        # 执行图
        config_dict = {"configurable": {"thread_id": self._current_thread_id}}
        
        try:
            result = await self.graph.ainvoke(initial_state, config=config_dict)
            
            # 检查是否中断
            if result.get("should_interrupt"):
                return {
                    "status": "interrupted",
                    "reason": result.get("interrupt_reason"),
                    "thread_id": self._current_thread_id,
                    "pending_review": result.get("pending_human_review", []),
                    "review_package": result.get("human_review_package")
                }
            
            return {
                "status": "completed",
                "thread_id": self._current_thread_id,
                "stats": result.get("stats", {}),
                "report_path": result.get("report_path")
            }
            
        except Exception as e:
            return {
                "status": "error",
                "thread_id": self._current_thread_id,
                "error": str(e)
            }
    
    async def submit_human_review(
        self,
        thread_id: str,
        component_id: str,
        decision: ReviewDecision,
        feedback: str = "",
        modification: Optional[Dict[str, str]] = None,
        reviewer: str = "human"
    ) -> Dict[str, Any]:
        """
        提交人工审查结果
        
        Args:
            thread_id: 会话 ID
            component_id: 组件 ID
            decision: 审查决定
            feedback: 反馈说明
            modification: 代码修改 (当 decision 为 MODIFY 时)
            reviewer: 审查者标识
        
        Returns:
            继续执行后的状态
        """
        config_dict = {"configurable": {"thread_id": thread_id}}
        
        # 获取当前状态
        state = await self.graph.aget_state(config_dict)
        
        if not state or not state.values:
            return {
                "status": "error",
                "error": "Session not found"
            }
        
        current_state = state.values
        
        # 更新人工审查决定
        human_review_decisions = dict(current_state.get("human_review_decisions", {}))
        human_review_decisions[component_id] = {
            "decision": decision.value if isinstance(decision, ReviewDecision) else decision,
            "feedback": feedback,
            "modification": modification,
            "reviewer": reviewer
        }
        
        # 更新状态
        update = {
            "human_review_decisions": human_review_decisions,
            "should_interrupt": False
        }
        
        await self.graph.aupdate_state(config_dict, update)
        
        # 继续执行
        try:
            result = await self.graph.ainvoke(None, config=config_dict)
            
            if result.get("should_interrupt"):
                return {
                    "status": "interrupted",
                    "reason": result.get("interrupt_reason"),
                    "thread_id": thread_id,
                    "pending_review": result.get("pending_human_review", []),
                    "review_package": result.get("human_review_package")
                }
            
            return {
                "status": "completed",
                "thread_id": thread_id,
                "stats": result.get("stats", {}),
                "report_path": result.get("report_path")
            }
            
        except Exception as e:
            return {
                "status": "error",
                "thread_id": thread_id,
                "error": str(e)
            }
    
    async def get_migration_status(self, thread_id: str) -> Dict[str, Any]:
        """
        获取迁移状态
        
        Args:
            thread_id: 会话 ID
        
        Returns:
            当前迁移状态
        """
        config_dict = {"configurable": {"thread_id": thread_id}}
        state = await self.graph.aget_state(config_dict)
        
        if not state or not state.values:
            return {"status": "not_found"}
        
        values = state.values
        
        return {
            "status": "active",
            "thread_id": thread_id,
            "current_phase": values.get("current_phase"),
            "stats": values.get("stats", {}),
            "pending_human_review": values.get("pending_human_review", []),
            "errors": values.get("errors", []),
            "warnings": values.get("warnings", [])
        }
    
    async def get_review_package(
        self,
        thread_id: str,
        component_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        获取组件的审查数据包
        
        Args:
            thread_id: 会话 ID
            component_id: 组件 ID
        
        Returns:
            审查所需的所有数据
        """
        config_dict = {"configurable": {"thread_id": thread_id}}
        state = await self.graph.aget_state(config_dict)
        
        if not state or not state.values:
            return None
        
        values = state.values
        components = values.get("components", {})
        
        if component_id not in components:
            return None
        
        comp_data = components[component_id]
        
        return {
            "component_id": component_id,
            "original": {
                "htl": comp_data.get("aem_component", {}).get("htl_template", {}).get("raw_content", ""),
                "dialog": comp_data.get("aem_component", {}).get("dialog", {}),
                "js": comp_data.get("aem_component", {}).get("clientlib", {}).get("js_content", ""),
                "css": comp_data.get("aem_component", {}).get("clientlib", {}).get("css_content", "")
            },
            "generated": {
                "code": comp_data.get("react_component", {}).get("component_code", ""),
                "styles": comp_data.get("react_component", {}).get("styles_code", "")
            },
            "review_results": comp_data.get("review", {}),
            "mapping": comp_data.get("bdl_mapping", {})
        }


# ============================================================================
# CLI 接口
# ============================================================================

async def run_migration_cli():
    """命令行运行迁移"""
    import argparse
    
    parser = argparse.ArgumentParser(description="uce-adui CLI")
    parser.add_argument("source_path", help="Path to AEM components directory")
    parser.add_argument("--pages", nargs="*", help="AEM page JSON files")
    parser.add_argument("--output", default="./output", help="Output directory")
    parser.add_argument("--bdl-spec", help="Path to BDL specification file")
    parser.add_argument("--auto-approve", action="store_true", help="Skip human review")
    parser.add_argument("--graph", choices=["pipeline", "hybrid"], default="hybrid", help="Graph mode")
    
    args = parser.parse_args()
    
    config = {
        "output_dir": args.output,
        "bdl_spec_path": args.bdl_spec,
        "auto_approve_all": args.auto_approve,
        "graph_mode": args.graph,
    }
    
    engine = MigrationEngine(config=config)
    
    print(f"Starting migration from: {args.source_path}")
    print(f"Output directory: {args.output}")
    
    result = await engine.start_migration(
        source_path=args.source_path,
        aem_page_json_paths=args.pages
    )
    
    if result["status"] == "completed":
        print("\n[OK] Migration completed successfully!")
        print(f"Report: {result.get('report_path')}")
        stats = result.get("stats", {})
        print(f"Components processed: {stats.get('generated_components', 0)}")
        print(f"Components approved: {stats.get('approved_components', 0)}")
        print(f"Pages migrated: {stats.get('migrated_pages', 0)}")
    
    elif result["status"] == "interrupted":
        print("\n[PAUSED] Migration paused - Human review required")
        print(f"Thread ID: {result['thread_id']}")
        print(f"Pending review: {result.get('pending_review', [])}")
        print("\nUse the API to submit reviews and continue.")
    
    else:
        print(f"\n[ERROR] Migration failed: {result.get('error')}")


if __name__ == "__main__":
    asyncio.run(run_migration_cli())
