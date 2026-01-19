"""
混合架构主图 - Pipeline + Agent
使用 LangGraph 1.0+ 最新特性

架构:
- 87% Pipeline 节点（确定性流程）
- 13% Agent 节点（智能决策和迭代优化）

Agent 节点位置:
1. map_to_bdl → bdl_mapping_agent（智能组件映射）
2. generate_react → code_generator_agent（代码生成+验证循环）
3. code_quality_review → code_reviewer_agent（智能审查）
4. analyze_editables → editor_designer_agent（UX 设计）

LangGraph 1.0+ Features:
- create_react_agent: 预构建 ReAct Agent
- 条件边循环: 支持重新生成
- State Reducers: 自动合并状态
- Checkpointer: 持久化和恢复

================================================================================
⚠️ 使用说明:
================================================================================
这是混合架构实现，与原 graph.py 并存。

默认使用混合版:
- 默认: from src.core.graph_hybrid import compile_hybrid_graph
- 可选: from src.core.graph import compile_graph (仅 Pipeline)
================================================================================
"""

import os
from typing import Any, Dict, List, Literal, Optional

from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import Send
from langgraph.graph import END, START, StateGraph

from src.core.state import MigrationGraphState, Phase


# ============================================================================
# 条件边函数
# ============================================================================

def should_continue_parsing(state: MigrationGraphState) -> Literal["parse_next", "analyze_all"]:
    """判断是否继续解析"""
    if state.get("component_queue"):
        return "parse_next"
    return "analyze_all"


def after_review_router(state: MigrationGraphState) -> Literal["regenerate", "continue", "human_wait"]:
    """
    Review 后的路由决策
    
    ⭐ 支持循环: 可以返回到生成节点
    """
    components = state.get("components", {})
    
    # 统计状态
    needs_regeneration = []
    needs_human = []
    
    for comp_id, comp_data in components.items():
        status = comp_data.get("status", "")
        retry_count = comp_data.get("retry_count", 0)
        
        if status == "rejected" and retry_count < 3:
            needs_regeneration.append(comp_id)
        elif status == "human_reviewing":
            needs_human.append(comp_id)
    
    # 决策优先级
    if needs_human:
        return "human_wait"  # 需要人工
    elif needs_regeneration:
        return "regenerate"  # ⭐ 循环回去
    else:
        return "continue"  # 继续


def page_migration_router(state: MigrationGraphState) -> Literal["migrate_next", "complete"]:
    """页面迁移路由"""
    page_queue = state.get("page_queue", [])
    return "migrate_next" if page_queue else "complete"


def _review_settings(state: MigrationGraphState) -> Dict[str, Any]:
    config = state.get("config", {})
    review = config.get("review", {})
    return review if isinstance(review, dict) else {}


def _review_enabled(state: MigrationGraphState, review_name: str) -> bool:
    review = _review_settings(state)
    enabled = review.get("enabled")
    if isinstance(enabled, dict) and review_name in enabled:
        return bool(enabled.get(review_name))
    return True


def _runtime_check_enabled(state: MigrationGraphState) -> bool:
    if not _review_enabled(state, "runtime_check"):
        return False
    review = _review_settings(state)
    runtime_check = review.get("runtime_check", {})
    if not isinstance(runtime_check, dict):
        runtime_check = {}
    command_value = runtime_check.get("command", "")
    command = str(command_value).strip() if command_value is not None else ""
    enabled_value = runtime_check.get("enabled")
    enabled = bool(command) if enabled_value is None else bool(enabled_value)
    return bool(command) and enabled


def _review_strategy(state: MigrationGraphState, review_name: str) -> Optional[str]:
    review = _review_settings(state)
    strategy = review.get("strategy")
    if isinstance(strategy, dict):
        value = strategy.get(review_name)
        return str(value).lower() if value else None
    if isinstance(strategy, str) and review_name == "code_quality":
        return strategy.lower()
    return None


def _should_skip_review(state: MigrationGraphState) -> bool:
    config = state.get("config", {})
    review = _review_settings(state)
    if review.get("skip") is True:
        return True
    if config.get("auto_approve_all"):
        return True
    if not any(
        (_runtime_check_enabled(state) if name == "runtime_check" else _review_enabled(state, name))
        for name in (
            "code_quality",
            "bdl_compliance",
            "function_parity",
            "accessibility",
            "security",
            "editor_schema",
            "runtime_check",
        )
    ):
        return True
    return os.getenv("MIGRATION_SKIP_REVIEW", "").lower() in {"1", "true", "yes"}


def route_to_parallel_reviews(state: MigrationGraphState) -> List[Send]:
    """Send API ?? - ????"""
    if _should_skip_review(state):
        return [Send("merge_review_results", state)]

    components = state.get("components", {})
    has_components = any(
        comp_data.get("status") == "generating"
        for comp_data in components.values()
    )

    if not has_components:
        return [Send("merge_review_results", state)]

    review_nodes: List[str] = []
    if _review_enabled(state, "code_quality"):
        strategy = _review_strategy(state, "code_quality")
        if strategy == "pipeline":
            review_nodes.append("code_quality_review")
        else:
            review_nodes.append("code_quality_review_agent")
    if _review_enabled(state, "bdl_compliance"):
        review_nodes.append("bdl_compliance_review")
    if _review_enabled(state, "function_parity"):
        review_nodes.append("function_parity_review")
    if _review_enabled(state, "accessibility"):
        review_nodes.append("accessibility_review")
    if _review_enabled(state, "security"):
        review_nodes.append("security_review")
    if _review_enabled(state, "editor_schema"):
        review_nodes.append("editor_schema_review")
    if _runtime_check_enabled(state):
        review_nodes.append("runtime_check_review")

    if not review_nodes:
        return [Send("merge_review_results", state)]

    return [Send(node, state) for node in review_nodes]

def create_component_conversion_subgraph_hybrid() -> StateGraph:
    """
    组件转换子图 - 混合架构
    
    ⭐ Agent 节点:
    - map_to_bdl → bdl_mapping_agent
    - generate_react → code_generator_agent
    
    Pipeline 节点:
    - ingest_source, parse_aem, analyze_component, transform_logic
    """
    subgraph = StateGraph(MigrationGraphState)
    
    # Pipeline 节点
    subgraph.add_node("ingest_source", ingest_source_node)
    subgraph.add_node("parse_aem", parse_aem_node)
    subgraph.add_node("analyze_component", analyze_component_node)
    
    # ⭐ Agent 节点
    subgraph.add_node("map_to_bdl_agent", bdl_mapping_agent_node)
    
    # Pipeline 节点
    subgraph.add_node("transform_logic", transform_logic_node)
    
    # ⭐ Agent 节点
    subgraph.add_node("generate_react_agent", code_generator_agent_node)
    
    # 流程
    subgraph.add_edge(START, "ingest_source")
    subgraph.add_edge("ingest_source", "parse_aem")
    subgraph.add_conditional_edges(
        "parse_aem",
        should_continue_parsing,
        {
            "parse_next": "parse_aem",
            "analyze_all": "analyze_component"
        }
    )
    subgraph.add_edge("analyze_component", "map_to_bdl_agent")  # ⭐ Agent
    subgraph.add_edge("map_to_bdl_agent", "transform_logic")
    subgraph.add_edge("transform_logic", "generate_react_agent")  # ⭐ Agent
    subgraph.add_edge("generate_react_agent", END)
    
    return subgraph


# ============================================================================
# 子图定义 - Config Generation (混合)
# ============================================================================

def create_config_generation_subgraph_hybrid() -> StateGraph:
    """
    配置生成子图 - 混合架构
    
    ⭐ Agent 节点:
    - analyze_editables → editor_designer_agent
    
    Pipeline 节点:
    - extract_props, generate_schema, validate_config
    """
    subgraph = StateGraph(MigrationGraphState)
    
    # Pipeline 节点
    subgraph.add_node("extract_props", extract_props_node)
    
    # ⭐ Agent 节点
    subgraph.add_node("analyze_editables_agent", editor_designer_agent_node)
    
    # Pipeline 节点
    subgraph.add_node("generate_schema", generate_schema_node)
    subgraph.add_node("validate_config", validate_config_node)
    
    # 流程
    subgraph.add_edge(START, "extract_props")
    subgraph.add_edge("extract_props", "analyze_editables_agent")  # ⭐ Agent
    subgraph.add_edge("analyze_editables_agent", "generate_schema")
    subgraph.add_edge("generate_schema", "validate_config")
    subgraph.add_edge("validate_config", END)
    
    return subgraph


# ============================================================================
# 子图定义 - Review System (混合)
# ============================================================================

def create_review_subgraph_hybrid() -> StateGraph:
    """
    审查子图 - 混合架构
    
    ⭐ Agent 节点:
    - code_quality_review → code_reviewer_agent
    
    Pipeline 节点:
    - bdl_compliance_review, function_parity_review
    """
    subgraph = StateGraph(MigrationGraphState)
    
    # Send API 节点
    subgraph.add_node("distribute_reviews", distribute_reviews_node)
    
    # ⭐ 1 个 Agent + 2 个 Pipeline（混合）
    subgraph.add_node("code_quality_review_agent", code_reviewer_agent_node)  # ⭐ Agent
    subgraph.add_node("code_quality_review", code_quality_review_node)  # Pipeline
    subgraph.add_node("bdl_compliance_review", bdl_compliance_review_node)    # Pipeline
    subgraph.add_node("function_parity_review", function_parity_review_node)  # Pipeline
    subgraph.add_node("accessibility_review", accessibility_review_node)  # Pipeline
    subgraph.add_node("security_review", security_review_node)  # Pipeline
    subgraph.add_node("editor_schema_review", editor_schema_review_node)  # Pipeline
    subgraph.add_node("runtime_check_review", runtime_check_review_node)  # Pipeline
    
    subgraph.add_node("merge_review_results", merge_review_results_node)
    subgraph.add_node("aggregate_reviews", aggregate_reviews_node)
    
    # 人工审查节点
    subgraph.add_node("prepare_human_review", prepare_human_review_node)
    subgraph.add_node("human_review", human_review_node)
    subgraph.add_node("process_human_decision", process_human_decision_node)
    
    # 后续处理
    subgraph.add_node("auto_fix", auto_fix_node)
    subgraph.add_node("auto_approve", auto_approve_node)
    subgraph.add_node("handle_rejection", handle_rejection_node)
    subgraph.add_node("apply_modifications", apply_modifications_node)
    
    # 流程
    subgraph.add_edge(START, "distribute_reviews")
    
    subgraph.add_conditional_edges(
        "distribute_reviews",
        route_to_parallel_reviews,
        [
            "code_quality_review_agent",
            "code_quality_review",
            "bdl_compliance_review",
            "function_parity_review",
            "accessibility_review",
            "security_review",
            "editor_schema_review",
            "runtime_check_review",
            "merge_review_results",
        ]
    )
    
    subgraph.add_edge("code_quality_review_agent", "merge_review_results")
    subgraph.add_edge("code_quality_review", "merge_review_results")
    subgraph.add_edge("bdl_compliance_review", "merge_review_results")
    subgraph.add_edge("function_parity_review", "merge_review_results")
    subgraph.add_edge("accessibility_review", "merge_review_results")
    subgraph.add_edge("security_review", "merge_review_results")
    subgraph.add_edge("editor_schema_review", "merge_review_results")
    subgraph.add_edge("runtime_check_review", "merge_review_results")
    
    subgraph.add_edge("merge_review_results", "aggregate_reviews")
    
    subgraph.add_conditional_edges(
        "aggregate_reviews",
        review_decision_router,
        {
            "human_review": "prepare_human_review",
            "auto_fix": "auto_fix",
            "auto_approve": "auto_approve",
            "regenerate": "handle_rejection"
        }
    )
    
    subgraph.add_edge("prepare_human_review", "human_review")
    subgraph.add_edge("human_review", "process_human_decision")
    
    subgraph.add_conditional_edges(
        "process_human_decision",
        human_review_router,
        {
            "approve": "auto_approve",
            "reject": "handle_rejection",
            "modify": "apply_modifications",
            "escalate": END
        }
    )
    
    subgraph.add_edge("auto_fix", "distribute_reviews")
    subgraph.add_edge("auto_approve", END)
    subgraph.add_edge("handle_rejection", END)
    subgraph.add_edge("apply_modifications", END)
    
    return subgraph


# ============================================================================
# 主图 - 混合架构（⭐ 支持循环）
# ============================================================================

def create_main_hybrid_graph() -> StateGraph:
    """
    主图 - 混合架构
    
    ⭐ 新特性:
    1. 5 个 Agent 节点（智能决策和迭代）
    2. 支持循环（Review → Generate）
    3. 反馈机制（节点间通信）
    4. 27 个 Pipeline 节点（确定性流程）
    """
    graph = StateGraph(MigrationGraphState)
    
    # 初始化节点
    graph.add_node("initialize", initialize_node)
    graph.add_node("load_bdl_spec", load_bdl_spec_node)
    
    # 子图
    component_subgraph = create_component_conversion_subgraph_hybrid()  # ⭐ 含 Agent
    config_subgraph = create_config_generation_subgraph_hybrid()        # ⭐ 含 Agent
    review_subgraph = create_review_subgraph_hybrid()                   # ⭐ 含 Agent
    
    # Pipeline 子图
    from src.core.graph import create_page_migration_subgraph
    page_subgraph = create_page_migration_subgraph()
    
    graph.add_node("component_conversion", component_subgraph.compile())
    graph.add_node("config_generation", config_subgraph.compile())
    graph.add_node("review_system", review_subgraph.compile(interrupt_before=["human_review"]))
    graph.add_node("page_migration", page_subgraph.compile())
    
    # 最终节点
    graph.add_node("finalize", finalize_node)
    graph.add_node("generate_report", generate_report_node)
    
    # 流程
    graph.add_edge(START, "initialize")
    graph.add_edge("initialize", "load_bdl_spec")
    graph.add_edge("load_bdl_spec", "component_conversion")
    graph.add_edge("component_conversion", "config_generation")
    
    # ⭐⭐⭐ 关键: Review 后支持循环
    graph.add_conditional_edges(
        "review_system",
        after_review_router,
        {
            "regenerate": "component_conversion",  # ⭐ 循环边！
            "continue": "page_migration",
            "human_wait": END  # 人工审查中断
        }
    )
    
    graph.add_edge("page_migration", "finalize")
    graph.add_edge("finalize", "generate_report")
    graph.add_edge("generate_report", END)
    
    return graph


# ============================================================================
# 编译函数
# ============================================================================

def compile_hybrid_graph(checkpointer=None, debug: bool = False):
    """
    编译混合架构图
    
    ⭐ 使用此函数替代原 compile_graph
    
    Args:
        checkpointer: 状态检查点（默认 MemorySaver）
        debug: 调试模式
    
    Returns:
        编译后的图
    """
    graph = create_main_hybrid_graph()
    
    if checkpointer is None:
        checkpointer = MemorySaver()
    
    compiled = graph.compile(
        checkpointer=checkpointer,
        debug=debug,
    )
    
    return compiled


# ============================================================================
# 节点实现占位符 - 导入现有节点
# ============================================================================

def initialize_node(state: MigrationGraphState) -> Dict[str, Any]:
    from src.nodes.pipeline.initialization import initialize
    return initialize(state)

def load_bdl_spec_node(state: MigrationGraphState) -> Dict[str, Any]:
    from src.nodes.pipeline.initialization import load_bdl_spec
    return load_bdl_spec(state)

def ingest_source_node(state: MigrationGraphState) -> Dict[str, Any]:
    from src.nodes.pipeline.component_conversion import ingest_source
    return ingest_source(state)

async def parse_aem_node(state: MigrationGraphState) -> Dict[str, Any]:
    from src.nodes.pipeline.component_conversion import parse_aem
    return await parse_aem(state)

async def analyze_component_node(state: MigrationGraphState) -> Dict[str, Any]:
    from src.nodes.pipeline.component_conversion import analyze_component
    return await analyze_component(state)

async def transform_logic_node(state: MigrationGraphState) -> Dict[str, Any]:
    from src.nodes.pipeline.component_conversion import transform_logic
    return await transform_logic(state)

# ⭐ Agent 节点
async def bdl_mapping_agent_node(state: MigrationGraphState) -> Dict[str, Any]:
    from src.nodes.intelligent.bdl_mapping import bdl_mapping_node
    return await bdl_mapping_node(state)

async def code_generator_agent_node(state: MigrationGraphState) -> Dict[str, Any]:
    from src.nodes.intelligent.code_generation import code_generation_node
    return await code_generation_node(state)

async def extract_props_node(state: MigrationGraphState) -> Dict[str, Any]:
    from src.nodes.pipeline.config_generation import extract_props
    return await extract_props(state)

async def editor_designer_agent_node(state: MigrationGraphState) -> Dict[str, Any]:
    from src.nodes.intelligent.editor_design import editor_design_node
    return await editor_design_node(state)

def generate_schema_node(state: MigrationGraphState) -> Dict[str, Any]:
    from src.nodes.pipeline.config_generation import generate_schema
    return generate_schema(state)

def validate_config_node(state: MigrationGraphState) -> Dict[str, Any]:
    from src.nodes.pipeline.config_generation import validate_config
    return validate_config(state)

# Review 节点
async def distribute_reviews_node(state: MigrationGraphState) -> Dict[str, Any]:
    from src.nodes.pipeline.review import distribute_reviews_node as impl
    return await impl(state)

async def code_reviewer_agent_node(state: MigrationGraphState) -> Dict[str, Any]:
    """⭐ Agent 审查节点"""
    from src.nodes.intelligent.code_review import code_review_node
    return await code_review_node(state)


async def code_quality_review_node(state: MigrationGraphState) -> Dict[str, Any]:
    """Pipeline ????"""
    from src.nodes.pipeline.review import code_quality_review_v2
    return await code_quality_review_v2(state)


async def accessibility_review_node(state: MigrationGraphState) -> Dict[str, Any]:
    """Pipeline ????"""
    from src.nodes.pipeline.review import accessibility_review_v2
    return await accessibility_review_v2(state)

async def security_review_node(state: MigrationGraphState) -> Dict[str, Any]:
    """Pipeline ????"""
    from src.nodes.pipeline.review import security_review_v2
    return await security_review_v2(state)

async def editor_schema_review_node(state: MigrationGraphState) -> Dict[str, Any]:
    """Pipeline ????"""
    from src.nodes.pipeline.review import editor_schema_review_v2
    return await editor_schema_review_v2(state)

async def runtime_check_review_node(state: MigrationGraphState) -> Dict[str, Any]:
    """Pipeline ????"""
    from src.nodes.pipeline.review import runtime_check_review_v2
    return await runtime_check_review_v2(state)
async def bdl_compliance_review_node(state: MigrationGraphState) -> Dict[str, Any]:
    """Pipeline 审查节点（或未来也可改为 Agent）"""
    from src.nodes.pipeline.review import bdl_compliance_review_v2
    return await bdl_compliance_review_v2(state)

async def function_parity_review_node(state: MigrationGraphState) -> Dict[str, Any]:
    """Pipeline 审查节点"""
    from src.nodes.pipeline.review import function_parity_review_v2
    return await function_parity_review_v2(state)

def merge_review_results_node(state: MigrationGraphState) -> Dict[str, Any]:
    from src.nodes.pipeline.review import merge_review_results_node as impl
    return impl(state)

def aggregate_reviews_node(state: MigrationGraphState) -> Dict[str, Any]:
    from src.nodes.pipeline.review import aggregate_reviews
    return aggregate_reviews(state)

def prepare_human_review_node(state: MigrationGraphState) -> Dict[str, Any]:
    from src.nodes.pipeline.review import prepare_human_review
    return prepare_human_review(state)

def human_review_node(state: MigrationGraphState) -> Dict[str, Any]:
    from src.nodes.pipeline.review import human_review
    return human_review(state)

def process_human_decision_node(state: MigrationGraphState) -> Dict[str, Any]:
    from src.nodes.pipeline.review import process_human_decision
    return process_human_decision(state)

def auto_approve_node(state: MigrationGraphState) -> Dict[str, Any]:
    from src.nodes.pipeline.review import auto_approve
    return auto_approve(state)


async def auto_fix_node(state: MigrationGraphState) -> Dict[str, Any]:
    """Auto fix"""
    from src.nodes.intelligent.code_fix import code_fix_node
    return await code_fix_node(state)

def handle_rejection_node(state: MigrationGraphState) -> Dict[str, Any]:
    from src.nodes.pipeline.review import handle_rejection
    return handle_rejection(state)

def apply_modifications_node(state: MigrationGraphState) -> Dict[str, Any]:
    from src.nodes.pipeline.review import apply_modifications
    return apply_modifications(state)

def review_decision_router(state: MigrationGraphState) -> str:
    """Review 决策路由"""
    pending = state.get("pending_human_review", [])
    if pending:
        return "human_review"

    pending_auto_fix = state.get("pending_auto_fix", [])
    if pending_auto_fix:
        return "auto_fix"
    
    components = state.get("components", {})
    for comp_data in components.values():
        if comp_data.get("status") == "rejected":
            return "regenerate"
        review = comp_data.get("review", {})
        aggregated = review.get("aggregated", {})
        if aggregated.get("decision") == "regenerate":
            return "regenerate"
        if review.get("human_decision") == "reject":
            return "regenerate"
    
    return "auto_approve"

def human_review_router(state: MigrationGraphState) -> str:
    """人工审查路由"""
    decisions = state.get("human_review_decisions", {})
    current_id = state.get("current_component_id")
    
    if current_id and current_id in decisions:
        decision = decisions[current_id].get("decision", "approve")
        return decision
    
    return "approve"

def finalize_node(state: MigrationGraphState) -> Dict[str, Any]:
    from src.nodes.pipeline.finalization import finalize
    return finalize(state)

def generate_report_node(state: MigrationGraphState) -> Dict[str, Any]:
    from src.nodes.pipeline.finalization import generate_report
    return generate_report(state)
