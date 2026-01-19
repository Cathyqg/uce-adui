"""
uce-adui - LangGraph 图定义
定义完整的多智能体工作流图

LangGraph 1.0 Best Practices:
1. 使用 StateGraph 和明确的状态类型
2. 使用 interrupt_before/after 进行人工介入
3. 子图组合实现模块化
4. 使用 checkpointer 支持持久化和恢复
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Sequence

from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import Send
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from src.core.state import MigrationGraphState, Phase


# ============================================================================
# 条件边函数 - 决定下一个节点
# ============================================================================

def should_continue_parsing(state: MigrationGraphState) -> Literal["parse_next", "analyze_all"]:
    """判断是否继续解析更多组件"""
    if state.get("component_queue"):
        return "parse_next"
    return "analyze_all"


def should_continue_generating(state: MigrationGraphState) -> Literal["generate_next", "start_config"]:
    """判断是否继续生成更多组件"""
    components = state.get("components", {})
    for comp_id, comp_data in components.items():
        if comp_data.get("status") == "mapping":
            return "generate_next"
    return "start_config"


def review_decision_router(state: MigrationGraphState) -> Literal["human_review", "auto_fix", "auto_approve", "regenerate"]:
    """根据审查结果决定下一步"""
    pending = state.get("pending_human_review", [])
    if pending:
        return "human_review"

    pending_auto_fix = state.get("pending_auto_fix", [])
    if pending_auto_fix:
        return "auto_fix"
    
    # 检查是否有需要重新生成的
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


def human_review_router(state: MigrationGraphState) -> Literal["approve", "reject", "modify", "escalate"]:
    """人工审查后的路由"""
    decisions = state.get("human_review_decisions", {})
    current_id = state.get("current_component_id")
    
    if current_id and current_id in decisions:
        decision = decisions[current_id].get("decision", "approve")
        if decision == "approve":
            return "approve"
        elif decision == "reject":
            return "reject"
        elif decision == "modify":
            return "modify"
        else:
            return "escalate"
    
    return "approve"


def page_migration_router(state: MigrationGraphState) -> Literal["migrate_next", "complete"]:
    """页面迁移路由"""
    page_queue = state.get("page_queue", [])
    if page_queue:
        return "migrate_next"
    return "complete"


# ============================================================================
# 子图定义 - Component Conversion Subgraph
# ============================================================================

def create_component_conversion_subgraph() -> StateGraph:
    """创建组件转换子图"""
    
    subgraph = StateGraph(MigrationGraphState)
    
    # 添加节点
    subgraph.add_node("ingest_source", ingest_source_node)
    subgraph.add_node("parse_aem", parse_aem_node)
    subgraph.add_node("analyze_component", analyze_component_node)
    subgraph.add_node("map_to_bdl", map_to_bdl_node)
    subgraph.add_node("transform_logic", transform_logic_node)
    subgraph.add_node("generate_react", generate_react_node)
    
    # 定义边
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
    subgraph.add_edge("analyze_component", "map_to_bdl")
    subgraph.add_edge("map_to_bdl", "transform_logic")
    subgraph.add_edge("transform_logic", "generate_react")
    subgraph.add_edge("generate_react", END)
    
    return subgraph


# ============================================================================
# 子图定义 - Config Generation Subgraph
# ============================================================================

def create_config_generation_subgraph() -> StateGraph:
    """创建配置生成子图"""
    
    subgraph = StateGraph(MigrationGraphState)
    
    subgraph.add_node("extract_props", extract_props_node)
    subgraph.add_node("analyze_editables", analyze_editables_node)
    subgraph.add_node("generate_schema", generate_schema_node)
    subgraph.add_node("validate_config", validate_config_node)
    
    subgraph.add_edge(START, "extract_props")
    subgraph.add_edge("extract_props", "analyze_editables")
    subgraph.add_edge("analyze_editables", "generate_schema")
    subgraph.add_edge("generate_schema", "validate_config")
    subgraph.add_edge("validate_config", END)
    
    return subgraph


# ============================================================================
# 子图定义 - Review Subgraph (Send API 模式)
# ============================================================================

def create_review_subgraph() -> StateGraph:
    """
    创建审查子图 - 使用 LangGraph Send API 实现图级并行
    
    LangGraph 1.0+ Send API 模式:
    ═══════════════════════════════════════════════════════════
    
                    ┌─────────────────────────┐
                    │   distribute_reviews    │  (准备分发)
                    └─────────────────────────┘
                           │
                    ┌──────┴──────┐
                    │  Send API   │  (返回多个 Send 对象)
                    └──────┬──────┘
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │  code_   │ │   bdl_   │ │function_ │  ← 三个节点图级并行
        │ quality  │ │compliance│ │ parity   │    LangSmith 可追踪
        └──────────┘ └──────────┘ └──────────┘
              │            │            │
              └────────────┼────────────┘
                           │
                    ┌──────┴──────┐
                    │   Reducer   │  (自动合并状态)
                    └──────┬──────┘
                           ▼
                    ┌─────────────────────────┐
                    │  merge_review_results   │  (写入组件)
                    └─────────────────────────┘
                           │
                           ▼
                    ┌─────────────────────────┐
                    │   aggregate_reviews     │  (决策)
                    └─────────────────────────┘
    
    优势:
    1. 图级并行 - LangSmith 可以看到每个审查节点的执行
    2. 自动状态合并 - 使用 Reducer 自动合并并行结果
    3. 更好的错误隔离 - 单个审查失败不影响其他
    4. 更好的可观测性 - 每个节点独立追踪
    """
    
    subgraph = StateGraph(MigrationGraphState)
    
    # === Send API 并行审查节点 ===
    subgraph.add_node("distribute_reviews", distribute_reviews_node)
    subgraph.add_node("code_quality_review", code_quality_review_v2_node)
    subgraph.add_node("bdl_compliance_review", bdl_compliance_review_v2_node)
    subgraph.add_node("function_parity_review", function_parity_review_v2_node)
    subgraph.add_node("accessibility_review", accessibility_review_v2_node)
    subgraph.add_node("security_review", security_review_v2_node)
    subgraph.add_node("editor_schema_review", editor_schema_review_v2_node)
    subgraph.add_node("runtime_check_review", runtime_check_review_v2_node)
    subgraph.add_node("merge_review_results", merge_review_results_node)
    
    # === 聚合与决策节点 ===
    subgraph.add_node("aggregate_reviews", aggregate_reviews_node)
    
    # === 人工审查节点 ===
    subgraph.add_node("prepare_human_review", prepare_human_review_node)
    subgraph.add_node("human_review", human_review_node)  # 会被中断
    subgraph.add_node("process_human_decision", process_human_decision_node)
    
    # === 后续处理节点 ===
    subgraph.add_node("auto_fix", auto_fix_node)
    subgraph.add_node("auto_approve", auto_approve_node)
    subgraph.add_node("handle_rejection", handle_rejection_node)
    subgraph.add_node("apply_modifications", apply_modifications_node)
    
    # === 流程定义 ===
    
    # 1. 开始 → 分发节点
    subgraph.add_edge(START, "distribute_reviews")
    
    # 2. 分发节点 → Send API 并行分发到三个审查节点
    subgraph.add_conditional_edges(
        "distribute_reviews",
        route_to_parallel_reviews,  # 返回 [Send("code_quality_review"), Send("bdl_compliance_review"), Send("function_parity_review")]
        [
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
    
    # 3. 三个并行审查节点 → 合并节点 (LangGraph 自动等待所有并行节点完成)
    subgraph.add_edge("code_quality_review", "merge_review_results")
    subgraph.add_edge("bdl_compliance_review", "merge_review_results")
    subgraph.add_edge("function_parity_review", "merge_review_results")
    subgraph.add_edge("accessibility_review", "merge_review_results")
    subgraph.add_edge("security_review", "merge_review_results")
    subgraph.add_edge("editor_schema_review", "merge_review_results")
    subgraph.add_edge("runtime_check_review", "merge_review_results")
    
    # 4. 合并节点 → 聚合决策
    subgraph.add_edge("merge_review_results", "aggregate_reviews")
    
    # 5. 聚合后的条件路由
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
    
    # 6. 人工审查流程
    subgraph.add_edge("prepare_human_review", "human_review")
    subgraph.add_edge("human_review", "process_human_decision")
    
    # 7. 人工决定后的路由
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
    
    # 8. 终止节点
    subgraph.add_edge("auto_fix", "distribute_reviews")
    subgraph.add_edge("auto_approve", END)
    subgraph.add_edge("handle_rejection", END)
    subgraph.add_edge("apply_modifications", END)
    
    return subgraph


def route_to_parallel_reviews(state: MigrationGraphState) -> List[Send]:
    """
    Send API 路由函数 - 返回 Send 对象列表实现 Fan-out
    
    LangGraph 会自动:
    1. 并行执行返回的所有 Send 目标节点
    2. 等待所有节点完成
    3. 使用 Reducer 合并各节点的状态更新
    """
    from src.nodes.pipeline.review import route_to_parallel_reviews as impl

    return impl(state)


# ============================================================================
# 子图定义 - Page Migration Subgraph
# ============================================================================

def create_page_migration_subgraph() -> StateGraph:
    """创建页面迁移子图"""
    
    subgraph = StateGraph(MigrationGraphState)
    
    subgraph.add_node("parse_aem_json", parse_aem_json_node)
    subgraph.add_node("map_page_components", map_page_components_node)
    subgraph.add_node("transform_structure", transform_structure_node)
    subgraph.add_node("generate_cms_json", generate_cms_json_node)
    subgraph.add_node("validate_page", validate_page_node)
    
    subgraph.add_edge(START, "parse_aem_json")
    subgraph.add_edge("parse_aem_json", "map_page_components")
    subgraph.add_edge("map_page_components", "transform_structure")
    subgraph.add_edge("transform_structure", "generate_cms_json")
    subgraph.add_edge("generate_cms_json", "validate_page")
    subgraph.add_conditional_edges(
        "validate_page",
        page_migration_router,
        {
            "migrate_next": "parse_aem_json",
            "complete": END
        }
    )
    
    return subgraph


# ============================================================================
# 主图定义
# ============================================================================

def create_main_graph() -> StateGraph:
    """创建主工作流图"""
    
    graph = StateGraph(MigrationGraphState)
    
    # 初始化节点
    graph.add_node("initialize", initialize_node)
    graph.add_node("load_bdl_spec", load_bdl_spec_node)
    
    # 子图作为节点
    component_subgraph = create_component_conversion_subgraph()
    config_subgraph = create_config_generation_subgraph()
    review_subgraph = create_review_subgraph()
    page_subgraph = create_page_migration_subgraph()
    
    graph.add_node("component_conversion", component_subgraph.compile())
    graph.add_node("config_generation", config_subgraph.compile())
    graph.add_node("review_system", review_subgraph.compile(interrupt_before=["human_review"]))
    graph.add_node("page_migration", page_subgraph.compile())
    
    # 最终节点
    graph.add_node("finalize", finalize_node)
    graph.add_node("generate_report", generate_report_node)
    
    # 定义流程
    graph.add_edge(START, "initialize")
    graph.add_edge("initialize", "load_bdl_spec")
    graph.add_edge("load_bdl_spec", "component_conversion")
    graph.add_edge("component_conversion", "config_generation")
    graph.add_edge("config_generation", "review_system")
    graph.add_edge("review_system", "page_migration")
    graph.add_edge("page_migration", "finalize")
    graph.add_edge("finalize", "generate_report")
    graph.add_edge("generate_report", END)
    
    return graph


def compile_graph(checkpointer=None, debug: bool = False):
    """
    编译图并配置检查点
    
    LangGraph 1.0 Best Practices:
    1. 使用 checkpointer 支持状态持久化
    2. 使用 interrupt_before 实现 Human-in-the-Loop
    3. 支持 debug 模式便于开发调试
    
    Args:
        checkpointer: 状态检查点存储 (默认 MemorySaver)
        debug: 是否启用调试模式
    
    Returns:
        编译后的图
    """
    graph = create_main_graph()
    
    if checkpointer is None:
        checkpointer = MemorySaver()
    
    # 配置人工审查中断点 - LangGraph 1.0 标准方式
    compiled = graph.compile(
        checkpointer=checkpointer,
        debug=debug,  # 启用调试日志
    )
    
    return compiled


def create_graph_with_tracing(checkpointer=None):
    """
    创建带有 LangSmith tracing 的图
    
    LangGraph 1.0 Best Practice:
    - 集成 LangSmith 进行可观测性监控
    """
    import os
    
    # 启用 LangSmith tracing (如果配置了)
    if os.getenv("LANGCHAIN_TRACING_V2"):
        os.environ.setdefault("LANGCHAIN_PROJECT", "uce-adui")
    
    return compile_graph(checkpointer, debug=True)


# ============================================================================
# 节点实现占位符 - 实际实现在 nodes/ 目录
# ============================================================================

def initialize_node(state: MigrationGraphState) -> Dict[str, Any]:
    """初始化节点"""
    from src.nodes.pipeline.initialization import initialize
    return initialize(state)


def load_bdl_spec_node(state: MigrationGraphState) -> Dict[str, Any]:
    """加载 BDL 规范"""
    from src.nodes.pipeline.initialization import load_bdl_spec
    return load_bdl_spec(state)


def ingest_source_node(state: MigrationGraphState) -> Dict[str, Any]:
    """源码摄入"""
    from src.nodes.pipeline.component_conversion import ingest_source
    return ingest_source(state)


async def parse_aem_node(state: MigrationGraphState) -> Dict[str, Any]:
    """解析 AEM 组件"""
    from src.nodes.pipeline.component_conversion import parse_aem
    return await parse_aem(state)


async def analyze_component_node(state: MigrationGraphState) -> Dict[str, Any]:
    """分析组件"""
    from src.nodes.pipeline.component_conversion import analyze_component
    return await analyze_component(state)


async def map_to_bdl_node(state: MigrationGraphState) -> Dict[str, Any]:
    """映射到 BDL"""
    from src.nodes.pipeline.component_conversion import map_to_bdl
    return await map_to_bdl(state)


async def transform_logic_node(state: MigrationGraphState) -> Dict[str, Any]:
    """转换业务逻辑"""
    from src.nodes.pipeline.component_conversion import transform_logic
    return await transform_logic(state)


async def generate_react_node(state: MigrationGraphState) -> Dict[str, Any]:
    """生成 React 组件"""
    from src.nodes.pipeline.component_conversion import generate_react
    return await generate_react(state)


async def extract_props_node(state: MigrationGraphState) -> Dict[str, Any]:
    """提取 Props"""
    from src.nodes.pipeline.config_generation import extract_props
    return await extract_props(state)


async def analyze_editables_node(state: MigrationGraphState) -> Dict[str, Any]:
    """分析可编辑区域"""
    from src.nodes.pipeline.config_generation import analyze_editables
    return await analyze_editables(state)


def generate_schema_node(state: MigrationGraphState) -> Dict[str, Any]:
    """生成 Schema"""
    from src.nodes.pipeline.config_generation import generate_schema
    return generate_schema(state)


def validate_config_node(state: MigrationGraphState) -> Dict[str, Any]:
    """验证配置"""
    from src.nodes.pipeline.config_generation import validate_config
    return validate_config(state)


# ============================================================================
# Send API 并行审查节点实现
# ============================================================================

async def distribute_reviews_node(state: MigrationGraphState) -> Dict[str, Any]:
    """分发审查任务节点 - Fan-out 起点"""
    from src.nodes.pipeline.review import distribute_reviews_node as impl
    return await impl(state)


async def code_quality_review_v2_node(state: MigrationGraphState) -> Dict[str, Any]:
    """代码质量审查节点 (Send API 版本)"""
    from src.nodes.pipeline.review import code_quality_review_v2
    return await code_quality_review_v2(state)


async def bdl_compliance_review_v2_node(state: MigrationGraphState) -> Dict[str, Any]:
    """BDL 合规性审查节点 (Send API 版本)"""
    from src.nodes.pipeline.review import bdl_compliance_review_v2
    return await bdl_compliance_review_v2(state)


async def function_parity_review_v2_node(state: MigrationGraphState) -> Dict[str, Any]:
    """功能一致性审查节点 (Send API 版本)"""
    from src.nodes.pipeline.review import function_parity_review_v2
    return await function_parity_review_v2(state)

async def accessibility_review_v2_node(state: MigrationGraphState) -> Dict[str, Any]:
    """可访问性审查节点 (Send API 版本)"""
    from src.nodes.pipeline.review import accessibility_review_v2
    return await accessibility_review_v2(state)

async def security_review_v2_node(state: MigrationGraphState) -> Dict[str, Any]:
    """安全审查节点 (Send API 版本)"""
    from src.nodes.pipeline.review import security_review_v2
    return await security_review_v2(state)

async def editor_schema_review_v2_node(state: MigrationGraphState) -> Dict[str, Any]:
    """编辑器 Schema 审查节点 (Send API 版本)"""
    from src.nodes.pipeline.review import editor_schema_review_v2
    return await editor_schema_review_v2(state)


async def runtime_check_review_v2_node(state: MigrationGraphState) -> Dict[str, Any]:
    """运行检查节点 (Send API 版本)"""
    from src.nodes.pipeline.review import runtime_check_review_v2
    return await runtime_check_review_v2(state)


def merge_review_results_node(state: MigrationGraphState) -> Dict[str, Any]:
    """合并审查结果节点 - Fan-in 终点"""
    from src.nodes.pipeline.review import merge_review_results_node as impl
    return impl(state)


def aggregate_reviews_node(state: MigrationGraphState) -> Dict[str, Any]:
    """聚合审查结果"""
    from src.nodes.pipeline.review import aggregate_reviews
    return aggregate_reviews(state)


def prepare_human_review_node(state: MigrationGraphState) -> Dict[str, Any]:
    """准备人工审查"""
    from src.nodes.pipeline.review import prepare_human_review
    return prepare_human_review(state)


def human_review_node(state: MigrationGraphState) -> Dict[str, Any]:
    """人工审查节点 - 会被中断"""
    from src.nodes.pipeline.review import human_review
    return human_review(state)


def process_human_decision_node(state: MigrationGraphState) -> Dict[str, Any]:
    """处理人工决定"""
    from src.nodes.pipeline.review import process_human_decision
    return process_human_decision(state)


def auto_approve_node(state: MigrationGraphState) -> Dict[str, Any]:
    """自动批准"""
    from src.nodes.pipeline.review import auto_approve
    return auto_approve(state)



async def auto_fix_node(state: MigrationGraphState) -> Dict[str, Any]:
    """Auto fix"""
    from src.nodes.intelligent.code_fix import code_fix_node
    return await code_fix_node(state)
def handle_rejection_node(state: MigrationGraphState) -> Dict[str, Any]:
    """处理拒绝"""
    from src.nodes.pipeline.review import handle_rejection
    return handle_rejection(state)


def apply_modifications_node(state: MigrationGraphState) -> Dict[str, Any]:
    """应用修改"""
    from src.nodes.pipeline.review import apply_modifications
    return apply_modifications(state)


def parse_aem_json_node(state: MigrationGraphState) -> Dict[str, Any]:
    """解析 AEM JSON"""
    from src.nodes.pipeline.page_migration import parse_aem_json
    return parse_aem_json(state)


def map_page_components_node(state: MigrationGraphState) -> Dict[str, Any]:
    """映射页面组件"""
    from src.nodes.pipeline.page_migration import map_page_components
    return map_page_components(state)


def transform_structure_node(state: MigrationGraphState) -> Dict[str, Any]:
    """转换结构"""
    from src.nodes.pipeline.page_migration import transform_structure
    return transform_structure(state)


def generate_cms_json_node(state: MigrationGraphState) -> Dict[str, Any]:
    """生成 CMS JSON"""
    from src.nodes.pipeline.page_migration import generate_cms_json
    return generate_cms_json(state)


def validate_page_node(state: MigrationGraphState) -> Dict[str, Any]:
    """验证页面"""
    from src.nodes.pipeline.page_migration import validate_page
    return validate_page(state)


def finalize_node(state: MigrationGraphState) -> Dict[str, Any]:
    """最终处理"""
    from src.nodes.pipeline.finalization import finalize
    return finalize(state)


def generate_report_node(state: MigrationGraphState) -> Dict[str, Any]:
    """生成报告"""
    from src.nodes.pipeline.finalization import generate_report
    return generate_report(state)
