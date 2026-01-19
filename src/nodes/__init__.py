"""
节点模块 - 统一导出
所有 LangGraph 节点的统一入口

架构:
├── pipeline/        # Pipeline 节点（确定性流程）
│   - 直接 LLM 调用
│   - 快速、可预测
│   - 适合解析、转换等任务
│
└── intelligent/     # Intelligent 节点（智能决策）
    - ReAct Agent 循环
    - 使用工具验证
    - 适合映射、生成、审查等任务

使用方式:
    # Pipeline 节点
    from src.nodes import parse_aem, analyze_component
    
    # Intelligent 节点
    from src.nodes import bdl_mapping_node, code_generation_node
    
    # 或按类型导入
    from src.nodes.pipeline import parse_aem
    from src.nodes.intelligent import bdl_mapping_node
"""

# ============================================================================
# Pipeline 节点
# ============================================================================

from src.nodes.pipeline.component_conversion import (
    ingest_source,
    parse_aem,
    analyze_component,
    map_to_bdl,
    transform_logic,
    generate_react,
)

from src.nodes.pipeline.config_generation import (
    extract_props,
    analyze_editables,
    generate_schema,
    validate_config,
)

from src.nodes.pipeline.page_migration import (
    parse_aem_json,
    map_page_components,
    transform_structure,
    generate_cms_json,
    validate_page,
)

from src.nodes.pipeline.review import (
    distribute_reviews_node,
    code_quality_review_v2,
    bdl_compliance_review_v2,
    function_parity_review_v2,
    accessibility_review_v2,
    security_review_v2,
    editor_schema_review_v2,
    runtime_check_review_v2,
    merge_review_results_node,
    aggregate_reviews,
    prepare_human_review,
    human_review,
    process_human_decision,
    auto_approve,
    handle_rejection,
    apply_modifications,
)

from src.nodes.pipeline.initialization import (
    initialize,
    load_bdl_spec,
)

from src.nodes.pipeline.finalization import (
    finalize,
    generate_report,
)


# ============================================================================
# Intelligent 节点（使用 Agent）
# ============================================================================

from src.nodes.intelligent.bdl_mapping import bdl_mapping_node
from src.nodes.intelligent.code_fix import code_fix_node
from src.nodes.intelligent.code_generation import code_generation_node
from src.nodes.intelligent.code_review import code_review_node
from src.nodes.intelligent.editor_design import editor_design_node


# ============================================================================
# 统一导出
# ============================================================================

__all__ = [
    # ========== Pipeline 节点 ==========
    # Component Conversion
    "ingest_source",
    "parse_aem",
    "analyze_component",
    "map_to_bdl",
    "transform_logic",
    "generate_react",
    
    # Config Generation
    "extract_props",
    "analyze_editables",
    "generate_schema",
    "validate_config",
    
    # Page Migration
    "parse_aem_json",
    "map_page_components",
    "transform_structure",
    "generate_cms_json",
    "validate_page",
    
    # Review
    "distribute_reviews_node",
    "code_quality_review_v2",
    "bdl_compliance_review_v2",
    "function_parity_review_v2",
    "accessibility_review_v2",
    "security_review_v2",
    "editor_schema_review_v2",
    "runtime_check_review_v2",
    "merge_review_results_node",
    "aggregate_reviews",
    "prepare_human_review",
    "human_review",
    "process_human_decision",
    "auto_approve",
    "handle_rejection",
    "apply_modifications",
    
    # Initialization & Finalization
    "initialize",
    "load_bdl_spec",
    "finalize",
    "generate_report",
    
    # ========== Intelligent 节点 ==========
    "bdl_mapping_node",
    "code_fix_node",
    "code_generation_node",
    "code_review_node",
    "editor_design_node",
]
