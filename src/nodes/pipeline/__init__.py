"""
Pipeline 节点模块
确定性流程节点，使用直接 LLM 调用

特点:
- 简单直接的 LLM 调用
- 确定性输出
- 快速执行
- 不需要工具和迭代
"""

# Component Conversion
from src.nodes.pipeline.component_conversion import (
    ingest_source,
    parse_aem,
    analyze_component,
    map_to_bdl,
    transform_logic,
    generate_react,
)

# Config Generation
from src.nodes.pipeline.config_generation import (
    extract_props,
    analyze_editables,
    generate_schema,
    validate_config,
)

# Page Migration
from src.nodes.pipeline.page_migration import (
    parse_aem_json,
    map_page_components,
    transform_structure,
    generate_cms_json,
    validate_page,
)

# Review
from src.nodes.pipeline.review import (
    distribute_reviews_node,
    code_quality_review_v2,
    bdl_compliance_review_v2,
    function_parity_review_v2,
    merge_review_results_node,
    aggregate_reviews,
    prepare_human_review,
    human_review,
    process_human_decision,
    auto_approve,
    handle_rejection,
    apply_modifications,
)

# Initialization
from src.nodes.pipeline.initialization import (
    initialize,
    load_bdl_spec,
)

# Finalization
from src.nodes.pipeline.finalization import (
    finalize,
    generate_report,
)


__all__ = [
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
    "merge_review_results_node",
    "aggregate_reviews",
    "prepare_human_review",
    "human_review",
    "process_human_decision",
    "auto_approve",
    "handle_rejection",
    "apply_modifications",
    # Initialization
    "initialize",
    "load_bdl_spec",
    # Finalization
    "finalize",
    "generate_report",
]
