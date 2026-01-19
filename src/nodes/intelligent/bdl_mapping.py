"""
BDL 映射智能节点
使用 Agent 智能搜索和选择最佳 BDL 组件映射

从 src/agents/bdl_mapper.py 重构而来
"""
from typing import Any, Dict
import json
import asyncio

from langchain_core.messages import HumanMessage

from src.core.state import MigrationGraphState
from src.agents import (
    PromptTemplates,
    BDLMappingOutput,
    create_structured_agent,
    invoke_agent_with_retry,
    create_error_result,
)
from src.llm import get_llm
from src.tools import (
    search_bdl_components,
    get_bdl_component_spec,
    list_bdl_components,
    get_bdl_design_token,
)
# ============================================================================
# 辅助函数
# ============================================================================

def _create_bdl_spec_summary(bdl_spec: Dict[str, Any]) -> str:
    """
    创建 BDL spec 的索引摘要，而不是硬截断
    
    生成一个精简的索引，包含：
    - 组件名称列表
    - 每个组件的简短描述
    - 常用 props 和 variants 的索引
    
    这样能在有限字符内提供最有价值的信息，避免截断导致 JSON 破损
    
    Args:
        bdl_spec: 完整的 BDL 规范
    
    Returns:
        索引摘要字符串
    """
    if not bdl_spec or not isinstance(bdl_spec, dict):
        return "No BDL spec available. Use tools to search and retrieve component specifications."
    
    summary_parts = ["# BDL Components Index\n"]
    
    # 提取组件列表
    components = bdl_spec.get("components", {})
    if not components:
        return "BDL spec loaded but no components found. Use search_bdl_components to find suitable components."
    
    summary_parts.append(f"**Total Components**: {len(components)}\n")
    summary_parts.append("\n## Available Components:\n")
    
    # 为每个组件创建简短的索引条目
    for comp_name, comp_spec in list(components.items())[:20]:  # 限制最多20个组件
        # 提取关键信息
        description = comp_spec.get("description", "No description")[:100]  # 限制描述长度
        
        # 提取 props 名称（不包括详细定义）
        props = comp_spec.get("props", {})
        prop_names = list(props.keys())[:5]  # 只列出前5个
        props_summary = ", ".join(prop_names)
        if len(props) > 5:
            props_summary += f" ... (+{len(props) - 5} more)"
        
        # 提取 variants 名称
        variants = comp_spec.get("variants", {})
        variant_names = list(variants.keys())[:3]  # 只列出前3个
        variants_summary = ", ".join(variant_names) if variant_names else "none"
        
        summary_parts.append(
            f"- **{comp_name}**: {description}\n"
            f"  - Props: {props_summary or 'none'}\n"
            f"  - Variants: {variants_summary}\n"
        )
    
    if len(components) > 20:
        summary_parts.append(f"\n... and {len(components) - 20} more components.\n")
    
    summary_parts.append(
        "\n**Note**: This is an index only. Use `get_bdl_component_spec(component_name)` "
        "to retrieve full specifications for any component.\n"
    )
    
    return "".join(summary_parts)


def _create_analyzed_component_summary(analyzed: Dict[str, Any]) -> str:
    """
    创建已分析组件的精简摘要
    
    Args:
        analyzed: 组件分析结果
    
    Returns:
        摘要字符串
    """
    if not analyzed:
        return "No analysis available."
    
    summary_parts = ["# Component Analysis Summary\n"]
    
    # 关键特征
    features = analyzed.get("features", {})
    if features:
        summary_parts.append("\n## Key Features:\n")
        for feature_name, feature_value in list(features.items())[:10]:
            summary_parts.append(f"- {feature_name}: {json.dumps(feature_value)[:50]}\n")
    
    # 复杂度
    complexity = analyzed.get("complexity", {})
    if complexity:
        summary_parts.append(f"\n## Complexity: {complexity.get('level', 'unknown')}\n")
    
    # 依赖
    dependencies = analyzed.get("dependencies", [])
    if dependencies:
        summary_parts.append(f"\n## Dependencies: {', '.join(dependencies[:5])}\n")
    
    return "".join(summary_parts)


# ============================================================================
# Agent 创建（内部函数）
# ============================================================================

def _create_bdl_mapper():
    """创建 BDL Mapper Agent（内部函数）"""
    llm = get_llm(task="analysis", temperature=0.1)
    
    tools = [
        search_bdl_components,
        get_bdl_component_spec,
        list_bdl_components,
        get_bdl_design_token,
    ]
    
    system_prompt = PromptTemplates.BDL_MAPPER
    
    return create_structured_agent(
        llm,
        tools,
        system_prompt=system_prompt,
        response_format=BDLMappingOutput,
    )


# ============================================================================
# 节点实现
# ============================================================================

async def _process_single_component(
    comp_id: str,
    comp_data: Dict[str, Any],
    bdl_spec: Dict[str, Any],
    agent,
    semaphore: asyncio.Semaphore,
) -> tuple[str, Dict[str, Any]]:
    """
    处理单个组件的映射（并发任务）
    
    Args:
        comp_id: 组件 ID
        comp_data: 组件数据
        bdl_spec: BDL 规范
        agent: Agent 实例
        semaphore: 并发控制信号量
    
    Returns:
        (comp_id, updated_comp_data) 元组
    """
    async with semaphore:  # 控制并发数量
        updated_data = dict(comp_data)
        
        try:
            aem_component = comp_data.get("aem_component", {})
            analyzed = comp_data.get("analyzed", {})
            
            # 创建智能摘要而非硬截断
            bdl_summary = _create_bdl_spec_summary(bdl_spec)
            analyzed_summary = _create_analyzed_component_summary(analyzed)
            
            user_message = f"""Map this AEM component to the best BDL component:

**Component ID**: {comp_id}
**Type**: {analyzed.get('component_type', 'unknown')}
**Dialog Fields Count**: {len(aem_component.get('dialog', {}).get('fields', []))}

{analyzed_summary}

---

{bdl_summary}

---

**Your Task**:
1. Use `search_bdl_components()` to find candidate components
2. Use `get_bdl_component_spec()` to analyze each candidate in detail
3. Compare candidates and select the best match
4. Return structured mapping with confidence score and reasoning

Remember: Use tools to access full specifications. Don't rely only on the summary above."""
            
            # response_format 已在 agent 创建时指定，无需重复传递
            result = await invoke_agent_with_retry(
                agent,
                messages=[HumanMessage(content=user_message)],
            )
            
            mapping_obj: BDLMappingOutput = result.get("structured_response")
            
            if mapping_obj:
                updated_data["bdl_mapping"] = {
                    "bdl_component_name": mapping_obj.bdl_component_name,
                    "confidence_score": mapping_obj.confidence_score,
                    "prop_mappings": [m.model_dump() for m in mapping_obj.prop_mappings],
                    "missing_features": mapping_obj.missing_features,
                    "reasoning": mapping_obj.reasoning,
                }
            else:
                # fallback: 使用 'custom' 而非 None，保持类型一致性
                updated_data["bdl_mapping"] = {
                    "bdl_component_name": "custom",
                    "confidence_score": 0.5,
                    "prop_mappings": [],
                    "missing_features": [],
                    "reasoning": "Failed to parse structured response"
                }
            
            updated_data["status"] = "mapping"
            updated_data["node_type"] = "intelligent"
            
        except Exception as e:
            error = create_error_result(e, comp_id, "bdl_mapping_node")
            if "errors" not in updated_data:
                updated_data["errors"] = []
            updated_data["errors"].append(error["error"])
        
        return comp_id, updated_data


async def bdl_mapping_node(state: MigrationGraphState) -> Dict[str, Any]:
    """
    BDL 映射智能节点
    
    使用 Agent 智能搜索和映射 BDL 组件
    
    特性：
    - 受控并发处理（默认最多5个组件同时处理）
    - 智能上下文注入（索引摘要而非硬截断）
    - 工具错误自动恢复
    - 结构化输出强制验证
    """
    components = state.get("components", {})
    bdl_spec = state.get("bdl_spec", {})
    updated_components = dict(components)
    
    # 筛选需要处理的组件
    components_to_process = {
        comp_id: comp_data
        for comp_id, comp_data in components.items()
        if comp_data.get("status") == "analyzing"
    }
    
    if not components_to_process:
        return {"components": updated_components}
    
    # 创建 agent（共享实例以节省资源）
    agent = _create_bdl_mapper()
    
    # 创建并发控制信号量（最多5个并发，可根据 provider rate limit 调整）
    semaphore = asyncio.Semaphore(5)
    
    # 创建并发任务
    tasks = [
        _process_single_component(comp_id, comp_data, bdl_spec, agent, semaphore)
        for comp_id, comp_data in components_to_process.items()
    ]
    
    # 并发执行所有任务
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # 合并结果
    for result in results:
        if isinstance(result, Exception):
            # 记录异常但不中断整体流程
            print(f"Component processing failed: {result}")
            continue
        
        comp_id, updated_data = result
        updated_components[comp_id] = updated_data
    
    return {"components": updated_components}
