"""
编辑器设计智能节点
使用 Agent 设计用户友好的 CMS 编辑器界面

从 src/agents/editor_designer.py 重构而来
"""
from typing import Any, Dict

from langchain_core.messages import HumanMessage

from src.core.state import MigrationGraphState
from src.agents import (
    PromptTemplates,
    EditorDesignOutput,
    create_structured_agent,
    invoke_agent_with_retry,
    create_error_result,
)
from src.llm import get_llm
# ============================================================================
# Agent 创建（内部函数）
# ============================================================================

def _create_editor_designer():
    """创建 Editor Designer Agent（内部函数）"""
    llm = get_llm(task="analysis", temperature=0.3)
    
    system_prompt = PromptTemplates.EDITOR_DESIGNER
    
    return create_structured_agent(
        llm,
        tools=[],  # 未来可添加 UX 验证工具
        system_prompt=system_prompt,
        response_format=EditorDesignOutput,
    )


# ============================================================================
# 节点实现
# ============================================================================

async def editor_design_node(state: MigrationGraphState) -> Dict[str, Any]:
    """
    编辑器设计智能节点
    
    使用 Agent 设计编辑器界面
    """
    components = state.get("components", {})
    updated_components = dict(components)
    
    agent = _create_editor_designer()
    
    for comp_id, comp_data in components.items():
        extracted_props = comp_data.get("extracted_props", [])
        if not extracted_props:
            continue
        
        aem_component = comp_data.get("aem_component", {})
        
        # 简化上下文注入，避免硬截断
        cms_capabilities = state.get("config", {}).get("cms_field_types", {})
        cms_summary = f"Available field types: {', '.join(list(cms_capabilities.keys())[:10])}" if cms_capabilities else "Standard CMS fields available"
        
        user_message = f"""Design editor for: {comp_id}

**Props**: {extracted_props}
**AEM Dialog Fields**: {len(aem_component.get('dialog', {}).get('fields', []))} fields
**{cms_summary}**

Design optimal editor layout with appropriate field types and grouping."""
        
        try:
            # response_format 已在 agent 创建时指定，无需重复传递
            result = await invoke_agent_with_retry(
                agent,
                messages=[HumanMessage(content=user_message)],
            )
            
            design_output: EditorDesignOutput = result.get("structured_response")
            
            if design_output:
                updated_components[comp_id]["editables"] = {
                    "editable_fields": [f.model_dump() for f in design_output.editable_fields],
                    "field_groups": [g.model_dump() for g in design_output.field_groups],
                    "editor_layout": design_output.editor_layout,
                    "ux_notes": design_output.ux_notes,
                }
                updated_components[comp_id]["node_type"] = "intelligent"
            
        except Exception as e:
            error = create_error_result(e, comp_id, "editor_design_node")
            updated_components[comp_id]["errors"].append(error["error"])
    
    return {"components": updated_components}
