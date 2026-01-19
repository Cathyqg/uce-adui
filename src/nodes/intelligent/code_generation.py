"""
代码生成智能节点
使用 Agent 生成并验证 React 组件代码

从 src/agents/code_generator.py 重构而来
"""
from typing import Any, Dict

from langchain_core.messages import HumanMessage

from src.core.state import MigrationGraphState
from src.agents import (
    PromptTemplates,
    CodeGeneratorOutput,
    create_structured_agent,
    invoke_agent_with_retry,
    create_error_result,
)
from src.llm import get_llm
from src.tools import (
    validate_typescript_syntax,
    lint_react_code,
    format_with_prettier,
)
# ============================================================================
# Agent 创建（内部函数）
# ============================================================================

def _create_code_generator():
    """创建 Code Generator Agent（内部函数）"""
    llm = get_llm(task="generation", temperature=0)
    
    tools = [
        validate_typescript_syntax,
        lint_react_code,
        format_with_prettier,
    ]
    
    system_prompt = PromptTemplates.CODE_GENERATOR
    
    return create_structured_agent(
        llm,
        tools,
        system_prompt=system_prompt,
        response_format=CodeGeneratorOutput,
    )


# ============================================================================
# 节点实现
# ============================================================================

async def code_generation_node(state: MigrationGraphState) -> Dict[str, Any]:
    """
    代码生成智能节点
    
    使用 Agent 生成并验证代码
    """
    components = state.get("components", {})
    updated_components = dict(components)
    component_registry = dict(state.get("component_registry", {}))
    stats = dict(state.get("stats", {}))
    
    agent = _create_code_generator()
    
    for comp_id, comp_data in components.items():
        if comp_data.get("status") not in ["transforming", "rejected"]:
            continue
        
        transformed = comp_data.get("transformed", {})
        bdl_mapping = comp_data.get("bdl_mapping", {})
        is_regeneration = comp_data.get("status") == "rejected"
        retry_count = comp_data.get("retry_count", 0)
        
        if is_regeneration and retry_count > 0:
            review_issues = comp_data.get("review", {}).get("aggregated", {}).get("issues", [])
            user_message = f"""⚠️ REGENERATION (Attempt {retry_count + 1}/3)

Previous rejected. Issues:
{_format_issues(review_issues[:5])}

Component: {comp_id}
BDL: {bdl_mapping.get('bdl_component_name')}
Props: {transformed.get('typescript_interface', '')}"""
        else:
            user_message = f"""Generate React component:

Component: {comp_id}
BDL: {bdl_mapping.get('bdl_component_name', 'custom')}
Interface: {transformed.get('typescript_interface', '')}
JSX: {transformed.get('jsx_template', '')}
CSS: {transformed.get('css_module', '')}"""
        
        try:
            # response_format 已在 agent 创建时指定，无需重复传递
            result = await invoke_agent_with_retry(
                agent,
                messages=[HumanMessage(content=user_message)],
            )
            
            code_output: CodeGeneratorOutput = result.get("structured_response")
            
            if code_output:
                component_name = _to_pascal_case(comp_id)
                updated_components[comp_id]["react_component"] = {
                    "component_id": comp_id,
                    "component_name": component_name,
                    "component_code": code_output.component_code,
                    "styles_code": code_output.styles_code,
                    "index_code": code_output.index_code or _generate_index(comp_id),
                    "validation_passed": code_output.validation_passed,
                    "validation_summary": code_output.validation_summary,
                    "tool_calls_made": code_output.tool_calls_made,
                }
                updated_components[comp_id]["status"] = "generating"
                updated_components[comp_id]["node_type"] = "intelligent"

                resource_type = comp_data.get("aem_component", {}).get("resource_type", "")
                if resource_type:
                    component_registry[resource_type] = component_name

                stats["generated_components"] = stats.get("generated_components", 0) + 1
            
        except Exception as e:
            error = create_error_result(e, comp_id, "code_generation_node")
            updated_components[comp_id]["errors"].append(error["error"])
            updated_components[comp_id]["status"] = "failed"
    
    return {
        "components": updated_components,
        "component_registry": component_registry,
        "stats": stats,
    }


# ============================================================================
# 辅助函数
# ============================================================================

def _format_issues(issues: list[dict]) -> str:
    """格式化问题列表"""
    return "\n".join([
        f"- [{i.get('severity')}] {i.get('title')}: {i.get('description')}"
        for i in issues
    ])


def _to_pascal_case(text: str) -> str:
    """转换为 PascalCase"""
    return "".join(word.capitalize() for word in text.replace("-", " ").replace("_", " ").split())


def _generate_index(comp_id: str) -> str:
    """生成默认 index"""
    name = _to_pascal_case(comp_id)
    return f"export {{ {name} }} from './{name}';\nexport type {{ {name}Props }} from './{name}';\n"
