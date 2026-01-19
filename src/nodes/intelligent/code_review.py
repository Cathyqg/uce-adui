"""
代码审查智能节点
使用 Agent 进行全面的代码质量审查

从 src/agents/code_reviewer.py 重构而来
"""
from typing import Any, Dict

from langchain_core.messages import HumanMessage

from src.core.state import MigrationGraphState
from src.agents import (
    PromptTemplates,
    CodeReviewOutput,
    create_structured_agent,
    invoke_agent_with_retry,
    create_error_result,
)
from src.llm import get_llm
from src.tools import (
    validate_typescript_syntax,
    lint_react_code,
    validate_bdl_compliance,
)
# ============================================================================
# Agent 创建（内部函数）
# ============================================================================

def _create_code_reviewer():
    """创建 Code Reviewer Agent（内部函数）"""
    llm = get_llm(task="review", temperature=0)
    
    tools = [
        validate_typescript_syntax,
        lint_react_code,
        validate_bdl_compliance,
    ]
    
    system_prompt = PromptTemplates.CODE_REVIEWER
    
    return create_structured_agent(
        llm,
        tools,
        system_prompt=system_prompt,
        response_format=CodeReviewOutput,
    )


# ============================================================================
# 节点实现
# ============================================================================

async def code_review_node(state: MigrationGraphState) -> Dict[str, Any]:
    """
    代码审查智能节点
    
    使用 Agent 进行全面审查
    """
    components = state.get("components", {})
    results = {}
    
    agent = _create_code_reviewer()
    
    for comp_id, comp_data in components.items():
        if comp_data.get("status") != "generating":
            continue
        
        react_component = comp_data.get("react_component", {})
        component_code = react_component.get("component_code", "")
        styles_code = react_component.get("styles_code", "")
        
        if not component_code:
            continue
        
        # 限制代码长度，避免超出上下文
        code_preview = component_code[:2000] + ("..." if len(component_code) > 2000 else "")
        styles_preview = styles_code[:1000] + ("..." if len(styles_code) > 1000 else "")
        
        user_message = f"""Review this component:

**Component**: {comp_id}

**Code**:
```tsx
{code_preview}
```

**Styles**:
```css
{styles_preview}
```

Use tools to validate syntax, linting, and BDL compliance. Provide comprehensive review."""
        
        try:
            # response_format 已在 agent 创建时指定，无需重复传递
            result = await invoke_agent_with_retry(
                agent,
                messages=[HumanMessage(content=user_message)],
            )
            
            review_output: CodeReviewOutput = result.get("structured_response")
            
            if review_output:
                results[comp_id] = {
                    "code_quality": {
                        "reviewer": "intelligent_reviewer",
                        "score": review_output.overall_score,
                        "passed": review_output.decision == "APPROVE",
                        "issues": [i.model_dump() for i in review_output.issues],
                        "decision": review_output.decision,
                        "summary": review_output.summary,
                        "node_type": "intelligent"
                    }
                }
            
        except Exception as e:
            error = create_error_result(e, comp_id, "code_review_node")
            results[comp_id] = {
                "code_quality": {
                    "score": 0,
                    "passed": False,
                    "error": error["error"]["message"]
                }
            }
    
    return {"parallel_review_results": results}
