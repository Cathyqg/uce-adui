"""
代码审查智能节点
使用 Agent 进行全面的代码质量审查

从 src/agents/code_reviewer.py 重构而来
"""
from datetime import datetime
from typing import Any, Dict, Optional
from uuid import uuid4

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

def _normalize_issue(issue: Any, category: str) -> Dict[str, Any]:
    if isinstance(issue, str):
        data = {"title": issue}
    elif isinstance(issue, dict):
        data = dict(issue)
    else:
        model_dump = getattr(issue, "model_dump", None)
        if callable(model_dump):
            data = model_dump()
        else:
            return {}
    return {
        "issue_id": data.get("issue_id", str(uuid4())),
        "category": data.get("category", category),
        "severity": data.get("severity", "minor"),
        "title": data.get("title", ""),
        "description": data.get("description", ""),
        "location": data.get("location", ""),
        "suggestion": data.get("suggestion", ""),
        "auto_fixable": bool(data.get("auto_fixable", False)),
    }


def _build_review_result(
    reviewer: str,
    score: int,
    passed: bool,
    issues: list[dict],
    metrics: Optional[Dict[str, Any]] = None,
    summary: str = "",
    reviewer_type: str = "agent",
    strategy: str = "agent",
) -> Dict[str, Any]:
    return {
        "reviewer": reviewer,
        "reviewer_type": reviewer_type,
        "strategy": strategy,
        "score": score,
        "passed": passed,
        "issues": issues,
        "metrics": metrics or {},
        "summary": summary,
        "timestamp": datetime.now().isoformat(),
    }


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
                issues = []
                for issue in review_output.issues:
                    normalized = _normalize_issue(issue, "code_quality")
                    if normalized:
                        issues.append(normalized)

                review_result = _build_review_result(
                    reviewer="code_quality",
                    score=review_output.overall_score,
                    passed=review_output.decision == "APPROVE",
                    issues=issues,
                    metrics=review_output.tool_validation_results,
                    summary=review_output.summary,
                    reviewer_type="agent",
                    strategy="agent",
                )
                review_result["decision"] = review_output.decision
                results[comp_id] = {"code_quality": review_result}
            
        except Exception as e:
            error = create_error_result(e, comp_id, "code_review_node")
            error_message = error["error"]["message"]
            review_result = _build_review_result(
                reviewer="code_quality",
                score=0,
                passed=False,
                issues=[],
                metrics={},
                summary=f"error: {error_message}",
                reviewer_type="agent",
                strategy="agent",
            )
            review_result["error"] = error_message
            results[comp_id] = {"code_quality": review_result}
    
    return {"parallel_review_results": results}
