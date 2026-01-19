"""
Auto-fix node for generated React components.
Uses an agent to apply small, targeted fixes based on review issues.
"""
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage

from src.core.state import MigrationGraphState
from src.agents import (
    CodeGeneratorOutput,
    create_code_fixer_agent,
    invoke_agent_with_retry,
    create_error_result,
)
 
def _create_code_fixer():
    return create_code_fixer_agent()


def _format_issues(issues: List[Dict[str, Any]]) -> str:
    if not issues:
        return "- None"
    return "\n".join(
        f"- [{issue.get('severity', 'minor')}] {issue.get('title', '')}: {issue.get('description', '')}"
        for issue in issues
    )


async def code_fix_node(state: MigrationGraphState) -> Dict[str, Any]:
    components = state.get("components", {})
    updated_components = dict(components)
    stats = dict(state.get("stats", {}))

    pending = list(state.get("pending_auto_fix", []))
    if not pending:
        pending = [
            comp_id
            for comp_id, comp_data in components.items()
            if comp_data.get("status") == "auto_fixing"
        ]

    if not pending:
        return {"components": updated_components, "stats": stats, "pending_auto_fix": []}

    agent = _create_code_fixer()

    for comp_id in pending:
        comp_data = components.get(comp_id, {})
        react_component = comp_data.get("react_component", {})
        component_code = react_component.get("component_code", "")
        styles_code = react_component.get("styles_code", "")
        if not component_code:
            continue

        issues = comp_data.get("review", {}).get("aggregated", {}).get("issues", [])
        user_message = f"""Fix the following review issues with minimal changes.

Component: {comp_id}

Issues:
{_format_issues(issues[:10])}

Current component code:
```tsx
{component_code}
```

Current styles:
```css
{styles_code}
```

Return updated component code and styles, preserving existing API and structure.
"""

        try:
            result = await invoke_agent_with_retry(
                agent,
                messages=[HumanMessage(content=user_message)],
            )
            fix_output: CodeGeneratorOutput = result.get("structured_response")
            if not fix_output:
                continue

            updated_components.setdefault(comp_id, {})
            updated_components[comp_id].setdefault("react_component", {})
            updated_components[comp_id]["react_component"]["component_code"] = fix_output.component_code
            updated_components[comp_id]["react_component"]["styles_code"] = fix_output.styles_code
            if fix_output.index_code:
                updated_components[comp_id]["react_component"]["index_code"] = fix_output.index_code
            updated_components[comp_id]["react_component"]["validation_passed"] = fix_output.validation_passed
            updated_components[comp_id]["react_component"]["validation_summary"] = fix_output.validation_summary
            updated_components[comp_id]["react_component"]["tool_calls_made"] = fix_output.tool_calls_made
            updated_components[comp_id]["react_component"]["issues_fixed"] = fix_output.issues_fixed

            updated_components[comp_id]["status"] = "generating"
            updated_components[comp_id]["auto_fix_attempts"] = (
                comp_data.get("auto_fix_attempts", 0) + 1
            )
            stats["auto_fix_count"] = stats.get("auto_fix_count", 0) + 1

        except Exception as exc:
            error = create_error_result(exc, comp_id, "code_fix_node")
            updated_components[comp_id].setdefault("errors", [])
            updated_components[comp_id]["errors"].append(error["error"])
            updated_components[comp_id]["status"] = "rejected"
            updated_components[comp_id]["auto_fix_attempts"] = (
                comp_data.get("auto_fix_attempts", 0) + 1
            )

    return {
        "components": updated_components,
        "stats": stats,
        "pending_auto_fix": [],
    }
