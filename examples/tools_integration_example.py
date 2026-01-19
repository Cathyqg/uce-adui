"""
Tools 集成示例
展示如何在 LangGraph 节点中使用工具

LangGraph 1.0+ 工具使用模式:
1. 直接调用模式 - 在节点中直接调用工具
2. ToolNode 模式 - 让 Agent 自主选择工具
3. ReAct Agent 模式 - 推理 + 行动循环
"""

from typing import Any, Dict
from langgraph.prebuilt import create_react_agent, ToolNode  # LangGraph 1.0+

from src.llm import get_llm
from src.core.state import MigrationGraphState
from src.tools import (
    validate_typescript_syntax,
    lint_react_code,
    search_bdl_components,
    get_tools_for_node,
)


# ============================================================================
# 模式 1: 直接调用工具（推荐用于确定性流程）
# ============================================================================

async def code_quality_review_with_tools(state: MigrationGraphState) -> Dict[str, Any]:
    """
    代码质量审查节点 - 工具增强版
    
    LangGraph 1.0+ 最佳实践:
    - 先用工具快速验证（确定性、免费、快速）
    - 再用 LLM 深度分析（需要推理的部分）
    - 合并结果
    """
    components = state.get("components", {})
    results = {}
    
    for comp_id, comp_data in components.items():
        if comp_data.get("status") != "generating":
            continue
        
        react_component = comp_data.get("react_component", {})
        component_code = react_component.get("component_code", "")
        styles_code = react_component.get("styles_code", "")
        
        if not component_code:
            continue
        
        # === 阶段 1: 工具快速验证 ===
        
        # 1.1 TypeScript 语法验证（100% 准确，~0.5s，免费）
        ts_validation = validate_typescript_syntax.invoke({"code": component_code})
        
        if not ts_validation["valid"]:
            # 语法错误，直接失败，无需 LLM 审查
            results[comp_id] = {
                "code_quality": {
                    "reviewer": "typescript_compiler",
                    "score": 0,
                    "passed": False,
                    "issues": [
                        {
                            "severity": "critical",
                            "title": "TypeScript Syntax Error",
                            "description": err,
                            "source": "tsc",
                            "auto_fixable": False
                        }
                        for err in ts_validation["errors"]
                    ],
                    "tool_validation": ts_validation,
                    "skipped_llm": True  # 跳过 LLM 节省成本
                }
            }
            continue
        
        # 1.2 ESLint 规范检查（~1s，免费）
        lint_result = lint_react_code.invoke({"code": component_code})
        
        tool_issues = [
            {
                "severity": "error" if issue["severity"] == "error" else "minor",
                "title": f"ESLint: {issue['rule']}",
                "description": issue["message"],
                "location": f"line {issue['line']}:{issue['column']}",
                "source": "eslint",
                "auto_fixable": issue["fixable"]
            }
            for issue in lint_result["issues"]
        ]
        
        # === 阶段 2: LLM 深度审查（仅审查架构和最佳实践）===
        
        from src.nodes.review import _run_code_quality_review
        
        try:
            # LLM 只需要审查工具无法检查的部分
            # - 架构设计
            # - React 最佳实践
            # - 性能优化
            # - 可访问性
            llm_review = await _run_code_quality_review(
                comp_id, component_code, styles_code
            )
        except Exception as e:
            llm_review = {
                "score": 50,
                "issues": [],
                "error": str(e)
            }
        
        # === 阶段 3: 合并结果 ===
        
        all_issues = tool_issues + llm_review.get("issues", [])
        
        # 重新计算分数
        from src.nodes.review import REVIEW_CONFIG
        score = 100
        for issue in all_issues:
            severity = issue.get("severity", "minor")
            weight = REVIEW_CONFIG["code_quality"]["weights"].get(severity, -1)
            score += weight
        score = max(0, min(100, score))
        
        results[comp_id] = {
            "code_quality": {
                "reviewer": "code_quality_with_tools",
                "score": score,
                "passed": score >= REVIEW_CONFIG["code_quality"]["threshold"],
                "issues": all_issues,
                "tool_results": {
                    "typescript": ts_validation,
                    "eslint": lint_result
                },
                "llm_review": llm_review
            }
        }
    
    return {"parallel_review_results": results}


# ============================================================================
# 模式 2: ToolNode 模式（Agent 可以使用工具）
# ============================================================================

from langgraph.graph import StateGraph

def create_review_subgraph_with_tools():
    """
    创建带工具的审查子图
    
    LangGraph 1.0+ ToolNode 模式:
    - Agent 执行后可以选择调用工具
    - 工具执行完返回 Agent 继续处理
    """
    subgraph = StateGraph(MigrationGraphState)
    
    # 添加审查节点
    subgraph.add_node("code_quality_review", code_quality_review_node)
    
    # 添加 ToolNode
    tools = get_tools_for_node("code_quality_review")
    tool_node = ToolNode(tools)
    subgraph.add_node("tools", tool_node)
    
    # 条件边: Agent 决定是否需要调用工具
    subgraph.add_conditional_edges(
        "code_quality_review",
        should_use_tools,  # 检查 Agent 是否请求工具
        {
            "tools": "tools",        # 调用工具
            "continue": "next_node"  # 继续流程
        }
    )
    
    # 工具执行后返回 Agent
    subgraph.add_edge("tools", "code_quality_review")
    
    return subgraph


def should_use_tools(state: MigrationGraphState) -> str:
    """判断是否需要调用工具"""
    messages = state.get("messages", [])
    
    if not messages:
        return "continue"
    
    last_message = messages[-1]
    
    # 检查最后一条消息是否包含工具调用
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    
    return "continue"


# ============================================================================
# 模式 3: ReAct Agent 模式（最智能，让 Agent 自主决策）
# ============================================================================

def create_smart_code_quality_agent():
    """
    创建智能代码质量审查 Agent
    
    LangGraph 1.0+ ReAct 模式:
    - Agent 可以推理并决定使用哪些工具
    - 支持多轮工具调用
    - 自动循环直到完成任务
    """
    # 选择相关工具
    tools = get_tools_for_node("code_quality_review")
    
    # 创建 ReAct Agent - 使用工厂方法和新 API
    llm = get_llm(task="review", temperature=0)
    
    # LangGraph 1.0+: 使用 system_prompt 而不是 state_modifier
    system_prompt = """You are an expert code quality reviewer for React components.

Your task:
1. First, use validate_typescript_syntax to check for syntax errors
2. Then, use lint_react_code to check code style
3. If needed, use format_with_prettier to format the code
4. Finally, provide a comprehensive review of:
   - Architecture and design patterns
   - React best practices
   - Performance considerations
   - Accessibility

Use tools wisely to validate facts before making judgments.
"""
    
    agent = create_react_agent(
        llm,  # 第一个参数是 model
        tools,  # 第二个参数是 tools
        system_prompt=system_prompt,  # 使用 system_prompt
    )
    
    return agent


# ============================================================================
# 使用示例
# ============================================================================

async def example_usage():
    """Tools 使用示例"""
    
    # 示例 1: 直接调用工具
    code = """
    import React from 'react';
    
    const MyComponent: React.FC = () => {
        return <div>Hello</div>;
    };
    """
    
    result = validate_typescript_syntax.invoke({"code": code})
    print(f"Valid: {result['valid']}")
    
    # 示例 2: 搜索 BDL 组件
    bdl_spec = {...}  # 假设已加载
    matches = search_bdl_components.invoke({
        "query": "button with loading state",
        "bdl_spec": bdl_spec,
        "top_k": 3
    })
    print(f"Found {len(matches)} matching components")
    
    # 示例 3: 读取 AEM 组件
    aem_comp = read_aem_component_from_repo.invoke({
        "repo_path": "/path/to/aem-project",
        "component_name": "hero-banner"
    })
    if aem_comp["success"]:
        print(f"HTL content: {aem_comp['htl_content'][:100]}...")


if __name__ == "__main__":
    import asyncio
    asyncio.run(example_usage())
