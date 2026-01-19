"""
审查系统节点实现
包含代码质量、BDL 合规性、功能一致性审查以及人工审查

LangGraph 1.0+ Best Practices:
1. 使用 Send API 实现图级并行 (Fan-out/Fan-in 模式)
2. 使用 Pydantic 进行结构化输出验证
3. 使用 with_structured_output() 替代手动 JSON 解析
4. 实现重试机制和优雅降级
5. 使用 Reducer 自动合并并行节点状态

架构:
    distribute_reviews → [Send] → code_quality_review_v2
                      → [Send] → bdl_compliance_review_v2    → merge_review_results
                      → [Send] → function_parity_review_v2

================================================================================
⚠️ 需要定制的部分标记说明:
================================================================================
- [CUSTOMIZE]    = 需要根据实际项目定制的逻辑/配置
- [PLACEHOLDER]  = 占位符代码，需要完整实现
- [EXAMPLE]      = 示例代码，需要根据实际情况替换
- [PROMPT]       = LLM Prompt，可能需要根据实际需求调整

搜索这些标记可以快速定位需要修改的地方:
    grep -r "\\[CUSTOMIZE\\]\\|\\[PLACEHOLDER\\]\\|\\[EXAMPLE\\]\\|\\[PROMPT\\]" src/
================================================================================
"""
from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime
import os
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, TypeVar
from uuid import uuid4

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.llm import get_llm
from src.core.state import (
    AggregatedReview,
    BDLComplianceOutput,
    CodeQualityOutput,
    FunctionParityOutput,
    IssueSeverity,
    MigrationGraphState,
    Phase,
    ReviewDecision,
    ReviewIssue,
    ReviewResult,
    ReviewVerdict,
)


# ============================================================================
# 重试装饰器 - Best Practice for LLM calls
# ============================================================================

T = TypeVar('T')

def with_retry(
    max_attempts: int = 3,
    min_wait: float = 1,
    max_wait: float = 10
) -> Callable:
    """
    LLM 调用重试装饰器
    
    Best Practice:
    - 对瞬时错误 (rate limit, timeout) 进行重试
    - 使用指数退避避免雪崩
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        wait_time = min(min_wait * (2 ** attempt), max_wait)
                        await asyncio.sleep(wait_time)
            raise last_exception
        return wrapper
    return decorator


# ============================================================================
# Review Skip Helper
# ============================================================================

def _should_skip_review(state: MigrationGraphState) -> bool:
    """Return True when review should be bypassed."""
    config = state.get("config", {})
    if config.get("auto_approve_all"):
        return True
    return os.getenv("MIGRATION_SKIP_REVIEW", "").lower() in {"1", "true", "yes"}


# ============================================================================
# 配置
# ============================================================================

# [CUSTOMIZE] 审查配置 - 根据项目质量标准调整
# - threshold: 各类审查的及格分数线
# - weights: 各严重级别的扣分权重
# - auto_approve_threshold: 自动通过的最低平均分
# - require_human_for: 需要强制人工审查的组件路径模式
REVIEW_CONFIG = {
    "code_quality": {
        "threshold": 80,  # [CUSTOMIZE] 代码质量及格线
        "weights": {
            "critical": -100,  # 直接失败
            "major": -10,
            "minor": -2,
            "suggestion": -1
        }
    },
    "bdl_compliance": {
        "threshold": 85,  # [CUSTOMIZE] BDL 合规性及格线
        "weights": {
            "critical": -100,
            "major": -15,
            "minor": -5,
            "suggestion": -2
        }
    },
    "function_parity": {
        "threshold": 75,  # [CUSTOMIZE] 功能一致性及格线
        "weights": {
            "critical": -100,
            "major": -20,
            "minor": -3,
            "suggestion": -1
        }
    },
    "auto_approve_threshold": 85,  # [CUSTOMIZE] 自动通过阈值
    "require_human_for": ["core/*", "payment/*", "auth/*"]  # [CUSTOMIZE] 强制人工审查的组件模式
}


# ============================================================================
# 代码质量审查 Agent
# ============================================================================

# [PROMPT] 代码质量审查 Prompt
# [CUSTOMIZE] 可根据团队代码规范调整审查标准
CODE_QUALITY_SYSTEM_PROMPT = """You are an expert code reviewer specializing in React and TypeScript.
Your task is to review generated React component code for quality issues.

Review the code for:
1. **Syntax Correctness**
   - TypeScript compilation errors
   - JSX syntax errors
   - Import/export issues

2. **Code Standards**
   - Naming conventions (camelCase for variables, PascalCase for components)
   - File structure
   - Import organization
   - Comment completeness

3. **Best Practices**
   - React Hooks rules (deps arrays, no conditional hooks)
   - Memory leak prevention (cleanup in useEffect)
   - Performance optimization (memo, useMemo, useCallback where appropriate)
   - Accessibility (a11y) - ARIA labels, keyboard navigation

4. **Security**
   - XSS prevention (dangerouslySetInnerHTML usage)
   - Sensitive data handling
   - User input sanitization

For each issue found, provide:
- severity: critical | major | minor | suggestion
- title: short description
- description: detailed explanation
- location: file:line if applicable
- suggestion: how to fix

Respond in JSON format:
{
  "issues": [...],
  "metrics": {
    "lines_of_code": number,
    "complexity_score": number,
    "test_coverage_estimate": number
  },
  "summary": "overall assessment"
}
"""


# ============================================================================
# BDL 合规性审查 Agent
# ============================================================================

# [PROMPT] BDL 合规性审查 Prompt
# [CUSTOMIZE] ⚠️ 这是最重要的定制点之一！
# 需要根据实际 BDL 规范调整:
#   - Design Tokens 命名规则 (例如: $color-primary vs --hsbc-color-primary)
#   - 组件结构规范
#   - 断点和响应式规则
#   - 主题支持方式
BDL_COMPLIANCE_SYSTEM_PROMPT = """You are an expert in BDL (Brand Design Language) component specifications, 
particularly for HSBC's frontend component framework.

Your task is to review React components for BDL compliance.

Check for:
1. **Design Tokens Usage**
   - Colors MUST use BDL tokens (e.g., $color-primary, $color-text-default)
   - Spacing MUST use BDL spacing scale (e.g., $spacing-xs, $spacing-md)
   - Typography MUST use BDL type scale (e.g., $font-heading-1, $font-body)
   - Shadows MUST use BDL shadow tokens
   (NOTE: [CUSTOMIZE] Replace token names with actual BDL tokens)

2. **Component Structure**
   - Follows BDL component hierarchy
   - Props naming matches BDL conventions
   - Variants are defined per BDL spec
   - Events follow BDL naming (onAction, onChange, etc.)
   (NOTE: [CUSTOMIZE] Update to match actual BDL component patterns)

3. **Responsive Design**
   - Uses BDL breakpoints ($breakpoint-sm, $breakpoint-md, etc.)
   - Mobile-first approach
   - Container query support where applicable
   (NOTE: [CUSTOMIZE] Update breakpoint token names)

4. **Theme Support**
   - Supports light/dark themes
   - Brandable through CSS custom properties
   - RTL support consideration

Provide issues with severity and suggestions.
Respond in JSON format similar to code quality review.
"""


# ============================================================================
# 功能一致性审查 Agent
# ============================================================================

# [PROMPT] 功能一致性审查 Prompt - 可根据功能对比标准调整
FUNCTION_PARITY_SYSTEM_PROMPT = """You are an expert in component migration and functional testing.
Your task is to verify that the generated React component maintains functional parity with the original AEM component.

Analyze and compare:
1. **Visual Consistency**
   - Layout structure matches
   - Color and styling alignment
   - Typography consistency
   - Spacing and dimensions

2. **Interaction Consistency**
   - Click events preserved
   - Hover effects maintained
   - Animation behaviors kept
   - Focus states implemented

3. **Data Processing Consistency**
   - Same inputs produce same outputs
   - Validation logic preserved
   - Edge cases handled similarly
   - Error handling maintained

4. **Responsive Consistency**
   - Layout behaves same at each breakpoint
   - Touch device support
   - Keyboard navigation support

For discrepancies, rate severity:
- critical: Core functionality missing/broken
- major: Noticeable difference in behavior
- minor: Subtle visual or behavioral difference
- suggestion: Improvement opportunity

Respond in JSON format with issues and parity_score.
"""


# ============================================================================
# LLM 审查辅助函数 (被 Send API v2 节点使用)
# ============================================================================

@with_retry(max_attempts=3)
async def _run_code_quality_review(
    comp_id: str, 
    component_code: str, 
    styles_code: str
) -> Dict[str, Any]:
    """
    运行代码质量审查
    
    LangGraph 1.0 Best Practice:
    - 使用 with_structured_output() 获取结构化输出
    - 使用重试装饰器处理瞬时错误
    """
    # 创建带结构化输出的 LLM - 使用工厂方法
    llm = get_llm(task="review", temperature=0)
    structured_llm = llm.with_structured_output(CodeQualityOutput)
    
    messages = [
        SystemMessage(content=CODE_QUALITY_SYSTEM_PROMPT),
        HumanMessage(content=f"""
Please review the following React component code:

**Component Code:**
```tsx
{component_code}
```

**Styles:**
```css
{styles_code}
```

Provide a detailed quality review with score, issues, and metrics.
""")
    ]
    
    try:
        # 使用结构化输出 - LangGraph 1.0 推荐方式
        result: CodeQualityOutput = await structured_llm.ainvoke(messages)
        review_data = result.model_dump()
    except Exception:
        # 降级到普通解析
        response = await llm.ainvoke(messages)
        try:
            review_data = json.loads(response.content)
        except json.JSONDecodeError:
            json_match = re.search(r'\{[\s\S]*\}', response.content)
            if json_match:
                review_data = json.loads(json_match.group())
            else:
                review_data = {"score": 50, "issues": [], "metrics": {}, "summary": "Parse failed"}
    
    # 计算得分 (如果 LLM 没有直接给出)
    score = review_data.get("score", 100)
    if score == 100 and review_data.get("issues"):
        # 根据问题重新计算
        for issue in review_data.get("issues", []):
            severity = issue.get("severity", "minor")
            weight = REVIEW_CONFIG["code_quality"]["weights"].get(severity, -1)
            score += weight
    
    score = max(0, min(100, score))
    
    # 格式化问题列表
    issues = []
    for issue in review_data.get("issues", []):
        issues.append({
            "issue_id": str(uuid4()),
            "category": "code_quality",
            "severity": issue.get("severity", "minor"),
            "title": issue.get("title", ""),
            "description": issue.get("description", ""),
            "suggestion": issue.get("suggestion", "")
        })
    
    return {
        "reviewer": "code_quality",
        "score": score,
        "passed": score >= REVIEW_CONFIG["code_quality"]["threshold"],
        "issues": issues,
        "metrics": review_data.get("metrics", {}),
        "summary": review_data.get("summary", ""),
        "timestamp": datetime.now().isoformat()
    }


@with_retry(max_attempts=3)
async def _run_bdl_compliance_review(
    comp_id: str,
    component_code: str,
    styles_code: str,
    bdl_spec: Dict
) -> Dict[str, Any]:
    """
    运行 BDL 合规审查
    
    LangGraph 1.0 Best Practice:
    - 使用 with_structured_output() 获取结构化输出
    - 使用重试装饰器处理瞬时错误
    """
    # 使用工厂方法获取 LLM - LangGraph 1.0+ Best Practice
    llm = get_llm(task="review", temperature=0)
    structured_llm = llm.with_structured_output(BDLComplianceOutput)
    
    bdl_context = json.dumps(bdl_spec, indent=2)[:3000] if bdl_spec else "Standard BDL"
    
    messages = [
        SystemMessage(content=BDL_COMPLIANCE_SYSTEM_PROMPT),
        HumanMessage(content=f"""
Review this component for BDL compliance:

**BDL Specification:**
{bdl_context}

**Component Code:**
```tsx
{component_code}
```

**Styles:**
```css
{styles_code}
```

Provide BDL compliance review with score, issues, and metrics.
""")
    ]
    
    try:
        # 使用结构化输出 - LangGraph 1.0 推荐方式
        result: BDLComplianceOutput = await structured_llm.ainvoke(messages)
        review_data = result.model_dump()
    except Exception:
        # 降级到普通解析
        response = await llm.ainvoke(messages)
        try:
            review_data = json.loads(response.content)
        except json.JSONDecodeError:
            json_match = re.search(r'\{[\s\S]*\}', response.content)
            if json_match:
                review_data = json.loads(json_match.group())
            else:
                review_data = {"score": 50, "issues": [], "metrics": {}}
    
    score = review_data.get("score", 100)
    if score == 100 and review_data.get("issues"):
        # 根据问题重新计算
        for issue in review_data.get("issues", []):
            severity = issue.get("severity", "minor")
            weight = REVIEW_CONFIG["bdl_compliance"]["weights"].get(severity, -2)
            score += weight
    
    score = max(0, min(100, score))
    
    issues = []
    for issue in review_data.get("issues", []):
        issues.append({
            "issue_id": str(uuid4()),
            "category": "bdl_compliance",
            "severity": issue.get("severity", "minor"),
            "title": issue.get("title", ""),
            "description": issue.get("description", ""),
            "suggestion": issue.get("suggestion", "")
        })
    
    return {
        "reviewer": "bdl_compliance",
        "score": score,
        "passed": score >= REVIEW_CONFIG["bdl_compliance"]["threshold"],
        "issues": issues,
        "metrics": review_data.get("metrics", {}),
        "summary": review_data.get("summary", ""),
        "timestamp": datetime.now().isoformat()
    }


@with_retry(max_attempts=3)
async def _run_function_parity_review(
    comp_id: str,
    aem_component: Dict,
    react_component: Dict
) -> Dict[str, Any]:
    """
    运行功能一致性审查
    
    LangGraph 1.0 Best Practice:
    - 使用 with_structured_output() 获取结构化输出
    - 使用重试装饰器处理瞬时错误
    """
    # 使用工厂方法获取 LLM - LangGraph 1.0+ Best Practice
    llm = get_llm(task="review", temperature=0)
    structured_llm = llm.with_structured_output(FunctionParityOutput)
    
    htl_template = aem_component.get("htl_template", {})
    original_html = htl_template.get("raw_content", "") if isinstance(htl_template, dict) else ""
    
    messages = [
        SystemMessage(content=FUNCTION_PARITY_SYSTEM_PROMPT),
        HumanMessage(content=f"""
Compare functional parity:

**Original AEM HTL:**
```html
{original_html}
```

**Generated React:**
```tsx
{react_component.get("component_code", "")}
```

**Generated Styles:**
```css
{react_component.get("styles_code", "")}
```

Verify functional parity and report any discrepancies with score and issues.
""")
    ]
    
    try:
        # 使用结构化输出 - LangGraph 1.0 推荐方式
        result: FunctionParityOutput = await structured_llm.ainvoke(messages)
        review_data = result.model_dump()
    except Exception:
        # 降级到普通解析
        response = await llm.ainvoke(messages)
        try:
            review_data = json.loads(response.content)
        except json.JSONDecodeError:
            json_match = re.search(r'\{[\s\S]*\}', response.content)
            if json_match:
                review_data = json.loads(json_match.group())
            else:
                review_data = {"score": 50, "issues": [], "metrics": {}, "parity_score": 100}
    
    score = review_data.get("score", review_data.get("parity_score", 100))
    if score == 100 and review_data.get("issues"):
        # 根据问题重新计算
        for issue in review_data.get("issues", []):
            severity = issue.get("severity", "minor")
            weight = REVIEW_CONFIG["function_parity"]["weights"].get(severity, -1)
            score += weight
    
    score = max(0, min(100, score))
    
    issues = []
    for issue in review_data.get("issues", []):
        issues.append({
            "issue_id": str(uuid4()),
            "category": "function_parity",
            "severity": issue.get("severity", "minor"),
            "title": issue.get("title", ""),
            "description": issue.get("description", ""),
            "suggestion": issue.get("suggestion", "")
        })
    
    return {
        "reviewer": "function_parity",
        "score": score,
        "passed": score >= REVIEW_CONFIG["function_parity"]["threshold"],
        "issues": issues,
        "metrics": review_data.get("metrics", {"parity_score": score}),
        "summary": review_data.get("summary", ""),
        "timestamp": datetime.now().isoformat()
    }


# ============================================================================
# 审查聚合
# ============================================================================

def aggregate_reviews(state: MigrationGraphState) -> Dict[str, Any]:
    """
    聚合所有审查结果，决定是否需要人工审查
    
    决策逻辑:
    1. 任何 critical 问题 -> 需要人工审查
    2. 任何审查未通过 -> 需要人工审查
    3. 组件在强制人工审查列表中 -> 需要人工审查
    4. 平均分低于阈值 -> 需要人工审查
    5. 否则 -> 自动通过
    """
    components = state.get("components", {})
    updated_components = dict(components)
    pending_human_review = []
    
    for comp_id, comp_data in components.items():
        review = comp_data.get("review", {})
        
        if not review:
            continue
        
        # 获取各项审查结果
        code_quality = review.get("code_quality", {})
        bdl_compliance = review.get("bdl_compliance", {})
        function_parity = review.get("function_parity", {})
        
        scores = []
        if code_quality:
            scores.append(code_quality.get("score", 0))
        if bdl_compliance:
            scores.append(bdl_compliance.get("score", 0))
        if function_parity:
            scores.append(function_parity.get("score", 0))
        
        overall_score = sum(scores) / len(scores) if scores else 0
        
        # 检查是否有 critical 问题
        has_critical = False
        all_issues = []
        for r in [code_quality, bdl_compliance, function_parity]:
            for issue in r.get("issues", []):
                if isinstance(issue, dict):
                    all_issues.append(issue)
                    if issue.get("severity") == "critical":
                        has_critical = True
        
        # 检查是否在强制人工审查列表
        requires_human = False
        for pattern in REVIEW_CONFIG["require_human_for"]:
            if pattern.endswith("/*"):
                prefix = pattern[:-2]
                if comp_id.startswith(prefix):
                    requires_human = True
                    break
        
        # 决定是否需要人工审查
        needs_human_review = (
            has_critical or
            not code_quality.get("passed", True) or
            not bdl_compliance.get("passed", True) or
            not function_parity.get("passed", True) or
            overall_score < REVIEW_CONFIG["auto_approve_threshold"] or
            requires_human
        )
        
        # 更新聚合结果
        updated_components[comp_id]["review"]["aggregated"] = {
            "overall_score": overall_score,
            "auto_approved": not needs_human_review,
            "requires_human_review": needs_human_review,
            "total_issues": len(all_issues),
            "critical_issues": sum(1 for i in all_issues if i.get("severity") == "critical"),
            "timestamp": datetime.now().isoformat()
        }
        
        if needs_human_review:
            pending_human_review.append(comp_id)
            updated_components[comp_id]["status"] = "human_reviewing"
        else:
            updated_components[comp_id]["status"] = "approved"
    
    return {
        "components": updated_components,
        "pending_human_review": pending_human_review
    }


# ============================================================================
# 人工审查相关节点
# ============================================================================

def prepare_human_review(state: MigrationGraphState) -> Dict[str, Any]:
    """
    准备人工审查数据
    
    生成审查所需的所有上下文:
    - 原始代码
    - 生成代码
    - 审查报告
    - Diff 视图
    """
    pending = state.get("pending_human_review", [])
    components = state.get("components", {})
    
    if not pending:
        return {"should_interrupt": False}
    
    current_component_id = pending[0]
    comp_data = components.get(current_component_id, {})
    
    # 构建审查数据包
    review_package = {
        "component_id": current_component_id,
        "original": {
            "htl": comp_data.get("aem_component", {}).get("htl_template", {}).get("raw_content", ""),
            "dialog": comp_data.get("aem_component", {}).get("dialog", {}),
        },
        "generated": {
            "code": comp_data.get("react_component", {}).get("component_code", ""),
            "styles": comp_data.get("react_component", {}).get("styles_code", ""),
        },
        "review_results": comp_data.get("review", {}),
        "issues_summary": _summarize_issues(comp_data.get("review", {}))
    }
    
    return {
        "current_component_id": current_component_id,
        "should_interrupt": True,
        "interrupt_reason": "human_review_required",
        "human_review_package": review_package
    }


def _summarize_issues(review: Dict) -> Dict[str, Any]:
    """汇总问题"""
    summary = {
        "critical": [],
        "major": [],
        "minor": [],
        "suggestion": []
    }
    
    for category in ["code_quality", "bdl_compliance", "function_parity"]:
        category_review = review.get(category, {})
        for issue in category_review.get("issues", []):
            if isinstance(issue, dict):
                severity = issue.get("severity", "minor")
                if severity in summary:
                    summary[severity].append({
                        "category": category,
                        "title": issue.get("title", ""),
                        "suggestion": issue.get("suggestion", "")
                    })
    
    return summary


def human_review(state: MigrationGraphState) -> Dict[str, Any]:
    """
    人工审查节点
    
    这个节点会被 LangGraph 中断，等待人工输入
    人工审查后通过 graph.update_state() 更新状态
    """
    # 这个节点本身不做什么，它的作用是触发中断
    # 人工审查的结果会通过外部 API 注入到状态中
    return {
        "should_interrupt": True,
        "interrupt_reason": "awaiting_human_review"
    }


def process_human_decision(state: MigrationGraphState) -> Dict[str, Any]:
    """
    处理人工审查决定
    """
    current_id = state.get("current_component_id")
    decisions = state.get("human_review_decisions", {})
    components = state.get("components", {})
    pending = list(state.get("pending_human_review", []))
    
    if not current_id or current_id not in decisions:
        return {}
    
    decision_data = decisions[current_id]
    decision = decision_data.get("decision", "approve")
    feedback = decision_data.get("feedback", "")
    reviewer = decision_data.get("reviewer", "unknown")
    
    updated_components = dict(components)
    
    # 更新组件的审查决定
    if current_id in updated_components:
        updated_components[current_id]["review"]["human_decision"] = {
            "decision": decision,
            "feedback": feedback,
            "reviewer": reviewer,
            "timestamp": datetime.now().isoformat()
        }
    
    # 从待审列表移除
    if current_id in pending:
        pending.remove(current_id)
    
    return {
        "components": updated_components,
        "pending_human_review": pending,
        "should_interrupt": False
    }


def auto_approve(state: MigrationGraphState) -> Dict[str, Any]:
    """
    自动批准节点
    """
    components = state.get("components", {})
    updated_components = dict(components)
    stats = dict(state.get("stats", {}))
    skip_review = _should_skip_review(state)
    
    for comp_id, comp_data in components.items():
        if skip_review:
            if comp_data.get("status") == "failed":
                continue
            if not comp_data.get("react_component"):
                continue
            if comp_data.get("status") != "approved":
                updated_components[comp_id]["status"] = "approved"
                stats["approved_components"] = stats.get("approved_components", 0) + 1
            review = updated_components[comp_id].setdefault("review", {})
            review.setdefault("aggregated", {
                "overall_score": 100.0,
                "auto_approved": True,
                "requires_human_review": False,
                "total_issues": 0,
                "critical_issues": 0,
                "timestamp": datetime.now().isoformat(),
                "skipped_review": True
            })
            continue

        review = comp_data.get("review", {})
        aggregated = review.get("aggregated", {})
        
        if aggregated.get("auto_approved"):
            if comp_data.get("status") != "approved":
                updated_components[comp_id]["status"] = "approved"
                stats["approved_components"] = stats.get("approved_components", 0) + 1
    
    return {
        "components": updated_components,
        "stats": stats
    }


def handle_rejection(state: MigrationGraphState) -> Dict[str, Any]:
    """
    处理拒绝 - 标记组件需要重新生成
    """
    current_id = state.get("current_component_id")
    components = state.get("components", {})
    decisions = state.get("human_review_decisions", {})
    
    updated_components = dict(components)
    
    if current_id and current_id in updated_components:
        feedback = decisions.get(current_id, {}).get("feedback", "")
        
        updated_components[current_id]["status"] = "rejected"
        updated_components[current_id]["rejection_feedback"] = feedback
        updated_components[current_id]["retry_count"] = \
            updated_components[current_id].get("retry_count", 0) + 1
    
    return {
        "components": updated_components,
        "should_regenerate": True
    }


def apply_modifications(state: MigrationGraphState) -> Dict[str, Any]:
    """
    应用人工修改
    """
    current_id = state.get("current_component_id")
    components = state.get("components", {})
    decisions = state.get("human_review_decisions", {})
    
    updated_components = dict(components)
    
    if current_id and current_id in decisions:
        modification = decisions[current_id].get("modification", {})
        
        if modification:
            # 应用代码修改
            if "code" in modification:
                if "react_component" not in updated_components[current_id]:
                    updated_components[current_id]["react_component"] = {}
                updated_components[current_id]["react_component"]["component_code"] = \
                    modification["code"]
            
            if "styles" in modification:
                updated_components[current_id]["react_component"]["styles_code"] = \
                    modification["styles"]
            
            updated_components[current_id]["status"] = "approved"
            updated_components[current_id]["modified_by_human"] = True
    
    return {"components": updated_components}


# ============================================================================
# LangGraph Send API 模式 - 图级并行审查节点
# ============================================================================
#
# 这些函数是为 LangGraph Send API (Fan-out/Fan-in) 模式设计的独立节点
# 每个节点在图级别并行执行，LangSmith 可以追踪每个节点的执行情况
#
# 流程:
#   distribute_reviews → [Send] → code_quality_review_v2
#                     → [Send] → bdl_compliance_review_v2    → [自动合并] → merge_review_results_node
#                     → [Send] → function_parity_review_v2
#
# ============================================================================

async def distribute_reviews_node(state: MigrationGraphState) -> Dict[str, Any]:
    """
    分发审查任务节点 - Fan-out 起点
    
    此节点准备数据，后续由 Send API 分发到三个并行审查节点
    """
    components = state.get("components", {})
    
    # 找出需要审查的组件
    components_to_review = []
    for comp_id, comp_data in components.items():
        if comp_data.get("status") == "generating":
            components_to_review.append(comp_id)
    
    return {
        "components_to_review": components_to_review,
        "current_phase": Phase.AUTO_REVIEW.value
    }


async def code_quality_review_v2(state: MigrationGraphState) -> Dict[str, Any]:
    """
    代码质量审查节点 (Send API 版本)
    
    LangGraph Send API 模式:
    - 作为独立节点在图级别执行
    - 结果通过 parallel_review_results Reducer 自动合并
    - LangSmith 可以单独追踪此节点
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
        
        try:
            review_result = await _run_code_quality_review(
                comp_id, component_code, styles_code
            )
            results[comp_id] = {"code_quality": review_result}
        except Exception as e:
            results[comp_id] = {
                "code_quality": {
                    "reviewer": "code_quality",
                    "score": 0,
                    "passed": False,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
            }
    
    # 返回到 parallel_review_results，由 Reducer 合并
    return {"parallel_review_results": results}


async def bdl_compliance_review_v2(state: MigrationGraphState) -> Dict[str, Any]:
    """
    BDL 合规性审查节点 (Send API 版本)
    
    LangGraph Send API 模式:
    - 与 code_quality_review_v2 并行执行
    - 结果自动合并到 parallel_review_results
    """
    components = state.get("components", {})
    bdl_spec = state.get("bdl_spec", {})
    results = {}
    
    for comp_id, comp_data in components.items():
        if comp_data.get("status") != "generating":
            continue
        
        react_component = comp_data.get("react_component", {})
        component_code = react_component.get("component_code", "")
        styles_code = react_component.get("styles_code", "")
        
        if not component_code:
            continue
        
        try:
            review_result = await _run_bdl_compliance_review(
                comp_id, component_code, styles_code, bdl_spec
            )
            results[comp_id] = {"bdl_compliance": review_result}
        except Exception as e:
            results[comp_id] = {
                "bdl_compliance": {
                    "reviewer": "bdl_compliance",
                    "score": 0,
                    "passed": False,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
            }
    
    return {"parallel_review_results": results}


async def function_parity_review_v2(state: MigrationGraphState) -> Dict[str, Any]:
    """
    功能一致性审查节点 (Send API 版本)
    
    LangGraph Send API 模式:
    - 与其他审查节点并行执行
    - 结果自动合并到 parallel_review_results
    """
    components = state.get("components", {})
    results = {}
    
    for comp_id, comp_data in components.items():
        if comp_data.get("status") != "generating":
            continue
        
        aem_component = comp_data.get("aem_component", {})
        react_component = comp_data.get("react_component", {})
        
        if not react_component:
            continue
        
        try:
            review_result = await _run_function_parity_review(
                comp_id, aem_component, react_component
            )
            results[comp_id] = {"function_parity": review_result}
        except Exception as e:
            results[comp_id] = {
                "function_parity": {
                    "reviewer": "function_parity",
                    "score": 0,
                    "passed": False,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
            }
    
    return {"parallel_review_results": results}


def merge_review_results_node(state: MigrationGraphState) -> Dict[str, Any]:
    """
    合并审查结果节点 - Fan-in 终点
    
    LangGraph Send API 模式:
    - 三个并行审查节点的结果已通过 Reducer 自动合并到 parallel_review_results
    - 此节点将合并后的结果写入 components 的 review 字段
    """
    components = state.get("components", {})
    parallel_results = state.get("parallel_review_results", {})
    updated_components = dict(components)
    
    for comp_id, review_results in parallel_results.items():
        if comp_id not in updated_components:
            continue
        
        # 将并行审查结果写入组件的 review 字段
        if "review" not in updated_components[comp_id]:
            updated_components[comp_id]["review"] = {}
        
        for review_type, result in review_results.items():
            updated_components[comp_id]["review"][review_type] = result
    
    return {
        "components": updated_components,
        # 清空 parallel_review_results 供下次使用
        "parallel_review_results": {"__clear__": True}
    }


# ============================================================================
# Send API 路由函数
# ============================================================================

def route_to_parallel_reviews(state: MigrationGraphState) -> List:
    """
    路由函数 - 返回 Send 对象列表实现 Fan-out
    
    LangGraph 会自动:
    1. 并行执行返回的所有 Send 目标节点
    2. 等待所有节点完成
    3. 使用 Reducer 合并各节点的状态更新
    
    Usage in graph:
        from langgraph.constants import Send
        
        graph.add_conditional_edges(
            "distribute_reviews",
            route_to_parallel_reviews,
            ["code_quality_review", "bdl_compliance_review", "function_parity_review"]
        )
    """
    # 导入放在函数内部避免循环导入
    from langgraph.constants import Send
    
    if _should_skip_review(state):
        return [Send("merge_review_results", state)]

    # 检查是否有组件需要审查
    components = state.get("components", {})
    has_components_to_review = any(
        comp_data.get("status") == "generating"
        for comp_data in components.values()
    )
    
    if not has_components_to_review:
        # 没有组件需要审查，直接跳到合并节点
        return [Send("merge_review_results", state)]
    
    # 返回三个 Send，LangGraph 会并行执行这三个节点
    return [
        Send("code_quality_review", state),
        Send("bdl_compliance_review", state),
        Send("function_parity_review", state),
    ]
