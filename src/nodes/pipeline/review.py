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
import subprocess
import tempfile
import time
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
# Review Config Helpers
# ============================================================================

def _review_settings(state: MigrationGraphState) -> Dict[str, Any]:
    config = state.get("config", {})
    review = config.get("review", {})
    return review if isinstance(review, dict) else {}


def _review_enabled(state: MigrationGraphState, review_name: str) -> bool:
    review = _review_settings(state)
    enabled = review.get("enabled")
    if isinstance(enabled, dict) and review_name in enabled:
        return bool(enabled.get(review_name))
    return True


def _runtime_check_settings(state: MigrationGraphState) -> Dict[str, Any]:
    review = _review_settings(state)
    runtime_check = review.get("runtime_check", {})
    if not isinstance(runtime_check, dict):
        runtime_check = {}
    command_value = runtime_check.get("command", "")
    command = str(command_value).strip() if command_value is not None else ""
    enabled_value = runtime_check.get("enabled")
    enabled = bool(command) if enabled_value is None else bool(enabled_value)
    timeout_value = runtime_check.get("timeout_seconds", 60)
    try:
        timeout_seconds = int(timeout_value)
    except (TypeError, ValueError):
        timeout_seconds = 60
    cwd_value = runtime_check.get("cwd", "")
    cwd = str(cwd_value).strip() if isinstance(cwd_value, str) else ""
    return {
        "enabled": enabled,
        "command": command,
        "timeout_seconds": max(1, timeout_seconds),
        "cwd": cwd or None,
        "shell": bool(runtime_check.get("shell", True)),
    }


def _runtime_check_enabled(state: MigrationGraphState) -> bool:
    if not _review_enabled(state, "runtime_check"):
        return False
    settings = _runtime_check_settings(state)
    return bool(settings.get("enabled")) and bool(settings.get("command"))


def _enabled_reviews(state: MigrationGraphState) -> List[str]:
    review_types = (
        "code_quality",
        "bdl_compliance",
        "function_parity",
        "accessibility",
        "security",
        "editor_schema",
        "runtime_check",
    )
    enabled = []
    for name in review_types:
        if name == "runtime_check":
            if _runtime_check_enabled(state):
                enabled.append(name)
            continue
        if _review_enabled(state, name):
            enabled.append(name)
    return enabled


def _should_skip_review(state: MigrationGraphState) -> bool:
    """Return True when review should be bypassed."""
    config = state.get("config", {})
    review = _review_settings(state)
    if review.get("skip") is True:
        return True
    if config.get("auto_approve_all"):
        return True
    if not _enabled_reviews(state):
        return True
    return os.getenv("MIGRATION_SKIP_REVIEW", "").lower() in {"1", "true", "yes"}


def _auto_fix_settings(state: MigrationGraphState) -> Dict[str, Any]:
    review = _review_settings(state)
    auto_fix = review.get("auto_fix", {})
    if not isinstance(auto_fix, dict):
        auto_fix = {}
    return {
        "enabled": auto_fix.get("enabled", True),
        "max_attempts": int(auto_fix.get("max_attempts", 2)),
        "max_severity": str(auto_fix.get("max_severity", "minor")).lower(),
    }


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
    issues: List[Dict[str, Any]],
    metrics: Optional[Dict[str, Any]] = None,
    summary: str = "",
    reviewer_type: str = "pipeline",
    strategy: str = "pipeline",
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


def _score_from_issues(review_name: str, issues: List[Dict[str, Any]]) -> int:
    score = 100
    weights = REVIEW_CONFIG.get(review_name, {}).get("weights", {})
    for issue in issues:
        severity = issue.get("severity", "minor")
        score += weights.get(severity, -1)
    return max(0, min(100, score))


def _severity_rank(severity: str) -> int:
    ranking = {"critical": 4, "major": 3, "minor": 2, "suggestion": 1}
    return ranking.get(severity, 1)


def _max_issue_severity(issues: List[Dict[str, Any]]) -> str:
    if not issues:
        return ""
    max_issue = max(issues, key=lambda issue: _severity_rank(issue.get("severity", "minor")))
    return str(max_issue.get("severity", "minor")).lower()


def _truncate_text(value: str, limit: int = 2000) -> str:
    text = "" if value is None else str(value)
    if len(text) <= limit:
        return text
    return f"{text[:limit]}... (truncated {len(text) - limit} chars)"


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
    "accessibility": {
        "threshold": 90,  # [CUSTOMIZE] 可访问性及格线
        "weights": {
            "critical": -100,
            "major": -15,
            "minor": -5,
            "suggestion": -2
        }
    },
    "security": {
        "threshold": 95,  # [CUSTOMIZE] 安全审查及格线
        "weights": {
            "critical": -100,
            "major": -25,
            "minor": -5,
            "suggestion": -1
        }
    },
    "editor_schema": {
        "threshold": 85,  # [CUSTOMIZE] 编辑器配置及格线
        "weights": {
            "critical": -100,
            "major": -15,
            "minor": -3,
            "suggestion": -1
        }
    },
    "runtime_check": {
        "threshold": 90,  # [CUSTOMIZE] 运行检查及格线
        "weights": {
            "critical": -100,
            "major": -50,
            "minor": -10,
            "suggestion": -2
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
        normalized = _normalize_issue(issue, "code_quality")
        if normalized:
            issues.append(normalized)

    return _build_review_result(
        reviewer="code_quality",
        score=score,
        passed=score >= REVIEW_CONFIG["code_quality"]["threshold"],
        issues=issues,
        metrics=review_data.get("metrics", {}),
        summary=review_data.get("summary", ""),
        reviewer_type="pipeline",
        strategy="pipeline",
    )


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
        normalized = _normalize_issue(issue, "bdl_compliance")
        if normalized:
            issues.append(normalized)

    return _build_review_result(
        reviewer="bdl_compliance",
        score=score,
        passed=score >= REVIEW_CONFIG["bdl_compliance"]["threshold"],
        issues=issues,
        metrics=review_data.get("metrics", {}),
        summary=review_data.get("summary", ""),
        reviewer_type="pipeline",
        strategy="pipeline",
    )


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
        normalized = _normalize_issue(issue, "function_parity")
        if normalized:
            issues.append(normalized)

    return _build_review_result(
        reviewer="function_parity",
        score=score,
        passed=score >= REVIEW_CONFIG["function_parity"]["threshold"],
        issues=issues,
        metrics=review_data.get("metrics", {"parity_score": score}),
        summary=review_data.get("summary", ""),
        reviewer_type="pipeline",
        strategy="pipeline",
    )


def _run_accessibility_review(component_code: str, styles_code: str) -> Dict[str, Any]:
    issues_raw: List[Dict[str, Any]] = []

    for match in re.finditer(r"<img\b[^>]*>", component_code, flags=re.IGNORECASE):
        tag = match.group(0)
        if not re.search(r"\balt\s*=", tag, flags=re.IGNORECASE):
            issues_raw.append({
                "severity": "major",
                "title": "Missing alt text on img",
                "description": "Image tags should include alt text for screen readers.",
                "location": "component",
                "suggestion": "Add a meaningful alt attribute."
            })

    for match in re.finditer(r"<input\b[^>]*>", component_code, flags=re.IGNORECASE):
        tag = match.group(0)
        if not re.search(r"\baria-(label|labelledby)\s*=", tag, flags=re.IGNORECASE):
            issues_raw.append({
                "severity": "minor",
                "title": "Input missing accessible label",
                "description": "Inputs should have aria-label or aria-labelledby when no label is present.",
                "location": "component",
                "suggestion": "Add aria-label/aria-labelledby or ensure a visible label is linked."
            })

    for match in re.finditer(r"<(div|span|section|li|p)\b[^>]*>", component_code, flags=re.IGNORECASE):
        tag = match.group(0)
        if "onClick" in tag and not re.search(r"\b(role|tabIndex)\s*=", tag, flags=re.IGNORECASE):
            issues_raw.append({
                "severity": "minor",
                "title": "Clickable non-interactive element",
                "description": "Clickable elements should be keyboard accessible and have a role.",
                "location": "component",
                "suggestion": "Use a button or add role and tabIndex."
            })

    if re.search(r"outline\s*:\s*(none|0)\b", styles_code, flags=re.IGNORECASE):
        issues_raw.append({
            "severity": "suggestion",
            "title": "Focus outline removed",
            "description": "Removing focus outlines reduces keyboard accessibility.",
            "location": "styles",
            "suggestion": "Preserve focus indicators or provide a visible alternative."
        })

    issues: List[Dict[str, Any]] = []
    for issue in issues_raw:
        normalized = _normalize_issue(issue, "accessibility")
        if normalized:
            issues.append(normalized)

    score = _score_from_issues("accessibility", issues)
    return _build_review_result(
        reviewer="accessibility",
        score=score,
        passed=score >= REVIEW_CONFIG["accessibility"]["threshold"],
        issues=issues,
        metrics={
            "missing_alt": sum(1 for i in issues if i.get("title") == "Missing alt text on img"),
            "missing_input_labels": sum(1 for i in issues if i.get("title") == "Input missing accessible label"),
            "non_interactive_clicks": sum(1 for i in issues if i.get("title") == "Clickable non-interactive element"),
        },
        summary="Accessibility heuristic review",
        reviewer_type="pipeline",
        strategy="pipeline",
    )


def _run_security_review(component_code: str, styles_code: str) -> Dict[str, Any]:
    issues_raw: List[Dict[str, Any]] = []

    if "dangerouslySetInnerHTML" in component_code:
        issues_raw.append({
            "severity": "critical",
            "title": "dangerouslySetInnerHTML usage",
            "description": "Direct HTML injection can lead to XSS vulnerabilities.",
            "location": "component",
            "suggestion": "Sanitize HTML before rendering or avoid direct injection."
        })

    if re.search(r"\beval\s*\(", component_code):
        issues_raw.append({
            "severity": "critical",
            "title": "eval usage detected",
            "description": "eval introduces severe security risks.",
            "location": "component",
            "suggestion": "Remove eval usage."
        })

    if re.search(r"\bnew\s+Function\s*\(", component_code):
        issues_raw.append({
            "severity": "critical",
            "title": "Function constructor usage",
            "description": "new Function behaves like eval and can be exploited.",
            "location": "component",
            "suggestion": "Remove dynamic code execution."
        })

    for match in re.finditer(r"<a\b[^>]*target=[\"']_blank[\"'][^>]*>", component_code, flags=re.IGNORECASE):
        tag = match.group(0)
        if not re.search(r"\brel\s*=\s*[\"'][^\"']*(noopener|noreferrer)[^\"']*[\"']", tag, flags=re.IGNORECASE):
            issues_raw.append({
                "severity": "major",
                "title": "Missing rel on target=_blank",
                "description": "Links with target=_blank should set rel=noopener noreferrer.",
                "location": "component",
                "suggestion": "Add rel=\"noopener noreferrer\"."
            })

    if re.search(r"href\s*=\s*[\"']javascript:", component_code, flags=re.IGNORECASE):
        issues_raw.append({
            "severity": "critical",
            "title": "javascript: URL detected",
            "description": "javascript: URLs can be exploited for XSS.",
            "location": "component",
            "suggestion": "Remove javascript: URLs."
        })

    issues: List[Dict[str, Any]] = []
    for issue in issues_raw:
        normalized = _normalize_issue(issue, "security")
        if normalized:
            issues.append(normalized)

    score = _score_from_issues("security", issues)
    return _build_review_result(
        reviewer="security",
        score=score,
        passed=score >= REVIEW_CONFIG["security"]["threshold"],
        issues=issues,
        metrics={
            "xss_risks": sum(
                1 for i in issues
                if i.get("title") in {
                    "dangerouslySetInnerHTML usage",
                    "eval usage detected",
                    "Function constructor usage",
                    "javascript: URL detected",
                }
            ),
            "target_blank_without_rel": sum(
                1 for i in issues if i.get("title") == "Missing rel on target=_blank"
            ),
        },
        summary="Security heuristic review",
        reviewer_type="pipeline",
        strategy="pipeline",
    )


def _run_editor_schema_review(comp_id: str, cms_config: Dict[str, Any]) -> Dict[str, Any]:
    issues_raw: List[Dict[str, Any]] = []

    if not cms_config:
        issues_raw.append({
            "severity": "major",
            "title": "Missing CMS config",
            "description": "Component does not have a generated CMS configuration schema.",
            "location": "config",
            "suggestion": "Ensure config_generation ran and produced cms_config."
        })

    json_schema = cms_config.get("json_schema", {}) if cms_config else {}
    properties = json_schema.get("properties", {}) if isinstance(json_schema, dict) else {}
    required = json_schema.get("required", []) if isinstance(json_schema, dict) else []
    editable_fields = cms_config.get("editable_fields", []) if cms_config else []
    field_groups = cms_config.get("field_groups", []) if cms_config else []

    if not properties:
        issues_raw.append({
            "severity": "major",
            "title": "Schema has no properties",
            "description": "JSON schema properties are empty.",
            "location": "config",
            "suggestion": "Populate schema properties from editor fields."
        })

    for req in required:
        if req not in properties:
            issues_raw.append({
                "severity": "critical",
                "title": "Required field missing from schema",
                "description": f"Required field '{req}' is not present in schema properties.",
                "location": "config",
                "suggestion": "Ensure required fields exist in properties."
            })

    field_ids = set()
    for field in editable_fields:
        prop_name = field.get("prop_name") or field.get("field_id") or ""
        if prop_name:
            field_ids.add(field.get("field_id", prop_name))
            if prop_name not in properties:
                issues_raw.append({
                    "severity": "major",
                    "title": "Editable field not in schema",
                    "description": f"Editable field '{prop_name}' is missing from schema properties.",
                    "location": "config",
                    "suggestion": "Sync editable fields and schema properties."
                })

    if editable_fields == []:
        issues_raw.append({
            "severity": "minor",
            "title": "No editable fields",
            "description": "CMS config has no editable fields defined.",
            "location": "config",
            "suggestion": "Define editable fields for the component."
        })

    for group in field_groups:
        for field_id in group.get("fields", []):
            if field_id not in field_ids:
                issues_raw.append({
                    "severity": "minor",
                    "title": "Field group references unknown field",
                    "description": f"Field group references '{field_id}' which is not in editable_fields.",
                    "location": "config",
                    "suggestion": "Ensure field_groups only reference existing fields."
                })

    issues: List[Dict[str, Any]] = []
    for issue in issues_raw:
        normalized = _normalize_issue(issue, "editor_schema")
        if normalized:
            issues.append(normalized)

    score = _score_from_issues("editor_schema", issues)
    return _build_review_result(
        reviewer="editor_schema",
        score=score,
        passed=score >= REVIEW_CONFIG["editor_schema"]["threshold"],
        issues=issues,
        metrics={
            "properties_count": len(properties) if isinstance(properties, dict) else 0,
            "required_count": len(required) if isinstance(required, list) else 0,
            "editable_fields_count": len(editable_fields) if isinstance(editable_fields, list) else 0,
        },
        summary=f"Editor schema review for {comp_id}",
        reviewer_type="pipeline",
        strategy="pipeline",
    )


def _run_runtime_check(
    comp_id: str,
    component_code: str,
    styles_code: str,
    settings: Dict[str, Any],
) -> Dict[str, Any]:
    command = settings.get("command", "")
    if not settings.get("enabled") or not command:
        return _build_review_result(
            reviewer="runtime_check",
            score=100,
            passed=True,
            issues=[],
            metrics={
                "skipped": True,
                "reason": "runtime check disabled or missing command",
            },
            summary=f"Runtime check skipped for {comp_id}",
            reviewer_type="pipeline",
            strategy="pipeline",
        )

    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", comp_id or "component")
    with tempfile.TemporaryDirectory(prefix="uce_runtime_") as temp_dir:
        component_path = os.path.join(temp_dir, f"{safe_id}.tsx")
        styles_path = os.path.join(temp_dir, f"{safe_id}.module.css")
        context_path = os.path.join(temp_dir, "context.json")

        with open(component_path, "w", encoding="utf-8") as handle:
            handle.write(component_code or "")
        with open(styles_path, "w", encoding="utf-8") as handle:
            handle.write(styles_code or "")

        context = {
            "component_id": comp_id,
            "component_path": component_path,
            "styles_path": styles_path,
        }
        with open(context_path, "w", encoding="utf-8") as handle:
            json.dump(context, handle)

        formatted_command = command
        for key, value in {
            "component_path": component_path,
            "styles_path": styles_path,
            "component_id": comp_id,
            "context_path": context_path,
            "temp_dir": temp_dir,
        }.items():
            formatted_command = formatted_command.replace(f"{{{key}}}", value)

        start_time = time.monotonic()
        try:
            result = subprocess.run(
                formatted_command,
                shell=settings.get("shell", True),
                cwd=settings.get("cwd") or None,
                capture_output=True,
                text=True,
                timeout=settings.get("timeout_seconds", 60),
            )
            duration_ms = int((time.monotonic() - start_time) * 1000)
        except subprocess.TimeoutExpired:
            issues = [
                {
                    "issue_id": str(uuid4()),
                    "category": "runtime_check",
                    "severity": "critical",
                    "title": "Runtime check timed out",
                    "description": (
                        f"Command timed out after {settings.get('timeout_seconds', 60)} seconds."
                    ),
                    "location": "",
                    "suggestion": "Increase timeout or optimize the runtime check command.",
                    "auto_fixable": False,
                }
            ]
            score = _score_from_issues("runtime_check", issues)
            return _build_review_result(
                reviewer="runtime_check",
                score=score,
                passed=False,
                issues=issues,
                metrics={
                    "command": formatted_command,
                    "timeout_seconds": settings.get("timeout_seconds", 60),
                    "duration_ms": int((time.monotonic() - start_time) * 1000),
                },
                summary=f"Runtime check timed out for {comp_id}",
                reviewer_type="pipeline",
                strategy="pipeline",
            )
        except Exception as exc:
            issues = [
                {
                    "issue_id": str(uuid4()),
                    "category": "runtime_check",
                    "severity": "major",
                    "title": "Runtime check failed to execute",
                    "description": str(exc),
                    "location": "",
                    "suggestion": "Verify runtime check command and environment.",
                    "auto_fixable": False,
                }
            ]
            score = _score_from_issues("runtime_check", issues)
            return _build_review_result(
                reviewer="runtime_check",
                score=score,
                passed=False,
                issues=issues,
                metrics={"command": formatted_command},
                summary=f"Runtime check execution error for {comp_id}",
                reviewer_type="pipeline",
                strategy="pipeline",
            )

        stdout = _truncate_text(result.stdout)
        stderr = _truncate_text(result.stderr)

        if result.returncode != 0:
            issues = [
                {
                    "issue_id": str(uuid4()),
                    "category": "runtime_check",
                    "severity": "major",
                    "title": "Runtime check failed",
                    "description": stderr or stdout or f"Exit code {result.returncode}.",
                    "location": "",
                    "suggestion": "Fix compilation/runtime errors reported by the check.",
                    "auto_fixable": False,
                }
            ]
            score = _score_from_issues("runtime_check", issues)
            return _build_review_result(
                reviewer="runtime_check",
                score=score,
                passed=False,
                issues=issues,
                metrics={
                    "command": formatted_command,
                    "exit_code": result.returncode,
                    "duration_ms": duration_ms,
                    "stdout": stdout,
                    "stderr": stderr,
                },
                summary=f"Runtime check failed for {comp_id}",
                reviewer_type="pipeline",
                strategy="pipeline",
            )

        return _build_review_result(
            reviewer="runtime_check",
            score=100,
            passed=True,
            issues=[],
            metrics={
                "command": formatted_command,
                "exit_code": result.returncode,
                "duration_ms": duration_ms,
                "stdout": stdout,
                "stderr": stderr,
            },
            summary=f"Runtime check passed for {comp_id}",
            reviewer_type="pipeline",
            strategy="pipeline",
        )


# ============================================================================
# 审查聚合
# ============================================================================

def aggregate_reviews(state: MigrationGraphState) -> Dict[str, Any]:
    """
    聚合所有审查结果，决定是否需要人工审查
    
    决策逻辑:
    1. 最大严重级别 >= major -> regenerate
    2. 有问题且满足 auto_fix 条件 -> auto_fix
    3. 命中强制人工审查列表 -> human_review
    4. 有问题或平均分低于阈值 -> human_review
    5. 否则 -> auto_approve
    """
    components = state.get("components", {})
    updated_components = dict(components)
    pending_human_review = []
    pending_auto_fix = []
    auto_fix_settings = _auto_fix_settings(state)
    
    for comp_id, comp_data in components.items():
        review = comp_data.get("review", {})
        
        if not review:
            continue
        
        # 获取各项审查结果
        code_quality = review.get("code_quality", {})
        bdl_compliance = review.get("bdl_compliance", {})
        function_parity = review.get("function_parity", {})
        accessibility = review.get("accessibility", {})
        security = review.get("security", {})
        editor_schema = review.get("editor_schema", {})
        runtime_check = review.get("runtime_check", {})
        
        scores = []
        if code_quality:
            scores.append(code_quality.get("score", 0))
        if bdl_compliance:
            scores.append(bdl_compliance.get("score", 0))
        if function_parity:
            scores.append(function_parity.get("score", 0))
        if accessibility:
            scores.append(accessibility.get("score", 0))
        if security:
            scores.append(security.get("score", 0))
        if editor_schema:
            scores.append(editor_schema.get("score", 0))
        if runtime_check:
            scores.append(runtime_check.get("score", 0))
        
        overall_score = sum(scores) / len(scores) if scores else 0
        
        # 检查问题严重级别
        all_issues = []
        severity_counts = {"critical": 0, "major": 0, "minor": 0, "suggestion": 0}
        review_items = [
            ("code_quality", code_quality),
            ("bdl_compliance", bdl_compliance),
            ("function_parity", function_parity),
            ("accessibility", accessibility),
            ("security", security),
            ("editor_schema", editor_schema),
            ("runtime_check", runtime_check),
        ]
        for review_name, result in review_items:
            if not isinstance(result, dict) or not result:
                continue
            issues = list(result.get("issues", [])) if isinstance(result.get("issues"), list) else []
            if result.get("passed") is False and not issues:
                summary = str(result.get("summary", "")).strip()
                issues.append({
                    "issue_id": str(uuid4()),
                    "category": review_name,
                    "severity": "major",
                    "title": "Review failed",
                    "description": summary or "Review did not pass but no issues were reported.",
                    "location": "review",
                    "suggestion": "Inspect the review output or re-run the review with diagnostics enabled.",
                    "auto_fixable": False,
                })
            for issue in issues:
                if isinstance(issue, dict):
                    severity = str(issue.get("severity", "minor")).lower()
                    if severity not in severity_counts:
                        severity = "minor"
                    severity_counts[severity] += 1
                    all_issues.append(issue)
        has_issues = len(all_issues) > 0
        max_severity = _max_issue_severity(all_issues)
        max_severity_rank = _severity_rank(max_severity)
        
        # 检查是否在强制人工审查列表
        requires_human = False
        for pattern in REVIEW_CONFIG["require_human_for"]:
            if pattern.endswith("/*"):
                prefix = pattern[:-2]
                if comp_id.startswith(prefix):
                    requires_human = True
                    break
        
        max_allowed = _severity_rank(auto_fix_settings["max_severity"])
        attempts = int(comp_data.get("auto_fix_attempts", 0))
        auto_fix_allowed = (
            auto_fix_settings["enabled"]
            and has_issues
            and attempts < auto_fix_settings["max_attempts"]
            and max_severity_rank <= max_allowed
        )

        # 决定下一步动作
        decision = "auto_approve"
        if max_severity_rank >= _severity_rank("major"):
            decision = "regenerate"
        elif auto_fix_allowed:
            decision = "auto_fix"
        elif requires_human or has_issues or overall_score < REVIEW_CONFIG["auto_approve_threshold"]:
            decision = "human_review"
        
        # 更新聚合结果
        updated_components[comp_id]["review"]["aggregated"] = {
            "overall_score": overall_score,
            "auto_approved": decision == "auto_approve",
            "requires_human_review": requires_human or decision == "human_review",
            "decision": decision,
            "issues": all_issues,
            "total_issues": len(all_issues),
            "critical_issues": severity_counts["critical"],
            "major_issues": severity_counts["major"],
            "minor_issues": severity_counts["minor"],
            "suggestion_issues": severity_counts["suggestion"],
            "timestamp": datetime.now().isoformat()
        }

        if decision == "human_review":
            pending_human_review.append(comp_id)
            updated_components[comp_id]["status"] = "human_reviewing"
        elif decision == "auto_fix":
            pending_auto_fix.append(comp_id)
            updated_components[comp_id]["status"] = "auto_fixing"
        elif decision == "regenerate":
            updated_components[comp_id]["status"] = "rejected"
        else:
            updated_components[comp_id]["status"] = "approved"
    
    return {
        "components": updated_components,
        "pending_human_review": pending_human_review,
        "pending_auto_fix": pending_auto_fix,
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
    
    for category in [
        "code_quality",
        "bdl_compliance",
        "function_parity",
        "accessibility",
        "security",
        "editor_schema",
        "runtime_check",
    ]:
        category_review = review.get(category, {})
        issues = list(category_review.get("issues", [])) if isinstance(category_review, dict) else []
        if isinstance(category_review, dict) and category_review.get("passed") is False and not issues:
            issues.append({
                "severity": "major",
                "title": "Review failed",
                "suggestion": "Inspect the review output or re-run the review.",
            })
        for issue in issues:
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
            error_result = _build_review_result(
                reviewer="code_quality",
                score=0,
                passed=False,
                issues=[],
                metrics={},
                summary=f"error: {e}",
                reviewer_type="pipeline",
                strategy="pipeline",
            )
            error_result["error"] = str(e)
            results[comp_id] = {"code_quality": error_result}
    
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
            error_result = _build_review_result(
                reviewer="bdl_compliance",
                score=0,
                passed=False,
                issues=[],
                metrics={},
                summary=f"error: {e}",
                reviewer_type="pipeline",
                strategy="pipeline",
            )
            error_result["error"] = str(e)
            results[comp_id] = {"bdl_compliance": error_result}
    
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
            error_result = _build_review_result(
                reviewer="function_parity",
                score=0,
                passed=False,
                issues=[],
                metrics={},
                summary=f"error: {e}",
                reviewer_type="pipeline",
                strategy="pipeline",
            )
            error_result["error"] = str(e)
            results[comp_id] = {"function_parity": error_result}
    
    return {"parallel_review_results": results}


async def accessibility_review_v2(state: MigrationGraphState) -> Dict[str, Any]:
    """
    可访问性审查节点 (Send API 版本)
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
            review_result = _run_accessibility_review(component_code, styles_code)
            results[comp_id] = {"accessibility": review_result}
        except Exception as e:
            error_result = _build_review_result(
                reviewer="accessibility",
                score=0,
                passed=False,
                issues=[],
                metrics={},
                summary=f"error: {e}",
                reviewer_type="pipeline",
                strategy="pipeline",
            )
            error_result["error"] = str(e)
            results[comp_id] = {"accessibility": error_result}

    return {"parallel_review_results": results}


async def security_review_v2(state: MigrationGraphState) -> Dict[str, Any]:
    """
    安全审查节点 (Send API 版本)
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
            review_result = _run_security_review(component_code, styles_code)
            results[comp_id] = {"security": review_result}
        except Exception as e:
            error_result = _build_review_result(
                reviewer="security",
                score=0,
                passed=False,
                issues=[],
                metrics={},
                summary=f"error: {e}",
                reviewer_type="pipeline",
                strategy="pipeline",
            )
            error_result["error"] = str(e)
            results[comp_id] = {"security": error_result}

    return {"parallel_review_results": results}


async def editor_schema_review_v2(state: MigrationGraphState) -> Dict[str, Any]:
    """
    编辑器 Schema 审查节点 (Send API 版本)
    """
    components = state.get("components", {})
    results = {}

    for comp_id, comp_data in components.items():
        if comp_data.get("status") != "generating":
            continue

        cms_config = comp_data.get("cms_config", {})

        try:
            review_result = _run_editor_schema_review(comp_id, cms_config)
            results[comp_id] = {"editor_schema": review_result}
        except Exception as e:
            error_result = _build_review_result(
                reviewer="editor_schema",
                score=0,
                passed=False,
                issues=[],
                metrics={},
                summary=f"error: {e}",
                reviewer_type="pipeline",
                strategy="pipeline",
            )
            error_result["error"] = str(e)
            results[comp_id] = {"editor_schema": error_result}

    return {"parallel_review_results": results}


async def runtime_check_review_v2(state: MigrationGraphState) -> Dict[str, Any]:
    """
    运行检查节点 (Send API 版本)
    """
    components = state.get("components", {})
    results = {}
    settings = _runtime_check_settings(state)

    for comp_id, comp_data in components.items():
        if comp_data.get("status") != "generating":
            continue

        react_component = comp_data.get("react_component", {})
        component_code = react_component.get("component_code", "")
        styles_code = react_component.get("styles_code", "")

        if not component_code:
            continue

        try:
            review_result = await asyncio.to_thread(
                _run_runtime_check, comp_id, component_code, styles_code, settings
            )
            results[comp_id] = {"runtime_check": review_result}
        except Exception as e:
            error_result = _build_review_result(
                reviewer="runtime_check",
                score=0,
                passed=False,
                issues=[],
                metrics={},
                summary=f"error: {e}",
                reviewer_type="pipeline",
                strategy="pipeline",
            )
            error_result["error"] = str(e)
            results[comp_id] = {"runtime_check": error_result}

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

    review_nodes: List[str] = []
    if _review_enabled(state, "code_quality"):
        review_nodes.append("code_quality_review")
    if _review_enabled(state, "bdl_compliance"):
        review_nodes.append("bdl_compliance_review")
    if _review_enabled(state, "function_parity"):
        review_nodes.append("function_parity_review")
    if _review_enabled(state, "accessibility"):
        review_nodes.append("accessibility_review")
    if _review_enabled(state, "security"):
        review_nodes.append("security_review")
    if _review_enabled(state, "editor_schema"):
        review_nodes.append("editor_schema_review")
    if _runtime_check_enabled(state):
        review_nodes.append("runtime_check_review")

    if not review_nodes:
        return [Send("merge_review_results", state)]

    return [Send(node, state) for node in review_nodes]
