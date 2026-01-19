"""
最终化节点实现
负责输出生成和报告生成

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

import json
import os
from datetime import datetime
from typing import Any, Dict

from src.core.state import MigrationGraphState, Phase


# ============================================================================
# 最终化节点
# ============================================================================

def finalize(state: MigrationGraphState) -> Dict[str, Any]:
    """
    最终化节点
    
    执行:
    1. 写入所有生成的 React 组件文件
    2. 写入 CMS 配置文件
    3. 写入页面 JSON 文件
    4. 生成 package.json 和 index 文件
    """
    config = state.get("config", {})
    components = state.get("components", {})
    configs = state.get("configs", {})
    pages = state.get("pages", {})
    
    output_dir = config.get("output_dir", "./output")
    
    # 创建输出目录结构
    components_dir = os.path.join(output_dir, "components")
    configs_dir = os.path.join(output_dir, "configs")
    pages_dir = os.path.join(output_dir, "pages")
    
    for dir_path in [components_dir, configs_dir, pages_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    written_files = []
    errors = list(state.get("errors", []))
    
    # 写入 React 组件
    for comp_id, comp_data in components.items():
        if comp_data.get("status") != "approved":
            continue
        
        react_component = comp_data.get("react_component", {})
        component_name = react_component.get("component_name", comp_id)
        
        comp_dir = os.path.join(components_dir, component_name)
        os.makedirs(comp_dir, exist_ok=True)
        
        try:
            # 写入组件文件
            component_code = react_component.get("component_code", "")
            if component_code:
                file_path = os.path.join(comp_dir, f"{component_name}.tsx")
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(component_code)
                written_files.append(file_path)
            
            # 写入样式文件
            styles_code = react_component.get("styles_code", "")
            if styles_code:
                file_path = os.path.join(comp_dir, f"{component_name}.module.css")
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(styles_code)
                written_files.append(file_path)
            
            # 写入 index 文件
            index_code = react_component.get("index_code", "")
            if not index_code:
                index_code = f"export * from './{component_name}';\nexport {{ default }} from './{component_name}';"
            file_path = os.path.join(comp_dir, "index.ts")
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(index_code)
            written_files.append(file_path)
            
        except Exception as e:
            errors.append({
                "error_id": str(id(e)),
                "severity": "recoverable",
                "error_type": "FileWriteError",
                "message": f"Failed to write component {comp_id}: {str(e)}",
                "component_id": comp_id
            })
    
    # 写入 CMS 配置
    for config_id, config_data in configs.items():
        try:
            file_path = os.path.join(configs_dir, f"{config_id}.config.json")
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            written_files.append(file_path)
        except Exception as e:
            errors.append({
                "error_id": str(id(e)),
                "severity": "recoverable",
                "error_type": "FileWriteError",
                "message": f"Failed to write config {config_id}: {str(e)}"
            })
    
    # 写入页面 JSON
    for page_id, page_data in pages.items():
        cms_json = page_data.get("cms_json", {})
        if not cms_json:
            continue
        
        try:
            slug = cms_json.get("page", {}).get("slug", page_id)
            safe_name = slug.replace("/", "_").strip("_") or page_id
            file_path = os.path.join(pages_dir, f"{safe_name}.page.json")
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(cms_json, f, indent=2, ensure_ascii=False)
            written_files.append(file_path)
        except Exception as e:
            errors.append({
                "error_id": str(id(e)),
                "severity": "recoverable",
                "error_type": "FileWriteError",
                "message": f"Failed to write page {page_id}: {str(e)}"
            })
    
    # 生成主 index 文件
    _generate_main_index(components_dir, components)
    
    # 生成 package.json
    _generate_package_json(output_dir, components)
    
    # 更新统计
    stats = dict(state.get("stats", {}))
    stats["written_files"] = len(written_files)
    stats["end_time"] = datetime.now().isoformat()
    
    return {
        "stats": stats,
        "errors": errors,
        "output_files": written_files,
        "current_phase": Phase.COMPLETED.value
    }


def _generate_main_index(components_dir: str, components: Dict) -> None:
    """生成组件库主入口文件"""
    exports = []
    
    for comp_id, comp_data in components.items():
        if comp_data.get("status") != "approved":
            continue
        
        component_name = comp_data.get("react_component", {}).get("component_name", "")
        if component_name:
            exports.append(f"export * from './{component_name}';")
    
    if exports:
        index_content = "// Auto-generated component library index\n\n"
        index_content += "\n".join(sorted(exports))
        
        with open(os.path.join(components_dir, "index.ts"), 'w', encoding='utf-8') as f:
            f.write(index_content)


def _generate_package_json(output_dir: str, components: Dict) -> None:
    """生成 package.json"""
    # [CUSTOMIZE] React 版本根据项目需求调整
    peer_deps = {
        "react": "^18.0.0",
        "react-dom": "^18.0.0"
    }
    
    for comp_data in components.values():
        react_component = comp_data.get("react_component", {})
        deps = react_component.get("peer_dependencies", {})
        peer_deps.update(deps)
    
    # [CUSTOMIZE] package.json 配置需要根据实际项目调整
    package_json = {
        "name": "@mysite/components",  # [CUSTOMIZE] 替换为实际包名
        "version": "1.0.0",
        "description": "Migrated component library from AEM to React",
        "main": "components/index.ts",
        "types": "components/index.ts",
        "scripts": {
            "build": "tsc",
            "test": "jest",
            "storybook": "storybook dev -p 6006",
            "lint": "eslint . --ext .ts,.tsx"
        },
        "peerDependencies": peer_deps,
        "devDependencies": {
            "typescript": "^5.0.0",
            "@types/react": "^18.0.0",
            "@types/react-dom": "^18.0.0",
            "jest": "^29.0.0",
            "@testing-library/react": "^14.0.0",
            "@storybook/react": "^7.0.0",
            "eslint": "^8.0.0",
            "@typescript-eslint/eslint-plugin": "^6.0.0"
        },
        "keywords": ["react", "components", "bdl", "cms"],
        "license": "MIT"
    }
    
    with open(os.path.join(output_dir, "package.json"), 'w', encoding='utf-8') as f:
        json.dump(package_json, f, indent=2)


# ============================================================================
# 报告生成节点
# ============================================================================

def generate_report(state: MigrationGraphState) -> Dict[str, Any]:
    """
    生成迁移报告节点
    
    创建详细的迁移报告包括:
    - 统计摘要
    - 组件状态详情
    - 错误和警告汇总
    - 审查结果汇总
    """
    config = state.get("config", {})
    components = state.get("components", {})
    pages = state.get("pages", {})
    stats = state.get("stats", {})
    errors = state.get("errors", [])
    warnings = state.get("warnings", [])
    
    output_dir = config.get("output_dir", "./output")
    
    # 计算时长
    start_time = stats.get("start_time", "")
    end_time = stats.get("end_time", "")
    duration = ""
    
    if start_time and end_time:
        try:
            start = datetime.fromisoformat(start_time)
            end = datetime.fromisoformat(end_time)
            duration_seconds = (end - start).total_seconds()
            duration = f"{duration_seconds:.2f} seconds"
        except:
            pass
    
    # 组件状态统计
    component_stats = {
        "approved": 0,
        "rejected": 0,
        "failed": 0,
        "pending": 0
    }
    
    component_details = []
    
    for comp_id, comp_data in components.items():
        status = comp_data.get("status", "pending")
        component_stats[status] = component_stats.get(status, 0) + 1
        
        review = comp_data.get("review", {})
        aggregated = review.get("aggregated", {})
        
        component_details.append({
            "component_id": comp_id,
            "component_name": comp_data.get("react_component", {}).get("component_name", ""),
            "status": status,
            "scores": {
                "code_quality": review.get("code_quality", {}).get("score"),
                "bdl_compliance": review.get("bdl_compliance", {}).get("score"),
                "function_parity": review.get("function_parity", {}).get("score"),
                "overall": aggregated.get("overall_score")
            },
            "issues_count": aggregated.get("total_issues", 0),
            "human_reviewed": bool(review.get("human_decision"))
        })
    
    # 审查评分统计
    review_scores = {
        "code_quality": [],
        "bdl_compliance": [],
        "function_parity": []
    }
    
    for comp_data in components.values():
        review = comp_data.get("review", {})
        for key in review_scores:
            score = review.get(key, {}).get("score")
            if score is not None:
                review_scores[key].append(score)
    
    avg_scores = {}
    for key, scores in review_scores.items():
        if scores:
            avg_scores[key] = sum(scores) / len(scores)
    
    # 构建报告
    report = {
        "report_generated_at": datetime.now().isoformat(),
        "session_id": state.get("session_id", ""),
        "duration": duration,
        
        "summary": {
            "total_components": stats.get("total_components", 0),
            "generated_components": stats.get("generated_components", 0),
            "approved_components": component_stats.get("approved", 0),
            "rejected_components": component_stats.get("rejected", 0),
            "failed_components": component_stats.get("failed", 0),
            "total_pages": stats.get("total_pages", 0),
            "migrated_pages": stats.get("migrated_pages", 0),
            "human_review_count": stats.get("human_review_count", 0),
            "total_errors": len(errors),
            "total_warnings": len(warnings)
        },
        
        "review_scores": {
            "average_code_quality": avg_scores.get("code_quality"),
            "average_bdl_compliance": avg_scores.get("bdl_compliance"),
            "average_function_parity": avg_scores.get("function_parity")
        },
        
        "components": component_details,
        
        "errors": [
            {
                "severity": e.get("severity"),
                "type": e.get("error_type"),
                "message": e.get("message"),
                "component_id": e.get("component_id")
            }
            for e in errors
        ],
        
        "warnings": [
            {
                "message": w.get("message"),
                "component_id": w.get("component_id")
            }
            for w in warnings
        ],
        
        "output": {
            "components_dir": os.path.join(output_dir, "components"),
            "configs_dir": os.path.join(output_dir, "configs"),
            "pages_dir": os.path.join(output_dir, "pages"),
            "written_files": stats.get("written_files", 0)
        }
    }
    
    # 写入报告文件
    report_path = os.path.join(output_dir, "migration_report.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # 生成 Markdown 报告
    md_report = _generate_markdown_report(report)
    md_path = os.path.join(output_dir, "migration_report.md")
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md_report)
    
    return {
        "report": report,
        "report_path": report_path
    }


def _generate_markdown_report(report: Dict) -> str:
    """生成 Markdown 格式报告"""
    summary = report.get("summary", {})
    scores = report.get("review_scores", {})
    
    md = f"""# uce-adui Migration Report

**Generated:** {report.get("report_generated_at", "")}
**Duration:** {report.get("duration", "N/A")}
**Session ID:** {report.get("session_id", "")}

## Summary

| Metric | Value |
|--------|-------|
| Total Components | {summary.get("total_components", 0)} |
| Generated | {summary.get("generated_components", 0)} |
| Approved | {summary.get("approved_components", 0)} |
| Rejected | {summary.get("rejected_components", 0)} |
| Failed | {summary.get("failed_components", 0)} |
| Total Pages | {summary.get("total_pages", 0)} |
| Migrated Pages | {summary.get("migrated_pages", 0)} |
| Human Reviews | {summary.get("human_review_count", 0)} |
| Errors | {summary.get("total_errors", 0)} |
| Warnings | {summary.get("total_warnings", 0)} |

## Review Scores

| Category | Average Score |
|----------|---------------|
| Code Quality | {scores.get("average_code_quality", "N/A"):.1f if scores.get("average_code_quality") else "N/A"} |
| BDL Compliance | {scores.get("average_bdl_compliance", "N/A"):.1f if scores.get("average_bdl_compliance") else "N/A"} |
| Function Parity | {scores.get("average_function_parity", "N/A"):.1f if scores.get("average_function_parity") else "N/A"} |

## Components

| Component | Status | Code Quality | BDL | Function | Overall |
|-----------|--------|--------------|-----|----------|---------|
"""

    for comp in report.get("components", []):
        scores_data = comp.get("scores", {})
        md += f"| {comp.get('component_name', comp.get('component_id'))} | {comp.get('status')} | "
        md += f"{scores_data.get('code_quality', '-')} | "
        md += f"{scores_data.get('bdl_compliance', '-')} | "
        md += f"{scores_data.get('function_parity', '-')} | "
        md += f"{scores_data.get('overall', '-'):.1f if scores_data.get('overall') else '-'} |\n"

    # Errors section
    errors = report.get("errors", [])
    if errors:
        md += "\n## Errors\n\n"
        for error in errors:
            md += f"- **[{error.get('severity')}]** {error.get('type')}: {error.get('message')}\n"
    
    # Warnings section
    warnings = report.get("warnings", [])
    if warnings:
        md += "\n## Warnings\n\n"
        for warning in warnings[:20]:  # Limit to 20
            md += f"- {warning.get('message')}\n"
        if len(warnings) > 20:
            md += f"\n*... and {len(warnings) - 20} more warnings*\n"
    
    md += f"""
## Output

- Components: `{report.get("output", {}).get("components_dir")}`
- Configs: `{report.get("output", {}).get("configs_dir")}`
- Pages: `{report.get("output", {}).get("pages_dir")}`
- Files Written: {report.get("output", {}).get("written_files", 0)}
"""

    return md
