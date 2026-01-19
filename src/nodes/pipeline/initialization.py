"""
初始化节点实现
负责系统初始化和 BDL 规范加载

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
from uuid import uuid4

from src.core.state import MigrationGraphState, Phase


# ============================================================================
# 默认 BDL 规范
# ============================================================================
# 
# ⚠️ TODO [CUSTOMIZE]: 这是示例 BDL 规范，实际项目中需要替换
# 
# 获取真实 BDL 规范的方式:
# 1. 联系 HSBC 设计系统团队获取官方规范
# 2. 从 Figma Design Tokens 导出
# 3. 从 @hsbc/bdl-components NPM 包分析
# 4. 使用 --bdl-spec 参数指定自定义规范文件
#
# 标记说明:
# - [CUSTOMIZE] = 需要根据实际项目定制
# - [PLACEHOLDER] = 占位符，需要实现
# - [EXAMPLE] = 示例代码，需要替换
# ============================================================================

DEFAULT_BDL_SPEC = {  # [EXAMPLE] 整个对象需要替换为真实 BDL 规范
    "version": "2.0.0",
    "design_tokens": {
        "colors": {
            "primary": {
                "50": "#fff1f1",
                "100": "#ffe1e1",
                "500": "#db0011",  # HSBC Red
                "600": "#c4000f",
                "700": "#9a000c"
            },
            "neutral": {
                "50": "#f8f9fa",
                "100": "#f1f3f5",
                "200": "#e9ecef",
                "500": "#868e96",
                "700": "#495057",
                "900": "#212529"
            },
            "semantic": {
                "success": "#2e7d32",
                "warning": "#ed6c02",
                "error": "#d32f2f",
                "info": "#0288d1"
            }
        },
        "spacing": {
            "xs": "4px",
            "sm": "8px",
            "md": "16px",
            "lg": "24px",
            "xl": "32px",
            "2xl": "48px",
            "3xl": "64px"
        },
        "typography": {
            "fontFamily": {
                "primary": "Univers, Arial, sans-serif",
                "secondary": "Georgia, serif",
                "mono": "Consolas, monospace"
            },
            "fontSize": {
                "xs": "12px",
                "sm": "14px",
                "base": "16px",
                "lg": "18px",
                "xl": "20px",
                "2xl": "24px",
                "3xl": "30px",
                "4xl": "36px"
            },
            "fontWeight": {
                "normal": 400,
                "medium": 500,
                "semibold": 600,
                "bold": 700
            },
            "lineHeight": {
                "tight": 1.25,
                "normal": 1.5,
                "relaxed": 1.75
            }
        },
        "breakpoints": {
            "xs": "0px",
            "sm": "576px",
            "md": "768px",
            "lg": "992px",
            "xl": "1200px",
            "2xl": "1400px"
        },
        "shadows": {
            "sm": "0 1px 2px rgba(0, 0, 0, 0.05)",
            "md": "0 4px 6px rgba(0, 0, 0, 0.1)",
            "lg": "0 10px 15px rgba(0, 0, 0, 0.1)",
            "xl": "0 20px 25px rgba(0, 0, 0, 0.15)"
        },
        "borderRadius": {
            "none": "0",
            "sm": "2px",
            "md": "4px",
            "lg": "8px",
            "xl": "12px",
            "full": "9999px"
        }
    },
    "components": {
        "Button": {
            "props": ["variant", "size", "disabled", "loading", "onClick", "children"],
            "variants": ["primary", "secondary", "outline", "ghost", "link"],
            "sizes": ["sm", "md", "lg"]
        },
        "Input": {
            "props": ["type", "value", "onChange", "placeholder", "disabled", "error", "helperText"],
            "types": ["text", "email", "password", "number", "tel"]
        },
        "Card": {
            "props": ["variant", "padding", "shadow", "children"],
            "variants": ["elevated", "outlined", "filled"]
        },
        "Typography": {
            "props": ["variant", "component", "color", "align", "children"],
            "variants": ["h1", "h2", "h3", "h4", "h5", "h6", "body1", "body2", "caption", "overline"]
        },
        "Image": {
            "props": ["src", "alt", "width", "height", "objectFit", "loading"],
            "objectFit": ["cover", "contain", "fill", "none"]
        },
        "Link": {
            "props": ["href", "target", "children", "variant"],
            "variants": ["default", "underline", "button"]
        },
        "Hero": {
            "props": ["title", "subtitle", "backgroundImage", "cta", "variant"],
            "variants": ["centered", "left-aligned", "split"]
        },
        "Grid": {
            "props": ["columns", "gap", "alignItems", "justifyContent", "children"]
        },
        "Container": {
            "props": ["maxWidth", "padding", "children"],
            "maxWidths": ["sm", "md", "lg", "xl", "full"]
        },
        "Accordion": {
            "props": ["items", "allowMultiple", "defaultExpanded"],
            "itemProps": ["title", "content", "disabled"]
        },
        "Tabs": {
            "props": ["items", "defaultValue", "onChange", "variant"],
            "variants": ["default", "pills", "underline"]
        },
        "Modal": {
            "props": ["open", "onClose", "title", "size", "children"],
            "sizes": ["sm", "md", "lg", "xl", "full"]
        },
        "Form": {
            "props": ["onSubmit", "children", "validationSchema"]
        },
        "Select": {
            "props": ["options", "value", "onChange", "placeholder", "multiple", "disabled"]
        },
        "Checkbox": {
            "props": ["checked", "onChange", "label", "disabled", "indeterminate"]
        },
        "Radio": {
            "props": ["options", "value", "onChange", "name", "disabled"]
        },
        "Table": {
            "props": ["columns", "data", "sortable", "pagination"]
        },
        "Carousel": {
            "props": ["items", "autoPlay", "interval", "showDots", "showArrows"]
        },
        "Breadcrumb": {
            "props": ["items", "separator"]
        },
        "Navigation": {
            "props": ["items", "orientation", "variant"],
            "variants": ["horizontal", "vertical"]
        }
    },
    "patterns": {
        "naming": {
            "components": "PascalCase",
            "props": "camelCase",
            "events": "onAction (e.g., onClick, onChange)",
            "css_classes": "kebab-case",
            "css_modules": "camelCase"
        },
        "file_structure": {
            "component": "{ComponentName}/index.ts",
            "styles": "{ComponentName}/{ComponentName}.module.css",
            "types": "{ComponentName}/{ComponentName}.types.ts",
            "stories": "{ComponentName}/{ComponentName}.stories.tsx",
            "tests": "{ComponentName}/{ComponentName}.test.tsx"
        }
    },
    "accessibility": {
        "required": [
            "All interactive elements must be keyboard accessible",
            "All images must have alt text",
            "Color contrast must meet WCAG AA standards",
            "Focus states must be visible",
            "Semantic HTML elements should be used"
        ],
        "aria_patterns": {
            "Modal": ["aria-modal", "aria-labelledby", "role=dialog"],
            "Accordion": ["aria-expanded", "aria-controls"],
            "Tabs": ["role=tablist", "role=tab", "role=tabpanel", "aria-selected"],
            "Menu": ["role=menu", "role=menuitem", "aria-expanded"]
        }
    }
}


# ============================================================================
# 初始化节点
# ============================================================================

def initialize(state: MigrationGraphState) -> Dict[str, Any]:
    """
    初始化节点
    
    执行:
    1. 验证输入参数
    2. 初始化统计数据
    3. 准备工作目录
    4. 记录会话开始
    """
    source_path = state.get("source_path", "")
    config = state.get("config", {})
    
    errors = []
    warnings = []
    
    # 验证源路径
    if not source_path:
        errors.append({
            "error_id": str(uuid4()),
            "severity": "fatal",
            "error_type": "ConfigurationError",
            "message": "source_path is required"
        })
    elif not os.path.exists(source_path):
        errors.append({
            "error_id": str(uuid4()),
            "severity": "fatal",
            "error_type": "ConfigurationError",
            "message": f"Source path does not exist: {source_path}"
        })
    
    # 验证输出目录
    output_dir = config.get("output_dir", "./output")
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception as e:
            warnings.append({
                "warning_id": str(uuid4()),
                "message": f"Could not create output directory: {e}"
            })
    
    # 验证 AEM 页面 JSON 路径
    aem_page_json_paths = state.get("aem_page_json_paths", [])
    valid_paths = []
    for path in aem_page_json_paths:
        if os.path.exists(path):
            valid_paths.append(path)
        else:
            warnings.append({
                "warning_id": str(uuid4()),
                "message": f"AEM page JSON not found: {path}"
            })
    
    # 初始化统计
    stats = {
        "total_components": 0,
        "parsed_components": 0,
        "generated_components": 0,
        "approved_components": 0,
        "failed_components": 0,
        "total_pages": len(valid_paths),
        "migrated_pages": 0,
        "human_review_count": 0,
        "auto_fix_count": 0,
        "start_time": datetime.now().isoformat()
    }
    
    # 确定初始阶段
    current_phase = Phase.INITIALIZATION.value if not errors else Phase.FAILED.value
    
    return {
        "current_phase": current_phase,
        "aem_page_json_paths": valid_paths,
        "stats": stats,
        "errors": errors,
        "warnings": warnings,
        "config": {
            **config,
            "output_dir": output_dir,
            "initialized_at": datetime.now().isoformat()
        }
    }


# ============================================================================
# BDL 规范加载节点
# ============================================================================

def load_bdl_spec(state: MigrationGraphState) -> Dict[str, Any]:
    """
    加载 BDL 规范节点
    
    支持多种加载方式:
    1. 从 JSON 文件加载 (推荐 - 生产环境)
    2. 从 URL 加载 (设计系统 API)
    3. 从 NPM 包分析 (逆向工程)
    4. 使用默认规范 (仅演示)
    
    ⚠️ 注意: DEFAULT_BDL_SPEC 只是示例，实际项目中应该:
    - 从 HSBC 设计系统团队获取官方规范
    - 从 Figma Design Tokens 导出
    - 从现有 @hsbc/bdl-components 包分析
    """
    config = state.get("config", {})
    warnings = list(state.get("warnings", []))
    
    bdl_spec = None
    bdl_source = "default"  # 记录规范来源
    
    # =========================================
    # 方式1: 从 JSON 文件加载 (推荐)
    # =========================================
    bdl_spec_path = config.get("bdl_spec_path")
    if bdl_spec_path and os.path.exists(bdl_spec_path):
        try:
            with open(bdl_spec_path, 'r', encoding='utf-8') as f:
                bdl_spec = json.load(f)
                bdl_source = f"file:{bdl_spec_path}"
        except Exception as e:
            warnings.append({
                "warning_id": str(uuid4()),
                "message": f"Failed to load BDL spec from {bdl_spec_path}: {e}"
            })
    
    # =========================================
    # 方式2: 从 URL 加载 (设计系统 API)
    # =========================================
    if not bdl_spec:
        bdl_spec_url = config.get("bdl_spec_url")
        if bdl_spec_url:
            try:
                import httpx
                response = httpx.get(bdl_spec_url, timeout=30)
                response.raise_for_status()
                bdl_spec = response.json()
                bdl_source = f"url:{bdl_spec_url}"
            except Exception as e:
                warnings.append({
                    "warning_id": str(uuid4()),
                    "message": f"Failed to load BDL spec from URL {bdl_spec_url}: {e}"
                })
    
    # =========================================
    # 方式3: 使用默认规范 (仅作演示)
    # =========================================
    if not bdl_spec:
        bdl_spec = DEFAULT_BDL_SPEC
        bdl_source = "default"
        warnings.append({
            "warning_id": str(uuid4()),
            "message": (
                "Using DEFAULT_BDL_SPEC (demo only). "
                "For production, please provide official BDL spec via: "
                "--bdl-spec <path> or config.bdl_spec_url"
            )
        })
    
    # 验证规范结构
    required_sections = ["design_tokens", "components", "patterns"]
    for section in required_sections:
        if section not in bdl_spec:
            warnings.append({
                "warning_id": str(uuid4()),
                "message": f"BDL spec missing required section: {section}"
            })
    
    # 记录规范版本和来源
    bdl_version = bdl_spec.get("version", "unknown")
    
    return {
        "bdl_spec": bdl_spec,
        "warnings": warnings,
        "current_phase": Phase.COMPONENT_PARSING.value,
        "stats": {
            "bdl_spec_version": bdl_version,
            "bdl_spec_source": bdl_source,
            "bdl_components_count": len(bdl_spec.get("components", {}))
        }
    }


def create_bdl_spec_template() -> Dict[str, Any]:
    """
    创建 BDL 规范模板
    
    供用户参考，创建自己的规范文件
    
    使用方式:
    1. 调用此函数获取模板
    2. 根据实际 BDL 设计系统填充内容
    3. 保存为 JSON 文件
    4. 使用 --bdl-spec 参数指定
    """
    return {
        "_comment": "BDL Specification Template - Please fill with your actual design system",
        "version": "1.0.0",
        "design_tokens": {
            "colors": {
                "_comment": "从 Figma/设计系统获取实际颜色值",
                "primary": {},
                "secondary": {},
                "neutral": {},
                "semantic": {}
            },
            "spacing": {
                "_comment": "间距系统",
            },
            "typography": {
                "_comment": "字体系统",
                "fontFamily": {},
                "fontSize": {},
                "fontWeight": {},
                "lineHeight": {}
            },
            "breakpoints": {
                "_comment": "响应式断点",
            },
            "shadows": {},
            "borderRadius": {}
        },
        "components": {
            "_comment": "组件规范 - 从 BDL 组件库文档获取",
            "Button": {
                "props": ["variant", "size", "disabled", "onClick", "children"],
                "variants": ["primary", "secondary"],
                "sizes": ["sm", "md", "lg"]
            }
            # ... 添加更多组件
        },
        "patterns": {
            "naming": {
                "components": "PascalCase",
                "props": "camelCase",
                "events": "onAction"
            },
            "file_structure": {}
        },
        "accessibility": {
            "required": [],
            "aria_patterns": {}
        }
    }
