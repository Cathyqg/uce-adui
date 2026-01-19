"""
页面迁移节点实现
负责将 AEM 页面 JSON 转换为新 CMS 页面 JSON

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
from typing import Any, Dict, List, Optional
from uuid import uuid4

from src.core.state import MigrationGraphState, Phase


# ============================================================================
# AEM JSON 解析节点
# ============================================================================

def parse_aem_json(state: MigrationGraphState) -> Dict[str, Any]:
    """
    解析 AEM 页面 JSON 节点
    
    AEM 导出的页面 JSON 通常包含:
    - jcr:primaryType
    - jcr:content 节点
    - 嵌套的组件节点
    - 属性值
    """
    page_queue = list(state.get("page_queue", []))
    aem_page_json_paths = state.get("aem_page_json_paths", [])
    pages = dict(state.get("pages", {}))
    
    # 如果队列为空但有页面路径，初始化队列
    if not page_queue and aem_page_json_paths:
        page_queue = list(range(len(aem_page_json_paths)))
    
    if not page_queue:
        return {
            "pages": pages,
            "page_queue": [],
            "current_phase": Phase.COMPLETED.value
        }
    
    # 处理下一个页面
    current_index = page_queue.pop(0)
    
    if current_index >= len(aem_page_json_paths):
        return {
            "pages": pages,
            "page_queue": page_queue
        }
    
    json_path = aem_page_json_paths[current_index]
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            aem_json = json.load(f)
    except Exception as e:
        return {
            "pages": pages,
            "page_queue": page_queue,
            "errors": state.get("errors", []) + [{
                "error_id": str(uuid4()),
                "severity": "recoverable",
                "error_type": "PageParseError",
                "message": f"Failed to parse {json_path}: {str(e)}"
            }]
        }
    
    # 解析 JCR 结构
    page_id = f"page_{current_index}"
    parsed_page = _parse_jcr_structure(aem_json, page_id)
    
    pages[page_id] = {
        "page_id": page_id,
        "source_path": json_path,
        "aem_json": aem_json,
        "parsed": parsed_page,
        "status": "parsed"
    }
    
    return {
        "pages": pages,
        "page_queue": page_queue,
        "current_page_id": page_id
    }


def _parse_jcr_structure(aem_json: Dict, page_id: str) -> Dict[str, Any]:
    """解析 JCR 节点结构"""
    
    # 提取页面元数据
    jcr_content = aem_json.get("jcr:content", aem_json)
    
    parsed = {
        "page_path": jcr_content.get("cq:path", ""),
        "page_title": jcr_content.get("jcr:title", jcr_content.get("pageTitle", "")),
        "template": jcr_content.get("cq:template", ""),
        "components": [],
        "metadata": {
            "language": jcr_content.get("jcr:language", "en"),
            "description": jcr_content.get("jcr:description", ""),
            "keywords": jcr_content.get("cq:tags", []),
            "lastModified": jcr_content.get("cq:lastModified", ""),
            "created": jcr_content.get("jcr:created", "")
        }
    }
    
    # 递归提取组件
    components = _extract_components(jcr_content)
    parsed["components"] = components
    
    return parsed


def _extract_components(node: Dict, parent_path: str = "") -> List[Dict]:
    """递归提取组件"""
    components = []
    
    for key, value in node.items():
        if key.startswith("jcr:") or key.startswith("cq:") or key.startswith("sling:"):
            continue
        
        if isinstance(value, dict):
            # 检查是否是组件节点
            resource_type = _get_resource_type(value)
            
            if resource_type:
                component = {
                    "instance_id": f"{parent_path}/{key}" if parent_path else key,
                    "node_name": key,
                    "resource_type": resource_type,
                    "props": _extract_props(value),
                    "children": _extract_components(value, f"{parent_path}/{key}" if parent_path else key)
                }
                components.append(component)
            else:
                # 继续深入查找
                nested = _extract_components(value, f"{parent_path}/{key}" if parent_path else key)
                components.extend(nested)
    
    return components


def _get_resource_type(node: Dict[str, Any]) -> str:
    """Return resourceType value from known or namespaced keys."""
    if not isinstance(node, dict):
        return ""

    for key in ("sling:resourceType", "resourceType"):
        value = node.get(key)
        if value:
            return value

    for key, value in node.items():
        if isinstance(key, str) and key.lower().endswith("resourcetype") and value:
            return value

    return ""


def _extract_props(node: Dict) -> Dict[str, Any]:
    """提取组件属性"""
    props = {}
    
    for key, value in node.items():
        # 跳过 JCR 系统属性
        if key.startswith("jcr:") or key.startswith("cq:") or key.startswith("sling:"):
            continue
        
        # 跳过子节点 (已在 children 中处理)
        if isinstance(value, dict):
            continue
        
        props[key] = value
    
    return props


# ============================================================================
# 组件映射节点
# ============================================================================

def map_page_components(state: MigrationGraphState) -> Dict[str, Any]:
    """
    映射页面组件节点
    
    将 AEM resourceType 映射到新 CMS 的 React 组件
    """
    current_page_id = state.get("current_page_id")
    pages = dict(state.get("pages", {}))
    component_registry = state.get("component_registry", {})
    warnings = list(state.get("warnings", []))
    
    if not current_page_id or current_page_id not in pages:
        return {"pages": pages}
    
    page_data = pages[current_page_id]
    parsed = page_data.get("parsed", {})
    components = parsed.get("components", [])
    
    # 映射组件
    mapped_components = []
    unmapped_types = set()
    
    for comp in components:
        resource_type = comp.get("resource_type", "")
        react_component = component_registry.get(resource_type)
        
        if react_component:
            mapped = {
                "instance_id": comp.get("instance_id"),
                "component_type": react_component,
                "original_resource_type": resource_type,
                "props": _transform_props(comp.get("props", {}), resource_type, component_registry),
                "children": _map_children(comp.get("children", []), component_registry)
            }
            mapped_components.append(mapped)
        else:
            unmapped_types.add(resource_type)
            # 添加占位符
            mapped = {
                "instance_id": comp.get("instance_id"),
                "component_type": "UnmappedComponent",
                "original_resource_type": resource_type,
                "props": comp.get("props", {}),
                "children": _map_children(comp.get("children", []), component_registry),
                "_unmapped": True
            }
            mapped_components.append(mapped)
    
    # 记录未映射的组件类型
    for resource_type in unmapped_types:
        warnings.append({
            "warning_id": str(uuid4()),
            "message": f"No mapping found for resource type: {resource_type}",
            "component_id": current_page_id
        })
    
    pages[current_page_id]["mapped_components"] = mapped_components
    pages[current_page_id]["status"] = "mapped"
    
    return {
        "pages": pages,
        "warnings": warnings
    }


def _map_children(children: List[Dict], registry: Dict[str, str]) -> List[Dict]:
    """递归映射子组件"""
    mapped = []
    
    for child in children:
        resource_type = child.get("resource_type", "")
        react_component = registry.get(resource_type, "UnmappedComponent")
        
        mapped.append({
            "instance_id": child.get("instance_id"),
            "component_type": react_component,
            "original_resource_type": resource_type,
            "props": child.get("props", {}),
            "children": _map_children(child.get("children", []), registry)
        })
    
    return mapped


def _transform_props(props: Dict, resource_type: str, registry: Dict) -> Dict:
    """转换属性名称和值"""
    # [CUSTOMIZE] 通用属性转换规则 - 根据 AEM 项目实际属性命名调整
    # 左边是 AEM 属性名，右边是目标 CMS 的属性名
    prop_transforms = {
        "jcr:title": "title",
        "text": "content",
        "fileReference": "imageSrc",
        "linkURL": "href",
        "openInNewWindow": "target",
        "altText": "alt"
    }
    
    transformed = {}
    
    for key, value in props.items():
        new_key = prop_transforms.get(key, key)
        
        # 特殊值转换
        if key == "openInNewWindow":
            transformed["target"] = "_blank" if value else "_self"
        elif key == "fileReference" and value:
            # 转换 DAM 路径
            transformed["imageSrc"] = _transform_dam_path(value)
        else:
            transformed[new_key] = value
    
    return transformed


def _transform_dam_path(dam_path: str) -> str:
    """转换 AEM DAM 路径到新系统路径"""
    # [CUSTOMIZE] [EXAMPLE] DAM 路径转换规则
    # 示例: /content/dam/mysite/images/hero.jpg -> /assets/images/hero.jpg
    # 需要根据实际的 AEM DAM 结构和目标资产存储路径调整
    if dam_path.startswith("/content/dam/"):
        return dam_path.replace("/content/dam/mysite/", "/assets/")  # [CUSTOMIZE] 替换 'mysite'
    return dam_path


# ============================================================================
# 结构转换节点
# ============================================================================

def transform_structure(state: MigrationGraphState) -> Dict[str, Any]:
    """
    结构转换节点
    
    将组件层级转换为新 CMS 的结构
    """
    current_page_id = state.get("current_page_id")
    pages = dict(state.get("pages", {}))
    
    if not current_page_id or current_page_id not in pages:
        return {"pages": pages}
    
    page_data = pages[current_page_id]
    parsed = page_data.get("parsed", {})
    mapped_components = page_data.get("mapped_components", [])
    
    # 构建新结构
    transformed = {
        "layout": _detect_layout(mapped_components),
        "sections": _organize_into_sections(mapped_components),
        "metadata": parsed.get("metadata", {})
    }
    
    pages[current_page_id]["transformed"] = transformed
    pages[current_page_id]["status"] = "transformed"
    
    return {"pages": pages}


def _detect_layout(components: List[Dict]) -> str:
    """检测页面布局类型"""
    # 简单启发式检测
    component_types = [c.get("component_type", "").lower() for c in components]
    
    if any("hero" in t or "banner" in t for t in component_types):
        return "landing"
    elif any("sidebar" in t for t in component_types):
        return "with-sidebar"
    elif any("form" in t for t in component_types):
        return "form"
    else:
        return "default"


def _organize_into_sections(components: List[Dict]) -> List[Dict]:
    """组织组件到逻辑区块"""
    sections = []
    current_section = {
        "section_id": str(uuid4()),
        "section_type": "main",
        "components": []
    }
    
    for comp in components:
        comp_type = comp.get("component_type", "").lower()
        
        # 某些组件类型标志新区块开始
        if any(marker in comp_type for marker in ["header", "hero", "footer", "nav"]):
            if current_section["components"]:
                sections.append(current_section)
            
            section_type = "header" if "header" in comp_type or "nav" in comp_type else \
                          "hero" if "hero" in comp_type else \
                          "footer" if "footer" in comp_type else "main"
            
            current_section = {
                "section_id": str(uuid4()),
                "section_type": section_type,
                "components": [comp]
            }
        else:
            current_section["components"].append(comp)
    
    if current_section["components"]:
        sections.append(current_section)
    
    return sections


# ============================================================================
# CMS JSON 生成节点
# ============================================================================

def generate_cms_json(state: MigrationGraphState) -> Dict[str, Any]:
    """
    生成 CMS JSON 节点
    
    生成最终的 CMS 页面 JSON 格式
    """
    current_page_id = state.get("current_page_id")
    pages = dict(state.get("pages", {}))
    
    if not current_page_id or current_page_id not in pages:
        return {"pages": pages}
    
    page_data = pages[current_page_id]
    parsed = page_data.get("parsed", {})
    transformed = page_data.get("transformed", {})
    mapped_components = page_data.get("mapped_components", [])
    
    # 生成 CMS JSON 格式
    # [CUSTOMIZE] Schema URL 需要替换为实际的 CMS 域名
    cms_json = {
        "$schema": "https://cms.example.com/schemas/page.json",  # [CUSTOMIZE] 替换域名
        "version": "1.0.0",
        "page": {
            "id": current_page_id,
            "slug": _generate_slug(parsed.get("page_path", "")),
            "title": parsed.get("page_title", ""),
            "template": _map_template(parsed.get("template", "")),
            "status": "draft",
            "createdAt": transformed.get("metadata", {}).get("created", ""),
            "updatedAt": transformed.get("metadata", {}).get("lastModified", "")
        },
        "seo": {
            "title": parsed.get("page_title", ""),
            "description": transformed.get("metadata", {}).get("description", ""),
            "keywords": transformed.get("metadata", {}).get("keywords", []),
            "ogImage": None,
            "robots": "index, follow"
        },
        "layout": transformed.get("layout", "default"),
        "content": {
            "sections": transformed.get("sections", [])
        },
        "components": _format_components_for_cms(mapped_components)
    }
    
    pages[current_page_id]["cms_json"] = cms_json
    pages[current_page_id]["status"] = "generated"
    
    # 更新统计
    stats = dict(state.get("stats", {}))
    stats["migrated_pages"] = stats.get("migrated_pages", 0) + 1
    
    return {
        "pages": pages,
        "stats": stats
    }


def _generate_slug(page_path: str) -> str:
    """从 AEM 路径生成 slug"""
    # [CUSTOMIZE] Slug 生成逻辑需要根据实际 AEM 路径结构调整
    # 示例: /content/mysite/en/products/detail -> /products/detail
    if "/content/" in page_path:
        parts = page_path.split("/")
        # [CUSTOMIZE] 语言代码列表根据实际支持的语言调整
        try:
            lang_idx = next(i for i, p in enumerate(parts) if p in ["en", "zh", "de", "fr", "es"])
            return "/" + "/".join(parts[lang_idx + 1:])
        except StopIteration:
            pass
    
    return page_path.replace("/content/mysite", "")  # [CUSTOMIZE] 替换 'mysite'


def _map_template(aem_template: str) -> str:
    """映射 AEM 模板到新系统模板"""
    # [CUSTOMIZE] [EXAMPLE] 模板映射 - 需要根据实际 AEM 模板和目标 CMS 模板调整
    # 左边是 AEM 模板名称，右边是目标 CMS 模板名称
    template_map = {
        "page-content": "default",
        "page-landing": "landing",
        "page-article": "article",
        "page-product": "product"
    }
    
    template_name = aem_template.split("/")[-1] if "/" in aem_template else aem_template
    return template_map.get(template_name, "default")


def _format_components_for_cms(components: List[Dict]) -> List[Dict]:
    """格式化组件为 CMS 格式"""
    formatted = []
    
    for comp in components:
        formatted_comp = {
            "id": str(uuid4()),
            "type": comp.get("component_type"),
            "props": comp.get("props", {}),
            "metadata": {
                "originalId": comp.get("instance_id"),
                "originalType": comp.get("original_resource_type")
            }
        }
        
        if comp.get("children"):
            formatted_comp["children"] = _format_components_for_cms(comp["children"])
        
        if comp.get("_unmapped"):
            formatted_comp["_warning"] = "Component type not mapped"
        
        formatted.append(formatted_comp)
    
    return formatted


# ============================================================================
# 页面验证节点
# ============================================================================

def validate_page(state: MigrationGraphState) -> Dict[str, Any]:
    """
    验证页面节点
    
    确保生成的页面 JSON 有效
    """
    current_page_id = state.get("current_page_id")
    pages = dict(state.get("pages", {}))
    errors = list(state.get("errors", []))
    warnings = list(state.get("warnings", []))
    
    if not current_page_id or current_page_id not in pages:
        return {
            "pages": pages,
            "errors": errors,
            "warnings": warnings
        }
    
    page_data = pages[current_page_id]
    cms_json = page_data.get("cms_json", {})
    
    # 验证必需字段
    page_info = cms_json.get("page", {})
    
    if not page_info.get("title"):
        warnings.append({
            "warning_id": str(uuid4()),
            "message": f"Page {current_page_id} has no title",
            "component_id": current_page_id
        })
    
    if not page_info.get("slug"):
        errors.append({
            "error_id": str(uuid4()),
            "severity": "critical",
            "error_type": "PageValidationError",
            "message": f"Page {current_page_id} has no slug",
            "component_id": current_page_id
        })
    
    # 检查未映射组件
    components = cms_json.get("components", [])
    unmapped_count = sum(1 for c in components if c.get("_warning"))
    
    if unmapped_count > 0:
        warnings.append({
            "warning_id": str(uuid4()),
            "message": f"Page {current_page_id} has {unmapped_count} unmapped components",
            "component_id": current_page_id
        })
    
    pages[current_page_id]["status"] = "validated"
    pages[current_page_id]["validation"] = {
        "errors": [e for e in errors if e.get("component_id") == current_page_id],
        "warnings": [w for w in warnings if w.get("component_id") == current_page_id]
    }
    
    return {
        "pages": pages,
        "errors": errors,
        "warnings": warnings
    }
