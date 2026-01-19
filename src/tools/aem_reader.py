"""
AEM 项目读取工具
从实际的 AEM 项目结构中读取组件定义

LangGraph 1.0+ Tools Best Practices:
1. 使用 @tool 装饰器
2. 详细的 docstring
3. 结构化返回值
4. 完整的错误处理

================================================================================
⚠️ 需要定制的部分标记说明:
================================================================================
- [CUSTOMIZE]    = 需要根据实际 AEM 项目定制
- [PLACEHOLDER]  = 占位符代码，需要完整实现
- [EXAMPLE]      = 示例代码，需要根据实际情况替换

实际使用场景:
1. 从本地 AEM 项目仓库读取组件
2. 从远程 Git 仓库克隆并读取
3. 从 AEM 实例通过 API 读取
================================================================================
"""

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from xml.etree import ElementTree as ET

from langchain_core.tools import tool


@tool
def read_aem_component_from_repo(
    repo_path: str,
    component_name: str,
    apps_path: str = "ui.apps/src/main/content/jcr_root/apps"
) -> Dict[str, Any]:
    """
    从 AEM 项目仓库读取组件定义
    
    实际使用场景:
    1. 从本地克隆的 AEM 项目读取组件
    2. 支持标准的 AEM Maven 项目结构
    
    Args:
        repo_path: AEM 项目仓库根路径
        component_name: 组件名称（如 "hero-banner"）
        apps_path: apps 目录的相对路径
    
    Returns:
        {
            "success": bool,
            "component_id": str,
            "htl_content": str,              # HTL 模板内容
            "dialog_xml": str,               # Dialog 配置 XML
            "js_content": str,               # JavaScript 内容
            "css_content": str,              # CSS/SCSS 内容
            "metadata": {...},               # 组件元数据
            "source_path": str,              # 源路径
            "error": str                     # 错误信息
        }
    """
    # [CUSTOMIZE] AEM 项目结构
    # 标准 AEM Maven 项目结构:
    # repo/
    # ├── ui.apps/
    # │   └── src/main/content/jcr_root/apps/
    # │       └── mysite/components/
    # │           └── hero-banner/
    # │               ├── hero-banner.html
    # │               ├── _cq_dialog/.content.xml
    # │               └── clientlibs/
    
    try:
        # [CUSTOMIZE] 项目名称需要替换
        project_name = "mysite"  # [CUSTOMIZE] 替换为实际项目名
        
        # 构建组件路径
        component_base = Path(repo_path) / apps_path / project_name / "components" / component_name
        
        if not component_base.exists():
            # 尝试其他可能的路径
            alternative_paths = [
                Path(repo_path) / "apps" / project_name / "components" / component_name,
                Path(repo_path) / "src" / "components" / component_name,
            ]
            component_base = next((p for p in alternative_paths if p.exists()), None)
        
        if not component_base or not component_base.exists():
            return {
                "success": False,
                "error": f"Component not found: {component_name} in {repo_path}",
                "component_id": component_name
            }
        
        result = {
            "success": True,
            "component_id": component_name,
            "source_path": str(component_base),
            "htl_content": "",
            "dialog_xml": "",
            "js_content": "",
            "css_content": "",
            "metadata": {},
            "error": ""
        }
        
        # 读取 HTL 模板
        htl_files = list(component_base.glob(f"{component_name}.html")) + \
                    list(component_base.glob("*.html"))
        if htl_files:
            with open(htl_files[0], 'r', encoding='utf-8') as f:
                result["htl_content"] = f.read()
        
        # 读取 Dialog 配置
        dialog_path = component_base / "_cq_dialog" / ".content.xml"
        if dialog_path.exists():
            with open(dialog_path, 'r', encoding='utf-8') as f:
                result["dialog_xml"] = f.read()
        
        # 读取 ClientLib JavaScript
        js_dir = component_base / "clientlibs" / "js"
        if js_dir.exists():
            js_files = list(js_dir.glob("*.js"))
            js_contents = []
            for js_file in js_files:
                with open(js_file, 'r', encoding='utf-8') as f:
                    js_contents.append(f.read())
            result["js_content"] = "\n\n".join(js_contents)
        
        # 读取 ClientLib CSS/SCSS
        for css_dir_name in ["css", "scss", "less"]:
            css_dir = component_base / "clientlibs" / css_dir_name
            if css_dir.exists():
                css_files = list(css_dir.glob(f"*.{css_dir_name}"))
                css_contents = []
                for css_file in css_files:
                    with open(css_file, 'r', encoding='utf-8') as f:
                        css_contents.append(f.read())
                result["css_content"] = "\n\n".join(css_contents)
                break
        
        # 读取组件元数据
        content_xml = component_base / ".content.xml"
        if content_xml.exists():
            try:
                tree = ET.parse(content_xml)
                root = tree.getroot()
                result["metadata"] = {
                    "title": root.get('{http://www.jcp.org/jcr/1.0}title', ''),
                    "description": root.get('{http://www.jcp.org/jcr/1.0}description', ''),
                    "componentGroup": root.get('componentGroup', ''),
                    "isContainer": root.get('isContainer', 'false')
                }
            except:
                pass
        
        return result
    
    except Exception as e:
        return {
            "success": False,
            "component_id": component_name,
            "error": f"Failed to read AEM component: {str(e)}"
        }


@tool
def list_aem_components_in_repo(
    repo_path: str,
    apps_path: str = "ui.apps/src/main/content/jcr_root/apps",
    project_name: str = "mysite"
) -> Dict[str, Any]:
    """
    列出 AEM 仓库中的所有组件
    
    实际使用场景:
    1. 扫描整个 AEM 项目，发现所有可迁移的组件
    2. 生成组件清单
    
    Args:
        repo_path: AEM 项目仓库根路径
        apps_path: apps 目录的相对路径
        project_name: AEM 项目名称
    
    Returns:
        {
            "success": bool,
            "components": [
                {
                    "name": str,
                    "path": str,
                    "has_htl": bool,
                    "has_dialog": bool,
                    "has_clientlib": bool
                }
            ],
            "count": int,
            "error": str
        }
    """
    # [CUSTOMIZE] AEM 项目结构
    try:
        components_dir = Path(repo_path) / apps_path / project_name / "components"
        
        if not components_dir.exists():
            return {
                "success": False,
                "components": [],
                "count": 0,
                "error": f"Components directory not found: {components_dir}"
            }
        
        components = []
        
        for item in components_dir.iterdir():
            if not item.is_dir():
                continue
            
            # 检查是否是有效的 AEM 组件
            has_content_xml = (item / ".content.xml").exists()
            if not has_content_xml:
                continue
            
            comp_info = {
                "name": item.name,
                "path": str(item),
                "has_htl": any(item.glob("*.html")),
                "has_dialog": (item / "_cq_dialog" / ".content.xml").exists(),
                "has_clientlib": (item / "clientlibs").exists()
            }
            
            components.append(comp_info)
        
        return {
            "success": True,
            "components": components,
            "count": len(components),
            "error": ""
        }
    
    except Exception as e:
        return {
            "success": False,
            "components": [],
            "count": 0,
            "error": f"Failed to list components: {str(e)}"
        }