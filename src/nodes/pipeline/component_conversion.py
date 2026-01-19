"""
组件转换节点实现
负责 AEM 组件到 React 组件的转换流程

================================================================================
⚠️ 需要定制的部分标记说明:
================================================================================
- [CUSTOMIZE]    = 需要根据实际项目定制的逻辑
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
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4
from xml.etree import ElementTree as ET

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src.llm import get_llm
from src.core.state import (
    AEMComponent,
    AEMDialog,
    AnalyzedComponent,
    BDLMapping,
    ClientLib,
    ComponentComplexity,
    ComponentDependency,
    ComponentStatus,
    HTLTemplate,
    MigrationGraphState,
    Phase,
    PropMapping,
    ReactComponent,
    StyleMapping,
)


# ============================================================================
# Pydantic 输出模型 - LangGraph 1.0 结构化输出
# ============================================================================

class DataSlyUseItem(BaseModel):
    """HTL data-sly-use 指令"""
    variable: str = Field(description="Variable name")
    class_name: str = Field(alias="class", default="", description="Java/JS class")


class DataSlyListItem(BaseModel):
    """HTL data-sly-list 指令"""
    variable: str = Field(description="List variable")
    item_variable: str = Field(default="item", description="Item variable name")
    source: str = Field(description="Data source expression")


class DataSlyTestItem(BaseModel):
    """HTL data-sly-test 指令"""
    variable: str = Field(default="", description="Variable to store result")
    condition: str = Field(description="Test condition")


class DataSlyResourceItem(BaseModel):
    """HTL data-sly-resource 指令"""
    path: str = Field(description="Resource path")
    resource_type: str = Field(default="", description="Resource type")


class HTLParseOutput(BaseModel):
    """HTL 解析结构化输出"""
    data_sly_use: List[DataSlyUseItem] = Field(default_factory=list)
    data_sly_list: List[DataSlyListItem] = Field(default_factory=list)
    data_sly_test: List[DataSlyTestItem] = Field(default_factory=list)
    data_sly_resource: List[DataSlyResourceItem] = Field(default_factory=list)
    text_expressions: List[str] = Field(default_factory=list)
    html_structure: str = Field(default="", description="Simplified DOM structure")
    sling_model_class: Optional[str] = Field(None, description="Sling model class if identifiable")


class ComponentComplexityOutput(BaseModel):
    """组件复杂度"""
    lines_of_code: int = Field(default=0)
    cyclomatic_complexity: int = Field(default=1)
    dependency_count: int = Field(default=0)
    interaction_points: int = Field(default=0)


class ComponentFeaturesOutput(BaseModel):
    """组件特性"""
    has_form: bool = Field(default=False)
    has_animation: bool = Field(default=False)
    has_responsive: bool = Field(default=False)
    has_accessibility: bool = Field(default=False)


class DependencyOutput(BaseModel):
    """组件依赖"""
    component_id: str = Field(description="Dependent component ID")
    type: str = Field(description="Dependency type: component|model|service")
    required: bool = Field(default=True)


class AnalyzeComponentOutput(BaseModel):
    """组件分析结构化输出"""
    component_type: str = Field(description="ui|container|layout|utility")
    is_dynamic: bool = Field(default=False)
    complexity: ComponentComplexityOutput = Field(default_factory=ComponentComplexityOutput)
    features: ComponentFeaturesOutput = Field(default_factory=ComponentFeaturesOutput)
    bdl_mapping_feasibility: float = Field(ge=0, le=1, default=0.5, description="BDL mapping score 0-1")
    mapping_notes: List[str] = Field(default_factory=list)
    dependencies: List[DependencyOutput] = Field(default_factory=list)


class PropMappingOutput(BaseModel):
    """属性映射"""
    aem_prop: str = Field(description="AEM property name")
    bdl_prop: str = Field(description="BDL property name")
    transform: Optional[str] = Field(None, description="Transform function if needed")


class StyleMappingOutput(BaseModel):
    """样式映射"""
    aem_class: str = Field(description="AEM CSS class")
    bdl_token: str = Field(description="BDL design token")
    css_property: str = Field(default="", description="CSS property")


class BDLMappingOutput(BaseModel):
    """BDL 映射结构化输出"""
    bdl_component_name: Optional[str] = Field(None, description="Closest BDL component or null")
    prop_mappings: List[PropMappingOutput] = Field(default_factory=list)
    style_mappings: List[StyleMappingOutput] = Field(default_factory=list)
    event_mappings: Dict[str, str] = Field(default_factory=dict)
    missing_in_bdl: List[str] = Field(default_factory=list)
    requires_custom: List[str] = Field(default_factory=list)
    confidence_score: float = Field(ge=0, le=1, default=0.5)


class HookOutput(BaseModel):
    """React Hook 定义"""
    name: str = Field(description="Hook name")
    code: str = Field(description="Hook implementation")


class EventHandlerOutput(BaseModel):
    """事件处理器定义"""
    name: str = Field(description="Handler name")
    code: str = Field(description="Handler implementation")


class TransformLogicOutput(BaseModel):
    """逻辑转换结构化输出"""
    typescript_interface: str = Field(default="", description="TypeScript interface code")
    jsx_template: str = Field(default="", description="JSX template code")
    hooks: List[HookOutput] = Field(default_factory=list)
    event_handlers: List[EventHandlerOutput] = Field(default_factory=list)
    css_module: str = Field(default="", description="CSS module code")


class GenerateReactOutput(BaseModel):
    """React 代码生成结构化输出"""
    component_code: str = Field(description="Full .tsx file content")
    styles_code: str = Field(default="", description="Full .module.css content")
    index_code: str = Field(default="", description="Index file exports")
    types_code: Optional[str] = Field(None, description="Separate types file if complex")
    imports: List[str] = Field(default_factory=list)
    peer_dependencies: Dict[str, str] = Field(default_factory=dict)


# ============================================================================
# 重试装饰器 - LangGraph 1.0 Best Practice
# ============================================================================

def with_retry(max_attempts: int = 3, min_wait: float = 1, max_wait: float = 10):
    """LLM 调用重试装饰器"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            import asyncio
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
# 源码摄入节点
# ============================================================================

def ingest_source(state: MigrationGraphState) -> Dict[str, Any]:
    """
    源码摄入节点
    
    扫描 AEM 组件目录，识别和读取所有组件文件
    
    AEM 组件典型目录结构:
    /apps/mysite/components/
    ├── hero-banner/
    │   ├── hero-banner.html          # HTL 模板
    │   ├── _cq_dialog/.content.xml   # 对话框定义
    │   ├── .content.xml              # 组件元数据
    │   └── clientlibs/               # 客户端库
    │       ├── js/
    │       └── css/
    """
    source_path = state.get("source_path", "")
    config = state.get("config", {})
    
    if not source_path or not os.path.exists(source_path):
        return {
            "errors": state.get("errors", []) + [{
                "error_id": str(uuid4()),
                "severity": "fatal",
                "error_type": "SourceNotFound",
                "message": f"Source path not found: {source_path}"
            }],
            "current_phase": Phase.FAILED.value
        }
    
    # 扫描组件目录
    components = {}
    component_queue = []
    
    base_path = Path(source_path)
    component_filter = config.get("component_filter", [])
    
    for item in base_path.iterdir():
        if not item.is_dir():
            continue
            
        # 检查是否是有效的 AEM 组件 (包含 .content.xml)
        content_xml = item / ".content.xml"
        if not content_xml.exists():
            continue
        
        # 应用过滤器
        component_id = item.name
        if component_filter and component_id not in component_filter:
            continue
        
        # 读取组件元数据
        try:
            metadata = _parse_component_metadata(content_xml)
        except Exception as e:
            metadata = {"title": component_id, "componentGroup": "unknown"}
        
        # 读取 HTL 模板
        htl_content = ""
        htl_path = ""
        for ext in [".html", ".htm"]:
            htl_file = item / f"{component_id}{ext}"
            if htl_file.exists():
                htl_content = htl_file.read_text(encoding="utf-8")
                htl_path = str(htl_file)
                break
        
        # 读取对话框
        dialog_path = item / "_cq_dialog" / ".content.xml"
        dialog_content = ""
        if dialog_path.exists():
            dialog_content = dialog_path.read_text(encoding="utf-8")
        
        # 读取客户端库
        clientlib_info = _scan_clientlib(item / "clientlibs")
        
        # 构建组件数据
        # [CUSTOMIZE] resource_type 格式需要根据实际 AEM 项目结构调整
        aem_component = {
            "component_id": component_id,
            "resource_type": f"mysite/components/{component_id}",  # [CUSTOMIZE] 替换 'mysite' 为实际项目名
            "component_group": metadata.get("componentGroup", "unknown"),
            "title": metadata.get("title", component_id),
            "description": metadata.get("description", ""),
            "source_path": str(item),
            "htl_path": htl_path,
            "dialog_path": str(dialog_path) if dialog_path.exists() else "",
            "htl_template": {
                "raw_content": htl_content
            },
            "dialog": {
                "raw_xml": dialog_content
            },
            "clientlib": clientlib_info,
            "is_container": metadata.get("isContainer", False),
            "allowed_children": metadata.get("allowedChildren", [])
        }
        
        components[component_id] = {
            "component_id": component_id,
            "status": ComponentStatus.PENDING.value,
            "aem_component": aem_component,
            "errors": [],
            "warnings": [],
            "retry_count": 0
        }
        
        component_queue.append(component_id)
    
    # 更新统计
    stats = dict(state.get("stats", {}))
    stats["total_components"] = len(components)
    
    return {
        "components": components,
        "component_queue": component_queue,
        "stats": stats,
        "current_phase": Phase.COMPONENT_PARSING.value
    }


def _parse_component_metadata(content_xml: Path) -> Dict[str, Any]:
    """解析组件 .content.xml 获取元数据"""
    tree = ET.parse(content_xml)
    root = tree.getroot()
    
    # 处理 JCR 命名空间
    namespaces = {
        'jcr': 'http://www.jcp.org/jcr/1.0',
        'cq': 'http://www.day.com/jcr/cq/1.0'
    }
    
    metadata = {
        "title": root.get('{http://www.jcp.org/jcr/1.0}title', ''),
        "description": root.get('{http://www.jcp.org/jcr/1.0}description', ''),
        "componentGroup": root.get('componentGroup', 'General'),
        "isContainer": root.get('isContainer', 'false').lower() == 'true'
    }
    
    return metadata


def _scan_clientlib(clientlib_path: Path) -> Dict[str, Any]:
    """扫描客户端库"""
    if not clientlib_path.exists():
        return {}
    
    clientlib = {
        "js_files": [],
        "css_files": [],
        "js_content": "",
        "css_content": ""
    }
    
    # 扫描 JS
    js_dir = clientlib_path / "js"
    if js_dir.exists():
        for js_file in js_dir.glob("*.js"):
            clientlib["js_files"].append(str(js_file))
            clientlib["js_content"] += js_file.read_text(encoding="utf-8") + "\n"
    
    # 扫描 CSS/SCSS
    for css_dir_name in ["css", "scss"]:
        css_dir = clientlib_path / css_dir_name
        if css_dir.exists():
            for css_file in css_dir.glob("*"):
                if css_file.suffix in [".css", ".scss", ".less"]:
                    clientlib["css_files"].append(str(css_file))
                    clientlib["css_content"] += css_file.read_text(encoding="utf-8") + "\n"
    
    return clientlib


# ============================================================================
# HTL 解析节点
# ============================================================================

# [PROMPT] HTL 解析 Prompt - 可根据实际 AEM 版本和 HTL 用法调整
HTL_PARSE_PROMPT = """You are an expert in AEM HTL (Sightly) template parsing.
Analyze the following HTL template and extract structured information.

Extract:
1. **data-sly-use** directives - model bindings
2. **data-sly-list** loops - iteration variables and source
3. **data-sly-test** conditions - conditional logic
4. **data-sly-resource** - nested component includes
5. **${{...}}** expressions - text and attribute expressions
6. **HTML structure** - DOM hierarchy

Return JSON format:
{
  "data_sly_use": [{"variable": "...", "class": "..."}],
  "data_sly_list": [{"variable": "...", "item_variable": "...", "source": "..."}],
  "data_sly_test": [{"variable": "...", "condition": "..."}],
  "data_sly_resource": [{"path": "...", "resource_type": "..."}],
  "text_expressions": ["..."],
  "html_structure": "simplified DOM structure",
  "sling_model_class": "if identifiable"
}
"""


async def parse_aem(state: MigrationGraphState) -> Dict[str, Any]:
    """
    解析 AEM 组件节点
    
    深度解析:
    - HTL 模板语法
    - Dialog 字段定义
    - ClientLib JS/CSS
    """
    components = state.get("components", {})
    component_queue = list(state.get("component_queue", []))
    updated_components = dict(components)
    stats = dict(state.get("stats", {}))
    
    if not component_queue:
        return {
            "components": updated_components,
            "component_queue": [],
            "current_phase": Phase.COMPONENT_ANALYSIS.value
        }
    
    # 处理队列中的下一个组件
    current_id = component_queue.pop(0)
    comp_data = updated_components.get(current_id, {})
    
    if not comp_data:
        return {
            "components": updated_components,
            "component_queue": component_queue
        }
    
    updated_components[current_id]["status"] = ComponentStatus.PARSING.value
    
    # 使用工厂方法获取 LLM - LangGraph 1.0+ Best Practice
    llm = get_llm(task="parsing", temperature=0)
    structured_llm = llm.with_structured_output(HTLParseOutput)
    
    try:
        aem_component = comp_data.get("aem_component", {})
        htl_template = aem_component.get("htl_template", {})
        htl_content = htl_template.get("raw_content", "")
        
        if htl_content:
            # 使用 LLM 解析 HTL - 结构化输出
            messages = [
                SystemMessage(content=HTL_PARSE_PROMPT),
                HumanMessage(content=f"Parse this HTL template:\n\n```html\n{htl_content}\n```")
            ]
            
            try:
                # 优先使用结构化输出
                result: HTLParseOutput = await structured_llm.ainvoke(messages)
                parsed_htl = result.model_dump(by_alias=True)
            except Exception:
                # Fallback: 普通解析
                response = await llm.ainvoke(messages)
                try:
                    parsed_htl = json.loads(response.content)
                except json.JSONDecodeError:
                    json_match = re.search(r'\{[\s\S]*\}', response.content)
                    if json_match:
                        parsed_htl = json.loads(json_match.group())
                    else:
                        parsed_htl = {}
            
            # 更新 HTL 模板解析结果
            updated_components[current_id]["aem_component"]["htl_template"].update(parsed_htl)
        
        # 解析 Dialog
        dialog = aem_component.get("dialog", {})
        dialog_xml = dialog.get("raw_xml", "")
        
        if dialog_xml:
            parsed_dialog = _parse_dialog_xml(dialog_xml)
            updated_components[current_id]["aem_component"]["dialog"].update(parsed_dialog)
        
        stats["parsed_components"] = stats.get("parsed_components", 0) + 1
        
    except Exception as e:
        updated_components[current_id]["errors"].append({
            "error_id": str(uuid4()),
            "severity": "recoverable",
            "error_type": "ParseError",
            "message": str(e)
        })
    
    return {
        "components": updated_components,
        "component_queue": component_queue,
        "current_component_id": current_id,
        "stats": stats
    }


def _parse_dialog_xml(xml_content: str) -> Dict[str, Any]:
    """解析 AEM 对话框 XML"""
    try:
        root = ET.fromstring(xml_content)
    except ET.ParseError:
        return {"fields": [], "tabs": []}
    
    fields = []
    tabs = []
    
    # 递归查找所有字段
    def extract_fields(element, prefix=""):
        for child in element:
            # 检查是否是表单字段
            field_type = _get_xml_attr(child, "sling:resourceType")
            if "granite/ui/components/coral/foundation/form" in field_type:
                field = {
                    "name": _get_xml_attr(child, "name"),
                    "type": field_type.split("/")[-1],
                    "label": _get_xml_attr(child, "fieldLabel"),
                    "description": _get_xml_attr(child, "fieldDescription"),
                    "required": _get_xml_attr(child, "required") == "true"
                }
                fields.append(field)
            
            # 检查是否是 Tab
            if "tabs" in field_type.lower():
                tab = {
                    "title": _get_xml_attr(child, "jcr:title"),
                    "fields": []
                }
                tabs.append(tab)
            
            extract_fields(child, f"{prefix}/{child.tag}")
    
    extract_fields(root)
    
    return {
        "fields": fields,
        "tabs": tabs
    }


def _get_xml_attr(element: ET.Element, name: str) -> str:
    if name in element.attrib:
        return element.attrib.get(name, "")
    local = name.split(":", 1)[-1]
    for key, value in element.attrib.items():
        if key.endswith("}" + local):
            return value
    return ""


# ============================================================================
# 组件分析节点
# ============================================================================

# [PROMPT] 组件分析 Prompt - 可根据组件复杂度评估标准调整
ANALYZE_PROMPT = """You are an expert in AEM component architecture.
Analyze this component and provide insights about its structure, complexity, and migration considerations.

Analyze:
1. **Component Type**: UI component, container, layout, or utility
2. **Complexity**: lines of code, logic complexity, dependencies
3. **Features**: forms, animations, responsive behavior, accessibility
4. **BDL Mapping Feasibility**: how well it maps to standard BDL components (0-1 score)

Return JSON:
{
  "component_type": "ui|container|layout|utility",
  "is_dynamic": boolean,
  "complexity": {
    "lines_of_code": number,
    "cyclomatic_complexity": number,
    "dependency_count": number,
    "interaction_points": number
  },
  "features": {
    "has_form": boolean,
    "has_animation": boolean,
    "has_responsive": boolean,
    "has_accessibility": boolean
  },
  "bdl_mapping_feasibility": 0.0-1.0,
  "mapping_notes": ["..."],
  "dependencies": [{"component_id": "...", "type": "component|model|service", "required": true}]
}
"""


async def analyze_component(state: MigrationGraphState) -> Dict[str, Any]:
    """
    组件分析节点
    
    分析:
    - 组件类型和特性
    - 复杂度评估
    - 依赖关系
    - BDL 映射可行性
    """
    components = state.get("components", {})
    updated_components = dict(components)
    
    # 使用工厂方法获取 LLM - LangGraph 1.0+ Best Practice
    llm = get_llm(task="analysis", temperature=0)
    structured_llm = llm.with_structured_output(AnalyzeComponentOutput)
    
    for comp_id, comp_data in components.items():
        if comp_data.get("status") != ComponentStatus.PARSING.value:
            continue
        
        updated_components[comp_id]["status"] = ComponentStatus.ANALYZING.value
        
        aem_component = comp_data.get("aem_component", {})
        htl = aem_component.get("htl_template", {})
        dialog = aem_component.get("dialog", {})
        clientlib = aem_component.get("clientlib", {})
        
        messages = [
            SystemMessage(content=ANALYZE_PROMPT),
            HumanMessage(content=f"""
Analyze this AEM component:

**Component ID**: {comp_id}
**Title**: {aem_component.get('title', '')}

**HTL Template**:
```html
{htl.get('raw_content', 'N/A')}
```

**Dialog Fields**:
{json.dumps(dialog.get('fields', []), indent=2)}

**JavaScript**:
```javascript
{clientlib.get('js_content', 'N/A')[:2000]}
```

**CSS**:
```css
{clientlib.get('css_content', 'N/A')[:2000]}
```
""")
        ]
        
        try:
            # 优先使用结构化输出
            result: AnalyzeComponentOutput = await structured_llm.ainvoke(messages)
            analysis = result.model_dump()
        except Exception:
            # Fallback: 普通解析
            try:
                response = await llm.ainvoke(messages)
                analysis = json.loads(response.content)
            except Exception:
                analysis = {
                    "component_type": "ui",
                    "is_dynamic": False,
                    "complexity": {},
                    "features": {},
                    "bdl_mapping_feasibility": 0.5,
                    "mapping_notes": [],
                    "dependencies": []
                }
        
        updated_components[comp_id]["analyzed"] = analysis
    
    return {
        "components": updated_components,
        "current_phase": Phase.BDL_MAPPING.value
    }


# ============================================================================
# BDL 映射节点
# ============================================================================

# [PROMPT] BDL 映射 Prompt
# [CUSTOMIZE] BDL Component Reference 需要替换为实际的 BDL 组件列表
BDL_MAPPING_PROMPT = """You are an expert in BDL (Brand Design Language) component framework.
Your task is to create a mapping from AEM component to BDL React component.

Given the AEM component analysis, determine:
1. Which BDL component(s) best match this component
2. How to map AEM dialog fields to React props
3. How to map AEM styles to BDL design tokens
4. What custom implementation is needed

BDL Component Reference:
- Button, Link, Icon
- Typography (Heading, Text, Label)
- Form elements (Input, Select, Checkbox, Radio, TextArea)
- Card, Hero, Banner
- Navigation (Navbar, Menu, Breadcrumb)
- Layout (Grid, Container, Stack, Flex)
- Modal, Drawer, Tooltip, Popover
- Table, List
- Carousel, Accordion, Tabs
(NOTE: [CUSTOMIZE] Replace this list with actual BDL components from your design system)

Return JSON:
{
  "bdl_component_name": "closest BDL component or null if custom",
  "prop_mappings": [
    {"aem_prop": "...", "bdl_prop": "...", "transform": "optional transform function"}
  ],
  "style_mappings": [
    {"aem_class": "...", "bdl_token": "...", "css_property": "..."}
  ],
  "event_mappings": {"aemEvent": "reactEvent"},
  "missing_in_bdl": ["features not in BDL"],
  "requires_custom": ["parts needing custom implementation"],
  "confidence_score": 0.0-1.0
}
"""


async def map_to_bdl(state: MigrationGraphState) -> Dict[str, Any]:
    """
    BDL 映射节点
    
    建立 AEM 组件到 BDL 的映射关系
    """
    components = state.get("components", {})
    bdl_spec = state.get("bdl_spec", {})
    updated_components = dict(components)
    
    # 使用工厂方法获取 LLM - LangGraph 1.0+ Best Practice
    llm = get_llm(task="analysis", temperature=0)
    structured_llm = llm.with_structured_output(BDLMappingOutput)
    
    for comp_id, comp_data in components.items():
        if comp_data.get("status") != ComponentStatus.ANALYZING.value:
            continue
        
        updated_components[comp_id]["status"] = ComponentStatus.MAPPING.value
        
        aem_component = comp_data.get("aem_component", {})
        analyzed = comp_data.get("analyzed", {})
        
        bdl_context = json.dumps(bdl_spec, indent=2)[:3000] if bdl_spec else "Standard BDL components"
        
        messages = [
            SystemMessage(content=BDL_MAPPING_PROMPT),
            HumanMessage(content=f"""
Map this AEM component to BDL:

**Component**: {comp_id}
**Type**: {analyzed.get('component_type', 'unknown')}
**Features**: {json.dumps(analyzed.get('features', {}))}

**Dialog Fields**:
{json.dumps(aem_component.get('dialog', {}).get('fields', []), indent=2)}

**BDL Specification**:
{bdl_context}

Create the optimal mapping.
""")
        ]
        
        try:
            # 优先使用结构化输出
            result: BDLMappingOutput = await structured_llm.ainvoke(messages)
            mapping = result.model_dump()
        except Exception:
            # Fallback: 普通解析
            try:
                response = await llm.ainvoke(messages)
                mapping = json.loads(response.content)
            except Exception:
                mapping = {
                    "bdl_component_name": None,
                    "prop_mappings": [],
                    "style_mappings": [],
                    "event_mappings": {},
                    "missing_in_bdl": [],
                    "requires_custom": [comp_id],
                    "confidence_score": 0.3
                }
        
        updated_components[comp_id]["bdl_mapping"] = mapping
    
    return {
        "components": updated_components
    }


# ============================================================================
# 逻辑转换节点
# ============================================================================

# [PROMPT] 逻辑转换 Prompt - 可根据目标 React 版本和代码规范调整
TRANSFORM_LOGIC_PROMPT = """You are an expert in migrating AEM HTL templates to React JSX.

Transform rules:
1. HTL `data-sly-use` → React imports/hooks
2. HTL `data-sly-list` → JavaScript map()
3. HTL `data-sly-test` → JSX conditional rendering
4. HTL `data-sly-resource` → React component inclusion
5. HTL `${{...}}` expressions → JSX expressions
6. jQuery/vanilla JS → React hooks and event handlers

Create TypeScript interfaces for props based on dialog fields.
Convert CSS to CSS Modules format.

Return JSON:
{
  "typescript_interface": "interface code",
  "jsx_template": "JSX code",
  "hooks": [{"name": "...", "code": "..."}],
  "event_handlers": [{"name": "...", "code": "..."}],
  "css_module": "CSS module code"
}
"""


async def transform_logic(state: MigrationGraphState) -> Dict[str, Any]:
    """
    逻辑转换节点
    
    将 AEM 业务逻辑转换为 React 代码
    """
    components = state.get("components", {})
    updated_components = dict(components)
    
    # 使用工厂方法获取 LLM - LangGraph 1.0+ Best Practice
    llm = get_llm(task="generation", temperature=0)
    
    for comp_id, comp_data in components.items():
        if comp_data.get("status") != ComponentStatus.MAPPING.value:
            continue
        
        updated_components[comp_id]["status"] = ComponentStatus.TRANSFORMING.value
        
        aem_component = comp_data.get("aem_component", {})
        bdl_mapping = comp_data.get("bdl_mapping", {})
        
        htl = aem_component.get("htl_template", {})
        clientlib = aem_component.get("clientlib", {})
        dialog = aem_component.get("dialog", {})
        
        messages = [
            SystemMessage(content=TRANSFORM_LOGIC_PROMPT),
            HumanMessage(content=f"""
Transform this AEM component to React:

**Component**: {comp_id}
**BDL Component**: {bdl_mapping.get('bdl_component_name', 'custom')}

**HTL Template**:
```html
{htl.get('raw_content', '')}
```

**HTL Analysis**:
- data-sly-use: {htl.get('data_sly_use', [])}
- data-sly-list: {htl.get('data_sly_list', [])}
- data-sly-test: {htl.get('data_sly_test', [])}

**Dialog Fields**:
{json.dumps(dialog.get('fields', []), indent=2)}

**JavaScript**:
```javascript
{clientlib.get('js_content', '')[:3000]}
```

**CSS**:
```css
{clientlib.get('css_content', '')[:3000]}
```

**Prop Mappings**:
{json.dumps(bdl_mapping.get('prop_mappings', []), indent=2)}

Transform to React.
""")
        ]
        
        structured_llm = llm.with_structured_output(TransformLogicOutput)
        
        try:
            # 优先使用结构化输出
            result: TransformLogicOutput = await structured_llm.ainvoke(messages)
            transformed = result.model_dump()
        except Exception:
            # Fallback: 普通解析
            try:
                response = await llm.ainvoke(messages)
                transformed = json.loads(response.content)
            except Exception:
                transformed = {
                    "typescript_interface": "",
                    "jsx_template": "",
                    "hooks": [],
                    "event_handlers": [],
                    "css_module": ""
                }
        
        updated_components[comp_id]["transformed"] = transformed
    
    return {"components": updated_components}


# ============================================================================
# React 生成节点
# ============================================================================

# [PROMPT] React 代码生成 Prompt
# [CUSTOMIZE] 以下需要根据实际项目调整:
#   - React 版本 (18 vs 19)
#   - 样式方案 (CSS Modules vs styled-components vs Tailwind)
#   - BDL 导入路径 (例如: @hsbc/bdl-react 或其他)
#   - 代码规范和 ESLint 配置
GENERATE_REACT_PROMPT = """You are an expert React developer.
Generate a complete, production-ready React component.

Requirements:
1. TypeScript with strict typing
2. Functional component with hooks
3. CSS Modules for styling
4. Proper imports from BDL design system
5. Accessibility support (ARIA, keyboard nav)
6. Responsive design
7. Comprehensive props interface

Generate:
1. Component file (.tsx)
2. Styles file (.module.css)
3. Index file (re-exports)
4. Types file if complex

Return JSON:
{
  "component_code": "full .tsx file content",
  "styles_code": "full .module.css content",
  "index_code": "export * from './ComponentName'",
  "types_code": "optional separate types file",
  "imports": ["react", "bdl/components", ...],
  "peer_dependencies": {"react": "^18.0.0", ...}
}
"""


async def generate_react(state: MigrationGraphState) -> Dict[str, Any]:
    """
    React 代码生成节点
    
    生成完整的 React 组件代码
    """
    components = state.get("components", {})
    updated_components = dict(components)
    component_registry = dict(state.get("component_registry", {}))
    stats = dict(state.get("stats", {}))
    
    # 使用工厂方法获取 LLM - LangGraph 1.0+ Best Practice
    llm = get_llm(task="generation", temperature=0)
    
    for comp_id, comp_data in components.items():
        if comp_data.get("status") != ComponentStatus.TRANSFORMING.value:
            continue
        
        updated_components[comp_id]["status"] = ComponentStatus.GENERATING.value
        
        aem_component = comp_data.get("aem_component", {})
        transformed = comp_data.get("transformed", {})
        bdl_mapping = comp_data.get("bdl_mapping", {})
        
        # 生成 React 组件名 (PascalCase)
        component_name = "".join(word.capitalize() for word in comp_id.split("-"))
        
        messages = [
            SystemMessage(content=GENERATE_REACT_PROMPT),
            HumanMessage(content=f"""
Generate React component:

**Component Name**: {component_name}
**Original AEM ID**: {comp_id}

**TypeScript Interface**:
```typescript
{transformed.get('typescript_interface', '')}
```

**JSX Template**:
```tsx
{transformed.get('jsx_template', '')}
```

**Hooks**:
{json.dumps(transformed.get('hooks', []), indent=2)}

**Event Handlers**:
{json.dumps(transformed.get('event_handlers', []), indent=2)}

**CSS Module**:
```css
{transformed.get('css_module', '')}
```

**BDL Mapping**:
{json.dumps(bdl_mapping, indent=2)}

Generate complete, production-ready files.
""")
        ]
        
        structured_llm = llm.with_structured_output(GenerateReactOutput)
        
        try:
            # 优先使用结构化输出
            result: GenerateReactOutput = await structured_llm.ainvoke(messages)
            generated = result.model_dump()
        except Exception:
            # Fallback: 普通解析
            try:
                response = await llm.ainvoke(messages)
                generated = json.loads(response.content)
            except Exception:
                generated = {
                    "component_code": f"// Failed to generate {component_name}",
                    "styles_code": "",
                    "index_code": "",
                    "imports": [],
                    "peer_dependencies": {}
                }
        
        # 保存生成结果
        updated_components[comp_id]["react_component"] = {
            "component_id": comp_id,
            "component_name": component_name,
            "component_code": generated.get("component_code", ""),
            "styles_code": generated.get("styles_code", ""),
            "index_code": generated.get("index_code", ""),
            "imports": generated.get("imports", []),
            "peer_dependencies": generated.get("peer_dependencies", {})
        }
        
        # 更新组件注册表
        resource_type = aem_component.get("resource_type", "")
        if resource_type:
            component_registry[resource_type] = component_name
        
        stats["generated_components"] = stats.get("generated_components", 0) + 1
    
    return {
        "components": updated_components,
        "component_registry": component_registry,
        "stats": stats,
        "current_phase": Phase.CONFIG_GENERATION.value
    }
