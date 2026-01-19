"""
配置生成节点实现
负责生成 CMS 编辑器配置文件

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
from typing import Any, Dict, List
from uuid import uuid4

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src.llm import get_llm
from src.core.state import (
    CMSConfig,
    EditableField,
    MigrationGraphState,
    Phase,
)


# ============================================================================
# Pydantic 输出模型 - LangGraph 1.0 结构化输出
# ============================================================================

class PropItem(BaseModel):
    """Props 项"""
    name: str = Field(description="Prop name")
    type: str = Field(description="Type: string|number|boolean|object|array")
    category: str = Field(default="content", description="content|style|behavior|layout")
    description: str = Field(default="")
    required: bool = Field(default=False)
    default_value: Any = Field(default=None)
    enum_values: List[Any] = Field(default_factory=list)


class ExtractPropsOutput(BaseModel):
    """Props 提取输出"""
    props: List[PropItem] = Field(default_factory=list)


class ValidationRule(BaseModel):
    """验证规则"""
    pattern: str = Field(default="")
    min_length: int = Field(default=0)
    max_length: int = Field(default=0)
    min: float = Field(default=0)
    max: float = Field(default=0)


class EditableFieldItem(BaseModel):
    """可编辑字段"""
    field_id: str = Field(description="Field ID")
    field_type: str = Field(description="TextInput|RichTextEditor|AssetPicker|...")
    prop_name: str = Field(description="Maps to which prop")
    label: str = Field(description="User-friendly label")
    description: str = Field(default="")
    placeholder: str = Field(default="")
    required: bool = Field(default=False)
    default_value: Any = Field(default=None)
    validation: ValidationRule = Field(default_factory=ValidationRule)
    conditional: Dict[str, Any] = Field(default_factory=dict)


class FieldGroup(BaseModel):
    """字段组"""
    group_id: str = Field(description="Group ID")
    label: str = Field(description="Group label")
    fields: List[str] = Field(default_factory=list, description="Field IDs in this group")
    collapsed: bool = Field(default=False)


class AnalyzeEditablesOutput(BaseModel):
    """可编辑区域分析输出"""
    editable_fields: List[EditableFieldItem] = Field(default_factory=list)
    field_groups: List[FieldGroup] = Field(default_factory=list)


# ============================================================================
# Props 提取节点
# ============================================================================

# [PROMPT] Props 提取 Prompt - 可根据 TypeScript 规范调整
EXTRACT_PROPS_PROMPT = """You are an expert in React and TypeScript.
Analyze the React component code and extract all configurable props.

Categorize props:
1. **Content Props**: text, images, links, rich text
2. **Style Props**: variants, sizes, colors, themes
3. **Behavior Props**: events, callbacks, toggles
4. **Layout Props**: alignment, spacing, responsive settings

For each prop, identify:
- name
- type (string, number, boolean, object, array)
- description
- required or optional
- default value if any
- valid values/enum if applicable

Return JSON:
{
  "props": [
    {
      "name": "...",
      "type": "...",
      "category": "content|style|behavior|layout",
      "description": "...",
      "required": boolean,
      "default_value": any,
      "enum_values": [...] or null
    }
  ]
}
"""


async def extract_props(state: MigrationGraphState) -> Dict[str, Any]:
    """
    提取组件 Props 节点
    
    从 React 组件代码中提取可配置属性
    """
    components = state.get("components", {})
    updated_components = dict(components)
    
    # 使用工厂方法获取 LLM - LangGraph 1.0+ Best Practice
    llm = get_llm(task="analysis", temperature=0)
    structured_llm = llm.with_structured_output(ExtractPropsOutput)
    
    for comp_id, comp_data in components.items():
        # 只处理已生成的组件
        react_component = comp_data.get("react_component", {})
        if not react_component:
            continue
        
        component_code = react_component.get("component_code", "")
        
        if not component_code:
            continue
        
        messages = [
            SystemMessage(content=EXTRACT_PROPS_PROMPT),
            HumanMessage(content=f"""
Extract props from this React component:

```tsx
{component_code}
```

Identify all configurable props with their types and metadata.
""")
        ]
        
        try:
            # 优先使用结构化输出 - LangGraph 1.0 推荐方式
            result: ExtractPropsOutput = await structured_llm.ainvoke(messages)
            props_data = result.model_dump()
        except Exception:
            # Fallback: 普通解析
            try:
                response = await llm.ainvoke(messages)
                props_data = json.loads(response.content)
            except:
                props_data = {"props": []}
        
        updated_components[comp_id]["extracted_props"] = props_data.get("props", [])
    
    return {"components": updated_components}


# ============================================================================
# 可编辑区域分析节点
# ============================================================================

# [PROMPT] 可编辑区域分析 Prompt
# [CUSTOMIZE] 编辑器字段类型映射需要根据目标 CMS 的编辑器能力调整
# 以下映射是示例，实际需要替换为您 CMS 支持的字段类型
EDITABLE_ANALYSIS_PROMPT = """You are an expert in CMS editor configuration.
Analyze the component props and AEM dialog fields to design the editor interface.

Map prop types to editor field types:
- string (short) → TextInput
- string (long) → TextArea
- string (rich) → RichTextEditor
- string (url) → LinkPicker
- image → AssetPicker/ImagePicker
- boolean → Toggle/Checkbox
- enum/select → Dropdown/RadioGroup
- number → NumberInput/Slider
- array → Repeater
- object → FieldGroup
(NOTE: [CUSTOMIZE] Replace these with actual field types supported by your CMS)

Consider:
1. User-friendly labels and help text
2. Field validation rules
3. Conditional field visibility
4. Field grouping for better UX
5. Inline editing opportunities

Return JSON:
{
  "editable_fields": [
    {
      "field_id": "...",
      "field_type": "TextInput|RichTextEditor|AssetPicker|...",
      "prop_name": "maps to which prop",
      "label": "user-friendly label",
      "description": "help text",
      "placeholder": "...",
      "required": boolean,
      "default_value": any,
      "validation": {
        "pattern": "regex if applicable",
        "min_length": number,
        "max_length": number,
        "min": number,
        "max": number
      },
      "show_when": {"field": "...", "value": "..."} or null
    }
  ],
  "field_groups": [
    {
      "group_id": "...",
      "label": "...",
      "fields": ["field_id1", "field_id2"]
    }
  ]
}
"""


async def analyze_editables(state: MigrationGraphState) -> Dict[str, Any]:
    """
    分析可编辑区域节点
    
    设计 CMS 编辑器界面
    """
    components = state.get("components", {})
    updated_components = dict(components)
    
    # 使用工厂方法获取 LLM - LangGraph 1.0+ Best Practice
    llm = get_llm(task="analysis", temperature=0.3)  # 稍高温度增加创造性
    
    for comp_id, comp_data in components.items():
        extracted_props = comp_data.get("extracted_props", [])
        aem_dialog = comp_data.get("aem_component", {}).get("dialog", {})
        
        if not extracted_props:
            continue
        
        # 创建结构化输出 LLM
        structured_llm = llm.with_structured_output(AnalyzeEditablesOutput)
        
        messages = [
            SystemMessage(content=EDITABLE_ANALYSIS_PROMPT),
            HumanMessage(content=f"""
Design editor interface for component: {comp_id}

**Extracted Props**:
{json.dumps(extracted_props, indent=2)}

**Original AEM Dialog Fields** (for reference):
{json.dumps(aem_dialog.get('fields', []), indent=2)}

Create optimal editor configuration.
""")
        ]
        
        try:
            # 优先使用结构化输出 - LangGraph 1.0 推荐方式
            result: AnalyzeEditablesOutput = await structured_llm.ainvoke(messages)
            editables = result.model_dump()
        except Exception:
            # Fallback: 普通解析
            try:
                response = await llm.ainvoke(messages)
                editables = json.loads(response.content)
            except:
                editables = {"editable_fields": [], "field_groups": []}
        
        updated_components[comp_id]["editables"] = editables
    
    return {"components": updated_components}


# ============================================================================
# Schema 生成节点
# ============================================================================

def generate_schema(state: MigrationGraphState) -> Dict[str, Any]:
    """
    生成 CMS 配置 Schema 节点
    
    创建符合 JSON Schema 的配置文件
    """
    components = state.get("components", {})
    updated_components = dict(components)
    configs = {}
    
    for comp_id, comp_data in components.items():
        editables = comp_data.get("editables", {})
        react_component = comp_data.get("react_component", {})
        aem_component = comp_data.get("aem_component", {})
        
        if not editables:
            continue
        
        editable_fields = editables.get("editable_fields", [])
        field_groups = editables.get("field_groups", [])
        
        # 构建 JSON Schema
        properties = {}
        required = []
        
        for field in editable_fields:
            field_id = field.get("field_id", "")
            field_type = field.get("field_type", "TextInput")
            
            # 映射到 JSON Schema 类型
            schema_type = _field_type_to_schema_type(field_type)
            
            prop_schema = {
                "type": schema_type,
                "title": field.get("label", field_id),
                "description": field.get("description", "")
            }
            
            # 添加验证规则
            validation = field.get("validation", {})
            if validation:
                if "pattern" in validation:
                    prop_schema["pattern"] = validation["pattern"]
                if "min_length" in validation:
                    prop_schema["minLength"] = validation["min_length"]
                if "max_length" in validation:
                    prop_schema["maxLength"] = validation["max_length"]
                if "min" in validation:
                    prop_schema["minimum"] = validation["min"]
                if "max" in validation:
                    prop_schema["maximum"] = validation["max"]
            
            # 默认值
            if field.get("default_value") is not None:
                prop_schema["default"] = field["default_value"]
            
            properties[field.get("prop_name", field_id)] = prop_schema
            
            if field.get("required"):
                required.append(field.get("prop_name", field_id))
        
        # [CUSTOMIZE] Schema URL 需要替换为实际的 CMS 域名
        json_schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "$id": f"https://cms.example.com/schemas/{comp_id}.json",  # [CUSTOMIZE] 替换 cms.example.com
            "type": "object",
            "title": react_component.get("component_name", comp_id),
            "description": aem_component.get("description", ""),
            "properties": properties,
            "required": required,
            "additionalProperties": False
        }
        
        # 构建完整配置
        cms_config = {
            "component_id": comp_id,
            "component_name": react_component.get("component_name", comp_id),
            "category": aem_component.get("component_group", "General"),
            "icon": _get_component_icon(comp_id),
            "json_schema": json_schema,
            "editable_fields": editable_fields,
            "field_groups": field_groups,
            "preview_config": {
                "preview_props": _get_preview_props(editable_fields),
                "viewport": "responsive"
            },
            "version": "1.0.0"
        }
        
        configs[comp_id] = cms_config
        updated_components[comp_id]["cms_config"] = cms_config
    
    return {
        "components": updated_components,
        "configs": configs
    }


def _field_type_to_schema_type(field_type: str) -> str:
    """将编辑器字段类型映射到 JSON Schema 类型"""
    # [CUSTOMIZE] 根据实际 CMS 支持的字段类型调整映射
    type_map = {
        "TextInput": "string",
        "TextArea": "string",
        "RichTextEditor": "string",
        "NumberInput": "number",
        "Slider": "number",
        "Toggle": "boolean",
        "Checkbox": "boolean",
        "Dropdown": "string",
        "RadioGroup": "string",
        "AssetPicker": "object",
        "ImagePicker": "object",
        "LinkPicker": "object",
        "Repeater": "array",
        "FieldGroup": "object"
    }
    return type_map.get(field_type, "string")


def _get_component_icon(comp_id: str) -> str:
    """根据组件 ID 推断图标"""
    # [CUSTOMIZE] 图标名称需要替换为实际 CMS 使用的图标库名称
    # 例如: Lucide Icons, FontAwesome, Material Icons 等
    icon_map = {
        "hero": "image",
        "banner": "image",
        "button": "cursor",
        "card": "credit-card",
        "carousel": "layers",
        "accordion": "list",
        "tabs": "folder",
        "form": "file-text",
        "nav": "menu",
        "footer": "layout",
        "header": "layout"
    }
    
    for key, icon in icon_map.items():
        if key in comp_id.lower():
            return icon
    
    return "component"


def _get_preview_props(fields: List[Dict]) -> Dict[str, Any]:
    """生成预览用的默认属性"""
    preview = {}
    
    for field in fields:
        prop_name = field.get("prop_name", field.get("field_id", ""))
        field_type = field.get("field_type", "")
        default = field.get("default_value")
        
        if default is not None:
            preview[prop_name] = default
        elif field_type in ["TextInput", "TextArea"]:
            preview[prop_name] = f"Sample {field.get('label', 'text')}"
        elif field_type == "RichTextEditor":
            preview[prop_name] = f"<p>Sample {field.get('label', 'content')}</p>"
        elif field_type in ["NumberInput", "Slider"]:
            preview[prop_name] = 0
        elif field_type in ["Toggle", "Checkbox"]:
            preview[prop_name] = False
        elif field_type in ["AssetPicker", "ImagePicker"]:
            preview[prop_name] = {"url": "https://via.placeholder.com/400x300"}
    
    return preview


# ============================================================================
# 配置验证节点
# ============================================================================

def validate_config(state: MigrationGraphState) -> Dict[str, Any]:
    """
    验证配置节点
    
    确保生成的配置有效且完整
    """
    configs = state.get("configs", {})
    errors = list(state.get("errors", []))
    warnings = list(state.get("warnings", []))
    
    for config_id, config in configs.items():
        # 验证 JSON Schema
        json_schema = config.get("json_schema", {})
        
        if not json_schema.get("properties"):
            warnings.append({
                "warning_id": str(uuid4()),
                "message": f"Config {config_id} has no properties defined",
                "component_id": config_id
            })
        
        # 验证必填字段
        required = json_schema.get("required", [])
        properties = json_schema.get("properties", {})
        
        for req in required:
            if req not in properties:
                errors.append({
                    "error_id": str(uuid4()),
                    "severity": "critical",
                    "error_type": "ConfigValidationError",
                    "message": f"Required field '{req}' not in properties for {config_id}",
                    "component_id": config_id
                })
        
        # 验证编辑字段与 props 一致性
        editable_fields = config.get("editable_fields", [])
        for field in editable_fields:
            prop_name = field.get("prop_name", "")
            if prop_name and prop_name not in properties:
                warnings.append({
                    "warning_id": str(uuid4()),
                    "message": f"Editable field '{prop_name}' not in schema for {config_id}",
                    "component_id": config_id
                })
    
    return {
        "errors": errors,
        "warnings": warnings,
        "current_phase": Phase.AUTO_REVIEW.value
    }
