"""
Agent 核心定义
所有 Agent 的创建函数集中在此

架构说明:
- 此文件只负责 Agent 的创建（create_*_agent 函数）
- 节点包装器已移动到 src/nodes/intelligent/
- 这样 agents/ 模块纯粹作为"Agent 基础设施库"

使用方式:
    from src.agents.core import create_bdl_mapping_agent
    agent = create_bdl_mapping_agent()
    
    # 或使用工厂
    from src.agents import AgentFactory, AgentType
    agent = AgentFactory().create_agent(AgentType.BDL_MAPPER)
"""
from pydantic import BaseModel, Field
from typing import List, Any, Dict

from src.llm import get_llm
from src.agents.config import PromptTemplates
from src.agents.utils import create_structured_agent
from src.tools import (
    search_bdl_components,
    get_bdl_component_spec,
    list_bdl_components,
    get_bdl_design_token,
    validate_typescript_syntax,
    lint_react_code,
    format_with_prettier,
    validate_bdl_compliance,
)


# ============================================================================
# 输出模型定义
# ============================================================================

class PropMappingItem(BaseModel):
    """属性映射项"""
    aem_prop: str
    bdl_prop: str
    transform: str = Field(default="")


class BDLMappingOutput(BaseModel):
    """BDL 映射输出"""
    bdl_component_name: str
    confidence_score: float = Field(ge=0, le=1)
    prop_mappings: List[PropMappingItem] = Field(default_factory=list)
    missing_features: List[str] = Field(default_factory=list)
    reasoning: str


class CodeGeneratorOutput(BaseModel):
    """代码生成输出"""
    component_code: str
    styles_code: str = Field(default="")
    index_code: str = Field(default="")
    validation_passed: bool
    validation_summary: str = Field(default="")
    tool_calls_made: int = Field(default=0)
    issues_fixed: List[str] = Field(default_factory=list)


class ReviewIssueItem(BaseModel):
    """审查问题项"""
    severity: str
    title: str
    description: str
    location: str = Field(default="")
    suggestion: str = Field(default="")
    auto_fixable: bool = Field(default=False)


class CodeReviewOutput(BaseModel):
    """代码审查输出"""
    overall_score: int = Field(ge=0, le=100)
    decision: str
    issues: List[ReviewIssueItem] = Field(default_factory=list)
    tool_validation_results: Dict[str, Any] = Field(default_factory=dict)
    summary: str
    reasoning: str


class EditableFieldItem(BaseModel):
    """可编辑字段"""
    field_id: str
    field_type: str
    prop_name: str
    label: str
    description: str = Field(default="")
    placeholder: str = Field(default="")
    required: bool = Field(default=False)


class FieldGroupItem(BaseModel):
    """字段分组"""
    group_id: str
    label: str
    fields: List[str]
    collapsed: bool = Field(default=False)


class EditorDesignOutput(BaseModel):
    """编辑器设计输出"""
    editable_fields: List[EditableFieldItem] = Field(default_factory=list)
    field_groups: List[FieldGroupItem] = Field(default_factory=list)
    editor_layout: str = Field(default="vertical")
    ux_notes: str


# ============================================================================
# Agent 创建函数
# ============================================================================

def create_bdl_mapping_agent(llm=None):
    """
    创建 BDL 映射 Agent
    
    功能: 智能搜索和映射 BDL 组件
    工具: search_bdl_components, get_bdl_component_spec, list_bdl_components
    """
    llm = llm or get_llm(task="analysis", temperature=0.1)
    
    tools = [
        search_bdl_components,
        get_bdl_component_spec,
        list_bdl_components,
        get_bdl_design_token,
    ]
    
    system_prompt = PromptTemplates.BDL_MAPPER
    
    return create_structured_agent(
        llm, tools,
        system_prompt=system_prompt,
        response_format=BDLMappingOutput,
    )


def create_code_generator_agent(llm=None):
    """
    创建代码生成 Agent
    
    功能: 生成并验证 React 组件代码
    工具: validate_typescript_syntax, lint_react_code, format_with_prettier
    """
    llm = llm or get_llm(task="generation", temperature=0)
    
    tools = [
        validate_typescript_syntax,
        lint_react_code,
        format_with_prettier,
    ]
    
    system_prompt = PromptTemplates.CODE_GENERATOR
    
    return create_structured_agent(
        llm, tools,
        system_prompt=system_prompt,
        response_format=CodeGeneratorOutput,
    )


def create_code_fixer_agent(llm=None):
    """
    创建代码修复 Agent
    
    功能: 针对已生成代码进行小范围修复
    工具: validate_typescript_syntax, lint_react_code, format_with_prettier
    """
    llm = llm or get_llm(task="generation", temperature=0)
    
    tools = [
        validate_typescript_syntax,
        lint_react_code,
        format_with_prettier,
    ]
    
    system_prompt = PromptTemplates.CODE_FIXER
    
    return create_structured_agent(
        llm, tools,
        system_prompt=system_prompt,
        response_format=CodeGeneratorOutput,
    )


def create_code_reviewer_agent(llm=None):
    """
    创建代码审查 Agent
    
    功能: 全面审查代码质量
    工具: validate_typescript_syntax, lint_react_code, validate_bdl_compliance
    """
    llm = llm or get_llm(task="review", temperature=0)
    
    tools = [
        validate_typescript_syntax,
        lint_react_code,
        validate_bdl_compliance,
    ]
    
    system_prompt = PromptTemplates.CODE_REVIEWER
    
    return create_structured_agent(
        llm, tools,
        system_prompt=system_prompt,
        response_format=CodeReviewOutput,
    )


def create_editor_designer_agent(llm=None):
    """
    创建编辑器设计 Agent
    
    功能: 设计用户友好的编辑器界面
    工具: 无（纯设计任务）
    """
    llm = llm or get_llm(task="analysis", temperature=0.3)
    
    system_prompt = PromptTemplates.EDITOR_DESIGNER
    
    return create_structured_agent(
        llm, [],
        system_prompt=system_prompt,
        response_format=EditorDesignOutput,
    )


# ============================================================================
# 导出
# ============================================================================

__all__ = [
    # Agent 创建函数
    "create_bdl_mapping_agent",
    "create_code_generator_agent",
    "create_code_reviewer_agent",
    "create_editor_designer_agent",
    # 输出模型
    "BDLMappingOutput",
    "CodeGeneratorOutput",
    "CodeReviewOutput",
    "EditorDesignOutput",
]
