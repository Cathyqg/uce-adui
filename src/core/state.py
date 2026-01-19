"""
uce-adui - State Definitions
核心状态管理

LangGraph 1.0 Best Practices:
1. 使用 Annotated + reducer 函数处理状态更新
2. 使用 Pydantic BaseModel 进行结构化输出验证
3. 状态字段明确定义默认值和更新策略
"""
from __future__ import annotations

import operator
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Dict, List, Literal, Optional, Sequence, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from pydantic import BaseModel, Field


# ============================================================================
# Enums - 状态枚举
# ============================================================================

class Phase(str, Enum):
    """迁移阶段"""
    INITIALIZATION = "initialization"
    COMPONENT_PARSING = "component_parsing"
    COMPONENT_ANALYSIS = "component_analysis"
    BDL_MAPPING = "bdl_mapping"
    CODE_GENERATION = "code_generation"
    CONFIG_GENERATION = "config_generation"
    PAGE_MIGRATION = "page_migration"
    AUTO_REVIEW = "auto_review"
    HUMAN_REVIEW = "human_review"
    COMPLETED = "completed"
    FAILED = "failed"


class ComponentStatus(str, Enum):
    """组件处理状态"""
    PENDING = "pending"
    PARSING = "parsing"
    ANALYZING = "analyzing"
    MAPPING = "mapping"
    TRANSFORMING = "transforming"
    GENERATING = "generating"
    AUTO_REVIEWING = "auto_reviewing"
    HUMAN_REVIEWING = "human_reviewing"
    APPROVED = "approved"
    REJECTED = "rejected"
    FAILED = "failed"


class ReviewDecision(str, Enum):
    """人工审查决定"""
    APPROVE = "approve"
    REJECT = "reject"
    MODIFY = "modify"
    SKIP = "skip"
    ESCALATE = "escalate"


class IssueSeverity(str, Enum):
    """问题严重程度"""
    CRITICAL = "critical"  # 阻断性问题
    MAJOR = "major"        # 严重问题
    MINOR = "minor"        # 轻微问题
    SUGGESTION = "suggestion"  # 建议


class ErrorSeverity(str, Enum):
    """错误严重程度"""
    FATAL = "fatal"
    CRITICAL = "critical"
    RECOVERABLE = "recoverable"
    WARNING = "warning"


# ============================================================================
# Source Code Structures - AEM 源码结构
# ============================================================================

@dataclass
class HTLTemplate:
    """HTL/Sightly 模板解析结果"""
    raw_content: str
    data_sly_use: List[Dict[str, str]] = field(default_factory=list)
    data_sly_list: List[Dict[str, Any]] = field(default_factory=list)
    data_sly_test: List[Dict[str, str]] = field(default_factory=list)
    data_sly_resource: List[Dict[str, str]] = field(default_factory=list)
    text_expressions: List[str] = field(default_factory=list)
    html_structure: Optional[str] = None


@dataclass
class AEMDialog:
    """AEM 组件对话框定义"""
    raw_xml: str
    fields: List[Dict[str, Any]] = field(default_factory=list)
    tabs: List[Dict[str, Any]] = field(default_factory=list)
    validation_rules: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ClientLib:
    """AEM 客户端库"""
    categories: List[str] = field(default_factory=list)
    js_files: List[str] = field(default_factory=list)
    css_files: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)


@dataclass
class AEMComponent:
    """AEM 组件完整定义"""
    component_id: str
    resource_type: str
    component_group: str
    title: str
    description: str = ""
    
    # 源码
    htl_template: Optional[HTLTemplate] = None
    dialog: Optional[AEMDialog] = None
    clientlib: Optional[ClientLib] = None
    
    # 文件路径
    source_path: str = ""
    htl_path: str = ""
    dialog_path: str = ""
    
    # 元数据
    sling_model: Optional[str] = None
    is_container: bool = False
    allowed_children: List[str] = field(default_factory=list)


# ============================================================================
# Analysis Structures - 分析结果结构
# ============================================================================

@dataclass
class ComponentDependency:
    """组件依赖关系"""
    component_id: str
    dependency_type: Literal["component", "model", "service", "clientlib"]
    required: bool = True


@dataclass 
class ComponentComplexity:
    """组件复杂度评估"""
    lines_of_code: int = 0
    cyclomatic_complexity: int = 0
    dependency_count: int = 0
    interaction_points: int = 0
    
    @property
    def overall_score(self) -> str:
        """整体复杂度评分"""
        total = self.lines_of_code / 100 + self.cyclomatic_complexity + self.dependency_count
        if total < 5:
            return "low"
        elif total < 15:
            return "medium"
        return "high"


@dataclass
class AnalyzedComponent:
    """组件分析结果"""
    component_id: str
    component_type: Literal["ui", "container", "layout", "utility"]
    is_dynamic: bool
    dependencies: List[ComponentDependency] = field(default_factory=list)
    complexity: ComponentComplexity = field(default_factory=ComponentComplexity)
    
    # 功能特性
    has_form: bool = False
    has_animation: bool = False
    has_responsive: bool = False
    has_accessibility: bool = False
    
    # BDL 映射可行性
    bdl_mapping_feasibility: float = 0.0  # 0-1
    mapping_notes: List[str] = field(default_factory=list)


# ============================================================================
# BDL Mapping Structures - BDL 映射结构
# ============================================================================

@dataclass
class PropMapping:
    """属性映射"""
    aem_prop: str
    bdl_prop: str
    transform: Optional[str] = None  # 转换函数名
    default_value: Any = None


@dataclass
class StyleMapping:
    """样式映射"""
    aem_class: str
    bdl_token: str
    css_property: str


@dataclass
class BDLMapping:
    """BDL 映射方案"""
    aem_component_id: str
    bdl_component_name: Optional[str] = None  # None 表示需要自定义
    
    # 映射详情
    prop_mappings: List[PropMapping] = field(default_factory=list)
    style_mappings: List[StyleMapping] = field(default_factory=list)
    event_mappings: Dict[str, str] = field(default_factory=dict)
    
    # 差异
    missing_in_bdl: List[str] = field(default_factory=list)  # BDL 缺失的功能
    requires_custom: List[str] = field(default_factory=list)  # 需要自定义的部分
    
    # 置信度
    confidence_score: float = 0.0


# ============================================================================
# Generated Code Structures - 生成代码结构
# ============================================================================

@dataclass
class TypeScriptInterface:
    """TypeScript 接口定义"""
    name: str
    properties: List[Dict[str, Any]]
    extends: List[str] = field(default_factory=list)
    code: str = ""


@dataclass
class ReactHook:
    """React Hook 定义"""
    name: str
    dependencies: List[str]
    code: str


@dataclass
class ReactComponent:
    """生成的 React 组件"""
    component_id: str
    component_name: str  # PascalCase
    
    # 代码
    interface_code: str = ""
    component_code: str = ""
    styles_code: str = ""
    index_code: str = ""
    
    # 测试和文档
    test_code: str = ""
    storybook_code: str = ""
    readme: str = ""
    
    # 依赖
    imports: List[str] = field(default_factory=list)
    peer_dependencies: Dict[str, str] = field(default_factory=dict)


# ============================================================================
# Config Structures - CMS 配置结构
# ============================================================================

@dataclass
class EditableField:
    """可编辑字段定义"""
    field_id: str
    field_type: str  # text, richtext, image, link, select, etc.
    label: str
    description: str = ""
    required: bool = False
    default_value: Any = None
    
    # 验证
    validation: Dict[str, Any] = field(default_factory=dict)
    
    # UI 配置
    placeholder: str = ""
    help_text: str = ""
    
    # 条件显示
    show_when: Optional[Dict[str, Any]] = None


@dataclass
class CMSConfig:
    """CMS 组件配置"""
    component_id: str
    component_name: str
    category: str
    icon: str = ""
    
    # Schema
    json_schema: Dict[str, Any] = field(default_factory=dict)
    
    # 编辑器配置
    editable_fields: List[EditableField] = field(default_factory=list)
    field_groups: List[Dict[str, Any]] = field(default_factory=list)
    
    # 预览配置
    preview_config: Dict[str, Any] = field(default_factory=dict)
    
    # 版本
    version: str = "1.0.0"


# ============================================================================
# Page Structures - 页面结构
# ============================================================================

@dataclass
class PageComponent:
    """页面中的组件实例"""
    instance_id: str
    component_type: str
    props: Dict[str, Any] = field(default_factory=dict)
    children: List["PageComponent"] = field(default_factory=list)
    grid_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AEMPage:
    """AEM 页面结构"""
    page_path: str
    page_title: str
    template: str
    components: List[PageComponent] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CMSPage:
    """新 CMS 页面结构"""
    page_id: str
    page_slug: str
    page_title: str
    template_id: str
    
    # 组件树
    components: List[Dict[str, Any]] = field(default_factory=list)
    
    # 元数据
    seo: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # 版本控制
    version: str = "1.0.0"
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


# ============================================================================
# Review Structures - 审查结构
# ============================================================================

@dataclass
class ReviewIssue:
    """审查问题"""
    issue_id: str
    category: str  # code_quality, bdl_compliance, function_parity
    severity: IssueSeverity
    title: str
    description: str
    location: Optional[str] = None  # 文件:行号
    suggestion: str = ""
    auto_fixable: bool = False


@dataclass
class ReviewResult:
    """单项审查结果"""
    reviewer: str  # code_quality, bdl_compliance, function_parity
    score: int  # 0-100
    passed: bool
    issues: List[ReviewIssue] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class AggregatedReview:
    """聚合审查结果"""
    component_id: str
    
    # 各项审查结果
    code_quality: Optional[ReviewResult] = None
    bdl_compliance: Optional[ReviewResult] = None
    function_parity: Optional[ReviewResult] = None
    
    # 综合评分
    overall_score: float = 0.0
    auto_approved: bool = False
    requires_human_review: bool = False
    
    # 人工审查
    human_decision: Optional[ReviewDecision] = None
    human_feedback: str = ""
    human_reviewer: str = ""
    human_review_timestamp: Optional[datetime] = None


# ============================================================================
# Error Structures - 错误结构
# ============================================================================

@dataclass
class MigrationError:
    """迁移错误"""
    error_id: str
    severity: ErrorSeverity
    error_type: str
    message: str
    component_id: Optional[str] = None
    phase: Optional[Phase] = None
    stack_trace: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    retry_count: int = 0


@dataclass
class MigrationWarning:
    """迁移警告"""
    warning_id: str
    message: str
    component_id: Optional[str] = None
    phase: Optional[Phase] = None
    timestamp: datetime = field(default_factory=datetime.now)


# ============================================================================
# Statistics - 统计结构
# ============================================================================

@dataclass
class MigrationStats:
    """迁移统计"""
    total_components: int = 0
    parsed_components: int = 0
    generated_components: int = 0
    approved_components: int = 0
    failed_components: int = 0
    
    total_pages: int = 0
    migrated_pages: int = 0
    
    avg_code_quality_score: float = 0.0
    avg_bdl_compliance_score: float = 0.0
    avg_function_parity_score: float = 0.0
    
    human_review_count: int = 0
    auto_fix_count: int = 0
    
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    @property
    def duration_seconds(self) -> float:
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0


# ============================================================================
# Component State - 单个组件状态
# ============================================================================

@dataclass
class ComponentState:
    """单个组件的完整状态"""
    component_id: str
    status: ComponentStatus = ComponentStatus.PENDING
    
    # 各阶段数据
    aem_component: Optional[AEMComponent] = None
    analyzed: Optional[AnalyzedComponent] = None
    bdl_mapping: Optional[BDLMapping] = None
    react_component: Optional[ReactComponent] = None
    cms_config: Optional[CMSConfig] = None
    
    # 审查
    review: Optional[AggregatedReview] = None
    
    # 错误处理
    errors: List[MigrationError] = field(default_factory=list)
    warnings: List[MigrationWarning] = field(default_factory=list)
    retry_count: int = 0
    last_error: Optional[str] = None


# ============================================================================
# Pydantic Models - 结构化输出验证 (LangGraph 1.0 Best Practice)
# ============================================================================

class ReviewVerdict(BaseModel):
    """审查结论的结构化输出"""
    verdict: Literal["APPROVE", "REJECT", "MODIFY", "ESCALATE"] = Field(
        ..., description="审查决定"
    )
    reason: str = Field(..., description="决定原因")
    confidence: float = Field(..., ge=0, le=1, description="置信度 0-1")
    suggestions: List[str] = Field(default_factory=list, description="改进建议")


class CodeQualityOutput(BaseModel):
    """代码质量审查的结构化输出"""
    score: int = Field(..., ge=0, le=100, description="得分 0-100")
    issues: List[Dict[str, Any]] = Field(default_factory=list)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    summary: str = Field(default="", description="总体评估")


class BDLComplianceOutput(BaseModel):
    """BDL 合规性审查的结构化输出"""
    score: int = Field(..., ge=0, le=100)
    token_violations: List[str] = Field(default_factory=list)
    structure_issues: List[str] = Field(default_factory=list)
    theme_support: bool = Field(default=False)
    responsive_compliant: bool = Field(default=False)


class FunctionParityOutput(BaseModel):
    """功能一致性审查的结构化输出"""
    parity_score: int = Field(..., ge=0, le=100)
    visual_diff: List[str] = Field(default_factory=list)
    behavior_diff: List[str] = Field(default_factory=list)
    data_handling_diff: List[str] = Field(default_factory=list)


# ============================================================================
# Reducer Functions - 状态更新策略 (LangGraph 1.0 核心模式)
# ============================================================================

def merge_dicts(left: Dict, right: Dict) -> Dict:
    """合并字典，right 覆盖 left"""
    if not left:
        return right or {}
    if not right:
        return left
    return {**left, **right}


def append_list(left: List, right: List) -> List:
    """追加列表项"""
    if not left:
        return right or []
    if not right:
        return left
    return left + right


def replace_value(left: Any, right: Any) -> Any:
    """直接替换值"""
    return right if right is not None else left


def merge_review_results(left: Dict, right: Dict) -> Dict:
    """
    合并并行审查结果的 Reducer
    
    LangGraph Send API 模式:
    - 每个并行审查节点返回 {"review_type": {...result...}}
    - 此 Reducer 将多个并行结果合并到同一个字典
    
    Example:
        left = {"code_quality": {...}}
        right = {"bdl_compliance": {...}}
        result = {"code_quality": {...}, "bdl_compliance": {...}}
    """
    if right and isinstance(right, dict) and right.get("__clear__"):
        return {}
    if not left:
        return right or {}
    if not right:
        return left
    
    merged = dict(left)
    for key, value in right.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            # 深度合并组件级审查结果
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    return merged


# ============================================================================
# LangGraph State - 图状态定义 (LangGraph 1.0 Best Practice)
# ============================================================================

class MigrationGraphState(TypedDict, total=False):
    """
    LangGraph 图的状态定义
    
    LangGraph 1.0 最佳实践:
    1. 使用 Annotated[Type, reducer] 定义可累积的字段
    2. messages 使用内置的 add_messages reducer
    3. 明确区分覆盖型和累积型字段
    """
    
    # === 会话信息 (覆盖型) ===
    session_id: str
    started_at: str  # ISO format datetime
    current_phase: str
    
    # === 输入配置 (覆盖型) ===
    source_path: str
    aem_page_json_paths: List[str]
    config: Dict[str, Any]
    
    # === 组件状态 (合并型 - 使用 Annotated + reducer) ===
    # 新的组件数据会与现有数据合并，而非完全覆盖
    components: Annotated[Dict[str, Dict[str, Any]], merge_dicts]
    component_queue: List[str]  # 待处理队列 - 覆盖型
    current_component_id: Optional[str]
    
    # === 配置生成状态 (合并型) ===
    configs: Annotated[Dict[str, Dict[str, Any]], merge_dicts]
    
    # === 页面迁移状态 (合并型) ===
    pages: Annotated[Dict[str, Dict[str, Any]], merge_dicts]
    page_queue: List[str]
    current_page_id: Optional[str]
    
    # === 审查状态 ===
    pending_human_review: List[str]  # 覆盖型 - 每次重新计算
    human_review_decisions: Annotated[Dict[str, Dict[str, Any]], merge_dicts]
    
    # === 并行审查结果 (Send API 模式) ===
    # 每个并行审查节点的结果会通过 merge_review_results 合并
    # 格式: {component_id: {review_type: {score, issues, ...}}}
    parallel_review_results: Annotated[Dict[str, Dict[str, Any]], merge_review_results]
    
    # === 全局资源 (合并型) ===
    bdl_spec: Dict[str, Any]
    component_registry: Annotated[Dict[str, str], merge_dicts]
    
    # === 错误和警告 (追加型 - 错误只增不减) ===
    errors: Annotated[List[Dict[str, Any]], append_list]
    warnings: Annotated[List[Dict[str, Any]], append_list]
    
    # === 统计 (合并型) ===
    stats: Annotated[Dict[str, Any], merge_dicts]
    
    # === 消息历史 (使用 LangGraph 内置 reducer) ===
    messages: Annotated[Sequence[AnyMessage], add_messages]
    
    # === 中断控制 (覆盖型) ===
    should_interrupt: bool
    interrupt_reason: Optional[str]
    human_review_package: Optional[Dict[str, Any]]  # 新增: 审查数据包


# ============================================================================
# Helper Functions - 辅助函数
# ============================================================================

def create_initial_state(
    source_path: str,
    aem_page_json_paths: Optional[List[str]] = None,
    config: Optional[Dict[str, Any]] = None
) -> MigrationGraphState:
    """创建初始状态"""
    import uuid
    
    return MigrationGraphState(
        session_id=str(uuid.uuid4()),
        started_at=datetime.now().isoformat(),
        current_phase=Phase.INITIALIZATION.value,
        source_path=source_path,
        aem_page_json_paths=aem_page_json_paths or [],
        config=config or {},
        components={},
        component_queue=[],
        current_component_id=None,
        configs={},
        pages={},
        page_queue=[],
        current_page_id=None,
        pending_human_review=[],
        human_review_decisions={},
        bdl_spec={},
        component_registry={},
        errors=[],
        warnings=[],
        stats={
            "total_components": 0,
            "parsed_components": 0,
            "generated_components": 0,
            "approved_components": 0,
            "failed_components": 0,
            "start_time": datetime.now().isoformat()
        },
        messages=[],
        should_interrupt=False,
        interrupt_reason=None
    )


def serialize_component_state(state: ComponentState) -> Dict[str, Any]:
    """序列化组件状态为字典"""
    from dataclasses import asdict
    return asdict(state)


def deserialize_component_state(data: Dict[str, Any]) -> ComponentState:
    """从字典反序列化组件状态"""
    # 需要特殊处理嵌套的 dataclass
    return ComponentState(**data)
