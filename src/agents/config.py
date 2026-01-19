"""
Agent 配置管理模块
集中管理所有 Agent 的配置、提示词和参数

设计模式:
- 单例模式: 全局配置管理器
- 构建者模式: 灵活构建配置
- 观察者模式: 配置变更通知（可选）

优势:
1. 集中管理提示词（便于调优）
2. 统一配置接口
3. 环境变量支持
4. 配置验证
5. 配置导出/导入
"""
from typing import Any, Dict, List, Optional
from enum import Enum
from pydantic import BaseModel, Field
import os
import json


# ============================================================================
# 提示词模板
# ============================================================================

class PromptTemplates:
    """Agent 提示词模板集合"""
    
    BDL_MAPPER = """You are an expert in component mapping and design system analysis.

Your task: Find the BEST BDL component mapping for an AEM component.

PROCESS (use tools!):
1. UNDERSTAND the AEM component structure and features
2. SEARCH for candidates using search_bdl_components
3. ANALYZE each candidate with get_bdl_component_spec
4. COMPARE and choose the best match
5. VALIDATE the chosen component

IMPORTANT:
- ALWAYS use tools (don't guess!)
- Compare at least 2-3 candidates
- Be thorough - mapping accuracy is critical

OUTPUT: Return structured JSON with mapping details."""

    CODE_GENERATOR = """You are an expert React + TypeScript developer.

Your task: Generate production-ready React component code.

PROCESS:
1. GENERATE initial code with strict typing
2. VALIDATE syntax with validate_typescript_syntax
3. CHECK style with lint_react_code  
4. FORMAT with format_with_prettier
5. ITERATE until all validations pass

IMPORTANT:
- ALWAYS validate before claiming done
- Fix ALL critical issues
- Maximum 3-4 tool call cycles

OUTPUT: Return component code, styles, and validation status."""

    CODE_REVIEWER = """You are an expert code reviewer for React, TypeScript, and design systems.

Your task: Review component code comprehensively.

PROCESS:
1. SYNTAX validation (validate_typescript_syntax)
2. STYLE check (lint_react_code)
3. BDL compliance (validate_bdl_compliance)
4. MANUAL review (React best practices, architecture, accessibility)
5. CALCULATE score and make decision

DECISION RULES:
- Score >= 85 + no critical issues → APPROVE
- Score >= 60 OR critical issues → NEEDS_HUMAN
- Score < 60 → REJECT

OUTPUT: Return score, decision, issues, and reasoning."""

    EDITOR_DESIGNER = """You are an expert UX designer for CMS editor interfaces.

Your task: Design intuitive editor interfaces for React components.

ANALYSIS:
1. UNDERSTAND component purpose and user needs
2. CATEGORIZE props (content, style, behavior, layout)
3. SELECT appropriate field types
4. ORGANIZE fields logically
5. ENHANCE UX (labels, help text, grouping)

BEST PRACTICES:
- Content fields first and prominent
- Group related fields
- Clear, friendly labels
- Sensible defaults

OUTPUT: Return editor configuration with fields, groups, and UX notes."""


# ============================================================================
# Agent 配置类
# ============================================================================

class AgentConfiguration(BaseModel):
    """单个 Agent 的配置"""
    
    # LLM 配置
    task_type: str = Field(description="parsing|analysis|generation|review")
    temperature: float = Field(ge=0, le=2, default=0.0)
    provider: Optional[str] = Field(default=None)
    model: Optional[str] = Field(default=None)
    
    # Agent 行为配置
    max_iterations: int = Field(ge=1, le=50, default=10)
    timeout_seconds: int = Field(default=300)
    
    # 提示词
    system_prompt: str = Field(description="Agent system prompt")
    
    # 工具配置
    enabled_tools: List[str] = Field(default_factory=list)
    
    # 输出配置
    response_format: Optional[str] = Field(default=None, description="Pydantic model class name")
    
    # 重试配置
    max_retries: int = Field(default=3)
    retry_delay: float = Field(default=1.0)
    
    # Metadata
    description: str = Field(default="")
    version: str = Field(default="1.0.0")


# ============================================================================
# 配置管理器
# ============================================================================

class AgentConfigManager:
    """
    Agent 配置管理器（单例）
    
    功能:
    1. 加载和保存配置
    2. 环境变量覆盖
    3. 配置验证
    4. 热重载支持
    
    Example:
        >>> manager = AgentConfigManager()
        >>> config = manager.get_config("bdl_mapper")
        >>> manager.update_config("bdl_mapper", {"temperature": 0.5})
        >>> manager.save_to_file("configs/agents.json")
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        初始化配置管理器
        
        Args:
            config_file: 配置文件路径（可选）
        """
        self.configs: Dict[str, AgentConfiguration] = {}
        self._load_default_configs()
        
        if config_file and os.path.exists(config_file):
            self.load_from_file(config_file)
        
        # 应用环境变量覆盖
        self._apply_env_overrides()
    
    def _load_default_configs(self):
        """加载默认配置"""
        self.configs = {
            "bdl_mapper": AgentConfiguration(
                task_type="analysis",
                temperature=0.1,
                provider="litellm",
                max_iterations=10,
                system_prompt=PromptTemplates.BDL_MAPPER,
                enabled_tools=["search_bdl_components", "get_bdl_component_spec"],
                description="Maps AEM components to BDL components",
            ),
            "code_generator": AgentConfiguration(
                task_type="generation",
                temperature=0.0,
                max_iterations=15,
                system_prompt=PromptTemplates.CODE_GENERATOR,
                enabled_tools=["validate_typescript_syntax", "lint_react_code", "format_with_prettier"],
                description="Generates React component code",
            ),
            "code_reviewer": AgentConfiguration(
                task_type="review",
                temperature=0.0,
                provider="litellm",
                max_iterations=5,
                system_prompt=PromptTemplates.CODE_REVIEWER,
                enabled_tools=["validate_typescript_syntax", "lint_react_code", "validate_bdl_compliance"],
                description="Reviews code quality and compliance",
            ),
            "editor_designer": AgentConfiguration(
                task_type="analysis",
                temperature=0.3,
                max_iterations=5,
                system_prompt=PromptTemplates.EDITOR_DESIGNER,
                enabled_tools=[],
                description="Designs CMS editor interfaces",
            ),
        }
    
    def _apply_env_overrides(self):
        """应用环境变量覆盖"""
        for agent_name in self.configs:
            # 温度覆盖
            env_key = f"AGENT_{agent_name.upper()}_TEMPERATURE"
            if os.getenv(env_key):
                try:
                    self.configs[agent_name].temperature = float(os.getenv(env_key))
                except ValueError:
                    pass
            
            # 提供商覆盖
            env_key = f"AGENT_{agent_name.upper()}_PROVIDER"
            if os.getenv(env_key):
                self.configs[agent_name].provider = os.getenv(env_key)
            
            # 模型覆盖
            env_key = f"AGENT_{agent_name.upper()}_MODEL"
            if os.getenv(env_key):
                self.configs[agent_name].model = os.getenv(env_key)
    
    def get_config(self, agent_name: str) -> AgentConfiguration:
        """获取 Agent 配置"""
        if agent_name not in self.configs:
            raise ValueError(f"Unknown agent: {agent_name}")
        return self.configs[agent_name]
    
    def update_config(self, agent_name: str, updates: Dict[str, Any]):
        """
        更新 Agent 配置
        
        Args:
            agent_name: Agent 名称
            updates: 更新的字段
        """
        if agent_name not in self.configs:
            raise ValueError(f"Unknown agent: {agent_name}")
        
        current = self.configs[agent_name]
        updated_data = {**current.model_dump(), **updates}
        self.configs[agent_name] = AgentConfiguration(**updated_data)
    
    def save_to_file(self, filepath: str):
        """保存配置到文件"""
        data = {
            name: config.model_dump()
            for name, config in self.configs.items()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def load_from_file(self, filepath: str):
        """从文件加载配置"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for name, config_dict in data.items():
            self.configs[name] = AgentConfiguration(**config_dict)
    
    def get_all_configs(self) -> Dict[str, AgentConfiguration]:
        """获取所有配置"""
        return self.configs.copy()
    
    def reset_to_defaults(self):
        """重置为默认配置"""
        self._load_default_configs()


# ============================================================================
# 全局配置管理器（单例）
# ============================================================================

_global_config_manager: Optional[AgentConfigManager] = None


def get_config_manager() -> AgentConfigManager:
    """
    获取全局配置管理器（单例）
    
    Returns:
        AgentConfigManager 实例
    
    Example:
        >>> from src.agents.config import get_config_manager
        >>> 
        >>> manager = get_config_manager()
        >>> config = manager.get_config("bdl_mapper")
    """
    global _global_config_manager
    if _global_config_manager is None:
        _global_config_manager = AgentConfigManager()
    return _global_config_manager


# ============================================================================
# 便捷函数
# ============================================================================

def get_agent_prompt(agent_name: str) -> str:
    """
    获取 Agent 提示词
    
    Args:
        agent_name: Agent 名称
    
    Returns:
        提示词字符串
    """
    manager = get_config_manager()
    config = manager.get_config(agent_name)
    return config.system_prompt


def update_agent_prompt(agent_name: str, new_prompt: str):
    """
    更新 Agent 提示词
    
    Args:
        agent_name: Agent 名称
        new_prompt: 新提示词
    """
    manager = get_config_manager()
    manager.update_config(agent_name, {"system_prompt": new_prompt})


# ============================================================================
# 使用示例
# ============================================================================

"""
使用示例：

# 1. 获取配置
from src.agents.config import get_config_manager

manager = get_config_manager()
config = manager.get_config("bdl_mapper")

print(f"Temperature: {config.temperature}")
print(f"Provider: {config.provider}")
print(f"Prompt: {config.system_prompt}")

# 2. 更新配置
manager.update_config("bdl_mapper", {
    "temperature": 0.5,
    "max_iterations": 15,
})

# 3. 保存配置
manager.save_to_file("configs/agents_custom.json")

# 4. 加载配置
manager = AgentConfigManager(config_file="configs/agents_custom.json")

# 5. 环境变量覆盖
# 在 .env 中设置：
# AGENT_BDL_MAPPER_TEMPERATURE=0.5
# AGENT_BDL_MAPPER_PROVIDER=litellm
# 
# 配置管理器会自动应用这些覆盖

# 6. 便捷函数
from src.agents.config import get_agent_prompt, update_agent_prompt

prompt = get_agent_prompt("code_generator")
update_agent_prompt("code_generator", "New prompt...")
"""
