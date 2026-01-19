"""
Agent 基础设施模块
提供 Agent 创建、调用、管理的完整工具集

架构说明:
- agents/ 现在是纯粹的"技术基础设施层"
- 节点实现已移动到 src/nodes/intelligent/
- 这样职责更清晰：nodes/ = 业务逻辑，agents/ = 技术工具

模块说明:
├── core.py               - Agent 创建函数和输出模型 ⭐
├── utils.py              - 统一工具（调用、解析、重试）
├── middleware.py         - Middleware 模式
├── factory.py            - 工厂模式
├── strategies.py         - 调用策略模式
└── config.py             - 配置管理

设计模式:
- 工厂模式: 统一创建 Agent
- 策略模式: 灵活的调用策略
- 中间件模式: 横切关注点
- 单例模式: 全局实例管理
- 建造者模式: 复杂配置构建
"""

# Agent 创建函数（从 core.py）
from src.agents.core import (
    create_bdl_mapping_agent,
    create_code_generator_agent,
    create_code_reviewer_agent,
    create_editor_designer_agent,
    # 输出模型
    BDLMappingOutput,
    CodeGeneratorOutput,
    CodeReviewOutput,
    EditorDesignOutput,
)

# 辅助工具 - 统一的 Agent 调用和解析
from src.agents.utils import (
    create_structured_agent,
    invoke_agent_with_retry,
    parse_structured_response,
    inject_context_to_message,
    create_error_result,
    parse_json_from_content,
)

# Middleware - 动态提示和横切关注点
from src.agents.middleware import (
    create_context_injector,
    create_error_handler,
    create_response_parser,
    create_monitor,
    compose_middlewares,
)

# 工厂模式 - 统一创建和管理 Agent
from src.agents.factory import (
    AgentFactory,
    AgentType,
    AgentConfig,
    AgentBuilder,
    get_agent_factory,
    create_agent,
    get_agent_config,
)

# 策略模式 - 灵活的调用策略
from src.agents.strategies import (
    AgentInvocationStrategy,
    SimpleInvocationStrategy,
    RetryInvocationStrategy,
    IterativeImprovementStrategy,
    CascadeInvocationStrategy,
    VotingInvocationStrategy,
    StrategyFactory,
)

# 配置管理 - 集中的配置和提示词管理
from src.agents.config import (
    AgentConfiguration,
    AgentConfigManager,
    PromptTemplates,
    get_config_manager,
    get_agent_prompt,
    update_agent_prompt,
)


__all__ = [
    # ========== Agent 创建和模型 ==========
    "create_bdl_mapping_agent",
    "create_code_generator_agent",
    "create_code_reviewer_agent",
    "create_editor_designer_agent",
    "BDLMappingOutput",
    "CodeGeneratorOutput",
    "CodeReviewOutput",
    "EditorDesignOutput",
    
    # ========== 工具函数 ==========
    "create_structured_agent",
    "invoke_agent_with_retry",
    "parse_structured_response",
    "inject_context_to_message",
    "create_error_result",
    "parse_json_from_content",
    
    # ========== Middleware ==========
    "create_context_injector",
    "create_error_handler",
    "create_response_parser",
    "create_monitor",
    "compose_middlewares",
    
    # ========== 工厂模式 ==========
    "AgentFactory",
    "AgentType",
    "AgentConfig",
    "AgentBuilder",
    "get_agent_factory",
    "create_agent",
    "get_agent_config",
    
    # ========== 策略模式 ==========
    "AgentInvocationStrategy",
    "SimpleInvocationStrategy",
    "RetryInvocationStrategy",
    "IterativeImprovementStrategy",
    "CascadeInvocationStrategy",
    "VotingInvocationStrategy",
    "StrategyFactory",
    
    # ========== 配置管理 ==========
    "AgentConfiguration",
    "AgentConfigManager",
    "PromptTemplates",
    "get_config_manager",
    "get_agent_prompt",
    "update_agent_prompt",
]

