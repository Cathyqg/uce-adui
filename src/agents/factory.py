"""
Agent 工厂模式
统一创建和管理所有 Agent 实例

设计模式:
1. 工厂模式 - 统一创建 Agent
2. 单例模式 - 缓存 Agent 实例
3. 策略模式 - 灵活配置 Agent 行为
4. 建造者模式 - 构建复杂 Agent 配置

LangGraph 1.0+ Best Practices:
- 统一的 Agent 创建接口
- 集中的配置管理
- 类型安全的 Agent registry
- 支持动态切换实现
"""
from typing import Any, Dict, Optional, Type
from enum import Enum
from pydantic import BaseModel

from src.llm import get_llm
from src.agents.utils import create_structured_agent


# ============================================================================
# Agent 类型枚举
# ============================================================================

class AgentType(str, Enum):
    """支持的 Agent 类型"""
    BDL_MAPPER = "bdl_mapper"
    CODE_GENERATOR = "code_generator"
    CODE_REVIEWER = "code_reviewer"
    EDITOR_DESIGNER = "editor_designer"


# ============================================================================
# Agent 配置
# ============================================================================

class AgentConfig(BaseModel):
    """Agent 配置"""
    task_type: str  # parsing, analysis, generation, review
    temperature: float = 0.0
    provider: Optional[str] = None
    model: Optional[str] = None
    max_iterations: int = 10
    response_format: Optional[Type[BaseModel]] = None


# 默认配置
AGENT_CONFIGS = {
    AgentType.BDL_MAPPER: AgentConfig(
        task_type="analysis",
        temperature=0.1,  # 稍高温度增加创造性
        provider="litellm",
    ),
    AgentType.CODE_GENERATOR: AgentConfig(
        task_type="generation",
        temperature=0.0,  # 确定性输出
        max_iterations=15,  # 生成+验证可能需要更多迭代
    ),
    AgentType.CODE_REVIEWER: AgentConfig(
        task_type="review",
        temperature=0.0,  # 严格审查
        provider="litellm",
    ),
    AgentType.EDITOR_DESIGNER: AgentConfig(
        task_type="analysis",
        temperature=0.3,  # UX 设计需要创造性
    ),
}


# ============================================================================
# Agent 工厂类
# ============================================================================

class AgentFactory:
    """
    Agent 工厂
    
    职责:
    1. 根据类型创建 Agent
    2. 缓存 Agent 实例（单例模式）
    3. 提供统一的配置接口
    4. 支持自定义配置
    
    Example:
        >>> factory = AgentFactory()
        >>> agent = factory.create_agent(AgentType.BDL_MAPPER)
        >>> 
        >>> # 自定义配置
        >>> agent = factory.create_agent(
        ...     AgentType.CODE_GENERATOR,
        ...     config=AgentConfig(temperature=0.5, max_iterations=20)
        ... )
    """
    
    def __init__(self):
        """初始化工厂"""
        self._agent_cache: Dict[str, Any] = {}
        self._config_overrides: Dict[AgentType, AgentConfig] = {}
    
    def create_agent(
        self,
        agent_type: AgentType,
        config: Optional[AgentConfig] = None,
        force_recreate: bool = False,
    ):
        """
        创建或获取 Agent 实例
        
        Args:
            agent_type: Agent 类型
            config: 自定义配置（可选）
            force_recreate: 强制重新创建（忽略缓存）
        
        Returns:
            Agent 实例
        """
        # 检查缓存
        cache_key = str(agent_type)
        if not force_recreate and cache_key in self._agent_cache:
            return self._agent_cache[cache_key]
        
        # 获取配置
        agent_config = config or self._config_overrides.get(agent_type) or AGENT_CONFIGS[agent_type]
        
        # 创建 Agent
        agent = self._create_agent_impl(agent_type, agent_config)
        
        # 缓存
        if not force_recreate:
            self._agent_cache[cache_key] = agent
        
        return agent
    
    def _create_agent_impl(self, agent_type: AgentType, config: AgentConfig):
        """实际创建 Agent 的实现"""
        # 创建 LLM
        llm = get_llm(
            task=config.task_type,
            provider=config.provider,
            model=config.model,
            temperature=config.temperature,
        )
        
        # 根据类型创建不同的 Agent
        if agent_type == AgentType.BDL_MAPPER:
            return self._create_bdl_mapper_agent(llm, config)
        elif agent_type == AgentType.CODE_GENERATOR:
            return self._create_code_generator_agent(llm, config)
        elif agent_type == AgentType.CODE_REVIEWER:
            return self._create_code_reviewer_agent(llm, config)
        elif agent_type == AgentType.EDITOR_DESIGNER:
            return self._create_editor_designer_agent(llm, config)
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
    
    def _create_bdl_mapper_agent(self, llm, config: AgentConfig):
        """创建 BDL Mapper Agent"""
        from src.agents.core import create_bdl_mapping_agent
        # 注意：这里实际上会调用原函数，只是通过工厂统一管理
        return create_bdl_mapping_agent(llm=llm)
    
    def _create_code_generator_agent(self, llm, config: AgentConfig):
        """创建 Code Generator Agent"""
        from src.agents.core import create_code_generator_agent
        return create_code_generator_agent(llm=llm)
    
    def _create_code_reviewer_agent(self, llm, config: AgentConfig):
        """创建 Code Reviewer Agent"""
        from src.agents.core import create_code_reviewer_agent
        return create_code_reviewer_agent(llm=llm)
    
    def _create_editor_designer_agent(self, llm, config: AgentConfig):
        """创建 Editor Designer Agent"""
        from src.agents.core import create_editor_designer_agent
        return create_editor_designer_agent(llm=llm)
    
    def set_config_override(self, agent_type: AgentType, config: AgentConfig):
        """
        设置配置覆盖
        
        Args:
            agent_type: Agent 类型
            config: 新配置
        
        Example:
            >>> factory = AgentFactory()
            >>> factory.set_config_override(
            ...     AgentType.CODE_GENERATOR,
            ...     AgentConfig(temperature=0.5, max_iterations=20)
            ... )
        """
        self._config_overrides[agent_type] = config
        # 清除缓存以使用新配置
        cache_key = str(agent_type)
        if cache_key in self._agent_cache:
            del self._agent_cache[cache_key]
    
    def clear_cache(self):
        """清除所有缓存的 Agent"""
        self._agent_cache.clear()
    
    def get_agent_info(self, agent_type: AgentType) -> Dict[str, Any]:
        """
        获取 Agent 信息
        
        Args:
            agent_type: Agent 类型
        
        Returns:
            Agent 配置和状态信息
        """
        config = self._config_overrides.get(agent_type) or AGENT_CONFIGS[agent_type]
        cache_key = str(agent_type)
        
        return {
            "agent_type": agent_type,
            "config": config.model_dump(),
            "cached": cache_key in self._agent_cache,
            "task_type": config.task_type,
            "temperature": config.temperature,
        }


# ============================================================================
# 全局工厂实例（单例模式）
# ============================================================================

_global_factory: Optional[AgentFactory] = None


def get_agent_factory() -> AgentFactory:
    """
    获取全局 Agent 工厂实例（单例）
    
    Returns:
        AgentFactory 实例
    
    Example:
        >>> from src.agents.factory import get_agent_factory, AgentType
        >>> 
        >>> factory = get_agent_factory()
        >>> agent = factory.create_agent(AgentType.BDL_MAPPER)
    """
    global _global_factory
    if _global_factory is None:
        _global_factory = AgentFactory()
    return _global_factory


# ============================================================================
# 便捷函数
# ============================================================================

def create_agent(
    agent_type: AgentType,
    config: Optional[AgentConfig] = None,
):
    """
    便捷函数：创建 Agent
    
    Args:
        agent_type: Agent 类型
        config: 自定义配置
    
    Returns:
        Agent 实例
    
    Example:
        >>> from src.agents.factory import create_agent, AgentType
        >>> 
        >>> agent = create_agent(AgentType.BDL_MAPPER)
    """
    factory = get_agent_factory()
    return factory.create_agent(agent_type, config)


def get_agent_config(agent_type: AgentType) -> AgentConfig:
    """
    获取 Agent 配置
    
    Args:
        agent_type: Agent 类型
    
    Returns:
        Agent 配置
    """
    return AGENT_CONFIGS[agent_type]


# ============================================================================
# Builder 模式 - 构建复杂 Agent
# ============================================================================

class AgentBuilder:
    """
    Agent 建造者
    
    用于构建具有复杂配置的 Agent
    
    Example:
        >>> from src.agents.factory import AgentBuilder, AgentType
        >>> from src.agents.middleware import create_context_injector, create_monitor
        >>> 
        >>> agent = (AgentBuilder(AgentType.BDL_MAPPER)
        ...     .with_temperature(0.2)
        ...     .with_max_iterations(15)
        ...     .with_middleware(create_context_injector(["bdl_spec"]))
        ...     .with_middleware(create_monitor(log_timing=True))
        ...     .build()
        ... )
    """
    
    def __init__(self, agent_type: AgentType):
        """初始化建造者"""
        self.agent_type = agent_type
        self.config = AGENT_CONFIGS[agent_type].model_copy()
        self.middlewares = []
    
    def with_temperature(self, temperature: float):
        """设置温度"""
        self.config.temperature = temperature
        return self
    
    def with_max_iterations(self, max_iterations: int):
        """设置最大迭代次数"""
        self.config.max_iterations = max_iterations
        return self
    
    def with_provider(self, provider: str):
        """设置提供商"""
        self.config.provider = provider
        return self
    
    def with_model(self, model: str):
        """设置模型"""
        self.config.model = model
        return self
    
    def with_response_format(self, response_format: Type[BaseModel]):
        """设置响应格式"""
        self.config.response_format = response_format
        return self
    
    def with_middleware(self, middleware):
        """添加 middleware"""
        self.middlewares.append(middleware)
        return self
    
    def build(self):
        """构建 Agent"""
        factory = get_agent_factory()
        agent = factory.create_agent(self.agent_type, self.config)
        
        # 应用 middleware
        if self.middlewares:
            from src.agents.middleware import compose_middlewares
            agent = compose_middlewares(*self.middlewares) | agent
        
        return agent


# ============================================================================
# 使用示例
# ============================================================================

"""
使用示例：

# 1. 简单使用（工厂模式）
from src.agents.factory import create_agent, AgentType

agent = create_agent(AgentType.BDL_MAPPER)

# 2. 自定义配置
from src.agents.factory import create_agent, AgentType, AgentConfig

custom_config = AgentConfig(
    task_type="analysis",
    temperature=0.5,
    provider="litellm",
    max_iterations=20,
)

agent = create_agent(AgentType.BDL_MAPPER, config=custom_config)

# 3. 使用 Builder 模式（复杂配置）
from src.agents.factory import AgentBuilder, AgentType
from src.agents.middleware import create_context_injector, create_monitor

agent = (AgentBuilder(AgentType.BDL_MAPPER)
    .with_temperature(0.2)
    .with_max_iterations(15)
    .with_middleware(create_context_injector(["bdl_spec"]))
    .with_middleware(create_monitor(log_timing=True))
    .build()
)

# 4. 使用全局工厂（单例）
from src.agents.factory import get_agent_factory, AgentType

factory = get_agent_factory()

# 获取 Agent（会缓存）
agent1 = factory.create_agent(AgentType.BDL_MAPPER)
agent2 = factory.create_agent(AgentType.BDL_MAPPER)  # 返回相同实例

# 强制重新创建
agent3 = factory.create_agent(AgentType.BDL_MAPPER, force_recreate=True)

# 5. 配置覆盖
factory = get_agent_factory()
factory.set_config_override(
    AgentType.CODE_GENERATOR,
    AgentConfig(temperature=0.5, max_iterations=20)
)

# 之后创建的 CODE_GENERATOR 都会使用新配置
agent = factory.create_agent(AgentType.CODE_GENERATOR)
"""
