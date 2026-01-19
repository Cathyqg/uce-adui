"""
LLM 提供商抽象层
支持多种模型提供商的统一接口

支持的提供商:
- LiteLLM (统一接口)
- 自定义 Copilot 接口

LangGraph 1.0+ Best Practices:
1. 使用统一的接口抽象
2. 支持 with_structured_output()
3. 支持 streaming
4. 配置驱动的模型选择
"""

from src.llm.factory import (
    get_llm,
    get_structured_llm,
    LLMProvider,
)

from src.llm.config import (
    LLM_CONFIG,
    update_llm_config,
    get_provider_config,
)

from src.llm.providers import (
    LiteLLMProvider,
    CopilotChatModel,
)
from src.llm.mock import MockLLM


__all__ = [
    # Factory functions
    "get_llm",
    "get_structured_llm",
    "LLMProvider",
    
    # Configuration
    "LLM_CONFIG",
    "update_llm_config",
    "get_provider_config",
    
    # Providers
    "LiteLLMProvider",
    "CopilotChatModel",
    "MockLLM",
]
