"""
LLM Factory - 统一的模型获取接口
根据配置自动选择和创建不同的 LLM 提供商

LangGraph 1.0+ Best Practices:
1. 使用工厂模式创建 LLM
2. 支持配置驱动的模型选择
3. 统一的接口（兼容 LangChain ChatModel）
4. 支持 with_structured_output()
"""

from enum import Enum
import os
from typing import Any, Dict, Optional, Type

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel

from src.llm.config import LLM_CONFIG, get_provider_config, get_model_for_task


# ============================================================================
# 提供商枚举
# ============================================================================

class LLMProvider(str, Enum):
    """支持的 LLM 提供商"""
    LITELLM = "litellm"
    COPILOT = "copilot"  # [COMPANY_SPECIFIC]
    MOCK = "mock"


# ============================================================================
# LLM Factory
# ============================================================================

def get_llm(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    task: Optional[str] = None,
    **kwargs
) -> BaseChatModel:
    """
    获取 LLM 实例
    
    LangGraph 1.0+ 统一接口:
    - 所有返回的 LLM 都兼容 LangChain ChatModel 接口
    - 支持 ainvoke(), with_structured_output() 等方法
    
    Args:
        provider: 提供商名称（如果不指定，使用默认）
        model: 模型名称（如果不指定，使用提供商默认）
        task: 任务类型（parsing | analysis | generation | review）
        **kwargs: 额外参数（temperature, max_tokens 等）
    
    Returns:
        LangChain ChatModel 实例
    
    Examples:
        >>> # 使用默认配置
        >>> llm = get_llm()
        
        >>> # 使用 LiteLLM
        >>> llm = get_llm(provider="litellm")
        
        >>> # 为特定任务选择模型
        >>> llm = get_llm(task="parsing")  # 自动选择适合解析的模型
        
        >>> # 自定义参数
        >>> llm = get_llm(temperature=0.7, max_tokens=2048)
    """
    # Mock override for offline runs
    if os.getenv("MIGRATION_USE_MOCK_LLM") == "1":
        from src.llm.mock import MockLLM
        return MockLLM()

    # Determine provider
    if not provider:
        if task:
            model_str = get_model_for_task(task)
            if "/" in model_str:
                provider, model = model_str.split("/", 1)
            else:
                provider = LLM_CONFIG["default_provider"]
        else:
            provider = LLM_CONFIG["default_provider"]

    if provider == "mock":
        from src.llm.mock import MockLLM
        return MockLLM()

    config = get_provider_config(provider)
    models = config.get("models", {})
    if model and model in models:
        model = models[model]
    if not model:
        model = models.get("default", model)

    params = {**config.get("default_params", {}), **kwargs}

    if provider == "litellm":
        return _create_litellm_llm(model, config, params)

    if provider == "copilot":
        return _create_copilot_llm(model, config, params)

    raise ValueError(f"Unsupported LLM provider: {provider}")

def get_structured_llm(
    output_schema: Type[BaseModel],
    provider: Optional[str] = None,
    model: Optional[str] = None,
    **kwargs
) -> BaseChatModel:
    """
    获取带结构化输出的 LLM
    
    LangGraph 1.0+ Best Practice:
    - 使用 with_structured_output() 确保输出格式
    
    Args:
        output_schema: Pydantic 模型类
        provider: 提供商
        model: 模型名称
        **kwargs: 额外参数
    
    Returns:
        配置了结构化输出的 LLM
    
    Example:
        >>> class ReviewOutput(BaseModel):
        ...     score: int
        ...     issues: List[str]
        
        >>> llm = get_structured_llm(ReviewOutput)
        >>> result = await llm.ainvoke(messages)
        >>> result.score  # 类型安全
    """
    llm = get_llm(provider=provider, model=model, **kwargs)
    return llm.with_structured_output(output_schema)


# ============================================================================
# 提供商特定的创建函数
# ============================================================================

def _create_litellm_llm(
    model: str,
    config: Dict[str, Any],
    params: Dict[str, Any]
) -> BaseChatModel:
    """
    创建 LiteLLM
    
    LiteLLM 提供统一接口访问 100+ LLM 提供商
    文档: https://docs.litellm.ai/
    """
    # [PLACEHOLDER] 需要安装: pip install litellm
    
    try:
        from langchain_community.chat_models import ChatLiteLLM
        
        return ChatLiteLLM(
            model=model,
            api_base=config.get("api_base"),
            api_key=config.get("api_key"),
            **params
        )
    except ImportError:
        raise RuntimeError(
            "Missing dependency: litellm. Install it or switch DEFAULT_LLM_PROVIDER to copilot."
        )


def _create_copilot_llm(
    model: str,
    config: Dict[str, Any],
    params: Dict[str, Any]
) -> BaseChatModel:
    """
    创建公司 Copilot LLM
    
    [COMPANY_SPECIFIC] [PLACEHOLDER]
    需要根据公司实际的 Copilot 实现方式调整
    
    可能的实现方式:
    1. 如果是 Azure OpenAI: 使用 AzureChatOpenAI
    2. 如果是自定义 HTTP API: 继承 BaseChatModel 实现
    3. 如果是通过代理: 使用 OpenAI compatible API
    """
    from src.llm.providers.copilot import CopilotChatModel
    
    return CopilotChatModel(
        model=model,
        config=config,
        **params
    )


# ============================================================================
# 便捷函数 - 为不同任务创建 LLM
# ============================================================================

def get_parsing_llm(**kwargs) -> BaseChatModel:
    """获取用于解析任务的 LLM（通常用较小/快速的模型）"""
    return get_llm(task="parsing", **kwargs)


def get_analysis_llm(**kwargs) -> BaseChatModel:
    """获取用于分析任务的 LLM（通常用较强的模型）"""
    return get_llm(task="analysis", **kwargs)


def get_generation_llm(**kwargs) -> BaseChatModel:
    """获取用于代码生成任务的 LLM"""
    return get_llm(task="generation", **kwargs)


def get_review_llm(**kwargs) -> BaseChatModel:
    """获取用于审查任务的 LLM"""
    return get_llm(task="review", **kwargs)
