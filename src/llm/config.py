"""
LLM 提供商配置
集中管理所有模型提供商的配置

================================================================================
⚠️ 需要定制的部分标记说明:
================================================================================
- [CUSTOMIZE]    = 需要根据实际项目定制
- [PLACEHOLDER]  = 占位符，需要完整实现
- [COMPANY_SPECIFIC] = 公司特定的配置，需要根据实际情况填写
================================================================================
"""

import os
from typing import Any, Dict, Optional


# ============================================================================
# 全局 LLM 配置
# ============================================================================

LLM_CONFIG = {
    # [CUSTOMIZE] 默认提供商
    "default_provider": "litellm",  # litellm | copilot | mock
    
    # [CUSTOMIZE] 不同任务使用不同模型
    "task_models": {
        "parsing": "litellm/default",
        "analysis": "litellm/default",
        "generation": "litellm/default",
        "review": "litellm/default",
    },
    
    # 全局参数
    "temperature": 0,        # 确定性输出
    "max_tokens": 4096,      # 最大输出 tokens
    "timeout": 60,           # 请求超时（秒）
}


# ============================================================================

LITELLM_CONFIG = {
    "provider": "litellm",
    
    # [CUSTOMIZE] LiteLLM 支持的模型前缀
    # 格式: provider/model-name
    "models": {
        "default": "company-default",
        "fast": "company-fast",
        "smart": "company-smart",
    },
    
    # [CUSTOMIZE] LiteLLM 环境变量（如果使用）
    # 参考: https://docs.litellm.ai/docs/
    "api_base": os.getenv("LITELLM_API_BASE"),
    "api_key": os.getenv("LITELLM_API_KEY"),
    
    "default_params": {
        "temperature": 0,
        "max_tokens": 4096,
    }
}


# ============================================================================
# Mock 配置（离线测试）
# ============================================================================

MOCK_CONFIG = {
    "provider": "mock",
    "models": {
        "default": "mock",
    },
    "default_params": {},
}


# ============================================================================
# 公司 Copilot 配置
# ============================================================================

# [COMPANY_SPECIFIC] [PLACEHOLDER]
# 这部分需要根据公司实际的 Copilot 调用方式实现
COPILOT_CONFIG = {
    "provider": "copilot",
    
    # [COMPANY_SPECIFIC] Copilot API 配置
    # 不同公司的实现方式可能完全不同，以下是常见的可能性:
    
    # 可能性 1: HTTP API 调用
    "api_endpoint": os.getenv("COPILOT_API_ENDPOINT"),  # [CUSTOMIZE] 如 https://copilot.company.com/api
    "api_key": os.getenv("COPILOT_API_KEY"),
    "api_version": os.getenv("COPILOT_API_VERSION", "v1"),
    
    # 可能性 2: 通过 Azure OpenAI
    "azure_endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
    "azure_deployment": os.getenv("AZURE_DEPLOYMENT_NAME"),
    
    # 可能性 3: 通过内部代理
    "proxy_url": os.getenv("COMPANY_LLM_PROXY"),
    
    # [COMPANY_SPECIFIC] 认证方式
    "auth_type": "bearer",  # [CUSTOMIZE] bearer | api_key | oauth | custom
    
    # [COMPANY_SPECIFIC] 请求格式
    "request_format": "compatible",  # [CUSTOMIZE] compatible | custom
    
    # [COMPANY_SPECIFIC] 模型映射
    "models": {
        "default": "copilot-gpt4",  # [CUSTOMIZE] 公司内部模型名称
        "fast": "copilot-gpt35",
        "smart": "copilot-gpt4-turbo",
    },
    
    # [COMPANY_SPECIFIC] 额外的请求头
    "headers": {
        "X-Company-ID": os.getenv("COMPANY_ID"),  # [CUSTOMIZE]
        "X-Project-ID": os.getenv("PROJECT_ID"),   # [CUSTOMIZE]
        # 其他公司特定的 headers
    },
    
    # [COMPANY_SPECIFIC] 额外的请求参数
    "extra_params": {
        # 可能需要的公司特定参数
        # "tenant_id": "...",
        # "cost_center": "...",
        # "compliance_level": "high",
    },
    
    "default_params": {
        "temperature": 0,
        "max_tokens": 4096,
    }
}


# ============================================================================
# 提供商注册表
# ============================================================================

PROVIDER_CONFIGS = {
    "litellm": LITELLM_CONFIG,
    "copilot": COPILOT_CONFIG,
    "mock": MOCK_CONFIG,
}


# ============================================================================
# 辅助函数
# ============================================================================

def get_provider_config(provider: str) -> Dict[str, Any]:
    """
    获取指定提供商的配置
    
    Args:
        provider: 提供商名称
    
    Returns:
        提供商配置
    """
    return PROVIDER_CONFIGS.get(provider, LITELLM_CONFIG)


def update_llm_config(updates: Dict[str, Any]) -> None:
    """
    运行时更新 LLM 配置
    
    Args:
        updates: 配置更新
    """
    LLM_CONFIG.update(updates)


def get_model_for_task(task: str) -> str:
    """
    根据任务类型获取推荐模型
    
    Args:
        task: 任务类型 (parsing | analysis | generation | review)
    
    Returns:
        模型标识符
    """
    return LLM_CONFIG["task_models"].get(task, "litellm/default")


# ============================================================================
# 环境变量配置（自动加载）
# ============================================================================

# 从环境变量覆盖默认配置
if os.getenv("DEFAULT_LLM_PROVIDER"):
    LLM_CONFIG["default_provider"] = os.getenv("DEFAULT_LLM_PROVIDER")

if os.getenv("LLM_TEMPERATURE"):
    LLM_CONFIG["temperature"] = float(os.getenv("LLM_TEMPERATURE"))
