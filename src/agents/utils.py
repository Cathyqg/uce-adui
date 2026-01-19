"""
Agent 辅助工具模块
提供统一的 Agent 调用、结构化输出和错误处理

LangGraph 1.0+ Best Practices:
1. 统一的 Agent 调用接口
2. 结构化输出支持（避免手动解析 JSON）
3. 统一的错误处理和重试机制
4. 上下文注入辅助函数
"""
from typing import Any, Dict, Optional, Type, TypeVar
import json
import re
from pydantic import BaseModel

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent  # 使用新的调用方式
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


T = TypeVar('T', bound=BaseModel)


# ============================================================================
# 工具错误处理（避免工具异常导致整个 Agent 失败）
# ============================================================================

def _wrap_tools_with_error_handling(tools: list) -> list:
    """
    包装工具，添加错误处理
    
    当工具执行失败时，返回友好的错误信息给模型，
    让模型可以尝试其他方案，而不是整个 Agent 崩溃
    
    Args:
        tools: 原始工具列表
    
    Returns:
        包装后的工具列表
    """
    from functools import wraps
    from langchain_core.tools import StructuredTool
    
    wrapped = []
    for tool in tools:
        original_func = tool.func if hasattr(tool, 'func') else tool
        
        @wraps(original_func)
        async def error_handled_func(*args, **kwargs):
            try:
                # 尝试执行原始工具
                if hasattr(original_func, '__call__'):
                    result = original_func(*args, **kwargs)
                    # 如果是异步函数
                    if hasattr(result, '__await__'):
                        result = await result
                    return result
                return original_func(*args, **kwargs)
            except Exception as e:
                # 返回友好的错误信息给模型，让它可以继续推理
                error_msg = f"Tool execution failed: {type(e).__name__}: {str(e)}. Please try a different approach or tool."
                return error_msg
        
        # 保持工具的元数据
        if isinstance(tool, StructuredTool):
            wrapped_tool = StructuredTool(
                name=tool.name,
                description=tool.description,
                func=error_handled_func,
                args_schema=tool.args_schema if hasattr(tool, 'args_schema') else None,
            )
            wrapped.append(wrapped_tool)
        else:
            # 对于简单的函数工具，直接包装
            wrapped.append(error_handled_func)
    
    return wrapped


# ============================================================================
# 结构化输出 Agent 创建器（LangGraph 1.0+ 推荐方式）
# ============================================================================

def create_structured_agent(
    llm,
    tools: list,
    system_prompt: str,
    response_format: Optional[Type[BaseModel]] = None,
    max_iterations: int = 10,
):
    """
    创建带结构化输出的 Agent
    
    LangGraph 1.0+ 推荐方式:
    - 使用 langchain.agents.create_react_agent
    - 使用 system_prompt 而不是 state_modifier
    - 支持 response_format 实现结构化输出
    - 使用 with_structured_output 强制结构化（在支持原生结构化输出的模型上更稳定）
    
    Args:
        llm: LLM 实例
        tools: 工具列表
        system_prompt: 系统提示
        response_format: Pydantic 模型（可选，用于结构化输出）
        max_iterations: 最大迭代次数
    
    Returns:
        Agent 实例
    
    Example:
        >>> from pydantic import BaseModel
        >>> class MappingOutput(BaseModel):
        ...     component_name: str
        ...     confidence: float
        
        >>> agent = create_structured_agent(
        ...     llm=get_llm(),
        ...     tools=[search_tool, validate_tool],
        ...     system_prompt="You are an expert...",
        ...     response_format=MappingOutput,
        ... )
    """
    # Mock LLM: return placeholder output without tool calls.
    if getattr(llm, "is_mock", False):
        from langchain_core.messages import AIMessage
        from src.llm.mock import build_placeholder_model

        class MockAgent:
            def __init__(self, schema: Optional[Type[BaseModel]]):
                if schema:
                    self._response_format = schema

            async def ainvoke(self, input_dict: Dict[str, Any]):
                if getattr(self, "_response_format", None):
                    content = build_placeholder_model(self._response_format)
                else:
                    content = "{}"
                return {"messages": [AIMessage(content=content)]}

        return MockAgent(response_format)

    # 如果指定了 response_format，使用 with_structured_output 绑定到 LLM
    # 这种方式对支持原生结构化输出的 provider 更稳定，避免解析失败
    effective_llm = llm
    if response_format:
        # 使用 with_structured_output 强制结构化输出
        # 当模型支持时，这会使用 provider native structured output
        effective_llm = llm.with_structured_output(
            response_format,
            method="tool_calling",  # 明确使用工具调用方式，更通用
            include_raw=False,  # 只返回结构化结果
        )
    
    # 包装工具，添加错误处理（避免工具异常导致整个 agent 失败）
    wrapped_tools = _wrap_tools_with_error_handling(tools)
    
    agent = create_react_agent(
        effective_llm,
        wrapped_tools,
        system_prompt=system_prompt,
    )
    
    # 保存 response_format 供后续使用
    if response_format:
        agent._response_format = response_format
    
    return agent


# ============================================================================
# 统一的 Agent 调用接口
# ============================================================================

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=1, max=10),
    retry=retry_if_exception_type((ConnectionError, TimeoutError)),
)
async def invoke_agent_with_retry(
    agent,
    messages: list,
    response_format: Optional[Type[T]] = None,
) -> Dict[str, Any]:
    """
    带重试的 Agent 调用
    
    LangGraph 1.0+ 最佳实践:
    - 自动重试瞬时错误
    - 支持结构化输出解析（但建议在 create_structured_agent 时指定）
    - 统一的错误处理
    
    注意：如果 agent 创建时已指定 response_format，则无需再传递此参数
    
    Args:
        agent: Agent 实例
        messages: 消息列表
        response_format: 期望的输出格式（Pydantic 模型，可选，通常不需要）
    
    Returns:
        Agent 执行结果
    
    Example:
        >>> # 推荐方式：在创建 agent 时指定 response_format
        >>> agent = create_structured_agent(..., response_format=MappingOutput)
        >>> result = await invoke_agent_with_retry(agent, messages=[...])
        >>> structured_result = result.get("structured_response")
    """
    result = await agent.ainvoke({"messages": messages})
    
    # 如果 agent 使用了 with_structured_output，结果已经是结构化的
    if hasattr(agent, '_response_format'):
        # 检查最后一条消息是否是结构化输出
        final_message = result.get("messages", [])[-1]
        if hasattr(final_message, 'content'):
            # 如果 content 是 Pydantic 对象，直接使用
            if isinstance(final_message.content, BaseModel):
                result["structured_response"] = final_message.content
            else:
                # 否则尝试解析
                structured = parse_structured_response(
                    final_message.content,
                    agent._response_format
                )
                result["structured_response"] = structured
    # 如果调用时显式传了 response_format（不推荐，但兼容旧代码）
    elif response_format:
        final_message = result.get("messages", [])[-1]
        structured = parse_structured_response(
            final_message.content,
            response_format
        )
        result["structured_response"] = structured
    
    return result


# ============================================================================
# 结构化响应解析（改进的 JSON 提取）
# ============================================================================

def parse_structured_response(
    content: str,
    model_class: Type[T],
    fallback_on_error: bool = True,
) -> Optional[T]:
    """
    从 Agent 响应中解析结构化输出
    
    LangGraph 1.0+ 推荐升级路径:
    - 当前：手动解析 JSON
    - 未来：使用 response_format 参数直接返回结构化对象
    
    Args:
        content: Agent 响应内容
        model_class: Pydantic 模型类
        fallback_on_error: 解析失败时是否返回默认值
    
    Returns:
        解析后的 Pydantic 对象，或 None
    
    Example:
        >>> class Output(BaseModel):
        ...     score: int
        ...     issues: List[str]
        
        >>> result = parse_structured_response(
        ...     agent_response,
        ...     Output
        ... )
        >>> print(result.score)
    """
    # 方法 1: 尝试整体解析为 JSON
    try:
        data = json.loads(content)
        return model_class.model_validate(data)
    except (json.JSONDecodeError, ValueError):
        pass
    
    # 方法 2: 提取 JSON 代码块
    json_match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', content)
    if json_match:
        try:
            data = json.loads(json_match.group(1))
            return model_class.model_validate(data)
        except (json.JSONDecodeError, ValueError):
            pass
    
    # 方法 3: 提取第一个 JSON 对象
    json_match = re.search(r'\{[\s\S]*\}', content)
    if json_match:
        try:
            data = json.loads(json_match.group())
            return model_class.model_validate(data)
        except (json.JSONDecodeError, ValueError):
            pass
    
    # 解析失败
    if fallback_on_error:
        # 返回默认值
        try:
            return model_class()
        except:
            return None
    else:
        return None


# ============================================================================
# 上下文注入辅助函数
# ============================================================================

def inject_context_to_message(
    user_message: str,
    context_data: Dict[str, Any],
    max_length: int = 3000,
) -> str:
    """
    将 state 中的上下文数据注入到用户消息中
    
    LangGraph 1.0+ 注意事项:
    - state 中的字段不会自动进入模型上下文
    - 需要显式包含在消息中
    - 或使用 middleware 动态注入
    
    Args:
        user_message: 原始用户消息
        context_data: 需要注入的上下文字典
        max_length: 每个上下文项的最大长度
    
    Returns:
        包含上下文的完整消息
    
    Example:
        >>> msg = inject_context_to_message(
        ...     "Map this component",
        ...     {"bdl_spec": {...}, "history": [...]},
        ... )
    """
    context_parts = []
    
    for key, value in context_data.items():
        if value is None:
            continue
        
        # 格式化上下文
        if isinstance(value, (dict, list)):
            formatted = json.dumps(value, indent=2)[:max_length]
            if len(json.dumps(value)) > max_length:
                formatted += "\n... (truncated)"
        else:
            formatted = str(value)[:max_length]
        
        context_parts.append(f"**{key.replace('_', ' ').title()}:**\n{formatted}")
    
    if context_parts:
        context_section = "\n\n".join(context_parts)
        return f"{user_message}\n\n---\n\n{context_section}"
    else:
        return user_message


# ============================================================================
# 错误处理辅助函数
# ============================================================================

def create_error_result(
    error: Exception,
    component_id: str,
    agent_name: str,
) -> Dict[str, Any]:
    """
    创建统一的错误结果
    
    Args:
        error: 异常对象
        component_id: 组件 ID
        agent_name: Agent 名称
    
    Returns:
        错误结果字典
    """
    return {
        "success": False,
        "error": {
            "type": type(error).__name__,
            "message": str(error),
            "component_id": component_id,
            "agent": agent_name,
        }
    }


# ============================================================================
# 向后兼容的解析函数（过渡期使用）
# ============================================================================

def parse_json_from_content(content: str, fallback: Optional[Dict] = None) -> Dict:
    """
    从文本中提取 JSON（向后兼容函数）
    
    注意：这是过渡期函数，最终应该用 response_format 替代
    
    Args:
        content: 文本内容
        fallback: 解析失败时的默认值
    
    Returns:
        解析后的字典
    """
    # 尝试直接解析
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    
    # 尝试提取 JSON 块
    json_match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', content)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass
    
    # 尝试提取第一个 JSON 对象
    json_match = re.search(r'\{[\s\S]*\}', content)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
    
    # 返回 fallback
    return fallback if fallback is not None else {}


# ============================================================================
# 使用示例
# ============================================================================

"""
使用示例：

# 1. 创建结构化 Agent
from pydantic import BaseModel
from src.llm import get_llm

class MappingOutput(BaseModel):
    bdl_component_name: str
    confidence_score: float
    prop_mappings: List[Dict]

llm = get_llm(task="analysis")
agent = create_structured_agent(
    llm,
    tools=[search_tool, spec_tool],
    system_prompt="You are an expert...",
    response_format=MappingOutput,  # 指定输出格式
)

# 2. 调用 Agent
result = await invoke_agent_with_retry(
    agent,
    messages=[HumanMessage(content="...")],
    response_format=MappingOutput,
)

# 3. 获取结构化结果
mapping: MappingOutput = result["structured_response"]
print(mapping.bdl_component_name)  # 类型安全！

# 4. 注入上下文
full_message = inject_context_to_message(
    "Map this component",
    {
        "bdl_spec": state.get("bdl_spec"),
        "analyzed_component": comp_data.get("analyzed"),
    }
)
"""
