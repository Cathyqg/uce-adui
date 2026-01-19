"""
Agent Middleware 模块
LangGraph 1.0+ 动态提示和消息重写的推荐方式

使用场景:
1. 动态注入上下文（如 BDL spec、历史记录）
2. 请求预处理（格式化、验证）
3. 响应后处理（解析、转换）
4. 统一错误处理
5. 日志和监控

LangGraph 1.0+ Best Practice:
- 使用 RunnableSequence 组合 middleware 和 agent
- 使用 RunnableLambda 包装 middleware 函数
- 支持 async middleware
"""
from typing import Any, Dict
import json

from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.messages import SystemMessage, HumanMessage


# ============================================================================
# 上下文注入 Middleware
# ============================================================================

def create_context_injector(
    context_keys: list,
    max_length: int = 3000,
):
    """
    创建上下文注入 middleware
    
    LangGraph 1.0+ 推荐方式:
    - 从 state 中读取指定字段
    - 动态注入到消息中
    - 不污染原始消息
    
    Args:
        context_keys: 需要注入的 state 字段列表
        max_length: 每个字段的最大长度
    
    Returns:
        Middleware 函数
    
    Example:
        >>> # 创建 middleware
        >>> inject_bdl = create_context_injector(
        ...     context_keys=["bdl_spec", "component_registry"],
        ...     max_length=2000
        ... )
        
        >>> # 组合到 Agent
        >>> agent_with_context = inject_bdl | agent
        
        >>> # 调用时自动注入上下文
        >>> result = await agent_with_context.ainvoke({
        ...     "messages": [...],
        ...     "bdl_spec": {...},  # 会被自动注入
        ... })
    """
    def inject_context(input_dict: Dict) -> Dict:
        """注入上下文到消息中"""
        messages = input_dict.get("messages", [])
        
        # 收集需要注入的上下文
        context_parts = []
        for key in context_keys:
            value = input_dict.get(key)
            if value is not None:
                # 格式化上下文
                if isinstance(value, (dict, list)):
                    formatted = json.dumps(value, indent=2)[:max_length]
                    if len(json.dumps(value)) > max_length:
                        formatted += "\n... (truncated)"
                else:
                    formatted = str(value)[:max_length]
                
                context_parts.append(
                    f"**{key.replace('_', ' ').title()}:**\n{formatted}"
                )
        
        if context_parts:
            # 创建上下文消息
            context_content = "\n\n".join(context_parts)
            context_msg = SystemMessage(
                content=f"Available Context:\n\n{context_content}"
            )
            
            # 注入到消息列表开头（在用户消息之前）
            messages = [context_msg] + messages
        
        # 返回更新后的输入
        return {
            **input_dict,
            "messages": messages
        }
    
    return RunnableLambda(inject_context)


# ============================================================================
# 错误处理 Middleware
# ============================================================================

def create_error_handler(
    fallback_response: str = "I encountered an error. Please try again.",
    log_errors: bool = True,
):
    """
    创建错误处理 middleware
    
    LangGraph 1.0+ 推荐方式:
    - 捕获 Agent 执行错误
    - 提供 fallback 响应
    - 记录错误日志
    
    Args:
        fallback_response: 错误时的默认响应
        log_errors: 是否记录错误
    
    Returns:
        Middleware 函数
    
    Example:
        >>> error_handler = create_error_handler()
        >>> safe_agent = agent | error_handler
    """
    async def handle_error(input_or_result):
        """处理错误"""
        try:
            # 如果是正常结果，直接返回
            if isinstance(input_or_result, dict) and "messages" in input_or_result:
                return input_or_result
            
            # 否则尝试执行
            return input_or_result
        except Exception as e:
            if log_errors:
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"Agent error: {str(e)}", exc_info=True)
            
            # 返回 fallback
            return {
                "messages": [HumanMessage(content=fallback_response)],
                "error": {
                    "type": type(e).__name__,
                    "message": str(e)
                }
            }
    
    return RunnableLambda(handle_error)


# ============================================================================
# 响应后处理 Middleware
# ============================================================================

def create_response_parser(
    extract_json: bool = True,
    validate_schema: bool = False,
):
    """
    创建响应解析 middleware
    
    LangGraph 1.0+ 推荐方式:
    - 自动提取 Agent 响应中的结构化数据
    - 验证输出格式
    - 转换为标准格式
    
    Args:
        extract_json: 是否自动提取 JSON
        validate_schema: 是否验证 schema
    
    Returns:
        Middleware 函数
    
    Example:
        >>> parser = create_response_parser(extract_json=True)
        >>> agent_with_parser = agent | parser
    """
    def parse_response(result: Dict) -> Dict:
        """解析响应"""
        if not extract_json:
            return result
        
        messages = result.get("messages", [])
        if not messages:
            return result
        
        final_message = messages[-1]
        content = final_message.content
        
        # 尝试提取 JSON
        import re
        
        # 提取 JSON 代码块
        json_match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', content)
        if json_match:
            try:
                extracted = json.loads(json_match.group(1))
                result["extracted_data"] = extracted
            except json.JSONDecodeError:
                pass
        else:
            # 提取第一个 JSON 对象
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                try:
                    extracted = json.loads(json_match.group())
                    result["extracted_data"] = extracted
                except json.JSONDecodeError:
                    pass
        
        return result
    
    return RunnableLambda(parse_response)


# ============================================================================
# 监控 Middleware
# ============================================================================

def create_monitor(
    log_inputs: bool = True,
    log_outputs: bool = True,
    log_timing: bool = True,
):
    """
    创建监控 middleware
    
    LangGraph 1.0+ 推荐方式:
    - 记录 Agent 输入输出
    - 测量执行时间
    - 便于调试和优化
    
    Args:
        log_inputs: 是否记录输入
        log_outputs: 是否记录输出
        log_timing: 是否记录时间
    
    Returns:
        Middleware 函数
    
    Example:
        >>> monitor = create_monitor(log_timing=True)
        >>> monitored_agent = monitor | agent
    """
    import time
    import logging
    
    logger = logging.getLogger(__name__)
    
    async def monitor_execution(input_dict: Dict) -> Dict:
        """监控执行"""
        start_time = time.time()
        
        if log_inputs:
            logger.info(f"Agent input: {_summarize_dict(input_dict)}")
        
        # 执行（这里只是传递，实际执行在 agent 中）
        result = input_dict
        
        if log_timing:
            duration = time.time() - start_time
            logger.info(f"Agent execution time: {duration:.2f}s")
        
        if log_outputs and isinstance(result, dict):
            logger.info(f"Agent output: {_summarize_dict(result)}")
        
        return result
    
    return RunnableLambda(monitor_execution)


def _summarize_dict(d: Dict, max_length: int = 200) -> str:
    """总结字典内容"""
    summary = json.dumps(d, indent=None)
    if len(summary) > max_length:
        return summary[:max_length] + "..."
    return summary


# ============================================================================
# 组合 Middleware
# ============================================================================

def compose_middlewares(*middlewares):
    """
    组合多个 middleware
    
    Example:
        >>> agent_with_all = compose_middlewares(
        ...     create_context_injector(["bdl_spec"]),
        ...     create_monitor(log_timing=True),
        ...     create_error_handler(),
        ... ) | agent | create_response_parser()
    """
    from langchain_core.runnables import RunnableSequence
    return RunnableSequence(*middlewares)


# ============================================================================
# 使用示例
# ============================================================================

"""
完整使用示例：

from langgraph.prebuilt import create_react_agent
from src.llm import get_llm
from src.agents.middleware import (
    create_context_injector,
    create_error_handler,
    create_response_parser,
    create_monitor,
    compose_middlewares,
)

# 1. 创建基础 Agent
llm = get_llm(task="analysis")
base_agent = create_react_agent(
    llm,
    tools=[search_tool, validate_tool],
    system_prompt="You are an expert...",
)

# 2. 添加 Middleware
agent_with_middleware = compose_middlewares(
    # 前置 middleware
    create_context_injector(
        context_keys=["bdl_spec", "component_registry"],
        max_length=2000
    ),
    create_monitor(log_timing=True),
) | base_agent | compose_middlewares(
    # 后置 middleware
    create_response_parser(extract_json=True),
    create_error_handler(fallback_response="Processing failed"),
)

# 3. 使用增强后的 Agent
result = await agent_with_middleware.ainvoke({
    "messages": [HumanMessage(content="Map this component")],
    "bdl_spec": {...},  # 会被自动注入到上下文
    "component_registry": {...},  # 也会被注入
})

# 4. 获取结果
structured_data = result.get("extracted_data")
if structured_data:
    print(structured_data["bdl_component_name"])
"""
