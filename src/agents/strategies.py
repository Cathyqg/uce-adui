"""
Agent 调用策略模式
定义不同场景下的 Agent 调用策略

设计模式:
- 策略模式: 封装不同的调用策略
- 模板方法模式: 定义调用流程模板
- 责任链模式: 多个 Agent 串联调用

使用场景:
1. 单次调用策略（简单任务）
2. 重试策略（需要容错）
3. 迭代策略（需要多次改进）
4. 级联策略（多个 Agent 协作）
5. 投票策略（多个 Agent 投票决策）
"""
from typing import Any, Dict, List, Optional, Callable
from abc import ABC, abstractmethod
from pydantic import BaseModel

from langchain_core.messages import HumanMessage
from src.agents.utils import invoke_agent_with_retry


# ============================================================================
# 策略基类
# ============================================================================

class AgentInvocationStrategy(ABC):
    """Agent 调用策略基类"""
    
    @abstractmethod
    async def invoke(
        self,
        agent,
        messages: List,
        response_format: Optional[type] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        调用 Agent
        
        Args:
            agent: Agent 实例
            messages: 消息列表
            response_format: 响应格式（Pydantic 模型）
            **kwargs: 额外参数
        
        Returns:
            调用结果
        """
        pass


# ============================================================================
# 策略 1: 简单调用策略
# ============================================================================

class SimpleInvocationStrategy(AgentInvocationStrategy):
    """
    简单调用策略
    
    适用场景:
    - 简单、快速的任务
    - 不需要重试
    - 结果确定性高
    """
    
    async def invoke(
        self,
        agent,
        messages: List,
        response_format: Optional[type] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """单次调用"""
        result = await agent.ainvoke({"messages": messages})
        
        # 如果需要结构化输出
        if response_format:
            from src.agents.utils import parse_structured_response
            final_message = result.get("messages", [])[-1]
            structured = parse_structured_response(
                final_message.content,
                response_format
            )
            result["structured_response"] = structured
        
        return result


# ============================================================================
# 策略 2: 重试策略
# ============================================================================

class RetryInvocationStrategy(AgentInvocationStrategy):
    """
    重试调用策略
    
    适用场景:
    - 可能有瞬时错误（网络、API 限流）
    - 需要容错
    
    配置:
    - max_retries: 最大重试次数
    - backoff: 退避策略
    """
    
    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries
    
    async def invoke(
        self,
        agent,
        messages: List,
        response_format: Optional[type] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """带重试的调用"""
        return await invoke_agent_with_retry(
            agent,
            messages,
            response_format
        )


# ============================================================================
# 策略 3: 迭代改进策略
# ============================================================================

class IterativeImprovementStrategy(AgentInvocationStrategy):
    """
    迭代改进策略
    
    适用场景:
    - 需要多次改进才能达到质量标准
    - 有明确的验证条件
    
    流程:
    1. 首次生成
    2. 验证结果
    3. 如果不满足，基于反馈重新生成
    4. 重复直到满足或达到最大次数
    
    Example:
        >>> strategy = IterativeImprovementStrategy(
        ...     max_iterations=3,
        ...     validator=lambda result: result["score"] >= 80
        ... )
        >>> result = await strategy.invoke(agent, messages)
    """
    
    def __init__(
        self,
        max_iterations: int = 3,
        validator: Optional[Callable[[Dict], bool]] = None,
    ):
        self.max_iterations = max_iterations
        self.validator = validator or (lambda r: True)
    
    async def invoke(
        self,
        agent,
        messages: List,
        response_format: Optional[type] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """迭代调用"""
        iteration_history = []
        
        for iteration in range(self.max_iterations):
            # 调用 Agent
            result = await invoke_agent_with_retry(
                agent,
                messages,
                response_format
            )
            
            iteration_history.append(result)
            
            # 验证结果
            if self.validator(result):
                # 满足条件，返回
                result["iterations"] = iteration + 1
                result["history"] = iteration_history
                return result
            
            # 不满足，准备下一次迭代
            if iteration < self.max_iterations - 1:
                # 构建反馈消息
                feedback = self._extract_feedback(result)
                messages = messages + [
                    HumanMessage(content=f"""
Previous attempt did not meet quality standards.

Feedback:
{feedback}

Please improve and try again.
""")
                ]
        
        # 达到最大迭代次数
        result["iterations"] = self.max_iterations
        result["history"] = iteration_history
        result["max_iterations_reached"] = True
        return result
    
    def _extract_feedback(self, result: Dict) -> str:
        """提取反馈信息"""
        structured = result.get("structured_response")
        if structured and hasattr(structured, "issues"):
            issues = structured.issues
            return "\n".join([f"- {issue.title}: {issue.description}" for issue in issues[:3]])
        
        return "Please improve the output quality."


# ============================================================================
# 策略 4: 级联调用策略
# ============================================================================

class CascadeInvocationStrategy(AgentInvocationStrategy):
    """
    级联调用策略
    
    适用场景:
    - 多个 Agent 串联协作
    - 前一个 Agent 的输出是后一个的输入
    
    Example:
        >>> strategy = CascadeInvocationStrategy([
        ...     (mapper_agent, MappingOutput),
        ...     (generator_agent, CodeOutput),
        ...     (reviewer_agent, ReviewOutput),
        ... ])
        >>> result = await strategy.invoke_cascade(initial_messages)
    """
    
    def __init__(self, agent_chain: List[tuple]):
        """
        初始化级联策略
        
        Args:
            agent_chain: [(agent, response_format), ...]
        """
        self.agent_chain = agent_chain
    
    async def invoke(
        self,
        agent,
        messages: List,
        response_format: Optional[type] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """不支持单个 Agent 调用"""
        raise NotImplementedError("Use invoke_cascade for cascade strategy")
    
    async def invoke_cascade(self, initial_messages: List) -> Dict[str, Any]:
        """
        级联调用多个 Agent
        
        Args:
            initial_messages: 初始消息
        
        Returns:
            所有 Agent 的结果
        """
        results = []
        current_messages = initial_messages
        
        for agent, response_format in self.agent_chain:
            # 调用当前 Agent
            result = await invoke_agent_with_retry(
                agent,
                current_messages,
                response_format
            )
            
            results.append(result)
            
            # 准备下一个 Agent 的输入
            final_message = result.get("messages", [])[-1]
            current_messages = [final_message]
        
        return {
            "cascade_results": results,
            "final_result": results[-1] if results else None,
        }


# ============================================================================
# 策略 5: 投票策略
# ============================================================================

class VotingInvocationStrategy(AgentInvocationStrategy):
    """
    投票策略
    
    适用场景:
    - 需要高可靠性的决策
    - 多个 Agent 投票决定
    
    Example:
        >>> strategy = VotingInvocationStrategy(
        ...     agents=[agent1, agent2, agent3],
        ...     voting_method="majority"
        ... )
        >>> result = await strategy.invoke_vote(messages)
    """
    
    def __init__(
        self,
        agents: List,
        voting_method: str = "majority",  # majority, unanimous, weighted
        weights: Optional[List[float]] = None,
    ):
        self.agents = agents
        self.voting_method = voting_method
        self.weights = weights or [1.0] * len(agents)
    
    async def invoke(
        self,
        agent,
        messages: List,
        response_format: Optional[type] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """不支持单个 Agent 调用"""
        raise NotImplementedError("Use invoke_vote for voting strategy")
    
    async def invoke_vote(
        self,
        messages: List,
        response_format: Optional[type] = None,
    ) -> Dict[str, Any]:
        """
        多个 Agent 投票
        
        Args:
            messages: 消息列表
            response_format: 响应格式
        
        Returns:
            投票结果
        """
        import asyncio
        
        # 并行调用所有 Agent
        tasks = [
            invoke_agent_with_retry(agent, messages, response_format)
            for agent in self.agents
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 过滤失败的结果
        valid_results = [r for r in results if not isinstance(r, Exception)]
        
        if not valid_results:
            return {
                "voting_result": None,
                "error": "All agents failed",
                "failed_count": len(results)
            }
        
        # 投票逻辑
        if self.voting_method == "majority":
            final_result = self._majority_vote(valid_results)
        elif self.voting_method == "unanimous":
            final_result = self._unanimous_vote(valid_results)
        elif self.voting_method == "weighted":
            final_result = self._weighted_vote(valid_results)
        else:
            final_result = valid_results[0]  # 默认使用第一个
        
        return {
            "voting_result": final_result,
            "all_results": valid_results,
            "votes_count": len(valid_results),
            "voting_method": self.voting_method,
        }
    
    def _majority_vote(self, results: List[Dict]) -> Dict:
        """多数投票"""
        # 简化实现：返回出现最多的决策
        decisions = [r.get("structured_response", {}).get("decision", "UNKNOWN") 
                    for r in results if r.get("structured_response")]
        
        if decisions:
            from collections import Counter
            most_common = Counter(decisions).most_common(1)[0][0]
            # 返回该决策对应的第一个结果
            for r in results:
                if r.get("structured_response", {}).get("decision") == most_common:
                    return r
        
        return results[0]
    
    def _unanimous_vote(self, results: List[Dict]) -> Optional[Dict]:
        """一致同意投票"""
        decisions = [r.get("structured_response", {}).get("decision") 
                    for r in results if r.get("structured_response")]
        
        if decisions and len(set(decisions)) == 1:
            # 全部一致
            return results[0]
        
        return None  # 不一致
    
    def _weighted_vote(self, results: List[Dict]) -> Dict:
        """加权投票"""
        # 使用分数加权
        weighted_scores = []
        
        for result, weight in zip(results, self.weights):
            structured = result.get("structured_response")
            if structured and hasattr(structured, "overall_score"):
                weighted_scores.append((result, structured.overall_score * weight))
        
        if weighted_scores:
            # 返回加权分数最高的
            best_result = max(weighted_scores, key=lambda x: x[1])
            return best_result[0]
        
        return results[0]


# ============================================================================
# 策略工厂
# ============================================================================

class StrategyFactory:
    """策略工厂"""
    
    @staticmethod
    def create_simple_strategy() -> AgentInvocationStrategy:
        """创建简单调用策略"""
        return SimpleInvocationStrategy()
    
    @staticmethod
    def create_retry_strategy(max_retries: int = 3) -> AgentInvocationStrategy:
        """创建重试策略"""
        return RetryInvocationStrategy(max_retries)
    
    @staticmethod
    def create_iterative_strategy(
        max_iterations: int = 3,
        validator: Optional[Callable] = None,
    ) -> AgentInvocationStrategy:
        """创建迭代改进策略"""
        return IterativeImprovementStrategy(max_iterations, validator)
    
    @staticmethod
    def create_cascade_strategy(agent_chain: List[tuple]) -> CascadeInvocationStrategy:
        """创建级联策略"""
        return CascadeInvocationStrategy(agent_chain)
    
    @staticmethod
    def create_voting_strategy(
        agents: List,
        voting_method: str = "majority",
        weights: Optional[List[float]] = None,
    ) -> VotingInvocationStrategy:
        """创建投票策略"""
        return VotingInvocationStrategy(agents, voting_method, weights)


# ============================================================================
# 使用示例
# ============================================================================

"""
使用示例：

# 1. 简单调用
from src.agents.strategies import SimpleInvocationStrategy

strategy = SimpleInvocationStrategy()
result = await strategy.invoke(agent, messages)

# 2. 带重试
from src.agents.strategies import RetryInvocationStrategy

strategy = RetryInvocationStrategy(max_retries=5)
result = await strategy.invoke(agent, messages)

# 3. 迭代改进（适合代码生成）
from src.agents.strategies import IterativeImprovementStrategy

def quality_validator(result):
    output = result.get("structured_response")
    return output and output.validation_passed

strategy = IterativeImprovementStrategy(
    max_iterations=3,
    validator=quality_validator
)

result = await strategy.invoke(
    code_generator_agent,
    messages=[HumanMessage(content="Generate component...")]
)

# 4. 级联调用（Agent 协作）
from src.agents.strategies import CascadeInvocationStrategy

strategy = CascadeInvocationStrategy([
    (mapper_agent, MappingOutput),
    (generator_agent, CodeOutput),
    (reviewer_agent, ReviewOutput),
])

result = await strategy.invoke_cascade([
    HumanMessage(content="Migrate this component...")
])

final_review = result["final_result"]["structured_response"]

# 5. 多 Agent 投票（高可靠性决策）
from src.agents.strategies import VotingInvocationStrategy

strategy = VotingInvocationStrategy(
    agents=[reviewer1, reviewer2, reviewer3],
    voting_method="majority"
)

result = await strategy.invoke_vote(
    messages=[HumanMessage(content="Review this code...")]
)

consensus = result["voting_result"]

# 6. 使用工厂创建策略
from src.agents.strategies import StrategyFactory

# 为代码生成任务创建迭代策略
gen_strategy = StrategyFactory.create_iterative_strategy(
    max_iterations=3,
    validator=lambda r: r.get("structured_response", {}).get("validation_passed", False)
)

# 为审查任务创建投票策略
review_strategy = StrategyFactory.create_voting_strategy(
    agents=[reviewer1, reviewer2, reviewer3],
    voting_method="majority"
)
"""
