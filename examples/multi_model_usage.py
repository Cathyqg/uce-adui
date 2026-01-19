"""
Multi-model usage examples (LiteLLM + Copilot).
"""

from typing import Any, Dict

from src.llm import get_llm, get_structured_llm
from src.core.state import MigrationGraphState, CodeQualityOutput


# ============================================================================
# Example 1: Use different providers
# ============================================================================

async def example_1_different_providers() -> None:
    """Use LiteLLM and Copilot providers."""
    from langchain_core.messages import HumanMessage

    litellm_llm = get_llm(provider="litellm", model="default")
    messages = [HumanMessage(content="Hello")]
    result_litellm = await litellm_llm.ainvoke(messages)
    print(f"LiteLLM: {result_litellm.content}")

    # Copilot is optional and may require internal configuration.
    try:
        copilot_llm = get_llm(provider="copilot", model="default")
        result_copilot = await copilot_llm.ainvoke(messages)
        print(f"Copilot: {result_copilot.content}")
    except Exception as exc:
        print(f"Copilot unavailable: {exc}")


# ============================================================================
# Example 2: Task-based model selection
# ============================================================================

async def example_2_task_based_models() -> None:
    """Select models by task (mapped via LLM_CONFIG)."""
    parsing_llm = get_llm(task="parsing")
    analysis_llm = get_llm(task="analysis")
    generation_llm = get_llm(task="generation")
    review_llm = get_llm(task="review")

    _ = (parsing_llm, analysis_llm, generation_llm, review_llm)


# ============================================================================
# Example 3: Use structured output in a node
# ============================================================================

async def code_quality_review_multi_model(
    state: MigrationGraphState,
) -> Dict[str, Any]:
    """Code quality review node using structured output."""
    components = state.get("components", {})
    results: Dict[str, Any] = {}

    llm = get_structured_llm(
        CodeQualityOutput,
        task="review",
        temperature=0,
    )

    for comp_id in components:
        result = await llm.ainvoke([])
        review_data = result.model_dump()
        results[comp_id] = {"code_quality": review_data}

    return {"parallel_review_results": results}


# ============================================================================
# Example 4: Choose models by cost mode (LiteLLM)
# ============================================================================

async def example_4_hybrid_models() -> None:
    """Switch between fast and smart LiteLLM models."""
    import os

    if os.getenv("COST_MODE") == "low":
        llm = get_llm(provider="litellm", model="fast")
    else:
        llm = get_llm(provider="litellm", model="smart")

    _ = llm


# ============================================================================
# Example 5: Copilot direct usage
# ============================================================================

async def example_5_copilot() -> None:
    """Use Copilot via the factory or directly."""
    from src.llm import get_llm
    from src.llm.providers import CopilotChatModel
    from langchain_core.messages import HumanMessage

    copilot_llm = get_llm(provider="copilot")

    direct = CopilotChatModel(
        model="copilot-default",
        config={
            "api_endpoint": "https://copilot.company.com/api",
            "api_key": "your_key",
            "headers": {"X-Company-ID": "your_company"},
        },
    )

    messages = [HumanMessage(content="Review this code...")]
    _ = await copilot_llm.ainvoke(messages)
    _ = await direct.ainvoke(messages)


if __name__ == "__main__":
    import asyncio

    asyncio.run(example_1_different_providers())
