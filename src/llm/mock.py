"""
Mock LLM implementation for offline runs and tests.
"""
from __future__ import annotations

from typing import Any, Type, Union, Literal, get_args, get_origin

from langchain_core.messages import AIMessage
from pydantic import BaseModel
from pydantic_core import PydanticUndefined


class MockLLM:
    """Simple mock LLM that returns placeholder content."""

    is_mock = True

    def __init__(self, response_text: str = "{}"):
        self.response_text = response_text

    async def ainvoke(self, *args: Any, **kwargs: Any) -> AIMessage:
        return AIMessage(content=self.response_text)

    def invoke(self, *args: Any, **kwargs: Any) -> AIMessage:
        return AIMessage(content=self.response_text)

    def with_structured_output(self, schema: Type[BaseModel], **kwargs: Any):
        return MockStructuredOutput(schema)


class MockStructuredOutput:
    """Mock structured output wrapper that returns a model instance."""

    def __init__(self, schema: Type[BaseModel]):
        self.schema = schema

    async def ainvoke(self, *args: Any, **kwargs: Any) -> BaseModel:
        return build_placeholder_model(self.schema)

    def invoke(self, *args: Any, **kwargs: Any) -> BaseModel:
        return build_placeholder_model(self.schema)


def build_placeholder_model(model_cls: Type[BaseModel]) -> BaseModel:
    """Build a minimal placeholder instance without validation."""
    values = {}
    for name, field in model_cls.model_fields.items():
        values[name] = _build_placeholder_value(field.annotation, field)
    return model_cls.model_construct(**values)


def _build_placeholder_value(annotation: Any, field: Any) -> Any:
    if field.default is not PydanticUndefined:
        return field.default
    if field.default_factory is not None:
        return field.default_factory()

    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin in (list, tuple, set):
        return []
    if origin is dict:
        return {}
    if origin is None and isinstance(annotation, type):
        if issubclass(annotation, BaseModel):
            return build_placeholder_model(annotation)
        if annotation is str:
            return "mock"
        if annotation is int:
            return 0
        if annotation is float:
            return 0.0
        if annotation is bool:
            return False
        return None
    if origin is None:
        return None

    # Optional/Union handling
    if origin is Union:
        non_none = [a for a in args if a is not type(None)]
        if non_none:
            return _build_placeholder_value(non_none[0], field)
        return None

    # Literal handling
    if origin is Literal:
        return args[0] if args else "mock"

    return None
