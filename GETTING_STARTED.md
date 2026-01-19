# 🚀 快速开始指南

## 📦 安装和配置

### 1. 安装依赖

```bash
# 创建虚拟环境（推荐）
python -m venv venv

# 激活虚拟环境
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
# 复制示例配置
cp env.example .env

# 编辑 .env 文件
notepad .env  # Windows
# 或
nano .env     # Linux/Mac
```

**必需的环境变量**：
```bash
# LLM provider
DEFAULT_LLM_PROVIDER=litellm

# LiteLLM (recommended)
LITELLM_API_BASE=http://localhost:8000
LITELLM_API_KEY=your-litellm-key

# Or Copilot
COPILOT_API_ENDPOINT=https://copilot.company.com/api/v1/chat
COPILOT_API_KEY=your-copilot-key

# Optional: offline test
MIGRATION_USE_MOCK_LLM=1

# Optional: LangSmith tracing
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=...
LANGCHAIN_PROJECT=uce-adui
```

---

## 🏃 运行项目

### 方式 1: 命令行（CLI）

```bash
# 基础用法
python -m src.main path/to/aem-components

# 完整参数
python -m src.main path/to/aem-components \
    --pages path/to/page1.json path/to/page2.json \
    --output ./output \
    --bdl-spec path/to/bdl-spec.json

# Default uses hybrid graph; pipeline-only:
# python -m src.main path/to/aem-components --graph pipeline
```

### 方式 2: Python 代码

```python
import asyncio
from src.main import MigrationEngine

async def main():
    # 创建引擎
    engine = MigrationEngine()
    
    # 启动迁移
    result = await engine.start_migration(
        source_path="examples/aem-components",
        aem_page_json_paths=["examples/aem-pages/home.json"],
    )
    
    # 检查结果
    if result["status"] == "completed":
        print(f"✅ 迁移完成！")
        print(f"组件数: {result['stats']['generated_components']}")
        print(f"报告: {result.get('report_path')}")
    elif result["status"] == "interrupted":
        print(f"⏸️ 等待人工审查")
        print(f"Thread ID: {result['thread_id']}")
    else:
        print(f"❌ 失败: {result.get('error')}")

if __name__ == "__main__":
    asyncio.run(main())
```

### 方式 3: FastAPI Server

```bash
# 启动服务器
python -m src.api.server --host 0.0.0.0 --port 8000

# 访问 API 文档
http://localhost:8000/docs
```

**API 调用示例**：
```bash
# 启动迁移
curl -X POST http://localhost:8000/migrations \
  -H "Content-Type: application/json" \
  -d '{
    "source_path": "path/to/aem-components",
    "config": {
      "output_dir": "./output"
    }
  }'

# 查询状态
curl http://localhost:8000/migrations/{thread_id}
```

---

## 🧪 测试单个节点

### 测试 Pipeline 节点

```python
# test_pipeline_node.py
import asyncio
from src.nodes.pipeline.component_conversion import parse_aem
from src.core.state import create_initial_state

async def test_parse_aem_node():
    """测试 AEM 解析节点"""
    
    # 准备测试状态
    state = create_initial_state(
        source_path="examples/aem-components",
    )
    
    # 添加测试数据
    state["components"] = {
        "hero-banner": {
            "component_id": "hero-banner",
            "status": "pending",
            "aem_component": {
                "htl_template": {
                    "raw_content": "<div>Test HTL</div>"
                }
            }
        }
    }
    state["component_queue"] = ["hero-banner"]
    
    # 调用节点
    result = await parse_aem(state)
    
    # 验证结果
    print(f"状态: {result['components']['hero-banner']['status']}")
    print(f"解析结果: {result['components']['hero-banner'].get('aem_component', {}).get('htl_template')}")
    
    assert "hero-banner" in result["components"]
    print("✅ 测试通过！")

if __name__ == "__main__":
    asyncio.run(test_parse_aem_node())
```

**运行**：
```bash
python test_pipeline_node.py
```

### 测试 Intelligent 节点

```python
# test_intelligent_node.py
import asyncio
from src.nodes.intelligent.bdl_mapping import bdl_mapping_node
from src.core.state import create_initial_state

async def test_bdl_mapping_node():
    """测试 BDL 映射节点（使用 Agent）"""
    
    # 准备测试状态
    state = create_initial_state(source_path="examples/aem-components")
    
    # 添加测试数据
    state["bdl_spec"] = {
        "components": {
            "Button": {"type": "button", "variants": ["primary", "secondary"]},
            "Hero": {"type": "hero", "props": ["title", "image"]},
        }
    }
    
    state["components"] = {
        "hero-banner": {
            "component_id": "hero-banner",
            "status": "analyzing",  # BDL mapping 需要这个状态
            "aem_component": {
                "component_group": "content",
                "title": "Hero Banner"
            },
            "analyzed": {
                "component_type": "ui",
                "is_dynamic": True,
                "features": {
                    "has_form": False,
                    "has_animation": True,
                },
                "complexity": {
                    "lines_of_code": 150,
                    "dependency_count": 2,
                }
            }
        }
    }
    
    # 调用节点（Agent 会自动搜索和映射）
    print("开始 BDL 映射（Agent 会调用工具）...")
    result = await bdl_mapping_node(state)
    
    # 验证结果
    mapping = result["components"]["hero-banner"].get("bdl_mapping", {})
    print(f"\n映射结果:")
    print(f"  BDL 组件: {mapping.get('bdl_component_name')}")
    print(f"  置信度: {mapping.get('confidence_score', 0):.2%}")
    print(f"  推理: {mapping.get('reasoning', '')[:100]}...")
    
    assert "bdl_mapping" in result["components"]["hero-banner"]
    print("\n✅ 测试通过！")

if __name__ == "__main__":
    asyncio.run(test_bdl_mapping_node())
```

**运行**：
```bash
python test_intelligent_node.py
```

---

## 🔬 测试单个 Agent

### 方式 1: 直接测试 Agent

```python
# test_agent_direct.py
import asyncio
from langchain_core.messages import HumanMessage

# 导入 Agent 创建函数
from src.agents.core import create_bdl_mapping_agent, BDLMappingOutput
from src.agents.utils import invoke_agent_with_retry

async def test_bdl_mapper_agent():
    """直接测试 BDL Mapper Agent"""
    
    # 创建 Agent
    agent = create_bdl_mapping_agent()
    print("✅ Agent 创建成功")
    
    # 准备测试消息
    test_message = """
Map this AEM component to BDL:

**Component**: hero-banner
**Type**: ui component
**Features**: Has image, title, subtitle, CTA button
**Dialog Fields**: title, subtitle, image, ctaText, ctaLink

Available BDL Components:
- Hero: Large hero banner with image and CTA
- Banner: Simple banner component
- Card: Card component with image

Use tools to find the best match.
"""
    
    # 调用 Agent（带结构化输出）
    print("调用 Agent...")
    result = await invoke_agent_with_retry(
        agent,
        messages=[HumanMessage(content=test_message)],
        response_format=BDLMappingOutput,
    )
    
    # 获取结构化结果
    mapping: BDLMappingOutput = result.get("structured_response")
    
    if mapping:
        print(f"\n✅ 映射结果:")
        print(f"  BDL 组件: {mapping.bdl_component_name}")
        print(f"  置信度: {mapping.confidence_score:.2%}")
        print(f"  属性映射: {len(mapping.prop_mappings)} 个")
        print(f"  推理: {mapping.reasoning[:150]}...")
    else:
        print("❌ 未获取到结构化输出")
    
    # 查看 Agent 执行的工具调用
    messages = result.get("messages", [])
    tool_calls = [m for m in messages if hasattr(m, 'tool_calls') and m.tool_calls]
    print(f"\n工具调用次数: {len(tool_calls)}")
    
    return mapping

if __name__ == "__main__":
    mapping = asyncio.run(test_bdl_mapper_agent())
```

**运行**：
```bash
python test_agent_direct.py
```

### 方式 2: 使用工厂测试

```python
# test_agent_factory.py
import asyncio
from langchain_core.messages import HumanMessage
from src.agents import AgentFactory, AgentType, invoke_agent_with_retry
from src.agents.core import BDLMappingOutput

async def test_with_factory():
    """使用工厂创建和测试 Agent"""
    
    # 使用工厂创建
    factory = AgentFactory()
    agent = factory.create_agent(AgentType.BDL_MAPPER)
    
    print(f"✅ 使用工厂创建 Agent")
    print(f"配置: {factory.get_agent_info(AgentType.BDL_MAPPER)}")
    
    # 调用
    result = await invoke_agent_with_retry(
        agent,
        messages=[HumanMessage(content="Map hero-banner to BDL...")],
        response_format=BDLMappingOutput,
    )
    
    mapping: BDLMappingOutput = result["structured_response"]
    print(f"\nBDL Component: {mapping.bdl_component_name}")
    print(f"Confidence: {mapping.confidence_score:.2%}")

if __name__ == "__main__":
    asyncio.run(test_with_factory())
```

### 方式 3: 使用 Middleware 测试

```python
# test_agent_middleware.py
import asyncio
from langchain_core.messages import HumanMessage
from src.agents.core import create_bdl_mapping_agent, BDLMappingOutput
from src.agents.middleware import create_context_injector, create_monitor, compose_middlewares

async def test_with_middleware():
    """测试带 Middleware 的 Agent"""
    
    # 创建基础 Agent
    base_agent = create_bdl_mapping_agent()
    
    # 添加 Middleware
    agent_enhanced = compose_middlewares(
        create_context_injector(["bdl_spec"]),     # 自动注入上下文
        create_monitor(log_timing=True),           # 监控执行时间
    ) | base_agent
    
    print("✅ Agent + Middleware 已就绪")
    
    # 调用（上下文会自动注入）
    result = await agent_enhanced.ainvoke({
        "messages": [HumanMessage(content="Map hero-banner...")],
        "bdl_spec": {  # 这个会被自动注入到消息中
            "components": {
                "Hero": {"type": "hero", "props": ["title", "image"]},
            }
        }
    })
    
    print(f"✅ Agent 执行完成")
    print(f"消息数量: {len(result.get('messages', []))}")

if __name__ == "__main__":
    asyncio.run(test_with_middleware())
```

---

## 🧪 单元测试

### 创建测试文件

```bash
# 创建测试目录
mkdir tests
mkdir tests/nodes
mkdir tests/agents
```

### 测试 Agent Utils

```python
# tests/agents/test_utils.py
import pytest
from pydantic import BaseModel, Field
from src.agents.utils import (
    parse_structured_response,
    inject_context_to_message,
    parse_json_from_content,
)

class TestOutput(BaseModel):
    value: str
    score: float = Field(ge=0, le=1)

def test_parse_json_block():
    """测试解析 JSON 代码块"""
    content = '```json\n{"value": "test", "score": 0.9}\n```'
    result = parse_structured_response(content, TestOutput)
    
    assert result is not None
    assert result.value == "test"
    assert result.score == 0.9

def test_parse_plain_json():
    """测试解析纯 JSON"""
    content = '{"value": "test", "score": 0.9}'
    result = parse_structured_response(content, TestOutput)
    
    assert result is not None
    assert result.value == "test"

def test_inject_context():
    """测试上下文注入"""
    message = "User query"
    context = {"key1": "value1", "key2": {"nested": "data"}}
    
    result = inject_context_to_message(message, context, max_length=100)
    
    assert "User query" in result
    assert "Key1" in result  # 标题化
    assert "value1" in result

def test_parse_json_from_content():
    """测试 JSON 提取"""
    content = 'Some text {"result": "success"} more text'
    result = parse_json_from_content(content)
    
    assert result["result"] == "success"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

**运行测试**：
```bash
pytest tests/agents/test_utils.py -v
```

### 测试节点

```python
# tests/nodes/test_pipeline_nodes.py
import pytest
import asyncio
from src.nodes.pipeline.component_conversion import ingest_source, parse_aem
from src.core.state import create_initial_state

@pytest.mark.asyncio
async def test_ingest_source():
    """测试源码摄入节点"""
    state = create_initial_state(
        source_path="examples/aem-components",
    )
    
    result = ingest_source(state)
    
    # 验证
    assert "components" in result
    assert len(result["components"]) > 0
    assert "hero-banner" in result["components"]

@pytest.mark.asyncio
async def test_parse_aem():
    """测试 AEM 解析节点"""
    state = create_initial_state(source_path="examples/aem-components")
    
    # 先摄入
    state = {**state, **ingest_source(state)}
    
    # 然后解析
    result = await parse_aem(state)
    
    # 验证
    assert "components" in result
    for comp_data in result["components"].values():
        if comp_data.get("status") == "parsing":
            assert "aem_component" in comp_data

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

**运行**：
```bash
pytest tests/nodes/test_pipeline_nodes.py -v
```

---

## 🔍 调试和开发

### 1. 交互式测试（IPython/Jupyter）

```bash
# 安装 IPython
pip install ipython

# 启动
ipython
```

```python
# 在 IPython 中
from src.llm import get_llm
from src.agents.core import create_bdl_mapping_agent
from langchain_core.messages import HumanMessage

# 创建 Agent
agent = create_bdl_mapping_agent()

# 测试调用
result = await agent.ainvoke({
    "messages": [HumanMessage(content="Test message")]
})

# 查看结果
result["messages"][-1].content
```

### 2. 使用日志调试

```python
# debug_example.py
import logging
import asyncio

# 配置日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from src.nodes.intelligent.bdl_mapping import bdl_mapping_node
from src.core.state import create_initial_state

async def debug_node():
    state = create_initial_state(source_path="examples/aem-components")
    
    # 添加测试数据...
    state["components"] = {...}
    
    # 调用会输出详细日志
    result = await bdl_mapping_node(state)
    
    return result

if __name__ == "__main__":
    asyncio.run(debug_node())
```

### 3. 使用 LangSmith 追踪

```python
# 启用 LangSmith（在 .env 中）
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your-key
LANGCHAIN_PROJECT=uce-adui

# 运行代码
python -m src.main path/to/components

# 在 LangSmith UI 中查看完整追踪
# https://smith.langchain.com/
```

---

## 🎯 常见场景

### 场景 1: 只测试某个步骤

```python
# test_specific_step.py
import asyncio
from src.nodes import analyze_component
from src.core.state import create_initial_state

async def test_analyze_only():
    """只测试组件分析步骤"""
    state = create_initial_state(source_path="examples/aem-components")
    
    # 准备已解析的组件数据
    state["components"] = {
        "hero-banner": {
            "status": "parsing",  # 分析需要这个状态
            "aem_component": {
                # ... AEM 组件数据
            }
        }
    }
    
    # 只调用分析节点
    result = await analyze_component(state)
    
    # 查看分析结果
    analyzed = result["components"]["hero-banner"]["analyzed"]
    print(f"组件类型: {analyzed['component_type']}")
    print(f"复杂度: {analyzed['complexity']}")
    print(f"BDL 可行性: {analyzed['bdl_mapping_feasibility']}")

asyncio.run(test_analyze_only())
```

### 场景 2: 测试完整流程（小规模）

```python
# test_full_flow.py
import asyncio
from src.main import MigrationEngine

async def test_small_migration():
    """测试小规模迁移（1个组件）"""
    
    engine = MigrationEngine(config={
        "component_filter": ["hero-banner"],  # 只迁移这一个
        "auto_approve_all": True,  # 跳过人工审查
    })
    
    result = await engine.start_migration(
        source_path="examples/aem-components",
    )
    
    print(f"状态: {result['status']}")
    print(f"统计: {result.get('stats', {})}")
    
    return result

asyncio.run(test_small_migration())
```

### 场景 3: 测试 Agent 工具调用

```python
# test_agent_tools.py
import asyncio
from src.tools import search_bdl_components, validate_typescript_syntax

def test_bdl_search():
    """测试 BDL 搜索工具"""
    result = search_bdl_components.invoke({
        "query": "button with loading state",
        "bdl_spec": {
            "components": {
                "Button": {"variants": ["primary", "loading"]},
                "LoadingButton": {"has_loading": True},
            }
        },
        "top_k": 3
    })
    
    print(f"搜索结果: {len(result.get('matches', []))} 个")
    for match in result.get("matches", []):
        print(f"  - {match['component_name']}: {match['score']:.2%}")

def test_typescript_validation():
    """测试 TypeScript 验证工具"""
    code = """
import React from 'react';

const Test: React.FC = () => {
    return <div>Hello</div>;
};
"""
    
    result = validate_typescript_syntax.invoke({"code": code})
    
    print(f"验证结果: {'✅ 通过' if result['valid'] else '❌ 失败'}")
    if result['errors']:
        print(f"错误: {result['errors']}")

if __name__ == "__main__":
    test_bdl_search()
    test_typescript_validation()
```

---

## 🐛 常见问题

### Q1: ModuleNotFoundError

```bash
# 确保在项目根目录
cd d:\Code\uce-adui

# 确保虚拟环境已激活
venv\Scripts\activate

# 重新安装依赖
pip install -r requirements.txt
```

### Q2: 找不到 API Key

```bash
# 检查 .env 文件是否存在
ls .env

# 检查环境变量是否加载
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print(os.getenv('LITELLM_API_KEY') or os.getenv('COPILOT_API_KEY'))"
```

### Q3: Agent 不调用工具

```python
# 检查工具是否正确注册
from src.agents.core import create_bdl_mapping_agent

agent = create_bdl_mapping_agent()

# 打印 Agent 信息
print(f"Agent type: {type(agent)}")
# 检查工具列表（根据 create_react_agent 的实现）
```

### Q4: 结构化输出解析失败

```python
# 方式 1: 检查 Agent 响应
result = await agent.ainvoke(...)
final_message = result["messages"][-1]
print("Agent 原始响应:")
print(final_message.content)

# 方式 2: 手动测试解析
from src.agents.utils import parse_structured_response

parsed = parse_structured_response(
    final_message.content,
    YourOutputModel
)
print(f"解析结果: {parsed}")
```

---

## 📊 性能测试

### 测试执行时间

```python
# benchmark.py
import asyncio
import time
from src.nodes import parse_aem, bdl_mapping_node
from src.core.state import create_initial_state

async def benchmark_nodes():
    """基准测试各节点性能"""
    state = create_initial_state(source_path="examples/aem-components")
    
    # 准备状态...
    state["components"] = {"hero-banner": {...}}
    
    # 测试 Pipeline 节点
    start = time.time()
    result1 = await parse_aem(state)
    pipeline_time = time.time() - start
    
    # 测试 Intelligent 节点
    start = time.time()
    result2 = await bdl_mapping_node(state)
    intelligent_time = time.time() - start
    
    print(f"Pipeline 节点: {pipeline_time:.2f}秒")
    print(f"Intelligent 节点: {intelligent_time:.2f}秒")
    print(f"差异: {intelligent_time / pipeline_time:.1f}x")

asyncio.run(benchmark_nodes())
```

---

## 🎯 快速测试检查清单

### 基础测试（5 分钟）

```bash
# 1. 测试导入
python -c "from src.nodes import parse_aem, bdl_mapping_node; print('✅ 导入成功')"

# 2. 测试 LLM
python -c "from src.llm import get_llm; llm = get_llm(); print('✅ LLM 创建成功')"

# 3. 测试 Agent 创建
python -c "from src.agents.core import create_bdl_mapping_agent; agent = create_bdl_mapping_agent(); print('✅ Agent 创建成功')"

# 4. 测试工具
python -c "from src.tools import search_bdl_components; print('✅ 工具导入成功')"
```

### 集成测试（15 分钟）

```bash
# 运行完整的小规模测试
python test_intelligent_node.py

# 运行单个组件迁移
python test_full_flow.py
```

---

## 📚 推荐的开发流程

### 1. 开发新节点

```python
# Step 1: 在 tests/ 中写测试
# tests/nodes/test_my_node.py
async def test_my_node():
    # 准备测试数据
    state = {...}
    # 调用节点
    result = await my_node(state)
    # 验证
    assert ...

# Step 2: 实现节点
# src/nodes/pipeline/my_node.py  或
# src/nodes/intelligent/my_node.py

# Step 3: 运行测试
pytest tests/nodes/test_my_node.py -v

# Step 4: 集成到图中
# src/core/graph.py
graph.add_node("my_node", my_node)
```

### 2. 调整 Agent 提示词

```python
# Step 1: 修改提示词
# src/agents/core.py
def create_bdl_mapping_agent():
    system_prompt = """
    Your new improved prompt...
    """
    ...

# Step 2: 测试效果
python test_agent_direct.py

# Step 3: 对比结果
# 使用不同提示词运行，对比输出质量
```

### 3. 添加新工具

```python
# Step 1: 定义工具
# src/tools/my_tool.py
from langchain_core.tools import tool

@tool
def my_validation_tool(code: str) -> Dict[str, Any]:
    """验证代码的某个方面"""
    # 实现...
    return {"valid": True, "issues": []}

# Step 2: 测试工具
def test_my_tool():
    result = my_validation_tool.invoke({"code": "test code"})
    assert result["valid"]

# Step 3: 添加到 Agent
# src/agents/core.py
tools = [..., my_validation_tool]
```

---

## 🎉 完整示例

创建一个完整的测试脚本：

```python
# run_complete_test.py
import asyncio
import logging
from src.main import MigrationEngine

# 配置日志
logging.basicConfig(level=logging.INFO)

async def main():
    print("=" * 60)
    print("uce-adui - 完整测试")
    print("=" * 60)
    
    # 创建引擎
    print("\n1. 创建迁移引擎...")
    engine = MigrationEngine(config={
        "component_filter": ["hero-banner"],  # 只测试一个组件
        "auto_approve_all": True,  # 自动通过审查
    })
    print("✅ 引擎就绪")
    
    # 启动迁移
    print("\n2. 启动迁移流程...")
    result = await engine.start_migration(
        source_path="examples/aem-components",
        aem_page_json_paths=["examples/aem-pages/home.json"],
    )
    
    # 输出结果
    print("\n3. 迁移结果:")
    print(f"   状态: {result['status']}")
    
    if result["status"] == "completed":
        stats = result.get("stats", {})
        print(f"   总组件: {stats.get('total_components', 0)}")
        print(f"   已生成: {stats.get('generated_components', 0)}")
        print(f"   已审批: {stats.get('approved_components', 0)}")
        print(f"   报告: {result.get('report_path', 'N/A')}")
        print("\n✅ 迁移成功完成！")
    elif result["status"] == "interrupted":
        print(f"   Thread ID: {result['thread_id']}")
        print(f"   待审查: {result.get('pending_review', [])}")
        print("\n⏸️ 等待人工审查")
    else:
        print(f"   错误: {result.get('error', 'Unknown')}")
        print("\n❌ 迁移失败")
    
    print("\n" + "=" * 60)
    return result

if __name__ == "__main__":
    result = asyncio.run(main())
```

**运行**：
```bash
python run_complete_test.py
```

---

## 💡 提示

### 快速验证安装

```bash
# 一行命令测试所有关键模块
python -c "from src.nodes import parse_aem; from src.agents.core import create_bdl_mapping_agent; from src.llm import get_llm; from src.tools import search_bdl_components; print('✅ 所有模块导入成功！')"
```

### 查看可用的节点

```python
from src.nodes import __all__ as node_exports

print("可用的节点:")
for node in node_exports:
    print(f"  - {node}")
```

### 查看可用的 Agent

```python
from src.agents.factory import AgentType

print("可用的 Agent:")
for agent_type in AgentType:
    print(f"  - {agent_type.value}")
```

---

**现在你可以轻松运行和测试项目了！** 🚀
