"""
Tools 模块
提供可复用的工具函数供 LangGraph 节点使用

LangGraph 1.0+ Tools Best Practices:
1. 使用 @tool 装饰器定义工具
2. 提供清晰的 docstring（Agent 会读取）
3. 明确的输入输出类型（Pydantic）
4. 工具应该是无状态的、幂等的
5. 包含完整的错误处理

使用方式:
    from src.tools import CORE_TOOLS, get_tools_by_category
    
    # 获取所有工具
    all_tools = CORE_TOOLS
    
    # 获取特定类别的工具
    validation_tools = get_tools_by_category("validation")
"""

from typing import List

from src.tools.code_validation import (
    validate_typescript_syntax,
    lint_react_code,
    format_with_prettier,
)

from src.tools.bdl_spec import (
    load_bdl_from_npm_package,
    load_bdl_from_figma_tokens,
    load_bdl_from_git_repo,
    search_bdl_components,
    get_bdl_design_token,
    validate_bdl_compliance,
    get_bdl_component_spec,
    list_bdl_components,
)

from src.tools.aem_reader import (
    read_aem_component_from_repo,
    list_aem_components_in_repo,
)

from src.tools.filesystem import (
    read_file_safe,
    write_file_safe,
    create_directory,
    list_directory_files,
    file_exists,
)


# ============================================================================
# 工具注册表 - LangGraph 1.0+ 模式
# ============================================================================

CORE_TOOLS = [
    # === 代码验证工具 ===
    validate_typescript_syntax,
    lint_react_code,
    format_with_prettier,
    
    # === BDL 规范工具 ===
    load_bdl_from_npm_package,      # 从 NPM 包加载
    load_bdl_from_figma_tokens,     # 从 Figma 导出加载
    load_bdl_from_git_repo,         # 从 Git 仓库加载
    search_bdl_components,          # 向量搜索组件
    get_bdl_design_token,           # 查询 Design Token
    validate_bdl_compliance,        # 静态验证 BDL 合规性
    get_bdl_component_spec,         # 获取组件规范
    list_bdl_components,            # 列出所有组件
    
    # === AEM 读取工具 ===
    read_aem_component_from_repo,   # 从仓库读取 AEM 组件
    list_aem_components_in_repo,    # 列出所有 AEM 组件
    
    # === 文件系统工具 ===
    read_file_safe,
    write_file_safe,
    create_directory,
    list_directory_files,
    file_exists,
]

# 按类别分组 - 方便不同节点选择需要的工具
TOOL_CATEGORIES = {
    "validation": [
        validate_typescript_syntax,
        lint_react_code,
    ],
    "formatting": [
        format_with_prettier,
    ],
    "bdl_loaders": [
        load_bdl_from_npm_package,
        load_bdl_from_figma_tokens,
        load_bdl_from_git_repo,
    ],
    "bdl_query": [
        search_bdl_components,
        get_bdl_design_token,
        validate_bdl_compliance,
        get_bdl_component_spec,
        list_bdl_components,
    ],
    "aem_readers": [
        read_aem_component_from_repo,
        list_aem_components_in_repo,
    ],
    "filesystem": [
        read_file_safe,
        write_file_safe,
        create_directory,
        list_directory_files,
        file_exists,
    ],
}


def get_tools_by_category(category: str) -> List:
    """
    根据类别获取工具列表
    
    Args:
        category: 工具类别
            - validation: 代码验证工具
            - formatting: 代码格式化工具
            - bdl_loaders: BDL 规范加载工具
            - bdl_query: BDL 查询工具
            - aem_readers: AEM 读取工具
            - filesystem: 文件系统工具
    
    Returns:
        工具列表
    """
    return TOOL_CATEGORIES.get(category, [])


def get_all_tools():
    """获取所有工具"""
    return CORE_TOOLS


def get_tools_for_node(node_name: str) -> List:
    """
    根据节点名称获取推荐的工具列表
    
    LangGraph Best Practice:
    - 不同节点使用不同的工具子集
    - 避免给 Agent 过多工具选择（导致混淆）
    
    Args:
        node_name: 节点名称
    
    Returns:
        工具列表
    """
    # [CUSTOMIZE] 根据实际节点需求调整工具分配
    node_tool_mapping = {
        "code_quality_review": TOOL_CATEGORIES["validation"] + TOOL_CATEGORIES["formatting"],
        "bdl_compliance_review": TOOL_CATEGORIES["bdl_query"],
        "map_to_bdl": TOOL_CATEGORIES["bdl_query"],
        "ingest_source": TOOL_CATEGORIES["aem_readers"] + TOOL_CATEGORIES["filesystem"],
        "finalize": TOOL_CATEGORIES["filesystem"] + TOOL_CATEGORIES["formatting"],
    }
    
    return node_tool_mapping.get(node_name, [])


__all__ = [
    # === 代码验证工具 ===
    "validate_typescript_syntax",
    "lint_react_code",
    "format_with_prettier",
    
    # === BDL 加载工具 ===
    "load_bdl_from_npm_package",
    "load_bdl_from_figma_tokens",
    "load_bdl_from_git_repo",
    
    # === BDL 查询工具 ===
    "search_bdl_components",
    "get_bdl_design_token",
    "validate_bdl_compliance",
    "get_bdl_component_spec",
    "list_bdl_components",
    
    # === AEM 读取工具 ===
    "read_aem_component_from_repo",
    "list_aem_components_in_repo",
    
    # === 文件系统工具 ===
    "read_file_safe",
    "write_file_safe",
    "create_directory",
    "list_directory_files",
    "file_exists",
    
    # === 注册表和辅助函数 ===
    "CORE_TOOLS",
    "TOOL_CATEGORIES",
    "get_tools_by_category",
    "get_all_tools",
    "get_tools_for_node",
]
