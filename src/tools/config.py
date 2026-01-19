"""
Tools 配置文件
集中管理所有工具的配置参数

================================================================================
⚠️ 需要定制的部分标记说明:
================================================================================
- [CUSTOMIZE]    = 需要根据实际项目定制
- [PLACEHOLDER]  = 占位符，需要实现
================================================================================
"""

from typing import Any, Dict, Optional
import os


# ============================================================================
# TypeScript 编译器配置
# ============================================================================

# [CUSTOMIZE] 根据项目的 TypeScript 配置调整
TYPESCRIPT_CONFIG = {
    "enabled": True,
    "command": "npx",  # [CUSTOMIZE] Windows: "npx", Linux/Mac: "npx" 或绝对路径
    "compiler_options": {
        "target": "ES2020",           # [CUSTOMIZE] ES2015, ES2020, ESNext
        "module": "ESNext",           # [CUSTOMIZE] CommonJS, ESNext
        "jsx": "react-jsx",           # [CUSTOMIZE] react (旧), react-jsx (React 17+)
        "strict": True,
        "esModuleInterop": True,
        "skipLibCheck": True,
        "forceConsistentCasingInFileNames": True,
        "lib": ["ES2020", "DOM", "DOM.Iterable"]
    },
    "timeout": 30,  # 秒
}


# ============================================================================
# ESLint 配置
# ============================================================================

# [CUSTOMIZE] 根据团队代码规范调整
ESLINT_CONFIG = {
    "enabled": True,
    "command": "npx",
    "extends": [
        "eslint:recommended",
        "plugin:react/recommended",
        "plugin:@typescript-eslint/recommended"
    ],
    "rules": {
        # [CUSTOMIZE] 自定义规则
        "react/react-in-jsx-scope": "off",  # React 17+ 不需要
        "react/prop-types": "off",          # 使用 TypeScript
        "@typescript-eslint/explicit-module-boundary-types": "off",
        "no-console": "warn",
    },
    "timeout": 30,
}


# ============================================================================
# Prettier 配置
# ============================================================================

# [CUSTOMIZE] 根据团队代码风格调整
PRETTIER_CONFIG = {
    "enabled": True,
    "command": "npx",
    "options": {
        "printWidth": 100,              # [CUSTOMIZE] 行宽
        "tabWidth": 2,
        "useTabs": False,
        "semi": True,                   # [CUSTOMIZE] 是否使用分号
        "singleQuote": True,            # [CUSTOMIZE] 单引号还是双引号
        "quoteProps": "as-needed",
        "jsxSingleQuote": False,
        "trailingComma": "es5",
        "bracketSpacing": True,
        "jsxBracketSameLine": False,
        "arrowParens": "always",
    },
    "timeout": 10,
}


# ============================================================================
# BDL 向量搜索配置
# ============================================================================

# [CUSTOMIZE] 选择嵌入模型和向量库
BDL_SEARCH_CONFIG = {
    "enabled": True,
    
    # 嵌入方案选择
    "embedding_provider": "in-memory",  # [CUSTOMIZE] "in-memory" | "sentence-transformers" | "cohere"
    
    # Sentence Transformers (本地模型)
    "sentence_transformers": {
        "model": "all-MiniLM-L6-v2",  # [CUSTOMIZE] all-MiniLM-L6-v2 | paraphrase-MiniLM-L6-v2
    },
    
    # 向量库选择
    "vector_store": "faiss",  # [CUSTOMIZE] "faiss" | "chromadb" | "in-memory"
    
    # 搜索参数
    "top_k": 5,                          # 返回 top K 个结果
    "similarity_threshold": 0.7,         # 最低相似度阈值
    "cache_embeddings": True,            # 缓存 embeddings
}


# ============================================================================
# AEM 读取配置
# ============================================================================

# [CUSTOMIZE] 实际 AEM 项目配置
AEM_READER_CONFIG = {
    # AEM 项目结构
    "default_apps_path": "ui.apps/src/main/content/jcr_root/apps",  # [CUSTOMIZE] Maven 标准路径
    "project_name": "mysite",  # [CUSTOMIZE] 替换为实际项目名
    
    # 组件路径模式
    "component_path_pattern": "{apps_path}/{project_name}/components/{component_name}",
    
    # 文件编码
    "encoding": "utf-8",
    
    # 是否读取 Sling Models
    "read_sling_models": False,  # [CUSTOMIZE] 如果需要读取 Java Sling Models
    "sling_models_path": "core/src/main/java",
}


# ============================================================================
# BDL 加载配置
# ============================================================================

# [CUSTOMIZE] BDL 规范来源配置
BDL_LOADER_CONFIG = {
    # 优先级顺序（按顺序尝试）
    "sources": [
        "npm_package",    # 优先从 NPM 包加载
        "git_repo",       # 然后从 Git 仓库
        "figma_export",   # 最后从 Figma 导出
        "default"         # 使用内置默认规范
    ],
    
    # NPM 包配置
    "npm_package": {
        "package_name": "@hsbc/bdl-components",  # [CUSTOMIZE] 实际包名
        "node_modules_path": "./node_modules",   # [CUSTOMIZE] node_modules 路径
    },
    
    # Git 仓库配置
    "git_repo": {
        "repo_path": "../hsbc-design-system",    # [CUSTOMIZE] Design System 仓库路径
        "spec_file": "bdl-spec.json",            # [CUSTOMIZE] 规范文件路径
        "auto_pull": False,                      # 是否自动 git pull 更新
    },
    
    # Figma 导出配置
    "figma_export": {
        "tokens_file": "./design-tokens/figma-tokens.json",  # [CUSTOMIZE] 导出文件路径
        "auto_convert": True,                                 # 自动转换格式
    },
}


# ============================================================================
# 工具全局配置
# ============================================================================

TOOL_GLOBAL_CONFIG = {
    # 并发控制
    "max_concurrent_tool_calls": 10,  # 最多同时执行的工具数
    
    # 超时配置
    "default_timeout": 30,  # 默认超时（秒）
    
    # 重试配置
    "retry_on_failure": True,
    "max_retries": 3,
    
    # 缓存配置
    "cache_enabled": True,
    "cache_ttl": 3600,  # 缓存过期时间（秒）
    
    # 日志级别
    "log_level": "INFO",  # [CUSTOMIZE] DEBUG, INFO, WARNING, ERROR
}


# ============================================================================
# 辅助函数 - 加载配置
# ============================================================================

def get_tool_config(tool_name: str) -> Dict[str, Any]:
    """
    获取指定工具的配置
    
    Args:
        tool_name: 工具名称
    
    Returns:
        工具配置字典
    """
    config_map = {
        "typescript": TYPESCRIPT_CONFIG,
        "eslint": ESLINT_CONFIG,
        "prettier": PRETTIER_CONFIG,
        "bdl_search": BDL_SEARCH_CONFIG,
        "aem_reader": AEM_READER_CONFIG,
        "bdl_loader": BDL_LOADER_CONFIG,
    }
    
    return config_map.get(tool_name, {})


def update_tool_config(tool_name: str, config: Dict[str, Any]) -> None:
    """
    运行时更新工具配置
    
    Args:
        tool_name: 工具名称
        config: 新的配置
    """
    global TYPESCRIPT_CONFIG, ESLINT_CONFIG, PRETTIER_CONFIG
    global BDL_SEARCH_CONFIG, AEM_READER_CONFIG, BDL_LOADER_CONFIG
    
    if tool_name == "typescript":
        TYPESCRIPT_CONFIG.update(config)
    elif tool_name == "eslint":
        ESLINT_CONFIG.update(config)
    elif tool_name == "prettier":
        PRETTIER_CONFIG.update(config)
    elif tool_name == "bdl_search":
        BDL_SEARCH_CONFIG.update(config)
    elif tool_name == "aem_reader":
        AEM_READER_CONFIG.update(config)
    elif tool_name == "bdl_loader":
        BDL_LOADER_CONFIG.update(config)


# ============================================================================
# 环境变量配置（覆盖默认值）
# ============================================================================

# 从环境变量读取配置
if os.getenv("AEM_PROJECT_NAME"):
    AEM_READER_CONFIG["project_name"] = os.getenv("AEM_PROJECT_NAME")

if os.getenv("BDL_NPM_PACKAGE"):
    BDL_LOADER_CONFIG["npm_package"]["package_name"] = os.getenv("BDL_NPM_PACKAGE")

if os.getenv("BDL_REPO_PATH"):
    BDL_LOADER_CONFIG["git_repo"]["repo_path"] = os.getenv("BDL_REPO_PATH")
