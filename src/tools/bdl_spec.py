"""
BDL 规范查询工具
提供 BDL 规范的智能查询和验证

LangGraph 1.0+ Tools Best Practices:
1. 使用 @tool 装饰器
2. 详细的 docstring
3. 结构化返回值
4. 错误处理

================================================================================
⚠️ 需要定制的部分标记说明:
================================================================================
- [CUSTOMIZE]    = 需要根据实际项目定制的逻辑/配置
- [PLACEHOLDER]  = 占位符代码，需要完整实现  
- [EXAMPLE]      = 示例代码，需要根据实际情况替换

实际使用场景:
1. 从本地 NPM 包读取 BDL 组件定义 (如 @hsbc/bdl-components)
2. 从 Figma Design Tokens 导出文件读取
3. 从公司内部 Design System 仓库读取
================================================================================
"""

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool


# ============================================================================
# BDL 规范加载工具 - 从实际项目读取
# ============================================================================

@tool
def load_bdl_from_npm_package(
    package_name: str = "@hsbc/bdl-components",
    package_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    从 NPM 包读取 BDL 组件定义
    
    实际使用场景:
    1. 从 node_modules/@hsbc/bdl-components 读取组件定义
    2. 解析 package.json 和 TypeScript 类型定义
    3. 提取组件 Props、Variants 等信息
    
    Args:
        package_name: NPM 包名（如 "@hsbc/bdl-components"）
        package_path: 可选的自定义包路径
    
    Returns:
        BDL 规范:
        {
            "version": str,
            "components": {...},
            "design_tokens": {...},
            "source": "npm_package"
        }
    """
    # [CUSTOMIZE] NPM 包路径
    # 实际项目中需要根据 node_modules 位置调整
    if not package_path:
        # 尝试常见位置
        possible_paths = [
            f"./node_modules/{package_name}",
            f"../node_modules/{package_name}",
            f"../../node_modules/{package_name}",
        ]
        package_path = next((p for p in possible_paths if os.path.exists(p)), None)
    
    if not package_path or not os.path.exists(package_path):
        return {
            "error": f"NPM package not found: {package_name}",
            "suggestion": f"Run: npm install {package_name}",
            "components": {},
            "design_tokens": {}
        }
    
    try:
        # [PLACEHOLDER] 实际项目中需要实现 NPM 包解析
        # 
        # 推荐实现步骤:
        # 1. 读取 package.json 获取版本信息
        # 2. 读取 TypeScript 类型定义文件（.d.ts）
        # 3. 解析组件 Props interface
        # 4. 读取 design tokens（如果有单独的 tokens.json）
        # 5. 构建标准化的 BDL 规范格式
        
        package_json_path = os.path.join(package_path, 'package.json')
        
        if os.path.exists(package_json_path):
            with open(package_json_path, 'r', encoding='utf-8') as f:
                package_info = json.load(f)
            
            version = package_info.get('version', 'unknown')
        else:
            version = 'unknown'
        
        # [CUSTOMIZE] 实际读取逻辑
        # 以下是示例，需要根据实际包结构调整
        components = {}
        design_tokens = {}
        
        # 尝试读取 design tokens
        tokens_paths = [
            os.path.join(package_path, 'tokens.json'),
            os.path.join(package_path, 'design-tokens.json'),
            os.path.join(package_path, 'dist', 'tokens.json'),
        ]
        
        for tokens_path in tokens_paths:
            if os.path.exists(tokens_path):
                with open(tokens_path, 'r', encoding='utf-8') as f:
                    design_tokens = json.load(f)
                break
        
        return {
            "version": version,
            "components": components,  # [PLACEHOLDER] 需要实现组件提取
            "design_tokens": design_tokens,
            "source": "npm_package",
            "package_name": package_name,
            "package_path": package_path
        }
    
    except Exception as e:
        return {
            "error": f"Failed to load NPM package: {str(e)}",
            "components": {},
            "design_tokens": {}
        }


@tool
def load_bdl_from_figma_tokens(
    tokens_file_path: str
) -> Dict[str, Any]:
    """
    从 Figma Design Tokens 导出文件读取 BDL 规范
    
    实际使用场景:
    1. 设计师从 Figma 导出 Design Tokens (JSON 格式)
    2. 使用此工具加载并转换为 BDL 规范格式
    
    Args:
        tokens_file_path: Figma tokens 文件路径
    
    Returns:
        BDL 规范（转换后的格式）
    """
    # [CUSTOMIZE] Figma tokens 文件路径
    # 实际项目中需要根据团队的 Figma 导出位置调整
    
    if not os.path.exists(tokens_file_path):
        return {
            "error": f"Figma tokens file not found: {tokens_file_path}",
            "design_tokens": {}
        }
    
    try:
        with open(tokens_file_path, 'r', encoding='utf-8') as f:
            figma_tokens = json.load(f)
        
        # [PLACEHOLDER] 实际项目中需要实现 Figma tokens 格式转换
        # 
        # Figma Design Tokens 通常格式:
        # {
        #   "colors": {
        #     "primary": {"value": "#db0011", "type": "color"},
        #     ...
        #   },
        #   "spacing": {...}
        # }
        #
        # 需要转换为 BDL 规范格式
        
        design_tokens = {}
        
        # 简化转换示例
        for category, tokens in figma_tokens.items():
            design_tokens[category] = {}
            for name, token_data in tokens.items():
                if isinstance(token_data, dict):
                    design_tokens[category][name] = token_data.get("value", token_data)
                else:
                    design_tokens[category][name] = token_data
        
        return {
            "version": "figma_export",
            "design_tokens": design_tokens,
            "source": "figma",
            "source_file": tokens_file_path
        }
    
    except Exception as e:
        return {
            "error": f"Failed to load Figma tokens: {str(e)}",
            "design_tokens": {}
        }


@tool
def load_bdl_from_git_repo(
    repo_path: str,
    spec_file: str = "bdl-spec.json"
) -> Dict[str, Any]:
    """
    从 Git 仓库读取 BDL 规范
    
    实际使用场景:
    1. 公司内部 Design System Git 仓库
    2. 支持版本控制和协作
    3. 可以切换不同分支/版本的规范
    
    Args:
        repo_path: Git 仓库路径（本地克隆）
        spec_file: 规范文件名（相对于仓库根目录）
    
    Returns:
        BDL 规范
    """
    # [CUSTOMIZE] Git 仓库路径
    # 实际项目中需要配置公司内部的 Design System 仓库
    # 例如: /path/to/hsbc-design-system
    
    full_path = os.path.join(repo_path, spec_file)
    
    if not os.path.exists(full_path):
        return {
            "error": f"BDL spec file not found in repo: {full_path}",
            "suggestion": f"Ensure {spec_file} exists in {repo_path}",
            "components": {},
            "design_tokens": {}
        }
    
    try:
        with open(full_path, 'r', encoding='utf-8') as f:
            bdl_spec = json.load(f)
        
        # [PLACEHOLDER] 可以添加 Git 版本信息
        # import subprocess
        # git_hash = subprocess.check_output(
        #     ['git', 'rev-parse', 'HEAD'],
        #     cwd=repo_path
        # ).decode().strip()
        
        bdl_spec["source"] = "git_repo"
        bdl_spec["repo_path"] = repo_path
        # bdl_spec["git_commit"] = git_hash
        
        return bdl_spec
    
    except Exception as e:
        return {
            "error": f"Failed to load from Git repo: {str(e)}",
            "components": {},
            "design_tokens": {}
        }


# ============================================================================
# BDL 组件搜索工具 - 向量搜索
# ============================================================================

@tool
def search_bdl_components(
    query: str,
    bdl_spec: Dict[str, Any],
    top_k: int = 3
) -> List[Dict[str, Any]]:
    """
    向量搜索 BDL 规范，找到最相关的组件
    
    使用语义搜索而非全量传递 BDL 规范给 LLM，可以：
    1. 减少 token 消耗（只传递相关组件）
    2. 提高准确性（精确匹配）
    3. 加快响应速度
    
    Args:
        query: 搜索查询（如 "button with icon and loading state"）
        bdl_spec: 完整的 BDL 规范
        top_k: 返回 top K 个最相关结果
    
    Returns:
        匹配的 BDL 组件列表:
        [
            {
                "name": str,              # 组件名称
                "props": [str],           # 支持的 Props
                "variants": [str],        # 可用变体
                "description": str,       # 组件说明
                "relevance_score": float  # 相关度分数 (0-1)
            }
        ]
    
    Example:
        >>> results = search_bdl_components(
        ...     "clickable button with loading",
        ...     bdl_spec
        ... )
        >>> results[0]["name"]
        'Button'
        >>> results[0]["props"]
        ['variant', 'size', 'loading', 'onClick']
    """
    # [PLACEHOLDER] 实际项目中需要实现语义向量搜索
    # 
    # 推荐实现方案:
    # ═══════════════════════════════════════════════════════════════
    # 
    # Option 1: FAISS vector store
    # --------------------------------------------
    # from langchain_community.vectorstores import FAISS
    # 
    # vectorstore = FAISS.from_texts(
    #     texts=[f"{name}: {comp}" for name, comp in components.items()],
    #     embedding=embeddings,
    #     metadatas=[{"name": name, **comp} for name, comp in components.items()]
    # )
    # results = vectorstore.similarity_search_with_score(query, k=top_k)
    #
    # 方案 2: Sentence Transformers + ChromaDB (本地)
    # --------------------------------------------
    # from sentence_transformers import SentenceTransformer
    # import chromadb
    # 
    # model = SentenceTransformer('all-MiniLM-L6-v2')
    # embeddings = model.encode([query] + component_descriptions)
    # similarities = cosine_similarity(embeddings[0:1], embeddings[1:])[0]
    #
    # 方案 3: 简单关键词匹配（当前实现，适合原型）
    # --------------------------------------------
    # 使用关键词和规则匹配，不需要额外依赖
    # ═══════════════════════════════════════════════════════════════
    
    components = bdl_spec.get("components", {})
    
    if not components:
        return []
    
    # [EXAMPLE] 简化的关键词匹配实现
    # 实际项目中应替换为上述向量搜索方案之一
    query_lower = query.lower()
    results = []
    
    for name, comp_spec in components.items():
        # 计算相关度分数
        score = 0.0
        
        # 1. 组件名称匹配
        if name.lower() in query_lower:
            score += 0.5
        if query_lower in name.lower():
            score += 0.3
        
        # 2. Props 匹配
        props = comp_spec.get("props", [])
        if isinstance(props, list):
            prop_matches = sum(
                1 for p in props 
                if isinstance(p, str) and p.lower() in query_lower
            )
            score += prop_matches * 0.1
        
        # 3. Variants 匹配
        variants = comp_spec.get("variants", [])
        if isinstance(variants, list):
            variant_matches = sum(
                1 for v in variants 
                if isinstance(v, str) and v.lower() in query_lower
            )
            score += variant_matches * 0.1
        
        # 4. 描述匹配
        description = comp_spec.get("description", "")
        if description and query_lower in description.lower():
            score += 0.2
        
        if score > 0:
            results.append({
                "name": name,
                "props": props if isinstance(props, list) else [],
                "variants": variants if isinstance(variants, list) else [],
                "description": description,
                "relevance_score": min(score, 1.0)
            })
    
    # 按相关度排序
    results.sort(key=lambda x: x["relevance_score"], reverse=True)
    
    return results[:top_k]


@tool
def get_bdl_design_token(
    token_path: str,
    bdl_spec: Dict[str, Any]
) -> Any:
    """
    查询 BDL Design Token 值
    
    用于验证代码中使用的 token 是否存在于 BDL 规范中。
    
    Args:
        token_path: Token 路径（如 "colors.primary.500" 或 "spacing.md"）
        bdl_spec: BDL 规范
    
    Returns:
        Token 值（如果存在）或 None
    
    Example:
        >>> get_bdl_design_token("colors.primary.500", bdl_spec)
        "#db0011"
        >>> get_bdl_design_token("spacing.md", bdl_spec)
        "16px"
    """
    keys = token_path.split('.')
    value = bdl_spec.get("design_tokens", {})
    
    for key in keys:
        if isinstance(value, dict):
            value = value.get(key)
        else:
            return None
    
    return value


@tool
def validate_bdl_compliance(
    component_code: str,
    styles_code: str,
    bdl_spec: Dict[str, Any]
) -> Dict[str, Any]:
    """
    验证代码是否符合 BDL 规范
    
    执行静态分析，检查：
    1. 是否使用了未定义的 design token
    2. 是否使用了硬编码的颜色/间距值
    3. 组件结构是否符合 BDL 模式
    
    Args:
        component_code: React 组件代码
        styles_code: CSS 样式代码
        bdl_spec: BDL 规范
    
    Returns:
        {
            "compliant": bool,
            "issues": [
                {
                    "type": "invalid_token" | "hardcoded_value" | "wrong_structure",
                    "severity": "error" | "warning",
                    "message": str,
                    "location": str,  # 文件:行号
                    "suggestion": str
                }
            ],
            "used_tokens": [str],      # 使用的 token 列表
            "undefined_tokens": [str]  # 未定义的 token
        }
    """
    # [PLACEHOLDER] 实际项目中需要实现静态分析
    # 推荐方案:
    # 1. 使用 CSS AST 解析器（如 postcss）
    # 2. 使用正则表达式提取 token 引用
    # 3. 与 BDL 规范对照验证
    
    import re
    
    issues = []
    used_tokens = []
    undefined_tokens = []
    
    # 提取 CSS 变量引用 (--token-name 或 var(--token-name))
    token_pattern = r'--[\w-]+'
    tokens_in_css = set(re.findall(token_pattern, styles_code))
    
    # 提取硬编码的颜色值
    hardcoded_color_pattern = r'#[0-9a-fA-F]{3,6}|rgb\([^)]+\)|rgba\([^)]+\)'
    hardcoded_colors = re.findall(hardcoded_color_pattern, styles_code)
    
    # 检查硬编码颜色
    for color in hardcoded_colors:
        issues.append({
            "type": "hardcoded_value",
            "severity": "warning",
            "message": f"Hardcoded color value found: {color}",
            "location": "styles",
            "suggestion": "Use BDL design token instead"
        })
    
    # 检查 token 是否定义（简化实现）
    # [CUSTOMIZE] 需要根据实际 BDL token 命名规则调整
    for token in tokens_in_css:
        used_tokens.append(token)
        # 简化检查：假设所有 --hsbc-* 开头的是有效的
        if not token.startswith('--hsbc-') and not token.startswith('--bdl-'):
            undefined_tokens.append(token)
            issues.append({
                "type": "invalid_token",
                "severity": "error",
                "message": f"Token may not be defined in BDL: {token}",
                "location": "styles",
                "suggestion": "Check BDL specification for valid tokens"
            })
    
    return {
        "compliant": len(issues) == 0,
        "issues": issues,
        "used_tokens": used_tokens,
        "undefined_tokens": undefined_tokens
    }


@tool
def get_bdl_component_spec(
    component_name: str,
    bdl_spec: Dict[str, Any]
) -> Dict[str, Any]:
    """
    获取指定 BDL 组件的完整规范
    
    Args:
        component_name: 组件名称（如 "Button"）
        bdl_spec: BDL 规范
    
    Returns:
        组件规范或空字典
    """
    components = bdl_spec.get("components", {})
    return components.get(component_name, {})


@tool
def list_bdl_components(
    bdl_spec: Dict[str, Any],
    category: str = None
) -> List[str]:
    """
    列出所有 BDL 组件名称
    
    Args:
        bdl_spec: BDL 规范
        category: 可选的类别过滤
    
    Returns:
        组件名称列表
    """
    components = bdl_spec.get("components", {})
    
    if category:
        # [CUSTOMIZE] 根据实际 BDL 规范的分类方式调整
        return [
            name for name, spec in components.items()
            if spec.get("category") == category
        ]
    
    return list(components.keys())
