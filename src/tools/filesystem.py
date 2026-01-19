"""
文件系统工具
提供安全的文件读写操作
"""

import os
from pathlib import Path
from typing import Any, Dict

from langchain_core.tools import tool


@tool
def read_file_safe(
    file_path: str,
    encoding: str = "utf-8",
    max_size_mb: float = 10.0
) -> Dict[str, Any]:
    """
    安全地读取文件内容
    
    包含文件大小限制和错误处理，防止读取过大文件导致内存问题。
    
    Args:
        file_path: 文件路径
        encoding: 文件编码（默认 utf-8）
        max_size_mb: 最大文件大小（MB）
    
    Returns:
        {
            "success": bool,
            "content": str,      # 文件内容（如果成功）
            "error": str,        # 错误信息（如果失败）
            "size_bytes": int,   # 文件大小
            "path": str          # 完整路径
        }
    
    Example:
        >>> result = read_file_safe("component.tsx")
        >>> if result["success"]:
        ...     code = result["content"]
    """
    try:
        path = Path(file_path)
        
        # 检查文件是否存在
        if not path.exists():
            return {
                "success": False,
                "content": "",
                "error": f"File not found: {file_path}",
                "size_bytes": 0,
                "path": str(path.absolute())
            }
        
        # 检查文件大小
        size_bytes = path.stat().st_size
        max_size_bytes = max_size_mb * 1024 * 1024
        
        if size_bytes > max_size_bytes:
            return {
                "success": False,
                "content": "",
                "error": f"File too large: {size_bytes / 1024 / 1024:.2f}MB (max: {max_size_mb}MB)",
                "size_bytes": size_bytes,
                "path": str(path.absolute())
            }
        
        # 读取文件
        with open(path, 'r', encoding=encoding) as f:
            content = f.read()
        
        return {
            "success": True,
            "content": content,
            "error": "",
            "size_bytes": size_bytes,
            "path": str(path.absolute())
        }
    
    except UnicodeDecodeError as e:
        return {
            "success": False,
            "content": "",
            "error": f"Encoding error: {str(e)}. Try different encoding.",
            "size_bytes": 0,
            "path": file_path
        }
    
    except Exception as e:
        return {
            "success": False,
            "content": "",
            "error": f"Read error: {str(e)}",
            "size_bytes": 0,
            "path": file_path
        }


@tool
def write_file_safe(
    file_path: str,
    content: str,
    encoding: str = "utf-8",
    create_dirs: bool = True
) -> Dict[str, Any]:
    """
    安全地写入文件内容
    
    包含目录创建和错误处理。
    
    Args:
        file_path: 文件路径
        content: 文件内容
        encoding: 文件编码（默认 utf-8）
        create_dirs: 是否自动创建目录
    
    Returns:
        {
            "success": bool,
            "path": str,        # 完整路径
            "size_bytes": int,  # 写入大小
            "error": str        # 错误信息（如果失败）
        }
    
    Example:
        >>> result = write_file_safe("output/Component.tsx", code)
        >>> if result["success"]:
        ...     print(f"Written to {result['path']}")
    """
    try:
        path = Path(file_path)
        
        # 创建目录
        if create_dirs and path.parent:
            path.parent.mkdir(parents=True, exist_ok=True)
        
        # 写入文件
        with open(path, 'w', encoding=encoding) as f:
            f.write(content)
        
        size_bytes = len(content.encode(encoding))
        
        return {
            "success": True,
            "path": str(path.absolute()),
            "size_bytes": size_bytes,
            "error": ""
        }
    
    except Exception as e:
        return {
            "success": False,
            "path": file_path,
            "size_bytes": 0,
            "error": f"Write error: {str(e)}"
        }


@tool
def create_directory(
    dir_path: str,
    parents: bool = True,
    exist_ok: bool = True
) -> Dict[str, Any]:
    """
    创建目录
    
    Args:
        dir_path: 目录路径
        parents: 是否创建父目录
        exist_ok: 目录已存在是否报错
    
    Returns:
        {
            "success": bool,
            "path": str,
            "created": bool,  # 是否新创建（False 表示已存在）
            "error": str
        }
    """
    try:
        path = Path(dir_path)
        existed = path.exists()
        
        path.mkdir(parents=parents, exist_ok=exist_ok)
        
        return {
            "success": True,
            "path": str(path.absolute()),
            "created": not existed,
            "error": ""
        }
    
    except Exception as e:
        return {
            "success": False,
            "path": dir_path,
            "created": False,
            "error": f"Directory creation error: {str(e)}"
        }


@tool
def list_directory_files(
    dir_path: str,
    pattern: str = "*",
    recursive: bool = False
) -> Dict[str, Any]:
    """
    列出目录中的文件
    
    Args:
        dir_path: 目录路径
        pattern: 文件名模式（如 "*.tsx"）
        recursive: 是否递归搜索
    
    Returns:
        {
            "success": bool,
            "files": [str],    # 文件路径列表
            "count": int,
            "error": str
        }
    """
    try:
        path = Path(dir_path)
        
        if not path.exists():
            return {
                "success": False,
                "files": [],
                "count": 0,
                "error": f"Directory not found: {dir_path}"
            }
        
        if recursive:
            files = [str(p) for p in path.rglob(pattern)]
        else:
            files = [str(p) for p in path.glob(pattern)]
        
        return {
            "success": True,
            "files": files,
            "count": len(files),
            "error": ""
        }
    
    except Exception as e:
        return {
            "success": False,
            "files": [],
            "count": 0,
            "error": f"List error: {str(e)}"
        }


@tool
def file_exists(file_path: str) -> bool:
    """
    检查文件是否存在
    
    Args:
        file_path: 文件路径
    
    Returns:
        是否存在
    """
    return Path(file_path).exists()
