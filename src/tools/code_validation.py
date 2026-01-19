"""
代码验证工具
使用实际的编译器和 Linter 而非仅依赖 LLM

LangGraph 1.0+ Tools Best Practices:
1. 使用 @tool 装饰器定义工具
2. 提供详细的 docstring（Agent 会读取作为工具说明）
3. 返回结构化数据（Dict/Pydantic）
4. 工具应无状态、幂等
5. 包含错误处理和超时控制

================================================================================
⚠️ 需要定制的部分标记说明:
================================================================================
- [CUSTOMIZE]    = 需要根据实际项目定制的逻辑/配置
- [PLACEHOLDER]  = 占位符代码，需要完整实现
- [EXAMPLE]      = 示例代码，需要根据实际情况替换
================================================================================
"""

import json
import os
import subprocess
import tempfile
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool


@tool
def validate_typescript_syntax(code: str, tsconfig: Optional[Dict] = None) -> Dict[str, Any]:
    """
    使用 TypeScript Compiler 验证代码语法
    
    比 LLM 更可靠、速度更快、成本更低。
    应该在 LLM 审查之前调用此工具进行快速验证。
    
    LangGraph Tool Best Practice:
    - 详细的 docstring（Agent 会读取）
    - 结构化返回值
    - 完整的错误处理
    
    Args:
        code: TypeScript/TSX 代码字符串
        tsconfig: 可选的 TypeScript 配置
    
    Returns:
        验证结果:
        {
            "valid": bool,              # 是否通过验证
            "errors": [str],            # 错误列表
            "warnings": [str],          # 警告列表
            "error_count": int,         # 错误数量
            "warning_count": int        # 警告数量
        }
    
    Example:
        >>> result = validate_typescript_syntax("const x: string = 123;")
        >>> result["valid"]
        False
        >>> result["errors"]
        ["Type 'number' is not assignable to type 'string'"]
    """
    # [CUSTOMIZE] TypeScript 编译器配置
    # 需要根据项目的 tsconfig.json 调整以下配置
    TS_CONFIG = tsconfig or {
        "compilerOptions": {
            "target": "ES2020",           # [CUSTOMIZE] 根据目标浏览器调整
            "module": "ESNext",
            "jsx": "react-jsx",           # [CUSTOMIZE] react-jsx (React 17+) 或 react
            "strict": True,
            "esModuleInterop": True,
            "skipLibCheck": True,
            "forceConsistentCasingInFileNames": True
        }
    }
    
    # [PLACEHOLDER] 实际项目中的实现选项：
    # 
    # 方案 1: 调用系统安装的 tsc（当前实现）
    # 方案 2: 使用 TypeScript Language Service API (需要 Node.js 绑定)
    # 方案 3: 使用在线 TypeScript Playground API
    # 方案 4: 集成到 CI/CD，使用项目本地的 tsconfig.json
    
    # 创建临时文件和配置
    temp_dir = tempfile.mkdtemp()
    temp_ts_path = os.path.join(temp_dir, 'component.tsx')
    temp_config_path = os.path.join(temp_dir, 'tsconfig.json')
    
    try:
        # 写入代码文件
        with open(temp_ts_path, 'w', encoding='utf-8') as f:
            f.write(code)
        
        # 写入 tsconfig.json
        with open(temp_config_path, 'w', encoding='utf-8') as f:
            json.dump(TS_CONFIG, f, indent=2)
        
        # [CUSTOMIZE] tsc 命令路径
        # Windows: 'npx' 或 'tsc'
        # Linux/Mac: 'npx' 或 '/usr/local/bin/tsc'
        tsc_command = 'npx'  # [CUSTOMIZE] 根据环境调整
        
        # 调用 TypeScript Compiler
        result = subprocess.run(
            [
                tsc_command,
                'tsc',
                '--project', temp_config_path,
                '--noEmit',  # 只验证，不生成文件
                temp_ts_path
            ],
            capture_output=True,
            text=True,
            timeout=30,  # [CUSTOMIZE] 超时时间可调整
            cwd=temp_dir
        )
        
        errors = []
        warnings = []
        
        if result.returncode != 0:
            # 解析 tsc 输出
            output = result.stdout + result.stderr
            for line in output.split('\n'):
                line = line.strip()
                if not line:
                    continue
                
                # [CUSTOMIZE] 错误解析逻辑可能需要根据 tsc 版本调整
                if 'error TS' in line:
                    errors.append(line)
                elif 'warning TS' in line:
                    warnings.append(line)
        
        return {
            "valid": result.returncode == 0 and len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "error_count": len(errors),
            "warning_count": len(warnings)
        }
    
    except FileNotFoundError:
        # [CUSTOMIZE] 错误消息可能需要根据操作系统调整
        return {
            "valid": False,
            "errors": [
                "TypeScript compiler not found.",
                "Please install: npm install -g typescript",
                "Or ensure 'npx' is available in PATH"
            ],
            "warnings": [],
            "error_count": 1,
            "warning_count": 0
        }
    
    except subprocess.TimeoutExpired:
        return {
            "valid": False,
            "errors": [f"Validation timeout (30s). Code may be too complex."],
            "warnings": [],
            "error_count": 1,
            "warning_count": 0
        }
    
    except Exception as e:
        return {
            "valid": False,
            "errors": [f"Validation error: {str(e)}"],
            "warnings": [],
            "error_count": 1,
            "warning_count": 0
        }
    
    finally:
        # 清理临时文件和目录
        try:
            import shutil
            shutil.rmtree(temp_dir)
        except:
            pass


@tool
def lint_react_code(
    code: str,
    config: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    使用 ESLint 检查 React 代码规范
    
    检查代码风格、最佳实践、潜在问题。
    
    Args:
        code: React 代码字符串
        config: ESLint 配置（可选）
    
    Returns:
        {
            "issues": [              # 问题列表
                {
                    "line": int,
                    "column": int,
                    "severity": "error" | "warning",
                    "message": str,
                    "rule": str,
                    "fixable": bool
                }
            ],
            "error_count": int,
            "warning_count": int,
            "fixable_count": int
        }
    """
    # [PLACEHOLDER] 实际项目中需要实现 ESLint 调用
    
    # 示例实现
    with tempfile.NamedTemporaryFile(
        mode='w',
        suffix='.tsx',
        delete=False,
        encoding='utf-8'
    ) as f:
        f.write(code)
        temp_path = f.name
    
    try:
        # 调用 ESLint
        result = subprocess.run(
            [
                'npx',
                'eslint',
                '--format', 'json',
                '--no-eslintrc',  # 使用默认配置
                temp_path
            ],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        try:
            lint_results = json.loads(result.stdout)
            
            if lint_results and len(lint_results) > 0:
                messages = lint_results[0].get("messages", [])
                
                issues = [
                    {
                        "line": msg.get("line", 0),
                        "column": msg.get("column", 0),
                        "severity": "error" if msg.get("severity") == 2 else "warning",
                        "message": msg.get("message", ""),
                        "rule": msg.get("ruleId", ""),
                        "fixable": msg.get("fix") is not None
                    }
                    for msg in messages
                ]
                
                error_count = sum(1 for i in issues if i["severity"] == "error")
                warning_count = sum(1 for i in issues if i["severity"] == "warning")
                fixable_count = sum(1 for i in issues if i["fixable"])
                
                return {
                    "issues": issues,
                    "error_count": error_count,
                    "warning_count": warning_count,
                    "fixable_count": fixable_count
                }
        except:
            pass
        
        return {
            "issues": [],
            "error_count": 0,
            "warning_count": 0,
            "fixable_count": 0
        }
    
    except FileNotFoundError:
        return {
            "issues": [{
                "line": 0,
                "column": 0,
                "severity": "error",
                "message": "ESLint not found. Please install: npm install -g eslint",
                "rule": "tool-missing",
                "fixable": False
            }],
            "error_count": 1,
            "warning_count": 0,
            "fixable_count": 0
        }
    
    finally:
        try:
            os.unlink(temp_path)
        except:
            pass


@tool
def format_with_prettier(
    code: str,
    parser: str = "typescript"
) -> str:
    """
    使用 Prettier 格式化代码
    
    确保生成的代码格式一致、可读性好。
    
    Args:
        code: 源代码
        parser: 解析器类型（typescript, babel, css等）
    
    Returns:
        格式化后的代码
    
    Example:
        >>> formatted = format_with_prettier("const  x=1;")
        >>> formatted
        'const x = 1;\n'
    """
    # [PLACEHOLDER] 实际项目中需要实现 Prettier 调用
    
    # 示例实现
    with tempfile.NamedTemporaryFile(
        mode='w',
        suffix='.tsx' if parser == 'typescript' else '.css',
        delete=False,
        encoding='utf-8'
    ) as f:
        f.write(code)
        temp_path = f.name
    
    try:
        result = subprocess.run(
            [
                'npx',
                'prettier',
                '--parser', parser,
                '--write',
                temp_path
            ],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        # 读取格式化后的内容
        with open(temp_path, 'r', encoding='utf-8') as f:
            formatted = f.read()
        
        return formatted
    
    except:
        # 格式化失败，返回原代码
        return code
    
    finally:
        try:
            os.unlink(temp_path)
        except:
            pass
