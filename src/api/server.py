"""
uce-adui - FastAPI Server
提供 REST API 用于:
- 启动迁移任务
- 人工审查界面交互
- 状态查询
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.core.state import ReviewDecision
from src.main import MigrationEngine


# ============================================================================
# Pydantic Models
# ============================================================================

class MigrationStartRequest(BaseModel):
    """启动迁移请求"""
    source_path: str = Field(..., description="AEM 组件源码路径")
    aem_page_json_paths: Optional[List[str]] = Field(default=None, description="AEM 页面 JSON 路径列表")
    output_dir: Optional[str] = Field(default="./output", description="输出目录")
    bdl_spec_path: Optional[str] = Field(default=None, description="BDL 规范文件路径")
    component_filter: Optional[List[str]] = Field(default=None, description="仅处理指定组件")


class HumanReviewRequest(BaseModel):
    """人工审查请求"""
    component_id: str = Field(..., description="组件 ID")
    decision: str = Field(..., description="审查决定: approve, reject, modify, skip, escalate")
    feedback: Optional[str] = Field(default="", description="反馈说明")
    modification: Optional[Dict[str, str]] = Field(default=None, description="代码修改")
    reviewer: Optional[str] = Field(default="anonymous", description="审查者")


class MigrationStatusResponse(BaseModel):
    """迁移状态响应"""
    status: str
    thread_id: Optional[str] = None
    current_phase: Optional[str] = None
    stats: Optional[Dict[str, Any]] = None
    pending_human_review: Optional[List[str]] = None
    errors: Optional[List[Dict[str, Any]]] = None
    warnings: Optional[List[Dict[str, Any]]] = None


class ReviewPackageResponse(BaseModel):
    """审查数据包响应"""
    component_id: str
    original: Dict[str, Any]
    generated: Dict[str, Any]
    review_results: Dict[str, Any]
    mapping: Optional[Dict[str, Any]] = None


# ============================================================================
# Application Setup
# ============================================================================

# 全局引擎实例
engine: Optional[MigrationEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global engine
    engine = MigrationEngine()
    yield
    # 清理


app = FastAPI(
    title="uce-adui API",
    description="API for migrating AEM components to React with LangGraph",
    version="1.0.0",
    lifespan=lifespan
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy"}


@app.post("/migrations", response_model=MigrationStatusResponse)
async def start_migration(request: MigrationStartRequest):
    """
    启动迁移任务
    
    返回迁移状态。如果需要人工审查会返回 interrupted 状态。
    """
    if not engine:
        raise HTTPException(status_code=500, detail="Engine not initialized")
    
    config = {
        "output_dir": request.output_dir,
        "bdl_spec_path": request.bdl_spec_path,
        "component_filter": request.component_filter
    }
    
    result = await engine.start_migration(
        source_path=request.source_path,
        aem_page_json_paths=request.aem_page_json_paths,
        config=config
    )
    
    return MigrationStatusResponse(**result)


@app.get("/migrations/{thread_id}", response_model=MigrationStatusResponse)
async def get_migration_status(thread_id: str):
    """获取迁移状态"""
    if not engine:
        raise HTTPException(status_code=500, detail="Engine not initialized")
    
    result = await engine.get_migration_status(thread_id)
    
    if result["status"] == "not_found":
        raise HTTPException(status_code=404, detail="Migration not found")
    
    return MigrationStatusResponse(**result)


@app.get("/migrations/{thread_id}/review/{component_id}", response_model=ReviewPackageResponse)
async def get_review_package(thread_id: str, component_id: str):
    """
    获取组件审查数据包
    
    返回审查所需的所有信息，包括原始代码、生成代码、审查结果等。
    """
    if not engine:
        raise HTTPException(status_code=500, detail="Engine not initialized")
    
    package = await engine.get_review_package(thread_id, component_id)
    
    if not package:
        raise HTTPException(status_code=404, detail="Component or migration not found")
    
    return ReviewPackageResponse(**package)


@app.post("/migrations/{thread_id}/review", response_model=MigrationStatusResponse)
async def submit_human_review(thread_id: str, request: HumanReviewRequest):
    """
    提交人工审查结果
    
    提交后系统会继续执行迁移流程。
    """
    if not engine:
        raise HTTPException(status_code=500, detail="Engine not initialized")
    
    # 验证决定
    try:
        decision = ReviewDecision(request.decision)
    except ValueError:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid decision. Must be one of: {[d.value for d in ReviewDecision]}"
        )
    
    result = await engine.submit_human_review(
        thread_id=thread_id,
        component_id=request.component_id,
        decision=decision,
        feedback=request.feedback or "",
        modification=request.modification,
        reviewer=request.reviewer or "anonymous"
    )
    
    if result["status"] == "error":
        raise HTTPException(status_code=400, detail=result.get("error"))
    
    return MigrationStatusResponse(**result)


@app.get("/migrations/{thread_id}/report")
async def get_migration_report(thread_id: str):
    """获取迁移报告"""
    if not engine:
        raise HTTPException(status_code=500, detail="Engine not initialized")
    
    status = await engine.get_migration_status(thread_id)
    
    if status["status"] == "not_found":
        raise HTTPException(status_code=404, detail="Migration not found")
    
    # 检查是否完成
    if status.get("current_phase") != "completed":
        raise HTTPException(status_code=400, detail="Migration not completed yet")
    
    # 返回报告路径或内容
    config = status.get("config", {})
    output_dir = config.get("output_dir", "./output")
    report_path = f"{output_dir}/migration_report.json"
    
    try:
        import json
        with open(report_path, 'r') as f:
            report = json.load(f)
        return report
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Report not found")


# ============================================================================
# WebSocket for real-time updates (可选)
# ============================================================================

from fastapi import WebSocket, WebSocketDisconnect
from typing import Set

# 活跃连接
active_connections: Dict[str, Set[WebSocket]] = {}


@app.websocket("/ws/migrations/{thread_id}")
async def websocket_endpoint(websocket: WebSocket, thread_id: str):
    """
    WebSocket 端点用于实时更新
    
    客户端可以订阅特定迁移任务的更新。
    """
    await websocket.accept()
    
    if thread_id not in active_connections:
        active_connections[thread_id] = set()
    active_connections[thread_id].add(websocket)
    
    try:
        while True:
            # 等待客户端消息或心跳
            data = await websocket.receive_text()
            
            if data == "ping":
                await websocket.send_text("pong")
            elif data == "status":
                # 发送当前状态
                if engine:
                    status = await engine.get_migration_status(thread_id)
                    await websocket.send_json(status)
    
    except WebSocketDisconnect:
        active_connections[thread_id].discard(websocket)
        if not active_connections[thread_id]:
            del active_connections[thread_id]


async def broadcast_update(thread_id: str, message: Dict[str, Any]):
    """向订阅者广播更新"""
    if thread_id in active_connections:
        for connection in active_connections[thread_id]:
            try:
                await connection.send_json(message)
            except:
                pass


# ============================================================================
# 运行服务器
# ============================================================================

def run_server(host: str = "0.0.0.0", port: int = 8000):
    """运行 API 服务器"""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
