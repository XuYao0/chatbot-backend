import os
import json
import time
import uuid
import asyncio
from typing import List, Dict, Any, Optional, Union, AsyncGenerator
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import requests
import uvicorn

# 导入自定义模块
from models.ChatCompletionRequest import ChatCompletionRequest
from services.milvus import MilvusMessageStore
from utils.log import setup_logger

# FastAPI 应用
app = FastAPI(title="Enhanced OpenAI Compatible API Server", version="2.0.0")

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化增强版 API 客户端
# try:
#     enhanced_api = EnhancedVolcEngineAPI(
#         api_key
#         milvus_host="localhost",
#         milvus_port=19530
#     )
# except Exception as e:
#     print(f"[ERROR] 初始化API客户端失败: {e}")
#     enhanced_api = None

@app.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """增强版聊天补全端点，支持上下文检索"""
    # 存储收到的消息
    msg = request.messages[-1]  # 从前端给的最后一条消息是最新用户消息
    msg = await milvus_store.store_message(msg)  # 存储消息，返回StoredMessage或None
    if msg:
        context_messages = await milvus_store.get_context_messages(msg)

    return None
    # try:
        
    #     if request.stream:
    #         # 流式响应
    #         generator = await enhanced_api.chat_completion(request)
    #         return StreamingResponse(
    #             generator,
    #             media_type="text/plain",
    #             headers={
    #                 "Cache-Control": "no-cache",
    #                 "Connection": "keep-alive",
    #                 "Content-Type": "text/plain; charset=utf-8"
    #             }
    #         )
    #     else:
    #         # 非流式响应
    #         response = await enhanced_api.chat_completion(request)
    #         return response
            
    # except Exception as e:
    #     raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """根路径信息"""
    return {
        "message": "Enhanced OpenAI Compatible API Server with Milvus Context Retrieval",
        "version": "2.0.0",
        "features": [
            "Message storage in Milvus vector database",
            "Context-aware responses using similarity search",
            "Recent message history integration",
            "Automatic message deduplication"
        ],
        "endpoints": {
            "chat": "/chat/completions"
        }
    }

if __name__ == "__main__":
    logger = setup_logger("logs/total.log")
    
    # 初始化存储器
    milvus_store = MilvusMessageStore(
        host="localhost",
        port=19530,
        collection_name="chat_messages"
    )
    
    # 启动服务器
    uvicorn.run(
        app, 
        host="10.176.56.192",
        port=8001,  # 使用不同端口避免冲突
        log_level="info"
    )
