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
from utils.log import setup_root_logger, get_logger
from utils.message_convert import search_messages_to_api_messages
from services.deepseek import DeepSeekAPI


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

# 首先设置根logger - 这很重要，必须在导入其他模块前设置
root_logger = setup_root_logger("logs/total.log")

# 获取主服务logger
logger = get_logger("main")

# 初始化存储器
try:
    milvus = MilvusMessageStore(
        host="localhost",
        port=19530,
        collection_name="chat_messages"
    )
except Exception as e:
    logger.error(f"初始化Milvus存储器失败: {e}")
    milvus = None

# 初始化DeepSeek API客户端
deepseekApi = DeepSeekAPI()

async def test_milvus_connection():
    # 简单测试
    from models import ReceivedMessage
    store = MilvusMessageStore(
        collection_name="chat_messages_test"
    )
    print("MilvusMessageStore 初始化完成")
    # 可以添加更多测试代码
    message = {
        "role": "user",
        "content": "测试消息",
    }
    await store.store_message(ReceivedMessage(**message))

# 测试Deepseek api功能
async def test_deepseek_api():
    from models import ApiMessage
    test_messages = [
        {
            "role": "user",
            "content": "你好，请介绍一下你自己。",
            "id": "user_text_001"
        }
    ]
    try:
        response = await deepseekApi.chat_completion(
            messages=[ApiMessage(**msg) for msg in test_messages],
            stream=False
        )
        print("DeepSeek API 响应:")
        print(response)
    except Exception as e:
        print(f"DeepSeek API 测试异常: {e}")
    

@app.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """增强版聊天补全端点，支持上下文检索"""
    # 存储收到的消息
    msg = request.messages[-1]  # 从前端给的最后一条消息是最新用户消息
    stored_msg = await milvus.store_message(msg)  # 存储消息，返回StoredMessage或None
    if stored_msg is None:
        raise HTTPException(status_code=500, detail="Failed to store message")
    
    context_messages = await milvus.get_context_messages(stored_msg)
    api_messages = search_messages_to_api_messages(context_messages)
    
    try:
        if request.stream:
            # 流式响应
            generator = await deepseekApi.chat_completion(
                messages=api_messages,
                max_tokens=request.max_tokens or 3000,
                temperature=request.temperature or 0.7,
                stream=True,
                # model=request.model
            )
            return StreamingResponse(
                generator,
                media_type="text/plain",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Content-Type": "text/plain; charset=utf-8"
                }
            )
        else:
            # 非流式响应
            response = await deepseekApi.chat_completion(
                messages=api_messages,
                max_tokens=request.max_tokens or 3000,
                temperature=request.temperature or 0.7,
                stream=False,
                # model=request.model
            )
            return response
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
    # 启动服务器
    uvicorn.run(
        app, 
        host="10.176.56.192",
        port=8001,  # 使用不同端口避免冲突
        log_level="info"
    )
