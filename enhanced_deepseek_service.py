"""
增强版的 DeepSeek API 服务
集成 Milvus 向量数据库进行消息存储和上下文检索
"""
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
from models import Message, SearchResult, ChatSession
from milvus_store import MilvusMessageStore
from image_handler import ImageHandler

class ChatCompletionRequest(BaseModel):
    model: Optional[str] = "deepseek-r1-250528"
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    frequency_penalty: Optional[float] = 0.0
    presence_penalty: Optional[float] = 0.0
    max_tokens: Optional[int] = 3000
    stream: Optional[bool] = False
    messages: List[Message]
    
    # 新增字段
    stop: Optional[Union[str, List[str]]] = None
    use_context: Optional[bool] = True  # 是否使用上下文检索
    user_id: Optional[str] = None  # 用户ID
    session_id: Optional[str] = None  # 会话ID

class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: Optional[Message] = None
    delta: Optional[Dict[str, Any]] = None
    finish_reason: Optional[str] = None

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: Optional[Usage] = None
    context_info: Optional[Dict[str, Any]] = None  # 上下文信息

class EnhancedVolcEngineAPI:
    """增强版火山引擎 API 调用类，集成 Milvus 存储"""
    
    def __init__(
        self, 
        api_key: Optional[str] = None, 
        base_url: Optional[str] = None, 
        model_name: str = "deepseek-r1-250528",
        milvus_host: str = "localhost",
        milvus_port: int = 19530
    ):
        self.api_key = api_key or os.getenv("VOLCENG_API_KEY")
        self.base_url = base_url or "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
        self.model_name = model_name
        
        if not self.api_key:
            raise ValueError("API key is required")
        
        # 初始化 Milvus 存储
        try:
            self.milvus_store = MilvusMessageStore(
                host=milvus_host,
                port=milvus_port
            )
            print(f"[INFO] Milvus 存储初始化成功")
        except Exception as e:
            print(f"[WARNING] Milvus 存储初始化失败: {e}")
            self.milvus_store = None
        
        # 初始化图片处理器
        try:
            self.image_handler = ImageHandler()
            print(f"[INFO] 图片处理器初始化成功")
        except Exception as e:
            print(f"[WARNING] 图片处理器初始化失败: {e}")
            self.image_handler = None
    
    def _convert_messages_for_api(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """将消息格式转换为 API 格式"""
        converted_messages = []
        
        for msg in messages:
            if isinstance(msg.content, list):
                # 多模态消息处理
                text_parts = []
                image_parts = []
                
                for content_item in msg.content:
                    if content_item.get("type") == "text":
                        text_parts.append(content_item.get("text", ""))
                    elif content_item.get("type") == "image_url":
                        image_parts.append(f"[图片: {content_item.get('image_url', {}).get('url', '')[:50]}...]")
                
                combined_content = " ".join(text_parts)
                if image_parts:
                    combined_content += " " + " ".join(image_parts)
                
                converted_messages.append({
                    "role": msg.role,
                    "content": combined_content
                })
            else:
                converted_messages.append({
                    "role": msg.role,
                    "content": str(msg.content)
                })
        
        return converted_messages
    
    async def _process_and_save_images(self, message: Message) -> Message:
        """处理消息中的图片并保存到本地，返回更新后的消息"""
        if not self.image_handler or not isinstance(message.content, list):
            return message
        
        updated_content = []
        
        for content_item in message.content:
            if content_item.get("type") == "image_url":
                image_url_data = content_item.get("image_url", {})
                image_data = image_url_data.get("url", "")
                
                # 处理 base64 图片数据
                if image_data.startswith("data:image/"):
                    try:
                        # 保存 base64 图片
                        saved_path = await asyncio.get_event_loop().run_in_executor(
                            None, 
                            self.image_handler.save_image_from_base64, 
                            image_data, 
                            f"chat_{message.id}_{int(time.time())}"
                        )
                        
                        if saved_path:
                            # 更新内容项，添加本地路径
                            updated_item = content_item.copy()
                            updated_item["image_url"]["local_path"] = saved_path
                            updated_content.append(updated_item)
                            print(f"[INFO] 图片已保存到: {saved_path}")
                        else:
                            updated_content.append(content_item)
                    except Exception as e:
                        print(f"[WARNING] 保存图片失败: {e}")
                        updated_content.append(content_item)
                else:
                    updated_content.append(content_item)
            else:
                updated_content.append(content_item)
        
        # 更新消息内容
        message.content = updated_content
        return message
    
    def _extract_latest_message(self, messages: List[Message], user_id: Optional[str] = None, session_id: Optional[str] = None) -> Optional[Message]:
        """从消息列表中提取最新的用户消息，并生成完整的元数据"""
        # 找到最后一条用户消息
        user_messages = [msg for msg in messages if msg.role == "user"]
        if not user_messages:
            return None
        
        latest_message = user_messages[-1]
        
        # 生成新的消息 ID 和时间戳（如果没有的话）
        if not latest_message.id:
            latest_message.id = f"user_{uuid.uuid4().hex[:8]}"
        
        if not latest_message.timestamp:
            latest_message.timestamp = int(time.time())
        
        # 设置用户和会话信息
        if user_id:
            latest_message.user_id = user_id
        if session_id:
            latest_message.session_id = session_id
        
        return latest_message
    
    async def _store_message(self, message: Message, user_id: Optional[str] = None, session_id: Optional[str] = None):
        """存储消息到 Milvus"""
        if not self.milvus_store:
            return
        
        try:
            # 设置用户和会话信息
            if user_id:
                message.user_id = user_id
            if session_id:
                message.session_id = session_id
            
            await self.milvus_store.store_message(message)
        except Exception as e:
            print(f"[WARNING] 存储消息失败: {e}")
    
    async def _get_context_messages(
        self, 
        query_message: Message, 
        user_id: Optional[str] = None
    ) -> Optional[SearchResult]:
        """获取上下文消息"""
        if not self.milvus_store:
            print(f"[WARNING] Milvus 存储未初始化，无法获取上下文")
            return None
        
        try:
            print(f"[INFO] 正在milvus_store.py中执行 get_context_messages 方法")
            context_result = await self.milvus_store.get_context_messages(
                query_message=query_message,
                user_id=user_id
            )
            print(f"[INFO] 上下文消息获取成")
            return context_result
        except Exception as e:
            print(f"[WARNING] 获取上下文失败: {e}")
            return None
    
    def _build_enhanced_messages(
        self, 
        original_messages: List[Message], 
        context_result: Optional[SearchResult]
    ) -> List[Message]:
        """构建增强的消息列表（包含上下文）"""
        if not context_result or not context_result.messages:
            return original_messages
        
        # 构建上下文提示
        context_prompt = "以下是相关的历史对话上下文，请参考这些信息回答用户问题：\n\n"
        
        for i, ctx_msg in enumerate(context_result.messages[:10]):  # 限制上下文数量
            context_prompt += f"[上下文{i+1}] {ctx_msg.role}: {ctx_msg.get_text_content()[:200]}...\n"
        
        context_prompt += "\n现在请基于上述上下文回答用户的问题："
        
        # 在系统消息后插入上下文
        enhanced_messages = []
        system_msg_added = False
        
        for msg in original_messages:
            if msg.role == "system" and not system_msg_added:
                # 在系统消息后添加上下文
                enhanced_messages.append(msg)
                context_message = Message(
                    role="system",
                    content=context_prompt,
                    id=f"context_{uuid.uuid4().hex[:8]}"
                )
                enhanced_messages.append(context_message)
                system_msg_added = True
            elif msg.role != "system":
                enhanced_messages.append(msg)
                if not system_msg_added:
                    # 如果没有系统消息，在第一个用户消息前添加上下文
                    context_message = Message(
                        role="system",
                        content=context_prompt,
                        id=f"context_{uuid.uuid4().hex[:8]}"
                    )
                    enhanced_messages.insert(-1, context_message)
                    system_msg_added = True
            else:
                enhanced_messages.append(msg)
        
        return enhanced_messages
    
    async def chat_completion(self, request: ChatCompletionRequest) -> Union[ChatCompletionResponse, AsyncGenerator]:
        """调用增强的聊天完成接口"""
        try:
            # 提取最新的用户消息（忽略历史对话）
            latest_user_message = self._extract_latest_message(
                request.messages, 
                user_id=request.user_id, 
                session_id=request.session_id
            )
            
            # 处理并保存图片（如果有的话）
            if latest_user_message:
                print(f"[INFO] 处理最新用户消息: {latest_user_message.id}")
                latest_user_message = await self._process_and_save_images(latest_user_message)
                
                # 存储处理后的用户消息到 Milvus
                await self._store_message(
                    latest_user_message, 
                    user_id=request.user_id,
                    session_id=request.session_id
                )
                print(f"[INFO] 最新消息已存储到 Milvus")
            
            # 为了兼容现有逻辑，仍然获取最后一条用户消息作为查询
            user_messages = [msg for msg in request.messages if msg.role == "user"]
            last_user_message = user_messages[-1] if user_messages else latest_user_message
            
            # 准备消息列表
            messages_to_send = request.messages
            context_info = {"context_used": False, "context_count": 0}
            
            # 如果启用上下文检索且存在用户消息
            if request.use_context and last_user_message and self.milvus_store:
                print(f"[INFO] 正在检索上下文消息...")
                context_result = await self._get_context_messages(
                    last_user_message, 
                    user_id=request.user_id
                )
                print(f"[INFO] 检索到 {context_result.total_count if context_result else 0} 条上下文消息")
                
                if context_result and context_result.messages:
                    messages_to_send = self._build_enhanced_messages(request.messages, context_result)
                    context_info["context_used"] = True
                    context_info["context_count"] = len(context_result.messages)
                    print(f"[INFO] 使用了 {context_info['context_count']} 条上下文消息")
            
            # 转换消息格式
            api_messages = self._convert_messages_for_api(messages_to_send)
            
            print(f"[DEBUG] 发送到API的消息数量: {len(api_messages)}")
            
            # 构建请求数据
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            data = {
                "messages": api_messages,
                "model": self.model_name,
                "stream": request.stream,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "max_tokens": request.max_tokens
            }
            
            # 移除 None 值
            data = {k: v for k, v in data.items() if v is not None}
            
            if request.stream:
                return self._stream_response(headers, data, self.model_name, context_info, request)
            else:
                return await self._non_stream_response(headers, data, self.model_name, context_info, request)
                
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"API call failed: {str(e)}")
    
    async def _non_stream_response(
        self, 
        headers: Dict, 
        data: Dict, 
        model: str, 
        context_info: Dict,
        request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        """处理非流式响应"""
        print(f"[DEBUG] 发送请求到火山引擎API: {self.base_url}")
        
        response = requests.post(self.base_url, headers=headers, json=data)
        
        if response.status_code != 200:
            print(f"[ERROR] API错误响应: {response.text}")
            raise HTTPException(status_code=response.status_code, detail=response.text)
        
        result = response.json()
        assistant_content = result["choices"][0]["message"]["content"]
        
        # 创建助手消息并存储
        assistant_message = Message(
            role="assistant",
            content=assistant_content,
            id=f"assistant_{uuid.uuid4().hex[:8]}"
        )
        
        await self._store_message(
            assistant_message,
            user_id=request.user_id,
            session_id=request.session_id
        )
        
        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:29]}",
            object="chat.completion",
            created=int(time.time()),
            model=model,
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=assistant_message,
                    finish_reason="stop"
                )
            ],
            usage=Usage(
                prompt_tokens=result.get("usage", {}).get("prompt_tokens", 0),
                completion_tokens=result.get("usage", {}).get("completion_tokens", 0),
                total_tokens=result.get("usage", {}).get("total_tokens", 0)
            ),
            context_info=context_info
        )
    
    async def _stream_response(
        self, 
        headers: Dict, 
        data: Dict, 
        model: str, 
        context_info: Dict,
        request: ChatCompletionRequest
    ) -> AsyncGenerator:
        """处理流式响应"""
        try:
            print(f"[DEBUG] 发送流式请求到火山引擎API")
            
            response = requests.post(self.base_url, headers=headers, json=data, stream=True)
            
            if response.status_code != 200:
                error_msg = f"API Error: {response.status_code}"
                yield f"data: {json.dumps({'error': error_msg})}\n\n"
                return
            
            chat_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"
            created_time = int(time.time())
            assistant_content = ""
            
            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    
                    if line_str.startswith("data: "):
                        try:
                            raw_data = line_str[6:]
                            if raw_data.strip() == "[DONE]":
                                # 存储完整的助手消息
                                if assistant_content:
                                    assistant_message = Message(
                                        role="assistant",
                                        content=assistant_content,
                                        id=f"assistant_{uuid.uuid4().hex[:8]}"
                                    )
                                    await self._store_message(
                                        assistant_message,
                                        user_id=request.user_id,
                                        session_id=request.session_id
                                    )
                                
                                yield "data: [DONE]\n\n"
                                break
                            
                            data_json = json.loads(raw_data)
                            
                            if "choices" in data_json and len(data_json["choices"]) > 0:
                                choice = data_json["choices"][0]
                                delta = choice.get("delta", {})
                                content = delta.get("content", "")
                                
                                if content:
                                    assistant_content += content
                                
                                openai_chunk = {
                                    "id": chat_id,
                                    "object": "chat.completion.chunk",
                                    "created": created_time,
                                    "model": model,
                                    "choices": [{
                                        "index": 0,
                                        "delta": {"content": content} if content else {},
                                        "finish_reason": choice.get("finish_reason")
                                    }],
                                    "context_info": context_info
                                }
                                
                                yield f"data: {json.dumps(openai_chunk)}\n\n"
                                
                        except json.JSONDecodeError as e:
                            print(f"[WARNING] JSON解析错误: {e}")
                            continue
                            
        except Exception as e:
            error_msg = f"流式响应异常: {str(e)}"
            print(f"[ERROR] {error_msg}")
            yield f"data: {json.dumps({'error': error_msg})}\n\n"

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
try:
    enhanced_api = EnhancedVolcEngineAPI(
        api_key=
        milvus_host="localhost",
        milvus_port=19530
    )
except Exception as e:
    print(f"[ERROR] 初始化API客户端失败: {e}")
    enhanced_api = None

@app.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """增强版聊天补全端点，支持上下文检索"""
    if not enhanced_api:
        raise HTTPException(status_code=500, detail="API client not initialized")
    
    try:
        print(f"\n[INFO] 收到聊天请求:")
        print(f"[INFO] 模型: {request.model}")
        print(f"[INFO] 消息数量: {len(request.messages)}")
        print(f"[INFO] 使用上下文: {request.use_context}")
        print(f"[INFO] 用户ID: {request.user_id}")
        print(f"[INFO] 会话ID: {request.session_id}")
        
        if request.stream:
            # 流式响应
            generator = await enhanced_api.chat_completion(request)
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
            response = await enhanced_api.chat_completion(request)
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

@app.get("/health")
async def health_check():
    """健康检查"""
    milvus_status = "connected" if enhanced_api and enhanced_api.milvus_store else "disconnected"
    return {
        "status": "healthy",
        "milvus": milvus_status,
        "timestamp": time.time()
    }

if __name__ == "__main__":
    # 确保设置了 API Key
    if not os.getenv("VOLCENG_API_KEY"):
        print("警告: 请设置环境变量 VOLCENG_API_KEY")
    
    # 启动服务器
    uvicorn.run(
        app, 
        host="10.176.56.192",
        port=8001,  # 使用不同端口避免冲突
        log_level="info"
    )
