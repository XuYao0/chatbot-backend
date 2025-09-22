import time
import uuid
import asyncio
import json
import requests
from typing import List, Dict, Any, Optional, Union, AsyncGenerator
from pydantic import BaseModel, Field
import os
from utils.log import get_logger
from models.ApiMessage import ApiMessage

class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: Optional[Dict[str, Any]] = None
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
    context_info: Optional[Dict[str, Any]] = None

logger = get_logger(__name__)

class DeepSeekAPI:
    """
    DeepSeek API 调用类
    """
    def __init__(
        self, 
        api_key: Optional[str] = None, 
        base_url: Optional[str] = None, 
        model_name: str = "deepseek-r1-250528",
    ):
        self.api_key = api_key or os.getenv("VOLCENG_API_KEY")
        self.base_url = base_url or "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
        self.model_name = model_name
        
        if not self.api_key:
            logger.error("API key is required for DeepSeekAPI")
            raise ValueError("API key is required")
    
    async def chat_completion(
        self,
        messages: List[ApiMessage],
        max_tokens: int = 3000,
        temperature: float = 0.7,
        stream: bool = False,
        model: Optional[str] = None
    ) -> Union[ChatCompletionResponse, AsyncGenerator]:
        """
        接收 ApiMessage 格式数据，调用模型推理，返回响应对象
        """
        model_name = model or self.model_name
        
        # 转换 ApiMessage 为字典格式
        api_messages = [msg.to_dict() for msg in messages]
        
        logger.info(f"[DEBUG] 发送到API的消息数量: {len(api_messages)}")
        
        # 构建请求数据
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        data = {
            "messages": api_messages,
            "model": model_name,
            "stream": stream,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        # 移除 None 值
        data = {k: v for k, v in data.items() if v is not None}

        if stream:
            return self._stream_response(headers, data, model_name)
        else:
            return await self._non_stream_response(headers, data, model_name)
    
    async def _non_stream_response(
        self, 
        headers: Dict, 
        data: Dict, 
        model: str
    ) -> ChatCompletionResponse:
        """处理非流式响应"""
        logger.info(f"[DEBUG] 发送请求到DeepSeek API: {self.base_url}")
        
        response = requests.post(self.base_url, headers=headers, json=data)
        
        if response.status_code != 200:
            logger.error(f"[ERROR] 火山引擎API错误响应: {response.text}")
            raise Exception(f"API Error: {response.status_code} - {response.text}")
        
        result = response.json()
        assistant_content = result["choices"][0]["message"]["content"]
        
        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:29]}",
            object="chat.completion",
            created=int(time.time()),
            model=model,
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message={"role": "assistant", "content": assistant_content},
                    finish_reason="stop"
                )
            ],
            usage=Usage(
                prompt_tokens=result.get("usage", {}).get("prompt_tokens", 0),
                completion_tokens=result.get("usage", {}).get("completion_tokens", 0),
                total_tokens=result.get("usage", {}).get("total_tokens", 0)
            ),
            context_info=None
        )
    
    async def _stream_response(
        self, 
        headers: Dict, 
        data: Dict, 
        model: str
    ) -> AsyncGenerator:
        """处理流式响应"""
        try:
            logger.info(f"[DEBUG] 发送流式请求到DeepSeek API")
            
            response = requests.post(self.base_url, headers=headers, json=data, stream=True)
            
            if response.status_code != 200:
                error_msg = f"火山引擎API Error: {response.status_code}"
                logger.error(f"[ERROR] {error_msg} - {response.text}")
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
                                    }]
                                }
                                
                                yield f"data: {json.dumps(openai_chunk)}\n\n"
                                
                        except json.JSONDecodeError as e:
                            logger.warning(f"[WARNING] JSON解析错误: {e}")
                            continue
                            
        except Exception as e:
            error_msg = f"流式响应异常: {str(e)}"
            logger.error(f"[ERROR] {error_msg}")
            yield f"data: {json.dumps({'error': error_msg})}\n\n"