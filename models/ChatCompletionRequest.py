"""接收前端传递来的数据"""
from pydantic import BaseModel, Field
from typing import List
from .ReceivedMessage import ReceivedMessage

class ChatCompletionRequest(BaseModel):
    """前端聊天完成请求模型"""
    model: str = Field(default="gpt-3.5-turbo", description="模型名称")
    temperature: float = Field(default=0.8, description="温度参数")
    top_p: float = Field(default=0.9, description="top_p参数")
    frequency_penalty: float = Field(default=0, description="频率惩罚")
    presence_penalty: float = Field(default=0, description="存在惩罚")
    max_tokens: int = Field(default=3000, description="最大token数")
    n: int = Field(default=1)
    stream: bool = Field(default=True, description="是否流式输出")
    messages: List[ReceivedMessage] = Field(..., description="消息列表")