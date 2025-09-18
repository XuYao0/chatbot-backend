"""
通用数据模型定义
包含消息、用户等通用数据结构
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import uuid
import time
import json


class ChatSession(BaseModel):
    """聊天会话模型"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="会话ID")
    user_id: str = Field(..., description="用户ID")
    title: str = Field(default="新对话", description="会话标题")
    created_at: float = Field(default_factory=time.time, description="创建时间")
    updated_at: float = Field(default_factory=time.time, description="更新时间")
    message_count: int = Field(default=0, description="消息数量")

class SearchResult(BaseModel):
    """搜索结果模型"""
    messages: List[Message] = Field(default_factory=list, description="搜索到的消息列表")
    total_count: int = Field(default=0, description="总数量")
    query_message: Optional[Message] = Field(default=None, description="查询消息")
    
    def deduplicate(self) -> "SearchResult":
        """去重操作"""
        seen_ids = set()
        unique_messages = []
        
        for message in self.messages:
            if message.id not in seen_ids:
                seen_ids.add(message.id)
                unique_messages.append(message)
        
        return SearchResult(
            messages=unique_messages,
            total_count=len(unique_messages),
            query_message=self.query_message
        )
