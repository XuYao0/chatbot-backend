"""存储的消息模型定义"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
import uuid
import time
import json

class StoredMessage(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="消息唯一ID")
    timestamp: float = Field(default_factory=time.time, description="消息时间戳")
    session_id: str = Field(..., description="会话ID")
    user_id: str = Field(..., description="用户ID")

    role: str = Field(..., description="消息角色: system, user, assistant")
    text_content: str = Field(default="", description="消息的文本内容")
    has_image: bool = Field(default=False, description="是否包含图片")
    image_paths: Optional[List[str]] = Field(default=None, description="消息中包含的图片文件路径列表")
    
    embedding: Optional[List[float]] = Field(default=None, description="消息的向量表示（CLIP模型生成）")
    
    def to_dict(self) -> Dict[str, Any]:
        """将消息对象转换为字典"""
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "role": self.role,
            "text_content": self.text_content,
            "has_image": self.has_image,
            "image_paths": json.dumps(self.image_paths) if self.image_paths else "[]",
            "embedding": self.embedding,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StoredMessage":
        """从字典创建消息对象"""
        return cls(
            session_id=data.get("session_id", ""),
            user_id=data.get("user_id", ""),
            role=data.get("role", ""),
            text_content=data.get("text_content", ""),
            has_image=data.get("has_image", False),
            image_paths=data.get("image_paths", []),
            embedding=data.get("embedding"),
        )