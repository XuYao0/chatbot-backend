from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union

class ContentItem(BaseModel):
    """消息内容项"""
    type: str = Field(..., description="内容类型: text 或 image_url")
    text: Optional[str] = Field(default=None, description="文本内容")
    image_url: Optional[str] = Field(default=None, description="图片URL信息")

class ReceivedMessage(BaseModel):
    """前端发送的消息模型"""
    role: str = Field(..., description="消息角色: system, user, assistant")
    content: Union[str, List[ContentItem]] = Field(..., description="消息内容，可以是字符串或内容项列表")