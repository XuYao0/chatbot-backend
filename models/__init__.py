"""
Models 包 - 数据模型定义
"""

# 导入所有模型类
from .ReceivedMessage import ReceivedMessage, ContentItem
from .StoredMessage import StoredMessage
from .SearchMessage import SearchMessage
from .ChatCompletionRequest import ChatCompletionRequest
from .ApiMessage import ApiMessage

# 定义包的公共接口
__all__ = [
    # 前端消息模型
    "ReceivedMessage",
    "ContentItem",
    
    # 存储消息模型
    "StoredMessage",
    
    # 搜索消息模型
    "SearchMessage",
    
    # API请求模型
    "ChatCompletionRequest",

    # 送给LLM API的消息类型
    "ApiMessage"
]