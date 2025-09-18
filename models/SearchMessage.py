"""
搜索消息模型定义 - 用于从Milvus查询返回的消息数据
"""
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import time
import json

class SearchMessage(BaseModel):
    """从Milvus搜索返回的消息模型"""
    id: str = Field(..., description="消息唯一ID")
    timestamp: float = Field(default_factory=time.time, description="消息时间戳")
    role: str = Field(..., description="消息角色: system, user, assistant")
    text_content: str = Field(default="", description="消息的文本内容")
    has_image: bool = Field(default=False, description="是否包含图片")
    image_path: Optional[str] = Field(default=None, description="图片路径（带pic_stored_path前缀）")
    
    # 可选的相似度评分字段
    similarity_score: Optional[float] = Field(default=None, description="与查询的相似度评分")
    
    @classmethod
    def from_milvus_hit(cls, hit_data: Dict[str, Any], pic_stored_path: str = "/home/xuyao/data/bzchat_pic") -> "SearchMessage":
        """
        从Milvus搜索结果创建SearchMessage对象
        
        Args:
            hit_data: Milvus返回的hit数据，例如:
                {
                    "id": "msg_12345678",
                    "role": "user", 
                    "text_content": "请帮我分析这张图片中的内容",
                    "timestamp": 1703030400.123,
                    "session_id": "session_abc123",
                    "user_id": "user_001",
                    "has_image": True,
                    "image_paths": '["uploads/2023/12/19/chat_img_001.jpg"]',
                    "score": 0.8756  # Milvus相似度评分
                }
            pic_stored_path: 图片存储路径前缀，例如 "/home/xuyao/data/bzchat_pic"
            
        Returns:
            SearchMessage对象
        """
        # 处理图片路径，添加前缀
        image_path = None
        if hit_data.get("has_image") and hit_data.get("image_paths"):
            # 假设image_paths是JSON字符串或列表
            paths = hit_data.get("image_paths", "[]")
            if isinstance(paths, str):
                try:
                    paths = json.loads(paths)
                except:
                    paths = []
            
            if paths and len(paths) > 0:
                # 添加pic_stored_path前缀
                raw_path = paths[0]  # 取第一个图片路径
                image_path = f"{pic_stored_path}/{raw_path}" if pic_stored_path else raw_path
        
        return cls(
            id=hit_data.get("id", ""),
            timestamp=hit_data.get("timestamp", time.time()),
            role=hit_data.get("role", ""),
            text_content=hit_data.get("text_content", ""),
            has_image=hit_data.get("has_image", False),
            image_path=image_path,
            similarity_score=hit_data.get("score")  # Milvus返回的相似度评分
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "role": self.role,
            "text_content": self.text_content,
            "has_image": self.has_image,
            "image_path": self.image_path,
            "similarity_score": self.similarity_score
        }