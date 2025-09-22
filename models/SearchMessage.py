"""
搜索消息模型定义 - 用于从Milvus查询返回的消息数据
"""
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import time
import json
from typing import List

class SearchMessage(BaseModel):
    """从Milvus搜索返回的消息模型"""
    id: str = Field(..., description="消息唯一ID")
    timestamp: float = Field(default_factory=time.time, description="消息时间戳")
    role: str = Field(..., description="消息角色: system, user, assistant")
    text_content: str = Field(default="", description="消息的文本内容")
    has_image: bool = Field(default=False, description="是否包含图片")
    image_path: Optional[str] = Field(default=None, description="图片路径（带pic_stored_path前缀）")
    image_desc: Optional[str] = Field(default=None, description="图片的描述列表")
    
    # 可选的相似度评分字段
    similarity_score: Optional[float] = Field(default=None, description="与查询的相似度评分")
    
    @classmethod
    def from_milvus_hit(cls, hit_data: Dict[str, Any], pic_stored_path: str = "/home/xuyao/data/bzchat_pic") -> "SearchMessage":
        """
        从Milvus搜索结果创建SearchMessage对象
        
        Args:
            hit_data: Milvus返回的hit数据
            pic_stored_path: 图片存储路径前缀

        Returns:
            SearchMessage对象
        """
        image_path = None
        if hit_data.get("has_image") and hit_data.get("image_path"):
            raw_path = hit_data.get("image_path")
            # 如果需要添加前缀且路径不是绝对路径
            if raw_path and pic_stored_path and not raw_path.startswith("/"):
                image_path = f"{pic_stored_path}/{raw_path}"
            else:
                image_path = raw_path
        
        return cls(
            id=hit_data.get("id", ""),
            timestamp=hit_data.get("timestamp", time.time()),
            role=hit_data.get("role", ""),
            text_content=hit_data.get("text_content", ""),
            has_image=hit_data.get("has_image", False),
            image_path=image_path,
            image_desc=hit_data.get("image_desc"),
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
            "image_desc": self.image_desc,
            "similarity_score": self.similarity_score
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SearchMessage":
        """从字典创建SearchMessage对象"""
        return cls(
            id=data.get("id", ""),
            timestamp=data.get("timestamp", time.time()),
            role=data.get("role", ""),
            text_content=data.get("text_content", ""),
            has_image=data.get("has_image", False),
            image_path=data.get("image_path"),
            image_desc=data.get("image_desc"),
            similarity_score=data.get("similarity_score")
        )