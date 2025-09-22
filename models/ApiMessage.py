"""规定送给openai api风格的消息格式"""

from pydantic import BaseModel, Field

class ApiMessage(BaseModel):
    role: str = Field(..., description="消息角色: system, user, assistant")
    content: str = Field(..., description="消息内容")

    def to_dict(self) -> dict:
        """转换为字典格式"""
        return {"role": self.role, "content": self.content}