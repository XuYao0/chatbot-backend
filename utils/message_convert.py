from typing import List
from models.SearchMessage import SearchMessage
from models.ApiMessage import ApiMessage

def search_messages_to_api_messages(search_messages: List[SearchMessage]) -> List[ApiMessage]:
    """
    将 SearchMessage 列表转换为 ApiMessage 列表
    content 字段为 text_content 和 image_desc（如有）拼接
    """
    api_messages = []
    api_messages.append(ApiMessage(role="user", content="你是一个有帮助的助手。以下内容包括用户与你聊天的历史消息，以及用户最后一次发送的消息，请参考历史消息和你的知识来回答用户最近一次发送的消息"))
    for msg in search_messages:
        # 拼接文本内容和图片描述
        if msg.has_image and msg.image_desc:
            content = f"{msg.text_content}\n[图片描述]: {msg.image_desc}"
        else:
            content = msg.text_content
        api_messages.append(ApiMessage(role=msg.role, content=content))
    return api_messages