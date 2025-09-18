"""
Milvus 向量数据库消息存储工具类（多模态版本）
实现消息的存储、检索和相似度搜索功能，支持 CLIP 模型
"""
import logging
import json
import time
from datetime import datetime
from typing import List, Optional, Dict, Any, Union
from PIL import Image
from pymilvus import (
    connections, Collection, CollectionSchema, FieldSchema, DataType,
    utility, MilvusException
)
from pymilvus import SearchFuture
from sentence_transformers import SentenceTransformer
import asyncio

from models import Message, SearchResult
from image_handler import ImageHandler

# 配置日志
logger = logging.getLogger(__name__)


class MilvusMessageStore:
    """Milvus 消息存储类（支持多模态）"""
    
    def __init__(
        self, 
        host: str = "localhost", 
        port: int = 19530,
        collection_name: str = "multimodal_chat_messages",
        embedding_model: str = "clip-ViT-L-14"
    ):
        """
        初始化 Milvus 连接和集合
        
        Args:
            host: Milvus 服务器地址
            port: Milvus 服务器端口
            collection_name: 集合名称
            embedding_model: CLIP 多模态嵌入模型名称
        """
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.embedding_dim = 768  # clip-ViT-L-14 的向量维度
        
        # 初始化多模态嵌入模型 (CLIP)
        try:
            self.embedding_model = SentenceTransformer(embedding_model)
            logger.info(f"成功加载 CLIP 多模态模型: {embedding_model}")
        except Exception as e:
            logger.error(f"加载 CLIP 模型失败: {e}")
            raise
        
        # 初始化图片处理器
        self.image_handler = ImageHandler()
        
        # 连接到 Milvus
        self._connect()
        
        # 创建或获取集合
        if utility.has_collection(self.collection_name):
            self.collection = Collection(self.collection_name)
            logger.info(f"集合 {self.collection_name} 已存在，直接使用")
        else:
            self._create_collection()
        
        # 创建索引
        self._create_index()
        
        # 加载集合到内存
        self.collection.load()
        logger.info(f"集合 {self.collection_name} 加载完成")
    
    def _connect(self):
        """连接到 Milvus 服务器"""
        try:
            # 使用 milvus-lite 进行本地测试
            connections.connect(
                uri="./milvus_lite.db"  # 使用本地文件数据库
            )
            logger.info("成功连接到 Milvus Lite 本地数据库")
        except Exception as e:
            logger.error(f"连接 Milvus 失败: {e}")
            raise
    
    def _disconnect(self):
        """断开 Milvus 连接"""
        try:
            connections.disconnect("default")
            logger.info("已断开 Milvus 连接")
        except Exception as e:
            logger.warning(f"断开连接时出现警告: {e}")
    
    def _create_collection(self):
        """创建多模态消息集合"""
        # 定义字段
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=64, is_primary=True),
            FieldSchema(name="role", dtype=DataType.VARCHAR, max_length=32),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=10000),
            FieldSchema(name="text_content", dtype=DataType.VARCHAR, max_length=10000),
            FieldSchema(name="timestamp", dtype=DataType.DOUBLE),
            FieldSchema(name="session_id", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="user_id", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="image_paths", dtype=DataType.VARCHAR, max_length=1000),  # 图片路径（JSON字符串）
            FieldSchema(name="has_image", dtype=DataType.BOOL),  # 是否包含图片
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim)  # CLIP向量
        ]
        
        # 创建集合模式
        schema = CollectionSchema(
            fields=fields,
            description="多模态聊天消息存储集合（支持文本+图片）"
        )
        
        # 创建集合
        self.collection = Collection(
            name=self.collection_name,
            schema=schema
        )
        
        logger.info(f"成功创建多模态集合: {self.collection_name}")
    
    def _create_index(self, index_name: str = "embedding_idx", recreate: bool = False):
        """
        创建向量索引（支持指定名称，避免歧义）

        Args:
            index_name (str): 索引名称，默认为 "embedding_idx"
            recreate (bool): 是否强制删除并重建索引
        """
        field_name = "embedding"
        
        try:
            # 获取当前所有索引
            existing_indexes = self.collection.indexes
            has_index = any(idx.index_name == index_name for idx in existing_indexes)

            if has_index:
                if not recreate:
                    logger.info(f"索引 '{index_name}' 已存在，跳过创建")
                    return
                else:
                    logger.info(f"正在删除已有索引 '{index_name}' 以重建...")
                    self.collection.drop_index(
                        field_name=field_name,
                        index_name=index_name
                    )

            # 定义索引参数（使用 COSINE 相似度）
            index_params = {
                "index_type": "IVF_FLAT",      # 适合高精度搜索
                "metric_type": "COSINE",       # 余弦相似度，适合向量归一化场景
                "params": {"nlist": 128}       # 聚类中心数量，一般设为数据量的1/1000~1/500
            }

            logger.info(f"正在为字段 '{field_name}' 创建索引 '{index_name}'...")
            self.collection.create_index(
                field_name=field_name,
                index_params=index_params,
                index_name=index_name
            )

            logger.info(f"成功创建向量索引 '{index_name}' (type={index_params['index_type']}, metric={index_params['metric_type']})")

        except Exception as e:
            logger.error(f"创建索引 '{index_name}' 失败: {e}")
            raise
    
    def _get_embedding(self, content: Union[str, Image.Image]) -> List[float]:
        """
        获取文本或图片的 CLIP 向量表示
        
        Args:
            content: 文本字符串或PIL图片对象
            
        Returns:
            List[float]: 向量表示
        """
        try:
            if isinstance(content, str):
                # 文本向量化
                embedding = self.embedding_model.encode(content, convert_to_numpy=True)
            elif isinstance(content, Image.Image):
                # 图片向量化
                embedding = self.embedding_model.encode(content, convert_to_numpy=True)
            else:
                raise ValueError(f"不支持的内容类型: {type(content)}")
            
            return embedding.tolist()
        except Exception as e:
            logger.error(f"生成向量失败: {e}")
            raise
    
    def _get_multimodal_embedding(self, message: Message, fusion_strategy: str = "priority") -> List[float]:
        """
        获取多模态消息的向量表示
        对于包含图片的消息，优先使用图片生成向量
        对于纯文本消息，使用文本生成向量
        """
        try:
            text_embedding = None
            image_embedding = None
            
            # 分别获取两种向量
            text_content = message.get_text_content()
            if text_content.strip():
                text_embedding = self._get_embedding(text_content)
                logger.info("生成文本向量")
            else:
                logger.info("消息无文本内容")
            
            if message.has_image and message.image_paths:
                image = self.image_handler.load_image(message.image_paths[0])
                if image:
                    image_embedding = self._get_embedding(image)
                    logger.info("生成图片向量")
            else:
                logger.info("消息无图片内容")

            logger.info(f"融合策略: {fusion_strategy}")
            
            # 根据策略融合
            if fusion_strategy == "average" and text_embedding and image_embedding:
                # 平均融合：保持768维
                return [(t + i) / 2 for t, i in zip(text_embedding, image_embedding)]
            elif fusion_strategy == "weighted" and text_embedding and image_embedding:
                # 加权融合：可调节重要性
                alpha, beta = 0.6, 0.4
                return [alpha * t + beta * i for t, i in zip(text_embedding, image_embedding)]
            else:
                # 回退到当前策略
                return image_embedding or text_embedding or [0.0] * 768
            
        except Exception as e:
            logger.error(f"生成多模态向量失败: {e}")
            raise
    
    async def store_message(self, message: Message) -> bool:
        """
        存储一条多模态消息到 Milvus
        
        Args:
            message: 要存储的消息对象
            
        Returns:
            bool: 是否存储成功
        """
        try:
            # 检查消息是否有有效内容（文本或图片）
            text_content = message.get_text_content()
            has_text = bool(text_content.strip())
            has_image = message.has_image and message.image_paths
            
            if not has_text and not has_image:
                logger.warning(f"消息 {message.id} 没有有效内容，跳过存储")
                return False
            
            # 使用多模态方法生成向量
            embedding = self._get_multimodal_embedding(message)
            message.embedding = embedding
            
            # 准备数据（按schema字段顺序）
            data = [
                [message.id],
                [message.role],
                [message.content if isinstance(message.content, str) else str(message.content)],
                [text_content],
                [message.timestamp],
                [message.session_id],
                [message.user_id],
                [json.dumps(message.image_paths) if message.image_paths else "[]"],
                [message.has_image],
                [embedding]
            ]
            
            # 插入数据
            mr = self.collection.insert(data)
            
            # 立即刷新以确保数据持久化
            self.collection.flush()
            
            logger.info(f"成功存储多模态消息: {message.id}")
            return True
            
        except Exception as e:
            logger.error(f"存储消息失败: {e}")
            return False
    
    async def get_recent_messages(self, limit: int = 20, user_id: Optional[str] = None, session_id: Optional[str] = None) -> List[Message]:
        """
        获取最近的消息
        
        Args:
            limit: 返回消息数量限制
            user_id: 用户ID过滤（可选）
            session_id: 会话ID过滤（可选）
            
        Returns:
            List[Message]: 最近的消息列表
        """
        try:
            # 构建过滤表达式
            filter_expr = self._build_filter_expr(user_id, session_id)
            
            # 执行查询，按时间戳降序排列
            results = self.collection.query(
                expr=filter_expr if filter_expr else "",
                output_fields=["id", "role", "content", "text_content", "timestamp", "session_id", "user_id", "image_paths", "has_image"],
                limit=limit
            )
            
            # 按时间戳排序
            results.sort(key=lambda x: x["timestamp"], reverse=True)
            
            # 转换为 Message 对象
            messages = []
            for result in results:
                # 解析图片路径
                result["image_paths"] = json.loads(result.get("image_paths", "[]"))
                
                message = Message.from_dict(result)
                messages.append(message)
            
            logger.info(f"获取到 {len(messages)} 条最近消息")
            print(f"[INFO] 获取到 {len(messages)} 条最近消息")
            return messages
            
        except Exception as e:
            logger.error(f"获取最近消息失败: {e}")
            return []
    
    async def search_similar_messages(
        self, 
        query_message: Message, 
        limit: int = 10,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> List[Message]:
        try:
            # 输入验证
            if not query_message:
                logger.error("查询消息不能为空")
                return []

            print(f"[INFO] 在 MilvusMessageStore.search_similar_messages 中搜索相似消息")

            # 获取多模态嵌入
            query_embedding = self._get_multimodal_embedding(query_message)
            if not query_embedding:
                logger.warning("无法生成查询嵌入")
                return []

            # 构建过滤表达式
            filter_expr = self._build_filter_expr(user_id, session_id)

            # 执行搜索
            results = self.collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param={"metric_type": "COSINE", "params": {"nprobe": 10}},
                limit=limit,
                expr=filter_expr,
                output_fields=["id", "role", "content", "text_content", "timestamp", "session_id", "user_id", "image_paths", "has_image"]
            )

            # 处理异步结果
            if isinstance(results, SearchFuture):
                results = await asyncio.wrap_future(results)

            # 处理结果
            messages = []
            for hit in results[0]:
                if hit.entity["id"] == query_message.id:
                    continue
                message = self._hit_to_message(hit)
                messages.append(message)

            logger.info(f"找到 {len(messages)} 条相似消息")
            return messages

        except MilvusException as e:
            logger.error(f"Milvus 搜索失败: {e}")
            raise
        except Exception as e:
            logger.warning(f"搜索过程中发生异常: {e}")
            return []

    def _build_filter_expr(self, user_id: Optional[str], session_id: Optional[str]) -> str:
        conditions = []
        if user_id:
            conditions.append(f"user_id == '{user_id}'")
        if session_id:
            conditions.append(f"session_id == '{session_id}'")
        return " and ".join(conditions)

    def _hit_to_message(self, hit) -> Message:
        data = {
            "id": hit.entity["id"],
            "image_paths": json.loads(hit.entity.get("image_paths", "[]")),
            "role": hit.entity["role"],
            "content": hit.entity["content"],
            "text_content": hit.entity["text_content"],
            "timestamp": hit.entity["timestamp"],
            "session_id": hit.entity["session_id"],
            "user_id": hit.entity["user_id"],
            "has_image": hit.entity["has_image"]
        }
        return Message.from_dict(data)
    
    async def get_context_messages(
        self, 
        query_message: Message, 
        user_id: Optional[str] = None
    ) -> SearchResult:
        """
        获取上下文消息：最近20条 + 相似度最高的10条，去重后返回
        
        Args:
            query_message: 查询消息
            user_id: 用户ID过滤（可选）
            
        Returns:
            Dict: 包含去重后的消息列表和元信息
        """
        try:
            # 并行获取最近消息和相似消息
            print(f"[INFO] 在 MilvusMessageStore.get_context_messages 中获取上下文消息")
            recent_messages = await self.get_recent_messages(limit=20, user_id=user_id)
            similar_messages = await self.search_similar_messages(
                query_message=query_message, 
                limit=10, 
                user_id=user_id
            )
            
            # 合并并去重
            all_messages = recent_messages + similar_messages
            seen_ids = set()
            unique_messages = []
            
            for message in all_messages:
                if message.id not in seen_ids:
                    seen_ids.add(message.id)
                    unique_messages.append(message)
            
            # 按时间戳排序
            unique_messages.sort(key=lambda x: x.timestamp, reverse=True)
            
            # 创建SearchResult对象
            result = SearchResult(
                messages=unique_messages,
                total_count=len(unique_messages),
                query_message=query_message
            )
            
            logger.info(f"获取到 {len(unique_messages)} 条上下文消息（去重后）")
            return result
            
        except Exception as e:
            logger.error(f"获取上下文消息失败: {e}")
            return SearchResult(
                messages=[],
                total_count=0,
                query_message=query_message
            )
    
    def __del__(self):
        """析构函数，清理资源"""
        try:
            self._disconnect()
        except:
            pass

if __name__ == "__main__":
    try:
        store = MilvusMessageStore()
    except Exception as e:
        print(f"初始化 MilvusMessageStore 失败: {e}")
        exit(1)
    
    # 这里可以添加一些测试代码
    try:
        # 创建一个测试消息
        test_message = Message(
            id="test123",
            role="user",
            content="这是一个包含图片的测试消息。",
            timestamp=time.time(),
            session_id="session_test",
            user_id="user_test",
            image_paths=["../test_image.jpg"],
            has_image=True
        )
        
        # 存储消息
        import asyncio
        asyncio.run(store.store_message(test_message))
        
        # 获取最近消息
        recent_msgs = asyncio.run(store.get_recent_messages(limit=5))
        print(f"最近消息数量: {len(recent_msgs)}")
        
        # 搜索相似消息
        similar_msgs = asyncio.run(store.search_similar_messages(test_message, limit=5))
        print(f"相似消息数量: {len(similar_msgs)}")
        
    except Exception as e:
        print(f"测试过程中发生错误: {e}")
    finally:
        del store