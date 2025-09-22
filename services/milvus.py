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

from models.StoredMessage import StoredMessage
from models.ReceivedMessage import ReceivedMessage  
from models.SearchMessage import SearchMessage
from utils.image_handler import ImageHandler
from utils.log import get_logger
from utils.image_describer import ImageDescriber

# 配置日志
logger = get_logger(__name__) 
pic_stored_path = "/home/xuyao/data/bzchat_pic"

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
            logger.info( f"成功加载 CLIP 多模态模型: {embedding_model}")
        except Exception as e:
            logger.error( f"加载 CLIP 模型失败: {e}")
            raise
        
        # 初始化图片处理器
        self.image_handler = ImageHandler(pic_stored_path)

        # 初始化图片描述器
        self.image_describer = ImageDescriber("/home/xuyao/data/Qwen2.5-VL-3B-Instruct")
        
        # 连接到 Milvus
        self._connect()
        
        # 创建或获取集合
        if utility.has_collection(self.collection_name):
            self.collection = Collection(self.collection_name)
            logger.info( f"集合 {self.collection_name} 已存在，直接使用")
        else:
            self._create_collection()
        
        # 创建索引
        self._create_index()
        
        # 加载集合到内存
        self.collection.load()
        logger.info( f"集合 {self.collection_name} 加载完成")
    
    def _connect(self):
        """连接到 Milvus 服务器"""
        try:
            # 使用 milvus-lite 进行本地测试
            connections.connect(
                host=self.host,
                port=self.port
                # uri="/home/xuyao/data/milvus_lite.db"
            )
            # print(utility.get_server_version()) 
            # logger.info( "成功连接到 Milvus Lite 本地数据库")
            logger.info( f"成功连接到 Milvus 服务器 {self.host}:{self.port}")
        except Exception as e:
            logger.error( f"连接 Milvus 失败: {e}")
            raise
    
    def _disconnect(self):
        """断开 Milvus 连接"""
        try:
            connections.disconnect("default")
            logger.info( "已断开 Milvus 连接")
        except Exception as e:
            logger.warning( f"断开连接时出现警告: {e}")
    
    def _create_collection(self):
        """创建多模态消息集合"""
        # 定义字段
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=64, is_primary=True),
            FieldSchema(name="timestamp", dtype=DataType.DOUBLE),
            FieldSchema(name="session_id", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="user_id", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="role", dtype=DataType.VARCHAR, max_length=32),
            FieldSchema(name="text_content", dtype=DataType.VARCHAR, max_length=10000),
            FieldSchema(name="has_image", dtype=DataType.BOOL),  # 是否包含图片
            FieldSchema(name="image_path", dtype=DataType.VARCHAR, max_length=1000),  # 图片路径（JSON字符串）
            FieldSchema(name="image_desc", dtype=DataType.VARCHAR, max_length=2000),  # 图片描述（JSON字符串）
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
        
        logger.info( f"成功创建多模态集合: {self.collection_name}")
    
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
                    logger.info( f"索引 '{index_name}' 已存在，跳过创建")
                    return
                else:
                    logger.info( f"正在删除已有索引 '{index_name}' 以重建...")
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

            logger.info( f"正在为字段 '{field_name}' 创建索引 '{index_name}'...")
            self.collection.create_index(
                field_name=field_name,
                index_params=index_params,
                index_name=index_name
            )

            logger.info( f"成功创建向量索引 '{index_name}' (type={index_params['index_type']}, metric={index_params['metric_type']})")

        except Exception as e:
            logger.error( f"创建索引 '{index_name}' 失败: {e}")
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
                embedding = self.embedding_model.encode([content], convert_to_numpy=True)
                return embedding[0].tolist()
            elif isinstance(content, Image.Image):
                # 图片向量化 - 这里需要使用CLIP模型的图片编码功能
                embedding = self.embedding_model.encode([content], convert_to_numpy=True)
                logger.info( "生成图片向量成功")
                return embedding[0].tolist()
            else:
                raise ValueError( f"不支持的内容类型: {type(content)}")
        except Exception as e:
            logger.error( f"生成向量失败: {e}")
            raise
    
    def _get_multimodal_embedding(self, text_content: Optional[str], image_path, fusion_strategy: str = "priority") -> List[float]:
        """
        获取多模态消息的向量表示
        对于包含图片的消息，优先使用图片生成向量
        对于纯文本消息，使用文本生成向量
        """
        try:
            text_embedding = None
            image_embedding = None
            
            # 分别获取两种向量
            if text_content:
                text_embedding = self._get_embedding(text_content)
                logger.info("生成文本向量")
            else:
                logger.info( "消息无文本内容")
            
            if image_path:
                image = self.image_handler.load_image(image_path)
                if image:
                    image_embedding = self._get_embedding(image)
                    logger.info( "生成图片向量")
            else:
                logger.info( "消息无图片内容")

            logger.info( f"融合策略: {fusion_strategy}")
            
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
            logger.error( f"生成多模态向量失败: {e}")
            raise
    
    async def store_message(self, message: ReceivedMessage) -> Optional[StoredMessage]:
        """
        存储一条多模态消息到 Milvus
        
        Args:
            message: 要存储的消息对象
            
        Returns:
            bool: 是否存储成功
        """
        try:
            # 检查消息是否有有效内容（文本或图片）
            has_image = False
            has_text = False
            text_content = None
            image_path = None
            image_desc = None
            if isinstance(message.content, str):
                text_content = message.content
            else:
                for item in message.content:
                    if item.type == "text" and item.text:
                        text_content = item.text
                    elif item.type == "image_url" and item.image_url:
                        base64_data = item.image_url
                        try:
                            image_path, _ = self.image_handler.save_image_from_base64(
                                base64_data=base64_data,
                                user_id="admin",
                                session_id="total"
                            )
                            has_image = True
                            logger.info( f"保存图片成功: {image_path}")
                        except Exception as e:
                            logger.error( f"保存图片失败: {e}")

            if text_content:
                has_text = bool(text_content.strip())
            if not has_text and not has_image:
                logger.warning("from milvus: "f"消息没有有效内容，跳过存储")
                return None
            
            # 使用多模态方法生成向量
            embedding = self._get_multimodal_embedding(text_content, image_path)

            # 生成图片描述（如果有图片）
            if has_image and image_path:
                try:
                    image_desc = self.image_describer.describe_image(
                        image_path=pic_stored_path + "/" + image_path,
                        context=text_content or "",
                    )
                    logger.info( f"生成图片描述成功: {image_desc}")
                except Exception as e:
                    logger.error( f"生成图片描述失败: {e}")
                    image_desc = None

            # 准备数据（按schema字段顺序）
            data = StoredMessage.from_dict({
                "session_id": "total",
                "user_id": "admin",
                "role": message.role,
                "text_content": text_content or "",
                "has_image": has_image,
                "image_path": image_path if has_image else "",
                "image_desc": image_desc if has_image else "",
                "embedding": embedding
            })

            # 插入数据
            mr = self.collection.insert(data.to_dict())
            
            # 立即刷新以确保数据持久化
            self.collection.flush()
            
            logger.info( f"成功存储多模态消息: {data.id}")
            return data
            
        except Exception as e:
            logger.error( f"存储消息失败: {e}")
            return None
    
    async def get_recent_messages(self, limit: int = 20, user_id: Optional[str] = "admin", session_id: Optional[str] = "total") -> List[SearchMessage]:
        """
        获取最近的消息
        
        Args:
            limit: 返回消息数量限制
            user_id: 用户ID过滤（可选）
            session_id: 会话ID过滤（可选）
            
        Returns:
            List[SearchMessage]: 最近的消息列表
        """
        try:
            # 构建过滤表达式
            filter_expr = self._build_filter_expr(user_id, session_id)
            
            # 执行查询，按时间戳降序排列
            results = self.collection.query(
                expr=filter_expr if filter_expr else "",
                output_fields=["id", "role", "text_content", "timestamp", "session_id", "user_id", "image_path", "has_image", "image_desc"],
                limit=limit
            )
            
            # 按时间戳排序
            results.sort(key=lambda x: x["timestamp"], reverse=True)
            
            # 转换为 SearchMessage 对象
            messages = []
            for hit in results:  # self.collection.query() 返回的是一个 字典列表（List[Dict]），而不是嵌套列表。
                # 创建SearchMessage对象
                search_message = SearchMessage.from_milvus_hit(hit, pic_stored_path)
                messages.append(search_message)
            
            logger.info( f"获取到 {len(messages)} 条最近消息")
            # 调试代码
            for m in messages:
                print(f"[Recent] {m.role}: {m.text_content[:50]}... (ID: {m.id})")
            return messages
            
        except Exception as e:
            logger.error( f"获取最近消息失败: {e}")
            return []
    
    async def search_similar_messages(
        self, 
        query_message: StoredMessage, 
        limit: int = 10,
    ) -> List[SearchMessage]:
        try:
            # 输入验证
            if not query_message:
                logger.error( "查询消息不能为空")
                return []

            # 获取多模态嵌入 - 需要提供text_content和image_path
            query_embedding = query_message.embedding
            if not query_embedding:
                logger.warning( "无法生成查询嵌入")
                return []

            # 构建过滤表达式
            filter_expr = self._build_filter_expr(query_message.user_id, query_message.session_id)

            # 执行搜索
            search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
            results = await asyncio.get_event_loop().run_in_executor(
                None,
                self._do_search,
                query_embedding,
                search_params,
                limit,
                filter_expr,
                ["id", "role", "text_content", "timestamp", "session_id", "user_id", "image_path", "has_image", "image_desc"]
            )

            # results = await asyncio.wrap_future(results)
            # results = await results

            # 转换为 SearchMessage 对象
            messages = []
            for hit in results[0]:
                # 创建SearchMessage对象
                search_message = SearchMessage.from_milvus_hit(hit.get("entity"), pic_stored_path)
                messages.append(search_message)

            logger.info( f"找到 {len(messages)} 条相似消息")
            # 调试代码
            for m in messages:
                print(f"[Similar] {m.role}: {m.text_content[:50]}... (ID: {m.id})")
            return messages

        except MilvusException as e:
            logger.error( f"Milvus 搜索失败: {e}")
            raise
        except Exception as e:
            logger.warning( f"搜索过程中发生异常: {e}")
            return []
        
    def _do_search(self, query_embedding, search_params, limit, filter_expr, output_fields):
        return self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=limit,
            expr=filter_expr,
            output_fields=output_fields
        )

    def _build_filter_expr(self, user_id: Optional[str], session_id: Optional[str]) -> str:
        conditions = []
        if user_id:
            conditions.append(f"user_id == '{user_id}'")
        if session_id:
            conditions.append(f"session_id == '{session_id}'")
        return " and ".join(conditions)

    async def get_context_messages(
        self, 
        query_message: StoredMessage, 
    ) -> List[SearchMessage]:
        """
        获取上下文消息：最近20条 + 相似度最高的10条，去重后返回
        
        Args:
            query_message: 查询消息
            user_id: 用户ID过滤（可选）
            session_id: 会话ID过滤（可选）
            
        Returns:
            List[SearchMessage]: 去重后的消息列表
        """
        try:
            # 并行获取最近消息和相似消息
            recent_messages = await self.get_recent_messages(limit=20, user_id=query_message.user_id, session_id=query_message.session_id)
            similar_messages = await self.search_similar_messages(query_message=query_message, limit=10)
            
            # 合并并去重
            all_messages = recent_messages + similar_messages
            seen_ids = set()
            unique_messages = []
            
            for message in all_messages:
                if message.id not in seen_ids:
                    seen_ids.add(message.id)
                    unique_messages.append(message)
            
            # 按时间戳升序排序
            unique_messages.sort(key=lambda x: x.timestamp, reverse=False)
            
            logger.info(f"获取到 {len(unique_messages)} 条上下文消息（去重后）")
            return unique_messages
            
        except Exception as e:
            logger.error(f"获取上下文消息失败: {e}")
            return []
    
    def __del__(self):
        """析构函数，清理资源"""
        try:
            self._disconnect()
        except:
            pass