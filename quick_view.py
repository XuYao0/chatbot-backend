#!/usr/bin/env python3
"""
快速查看Milvus数据的简单脚本
"""

from pymilvus import connections, Collection, utility
import json

def quick_view():
    """快速查看数据库概况"""
    try:
        # 连接数据库
        connections.connect(uri="./milvus_lite.db")
        
        # 查看集合
        collections = utility.list_collections()
        print(f"📊 集合数量: {len(collections)}")
        
        for collection_name in collections:
            collection = Collection(collection_name)
            collection.load()
            count = collection.num_entities
            print(f"  • {collection_name}: {count} 条记录")
            # 检查索引
            print("    索引信息:")
            print(collection.indexes)
            
            # 如果有数据，显示最新的一条
            if count > 0:
                try:
                    results = collection.query(
                        expr="",
                        output_fields=["id", "role", "content", "timestamp"],
                        limit=1
                    )
                    if results:
                        latest = results[0]
                        print(f"    最新记录: {latest.get('role')} - {str(latest.get('content'))[:30]}...")
                except:
                    pass
    except Exception as e:
        print(f"❌ 错误: {e}")

if __name__ == "__main__":
    quick_view()