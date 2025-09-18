#!/usr/bin/env python3
"""
Milvus数据查看工具
用于查看Milvus Lite数据库中的数据内容
"""

from pymilvus import connections, Collection, utility
import json
from datetime import datetime
import sys

def format_timestamp(timestamp):
    """格式化时间戳"""
    try:
        if timestamp:
            return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
        return "未知时间"
    except:
        return str(timestamp)

def truncate_text(text, max_length=50):
    """截断长文本"""
    if not text:
        return ""
    text_str = str(text)
    if len(text_str) <= max_length:
        return text_str
    return text_str[:max_length] + "..."

def main():
    try:
        print("=== Milvus数据库查看工具 ===")
        
        # 连接到Milvus Lite数据库
        connections.connect(uri="./milvus_lite.db")
        print("✅ 成功连接到 milvus_lite.db")
        
        # 1. 查看所有集合
        print("\n📋 数据库中的所有集合:")
        collections = utility.list_collections()
        if not collections:
            print("  ❌ 数据库中没有集合")
            return
        
        for i, collection_name in enumerate(collections, 1):
            print(f"  {i}. {collection_name}")
        
        # 2. 查看每个集合的详细信息
        for collection_name in collections:
            print(f"\n" + "="*60)
            print(f"🔍 集合: {collection_name}")
            print("="*60)
            
            try:
                collection = Collection(collection_name)
                collection.load()
                
                # 集合统计信息
                num_entities = collection.num_entities
                print(f"📊 记录数量: {num_entities}")
                
                # 查看集合结构
                print(f"\n🏗️  集合结构:")
                for field in collection.schema.fields:
                    field_info = f"  • {field.name}: {field.dtype}"
                    if field.is_primary:
                        field_info += " (主键)"
                    if hasattr(field, 'max_length') and field.max_length:
                        field_info += f" (最大长度: {field.max_length})"
                    if hasattr(field, 'dim') and field.dim:
                        field_info += f" (维度: {field.dim})"
                    print(field_info)
                
                # 如果有数据，查看前几条记录
                if num_entities > 0:
                    print(f"\n📄 数据预览 (前5条记录):")
                    try:
                        # 获取所有字段名（除了向量字段）
                        output_fields = []
                        for field in collection.schema.fields:
                            if field.dtype.name not in ['FLOAT_VECTOR', 'BINARY_VECTOR']:
                                output_fields.append(field.name)
                        
                        results = collection.query(
                            expr="",  # 空条件查询所有
                            output_fields=output_fields,
                            limit=5
                        )
                        
                        for i, result in enumerate(results, 1):
                            print(f"\n  📝 记录 {i}:")
                            for field_name in output_fields:
                                value = result.get(field_name)
                                
                                # 特殊处理不同类型的字段
                                if field_name == 'timestamp':
                                    display_value = format_timestamp(value)
                                elif field_name == 'image_paths':
                                    # 尝试解析JSON
                                    try:
                                        if isinstance(value, str):
                                            paths = json.loads(value)
                                            display_value = f"图片数量: {len(paths)}, 路径: {paths}"
                                        else:
                                            display_value = str(value)
                                    except:
                                        display_value = str(value)
                                elif field_name in ['content', 'text_content']:
                                    display_value = truncate_text(value, 80)
                                else:
                                    display_value = str(value)
                                
                                print(f"    {field_name}: {display_value}")
                    
                    except Exception as e:
                        print(f"    ❌ 查询数据失败: {e}")
                
                # 查看索引信息
                try:
                    print(f"\n🗂️  索引信息:")
                    indexes = collection.indexes
                    if indexes:
                        for index in indexes:
                            print(f"  • 字段: {index.field_name}")
                            print(f"    索引类型: {index.params.get('index_type', '未知')}")
                            print(f"    度量类型: {index.params.get('metric_type', '未知')}")
                            if hasattr(index, 'index_name'):
                                print(f"    索引名称: {index.index_name}")
                    else:
                        print("  暂无索引")
                except Exception as e:
                    print(f"  ❌ 获取索引信息失败: {e}")
                        
            except Exception as e:
                print(f"❌ 处理集合 {collection_name} 时出错: {e}")
        
        print(f"\n" + "="*60)
        print("✅ 数据查看完成")
        
    except Exception as e:
        print(f"❌ 连接数据库失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()