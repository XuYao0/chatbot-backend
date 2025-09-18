#!/usr/bin/env python3
"""
测试增强的聊天接口的图片上传功能
"""
import json
import requests
import base64
from datetime import datetime

def encode_image_to_base64(image_path: str) -> str:
    """将图片文件编码为 base64"""
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return f"data:image/jpeg;base64,{encoded_string}"

def test_chat_with_image():
    """测试带图片的聊天功能"""
    
    # 如果你有测试图片，可以使用这个路径
    # image_base64 = encode_image_to_base64("test_image.jpg")
    
    # 模拟 base64 图片数据（这里使用一个很小的测试图片）
    # 1x1 像素的红色 PNG 图片的 base64 编码
    test_image_base64 = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
    
    # 构建测试请求
    test_request = {
        "model": "deepseek-r1-250528",
        "temperature": 0.7,
        "stream": False,
        "use_context": True,
        "user_id": "test_user_123",
        "session_id": "test_session_456",
        "messages": [
            {
                "role": "system",
                "content": "你是一个有用的助手。",
                "id": "system_001"
            },
            {
                "role": "user", 
                "content": [
                    {
                        "type": "text",
                        "text": "请描述这张图片"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": test_image_base64
                        }
                    }
                ],
                "id": "user_001"
            }
        ]
    }
    
    try:
        # 发送请求到本地服务
        response = requests.post(
            "http://10.176.56.192:8001/chat/completions",
            json=test_request,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        print(f"响应状态码: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✓ 请求成功!")
            print(f"消息ID: {result.get('id', 'N/A')}")
            print(f"模型: {result.get('model', 'N/A')}")
            
            if result.get('choices') and len(result['choices']) > 0:
                assistant_message = result['choices'][0].get('message', {})
                print(f"助手回复: {assistant_message.get('content', 'N/A')[:200]}...")
            
            # 显示上下文信息
            context_info = result.get('context_info', {})
            print(f"使用了上下文: {context_info.get('context_used', False)}")
            print(f"上下文条数: {context_info.get('context_count', 0)}")
            
        else:
            print(f"✗ 请求失败: {response.status_code}")
            print(f"错误信息: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("✗ 连接失败: 请确保服务器正在运行 (python enhanced_deepseek_service.py)")
    except Exception as e:
        print(f"✗ 测试异常: {e}")

def test_text_only_chat():
    """测试纯文本聊天功能"""
    test_request = {
        "model": "deepseek-r1-250528", 
        "stream": False,
        "use_context": True,
        "user_id": "test_user_123",
        "session_id": "test_session_456",
        "messages": [
            {
                "role": "user",
                "content": "你好，请介绍一下你自己。",
                "id": "user_text_001"
            }
        ]
    }
    
    try:
        response = requests.post(
            "http://10.176.56.192:8001/chat/completions",
            json=test_request,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        print(f"\n纯文本测试 - 响应状态码: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✓ 纯文本请求成功!")
            if result.get('choices') and len(result['choices']) > 0:
                assistant_message = result['choices'][0].get('message', {})
                print(f"助手回复: {assistant_message.get('content', 'N/A')[:200]}...")
        else:
            print(f"✗ 纯文本请求失败: {response.status_code}")
            
    except Exception as e:
        print(f"✗ 纯文本测试异常: {e}")

if __name__ == "__main__":
    print("开始测试增强的聊天接口...")
    print("=" * 50)
    
    # 测试图片上传功能
    print("1. 测试图片上传功能:")
    test_chat_with_image()
    
    # 测试纯文本功能
    print("2. 测试纯文本功能:")
    test_text_only_chat()
    
    print("\n测试完成!")