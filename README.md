# quick start
#### 1.导入火山引擎deepseek API key，也可以选择其他LLM推理服务，但需要修改base_url和传递数据的格式
```
export VOLCENG_API_KEY=your_actual_api_key
```

#### 2.启动milvus服务
```
cd milvus_config
docker-compose up -d
```

#### 3.启动主服务程序
```
python mainService.py
```

#### 4.接口介绍
/chat/completions
可以接收来自前端的如下例子的数据：
```json
{
  "model": "deepseek-r1-250528",
  "temperature": 0.8,
  "top_p": 0.9,
  "frequency_penalty": 0.0,
  "presence_penalty": 0.0,
  "max_tokens": 3000,
  "n": 1,
  "stream": true,
  "messages": [
    {
      "role": "system",
      "content": "你是一个有帮助的AI助手。"
    },
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "请描述这张图片的内容。"
        },
        {
          "type": "image_url",
          "image_url": "base64字符串"
        }
      ]
    }
  ]
}
```
针对实验室前端项目传来的数据格式做了适配，但实际上并非所有参数都能应用到deepseek API的调用中去。后续可以修改