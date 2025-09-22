"""
图片描述生成工具类
使用 Qwen2.5-VL 模型生成图片描述
"""
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from typing import List, Optional, Dict, Any
import logging
from pathlib import Path
from utils.log import get_logger


logger = get_logger(__name__)
# logger = logging.getLogger(__name__)

class ImageDescriber:
    """图片描述生成工具类"""
    
    def __init__(self, model_path: str = "/home/xuyao/data/Qwen2.5-VL-3B-Instruct"):
        """
        初始化图片描述生成器
        
        Args:
            model_path: 模型路径，可以是本地路径或 HuggingFace 模型名称
        """
        self.model_path = model_path
        self.model = None
        self.processor = None
        self._load_model()
    
    def _load_model(self):
        """加载模型和处理器"""
        try:
            logger.info(f"正在加载图片描述模型: {self.model_path}")
            
            # 加载模型，启用 flash_attention_2 以获得更好的性能
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                # attn_implementation="flash_attention_2",
                device_map="auto",
            )
            
            # 加载处理器
            self.processor = AutoProcessor.from_pretrained(self.model_path)
            
            logger.info("图片描述模型加载成功")
            
        except Exception as e:
            logger.error(f"加载图片描述模型失败: {str(e)}")
            raise e
    
    def describe_image(
        self, 
        image_path: str, 
        context: str = "",
        max_new_tokens: int = 1024
    ) -> str:
        """
        为单张图片生成描述
        
        Args:
            image_path: 图片文件路径
            max_new_tokens: 最大生成token数
            
        Returns:
            图片描述文本
        """
        try:
            # 验证图片文件是否存在
            if not Path(image_path).exists():
                raise FileNotFoundError(f"图片文件不存在: {image_path}")
            
            prompt = """
请对图片进行详细、客观、结构化的描述，以便后续由纯文本语言模型进行推理。请按以下格式输出：

【场景类型】  
（例如：室内、室外、街道、办公室、餐厅、自然风光等）

【主要对象】  
- 列出图片中所有可见的显著物体或人物，包括其位置、状态和相互关系。
- 使用“主语 + 动作/状态 + 宾语”的方式描述，避免模糊词汇。

【文字内容】  
- 提取图片中所有可读的文字（如标语、屏幕显示、书名、车牌等），并注明其位置。
- 如果无文字，写“无”。

【颜色与风格】  
- 描述整体色调、光影、艺术风格（如写实、卡通、素描、水彩等）。

【潜在意图或上下文线索】  
- 基于视觉信息，推测可能的场景目的（如：广告、教学、导航、社交分享等）。
- 仅基于可见内容推理，避免过度猜测。

【注意】  
- 保持客观，不添加情感或主观评价。
- 不要回答图片之外的问题。
- 不要使用“图片中可能”、“也许”等不确定词汇，只陈述可见事实。

【上下文消息】
随图片而来的上下文消息（如果有）：
"""

            if context.strip():  # 如果有上下文信息，添加到提示词中
                prompt += f"\n{context.strip()}\n"

            # 构建消息
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            
            # 处理消息
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            
            # 准备输入
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")
            
            # 生成描述
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs, 
                    max_new_tokens=max_new_tokens,
                    do_sample=False,  # 使用贪心搜索获得更稳定的结果
                    temperature=0.7
                )
            
            # 解码结果
            generated_ids_trimmed = [
                out_ids[len(in_ids):] 
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )[0]
            
            logger.info(f"成功生成图片描述: {image_path}")
            return output_text.strip()
            
        except Exception as e:
            logger.error(f"生成图片描述失败 {image_path}: {str(e)}")
            raise e
    
    def unload_model(self):
        """卸载模型以释放内存"""
        try:
            if self.model is not None:
                del self.model
                self.model = None
            if self.processor is not None:
                del self.processor
                self.processor = None
            
            # 清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            logger.info("图片描述模型已卸载")
            
        except Exception as e:
            logger.error(f"卸载模型失败: {str(e)}")
    
    def __del__(self):
        """析构函数，自动卸载模型以释放资源"""
        try:
            # 检查模型是否已加载，避免重复卸载
            if hasattr(self, 'model') and self.model is not None:
                del self.model
                self.model = None
            if hasattr(self, 'processor') and self.processor is not None:
                del self.processor
                self.processor = None
            
            # 清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception:
            # 在析构函数中不应该抛出异常，静默处理错误
            # 避免使用logger，因为析构时模块可能已被清理
            pass


if __name__ == "__main__":
    # 简单测试
    describer = ImageDescriber("/home/xuyao/data/Qwen2.5-VL-3B-Instruct")
    test_image_path = "/home/xuyao/bzchat/backend/test.png"  # 替换为实际图片路径
    if Path(test_image_path).exists():
        description = describer.describe_image(
            image_path=test_image_path,
            context="请详细描述这张图片的内容"
        )
        print("图片描述结果:")
        print(description)
    else:
        print(f"测试图片不存在: {test_image_path}")
    
    # 卸载模型
    describer.unload_model()