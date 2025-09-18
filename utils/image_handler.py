"""
图片处理工具类
处理图片的二进制数据保存、加载和格式转换
"""
import os
import io
import uuid
import base64
import logging
from typing import Optional, Tuple
from PIL import Image
from pathlib import Path

logger = logging.getLogger(__name__)

class ImageHandler:
    """图片处理工具类"""
    
    def __init__(self, storage_path: str = "/home/xuyao/data/bzchat_pic"):
        """
        初始化图片处理器
        
        Args:
            storage_path: 图片存储路径
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # 支持的图片格式
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
        
        logger.info("from image_handler: " + f"图片存储路径: {self.storage_path}")
    
    def save_image_from_binary(
        self, 
        binary_data: bytes, 
        filename: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> Tuple[str, str]:
        """
        保存二进制图片数据到文件
        
        Args:
            binary_data: 图片的二进制数据
            filename: 可选的文件名
            user_id: 用户ID（用于文件夹分类）
            session_id: 会话ID（用于文件夹分类）
            
        Returns:
            Tuple[str, str]: (文件路径, 文件ID)
        """
        try:
            # 验证是否为有效图片
            image = Image.open(io.BytesIO(binary_data))
            image.verify()  # 验证图片完整性
            
            # 重新打开图片（verify后需要重新打开）
            image = Image.open(io.BytesIO(binary_data))
            
            # 生成文件ID和路径
            file_id = str(uuid.uuid4())
            
            # 创建分层目录结构
            sub_dirs = []
            sub_dirs.append(f"user_{user_id}")
            sub_dirs.append(f"session_{session_id}")
            
            # 创建目录
            save_dir = self.storage_path
            for sub_dir in sub_dirs:
                save_dir = save_dir / sub_dir  # / 操作符在这里是 Path 对象的重载操作，用于拼接路径，比字符串拼接更安全、跨平台。
                save_dir.mkdir(exist_ok=True)
            
            # 确定文件扩展名
            format_lower = image.format.lower() if image.format else 'jpeg'
            if format_lower == 'jpeg':
                ext = '.jpg'   # 一个常见的约定和兼容性实践，把 'jpeg' 扩展名统一为 '.jpg'
            else:
                ext = f'.{format_lower}'
            
            # 生成最终文件名
            if filename:
                # 使用提供的文件名，但确保扩展名正确
                name_without_ext = Path(filename).stem
                final_filename = f"{name_without_ext}_{file_id}{ext}"
            else:
                final_filename = f"image_{file_id}{ext}"
            
            file_path = save_dir / final_filename
            
            # 保存图片
            # 如果是 RGBA 但要保存为 JPEG，转换为 RGB
            if image.mode == 'RGBA' and ext == '.jpg':
                # 创建白色背景
                background = Image.new('RGB', image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[-1])  # 使用 alpha 通道作为 mask
                image = background
            
            image.save(file_path, optimize=True, quality=85)
            
            # 返回相对路径和文件ID
            relative_path = str(file_path.relative_to(self.storage_path))
            
            logger.info("from image_handler: " + f"图片保存成功: {relative_path}")
            return relative_path, file_id
            
        except Exception as e:
            logger.error("from image_handler: " + f"保存图片失败: {e}")
            raise ValueError(f"保存图片失败: {e}")
    
    def save_image_from_base64(
        self, 
        base64_data: str, 
        filename: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> Tuple[str, str]:
        """
        保存 base64 编码的图片数据
        
        Args:
            base64_data: base64 编码的图片数据
            filename: 可选的文件名
            user_id: 用户ID
            session_id: 会话ID
            
        Returns:
            Tuple[str, str]: (文件路径, 文件ID)
        """
        try:
            # 处理 data URL 格式 (data:image/jpeg;base64,...)
            if base64_data.startswith('data:'):
                base64_data = base64_data.split(',')[1]
            
            # 解码 base64
            binary_data = base64.b64decode(base64_data)
            
            return self.save_image_from_binary(binary_data, filename, user_id, session_id)
            
        except Exception as e:
            logger.error("from image_handler: " + f"保存 base64 图片失败: {e}")
            raise ValueError(f"保存 base64 图片失败: {e}")
    
    def load_image(self, relative_path: str) -> Optional[Image.Image]:
        """
        加载图片文件
        
        Args:
            relative_path: 相对于存储路径的图片路径
            
        Returns:
            PIL.Image.Image: 加载的图片对象，如果失败返回 None
        """
        try:
            full_path = self.storage_path / relative_path
            
            if not full_path.exists():
                logger.error("from image_handler: " + f"图片文件不存在: {full_path}")
                return None
            
            image = Image.open(full_path)
            return image
            
        except Exception as e:
            logger.error("from image_handler: " + f"加载图片失败: {e}")
            return None
    
    def get_image_info(self, relative_path: str) -> Optional[dict]:
        """
        获取图片信息
        
        Args:
            relative_path: 相对于存储路径的图片路径
            
        Returns:
            dict: 图片信息字典
        """
        try:
            full_path = self.storage_path / relative_path
            
            if not full_path.exists():
                return None
            
            image = Image.open(full_path)
            file_stat = full_path.stat()
            
            return {
                "path": relative_path,
                "full_path": str(full_path),
                "size": image.size,  # (width, height)
                "mode": image.mode,
                "format": image.format,
                "file_size": file_stat.st_size,
                "created_time": file_stat.st_ctime,
                "modified_time": file_stat.st_mtime
            }
            
        except Exception as e:
            logger.error("from image_handler: " + f"获取图片信息失败: {e}")
            return None
    
    def delete_image(self, relative_path: str) -> bool:
        """
        删除图片文件
        
        Args:
            relative_path: 相对于存储路径的图片路径
            
        Returns:
            bool: 是否删除成功
        """
        try:
            full_path = self.storage_path / relative_path
            
            if full_path.exists():
                full_path.unlink()
                logger.info("from image_handler: " + f"图片删除成功: {relative_path}")
                return True
            else:
                logger.warning("from image_handler: " + f"要删除的图片不存在: {relative_path}")
                return False
                
        except Exception as e:
            logger.error("from image_handler: " + f"删除图片失败: {e}")
            return False
    
    def cleanup_empty_dirs(self):
        """清理空的目录"""
        try:
            for root, dirs, files in os.walk(self.storage_path, topdown=False):
                for dir_name in dirs:
                    dir_path = Path(root) / dir_name
                    try:
                        if not any(dir_path.iterdir()):  # 如果目录为空
                            dir_path.rmdir()
                            logger.info("from image_handler: " + f"删除空目录: {dir_path}")
                    except OSError:
                        pass  # 目录不为空或其他错误，忽略
        except Exception as e:
            logger.error("from image_handler: " + f"清理空目录失败: {e}")
    
    def get_storage_stats(self) -> dict:
        """获取存储统计信息"""
        try:
            total_files = 0
            total_size = 0
            file_types = {}
            
            for file_path in self.storage_path.rglob("*"):
                if file_path.is_file():
                    total_files += 1
                    file_size = file_path.stat().st_size
                    total_size += file_size
                    
                    ext = file_path.suffix.lower()
                    file_types[ext] = file_types.get(ext, 0) + 1
            
            return {
                "total_files": total_files,
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "file_types": file_types,
                "storage_path": str(self.storage_path)
            }
            
        except Exception as e:
            logger.error("from image_handler: " + f"获取存储统计失败: {e}")
            return {}




# 全局图片处理器实例
image_handler = ImageHandler()


if __name__ == "__main__":
    # 测试代码
    handler = ImageHandler("/tmp/test_images")
    
    # 测试统计信息
    stats = handler.get_storage_stats()
    print("存储统计:", stats)
