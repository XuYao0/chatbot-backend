"""创建日志 - 支持层级logger管理"""

import logging
import os

# 根logger名称
ROOT_LOGGER_NAME = "bzchat"

def setup_root_logger(log_file="logs/total.log", level=logging.INFO):
    """
    设置根logger，所有子模块的logger都会继承这个配置
    
    Args:
        log_file: 日志文件路径
        level: 日志级别
    
    Returns:
        根logger实例
    """
    # 确保日志目录存在
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    root_logger = logging.getLogger(ROOT_LOGGER_NAME)
    root_logger.setLevel(level)

    # 防止重复添加 handler
    if root_logger.handlers:
        return root_logger

    # 创建文件 handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(level)

    # 创建控制台 handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  

    # 定义格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 添加 handler 到根logger
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    return root_logger

def get_logger(name: str):
    """
    获取子logger，会自动继承根logger的配置
    
    Args:
        name: logger名称，建议使用模块名如 'services.milvus', 'utils.image'
    
    Returns:
        子logger实例
    """
    full_name = f"{ROOT_LOGGER_NAME}.{name}"
    return logging.getLogger(full_name)