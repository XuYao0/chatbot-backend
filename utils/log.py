"""创建日志"""

import logging
import os

def setup_logger(log_file="milvus_store.log", level=logging.INFO):
    logger = logging.getLogger("MilvusMessageStore")
    logger.setLevel(level)

    # 防止重复添加 handler
    if logger.handlers:
        return logger

    # 创建文件 handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(level)

    # 创建控制台 handler（可选）
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)  # 控制台只显示警告以上

    # 定义格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 添加 handler
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger