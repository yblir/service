# -*- coding: utf-8 -*-
# @Time    : 2023/8/24 9:06
# @Author  : yblir
# @File    : transfer.py
# explain  : 初始化各种路径和变量
# =======================================================
import sys
from pathlib2 import Path
import yaml

# 当前目录
cur_path = Path(__file__).resolve()
# 根目录
root_path = str(cur_path.parent)
root_path2 = str(cur_path.parent.parent)
root_path3 = str(cur_path.parent.parent.parent)
root_path4 = str(cur_path.parent / "modules")
# 将根目录yolo_detectd添加到搜索路径中
sys.path.append(root_path)
sys.path.append(root_path2)
sys.path.append(root_path3)
sys.path.append(root_path4)
from utils.logger_modify import MyLogger

config_path = cur_path.parent / "config" / "config.yaml"

with open(str(config_path), "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# 被改下的logger日志, bool_file为True时,才会保存日志到本地
logger = MyLogger(log_level="INFO", bool_std=True, bool_file=True, log_file_path=config["log_path"]).get_logger()
