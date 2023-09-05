# -*- coding: utf-8 -*-
# @Time    : 2023/8/24 9:06
# @Author  : yblir
# @File    : transfer.py
# explain  : 
# =======================================================
from pathlib2 import Path
import yaml

from .utils.logger_modify import MyLogger

cur_path = Path(__file__).resolve()
config_path = cur_path.parent / "config" / "config.yaml"

with open(str(config_path), "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# 被改下的logger日志, bool_file为True时,才会保存日志到本地
logger = MyLogger(log_level="INFO", bool_std=True, bool_file=True, log_file_path=config["log_path"]).get_logger()
