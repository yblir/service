# -*- coding: utf-8 -*-
# @Time    : 2023/8/24 9:06
# @Author  : yblir
# @File    : transfer.py
# explain  : 
# =======================================================
import yaml

from .utils.logger_modify import MyLogger

with open("./config/config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# 被改下的logger日志, bool_file为True时,才会保存日志到本地
logger = MyLogger(log_level="INFO", bool_std=True, bool_file=True, log_file_path=config["log_path"]).get_logger()
