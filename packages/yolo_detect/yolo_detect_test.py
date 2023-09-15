# -*- coding: utf-8 -*-
# @Time    : 2023/9/15 14:16
# @Author  : yblir
# @File    : yolo_detect_test.py
# explain  : 
# =======================================================
import base64
import os

import cv2
from pathlib2 import Path
from pprint import pprint

from transfer import root_path, config, logger
# from services.service import Service
from modules.model_infer import ImageInferModule
from modules.predict import YoloPredictor
# 解码模块：图片微服务
from decode_tools.image_decode.fh_image import ImageDecodeCPU

# 解码模块：音频微服务
# from decode_tools.audio_decode.fh_audio import AudioDecode, AudioDecodeThread
# 解码模块：视频微服务
# from decode_tools.video_decode.fh_video import VideoDecodeGenerator

os.makedirs('log', exist_ok=True)

if __name__ == "__main__":
    # # data_type_keyword为请求体中数据的关键字，用于解析例如：images/image/video/audioData等..
    data_type_keyword = {"single": "image", "batch": "images"}  # 图片微服务关键字
    # data_type_keyword = {"single": "audio", "batch": "audios"}  # 音频微服务关键字
    # data_type_keyword = {"single": "video", "batch": "videos"}  # 视频微服务关键字

    # service = Service(data_type_keyword)
    # 初始化图片解码类
    decoder = ImageDecodeCPU(config)
    # 图片解码模块：多线程版
    # decoder = ImageDecodeCPUThread(config, thread_num=16)

    # 初始化模型推理类
    infer_module = ImageInferModule(config)

    # 单图片schema
    # json_schema_path = os.path.join(root_path, 'config', 'single_image_schema.json')
    # single_predictor = YoloPredictor(decoder, infer_module, json_schema_path, data_type_keyword, "")
    # batch图片schema
    json_schema_path2 = os.path.join(root_path, 'config', 'batch_image_schema.json')
    batch_predictor = YoloPredictor(decoder, infer_module, "", data_type_keyword, "")
    logger.info("batch_predictor init ok")
    imgs_dir = Path("/mnt/e/test_imgs")

    count = 0
    imgs = []
    b64_dict = {}
    for i, item in enumerate(imgs_dir.iterdir()):
        b64_data = base64.b64encode(open(str(item), "rb").read()).decode()
        b64_dict[count] = b64_data
        # print(i)
        count += 1
        if count < 3 and i != 6:
            continue
        count = 0

        request_data = {
            "noid"  : "111",
            "sysid" : "test",
            "images": b64_dict
        }

        # logger.info("===================")
        res = batch_predictor.predict(request_data)
        b64_dict = {}
        logger.info("===================")
        pprint(res)
        # logger.info("===================")
