# -*- coding: utf-8 -*-
# @Time    : 2023/4/11 18:38
# @Author  : yblir
# @File    : yolo_detect_service.py
# explain  : 
# =======================================================
import os
# import logging.config
from transfer import config, logger
# from decode_tools.util_func import config_dict
from packages.yolo_detect.base_interface.service import Service
from modules.model_infer import ImageInferModule
from modules.predict import YoloPredictor
# 解码模块：图片微服务
from decode_tools.image_decode.fh_image import ImageDecodeCPU

# 解码模块：音频微服务
# from decode_tools.audio_decode.fh_audio import AudioDecode, AudioDecodeThread
# 解码模块：视频微服务
# from decode_tools.video_decode.fh_video import VideoDecodeGenerator

os.makedirs('log', exist_ok=True)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(BASE_DIR, 'config', 'app.yaml')

if __name__ == "__main__":
    # # data_type_keyword为请求体中数据的关键字，用于解析例如：images/image/video/audioData等..
    data_type_keyword = {"single": "image", "batch": "images"}  # 图片微服务关键字

    # data_type_keyword = {"single": "audio", "batch": "audios"}  # 音频微服务关键字
    # data_type_keyword = {"single": "video", "batch": "videos"}  # 视频微服务关键字

    service = Service(data_type_keyword)
    # 初始化图片解码类
    decoder = ImageDecodeCPU(config)
    # 图片解码模块：多线程版
    # decoder = ImageDecodeCPUThread(config, thread_num=16)

    # 初始化模型推理类
    infer_module = ImageInferModule(config)

    # 单图片schema
    json_schema_path = os.path.join(BASE_DIR, 'config', 'single_image_schema.json')
    single_predictor = YoloPredictor(config, decoder, infer_module, json_schema_path, data_type_keyword)
    # batch图片schema
    json_schema_path2 = os.path.join(BASE_DIR, 'config', 'batch_image_schema.json')
    batch_predictor = YoloPredictor(config, decoder, infer_module, json_schema_path2, data_type_keyword)

    # 只注册一个url
    # service.register(config["url"],single_predictor)

    # 同时注册single与batch两个版本推理过程, 根据输入参数自行决定使用哪个
    service.register_list([
        [config['url'], single_predictor],
        [config['batch_url'], batch_predictor]
    ])

    try:
        service.run_list(port=config['server_port'], debug=False)
    except Exception as e:
        # 当服务出错时,必须保证可以是否推理资源
        infer_module.module_release()
        logger.exception(e)
