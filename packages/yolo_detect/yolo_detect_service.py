# -*- coding: utf-8 -*-
# @Time    : 2023/4/11 18:38
# @Author  : yblir
# @File    : yolo_detect_service.py
# explain  : 
# =======================================================
import os
# import logging.config
from transfer import config, logger
# from decode_tools.utils import config_dict
from decode_tools.aiservice import Service
from modules.example import ImageInferModule, AudioInferModule, VideoInferModule
from modules.predict import AttrPredictor
# 解码模块：图片微服务
from decode_tools.image_decode.fh_image import ImageDecodeCPU, ImageDecodeCPUThread, ImageDecodeGPU

# 解码模块：音频微服务
# from decode_tools.audio_decode.fh_audio import AudioDecode, AudioDecodeThread
# 解码模块：视频微服务
# from decode_tools.video_decode.fh_video import VideoDecodeGenerator

os.makedirs('log', exist_ok=True)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(BASE_DIR, 'config', 'app.yaml')

if __name__ == "__main__":
    # config = config_dict(config_path)
    # soLib_path = config['face_attribute']['libface_attr']
    # # data_type_keyword为请求体中数据的关键字，用于解析例如：images/image/video/audioData等..
    data_type_keyword = {"single": "image", "batch": "images"}  # 图片微服务关键字
    """
    data_type_keyword = {"single": "audioData", "batch": "audioDatas"}  # 音频微服务关键字
    data_type_keyword = {"single": "video", "batch": "videos"}  # 视频微服务关键字
    """

    service = Service(data_type_keyword)
    try:
        ####推理模块：图片微服务#林
        # 初始化图片解码类
        # decoder = ImageDecodeCPU(config)
        # 图片解码模块：单线程版
        # decoder = ImageDecodeCPUThread(config, thread_num=16)#片解码模块：多线程版
        decoder = ImageDecodeCPU(config)  # #图片解码模块:CPU版

        # 初始化模型推理类
        infer_module = ImageInferModule(config)
        # 单图片schema
        json_schema_path = os.path.join(BASE_DIR, 'config', 'single_image_schema.json')
        single_predictor = AttrPredictor(config, decoder, infer_module, json_schema_path, data_type_keyword)
        # batch图片schema
        json_schema_path2 = os.path.join(BASE_DIR, 'config', 'batch_image_schema.json')
        batch_predictor = AttrPredictor(config, decoder, infer_module, json_schema_path2, data_type_keyword)

        # 只注册一个url
        # service.register(config["url"],single_predictor)

        # 微服务默认不需要修改的部分
        service.register_list([
            [config['url'], single_predictor],
            [config['batch_url'], batch_predictor]
        ])
        service.run_list(port=config['server_port'], debug=True)

    except KeyboardInterrupt as e:
        logger.exception(e)
