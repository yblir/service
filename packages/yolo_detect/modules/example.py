# -*- coding: utf-8 -*-
# @Time    : 2023/4/13 8:53
# @Author  : yblir
# @File    : example.py
# explain  : 
# =======================================================
import sys

from ..transfer import config, logger
from ..decode_tools.exceptions import AILabException, AILabError
# from ..decode_tools.base_module_infer import BaseModuleInfer

from ctypes import cdll

try:
    from ..models import deployment
except:
    for lib_path in config["library"]:
        cdll.LoadLibrary(lib_path)
    from ..models import deployment

# 异常处理
ailab_error = AILabError()


class ImageInferModule:
    """
    图片推理模块demo
    """

    def __init__(self, params):
        """
        初始化
        :param params: 参数配置
        """
        super(ImageInferModule, self).__init__()

        self.param = deployment.ManualParam()
        self.engine = deployment.Engine()

        self.init_param(config["param_dict"])

        flag = self.engine.initEngine(self.param)
        if flag != 0:
            # logger.error('init Engine fail')
            sys.exit()

    def init_param(self, param_dict: dict):
        for k, v in param_dict.items():
            if hasattr(self.param, k):
                setattr(self.param, k, v)

    # todo 重写infer
    def infer(self, input_data):
        # results = []
        # infer_fail = {}
        rows = input_data["images"]
        try:
            future_infer_result = self.engine.inferEngine(rows)
        except Exception as e:
            logger.error(f"engine infer fail: {e}")
            future_infer_result = None

        return future_infer_result

    def age_to_agerange(self, age):
        """
        指定服务特有方法
        """
        pass

    def convert_hairstyle(self, hairstyle):
        """
        指定服务特有方法
        """
        pass

    def convert_hatcolor(self, hatcolor):
        """
        指定服务特有方法
        """
        pass


class AudioInferModule(BaseModuleInfer):
    """视频推理模块"""

    def __init__(self, params):
        """
        初始化
        """
        super(AudioInferModule, self).__init__(params)

    def module_infer(self, frame_list):
        # 对视频进行推理工作
        return True


class VideoInferModule(BaseModuleInfer):
    """
    视频推理模块demo
    """

    def __init__(self, params):
        """
        初始化
        :param params:参数置
        :param params:
        """
        super(VideoInferModule, self).__init__(params)

    @staticmethod
    def infer(input_data):
        """
            输入格式，为字典，每个字典中包含对应视频的解码返回信息，该信息也是一个字典，其中包含的关键字有：
            "ps":视频的fps信息
        :param input_data:
        :return:
        """
        predict_success = {}
        predict_fail = {}
        for k, v in input_data.items():
            one_video_data = input_data[k]
            # 取的
            video_generator = one_video_data["decode_generator"]
            frame_num = one_video_data["frame_num"]
            fps = one_video_data["fps"]
            duration = one_video_data["duration"]

            print("video NO.{}, fps={}, frame_num={}, duration={}".format(k, fps, frame_num, duration))

            frame_list = []
            # 示例：每个视频，获取其前10，进行推理
            for frame_index, frame in video_generator:
                if frame is not None:  # 若该帧效
                    frame_list.append(frame)
                if len(frame_list) == 10:
                    # 对视频帧list进行相应的推理工作
                    ret = module_infer(frame_list)
                    # 判断推理结果
                    if ret:
                        predict_success[k] = ["Predict Success"]
                    else:
                        predict_fail[k] = AILabError.ERROR_INFER_ERROR
                    break
                else:
                    logger.error("cannot get frame from video, no.{}".format(k))
                    predict_fail[k] = AILabError.ERROR_VIDEO_DECODE
                    break

        return predict_success, predict_fail