# -*- coding: utf-8 -*-
# @Time    : 2023/4/13 8:53
# @Author  : yblir
# @File    : model_infer.py
# explain  : 
# =======================================================
import math
import logging
from fhaiservice.exceptions import AILabException, AILabError
from fhaiservice.base_module_infer import BaseModuleInfer
from ctypes import c_void_p, cdll, byref, pointer, Structure, c_int, c_float, c_double, c_bool, c_char_p, c_char


class ImageInferModule(BaseModuleInfer):
    """
    图片推理模块demo
    """
    def __init__(self, params):
        """
        初始化
        :param params: 参数配置
        """
        super(ImageInferModule, self).__init__(params)

    def init_dynamicLib(self, soLib_path):
        # 2.加载动态库
        temp1 = cdll.LoadLibrary("/usr/local/lib64/libopencv_core.so.4.5")
        temp2 = cdll.LoadLibrary("/usr/local/lib64/libopencv_imgproc.so.4.5")
        temp3 = cdll.LoadLibrary("/usr/local/lib64/libopencv_dnn.so.4.5")
        temp4 = cdll.LoadLibrary("/usr/local/lib64/libopencv_imgcodecs.so.4.5")
        temp5 = cdll.LoadLibrary("/usr/local/lib64/libopencv_cudaarithm.so.4.5")
        temp6 = cdll.LoadLibrary("/usr/local/lib64/libopencv_cudawarping.so.4.5")
        temp7 = cdll.LoadLibrary("/usr/local/lib64/libopencv_cudaimgproc.so.4.5")
        temp8 = cdll.LoadLibrary("/usr/local/lib/libnvinfer.so.8")
        temp9 = cdll.LoadLibrary("/usr/local/lib/libnvinfer_plugin.so.8")
        temp10 = cdll.LoadLibrary("/usr/local/lib/libnvonnxparser.so")

        self.so = cdll.LoadLibrary(soLib_path)

        # 初始化C++ 动态库配置
        c_config = self.init_dynamicLib_params()
        # 初始化引擎
        self.engine = c_void_p(None)
        engine_state = self.so.initEngine(pointer(self.engine), pointer(c_config))
        if engine_state:
            raise
        print("init engine_state = ", engine_state, "engine =", self.engine)
        print('init finished.')

    def init_dynamicLib_params(self):
        """
        初始化C++ 动态库，将参数转成C++的格式，以便后面初始话动态库和构建引擎

        Returns:
            转成C++ 格式的配置信息
        """
        params = self.params["face_attribute"]
        base_param = self.params["base"]
        config = FaceAttrConf()
        config.model_file_age = bytes(params["model_file_age"], "utf8")
        config.model_file_gender = bytes(params["model_file_gender"], "utf8")
        config.model_file_mask = bytes(params["model_file_mask"], "utf8")
        config.model_file_hair = bytes(params["model_file_hair"], "utf8")
        config.model_file_hat = bytes(params["model_file_hat"], "utf8")
        config.model_file_must = bytes(params["model_file_must"], "utf8")
        config.batch_size = int(params["batchsize"])
        config.device_id = int(base_param["device_id"])
        config.fp16_flag = True if params['fp16_flag'].lower() == "true" else False

        return config
        results = []
        infer_fail = {}
        rows=input_data["imageRows"]
        rows_len=len(rows)
        # 遍历字典每张图片
        for i in range(rows_len):
            rect_points = rows[i]["rect"]
            key_points = rows[i]["faceKeyPoint"][i]
            image = input_data["images"][i]
    # todo 重写infer
    def infer(self, input_data):
        results = []
        infer_fail = {}
        # 遍历字典每张图片
        for i in range(len(input_data["images"])):
            rect_points = input_data["rect"][i]
            key_points = input_data["faceKeyPoint"][i]
            image = input_data["images"][i]

            rect_nums = len(rect_points)
            infer_results = (int(rect_nums) * FaceAttrRet)()
            height, width, channel = image.shape

            imgPixelFormat = 0
            face_info = (int(rect_nums) * FaceAttrInputInfo)()
            temp_err = []

            for k in range(rect_nums):
                # 将人脸坐标框的典转成列表映射给推理输入
                for l, item in enumerate([
                    rect_points[k]['x1'], rect_points[k]['y1'], rect_points[k]['x2'], rect_points[k]['y2']
                ]):
                    face_info[k].rect_points[l] = item
                # 将人脸关键点坐标转成列表映射给推理输入
                temp_list = []
                for key_point in key_points[k]:
                    temp_list.append(key_point["X"])
                    temp_list.append(key_point["Y"])
                for l, item in enumerate(temp_list):
                    face_info[k].key_points[l] = item
            # //////////////////////////////////////////////////////////////
            frame_data = image.ctypes.data_as(c_char_p)
            ret = self.so.inferFrameRect(self.engine, frame_data,
                                         width, height, imgPixelFormat, int(rect_nums), pointer(face_info),
                                         pointer(infer_results))
            if ret != 0:
                temp_err[i] = AILabError.ERROR_INFER_ERROR
                continue
            if temp_err:
                infer_fail[i] = temp_err
            results.append()

        return results, infer_fail

    def convert_to_jsons(self, rect_nums, infer_result):
        """
        将得到的结构体变为python list格式，如果输入是背景图+检测框则is_rect设置为true
        : param infer_result:
        : type infer_result:
        :return:
        :rtype:
        """
        result = []
        for i in range(int(rect_nums)):
            attr_res = {}
            attr_res["age"] = infer_result[i].age
            attr_res["gender"] = infer_result[i].gender
            attr_res["glasses"] = infer_result[i].glass
            attr_res["mask"] = infer_result[i].mask
            attr_res["hairstyle"] = infer_result[i].hair
            attr_res["hatcolor"] = infer_result[i].hat
            attr_res["mustachestyle"] = infer_result[i].must
            result.append(attr_res)
        return result


class FaceAttrConf(Structure):
    _fields_ = [
        ("model_file_age", c_char * 256),
        ("model_file_gender", c_char * 256),
        ("model_file_mask", c_char * 256),

        ("model_file_age", c_char * 256),
        ("model_file_gender", c_char * 256),
        ("model_file_mask", c_char * 256),
        ("model_file_hair", c_char * 256),
        ("model_file_hat", c_char * 256),
        ("model_file_must", c_char * 256),
        ("batch_size", c_int),
        ("device_id", c_int),
        ("fp16_flag", c_bool)
    ]


class FaceAttrRet(Structure):
    _fields_ = [
        ("age", c_int),
        ("gender", c_int),
        ("gender_conf", c_float),
        ("glass", c_int),
        ("glass_conf", c_float),
        ("mask", c_int),
        ("mask_conf", c_float),
        ("hair", c_int),
        ("hat", c_int),
        ("hat_conf", c_float),
        ("must", c_int),
        ("must_conf", c_float)
    ]


class FaceAttrInputInfo(Structure):
    fields_ = [
        ("rect_points", c_int * 4),
        ("key_points", c_int * 10)
    ]


class AudioInferModule(BaseModuleInfer):
    """视频推理模块"""

    def __init__(self, params):
        """
        初始化
        """
        super(AudioInferModule, self).__init__(params)


def module_infer(frame_list):
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
            # 示例：每个视频，获取共前10，进行推
            for frame_index, frame in video_generator:
                if frame is not None:  # 若效
                    frame_list.append(frame)
                if len(frame_list) == 10:
                    # 对视频list进行相应的推理工作
                    ret = module_infer(frame_list)
                    # 判断推理结果
                    if ret:
                        predict_success[k] = ["Predict Success"]
                    else:
                        predict_fail[k] = AILabError.ERROR_INFER_ERROR
                    break
                else:
                    logging.error("cannot get frame from yodeo, no.{}".format(k))
                    predict_fail[k] = AILabError.ERROR_VIDEO_DECODE
                    break

        return predict_success, predict_fail
