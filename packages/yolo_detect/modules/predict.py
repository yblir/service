# -*- coding: utf-8 -*-
# @Time    : 2023/4/13 9:03
# @Author  : yblir
# @File    : predict.py
# explain  : 
# ================================================================================
import json
import uuid
import time
import logging
import base64
import json

from typing import Dict
from ..decode_tools.predictor import Predictor
from ..decode_tools.exceptions import AILabException, AILabError

# 异常处理
ailab_error = AILabError()


# todo 修改Demo类为自己模型的推理过程
class YoloPredictor(Predictor):
    """
    Predictor的功能如下:
    1、处理所有与模型推理相关的事务;
    2.predict()方法:将所有模型的推理分为三步骤:前处理、推理和后处理;
    3.
    pre_process()方法:做前处理相关操作，包括，解析输入数据。对数据解码:
    4.postprocess()方法:将推理后的数据转成微服务输出的格式
    5.返回推理状态信息:请求总数、目标数量、解码失败数量和推理失败数量
    """

    def __init__(self, params, decoder, module_infer,
                 json_schema_path="",
                 data_type_keyword=Dict,
                 json_schema_output_path=""):
        """
        初始化Predictor
        :param params:配置参数
        :param decoder:初始化完成后的解码模块类
        :param module_infer:初始化完成后的推理模块类
        :param json_schema_path:输入的校验schema文档
        :param data_type_keyword:数据关键字，，用于提取发送请求中的数据
        :param json_schema_output_path:输出的校验schema文档
        """

        super(YoloPredictor, self).__init__(params, decoder, module_infer, json_schema_path, json_schema_output_path)
        self.module_infer = module_infer
        self.decoder = decoder
        self.data_type_keyword = data_type_keyword

        logging.info("init demo Predictor")

    def predict(self, request_data: Dict):
        """
        接收请求体数据，并将结果通过字典类型数ret_obj
        :param request data:
        :return:微服务的返回信息(Dict)、请求总数(int)、目标数量(int)、解码失败数量(int)。推理失败数量(int)
        """
        ret_obj = {}
        self.request_data = request_data

        # 1.解析请求体
        decode_time = time.time()
        self.start_time = decode_time
        decode_success, decode_fail = self.pre_process(request_data)
        self.decode_fail = decode_fail
        logging.debug(f"decode time: {format(time.time() - decode_time)}")

        # 替换解码正确的图片，去掉解码失败的
        for i in range(len(request_data[self.data_type_keyword["keyword batch"]])):
            if i in decode_success:
                request_data[self.data_type_keyword["keyword_batch"]][i]["imageData"] = decode_success[i]
            else:
                request_data[self.data_type_keyword["keyword batch"]][i]["imageData"] = None
        # 2.推理模块
        # 解码成功数据大于0，则进行推理
        predict_fail = {}
        if len(decode_success) > 0:
            infer_time = time.time()
            predict_success, predict_fail = self.module.infer(request_data)
            self.predict_success = predict_success
            self.predict_fail = predict_fail
            logging.debug("infer time: {}".format(time.time() - infer_time))
        else:
            logging.debug("cannot decode any images from request.")
            self.predict_success = {}
            self.predict_fail = {}

        # 3.根据请求体，对结果进行后处理，转成微服务输出格式
        post_process_time = time.time()
        target_num = self.post_process(ret_obj)
        logging.debug("post time: {}".format(time.time() - post_process_time))

        # 记录每个batch处理情况，成功多少张，失败多少张，解码失败多少张，结果为空多少张
        decode_fail_num = len(decode_fail.keys())
        predict_fail_num = len(predict_fail.keys())

        # total data_num = len(list(decode_success.keys()) + len(list(decode_fail.keys())
        total_data_num = target_num + decode_fail_num + predict_fail_num
        return ret_obj, total_data_num, target_num, decode_fail_num, predict_fail_num

    # todo 重写pre_process
    def pre_process(self, request):
        """
        解析请求体
        """
        raw_data_for_decode = None
        # 1、根据关键字，将数据从字典中读取出来
        keyword_single = self.data_type_keyword["keyword_single"]
        keyword_batch = self.data_type_keyword["keyword_batch"]

        if keyword_single in request:
            try:
                if not isinstance(request["faceKeyPoint"][0], list):
                    raise AILabException(ailab_error.ERROR_PARAMETER)
            except Exception:
                raise AILabException(ailab_error.ERROR_PARAMETER)
            raw_data_for_decode = request[keyword_single]
        # batch请求需将list里面的字典数据提取出来
        if keyword_batch in request:
            raw_data_for_decode = []
            for item in request[keyword_batch]:
                if "imageData" not in item.keys():
                    logging.error("no key imageData")
                    raise AILabException(ailab_error.ERROR_PARAMETER)
                raw_data_for_decode.append(item["imageData"])

        if not raw_data_for_decode:
            # 若请求体中无关键字，则前处理错误
            logging.error(ailab_error.ERROR_PARAMETER)
            raise AILabException(ailab_error.ERROR_PARAMETER)

        # 2.解码操作
        # 解码后,不论单张还是多batch,decode_imgs_dic都是字典,数字0,1,2,为键,键为nd.array格式图片矩阵
        decode_success, decode_fail = self.decoder.decode(raw_data_for_decode)

        return decode_success, decode_fail

    # todo 重写post
    def post_process(self, ret_obj: Dict):
        """
        将结果打包成接口文档规定的格式，将处理后的结果更新至ret_obj输出
        """
        # 解码相关信息
        decode_fail = self.decode_fail
        # 推理相关信息[[{...},{},...]]
        predict_success = self.predict_success
        predict_fail = self.predict_fail

        dict_sort_fail = {}  # 用于存放一批数据中返回失败的数据信息,类型是字典
        dict_sort_success = {}

        # 对解码失败的数据返回对应的错误状态
        for i in decode_fail.keys():
            # 对解码错误的数据结果设为空列表
            image_id = self.request_data[self.data_type_keyword]["keyword_batch"][i]["imageID"]
            temp_dict = ailab_error.ERROR_ERROR_IMAGE_DECODE
            temp_dict["imageID"] = image_id
            dict_sort_fail[i] = temp_dict

        # 对推理失败的数据返回对应的错误状态
        for i in predict_fail.keys():
            # 对推理错误的数据结果设为空列表
            image_id = self.request_data[self.data_type_keyword]["keyword_batch"][i]["imageID"]
            temp_dict = predict_fail[i]
            temp_dict["imageID"] = image_id
            dict_sort_fail[i] = temp_dict

        # 对推理成功的数据返回对应的推理结果
        target_num = 0  # 用于记录目标数据
        for i, item in enumerate(predict_success):
            dict_sort_success[i] = item
            target_num += 1

        # 将一批数据中返回成功和失败的两个字典拼接起来
        dict_sort = {**dict_sort_fail, **dict_sort_success}  # dict(dict1,**dict2)是两个字典的拼接

        # 按键把字典进行排序，目的是与输入数据的顺序保持一致
        ret_info = []
        for i in sorted(dict_sort):
            ret_info.append(dict_sort[i])

        # 将返回结果放在result中
        ret_obj['faceObjectRows'] = ret_info

        # 计时
        time_ms = time.time() - self.start_time
        # 获取noid
        try:
            # 发送请求获得的数据
            request_data = self.request_data
            noid = request_data['noid']
        except Exception as _:
            noid = str(uuid.uuid1())

        if noid == "":
            noid = str(uuid.uuid1())

        ret_obj["noid"] = noid
        ret_obj["costTime"] = int(time_ms * 1000)
        ret_obj["statusCode"] = "0"
        ret_obj["msg"] = "SUCCESS"

        return target_num

    def output_format_creator(self, ret_obj: Dict):
        """
        根据不同服务的输出格式,自行添加相应的参数
        如下事例添加time_ms 和noid
        获取noid
        """
        json_str = json.dumps(ret_obj)
        return json_str
