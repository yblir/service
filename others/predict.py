# -*- coding: utf-8 -*-
# @Time    : 2023/4/13 9:03
# @Author  : yblir
# @File    : predict.py
# explain  : 
# ================================================================================
import uuid
import time
import logging
from typing import Dict
from fhaiservice.predictor import Predictor
from fhaiservice.exceptions import AILabException, AILabError

# 异常处理
ailab_error = AILabError()

# todo 修改Demo类为自己的模型的推理过程
class DemoPredictor(Predictor):
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
        初始化predictor
        :param params:配置参数
        :param decoder:初始化完成后的解码模块类
        :param module_infer:初始化完成后的推理模块类
        :param json_schema_path:输入的校验schema文档
        :param data_type_keyword:数据关键字，，用于提取发送谎求中的数据
        :param json_schema_output_path:输出的校验ggheme文档
        """

        super(DemoPredictor, self).__init__(params, decoder, module_infer, json_schema_path, json_schema_output_path)
        self.module_infer = module_infer
        self.decoder = decoder
        self.data_type_keyword = data_type_keyword
        ###########
        # 自定义的参数
        ##############

        logging.info("init demo Predictor")

    def pre_process(self, request):
        """
        解析请术体
        """
        raw_data_for_decode = None
        # 1、根据关键字，将数据从字典中读取出来
        keyword_single = self.data_type_keyword["keyword_single"]
        keyword_batch = self.data_type_keyword["keyword_batch"]

        if keyword_single in request:
            raw_data_for_decode = request[keyword_single]
        # batch请求需将1ist里面的字典数据提取出来
        if keyword_batch in request:
            raw_data_for_decode = []
            data_list = request[keyword_batch]
            for data in data_list:
                raw_data_for_decode.append(data[keyword_single])
        if not raw_data_for_decode:
            # 若请求体中无天健字，则前处理错惧
            logging.error(ailab_error.ERROR_PREPROCESS)
            raise AILabException(ailab_error.ERROR_PREPROCESS)
        # 2.解码换作
        decode_success, decode_fail = self.decoder.decode(raw_data_for_decode)

        return decode_success, decode_fail

    def post_process(self, ret_obj: Dict):
        """
        将结桌打包成接口文档规定的格式，册将理情的果用新里上出
        """
        # 解码相关信息
        decode_fail = self.decode_fail
        # 推理相关信息
        predict_success = self.predict_success
        predict_fail = self.predict_fail
        dict_sort_fail = {}  # 用于存一批数据中返回失败的数据信息,类型型是序典
        dict_sort_success = {}

        # 对解码失败的数据返回对应的错误状态
        for i in decode_fail.keys():
            ret_data = {'data status': decode_fail[i]['status'],
                        'data_msg'   : decode_fail[i]['msg'],
                        'data result': []}
            # 对解码错误的数据结果设为空列表
            dict_sort_fail[i] = ret_data

        # 对推理失败的数据返回对应的错误状态
        for i in predict_fail.keys():
            ret_data = {'data_status': predict_fail[i]['status'],
                        'data msg'   : predict_fail[i]['msg'],
                        'data result': []}
            # 对推理错误的数据结果设为空列表
            dict_sort_fail[i] = ret_data

        # 对推理成功的数据返回对应的推理结果
        target_num = 0  # 记录标数据
        for i in predict_success._keys():
            ret_data = {'data status': "O",
                        'data msg'   : "SUCCESS",
                        'data result': predict_success[i]}
            dict_sort_success[i] = ret_data
            if len(predict_success[i]) != 0:
                target_num += 1

        # 将一批数据中返回成功和失败的两个字典拼接起来
        # todo 为什么出错
        dict_sort = (**dict_sort_fail, **dict_sort_success)  # dict(dictl,**dict2)是两个字典的拼接

        # 按键把字典进行排序，目的是与输入数据的顺序保持一致
        ret_info = []
        for i in sorted(dict_sort):
            ret_info.append(dict_sort[i])

        # 将返回结果放在result中
        ret_obj['results'] = ret_info
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
        ret_obj["time_ms"] = time_ms
        ret_obj["status"] = "0"
        ret_obj["msg"] = "SUCCESS"

        return target_num
