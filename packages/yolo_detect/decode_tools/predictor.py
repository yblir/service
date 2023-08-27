# -*- coding: utf-8 -*-
# @Time     : 2022/7/12 16:57
# @Author   : LiqunWang
# @FileName : image_decode.py
from typing import Dict
import json
import os
import re
import time
from jsonschema import validate
from jsonschema.exceptions import ValidationError
from abc import ABCMeta, abstractmethod
from .exceptions import AILabException, AILabError

# 异常处理
ailab_error = AILabError()


class Predictor(metaclass=ABCMeta):
    """
    建立Predictor抽象基类,用来定义Predictor的通用特性,虚函数不能直接被实例化,需要被继承,且实现对应的虚函数
    pre_process和post_process为虚函数,需要根据不同算法进行自定义实现

    Predictor的功能如下:
        1. 处理所有与模型推理有关的错误
        2. predict() 将所有模型的推理分为三个步骤:前处理,推理和后处理
        3. pre_process() 做前处理相关操作,包括:解析输入数据,对数据解码
        4. post_process() 将推理后的数据转成微服务输出的格式
        5. 返回推理状态信息: 请求总数,目标数量,解码失败数量和推理失败数量
    """

    def __init__(self, params, decoder, module_infer, json_schema_path: str = "", json_schema_output_path: str = ""):
        """
        初始化推理模块
        :param params: 参数
        :param decoder: 解码模块
        :param module_infer: 推理模块
        :param json_schema_path: json_schema路径
        """
        self.params = params  # 参数配置
        self.module = module_infer  # 初始化完成后的推理模块类
        self.decoder = decoder  # 初始化完成后的解码模块类
        self.request_data = None  # 请求数据,用于获取请求数据内的参数
        self.schema_pattern = re.compile(r'.*is too long')  # schema的模块
        if os.path.exists(json_schema_path):
            self.json_schema = json.load(open(json_schema_path))  # 输入的校验schema文档
        else:
            self.json_schema = None

        if os.path.exists(json_schema_output_path):
            self.json_schema_output = json.load(open(json_schema_output_path))  # 输出的校验schema文档
        else:
            self.json_schema_output = None

        # 初始化完成
        logging.info('Predictor init')

    def check(self, input_data: Dict):
        """
        对请求数据进行格式校验。如果没有指定校验文件，则不进行校验
        Arguments:
            input_data {Dict} -- 请求数据

        """
        if self.json_schema:
            try:
                validate(instance=input_data, schema=self.json_schema)
                valid_params = self.json_schema['properties'].keys()
                for k in input_data.keys():
                    if k not in valid_params:
                        assert False, "Undefined Parameters Found !"
            except ValidationError as v_e:
                if re.match(self.schema_pattern, v_e.message):
                    logging.error("input_data nums is {}, list len too large.".format(len(input_data['images'])))
                    raise AILabException(ailab_error.ERROR_PARAMETER)
                else:
                    logging.info("validator error:{}".format(v_e.message))
                    raise AILabException(ailab_error.ERROR_PARAMETER)
            except Exception as e:
                logging.error(e)
                raise AILabException(ailab_error.ERROR_PARAMETER)

    def check_out(self, output_data):
        """
        对请求数据进行格式校验,如果没有指定校验文件,则不进行校验
        :param output_data: 输出数据
        """
        try:
            if self.json_schema_output:
                validate(instance=output_data, schema=self.json_schema_output)
                valid_params = self.json_schema_output["properties"].keys()
                for k in output_data.keys():
                    if k not in valid_params:
                        assert False, "Undefined Parameters Found !"
        except ValidationError as v_e:
            logging.info("validator error:{}".format(v_e.message))
            raise AILabException(ailab_error.ERROR_CHECK_OUTPUT_FORMAT)
        except Exception as e:
            logging.error(e)
            raise AILabException(ailab_error.ERROR_CHECK_OUTPUT_FORMAT)

    def pre_process(self, request):
        """
        前处理,需要根据不同算法进行实现
        :param request: 请求数据
        :return 两个字典,分别为decode_success,decode_fail
        """
        raise NotImplementedError

    def post_process(self, ret_obj: Dict):
        """
        后处理,需要根据不同算法进行实现,将后处理结果更新至ret_obj
        :return 返回目标的数量
        """
        raise NotImplementedError

    def predict(self, request_data: Dict):
        """
        接收推理请求体数据,并将结果通过字典赋值给ret_obj
        :return: 微服务的返回信息(Dict),请求总数(int),目标数量(int),解码失败数量(int),推理失败数量(int)
        """
        ret_obj = {}
        self.request_data = request_data

        # 1.解析请求体
        decode_time = time.time()
        self.start_time = decode_time
        decode_success, decode_fail = self.pre_process(request_data)
        self.decode_fail = decode_fail
        logging.debug("decode time: {}".format(time.time() - decode_time))

        # 2.推理模块
        # 解码成功数据大于0，则进行推理
        predict_fail = {}
        if len(decode_success) > 0:
            infer_time = time.time()
            predict_success, predict_fail = self.module.infer(decode_success)
            self.predict_success = predict_success
            self.predict_fail = predict_fail
            logging.debug("infer time: {}".format(time.time() - infer_time))
        else:
            logging.debug("cannot decode any images from request.")
            self.predict_success = {}
            self.predict_fail = {}

        # 3.根据请求体，对结果进行后处理, 转成对应微服务输出格式
        post_process_time = time.time()
        result, target_num = self.post_process(ret_obj)
        logging.debug("post time: {}".format(time.time() - post_process_time))

        # 记录每个batch处理情况，成功多少张，失败多少张，解码失败多少张，结果为空多少张
        decode_fail_num = len(decode_fail.keys())
        predict_fail_num = len(predict_fail.keys())

        total_data_num = target_num + decode_fail_num + predict_fail_num

        return ret_obj, total_data_num, target_num, decode_fail_num, predict_fail_num

    def output_format_creator(self, ret_obj: Dict):
        """
        微服务输出格式包装
        :param ret_obj: 推理输出的结果
        :return
        """

        raise NotImplementedError
