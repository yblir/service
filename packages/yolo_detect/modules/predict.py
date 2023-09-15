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
from base_interface.base_predict import Predictor
from utils.exceptions import AILabException, ErrorCode
from transfer import logger

# 异常处理
error_code = ErrorCode()


# todo 修改Demo类为自己模型的推理过程
class YoloPredictor(Predictor):
    """
    Predictor的功能如下:
    1、处理所有与模型推理相关的事务;
    2.predict()方法:将所有模型的推理分为三步骤:前处理、推理和后处理;
    3.
    pre_process()方法:做前处理相关操作，包括，解析输入数据。对数据解码:
    4.post_process()方法:将推理后的数据转成微服务输出的格式
    5.返回推理状态信息:请求总数、目标数量、解码失败数量和推理失败数量
    """

    def __init__(self, decoder, module_infer,
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
        super(YoloPredictor, self).__init__(json_schema_path, json_schema_output_path)
        self.decode_success = None
        self.decode_fail = None
        # 标记输入时单张还是batch
        self.single_flag = True

        self.total_data_num = 0
        self.decode_fail = None

        self.module = module_infer
        self.decoder = decoder
        # self.data_type_keyword = data_type_keyword

        self.keyword_single = data_type_keyword["single"]
        self.keyword_batch = data_type_keyword["batch"]

        self.start_time = time.time()

        logger.info("init demo Predictor")

    # predict已在基类中实现, 所有服务可公用
    # def predict(self, request_data: Dict):
    #     pass

    # 重写基类pre_process
    def pre_process(self, request):
        """
        解析请求体
        {
            "noid":xxx,
            "sysid":xxx,
            "batch":{
                0:np.ndarray,   # 键值为 int 类型
                1:np.ndarray,
                2:np.ndarray,
                ...
            }
        }
        """
        # 记录预处理开始时间, 在post_process中调用
        self.start_time = time.time()

        # 1、根据关键字，将数据从字典中读取出来
        if self.keyword_single in request:
            self.single_flag = True
            raw_data_for_decode = request[self.keyword_single]
            self.total_data_num = 1
        # batch请求中列表是待推理的单个元素
        elif self.keyword_batch in request:
            # false, 表示batch
            self.single_flag = False
            # raw_data dict, {0:xxx, 1:xxx, ...}
            raw_data_batch = request[self.keyword_batch]
            raw_data_len = len(raw_data_batch)
            raw_data_for_decode = [None] * raw_data_len
            # 通过字典索引找值, 避免字典哈希索引乱序
            for i in range(raw_data_len):
                raw_data_for_decode[i] = raw_data_batch[i]

            self.total_data_num = raw_data_len
        else:
            logging.error("keyword must be single or batch")
            raise AILabException(error_code.ERROR_PARAMETER)

        # 2.解码操作
        # 解码后,batch,解码结果是字典,int 类型0,1,2,为键,值为nd.array格式图片矩阵
        decode_success, decode_fail = self.decoder.decode(raw_data_for_decode)

        return decode_success, decode_fail

    def predict(self, request_data: Dict):
        """
        request_data:通过request直接输入的dict
        共经历三个模块:
            预处理,推理,后处理
        """
        # 先建立一些输出字典的默认返回信息
        ret_obj = {"status_code": "0", "msg": "SUCCESS"}
        # 在后处理中取出uuid
        self.request_data = request_data
        # 只要输入图片格式没问题,不会出现推理失败情况, 最多什么都没预测出来
        predict_fail_num = 0
        # 1.解析请求体, single, batch,解码结果都是dict,int 类型0,1,2,为键,值为nd.array格式图片矩阵,decode_fail也是dict
        decode_success, self.decode_fail = self.pre_process(request_data)
        # 在后处理中有应用
        self.decode_success = decode_success

        # 2.推理模块
        if decode_success:
            # 将dict转为仅含numpy矩阵的list
            decode_success = [decode_success[i] for i in range(len(decode_success))]
            future_infer_result = self.module.module_infer(decode_success)
        else:
            logger.error("cannot decode any images from request.")
            raise

        if not future_infer_result:
            # 错误日志在module_infer中已经写过
            raise
        ret_obj["result"] = future_infer_result

        # 计算解码失败数量
        decode_fail_num = len(self.decode_fail)

        # 3.根据请求体，对结果进行后处理，转成微服务输出格式
        # todo 返回的结果的格式在后post_process定义
        target_num = self.post_process(ret_obj)

        return ret_obj, self.total_data_num, target_num, decode_fail_num, predict_fail_num

    # 重写基类post
    def post_process(self, ret_obj: Dict):
        """
        将结果打包成接口文档规定的格式，将处理后的结果更新至ret_obj输出
        """
        dict_sort_success, dict_sort_fail = {}, {}  # 用于存放一批数据中返回失败的数据信息,类型是字典
        # 获取noid
        try:
            # 发送请求获得的数据
            noid = self.request_data['noid']
        except Exception as _:
            noid = str(uuid.uuid1())

        ret_obj["noid"] = noid if noid else str(uuid.uuid1())

        # 收集解码失败信息
        for i in self.decode_fail.keys():
            dict_sort_fail[i] = {
                "status": self.decode_fail[i]["status"],
                "msg"   : self.decode_fail[i]["msg"],
                "result": []
            }

        # 收集推理失败的信息? 应该不存在

        # 通过get从C++取回推理结果, 如果没有推理完成,在此阻塞,直到拿到结果
        cur_infer_result = ret_obj["result"].get()

        if self.single_flag:
            ret_info, target_num = (cur_infer_result[0], 1) if self.decode_success else (dict_sort_fail[0], 0)
        else:
            # 将推理结果与图片编号对应起来, 因为有解码失败的风险存在, 所以decode_success中编号有可能不连续
            keys = sorted(self.decode_success.keys())
            dict_sort_success = dict(zip(keys, cur_infer_result))
            target_num = len(keys)
            # 将一批数据中返回成功和失败的两个字典拼接起来
            ret_info = {**dict_sort_fail, **dict_sort_success}  # dict(dict1,**dict2)是两个字典的拼接

        # 将返回结果放在result中
        ret_obj['result'] = ret_info

        ret_obj["cost_time"] = str(round(int((time.time() - self.start_time) * 1000), 3)) + " ms"

        return target_num

