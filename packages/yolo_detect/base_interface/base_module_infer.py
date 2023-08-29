# -*- coding: utf-8 -*-
# @Time    : 2023/3/11 19:38
# @Author  : yblir
# @File    : base_module_infer.py
# explain  :
# ==============================================================
# from ..transfer import logger
# from .exceptions import AILabException, AILabError


# 抽象类, 规范推理接口
class BaseModuleInfer:
    """
    推理模块基类的功能:
        1. 该模块聚焦于推理:获取解码后的数据,对数据进行推理,再将推理后的结果输出
    """

    def __init__(self):
        pass
        # 添加模型初始化操作
        # self.params = params
        ########################
        # 根据模型自定义初始化
        ########################

    # @staticmethod
    # def infer(input_data):
    #     """
    #     模型推理样例，
    #     捕获异常的同时，返回每个数据的异常状态到fail中
    #     :param input_data:解码后的数据,待推理
    #     :return: 推理之后的2个字典,分别为success,fail
    #     """
    #     predict_success = {}
    #     predict_fail = {}
    #     for k, v in input_data.items():
    #         # 推理样例，测试异常返回是否正确
    #         try:
    #             if k == 1:  # 第二张图片设置为异常
    #                 assert False, "Error Example"
    #             else:
    #                 predict_success[k] = ["Predict Success"]
    #         except Exception as e:
    #             logger.error("file: {}, predict error: {}".format(k, e))
    #             predict_fail[k] = AILabError.ERROR_INFER_ERROR
    #     return predict_success, predict_fail

    # 由继承该类的具体类实现
    def module_infer(self, input_data):
        raise NotImplementedError
