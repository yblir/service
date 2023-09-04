# -*- coding: utf-8 -*-
# @Time    : 2023/6/11 18:38
# @Author  : yblir
# @File    : base_decoder.py
# explain  :
# ==============================================================
import time
import base64
import numpy as np
import six

import queue as Queue
from threading import Thread
from ..util_func.exceptions import AILabException, AILabError, AILabException, ERROR_IMAGE_DECODE
from ..transfer import logger


def unpackage_request(raw_data):
    """
    将base64数据或二进制数据解成原生数据，为解码做准备
    param: ailab_error:
    :param raw_data:
    :return: 状态码,结果数据：状态码正常的时候为True，异常报错为False
    """
    # 二进制数据
    if isinstance(raw_data, bytes):
        return True, raw_data
    # base64数据
    unzip_data = AILabError.ERROR_DECODE
    try:
        if isinstance(raw_data, str):
            unzip_data = base64.b64decode(raw_data)
    except BaseException as _:
        logger.info("ERROR_DECODE_BASE64: decode base64 request error.")
        return False, AILabError.ERROR_DECODE_BASE64

    # 或添加自定义解包
    return True, unzip_data


class BaseDecoder(object):
    """
    BaseDecoder的功能如下：
    1. 处理所有与解码相关的事务
    2. 规范CPU解码和GPU解码的接口：
    3. 规范图片,视频和音频等输入形态的接口，使得框架有较大的兼容性；
    """

    def __init__(self, params, ailab_error=AILabError()):
        """
        解码模块的的始化
        :param: params 参数配置
        :param ailab_error 却始化完成的常处理类
        """
        self.params = params
        self.ailab_error = ailab_error
        self.unpackage_request = unpackage_request

    def decode_one_data(self, one_data):
        """
        根据不同的解码程序，对数据进行解码
        :param one_data 解pase64/解二进制后的数据
        :return:状态码,结果数据( 状应码:解码正确True,解码失败False)
        """
        raise NotImplementedError

    def decode(self, raw_data):
        """
        解码模块的主要解码接口
        :param raw_data 始数据
        :return: 解码之后的2个字典，分别为success，fail
        """
        success = {}
        fail = {}
        # 判断请求是单个数据还是batch数据(batch数据为list格式)
        if not isinstance(raw_data, list):
            # 单个请求
            status, unzip_data = self.unpackage_request(raw_data)  # 解base64或二进制数据
            if status:
                # 解包正确，进一步对数据进行解码
                status, _decoded_data = self.decode_one_data(unzip_data)
                if status:
                    success[0] = _decoded_data
                else:
                    fail[0] = _decoded_data
            else:
                # 解包失败
                fail[0] = unzip_data
        else:
            # 为batch请求
            for index, one_data in enumerate(raw_data):
                status, unzip_data = self.unpackage_request(one_data)  # 解base64或二进制数据
                if status:
                    status, _decoded_data = self.decode_one_data(unzip_data)
                    if status:
                        success[index] = _decoded_data
                    else:
                        fail[index] = _decoded_data
                else:
                    fail[index] = unzip_data

        return success, fail


class BaseDecoderProcessor(Thread):
    def __init__(self, thread_id, consumer_process, source_q, back_q, unpackage_request):
        Thread.__init__(self)

        self.thread_id = thread_id
        self.consumer_process = consumer_process
        self.source_q = source_q
        self.back_q = back_q
        self.unpackage_request = unpackage_request

    def run(self):
        while True:
            index, data = self.source_q.get()
            status, unzip_data = self.unpackage_request(data)  # 解base64或二进制数据
            if status:
                status, _decoded_data = self.consumer_process(unzip_data, self.thread_id)
                self.back_q.put((status, index, _decoded_data))
            else:
                self.back_q.put((status, index, unzip_data))


class BaseDecoderMultiThread(object):
    """
    BaseDecoderMultiThread的功能如下：
    1.使用多线程的方式进行解码，提升解码的效率；
    """

    def __init__(self, thread_num=16, ailab_error=AILabError()):
        """
        初始化
        param thread num:线程数
        :param ailab_error: 初始化完成的异常处理类
        """
        self.thread_num = int(thread_num)
        self.ailab_error = ailab_error
        self.unpackage_request = unpackage_request
        # 缓冲数据
        self.source_q = Queue.Queue()  # 通道的输入，用于数据的传输到解码模块中
        self.back_q = Queue.Queue()  # 通道的输出，用于将解码之后的数据传回
        for index in range(self.thread_num):
            p = BaseDecoderProcessor(index, self.decode_one_data, self.source_q, self.back_q, self.unpackage_request)
            p.daemon = True
            p.start()

    def decode_one_data(self, one_data, thread_id):
        """
        根据不同的解码程序，对数据进行解码
        :param thread_id:制定某个线程执行任务
        :parm one data:解base64/解二进制之后的数据
        :return:状态码,结果数据(状态码解码正确true,解码失败False)
        """
        raise NotImplementedError

    def decode(self, raw_data):
        """
        解码模块的主要解码接口
        param raw_data 原始数据
        :return：解码之后的2个字典，分别为success, fail
        """
        success = {}
        fail = {}
        # 判断请求是单个数据还是batch数据(batch数据为1ist格式)
        if not isinstance(raw_data, list):
            # 单个请求
            status, unzip_data = self.unpackage_request(raw_data)  # 解base64或二进制数据
            if status:
                status, _decoded_data = self.decode_one_data(unzip_data, 0)
                if status:
                    success[0] = _decoded_data
                else:
                    fail[0] = _decoded_data
            else:
                fail[0] = unzip_data
        else:
            # 为batch请求
            for index, one_data in enumerate(raw_data):
                self.source_q.put((index, one_data))

            batch_size = len(raw_data)

            for _ in range(batch_size):
                status, index, res = self.back_q.get()
                if status:
                    success[index] = res
                else:
                    fail[index] = res

        return success, fail
