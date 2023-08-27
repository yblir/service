# -*- coding: utf-8 -*-
# @Time    : 2023/6/20 16:03
# @Author  : yblir
# @File    : fh_image.py
# explain  :
# ================================================================================
import cv2
import numpy as np
import time
import base64
import logging

from ..base_decoder import BaseDecoder, BaseDecoderMultiThread
from ..exceptions import AILabException, AILabError

# 异常处理
ailab_error = AILabError()


class ImageDecodeCPU(BaseDecoder):
    """
    图片解码模块（单线程CPU版）
    """

    def __init__(self, params):
        super(ImageDecodeCPU, self)._init__(params=params)

    def decode_one_data(self, one_data, img_type=1):
        """
        图片解码
        :param one_data：解base64/解二进制之后的数据
        :param img_type: 1表示BGR, 0表示GRAY
        :return：状态码，结果数据（状态码：解码正确True,解码失败False)
        """
        try:
            img_array = np.frombuffer(one_data, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
            if img.dtype != 'uint8':
                print('convert dtype to uint8 by imdecode again.')
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        except BaseException as _:
            logging.warning(ailab_error.ERROR_IMAGE_DECODE)
            return False, ailab_error.ERROR_IMAGE_DECODE

        if img is None:
            logging.warning(ailab_error.ERROR_IMAGE_DECODE)
            return False, ailab_error.ERROR_IMAGE_DECODE
        if img_type:
            img = check_bgr(img, "")
        else:
            img = check_gray(img, "")

        return True, img


class ImageDecodeCPUThread(BaseDecoderMultiThread):
    """
    图片解码模块（单线程GPU版）
    """

    def __init__(self, params, thread_num=16):

        """
        初始化
        : param params: 配置参数
        : param thread_num: 多线程个数
        """
        super(ImageDecodeCPUThread, self).__init__(thread_num=thread_num)

    def decode_one_data(self, one_data, thread_id):
        """
        图片解码
        :param one_data: 解base64/解二进制之后的数据
        :param thread_id: 线程id
        :return：状态码,结果数据 (状态码：解码正确True，解码失败False)
        """
        try:
            img_array = np.frombuffer(one_data, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
            if img.dtype != 'uint8':
                print('convert dtype to uint8 by imdecode again.')
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        except BaseException as _:
            logging.warning(ailab_error.ERROR_INAGE_DECODE)
            return False, ailab_error.ERROR_IMAGE_DECODE

        if img is None:
            logging.warning(ailab_error.ERROR_IMAGE_DECODE)
            return False, ailab_error.ERROR_IMAGE_DECODE

        img = check_bgr(img, "")

        return True, img


class ImageDecodeGPU(BaseDecoder):
    def __init__(self, params):
        super(ImageDecodeGPU, self).__init_(params=params)
        from .fh_image_nvdecode import FAImageDecoder, FA_IMAGE_OUTPUT_FMT_RGB, FA_IMAGE_OUTPUT_FMT_BGR
        import nvidia.dali.types as types

        gpu_id = int(params['base']['device_id'])
        image_gpu_decode_thread_num = int(params['decode']['image_gpu_decode_thread_num'])
        gpu_decode_batch_size = int(params["decode"]["image_gpu_decode_batchsize"])
        gpu_ouput_img_type = int(params['decode']['image_gpu_decode_ouput_img_type'])
        ouput_img_type = FA_IMAGE_OUTPUT_FMT_RGB
        if gpu_ouput_img_type == 0:
            ouput_img_type = FA_IMAGE_OUTPUT_FMT_RGB
        elif gpu_ouput_img_type == 1:
            ouput_img_type = FA_IMAGE_OUTPUT_FMT_BGR
        try:
            max_size = eval(params['decode']['image_buffer_max_size'])
        except Exception as e:
            max_size = 1920 * 1080 * 3
            logging.error(e)
        max_size = max_size * gpu_decode_batch_size
        self.nvdecoder = FAImageDecoder(batch_size=gpu_decode_batch_size,
                                        device_id=gpu_id,
                                        nthread=image_gpu_decode_thread_num,
                                        output_type=ouput_img_type,
                                        max_size=max_size)

    def decode_one_data(self, one_data):
        try:
            img_array = np.frombuffer(one_data, np.uint8)

        except BaseException as _:
            return False, self.ailab_error.ERROR_DECODE
        return True, img_array

    def cpu_decode_process(self, images_nodecodes_list, unzip_success_key_list):
        """
        若GPU解码失败,则使用CPU进行解码

        :param images_nodecodes_list: unzip之后的图片数据
        :param unzip_success_key_list: 图片对应的key值，用于索引是第几个数据
        :return:
        """
        success = {}
        fail = {}
        img = None
        for index in range(images_nodecodes_list):
            one_data = images_nodecodes_list[index]
            key = unzip_success_key_list[index]
            try:
                img = cv2.imdecode(one_data, cv2.IMREAD_UNCHANGED)
                if img.dtype != 'uint8':
                    print('convert dtype to uint8 by imdecode again.')
                    img = cv2.imdecode(one_data, cv2.IMREAD_COLOR)
            except BaseException as _:
                logging.warning(ailab_error.ERROR_IMAGE_DECODE)
                fail[key] = ailab_error.ERROR_IMAGE_DECODE

            if img is None:
                logging.warning(ailab_error.ERROR_IMAGE_DECODE)
                fail[key] = ailab_error.ERROR_IMAGE_DECODE

            img = check_bgr(img, "")
            success[key] = img

        return success, fail

    def decode(self, raw_data):
        success = {}
        fail = {}
        # 判断请求是单个数据还是batch数据(batch数据为list格式)
        if not isinstance(raw_data, list):
            # 单个请求,改成list格式
            raw_data = [raw_data]
        images_nodecodes_list = []  # 等待GPU解码图片
        unzip_success_key_list = []  # 记录unzip成功的key值，用于对解码后的图片进行排序
        for index, one_data in enumerate(raw_data):
            status, unzip_data = self.unpackage_request(one_data)  # 解base64或二进制数据
            if status:
                status, _decoded_data = self.decode_one_data(unzip_data)
                if status:
                    images_nodecodes_list.append(_decoded_data)
                    unzip_success_key_list.append(index)
                else:
                    fail[index] = _decoded_data
            else:
                fail[index] = unzip_data

        if len(images_nodecodes_list) == 0:
            # 没有unzip成功的数据，则直接返回
            return success, fail

        success, _fail = self.gpu_decode(unzip_success_key_list, images_nodecodes_list)
        fail = {**_fail, **fail}  # 将失败的结果统一起来

        return success, fail

    def gpu_decode(self, unzip_success_key_list, images_nodecodes_list):
        fail = {}
        success = {}
        ret_size, ret_ptrs, ret_heights, ret_widths, ret_channels = \
            self.nvdecoder.batch_decode_buffer_to_host(images_nodecodes_list)
        succ_num = len(images_nodecodes_list)
        if ret_size != succ_num:
            logging.warning('nvdecoder has error, ret_size=(), need_decode_num={}'.format(ret_size, succ_num))
            if ret_size == 0:
                for i in range(succ_num):
                    fail[unzip_success_key_list[i]] = self.ailab_error.ERROR_IMAGE_DECODE
            elif ret_size < 0:
                logging.warning('nvdecoder error: ret_size={}, using cpu decode.'.format(ret_size))
                _success, _fail = self.cpu_decode_process(images_nodecodes_list, unzip_success_key_list)
                # 将CPU解码结果和之前的结果进行拼接
                success = {**_success, **success}
                fail = {**_fail, **fail}
            else:
                for i in range(succ_num):
                    if ret_heights[i] == 0:
                        fail[unzip_success_key_list[i]] = self.ailab_error.ERROR_IMAGE_DECODE
                    else:
                        success[unzip_success_key_list[i]] = ret_size, ret_ptrs[i], ret_heights[i], \
                                                             ret_widths[i], ret_channels[i]
        else:
            for i in range(ret_size):
                success[unzip_success_key_list[i]] = ret_size, ret_ptrs[i], ret_heights[i], \
                                                     ret_widths[i], ret_channels[i]

        return success, fail


def check_bgr(src, img_path):
    """
    检查BGR图片的通道数
    :param src: 待检查图片
    :param img_path: 图片的路径
    :return: 正常读取为numpy矩阵，否则为None
    """
    if len(src.shape) == 2:
        img = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
        # print('Warning:{} is gray'.format(img_path))
        return img
    elif len(src.shape) != 3:
        # print('decode img error:{}\t{}'.format(img_path, 'shape length is not 2/3'))
        return None
    elif src.shape[2] == 4:
        # print('Warning:{} is BGRA'.format(img_path))
        img = cv2.cvtColor(src, cv2.COLOR_BGRA2BGR)
        return img
    elif src.shape[2] != 3:
        # print('decode img error:{}\t{}'.format(img_path, 'channel is not 3/4'))
        return None
    return src


def check_gray(src, img_path):
    """
    检查灰度图片的通道数
    :param src: 待检查图片
    :param img_path: 图片的路径
    :return: 正常读取为numpy矩阵，否则为None
    """
    if len(src.shape) == 2:
        return src
    elif len(src.shape) != 3:
        # print('decode img error:{}\t{}'.format(img_path, 'shape length is not 2/3'))
        return None
    elif src.shape[2] == 4:
        # print('Warning:{} is BGRA'.format(img_path))
        img = cv2.cvtColor(src, cv2.COLOR_BGRA2GRAY)
        return img
    elif src.shape[2] == 3:
        # print('Warning:{} is BGR'.format(img_path))
        img = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        return img
    else:
        # print('decode img error:{}\t{}'.format(img_path, 'channel is not 3/4'))
        return None
