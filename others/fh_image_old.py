# -*- coding:utf-8 -*-
# @Time     :2018/8/6 13:38
# @Author   :zhuhejun


import cv2

cv2.setNumThreads(1)
import base64
import numpy as np
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

from packages.yolo_detect.decode_tools.exceptions import AILabException, ERROR_IMAGE_DECODE, ERROR_IMAGE_BASE64_DECODE, ERROR_RETURN_TO_BINARY_ERROR


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


def fh_imread_by_path(img_path, img_type=1):
    """
    使用opencv读取图片，并对异常图片进行判断
    :param img_path:图片的路径
    :param img_type:1表示BGR，0表示GRAY
    :return: 图片的numpy矩阵，异常时为None
    """
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print('decode img error: {}\t{}'.format(img_path, 'NoneType'))
        return None
    if img_type:
        img = check_bgr(img, img_path)
    else:
        img = check_gray(img, img_path)
    return img


def fh_imread_by_base64_python_decode(img_data, img_type=1):
    """
    使用base64读取图片，并对图片异常判断
    :param img_data， 图片的base64编码
    :param img_type: 1表示BGR, 0表示GRAY
    :return 图片的numpy矩阵， 异常时为None
    """
    try:
        img_b64decode = base64.b64decode(img_data)
    except BaseException as _:
        raise AILabException(ERROR_IMAGE_BASE64_DECODE)
    try:
        img_array = np.frombuffer(img_b64decode, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
        if img.dtype != 'uint8':
            print('convert dtype to uint8 by imdecode again.')
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    except BaseException as _:
        raise AILabException(ERROR_IMAGE_DECODE)
    if img is None:
        # print('decode img error:\t{}'.format('NoneType'))
        raise AILabException(ERROR_IMAGE_DECODE)
    if img_type:
        img = check_bgr(img, "")
    else:
        img = check_gray(img, "")
    return img


def fh_imread_by_binary_python_decode(img_data, img_type=1):
    """
    使用二进制读取图片，并对图片异常判断
    :param img_data， 图片的二进制编码
    :param img_type: 1表示BGR, 0表示GRAY
    :return 图片的numpy矩阵， 异常时为None
    """
    try:
        img_array = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
        if img.dtype != 'uint8':
            print('convert dtype to uint8 by imdecode again.')
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    except BaseException as _:
        raise AILabException(ERROR_IMAGE_DECODE)
    if img is None:
        raise AILabException(ERROR_IMAGE_DECODE)
    if img_type:
        img = check_bgr(img, "")
    else:
        img = check_gray(img, "")
    return img


def fh_imread_by_base64_c_decode(img_data):
    """
    通过流行是将base64数据转换为ndarry对象
    :param img_data， 图片的base64编码
    :return 未解码图片的numpy矩阵
    """
    try:
        img_b64decode = base64.b64decode(img_data)
    except BaseException as _:
        raise AILabException(ERROR_IMAGE_BASE64_DECODE)
    try:
        img_array = np.frombuffer(img_b64decode, np.uint8)
    except BaseException as _:
        raise AILabException(ERROR_RETURN_TO_BINARY_ERROR)
    return img_array


def fh_imread_by_binary_c_decode(img_data):
    """
    通过流行是将二进制数据转换为ndarry对象
    :param img_data， 图片的二进制编码
    :return 未解码图片的numpy矩阵
    """
    try:
        img_array = np.frombuffer(img_data, np.uint8)
    except BaseException as _:
        raise AILabException(ERROR_RETURN_TO_BINARY_ERROR)
    return img_array
