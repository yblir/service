# -*- coding: utf-8 -*-
# @Time    : 2023/6/20 18:25
# @Author  : yblir
# @File    : dasda.py
# explain  : 
# =======================================================
import numpy as np
from pathlib2 import Path
from pprint import pprint
import random
import warnings

warnings.filterwarnings("ignore")
dir_path=Path(r"E:\BaiduNetdiskDownload\service\packages\face_attribute_acc")

# a=[i for i in dir_path.iterdir()]
# pprint(len(a))
# print("-======================================")
# pprint(a)
b=np.linspace(0,9,min(15,10),dtype=np.int)
print(len(b),b)

def predict(self, request data: Dict):
    """
    接收请求体数据，并将结果通过字典类型数ret_obj
    :Rm request data:
    :return:微服务的返回信息(Dict)、请求总数(ant)、目标数量(int)、解码失败数量(ant)。推理失败数量(int)
    """

    ret_obj =()
    self. request_data = request_data
    #1.解析请求体
    decode_time = time. time()
    self. start_time = decode_time
    decode_success, decode_fail = self.pre process(request_data)
    self. decode_fail = decode_fail
    logging.debug("decode time: ()".format(time.time() - decode_time))
    # 替换解码正确的图片，去掉解码失败的
    for i in range(len(request_data[self.data_type_keyword_keyword["keyword batch"]])):
        if i in decode success:
            request_data[self.data_type_keyword["keyword_batch"]][i]["imageData"] = decode_success[i]
        else:
            request_data[self.data_type_keyword["keyword batch"]][i]["imageData"] = None
    #2.推理模块
    #解码成功数据大于，则进行推理
    predict_fail ={}
    if len(decode_success)> 0:
    infer_time = time. time()
    predict_success, predict_fail = self.module.infer(request_data)
    self.predict_success = predict_success
    self.predict_fail = predict_fail
    logging.debug("infer time: ()".format(time.time() - infer_time))
    else:
    logging.debug("cannot decode any images from request.")
    self.predict_success =()
    self.predict_fail =()
    #3.根据请求体，对结果进行后处理，转成微服务输出格式
    post_process_time = time. time()
    target_num = self.post process(ret_obj)
    logging.debug("post time: (}".format(time.time() - post_process_time))
    #记录每个batch处理情况，成功多少张，失败多少张，解码失败多少张，结果为空多少张
    decode_fail_num = len(decode_fail. keys())
    predict_fail_num = len(predict_fail. keys())
    # total data_num = len(list(decode_success.keys()) + len(list(decode_fail.keys())
    total_data_num = target_num + decode_fail_num + predict_fail_num
    return ret_obj, total_data_num, target_num, decode_fail_num, predict_fail_num
# [WindowsPath('E:/BaiduNetdiskDownload/service/packages/yolo_detect/config'),
#  WindowsPath('E:/BaiduNetdiskDownload/service/packages/yolo_detect/yolo_detect_service.py'),
#  WindowsPath('E:/BaiduNetdiskDownload/service/packages/yolo_detect/decode_tools'),
#  WindowsPath('E:/BaiduNetdiskDownload/service/packages/yolo_detect/lib'),
#  WindowsPath('E:/BaiduNetdiskDownload/service/packages/yolo_detect/log'),
#  WindowsPath('E:/BaiduNetdiskDownload/service/packages/yolo_detect/models'),
#  WindowsPath('E:/BaiduNetdiskDownload/service/packages/yolo_detect/modules'),
#  WindowsPath('E:/BaiduNetdiskDownload/service/packages/yolo_detect/__init__.py')]