# coding:utf-8
import json
import logging
import time
import uuid

import six
import tornado.web

from .predictor import Predictor
from .exceptions import *


class APIHandler(tornado.web.RequestHandler):
    """
    API接口处理的handler
    """

    def initialize(self, predictor: Predictor):
        self.predictor = predictor

    def get(self, *args, **kwargs):
        """
        GET请求
        """
        status = {
            "status": "ok",
            "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int(time.time())))
        }
        self.write(status)

    def post_old(self, *args, **kwargs):
        """
        处理post请求
        :param args:
        :param kwargs:
        :return:
        """
        start_time = time.time()
        ret_obj = {
            "status": "0",
            "msg": "SUCCESS"
        }

        request_data = None
        if len(self.request.body) > 200 * 1024 * 1024:
            ret_obj.update(ERROR_REQUEST_FILE_LARGE)
        else:
            request_data = self.get_request_data(request_data, ret_obj)

        try:
            noid = request_data['noid']
        except Exception as _:
            noid = str(uuid.uuid1())

        if noid == "":
            noid = str(uuid.uuid1())

        self.process_request_data(noid, request_data, ret_obj)

        end_time = time.time()
        ret_obj['noid'] = noid
        time_ms = int((end_time - start_time) * 1000)
        ret_obj['time_ms'] = time_ms
        try:
            json_str = json.dumps(ret_obj)
        except Exception as ee:
            logging.exception(ee)
            ret_obj.update(ERROR_RETURN_TO_JSON_ERROR)
            json_str = json.dumps(ret_obj)

        logging.info("noid:{}, cost:{}ms".format(noid, time_ms))
        self.write(json_str)

    def post(self, *args, **kwargs):
        """
        处理post请求
        :param args:
        :param kwargs:
        :return:
        """
        start_time = time.time()
        ret_obj = {
            "status": "0",
            "msg": "SUCCESS",
            "results": {}
        }

        request_data = None
        if len(self.request.body) > 200 * 1024 * 1024:
            ret_obj.update(ERROR_REQUEST_FILE_LARGE)
        else:
            request_data = self.get_request_data(request_data, ret_obj)

        try:
            noid = request_data['noid']
        except Exception as _:
            noid = ''

        self.process_request_data(noid, request_data, ret_obj)

        end_time = time.time()
        ret_obj['noid'] = noid
        time_ms = int((end_time - start_time) * 1000)
        ret_obj['time_ms'] = time_ms
        try:
            json_str = json.dumps(ret_obj)
        except Exception as ee:
            logging.exception(ee)
            ret_obj.update(ERROR_RETURN_TO_JSON_ERROR)
            json_str = json.dumps(ret_obj)

        logging.info("noid:{}, cost:{}ms".format(noid, time_ms))
        self.write(json_str)

    def get_request_data(self, request_data, ret_obj):
        # 传输数据为二进制
        if "Data-Head-Length" in self.request.headers:
            try:
                # 请求体头部长度
                len_header = int(self.request.headers["Data-Head-Length"])
                # 请求体头部
                request_data = json.loads(str(self.request.body[:len_header], "utf-8"))
            except Exception as e:
                logging.error(e)
                ret_obj.update(ERROR_REQUEST_BODY_FORMAT)

        # 传输数据为base64
        else:
            try:
                if six.PY3:
                    request_data = json.loads(self.request.body.decode())
                elif six.PY2:
                    request_data = json.loads(self.request.body)
            except Exception as e:
                logging.error(e)
                ret_obj.update(ERROR_REQUEST_BODY_FORMAT)
        return request_data

    def check_byte_request(self, request_data):
        try:
            if "Data-Head-Length" in self.request.headers:
                # 图片长度列表
                temp = request_data["images"]
                len_images_list = []
                for i in range(len(temp)):
                    len_images_list.append(int(temp[i]['image']))

                total_len = sum(len_images_list)
                # 二进制请求体提取二进制图片
                request_data_images = []
                # 请求体头部长度
                len_header = int(self.request.headers["Data-Head-Length"])

                if (total_len + len_header) != len(self.request.body):
                    raise AILabException(ERROR_REQUEST_BODY_FORMAT)

                data_images_start = len_header
                for i in range(len(len_images_list)):
                    request_data_images.append(
                        {"image": self.request.body[data_images_start: data_images_start + len_images_list[i]]})
                    data_images_start += len_images_list[i]
                request_data['images'] = request_data_images
        except Exception as _:
            raise AILabException(ERROR_REQUEST_BODY_FORMAT)

    def process_request_data(self, noid, request_data, ret_obj):
        if request_data is not None:
            target_num = 0
            decode_fail_num = 0
            other_fail_num = 0
            total_images = 0
            try:
                request_data['noid'] = noid
                self.predictor.check_input(request_data)

                if 'image' in request_data.keys():
                    total_images = 1  # 单张图片接口
                if 'images' in request_data.keys():
                    total_images = len(request_data['images'])  # batch接口
                # 检测是否为二进制方式输入，若为二进制输入，则进行解析
                self.check_byte_request(request_data)
                ret_dict = self.predictor.predict(request_data)
                ret_obj['results'] = ret_dict
                other_fail_num = 0
            except AILabException as e_ai:
                ret_obj.update({
                    'status': e_ai.errorCode,
                    'msg': e_ai.errorType
                })
                if e_ai.errorDict == ERROR_IMAGE_DECODE:
                    decode_fail_num = total_images
                else:
                    other_fail_num = total_images

            except Exception as e:
                logging.exception(e)
                ret_obj.update(ERROR_UNKNOWN)

                other_fail_num = total_images
            finally:
                logging.info("TotalNum:{}, TargetNum:{}, DecodeFailNum:{}, OtherFailNum:{}"
                             .format(total_images, target_num, decode_fail_num, other_fail_num))
                if self.predictor.collector.post_url:
                    self.predictor.collector.to_queue(total_images, target_num, decode_fail_num, other_fail_num)
