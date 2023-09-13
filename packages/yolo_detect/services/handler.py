import os
import six
import logging
import json
import time
import tornado.web
from typing import Dict

from packages.yolo_detect.utils.exceptions import AILabException, ErrorCode
from packages.yolo_detect.base_interface.base_predict import Predictor
from packages.yolo_detect.services.collector import Collector

collect = Collector()
# 异常处理
error_code = ErrorCode()


class ReadyHandler(tornado.web.RequestHandler):

    def get(self, *args, **kwargs):
        """
        GET请求，判断微服务是否正常运行
        :param args:基类的接口
        :param kwargs:基类的接口
        :return:返回微服务的状态信息以及对应的时间
        """

        status = {
            "status": 200,
            "time"  : time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int(time.time())))
        }
        self.write(status)


class APIHandler(tornado.web.RequestHandler):
    """
    微服务的请求接口类微服务接收post请求和get请求，基于tornado.web.RequestHandler
    """

    def initialize(self, data_type_keyword_dict: Dict, predictor: Predictor):
        """
        初始化微服务
        :param data_type_keyword_dict: 获取关键字的字典
        :param predictor: 初始化完成的预测模块类
        """
        self.data_type_keyword_list = []  # 获取关键字的字典，用于二进制数据的切分；
        for key in data_type_keyword_dict:
            self.data_type_keyword_list.append(data_type_keyword_dict[key])

        self.predictor = predictor  # 初始化完成的模型推理类

    def get(self, *args, **kwargs):
        """
        GET请求，判断微服务是否正常运行
        :param args: 基类的接口
        :param kwargs: 基类的接口
        :return: 返回微服务的状态信息以及对应的时间
        """
        status = {
            "status": "ok",
            "time"  : time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int(time.time())))
        }
        self.write(status)

    def post(self, *args, **kwargs):
        """
        处理post请术。微服务的主要入口，主要功能如下：
            1.解析输入数据
            2.校验输入数据格式：
            3.将数据转化为字典,交给推理块
            4.接收推理模块的输出，并进行输出格式校验
            5.将输出字典转成json格式，并输出
            6.异常处理:
            7.采集信息，并上传到监测平台
        :param args: 基类的接口
        """

        self.predictor.start_time = time.time()  # 记录开始时间
        target_num = 0
        decode_fail_num = 0
        predict_fail_num = 0
        other_fail_num = 0
        total_data_num = 0
        ret_obj = {}
        # 限制最大请求的尺寸为200M
        if len(self.request.body) > 200 * 1024 * 1024:
            logging.info(error_code.ERROR_REQUEST_FILE_LARGE)
            ret_obj.update(error_code.ERROR_REQUEST_FILE_LARGE)
        else:
            request_data = None
            try:
                # 将json格式数据解析成字典格式
                request_data = self.json_parser(self.request)

                self.parser_request_id(self.request)

                # 检查输入数据的格式是否符合schema的规定
                self.predictor.check_input(request_data)

                # 检测是否为二进制方式输入，若为二进制输入，则进行解析
                self.check_byte_request(request_data)
            except AILabException as e:
                logging.error(e.errorDict)
                ret_obj.update(e.errorDict)
                request_data = None

            if request_data:
                try:
                    # 处理微服务的请求
                    # ret_obj, total_data_num, target_num, decode_fail_num, predict_fail_num = \
                    #     self.predictor.predict(request_data)
                    ret_obj = self.predictor.predict(request_data)
                    # other_fail_num = predict_fail_num
                except AILabException as e:
                    logging.error(e.errorDict)
                    ret_obj.update(e.errorDict)
                except Exception as e:
                    logging.error(error_code.ERROR_UNKNOWN)
                    ret_obj.update(error_code.ERROR_UNKNOWN)
                finally:
                    # # 输出格式检查
                    # 微服务输出格式由写作者控制,不做输出格式校验
                    # try:
                    #     self.predictor.check_out(ret_obj)
                    # except Exception as _:
                    #     logging.debug("cannot check_input output format from self.predictor.check_out")

                    # logging.info("TotalNum:{}, TargetNum:{}, DecodeFailNum:{}, OtherFailNum:{}"
                    #              .format(total_data_num, target_num, decode_fail_num, other_fail_num))
                    if collect.post_url:
                        collect.to_queue(total_data_num, target_num, decode_fail_num, other_fail_num)
        try:
            json_str = self.predictor.output_format_creator(ret_obj)
        except Exception as ee:
            logging.exception(ee)
            ret_obj.update(error_code.ERROR_RETURN_TO_JSON_ERROR)
            json_str = self.predictor.output_format_creator(ret_obj)

        self.write(json_str)

    def parser_request_id(self, request_data):
        tmp = int(time.time() * 1000)
        if self.need_encrypt_or_not():
            if "Request-ID" in request_data.headers:
                request_id = int(request_data.headers["Request-ID"])
                id_data = self.solve(request_id)
                if (int(tmp) - int(id_data)) < 10000:
                    return
                else:
                    raise AILabException(error_code.MULTIMEDIA_AUTH_FAILED)
            else:
                raise AILabException(error_code.MULTIMEDIA_AUTH_FAILED)
        else:
            return

    def json_parser(self, request_data) -> Dict:
        """
        从request请求中解析json数据
        :param request_data:求体数据
        :param ret_obj:返回状态
        :return:解析后的数据，返回数据为Dict格式
        """
        data = None
        # 传输数据为二进制
        if "Data-Head-Length" in request_data.headers:
            try:
                # 请求体头部长度
                len_header = int(request_data.headers["Data-Head-Length"])
                # 请求体头部
                data = json.loads(str(request_data.body[:len_header], "utf-8"))
            except Exception as e:
                logging.error(e)
                raise AILabException(error_code.ERROR_REQUEST_BODY_FORMAT)

        # 传输数据为base64
        else:
            try:
                if six.PY3:
                    data = json.loads(request_data.body.decode())
                elif six.PY2:
                    data = json.loads(request_data.body)
            except Exception as e:
                logging.error(e)
                raise AILabException(error_code.ERROR_REQUEST_BODY_FORMAT)
        return data

    def parse_data_from_byte_request(self, request_data, key):
        """
        从二进制请求中，解析出数据，若为batch请求，则对请求数据进行切分
        :param request_data:发送请求数据
        :param key:根据关键词，从请求体的字典中，获取每个数据的size,根据size进行切分二进制数据
        :return:
        """
        # 根据关键字是否为list类型，来判断是否为batch的数据
        if not isinstance(request_data[key], list):
            # 为单个接口
            data_size = int(request_data[key])
            len_header = int(self.request.headers["Data-Head-Length"])
            if (data_size + len_header) != len(self.request.body):
                raise AILabException(error_code.ERROR_REQUEST_BODY_FORMAT)
            request_data[key] = self.request.body[len_header: len_header + data_size]
        else:
            # 为batch接口
            temp = request_data[key]
            data_len_list = []
            key_name_list = []
            _key_name = "imageData"  # 获取字典中的关键字
            for i in range(len(temp)):
                data_len_list.append(int(temp[i][_key_name]))
                key_name_list.append(_key_name)

            total_len = sum(data_len_list)
            # 二进制请求体提取二进制数据
            #  request data images =[]
            # 请求体头部长度
            len_header = int(self.request.headers["Data-Head-Length"])

            if (total_len + len_header) != len(self.request.body):
                raise AILabException(error_code.ERROR_REQUEST_BODY_FORMAT)

            data_start = len_header
            for i in range(len(data_len_list)):
                temp[i][_key_name] = self.request.body[data_start: data_start + data_len_list[i]]
                data_start += data_len_list[i]
            request_data[key] = temp

    def check_byte_request(self, request_data):
        """
        返回切片后的二进制数据，将数据重组成dict类型数据，
        :param request_data:解1son之后的数据
        :return:切片后的二进制数据
        """
        try:
            if "Data-Head-Length" in self.request.headers:
                # 判断下 self.data type keyword list 是否在输入请求体内
                request_keys = list(request_data.keys())
                for keyword in self.data_type_keyword_list:
                    if keyword in request_keys:
                        self.parse_data_from_byte_request(request_data, keyword)
        except Exception as _:
            raise AILabException(error_code.ERROR_REQUEST_BODY_FORMAT)

    @staticmethod
    def need_encrypt_or_not():
        keyname = "CRITICAL"
        key_value = 1024
        env_value = os.getenv(keyname)
        if env_value is not None:
            env_value = int(env_value)
            if env_value == key_value:
                return False
            else:
                return True
        else:
            return True

    @staticmethod
    def solve(data):
        data = str(data)
        substring = data[3:16]
        data_list = list(map(int, str(substring)))
        for i in range(len(data_list)):
            if 5 < data_list[i] < 10:
                data_list[i] -= 5
            else:
                data_list[i] += 5
                data_list[i] = data_list[i] % 10

        temp_list = [str(i) for i in data_list]
        strr = ""
        temp = strr.join(temp_list)
        ret = int(temp) >> 2

        return ret
