# -*- coding: utf-8 -*-
# @Time    : 2023/4/20 21:23
# @Author  : yblir
# @File    : collector.py
# explain  :
# ================================================================================
import os
import uuid
import json
import requests
import time
import threading
from packages.yolo_detect.transfer import config, logger
from packages.yolo_detect.utils.exceptions import AILabException, ErrorCode

# 异常处理
ailab_error = ErrorCode()

collector_total_images = 0
collector_target_num = 0
collector_decode_fail_num = 0
collector_other_fail_num = 0
lock = threading.Lock()


class Collector(object):
    def __init__(self, config_path="./config/app.yaml"):
        self.post_url = None
        try:
            # 读取配置文件中对应的参数
            params = config(config_path)
            self.server_port = config['server_port']
            self.post_url = config['post_url']
            self.send_info_time_interval = int(config['send_info_time_interval'])
        except Exception as _:
            assert AILabException(ailab_error.ERROR_COLLECTOR_INIT_FAILD)

        # 开启后台守护，用于收集本地信息并定期向服务器发送收集信息
        if self.post_url:
            # 获取相关信息
            self.instance = self.get_instance()
            self.real_ip = self.get_real_ip()
            self.version = self.get_version()
            self.server_name = self.get_server_name()
            self.pod_name = self.get_pod_name()
            url_path = "/multimedia/avatar/collector/v1/pushailab"

            self.last_status = True  # 记录发送请求的返回状态(判断是否连通)
            self.last_response_status = True  # 当发送请求状态正确的时候，返回具体的响应状态

            # 准备发送请求
            if self.post_url[:4] != "http":
                self.post_url = "http://" + self.post_url
            self.post_url = self.post_url + url_path

            host_process = threading.Thread(target=self.host, args=())
            host_process.start()
        else:
            logger.info("cannot find the post_url in app.yaml, cannot use collector.")

    def post(self, post_url, metrics):

        post_data = {
            "noid"      : str(uuid.uuid1()),
            "instance"  : self.instance,
            "realIp"    : self.real_ip,
            "version"   : self.version,
            "serverName": self.server_name,
            "podName"   : self.pod_name,
            "metrics"   : metrics
        }
        logger.debug("post_data={}, post_url={}".format(post_data, self.post_url))
        # print("post_data={}, post_url={}".format(post_data, self.post_url))
        try:
            _headers = {'Content-Type': 'application/json;charset=utf8'}
            response = requests.post(post_url, data=json.dumps(post_data), headers=_headers, timeout=7)
            self.last_status = True

            data = response.json()
            logger.debug("data={}".format(data))
            # print("data={}".format(data))
            if int(data['status']) != 200:
                self.last_response_status = False
                logger.error("cannot post info to collector server. response info = {}".format(data))
            else:
                self.last_response_status = True
        except Exception as e:
            if self.last_status:
                logger.error("something was wrong when send post, error info={}".format(e))
            self.last_status = False
            self.last_response_status = False

    @staticmethod
    def to_queue(_total_images, _target_num, _decode_fail_num, _other_fail_num):

        global collector_total_images
        global collector_target_num
        global collector_decode_fail_num
        global collector_other_fail_num

        # 上锁
        lock.acquire()
        collector_total_images += _total_images
        collector_target_num += _target_num
        collector_decode_fail_num += _decode_fail_num
        collector_other_fail_num += _other_fail_num
        # 锁释放
        lock.release()

    def host(self):
        """
        采集数据的守护进程，用于收集发送信息和向服务发送信息。
        :return:
        """
        last_time = time.time()

        global collector_total_images
        global collector_target_num
        global collector_decode_fail_num
        global collector_other_fail_num

        while True:
            now_time = time.time()
            time_interval = now_time - last_time
            if time_interval >= self.send_info_time_interval:
                # 更新上一次发送的时间
                last_time = now_time

                # 上锁
                lock.acquire()
                metrics = {
                    "TotalNum"     : collector_total_images,
                    "TargetNum"    : collector_target_num,
                    "DecodeFailNum": collector_decode_fail_num,
                    "OtherFailNum" : collector_other_fail_num
                }

                # 锁释放
                lock.release()

                # 向服务器发送数据
                self.post(self.post_url, metrics)

                # 当上一个发送请求被服务器成功接收了，则清楚状态。
                if self.last_response_status:
                    collector_total_images = 0
                    collector_target_num = 0
                    collector_decode_fail_num = 0
                    collector_other_fail_num = 0
            else:
                time.sleep(0.01)

    @staticmethod
    def get_real_ip():
        """
        获取服务器的真实IP地址
        :return:
        """
        host_ip = os.getenv("HOST_IP")
        if not host_ip:
            host_ip = "UNKNOWN"
        return host_ip

    @staticmethod
    def get_version():
        """
        获取该服务的版本号
        :return:
        """
        # 用于记录微服务的版本
        _version = os.getenv("IMAGE_TAG")
        if not _version:
            _version = "UNKNOWN"
        return _version

    @staticmethod
    def get_server_name():
        """
        获取服务名称
        :return:
        """
        image_name = os.getenv("$IMAGE_NAME")
        if not image_name:
            image_name = "UNKNOWN"
        return image_name

    def get_instance(self):
        """
        获取实例名
        :return:
        """
        tmp = os.popen("ifconfig | grep inet | grep -v 127.0.0.1")
        ip_list = tmp.readlines()
        try:
            ip = ip_list[0].split()[1]
        except Exception as _:
            ip = "UNKNOWN"

        instance_name = ip + ":" + str(self.server_port)
        return instance_name

    @staticmethod
    def get_pod_name():
        tmp = os.getenv("POD_NAME")
        if not tmp:
            tmp = "UNKNOWN"
        return tmp


if __name__ == '__main__':
    config_path = "../config/app.yaml"
    collector = Collector(config_path)

    while True:
        total_images = 1
        target_num = 2
        decode_fail_num = 3
        other_fail_num = 4
        collector.to_queue(total_images, target_num, decode_fail_num, other_fail_num)
        time.sleep(0.0005)
