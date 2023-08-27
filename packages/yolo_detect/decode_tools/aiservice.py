# -*- coding: utf-8 -*-
# @Time    : 2023/4/14 8:35
# @Author  : yblir
# @File    : aiservice.py
# explain  :
# ================================================================================
import tornado.ioloop
from .handler import APIHandler, ReadyHandler
from .predictor import Predictor
from ..transfer import logger


class Service(object):
    """
    微服务主类
    """

    def __init__(self, data_type_keyword=None, max_buffer_size=209715200, max_body_size=209715200):
        """服务初始化

        Keyword Arguments:
            max_buffer_size {int} -- [缓冲区最大值] (default: {209715200})
            max_body_size {int} -- [请求体最大值] (default: {209715200})
        """
        self.preditor = None
        self.url = None

        if data_type_keyword is None:
            self.data_type_keyword = {"single": "image", "batch": "images"}
        else:
            # data_type_keyword 请求体中数据的中关键字,dict,用于解析二进制数据, 关键字例如images/image/video/audioData等...
            self.data_type_keyword = data_type_keyword
        logger.info('service start')
        self.max_buffer_size = max_buffer_size
        self.max_body_size = max_body_size
        # 微服务
        # ready_url = "/ai_service/multimedia/ready"
        self.pairs = []
        # self.pairs.append(tuple([ready_url, ReadyHandler]))  # ready接口, 用于检测服务是否运行

    def register(self, url: str, predictor: Predictor):
        """
        指定向外暴露的调用url和推理类
        """
        self.url = url
        self.preditor = predictor

    def run(self, port=8080, debug=False):
        """
        模拟服务启动
        :return:
        """
        settings = {'debug': debug}
        self.pairs.append(
                tuple([self.url, APIHandler,
                       dict(data_type_keyword=self.data_type_keyword, predictor=self.preditor)])
        )
        app = tornado.web.Application(self.pairs, **settings)

        http_server = tornado.httpserver.HTTPServer(app,
                                                    max_buffer_size=self.max_buffer_size,
                                                    max_body_size=self.max_body_size)
        http_server.listen(port)
        logger.success("init finished.")
        tornado.ioloop.IOLoop.current().start()

    def register_list(self, params_list):
        for url, predictor in params_list:
            self.pairs.append(tuple([url, APIHandler, dict(data_type_keyword=self.data_type_keyword,
                                                           predictor=predictor)]))

    def run_list(self, port=8080, debug=False):
        """
        模拟服务启动
        :return:
        """
        settings = {'debug': debug}
        app = tornado.web.Application(self.pairs, **settings)

        http_server = tornado.httpserver.HTTPServer(app,
                                                    max_buffer_size=self.max_buffer_size,
                                                    max_body_size=self.max_body_size)
        http_server.listen(port)
        logger.success("init finished.")
        tornado.ioloop.IOLoop.current().start()
