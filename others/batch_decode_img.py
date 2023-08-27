# coding:utf-8

from threading import Thread
import logging
import time
import six

if six.PY2:
    import Queue
elif six.PY3:
    import queue as Queue

from packages.yolo_detect.decode_tools.image_decode.fh_image import fh_imread_by_base64_python_decode, fh_imread_by_binary_python_decode, \
    fh_imread_by_base64_c_decode, fh_imread_by_binary_c_decode
from packages.yolo_detect.decode_tools.exceptions import AILabException, ERROR_IMAGE_DECODE


class ImageProcessor(Thread):
    def __init__(self, q, back, img_type, decode_type):
        Thread.__init__(self)
        self.q = q
        self.back = back
        self.img_type = img_type
        self.decode_type = decode_type

    def run(self):
        while True:
            # 读取到的数据为单个字典，包含id和图片的base64
            image_id, noid, data, decode_type = self.q.get()
            starttime = time.time()
            img = None
            try:
                # cpu decode and input is base64
                if decode_type == 0 and isinstance(data, str):
                    img = fh_imread_by_base64_python_decode(data, self.img_type)
                # cpu decode and input is bytes
                elif decode_type == 0 and isinstance(data, bytes):
                    img = fh_imread_by_binary_python_decode(data, self.img_type)
                # gpu decode and input is base64
                elif decode_type == 1 and isinstance(data, str):
                    img = fh_imread_by_base64_c_decode(data)
                # gpu decode and input is bytes
                elif decode_type == 1 and isinstance(data, bytes):
                    img = fh_imread_by_binary_c_decode(data)
                # 添加图片解码失败的判断
                if img is None:
                    raise AILabException(ERROR_IMAGE_DECODE)
                self.back.put((image_id, img))
            except AILabException as e:
                logging.error("noid: {}, file: {}, preprocess error: {}".format(noid, image_id, e.msg))
                # 返回数据为二元组，id和结果，如果结果为None表示解码失败

                self.back.put((image_id, e.errorDict))

            endtime = time.time()
            logging.debug("noid: {}, img_id: {}, cost: {}s".format(noid, image_id, (endtime - starttime)))


class BatchDecode(object):
    def __init__(self, params):
        """
        批量解码类，内部调用多线程进行处理
        :param batchsize:批次的大小
        :param img_type:解码类型
        :channel 通道数，如果不一致则会强制转成指定的
        :decode_thread_num 线程数
        """

        # 缓冲数据
        self.source_q = Queue.Queue()
        self.back_q = Queue.Queue()

        self.decode_type = int(params['preprocess']['decode_type'])

        # 1表示要转成bgr， 0表示要转成gray,默认bgr
        img_type = int(params['preprocess']['cpu_decode_output_img_type'])

        try:
            self.cpu_decode_thread_num = int(params['preprocess']['cpu_decode_thread_num'])
            logging.info("cpu_decode_thread_num: {}".format(self.cpu_decode_thread_num))
        except:
            logging.warning("cpu_decode_thread_num use default value: 16")
            self.cpu_decode_thread_num = 16

        for _ in range(self.cpu_decode_thread_num):
            p = ImageProcessor(self.source_q, self.back_q, img_type, self.decode_type)
            p.daemon = True
            p.start()

    def process(self, noid, images):
        """
        批量解码并批量返回
        :param noid: 请求id标识
        :param images: [[]] 多，每批次中多个数据
        :return: 二元组：fail和success，每个都是字典表示解码失败和成功的
        """
        starttime = time.time()
        for id, image in enumerate(images):
            # 传进来的batch图片，每一个都是字典
            self.source_q.put((id, noid, image['image'], self.decode_type))
            # self.source_q.put((id, noid, image))

        batch_size = len(images)
        logging.debug("noid: {}, batch size: {}".format(noid, batch_size))

        fail = {}
        success = {}
        for _ in range(batch_size):
            idx, res = self.back_q.get()
            if isinstance(res, dict):
                fail[idx] = res
            else:
                success[idx] = res
        endtime = time.time()
        logging.debug("noid: {}, decode cost: {}s".format(noid, (endtime - starttime)))
        return success, fail

    def process_cpudec(self, noid, images):
        pattern = self.decode_type
        self.decode_type = 0
        success, fail = self.process(noid, images)
        self.decode_type = pattern
        return success, fail
