# -*- coding: utf-8 -*-
# @Time     : 2022/7/12 16:57
# @Author   : LiqunWang
# @FileName : image_decode.py
import time
import logging
from packages.yolo_detect.decode_tools.base_decoder import BaseDecoder

from packages.yolo_detect.decode_tools.image_decode.fh_image import fh_imread_by_base64_python_decode, fh_imread_by_binary_python_decode, \
    fh_imread_by_base64_c_decode, fh_imread_by_binary_c_decode
from packages.yolo_detect.decode_tools.exceptions import AILabException, ERROR_IMAGE_DECODE


class ImageDecoderSingle(BaseDecoder):
    def __init__(self, params):
        super().__init__(params)
        self.img_type = int(params['preprocess']['cpu_decode_output_img_type'])

    def image_decode(self, noid, image):
        try:
            # cpu decode and input is base64
            if self.decode_type == 0 and isinstance(image, str):
                img = fh_imread_by_base64_python_decode(image, self.img_type)
            # cpu decode and input is bytes
            elif self.decode_type == 0 and isinstance(image, bytes):
                img = fh_imread_by_binary_python_decode(image, self.img_type)
            # gpu decode and input is base64
            elif self.decode_type == 1 and isinstance(image, str):
                img = fh_imread_by_base64_c_decode(image)
            # gpu decode and input is bytes
            elif self.decode_type == 1 and isinstance(image, bytes):
                img = fh_imread_by_binary_c_decode(image)
            else:
                raise AILabException(ERROR_IMAGE_DECODE)
            if img is not None:
                return img
            else:
                raise AILabException(ERROR_IMAGE_DECODE)
        except AILabException as e:
            logging.error("noid: {}, file: {}, preprocess error: {}".format(noid, id, e.msg))
            return e.errorDict

    def cpu_decode_only(self, noid, image):
        try:
            # cpu decode and input is base64
            if isinstance(image, str):
                img = fh_imread_by_base64_python_decode(image, self.img_type)
            # cpu decode and input is bytes
            elif isinstance(image, bytes):
                img = fh_imread_by_binary_python_decode(image, self.img_type)
            else:
                raise AILabException(ERROR_IMAGE_DECODE)
            if img is not None:
                return img
            else:
                raise AILabException(ERROR_IMAGE_DECODE)
        except AILabException as e:
            logging.error("noid: {}, file: {}, preprocess error: {}".format(noid, id, e.msg))
            return e.errorDict

    def images_decode_cpu(self, noid, images):
        fail = {}
        success = {}
        for id, image in enumerate(images):
            img = image['image']
            ret = self.cpu_decode_only(noid, img)
            if isinstance(ret, dict):
                fail[id] = ret
            else:
                success[id] = ret
        return success, fail

    def images_decode(self, noid, images):
        decode_start = time.time()
        success = {}
        fail = {}
        for id, image in enumerate(images):
            img = image['image']
            ret = self.image_decode(noid, img)
            if isinstance(ret, dict):
                fail[id] = ret
            else:
                success[id] = ret

        if self.decode_type == 1 and len(success.keys()) > 0:
            success, fail = self.gpu_decode(noid, images, success, self.images_decode_cpu)

        decode_end = time.time()
        logging.debug("images_decode, noid: {}, cost: {}s".format(noid, (decode_end - decode_start)))

        return success, fail


class ImageDecoderMultiProcess(BaseDecoder):
    def __init__(self, params):
        super().__init__(params)
        from 其他.batch_decode_img import BatchDecode
        self.batch_decoder = BatchDecode(params)
        self.img_type = int(params['preprocess']['cpu_decode_output_img_type'])

    def images_decode(self, noid, images):
        gpu_decode_fail = {}

        decode_start = time.time()
        success, fail = self.batch_decoder.process(noid, images)
        if self.decode_type == 1 and len(success.keys()) > 0:
            success, gpu_decode_fail = self.gpu_decode(noid, images, success, self.batch_decoder.process_cpudec)

        fail.update(gpu_decode_fail)

        decode_end = time.time()
        logging.debug("images_decode, noid: {}, cost: {}s".format(noid, (decode_end - decode_start)))
        return success, fail
