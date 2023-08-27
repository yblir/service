# -*- coding: utf-8 -*-
# @Time     : 2022/7/13 9:12
# @Author   : LiqunWang
# @FileName : base_decoder.py
import time
import logging
from .exceptions import AILabException, ERROR_IMAGE_DECODE

try:
    from fhaiservice.fh_image_nvdecode import FAImageDecoder, FA_IMAGE_OUTPUT_FMT_RGB, FA_IMAGE_OUTPUT_FMT_BGR
    import nvidia.dali.types as types
except Exception as e:
    logging.error(e)


class BaseDecoder(object):
    def __init__(self, params):
        self.params = params
        self.decode_type = int(params['preprocess']['decode_type'])
        if self.decode_type == 1:
            logging.info("use gpu decode")
            gpu_decode_batch_size = int(params["base"]["batchsize"])
            gpu_decode_thread_num = int(params['preprocess']['gpu_decode_thread_num'])
            gpu_id = int(params['base']['device_id'])
            gpu_ouput_img_type = int(params['preprocess']['gpu_decode_ouput_img_type'])
            if gpu_ouput_img_type == 0:
                ouput_img_type = FA_IMAGE_OUTPUT_FMT_RGB
            elif gpu_ouput_img_type == 1:
                ouput_img_type = FA_IMAGE_OUTPUT_FMT_BGR
            try:
                max_size = eval(params['preprocess']['max_size'])
            except Exception as e:
                max_size = 1920 * 1080 * 3
                logging.error(e)
            max_size = max_size * gpu_decode_batch_size
            self.nvdecoder = FAImageDecoder(batch_size=gpu_decode_batch_size,
                                            device_id=gpu_id,
                                            nthread=gpu_decode_thread_num,
                                            output_type=ouput_img_type,
                                            max_size=max_size
                                            )

    def images_decode(self, noid, images):
        raise NotImplementedError

    def gpu_decode(self, noid, src_image, unbox_image_dict, cpu_decode_process):
        success = {}
        fail = {}
        key_list = []
        images_nodecodes = []
        for index, i in enumerate(unbox_image_dict.keys()):
            key_list.append(i)
            images_nodecodes.append(unbox_image_dict[i])
        nvdecode_time_start = time.time()
        ret_size, ret_ptrs, ret_heights, ret_widths, ret_channels = \
            self.nvdecoder.batch_decode_buffer_to_host(images_nodecodes)

        nvdecode_time_end = time.time()
        logging.debug("nvdecode time:{}".format(nvdecode_time_end - nvdecode_time_start))
        succ_num = len(unbox_image_dict)
        if ret_size != succ_num:
            logging.warning('nvdecoder has error, ret_size={}'.format(ret_size))
            if ret_size == 0:
                for i in range(succ_num):
                    fail[key_list[i]] = AILabException(ERROR_IMAGE_DECODE).errorDict
            elif ret_size < 0:
                logging.warning('nvdecoder error, using cpu decode, ret_size={}'.format(ret_size))
                success, fail = cpu_decode_process(noid, src_image)
            else:
                for i in range(succ_num):
                    if ret_heights[i] == 0:
                        fail[key_list[i]] = AILabException(ERROR_IMAGE_DECODE).errorDict
                    else:
                        success[key_list[i]] = ret_size, ret_ptrs[i], ret_heights[i], ret_widths[i], ret_channels[i]
        else:
            for i in range(ret_size):
                success[key_list[i]] = ret_size, ret_ptrs[i], ret_heights[i], ret_widths[i], ret_channels[i]
        return success, fail
