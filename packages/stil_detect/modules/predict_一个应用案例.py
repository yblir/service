# -*- coding: utf-8 -*-
# @Time    : 2023/4/13 9:03
# @Author  : yblir
# @File    : predict.py
# explain  :
# ================================================================================
import json
import uuid
import time
import torch

from typing import Dict
import albumentations as alb
from albumentations.pytorch.transforms import ToTensorV2

from base_interface.base_predict import Predictor
from utils.exceptions import AILabException, ErrorCode
from transfer import logger, config

from scrfd_opencv_gpu.scrfd_face_detect import SCRFD, get_max_face_box

# 异常处理
error_code = ErrorCode()

# 所用人脸检测模型
face_detect = SCRFD(config["scrfd_model"], confThreshold=0.5, nmsThreshold=0.5)


# todo 修改Demo类为自己模型的推理过程
class STILPredictor(Predictor):
    """
    Predictor的功能如下:
    1、处理所有与模型推理相关的事务;
    2.predict()方法:将所有模型的推理分为三步骤:前处理、推理和后处理;
    3.
    pre_process()方法:做前处理相关操作，包括，解析输入数据。对数据解码:
    4.post_process()方法:将推理后的数据转成微服务输出的格式
    5.返回推理状态信息:请求总数、目标数量、解码失败数量和推理失败数量
    """

    def __init__(self, decoder, module_infer,
                 json_schema_path="",
                 data_type_keyword=Dict,
                 json_schema_output_path=""):
        """
        初始化Predictor
        :param params:配置参数
        :param decoder:初始化完成后的解码模块类
        :param module_infer:初始化完成后的推理模块类
        :param json_schema_path:输入的校验schema文档
        :param data_type_keyword:数据关键字，，用于提取发送请求中的数据
        :param json_schema_output_path:输出的校验schema文档
        """
        super(STILPredictor, self).__init__(json_schema_path, json_schema_output_path)
        self.decode_success = None
        self.decode_fail = None
        # 标记输入时单张还是batch
        self.single_flag = True

        self.face_failure = None
        self.total_data_num = 0
        self.decode_fail = None

        self.module = module_infer
        self.decoder = decoder
        # self.data_type_keyword = data_type_keyword

        self.keyword_single = data_type_keyword["single"]
        self.keyword_batch = data_type_keyword["batch"]

        self.transform = alb.Compose([
            alb.Resize(224, 224),
            alb.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ], additional_targets={})

        self.start_time = time.time()

        logger.info("init demo Predictor")

    # predict已在基类中实现, 所有服务可公用
    # def predict(self, request_data: Dict):
    #     pass

    # 重写基类pre_process
    def pre_process(self, request):
        """
        解析请求体
        {
            "noid":xxx,
            "sysid":xxx,
            "batch":{
                0:np.ndarray,   # 键值为 int 类型
                1:np.ndarray,
                2:np.ndarray,
                ...
            }
        }
        """
        # 记录预处理开始时间, 在post_process中调用
        self.start_time = time.time()

        # 1、根据关键字，将数据从字典中读取出来
        if self.keyword_single in request:
            self.single_flag = True
            raw_data_for_decode = request[self.keyword_single]
            self.total_data_num = 1
        # batch请求中列表是待推理的单个元素
        elif self.keyword_batch in request:
            # false, 表示batch
            self.single_flag = False
            # raw_data dict, {0:np.ndarray, 1:np.ndarray, ...}
            raw_data_batch = request[self.keyword_batch]
            raw_data_len = len(raw_data_batch)
            raw_data_for_decode = [None] * raw_data_len
            # 通过字典索引找值, 避免字典哈希索引乱序
            for i in range(raw_data_len):
                raw_data_for_decode[i] = raw_data_batch[i]

            self.total_data_num = raw_data_len
        else:
            logger.error("keyword must be single or batch")
            # error_info = {"status": "1000", "msg": "ERROR_PARAMETER"}
            return None, error_code.ERROR_PARAMETER

        # 2.解码操作
        # 解码后,batch,解码结果是字典,int 类型0,1,2,为键,值为nd.array格式图片矩阵
        # decode_fail如果有,= {0:{"status":xxx,"msg":xxx, 1:{}, ...}}
        decode_success, decode_fail = self.decoder.decode(raw_data_for_decode)

        return decode_success, decode_fail

    def predict(self, request_data: Dict):
        """
        # predict 不抛出异常, 所有错误都在post中返回给调用者
        request_data:通过request直接输入的dict
        共经历三个模块:
            预处理,推理,后处理
        """

        future_infer_result = None

        # 先建立一些输出字典的默认返回信息
        ret_obj = {"status_code": "0", "msg": "SUCCESS"}
        # 在后处理中取出uuid
        self.request_data = request_data
        # 只要输入图片格式没问题,不会出现推理失败情况, 最多什么都没预测出来
        predict_fail_num = 0
        # todo 传入图片数量必须是16
        # 1.解析请求体, single, batch,解码结果都是dict,int 类型0,1,2,为键,值为nd.array格式图片矩阵,decode_fail也是dict
        # 有两种异常情况,decode_success= None, 关键字错误. ={},所有图片都解码错误
        decode_success, self.decode_fail = self.pre_process(request_data)
        # decode_success, self.decode_fail = request_data["images"], {}
        if len(decode_success) + len(self.decode_fail) != 16:
            ret_obj["cost_time"] = str(round((time.time() - self.start_time), 3)) + " s"
            ret_obj["result"] = {"status": "10086",
                                 "msg"   : f"image nums is not 16, but is {len(decode_success) + len(self.decode_fail)}"}
            return ret_obj, 0, 0, 0, 0

        # 在后处理中有应用
        self.decode_success = decode_success

        # 2.推理模块
        # 解码成功并且成功数量不少于16才进行推理. 因为少于16张不满足模型输入shape
        if decode_success:
            # [np.ndarray,np.ndarray,...] 共16个
            decode_faces = [decode_success[i] for i in range(len(decode_success))]
            decode_boxes = STILPredictor.get_faces_from_selected_frames(decode_faces)
            if decode_boxes is None:
                self.face_failure = True
            else:
                data_numpy = self.data_merge(decode_boxes)
                future_infer_result = self.module.module_infer(data_numpy)
        else:
            logger.error("cannot decode any images from request.")

        ret_obj["result"] = future_infer_result

        # 计算解码失败数量
        decode_fail_num = len(self.decode_fail)

        # 3.根据请求体，对结果进行后处理，转成微服务输出格式
        # todo 返回的结果的格式在后post_process定义
        target_num = self.post_process(ret_obj)
        # ret_obj["cost_time"] = str(round(int((time.time() - self.start_time) * 1000), 3)) + " ms"
        ret_obj["cost_time"] = str(round((time.time() - self.start_time), 3)) + " s"
        return ret_obj, self.total_data_num, target_num, decode_fail_num, predict_fail_num

    # 重写基类post
    def post_process(self, ret_obj: Dict):
        """
        将结果打包成接口文档规定的格式，将处理后的结果更新至ret_obj输出
        """
        # dict_sort_success, dict_sort_fail = {}, {}  # 用于存放一批数据中返回失败的数据信息,类型是字典
        # 获取noid
        try:
            # 发送请求获得的数据
            noid = self.request_data['noid']
        except Exception as _:
            noid = str(uuid.uuid1())

        ret_obj["noid"] = noid if noid else str(uuid.uuid1())

        # 收集解码失败信息
        for i in self.decode_fail.keys():
            ret_obj["msg"] = "ERROR"
            ret_obj['result'] = {
                "status": self.decode_fail[i]["status"],
                "msg"   : self.decode_fail[i]["msg"],
            }
            # 错误信息返回一次就够了
            return 0

        # 获取人脸信息失败
        if self.face_failure:
            ret_obj["msg"] = "ERROR"
            ret_obj['result'] = {
                "status": "1004",
                "msg"   : "have image no detect face"
            }
            return 0

        # 收集推理失败的信息? 应该不存在
        if ret_obj["result"]:
            print("111111")
            # 通过get从C++取回推理结果, 如果没有推理完成,在此阻塞,直到拿到结果
            # cur_infer_result = ret_obj["result"].get()
            judge = torch.softmax(torch.tensor([ret_obj["result"]]), dim=1)[:, 0].numpy()[0]
            judge = round(judge, 4)
            ret_obj['result'] = {"fake": round(1 - judge, 4), "msg": ""} \
                if judge <= 0.5 else \
                {"real": round(judge, 4), "msg": ""}

            return 1

    @staticmethod
    def get_faces_from_selected_frames(frames):
        """
        frames:从request解析出的16张视频帧,np.array
        """
        imgs = []
        img_h, img_w, _ = frames[0].shape

        for idx in range(len(frames)):
            try:
                img = frames[idx]
                res = face_detect.detect(img)
                if not res:
                    raise ValueError("no detect face")
                output_box, kpss = get_max_face_box(res)
                # rotated_img, _ = face_allign(kpss, img)
                # img = face_crop(rotated_img, output_box)

                x0, y0, x1, y1 = output_box
                # # logger.info(f"img.shape={img.shape}")
                x0, y0, x1, y1 = STILPredictor.get_enclosing_box(img_h, img_w, [x0, y0, x1, y1], config["margin"])
                img = img[y0:y1, x0:x1]
                imgs.append(img)
            except Exception as e:
                logger.error(f"image serial number {idx} get face failure: {str(e)}")
                return None
        return imgs

    @staticmethod
    def get_enclosing_box(img_h, img_w, box, margin):
        """Get the square-shape face bounding box after enlarging by a certain margin.

        Args:
            img_h (int): Image height.
            img_w (int): Image width.
            box (list): [x0, y0, x1, y1] format face bounding box.
            margin (float): The margin to enlarge.

        """
        x0, y0, x1, y1 = box
        w, h = x1 - x0, y1 - y0
        max_size = max(w, h)

        cx = x0 + w / 2
        cy = y0 + h / 2
        x0 = cx - max_size / 2
        y0 = cy - max_size / 2
        x1 = cx + max_size / 2
        y1 = cy + max_size / 2

        offset = max_size * (margin - 1) / 2
        x0 = int(max(x0 - offset, 0))
        y0 = int(max(y0 - offset, 0))
        x1 = int(min(x1 + offset, img_w))
        y1 = int(min(y1 + offset, img_h))

        return [x0, y0, x1, y1]

    def data_merge(self, frames):
        # 对16帧图片做增强
        additional_targets = {}
        tmp_imgs = {"image": frames[0]}
        for i in range(1, len(frames)):
            additional_targets[f"image{i}"] = "image"
            tmp_imgs[f"image{i}"] = frames[i]
        self.transform.add_targets(additional_targets)

        frames = self.transform(**tmp_imgs)

        # 排列字典frames 关键字,以从小到大形式排列,保证各帧时间连续性
        frames = [frames[i] for i in sorted(frames, reverse=False)]

        frames = torch.stack(frames)  # T, C, H, W
        process_img = frames.view(-1, frames.size(2), frames.size(3)).contiguous()  # TC, H, W

        return process_img.numpy()

    def output_format_creator(self, ret_obj: Dict):
        """
        微服务输出格式包装
        :param ret_obj: 推理输出的结果
        :return
        """
        json_str = json.dumps(ret_obj)

        return json_str
