# coding:utf-8

"""
异常定义
"""


class AILabException(Exception):
    """
    AILab自定义异常基类，继承自Exception
    """

    def __init__(self, exception_dict):
        self.status = exception_dict['status']
        self.msg = exception_dict['msg']

    @property
    def errorCode(self):
        return self.status

    @property
    def errorType(self):
        return self.msg

    @property
    def errorDict(self):
        return {"status": self.status, "msg": self.msg}


# 异常码
class ErrorCode:
    # 未知错误
    ERROR_UNKNOWN = {"status": "-1", "msg": "ERROR_UNKNOWN"}
    # 错误参数
    ERROR_PARAMETER = {"status": "1000", "msg": "ERROR_PARAMETER"}
    # 错误图片类型
    ERROR_IMAGE_TYPE = {"status": "1001", "msg": "ERROR_IMAGE_TYPE"}
    # 图片尺寸不对
    ERROR_IMAGE_SIZE = {"status": "1002", "msg": "ERROR_IMAGE_SIZE"}
    # 图片解码失败
    ERROR_IMAGE_DECODE = {"status": "1003", "msg": "ERROR_IMAGE_DECODE"}
    # 请求体积过大
    ERROR_REQUEST_FILE_LARGE = {"status": "1004", "msg": "ERROR_REQUEST_FILE_LARGE"}
    # 请求体格式异常
    ERROR_REQUEST_BODY_FORMAT = {"status": "1005", "msg": "ERROR_REQUEST_BODY_FORMAT"}
    # 图片base64解码失败
    ERROR_IMAGE_BASE64_DECODE = {"status": "1006", "msg": "ERROR_IMAGE_BASE64_DECODE"}
    # 返回结果转json失败
    ERROR_RETURN_TO_JSON_ERROR = {"status": "1007", "msg": "ERROR_RETURN_TO_JSON_ERROR"}
    # 日志分析打印错误
    ERROR_RESULT_ANALYZE = {"status": "1008", "msg": "ERROR_RESULT_ANALYZE"}
    # 二进制转换adarray失败
    ERROR_RETURN_TO_BINARY_ERROR = {"status": "1009", "msg": "ERROR_RETURN_TO_BINARY_ERROR"}
    # 调用SDK时，引擎初始化失败
    ERROR_ENGINE_INIT_FAILED = {"status": "1010", "msg": "ERROR_ENGINE_INIT_FAILED"}
    # 调用SDK时，推理失败
    ERROR_INFER_ERROR = {"status": "1011", "msg": "ERROR_INFER_ERROR"}
    # 前处理失败
    ERROR_PREPROCESS = {"status": "1012", "msg": "ERROR_PREPROCESS"}
    # 信息采集器初始化失败
    ERROR_COLLECTOR_INIT_FAILD = {"status": "1050", "msg": "ERROR_COLLECTOR_INIT_FAILD"}
