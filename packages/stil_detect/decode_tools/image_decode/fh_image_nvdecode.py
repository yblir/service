import os
import numpy as np
import pycuda.driver as cuda
from ctypes import *

# FAIMAGE_MAX_CHANNEL = 4
#
#
# class FAImage(Structure):
#     """
#     * @brief 解码后的图像数据
#     """
#     _fields_ = [
#         ("channel", POINTER(c_ubyte) * FAIMAGE_MAX_CHANNEL),  # 存储解码数据
#         ("size", c_size_t * FAIMAGE_MAX_CHANNEL)  # 每个通道的数据长度
#     ]
#
#
# class FAImageInfo(Structure):
#     """
#      * @brief 图像信息（宽,高,通道数，编码格式）
#     """
#     _fields_ = [
#         ("width", c_int),
#         ("height", c_int),
#         ("channel", c_int),
#         ("codec_id", c_int)
#     ]
#
#
# class FAImageResult(Structure):
#     """
#     * @brief 解码后的返回结果
#     """
#     _fields_ = [
#         ("info", FAImageInfo),  # 图像信息
#         ("image", FAImage)  # 解码后的图像数据
#     ]
#
#
# """
# @brief 解码后的图像像素格式
# """
# FAImageOutputFormat = (
#     FA_IMAGE_OUTPUT_FMT_RGB,  # RGBRGBRGB...
#     FA_IMAGE_OUTPUT_FMT_BGR,  # BGRBGRBGR...
#     FA_IMAGE_OUTPUT_FMT_GRAY,  # 灰度图
#     FA_IMAGE_OUTPUT_FMT_YCbCr,  # 该格式暂未支持
#     FA_IMAGE_OUTPUT_FMT_ANY_DATA  # 保持原有的通道数
# ) = (0, 1, 2, 3, 4)
#
# FAImageCodecID = (
#     FA_IMG_CODEC_ID_BMP,
#     FA_IMG_CODEC_ID_JPEG,
#     FA_IMG_CODEC_ID_PNG,
#     FA_IMG_CODEC_ID_PNM,
#     FA_IMG_CODEC_ID_TIFF,
#     FA_IMG_CODEC_ID_WEBP,
#     FA_IMG_CODEC_ID_JPEG2K,
#     FA_IMG_CODEC_ID_UNKNOWN
# ) = (0, 1, 2, 3, 4, 5, 6, -1)
#
# """
# @brief 解码错误返回状态码
# """
# FACodecError = (
#     FA_CODEC_ERROR_NOT_INITIALIZED,  # 未初始化
#     FA_CODEC_ERROR_INPUT_INVALID,  # 输入数据为空，或者输入数据数量超过预设值, 或者输入数据有问题
#     FA_CODEC_ERROR_BATCH_MEM_OVER_SIZE,  # 当前batch解码后占用的显存超过预设最大值
#     FA_CODEC_ERROR_UNKNOWN,  # 发生未知错误
# ) = (-1, -2, -3, -4)


class FAImageDecoder(object):
    def __init__(self,
                 batch_size=1,
                 device_id=0,
                 nthread=1,
                 output_type=FA_IMAGE_OUTPUT_FMT_BGR,
                 max_size=-1,
                 log_verbose=False):
        """
        图片解码类, 支持bmp，pbm, pgm, ppm, png, jpg, jpg2000, tiff, webp(lossy)图像格式解码
        :param batch_size: 批处理的图像数量
        :param device_id: 处理设备
        :param nthread: 处理的线程数
        :param output_type: 解码输出的像素排列格式. 当前仅支持BGR格式。后续会加入其他格式
        :param max_size: 限定单个batch解码后占用的最大显存/内存，如 1920*1080*3*batch_size. 如果max_size < 0,则不设上限,按需分配
        :param log_verbose: 当前参数目前暂不起作用
        :raise 如果输入的参数不合法，则抛出一个ValueError异常
        """
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        lib_file = cur_dir + "/libfanvcodec_c.so"
        self.so = cdll.LoadLibrary(lib_file)
        self.handle = c_void_p(None)

        self.batch_size = batch_size
        self.device_id = device_id
        self.nthread = nthread
        self.output_type = output_type
        self.max_size = max_size
        self.log_verbose = log_verbose

        self.cuda_ptrs = [None] * self.batch_size
        self.widths = (c_int * self.batch_size)()
        self.heights = (c_int * self.batch_size)()
        self.channels = (c_int * self.batch_size)()

        if self.batch_size < 1:
            raise ValueError("batch_size should be >= 1")
        if self.device_id < 0:
            raise ValueError("device id should be >= 0")
        if self.nthread <= 0:
            raise ValueError("thread nums should be > 0")

        if self._init_handle() != 0:
            raise RuntimeError("init handle failed.")

    def __del__(self):
        self._release_handle()

    def _init_handle(self) -> int:
        return self.so.init_handle(pointer(self.handle), self.batch_size, self.device_id, self.nthread,
                                   self.output_type, self.max_size)

    def _release_handle(self):
        self.so.release_handle(self.handle)

    def _check_input_size(self, images: list):
        if len(images) > self.batch_size:
            print("input image size is bigger than batch size")
            return FA_CODEC_ERROR_INPUT_INVALID
        return 0

    @staticmethod
    def def_c_input_file_params(self, images):
        nums = len(images)
        bimages = [c_char_p(bytes(img, "utf-8")) for img in images]
        pimages = (c_char_p * nums)(*bimages)
        results = (FAImageResult * nums)()
        c_nums = c_size_t(nums)

        return 0, pimages, c_nums, results

    @staticmethod
    def def_c_input_buffer_params(images):
        nums = len(images)
        sizes = [0] * nums
        image_buffs = (c_char_p * nums)()
        # 允许两种输入类型：numpy.ndarray和bytes，分别对应numpy.fromfile和file.open.read方法
        for i in range(nums):
            if isinstance(images[i], np.ndarray):
                image_buffs[i] = images[i].ctypes.data_as(c_char_p)
            elif isinstance(images[i], bytes):
                image_buffs[i] = cast(images[i], c_char_p)
            else:
                print("input images list should be bytes list or numpy.ndarray list")
                return FA_CODEC_ERROR_INPUT_INVALID, None, None, None, None

            sizes[i] = len(images[i])

        buff_sizes = (c_size_t * nums)(*sizes)
        results = (FAImageResult * nums)()
        c_nums = c_size_t(nums)

        return 0, image_buffs, buff_sizes, c_nums, results

    def _set_results(self, nums, results):
        for i in range(nums):
            self.widths[i] = results[i].info.width
            self.heights[i] = results[i].info.height
            self.channels[i] = results[i].info.channel
            if results[i].image.size[0] != 0:
                self.cuda_ptrs[i] = addressof(results[i].image.channel[0].contents)
            else:
                self.cuda_ptrs[i] = 0

    def batch_decode_file(self, images: list):
        """
        批处理解码文件图片
        :param images: 图片文件列表. list中的元素为图片的文件路径
        :return: 返回形式：(解码返回状态，指向解码结果的GPU指针，图像高度，图像宽度，图像通道数)
                如果解码期间出现出现异常，返回状态 < 0；否则返回状态是本次解码成功的图片数量。
                对于解码成功的图片，GPU指针指向解码数据存放的地址，宽、高、通道数均为有效值
                对于解码失败的图片，GPU指针为空指针，宽、高、通道数均为0
        """
        err = self._check_input_size(images)
        if err != 0:
            return err, self.cuda_ptrs, self.heights, self.widths, self.channels

        err, pimages, c_nums, results = self.def_c_input_file_params(images)
        c_ret = self.so.decode_batch(self.handle, pimages, c_nums, pointer(pointer(results)))
        if c_ret >= 0:
            self._set_results(c_nums.value, results)

        return c_ret, self.cuda_ptrs, self.heights, self.widths, self.channels

    def batch_decode_buffer(self, images: list):
        """
        批处理解码内存中的图片
        :param images: 图像数据列表. 列表中的元素需要是bytes类型或numpyt.ndaray类型
        :return: 返回形式：(解码返回状态，指向解码结果的GPU指针，图像高度，图像宽度，图像通道数)
                如果解码期间出现出现异常，返回状态 < 0；否则返回状态是本次解码成功的图片数量。
                对于解码成功的图片，GPU指针指向解码数据存放的地址，宽、高、通道数均为有效值
                对于解码失败的图片，GPU指针为空指针，宽、高、通道数均为0
        """
        err = self._check_input_size(images)
        if err != 0:
            return err, self.cuda_ptrs, self.heights, self.widths, self.channels

        err, image_buffs, buff_sizes, c_nums, results = self.def_c_input_buffer_params(images)
        if err != 0:
            return err, self.cuda_ptrs, self.heights, self.widths, self.channels
        c_ret = self.so.decode_batch_buffer(self.handle, pointer(image_buffs), pointer(buff_sizes),
                                            c_nums, pointer(pointer(results)))
        if c_ret >= 0:
            self._set_results(c_nums.value, results)

        return c_ret, self.cuda_ptrs, self.heights, self.widths, self.channels

    def batch_decode_file_to_host(self, images: list):
        """
        解码图片文件，jpg格式使用GPU加速，其余格式使用CPU解码，解码后的数据存放于内存中
        :param images:批量图片文件的路径，文件名需要是str或bytes格式
        :return: 返回形式：(解码返回状态，指向解码结果的GPU指针，图像高度，图像宽度，图像通道数)
                如果解码期间出现出现异常，返回状态 < 0；否则返回状态是本次解码成功的图片数量。
                对于解码成功的图片，指针指向解码数据存放的地址，宽、高、通道数均为有效值
                对于解码失败的图片，指针为空指针，宽、高、通道数均为0
        """
        err = self._check_input_size(images)
        if err != 0:
            return err, self.cuda_ptrs, self.heights, self.widths, self.channels

        err, pimages, c_nums, results = self.def_c_input_file_params(images)
        c_ret = self.so.decode_batch_to_host(self.handle, pimages, c_nums, pointer(pointer(results)))
        if c_ret >= 0:
            self._set_results(c_nums.value, results)

        return c_ret, self.cuda_ptrs, self.heights, self.widths, self.channels

    def batch_decode_buffer_to_host(self, images: list):
        """
        解码图片数据，jpg格式使用GPU加速，其余格式使用CPU解码，解码后的数据存放于内存中
        :param images:批量图片文件的路径，文件名需要是numpy.ndaray或bytes格式
        :return: 返回形式：(解码返回状态，指向解码结果的GPU指针，图像高度，图像宽度，图像通道数)
                如果解码期间出现出现异常，返回状态 < 0；否则返回状态是本次解码成功的图片数量。
                对于解码成功的图片，指针指向解码数据存放的地址，宽、高、通道数均为有效值
                对于解码失败的图片，指针为空指针，宽、高、通道数均为0
        """
        err = self._check_input_size(images)
        if err != 0:
            return err, self.cuda_ptrs, self.heights, self.widths, self.channels

        err, image_buffs, buff_sizes, c_nums, results = self.def_c_input_buffer_params(images)
        if err != 0:
            return err, self.cuda_ptrs, self.heights, self.widths, self.channels
        c_ret = self.so.decode_batch_buffer_to_host(self.handle, pointer(image_buffs), pointer(buff_sizes),
                                                    c_nums, pointer(pointer(results)))
        if c_ret >= 0:
            self._set_results(c_nums.value, results)

        return c_ret, self.cuda_ptrs, self.heights, self.widths, self.channels

    def batch_decode_file_by_cpu(self, images: list):
        """
        解码图片文件，所有格式均使用CPU解码，解码后的数据存放于内存中
        :param images:批量图片文件的路径，文件名需要是str或bytes格式
        :return: 返回形式：(解码返回状态，指向解码结果的GPU指针，图像高度，图像宽度，图像通道数)
                如果解码期间出现出现异常，返回状态 < 0；否则返回状态是本次解码成功的图片数量。
                对于解码成功的图片，指针指向解码数据存放的地址，宽、高、通道数均为有效值
                对于解码失败的图片，指针为空指针，宽、高、通道数均为0
        """
        err = self._check_input_size(images)
        if err != 0:
            return err, self.cuda_ptrs, self.heights, self.widths, self.channels

        err, pimages, c_nums, results = self.def_c_input_file_params(images)
        c_ret = self.so.decode_batch_by_cpu(self.handle, pimages, c_nums, pointer(pointer(results)))
        if c_ret >= 0:
            self._set_results(c_nums.value, results)

        return c_ret, self.cuda_ptrs, self.heights, self.widths, self.channels

    def batch_decode_buffer_by_cpu(self, images: list):
        """
        解码图片数据，所有格式均使用CPU解码，解码后的数据存放于内存中
        :param images:批量图片文件的路径，文件名需要是str或bytes格式
        :return: 返回形式：(解码返回状态，指向解码结果的GPU指针，图像高度，图像宽度，图像通道数)
                如果解码期间出现出现异常，返回状态 < 0；否则返回状态是本次解码成功的图片数量。
                对于解码成功的图片，指针指向解码数据存放的地址，宽、高、通道数均为有效值
                对于解码失败的图片，指针为空指针，宽、高、通道数均为0
        """
        err = self._check_input_size(images)
        if err != 0:
            return err, self.cuda_ptrs, self.heights, self.widths, self.channels

        err, image_buffs, buff_sizes, c_nums, results = self.def_c_input_buffer_params(images)
        if err != 0:
            return err, self.cuda_ptrs, self.heights, self.widths, self.channels
        c_ret = self.so.decode_batch_buffer_by_cpu(self.handle, pointer(image_buffs), pointer(buff_sizes),
                                                   c_nums, pointer(pointer(results)))
        if c_ret >= 0:
            self._set_results(c_nums.value, results)

        return c_ret, self.cuda_ptrs, self.heights, self.widths, self.channels

    def peek_info_file(self, image: 'str|bytes'):
        """
        解析图片文件头信息
        :param image:图片文件路径
        :return:(宽度，高度，通道数，编码格式)，编码格式id对应的图片格式见 FAImageCodecID
                如果解析失败，则返回(0, 0, 0, FA_IMG_CODEC_ID_UNKNOWN)

        """
        if isinstance(image, str):
            pimage = c_char_p(bytes(image, "utf-8"))
        elif isinstance(images, bytes):
            pimage = cast(image, c_char_p)
        else:
            print("input images list should be bytes or str")
            return 0, 0, 0, FA_IMG_CODEC_ID_UNKNOWN

        peek_info = self.so.peek_info
        peek_info.restype = FAImageInfo
        info = peek_info(self.handle, pimage)
        return info.width, info.height, info.channel, info.codec_id

    def peek_info_buffer(self, image: 'numpy.ndarry|bytes'):
        """
        解析图片文件头信息
        :param image:图片数据
        :return:(宽度，高度，通道数，编码格式)，编码格式id对应的图片格式见 FAImageCodecID
                如果解析失败，则返回(0, 0, 0, FA_IMG_CODEC_ID_UNKNOWN)
        """
        if isinstance(image, np.ndarray):
            image_buff = image.ctypes.data_as(c_char_p)
        elif isinstance(image, bytes):
            image_buff = cast(image, c_char_p)
        else:
            print("input images list should be bytes or numpy.ndarray")
            return 0, 0, 0, FA_IMG_CODEC_ID_UNKNOWN

        size = c_size_t(len(image))
        peek_info = self.so.peek_info_buffer
        peek_info.restype = FAImageInfo
        info = peek_info(self.handle, image_buff, size)
        return info.width, info.height, info.channel, info.codec_id

    def peek_info_batch(self, images: list):
        """
        读取图片文件头信息
        :param images:批量图片文件的路径，文件名需要是str或bytes格式
        :return: 返回形式：(解码返回状态，图片信息列表)
                如果解码期间出现出现异常，返回状态 < 0；否则返回状态是本次解码成功的图片数量。
                对于读取成功的图片，返回信息中的宽、高、通道数和编码格式均为有效值
                对于读取失败的图片，返回信息中的宽、高、通道数均为0，编码格式为unknown
        """
        err = self._check_input_size(images)
        if err != 0:
            return err, self.cuda_ptrs, self.heights, self.widths, self.channels

        nums = len(images)
        bimages = [c_char_p(bytes(img, "utf-8")) for img in images]
        pimages = (c_char_p * nums)(*bimages)
        results = (FAImageInfo * nums)()
        c_nums = c_size_t(nums)

        c_ret = self.so.peek_info_batch(self.handle, pimages, c_nums, pointer(pointer(results)))
        info_list = []
        for i in range(c_nums.value):
            info = results[i].width, results[i].height, results[i].channel, results[i].codec_id
            info_list.append(info)

        return c_ret, info_list

    def peek_info_batch_buffer(self, images: list):
        """
        读取图片文件头信息
        :param images:批量图片数据的路径，数据格式应为numpy.ndarray或bytes
        :return: 返回形式：(解码返回状态，图片信息列表)
                如果解码期间出现出现异常，返回状态 < 0；否则返回状态是本次解码成功的图片数量。
                对于读取成功的图片，返回信息中的宽、高、通道数和编码格式均为有效值
                对于读取失败的图片，返回信息中的宽、高、通道数均为0，编码格式为unknown
        """
        err = self._check_input_size(images)
        if err != 0:
            return err, self.cuda_ptrs, self.heights, self.widths, self.channels

        nums = len(images)
        sizes = [0] * nums
        image_buffs = (c_char_p * nums)()
        for i in range(nums):
            if isinstance(images[i], np.ndarray):
                image_buffs[i] = images[i].ctypes.data_as(c_char_p)
            elif isinstance(images[i], bytes):
                image_buffs[i] = cast(images[i], c_char_p)
            else:
                print("input images list should be bytes list or numpy.ndarray list")
                return FA_CODEC_ERROR_INPUT_INVALID, self.cuda_ptrs, self.heights, self.widths, self.channels

            sizes[i] = len(images[i])
        buff_sizes = (c_size_t * nums)(*sizes)
        results = (FAImageInfo * nums)()
        c_nums = c_size_t(nums)

        c_ret = self.so.peek_info_batch_buffer(self.handle, pointer(image_buffs), pointer(buff_sizes),
                                               c_nums, pointer(pointer(results)))

        info_list = []
        for i in range(c_nums.value):
            info = results[i].width, results[i].height, results[i].channel, results[i].codec_id
            info_list.append(info)

        return c_ret, info_list


"""
=============================================
                api example
=============================================
"""


# 遍历文件夹中的文件，返回list
def get_file_list(path):
    """
    获取当前文件夹下的所有文件列表
    :param path: 需要遍历的目录
    :return: 该目录下，所有文件格式的路径列表
    """
    file_list = []
    for dirpath, dirnames, files in os.walk(path):
        for file_name in files:
            file_path = os.path.join(dirpath, file_name)
            file_list.append(file_path)
    return file_list


# 使用pycuda读取显存指针中的数据，后用opencv保存为图片
def write_image(dptr, height, width, channel, save_path=None):
    if width == 0:
        return

    data = cuda.from_device(dptr, shape=(height, width, channel), dtype=np.uint8)
    if save_path is not None:
        import cv2
        cv2.imwrite(save_path, data)


# 图片文件解码
def decode_file_example(decoder, images: list, is_save=True):
    print("start decode file...")

    k = 0
    batch_images = []
    for img in images:
        batch_images.append(img)
        if len(batch_images) % batch_size == 0:
            ret_state, cuda_ptrs, heights, widths, channels = decoder.batch_decode_file(batch_images)

            if ret_state < 0:
                print("decode last batch failed.")
                batch_images.clear()
                continue
            elif ret_state != len(batch_images):
                print(len(batch_images) - ret_state, "images decode failed")

            if is_save:
                for j in range(len(batch_images)):
                    write_image(cuda_ptrs[j], heights[j], widths[j], channels[j], str(k) + "_file.bmp")
                    k += 1

            ret_state, info_list = decoder.peek_info_batch(batch_images)
            for info in info_list:
                print("w:", info[0], "h:", info[1], "c:", info[2], "codec id:", info[3])

            batch_images.clear()

    print("last batch size:", len(batch_images))
    ret_state, cuda_ptrs, heights, widths, channels = decoder.batch_decode_file(batch_images)
    if ret_state < 0:
        print("decode last batch failed.")
    elif ret_state != len(batch_images):
        print(len(batch_images) - ret_state, "images decode failed")

    if is_save:
        for j in range(len(batch_images)):
            write_image(cuda_ptrs[j], heights[j], widths[j], channels[j], str(k) + "_file.bmp")
            k += 1

    print("decode image complete")


# 图片数据解码
def decode_buffer_example(decoder, images: list, is_save=True):
    print("start decode buffer...")
    image_buffers = []
    for img in images:
        # 1. 使用file reader读进内存
        # with open(img, "rb") as f:
        #     data = f.read()
        #     image_buffers.append(data)

        # 2. 使用numpy.fromfile读进内存
        data = np.fromfile(img, dtype=np.uint8)
        image_buffers.append(data)

    k = 0
    batch_images = []
    for img in image_buffers:
        batch_images.append(img)
        if len(batch_images) % batch_size == 0:
            ret_state, cuda_ptrs, heights, widths, channels = decoder.batch_decode_buffer(batch_images)

            if ret_state < 0:
                print("decode last batch failed.")
                batch_images.clear()
                continue
            elif ret_state != len(batch_images):
                print(len(batch_images) - ret_state, "images decode failed")

            if is_save:
                for j in range(len(batch_images)):
                    write_image(cuda_ptrs[j], heights[j], widths[j], channels[j], str(k) + "_buffer.bmp")
                    k += 1

            ret_state, info_list = decoder.peek_info_batch_buffer(batch_images)
            for info in info_list:
                print("w:", info[0], "h:", info[1], "c:", info[2], "codec id:", info[3])

            batch_images.clear()

    print("last batch size:", len(batch_images))
    ret_state, cuda_ptrs, heights, widths, channels = decoder.batch_decode_buffer(batch_images)
    if ret_state < 0:
        print("decode last batch failed.")
    elif ret_state != len(batch_images):
        print(len(batch_images) - ret_state, "images decode failed")

    if is_save:
        for j in range(len(batch_images)):
            write_image(cuda_ptrs[j], heights[j], widths[j], channels[j], str(k) + "_buffer.bmp")
            k += 1

    print("decode image complete")


if __name__ == "__main__":
    img_dir = "../exp/images/"
    images = get_file_list(img_dir)
    print("there are", len(images), "files in '" + img_dir + "'")

    batch_size, device_id, nthread = 4, 4, 4
    max_size = -1
    decoder = FAImageDecoder(batch_size=batch_size, device_id=device_id, nthread=nthread, max_size=max_size)
    decode_file_example(decoder=decoder, images=images, is_save=True)
    decode_buffer_example(decoder=decoder, images=images, is_save=True)
