import onnxruntime
import numpy as np
import cv2
import math
import fastdeploy as fd
import numpy as np

class strLabelConverter(object):
    def __init__(self, alphabet):
        self.alphabet = alphabet + ' '  # for `-1` index
        self.dict = {}
        for i, char in enumerate(alphabet):
            # NOTE: 0 is reserved for 'blank' required by wrap_ctc
            self.dict[char] = i + 1
    def decode(self, t, length, raw=False):
        t = t[:length]
        if raw:
            return ''.join([self.alphabet[i - 1] for i in t])
        else:
            char_list = []
            for i in range(length):
                if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                    char_list.append(self.alphabet[t[i] - 1])
            return ''.join(char_list)

    # def decode2(self, text_index, text_prob=None, is_remove_duplicate=False):
    #     """ convert text-index into text-label. """
    #     result_list = []
    #     ignored_tokens = [0]
    #     batch_size = len(text_index)
    #     for batch_idx in range(batch_size):
    #         selection = np.ones(len(text_index[batch_idx]), dtype=bool)
    #         if is_remove_duplicate:
    #             selection[1:] = text_index[batch_idx][1:] != text_index[batch_idx][:-1]
    #         for ignored_token in ignored_tokens:
    #             selection &= text_index[batch_idx] != ignored_token
    #         char_list = [self.dict[text_id] for text_id in text_index[batch_idx][selection]]
    #         if text_prob is not None:
    #             conf_list = text_prob[batch_idx][selection]
    #         else:
    #             conf_list = [1] * len(selection)
    #         if len(conf_list) == 0:
    #             conf_list = [0]

    #         text = ''.join(char_list)
    #         result_list.append((text, np.mean(conf_list).tolist()))
    #     return result_list

class TextRecognizer:
    def __init__(self):
        # self.sess = onnxruntime.InferenceSession('weights/ch_PP-OCRv3_rec_infer.onnx')
        self.sess = onnxruntime.InferenceSession('python/weights/ch_PP-OCRv3_rec.onnx')
        self.alphabet = list(map(lambda x:x.decode('utf-8').strip("\n").strip("\r\n"), open('python/rec_word_dict.txt', 'rb').readlines()))
        self.converter = strLabelConverter(''.join(self.alphabet))
        self.rec_image_shape = [3, 48, 320]

        
        # option = fd.RuntimeOption()

        # option.set_model_path("python/inference_model/ppocrv3_rec.pdmodel",
        #                     "python/inference_model/ppocrv3_rec.pdiparams")

        # # **** CPU 配置 ****
        # option.use_cpu()
        # option.use_ort_backend()
        # option.set_cpu_thread_num(12)

        # # **** GPU 配置 ***
        # # 如需使用GPU，使用如下注释代码
        # # option.use_gpu(0)

        # # 初始化构造runtime
        # self.runtime = fd.Runtime(option)

        # # 获取模型输入名
        # self.input_name = self.runtime.get_input_info(0).name




    def resize_norm_img(self, img):
        imgC, imgH, imgW = self.rec_image_shape
        max_wh_ratio = imgW / imgH
        
        h, w = img.shape[0:2]
        wh_ratio = w * 1.0 / h
        max_wh_ratio = max(max_wh_ratio, wh_ratio)
                
        assert imgC == img.shape[2]
        imgW = int((imgH * max_wh_ratio))

        h, w = img.shape[:2]
        ratio = w / float(h)
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))

        resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype('float32')
        resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        return padding_im

    def predict_text(self, im):
        img = self.resize_norm_img(im)
        transformed_image = np.expand_dims(img, axis=0)

        ort_inputs = {i.name: transformed_image for i in self.sess.get_inputs()}
        preds = self.sess.run(None, ort_inputs)
        preds = preds[0]
         
        # np.random.rand(1, 3, 224, 224).astype("float32")
        # 构造随机数据进行推理
        # results = self.runtime.infer({
        #     self.input_name: transformed_image for i in self.sess.get_inputs()
        # })
        
        
        # print(results, '----')
        # print(preds, preds[0].shape)
        
       
        # print(preds, preds.shape, preds[0][0][:100])
        # preds_idx = preds.argmax(axis=2)
        # preds_prob = preds.max(axis=2)
        # rec_result = self.converter.decode2(preds_idx, preds_prob, is_remove_duplicate=True)
        # print(rec_result)
        
        _preds = preds.squeeze(axis=0)
        length  = _preds.shape[0]
        _preds = _preds.reshape(length,-1)
        # preds = softmax(preds)
        _preds = np.argmax(_preds,axis=1)
        _preds = _preds.reshape(-1)
        sim_pred = self.converter.decode(_preds, length, raw=False)
        
        # print(sim_pred)
        return sim_pred