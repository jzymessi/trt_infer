from utils.utils import BaseEngine
import numpy as np
import cv2

class FPN(BaseEngine):
    
    def __init__(self, engine_path):
        super().__init__(engine_path)
        self.class_num = 2
        self.weight = 64
        self.height = 64

    def preprocess_input(self, img_path):
        
        ## numpy实现
        img = cv2.imread(img_path)
        resized_img = cv2.resize(img, (self.height, self.weight)) 

        input_ = resized_img.transpose(2, 0, 1)  
        input_ = np.flip(input_, 0)

        mean = np.array([123.675, 116.28, 103.53]).reshape(3, 1, 1) 
        std = np.array([58.395, 57.12, 57.375]).reshape(3, 1, 1)
        input_ = (input_ - mean) / std
        input_ = np.ascontiguousarray(input_, dtype=np.float32)
        
        return input_
    
    def postprocess_output(self,datas):
        outputs = []
        channel = self.class_num
        size = self.height * self.weight
        for data in datas:
            result = np.empty(size)
            for i in range(size):
                index = np.argmax(data[:channel])
                result[i] = index
                data = data[channel:]
            outputs.append(result)
        return outputs
    
    def inference(self,img_path):
        input_data = self.preprocess_input(img_path)
        data = super().infer(input_data)
        data = self.postprocess_output(data)
        return data


