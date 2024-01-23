from utils.utils import BaseEngine
import numpy as np
import cv2

class Resnet(BaseEngine):
    
    def __init__(self, engine_path):
        super().__init__(engine_path)
        self.class_num = 3
        self.weight = 224
        self.height = 224
        
    def preprocess_input(self, img_path):
        
        img = cv2.imread(img_path)
        resized_img = cv2.resize(img, (self.weight, self.height)) 

        input_ = resized_img.transpose(2, 0, 1)  
        input_ = np.flip(input_, 0)

        mean = np.array([123.675, 116.28, 103.53]).reshape(3, 1, 1) 
        std = np.array([58.395, 57.12, 57.375]).reshape(3, 1, 1)
        input_ = (input_ - mean) / std
        input_ = np.ascontiguousarray(input_, dtype=np.float32)
        
        return input_
    
    def postprocess_output(self,data):
        probs = np.exp(data) / np.sum(np.exp(data))
        max_index = np.argmax(probs)
        return max_index
    
    def inference(self,img_path):
        input_data = self.preprocess_input(img_path)
        data = super().infer(input_data)
        max_index = self.postprocess_output(data)
        return max_index


