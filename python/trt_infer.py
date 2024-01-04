from resnet.resnet import Resnet
from segmentation.fpn import FPN
from yolox.yolox import Yolox,vis
import cv2
def test_resnet():
    engine = '/workspace/tensorrt/project/trt_benchmark/jd_resnet50_fp16_1.trt'
    pred = Resnet(engine_path=engine)
    image = "/workspace/tensorrt/project/trt_benchmark/fpn/fpn_imgs/100000.27291.c.-1647307451_16.jpg"
    output = pred.inference(image)
    # print(output)

def test_segmentation():
    engine = '/workspace/tensorrt/project/trt_benchmark/fpn/aoi_fpn_batch1.trt'
    pred = FPN(engine_path=engine)
    image = "/workspace/tensorrt/project/trt_benchmark/fpn/fpn_imgs/100000.27291.c.-1647307451_16.jpg"
    output = pred.inference(image)
    # print(output)

def test_yolox():
    class_names = [ 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
         'hair drier', 'toothbrush' ]
    
    engine = '/workspace/tensorrt/TensorRT-For-YOLO-Series/yolox_s.trt'
    pred = Yolox(engine_path=engine)
    image = "/workspace/tensorrt/trt_infer/data/dog.jpg"
    img = cv2.imread(image)
    end2end = True
    dets = pred.inference(image,end2end)
    print(dets)
    if dets is not None:
            final_boxes, final_scores, final_cls_inds = dets[:,:4], dets[:, 4], dets[:, 5]
            img = vis(img, final_boxes, final_scores, final_cls_inds,conf=0.5, class_names = class_names)
            cv2.imwrite("result.jpg",img)
    
if __name__ == '__main__':  
    test_yolox()