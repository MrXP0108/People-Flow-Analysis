from .mapper import Mapper
from torchvision import transforms
from ultralytics import YOLO
import torch
import cv2
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Detection:

    def __init__(self, id, bb_left = 0, bb_top = 0, bb_w = 0, bb_h = 0, conf = 0, det_class = 0):
        self.id = id
        self.bb_left = bb_left
        self.bb_top = bb_top
        self.bb_w = bb_w
        self.bb_h = bb_h
        self.conf = conf
        self.det_class = det_class
        self.track_id = 0
        self.y = np.zeros((2, 1))
        self.R = np.eye(4)


    def __str__(self):
        return 'd{}, bb_box:[{},{},{},{}], conf={:.2f}, class{}, uv:[{:.0f},{:.0f}], mapped to:[{:.1f},{:.1f}]'.format(
            self.id, self.bb_left, self.bb_top, self.bb_w, self.bb_h, self.conf, self.det_class,
            self.bb_left+self.bb_w/2,self.bb_top+self.bb_h,self.y[0,0],self.y[1,0])

    def __repr__(self):
        return self.__str__()

    def get_coord(self):
        return [int(self.bb_left), int(self.bb_top), int(self.bb_left+self.bb_w), int(self.bb_top+self.bb_h)]
    
class Detector:
    def __init__(self, yolo_version, cam_para_file):
        self.seq_length = 0
        self.gmc = None
        self.mapper = Mapper(cam_para_file)
        self.model = YOLO(f'pretrained/yolov{yolo_version}.pt').to(device)

    def get_dets(self, frame_img,conf_thresh = 0,det_classes = [0]):
        
        dets = []

        frame_img = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)
        img_tensor = transforms.ToTensor()(frame_img).unsqueeze(0).to(device)

        results = self.model(img_tensor)

        det_id = 0
        for box in results[0].boxes:
            conf = box.conf.to('cpu').numpy()[0]
            bbox = box.xyxy.to('cpu').numpy()[0]
            cls_id  = box.cls.to('cpu').numpy()[0]
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            if w <= 20 and h <= 20 or cls_id not in det_classes or conf <= conf_thresh:
                continue

            # 新建一个Detection对象
            det = Detection(det_id)
            det.bb_left = bbox[0]
            det.bb_top = bbox[1]
            det.bb_w = w
            det.bb_h = h
            det.conf = conf
            det.det_class = cls_id
            det.y,det.R = self.mapper.mapto([det.bb_left,det.bb_top,det.bb_w,det.bb_h])
            det_id += 1

            dets.append(det)

        return dets