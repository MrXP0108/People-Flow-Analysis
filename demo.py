import torch
from torchvision import transforms
from ultralytics import YOLO
from PIL import ImageFont, ImageDraw, Image
import cv2
import argparse
import matplotlib.pyplot as plt

from tracker.ucmc import UCMCTrack
from detector.mapper import Mapper
from util.mark_entrance import EntranceManager
import numpy as np

from enhancement.enhancement import Enhancer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 定义一个Detection类，包含id,bb_left,bb_top,bb_w,bb_h,conf,det_class
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

# Detector类，用于从Yolo检测器获取目标检测的结果
class Detector:
    def __init__(self):
        self.seq_length = 0
        self.gmc = None

    def load(self,cam_para_file):
        self.mapper = Mapper(cam_para_file,"MOT17")
        self.model = YOLO(f'pretrained/yolov{args.yolo_version}.pt').to(device)

    def get_dets(self, img,conf_thresh = 0,det_classes = [0]):
        
        global dets
        dets = []

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)

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

def update_value_display(n_person, in_pos_count, out_pos_count):
    value_display = np.zeros((400, 500), dtype=np.uint8)
    fontpath = 'fonts/GenRyuMin.ttc'
    font = ImageFont.truetype(fontpath, 20)
    img_pil = Image.fromarray(value_display)

    ImageDraw.Draw(img_pil).text((10, 10), f'畫面總人數：{n_person}', stroke_w=10, stroke_fill='black', fill=255, font=font)
    for i in range(len(in_pos_count)):
        ImageDraw.Draw(img_pil).text((10, 30*i+50), f'。由 {i} 號口進入：{in_pos_count[i]}', stroke_w=10, stroke_fill='black', fill=255, font=font)
    for i in range(len(out_pos_count)):
        ImageDraw.Draw(img_pil).text((250, 30*i+50), f'。由 {i} 號口離開：{out_pos_count[i]}', stroke_w=10, stroke_fill='black', fill=255, font=font)
    value_display = np.array(img_pil)
    cv2.imshow('Values', value_display)

def main(args):

    cap = cv2.VideoCapture(f'{args.source_folder}/{args.video}')

    # 获取视频的 fps
    fps = cap.get(cv2.CAP_PROP_FPS)

    detector = Detector()
    detector.load(f'{args.source_folder}/{args.cam_para}')
    dets = []
    
    track_manager = UCMCTrack(args.a, args.a, args.wx, args.wy, args.vmax, args.cdt, fps, "MOT", args.high_score,False,None)

    entrance_manager = EntranceManager(f'{args.source_folder}/{args.entrance_coords}')
    entrance_coords = entrance_manager.coords

    enhancer = Enhancer(lol_v2_real=True, best_GT_mean=True)

    n_person = 0
    in_pos_count = [0]*len(entrance_manager.coords)
    out_pos_count = [0]*len(entrance_manager.coords)

    # 循环读取视频帧
    frame_id = 1
    stop = False
    while True:
        ret, img = cap.read()
        if not ret:
            break
        h, w = img.shape[:2]

        # low-light enhancement
        # TODO: enhance if brightness is under some threshold:
        # frame_img = enhancer.enhance(img) if frame_id % 100 == 10 else img
        frame_img = enhancer.enhance(img)
        

        if frame_id == 1:
            if h >= 700 or w >= 700: entrance_manager.factor = 2.5
            if h <= 300:  entrance_manager.factor = 0.5
            for coord in reversed(entrance_coords):
                coord[0] = int(coord[0] / entrance_manager.factor)
                coord[1] = int(coord[1] / entrance_manager.factor)
                coord[2] = int(coord[2] / entrance_manager.factor)
                coord[3] = int(coord[3] / entrance_manager.factor)
        frame_img = cv2.resize(frame_img, \
            (int(w / entrance_manager.factor) // 32 * 32,
             int(h / entrance_manager.factor) // 32 * 32))
    
        if frame_id % args.detect_freq == 1:
            dets = detector.get_dets(frame_img,args.conf_thresh)
            all_out_pos = track_manager.update(dets,frame_id)
            for out_pos in all_out_pos:
                out_pos_count[out_pos] += 1

        # 標示進出口
        if args.show_entrances:
            for i in range(len(entrance_coords)):
                p1 = (entrance_coords[i][0], entrance_coords[i][1])
                p2 = (entrance_coords[i][2], entrance_coords[i][3])
                cv2.rectangle(frame_img, p1, p2, (0, 255, 255), 2)
                cv2.putText(frame_img, str(i), ((p1[0]+p2[0])//2, (p1[1]+p2[1])//2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        for det in dets:
            if det.track_id > 0:
                n_person += 1
                
                # 繪出 detection 的 bbox
                coord = det.get_coord()
                cv2.rectangle(frame_img, (coord[0], coord[1]), (coord[2], coord[3]), (0, 255, 0), 2)
                
                # 根據 bbox 判斷進出點
                tracker = track_manager.find_tracker_by_id(det.track_id)
                cv2.putText(frame_img, f'id: {det.track_id}, from: {tracker.in_pos if tracker.in_pos >= 0 else "detecting"}, to: {tracker.out_pos if tracker.out_pos >= 0 else "detecting"}', \
                    (int(det.bb_left), int(det.bb_top)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                if tracker.update_pos(det.get_coord(), entrance_coords):
                    in_pos_count[tracker.in_pos] += 1
                # 逐次降低 max_ios 以避免角度造成現在靠近的 ios 不夠大
                tracker.max_ios *= 0.95

        frame_id += 1

        # if frame_id % args.detect_freq == 1:
        #     # tracker = track_manager.find_tracker_by_id(1)
        #     # if tracker != None:
        #     #     print(f'status: {tracker.status}, associated detection: {tracker.detidx}, death count: {tracker.death_count}')
        #     for tracker in track_manager.trackers:
        #         print(f'{tracker.id}: ({tracker.status}, {tracker.death_count}')

        update_value_display(n_person, in_pos_count, out_pos_count)
        
        n_person = 0
        cv2.imshow('People Flow Analysis', frame_img)
        key = cv2.waitKey(50)
        # q: 離開
        if key == ord('q'):
            break
        # s: 暫停，再次按下可恢復播放
        elif key == ord('s'):
            stop = True
            while stop:
                key = cv2.waitKey(50)
                if key == ord('s'): stop = False
    
    cap.release()
    cv2.destroyAllWindows()



parser = argparse.ArgumentParser(description='Process some arguments.')
parser.add_argument('--yolo_version', type=str, default = "8m", help='used YOLO version')
parser.add_argument('--source_folder', type=str, default = "demo/demo5", help='folder for video, cam_para and entrance_coords')
parser.add_argument('--video', type=str, default = "demo.mp4", help='video file name')
parser.add_argument('--cam_para', type=str, default = "cam_para_test.txt", help='camera parameter file name')
parser.add_argument('--entrance_coords', type=str, default = "entrance_coords.txt", help='coordinates of all entrances')
parser.add_argument('--wx', type=float, default=5, help='wx')
parser.add_argument('--wy', type=float, default=5, help='wy')
parser.add_argument('--vmax', type=float, default=10, help='vmax')
parser.add_argument('--a', type=float, default=100.0, help='assignment threshold')
parser.add_argument('--cdt', type=int, default=5, help='coasted deletion time')
parser.add_argument('--high_score', type=float, default=0.3, help='high score threshold')
parser.add_argument('--conf_thresh', type=float, default=0.01, help='detection confidence threshold')
parser.add_argument('--detect_freq', type=int, default=10, help='frequency of YOLO detection')
parser.add_argument('--show_entrances', type=bool, default=True, help='show the bounding boxes of entrances or not')
args = parser.parse_args()

if __name__ == '__main__':
    main(args)



