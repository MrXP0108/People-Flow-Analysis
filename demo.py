from PIL import ImageFont, ImageDraw, Image
import cv2
import argparse
import numpy as np

from detector.detect import Detector
from tracker.ucmc import UCMCTrack
from util.mark_entrance import EntranceManager
from enhancement.lowlight import LowLightEnhancer
from enhancement.tampering import TamperHandler

# import timeit
# start_time = timeit.default_timer()
# print(f'elapse: {timeit.default_timer() - start_time}')

font = ImageFont.truetype('fonts/GenRyuMin.ttc', 20)

def show_updated_analysis(n_person, in_pos_count, out_pos_count):
    img = np.zeros((400, 500), dtype=np.uint8)
    img_pil = Image.fromarray(img)

    ImageDraw.Draw(img_pil).text((10, 10), f'畫面總人數：{n_person}', stroke_w=10, stroke_fill='black', fill=255, font=font)
    for i in range(len(in_pos_count)):
        ImageDraw.Draw(img_pil).text((10, 30*i+50), f'。由 {i} 號口進入：{in_pos_count[i]}', stroke_w=10, stroke_fill='black', fill=255, font=font)
    for i in range(len(out_pos_count)):
        ImageDraw.Draw(img_pil).text((250, 30*i+50), f'。由 {i} 號口離開：{out_pos_count[i]}', stroke_w=10, stroke_fill='black', fill=255, font=font)
    img = np.array(img_pil)
    cv2.imshow('Values', img)

def show_camera_warning():
    img = np.zeros((80, 210), dtype=np.uint8)
    img_pil = Image.fromarray(img)

    ImageDraw.Draw(img_pil).text((30, 30), '攝影機受到干擾！', stroke_w=10, stroke_fill='black', fill=255, font=font)
    img = np.array(img_pil)
    cv2.imshow('!!!', img)

def main(args):

    # video reader
    cap = cv2.VideoCapture(f'{args.source_folder}/{args.video}')
    fps = cap.get(cv2.CAP_PROP_FPS)

    # image enhancement
    enhancer = LowLightEnhancer()
    handler = TamperHandler()

    # person detection
    detector = Detector(args.yolo_version, f'{args.source_folder}/{args.cam_para}')
    
    # person tracking
    track_manager = UCMCTrack(args.a, args.a, args.wx, args.wy, args.vmax, args.cdt, fps, args.high_score)

    # entrance managing
    entrance_manager = EntranceManager(f'{args.source_folder}/{args.entrance_coords}')
    entrance_coords = entrance_manager.coords
    n_person = 0

    frame_id = 1
    while True:
        ret, frame_img = cap.read()
        if not ret: break
        if handler.fixed_tamper_flag:
            show_camera_warning()
            # if args.tamp_test: handler.show_result()
        h, w = frame_img.shape[:2]
        
        frame_img, blurred = enhancer.enhance(frame_img, \
            cv2.resize(handler.fgmask, (w, h)),
            cv2.resize(handler.e_bg, (w, h)))

        handler.detect(blurred, frame_id, args.tamp_thresh, visualized=args.tamp_test)

        if args.tamp_test:
            if cv2.waitKey(10) & 0xFF == ord('q'): break
            frame_id += 1
            continue

        if frame_id == 1:
            if h >= 700 or w >= 700: entrance_manager.factor = 2.5
            if h <= 300: entrance_manager.factor = 0.5
            for coord in reversed(entrance_coords):
                coord[0] = int(coord[0] / entrance_manager.factor)
                coord[1] = int(coord[1] / entrance_manager.factor)
                coord[2] = int(coord[2] / entrance_manager.factor)
                coord[3] = int(coord[3] / entrance_manager.factor)
        frame_img = cv2.resize(frame_img, \
            (int(w / entrance_manager.factor) // 32 * 32,
             int(h / entrance_manager.factor) // 32 * 32))

        dets = detector.get_dets(frame_img,args.conf_thresh)
        invalid_in_pos, all_out_pos = track_manager.update(dets,frame_id)
        for in_pos in invalid_in_pos:
            entrance_manager.in_pos_count[in_pos] -= 1
        for out_pos in all_out_pos:
            entrance_manager.out_pos_count[out_pos] += 1

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
                entrance = tracker.in_pos if tracker.in_pos >= 0 else 'detecting'
                exit = tracker.out_pos if tracker.out_pos >= 0 else 'detecting'
                cv2.putText(frame_img, f'id: {det.track_id}, from: {entrance}, to: {exit}', \
                    (int(det.bb_left), int(det.bb_top)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                if tracker.update_pos(det.get_coord(), entrance_coords):
                    entrance_manager.in_pos_count[tracker.in_pos] += 1

        show_updated_analysis(n_person, entrance_manager.in_pos_count, entrance_manager.out_pos_count)        
        cv2.imshow('People Flow Analysis', frame_img)
        if cv2.waitKey(50) & 0xFF == ord('q'): break

        frame_id += 1
        n_person = 0
    
    cap.release()
    cv2.destroyAllWindows()
    handler.show_result(force_show=True)

parser = argparse.ArgumentParser(description='Process some arguments.')
parser.add_argument('-y', '--yolo_version', type=str, default = "9e", help='used YOLO version')
parser.add_argument('-s', '--source_folder', type=str, default = "demo/demo", help='folder for video, cam_para and entrance_coords')
parser.add_argument('-v', '--video', type=str, default = "demo.mp4", help='video file name')
parser.add_argument('-c', '--cam_para', type=str, default = "cam_para_test.txt", help='camera parameter file name')
parser.add_argument('-e', '--entrance_coords', type=str, default = "entrance_coords.txt", help='coordinates of all entrances')
parser.add_argument('--wx', type=float, default=5, help='wx')
parser.add_argument('--wy', type=float, default=5, help='wy')
parser.add_argument('--vmax', type=float, default=10, help='vmax')
parser.add_argument('--a', type=float, default=100.0, help='assignment threshold')
parser.add_argument('--cdt', type=int, default=5, help='coasted deletion time')
parser.add_argument('--high_score', type=float, default=0.3, help='high score threshold')
parser.add_argument('--conf_thresh', type=float, default=0.01, help='detection confidence threshold')
parser.add_argument('--tamp_thresh', type=int, default=50, help='tampering confidence threshold')
parser.add_argument('--show_entrances', action='store_true', help='show the bounding boxes of entrances or not')
parser.add_argument('--tamp_test', action='store_true', help='debug mode for tamper handling')
args = parser.parse_args()

if __name__ == '__main__':
    main(args)
