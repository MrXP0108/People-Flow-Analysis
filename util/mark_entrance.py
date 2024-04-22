import argparse
import cv2

class EntranceManager:

    coords = []
    new_coord = []
    factor = 1
    
    def __init__(self,file):
        self.file = file
        with open(self.file, 'r') as f:
            data = f.readlines()
        for coord in data:
            self.coords.append([int(xyxy) for xyxy in coord.split()])

    def clear(self):
        open(self.file, 'w').close()
        self.coords.clear()

    def set_coord(self,event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.new_coord.extend([int(x*self.factor), int(y*self.factor)])
        
        if len(self.new_coord) == 4:
            self.write_coord()
            draw_entrance_box(self.new_coord, self.factor)
            self.new_coord.clear()
    
    def write_coord(self):
        with open(self.file, 'a') as f:
            for xyxy in self.new_coord:
                f.write(str(xyxy)+" ")
            f.write("\n")
        self.coords.append(self.new_coord)

def draw_entrance_box(coord, factor):
    cv2.rectangle(img, (int(coord[0] / factor), int(coord[1] / factor)), \
                       (int(coord[2] / factor), int(coord[3] / factor)), (0, 255, 0), 2)

def main(args):

    entrance_manager = EntranceManager(f'{args.source_folder}/{args.entrance_coords}')
    
    global img
    cap = cv2.VideoCapture(f'{args.source_folder}/{args.video}')
    ret, img = cap.read()

    height, width = img.shape[:2]
    if height >= 700:
        entrance_manager.factor = 2.5
        img = cv2.resize(img, (int(width / entrance_manager.factor), int(height / entrance_manager.factor)))

    clean_img = img.copy()

    cv2.namedWindow('出入口標示')
    cv2.setMouseCallback('出入口標示', entrance_manager.set_coord)
    for coord in entrance_manager.coords:
        draw_entrance_box(coord, entrance_manager.factor)

    while True:
        cv2.imshow('出入口標示', img)

        key = cv2.waitKey(50)
        # q: 離開
        if key == ord('q'):
            break
        # c: 清除全部標記
        elif key == ord('c'):
            entrance_manager.clear()
            img = clean_img
            clean_img = img.copy()
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some arguments.')
    parser.add_argument('--source_folder', type=str, default = "demo/demo5", help='folder for video and entrance_coords')
    parser.add_argument('--video', type=str, default="demo.mp4",help='video file name')
    parser.add_argument('--entrance_coords', type=str, default="entrance_coords.txt",help='The coodrinates of the entrance ')
    args = parser.parse_args()
    main(args)
