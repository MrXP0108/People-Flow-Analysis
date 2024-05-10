import numpy as np
import cv2
import matplotlib.pyplot as plt

class TamperHandler:
    def __init__(self):
        self.fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
        self.fgbg.setNMixtures(4)
        self.fgbg.setVarInit(100)

        self.t_fix = 100
        self.t_set = 3
        self.tamper_flag = 0         # f_{n}^{t}
        self.tamper_flag_count = 0   # count++ if tamper_flag=1
        self.tamper_validation_count = 0  # S
        self.fixed_tamper_flag = False    # set by hand, turn off by hand
        self.aedr = 0
        self.edr_sum = 0     # sum(edr) if no tamper attack
        self.bg_temp = None  # to save temp background
        self.w = 200         # size of the tamper validation window (length)   (w > t_fix)

        # store list for plot
        self.aedr_list = []
        self.edr_list = []
        self.aedr_th_list = []
        self.th_list = []

    def detect(self, frame, frame_id, visualized=False):
        frame = cv2.resize(frame, (320, 240))
        if visualized:
            cv2.imshow('frame_rgb', frame)
            cv2.moveWindow('frame_rgb', 500, 50)

        fgmask = self.fgbg.apply(frame, learningRate = 0.002)
        fgmask = cv2.erode(fgmask, np.ones((1,1)))
        fgmask = cv2.dilate(fgmask, np.ones((5, 5)))
        
        if self.tamper_flag == 1:
            bg = self.bg_temp
        else:
            bg = self.fgbg.getBackgroundImage()
            self.bg_temp = bg

        if visualized:
            cv2.imshow('fgmask', fgmask)
            cv2.moveWindow('fgmask', 900, 225)
            cv2.imshow('bg', bg)
            cv2.moveWindow('bg', 100, 50)

        # serve for edr
        background_gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if frame_id > 1:
            mask_to_exclude = (fgmask == 255)

            e_bg = self.sobel_edge_detection(background_gray)
            e_c = self.sobel_edge_detection(frame_gray)

            if visualized:
                cv2.imshow('bg_sobel', e_bg)
                cv2.moveWindow('bg_sobel', 100, 400)
                cv2.imshow('c_sobel', e_c)
                cv2.moveWindow('c_sobel', 500, 400)
            
            # th
            th = self.calculate_th(e_bg)

            # calculate edr
            e_bg[mask_to_exclude] = 0
            e_c[mask_to_exclude] = 0
            edr = 1 - (np.sum(e_bg & e_c)) / (np.sum(e_bg) + 1)     # +1 avoid 0/0

            # TODO: 改善遮擋偵測演算法
            # contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # m = 0.0
            # for contour in contours:
            #     # 計算輪廓的面積
            #     area = cv2.contourArea(contour)
            #     m = max(m, area)
            #     # 如果輪廓面積大於一定閾值，認為畫面中有一大塊白色
            #     if area > (320 * 240 / 2):
            #         self.fixed_tamper_flag = True
            #         break
            
            # aedr
            if self.tamper_flag == 0: # If any tamper attack occurs, the AEDR keeps the current value
                self.edr_sum += edr
                self.aedr = np.sum((1 - self.tamper_flag) * self.edr_sum) / (frame_id - self.tamper_flag_count)

            if frame_id > self.t_fix:    # generating background before monitoring
                # tamper validation count:
                self.tamper_validation_count = self.calculate_tamper_validation_count(self.tamper_validation_count, self.w, edr, self.aedr, th)

                # tamper flag setting:
                self.tamper_flag = 1 if self.tamper_validation_count > self.t_set else 0

                self.tamper_flag_count += self.tamper_flag

                if self.tamper_validation_count > 30:
                    self.fixed_tamper_flag = True

            self.aedr_list.append(self.aedr)
            self.edr_list.append(edr)
            self.aedr_th_list.append(self.aedr + th)
            self.th_list.append(th)

            if visualized: cv2.waitKey(10)

    def show_result(self):
        plt.figure(figsize=(16, 9))
        plt.plot(self.aedr_list, 'r', label='AEDR')
        plt.plot(self.edr_list, 'b', linestyle = ':', label='EDR')
        plt.plot(self.aedr_th_list, 'g', linestyle = '-.', label='AEDR + TH')
        plt.plot(self.th_list, 'brown', linestyle = '--', label='TH')
        plt.ylim([0, 1])
        plt.xlim([0, len(self.th_list)])
        plt.title('If the blue line exceeds the green line, it is camera tamper attack')
        plt.xlabel('Frame number')
        plt.ylabel('Edge disappearance ratio')
        plt.legend()
        plt.show()

    def sobel_edge_detection(self, image_gray, blur_ksize=7, sobel_ksize=1, skipping_threshold=10):
        """

        Input:
        ---
            image_gray: already read by opencv
            blur_ksize: kernel size parameter for Gaussian Blurry
            sobel_ksize: size of the extended Sobel kernel; it must be 1, 3, 5, or 7.
            skipping_threshold: ignore weakly edge

        Output:
        ---
            Edge matrix ∈ {0, 255}
        """
        img_blurred = cv2.GaussianBlur(image_gray, (blur_ksize, blur_ksize), 0)
        # img_blurred = cv2.filter2D(img_blurred, -1, np.array([[-1, -1, -1],[-1,  9, -1],[-1, -1, -1]]))
        img_blurred = cv2.filter2D(img_blurred, -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]))
        
        # sobel algorthm use cv2.CV_64F
        sobelx64f = cv2.Sobel(img_blurred, cv2.CV_64F, 1, 0, ksize=sobel_ksize)
        abs_sobelx64f = np.absolute(sobelx64f)
        img_sobelx = np.uint8(abs_sobelx64f)

        sobely64f = cv2.Sobel(img_blurred, cv2.CV_64F, 0, 1, ksize=sobel_ksize)
        abs_sobely64f = np.absolute(sobely64f)
        img_sobely = np.uint8(abs_sobely64f)
        
        # calculate magnitude
        img_sobel = (img_sobelx + img_sobely) / 2
        
        # ignore weakly pixel
        img_sobel[img_sobel < skipping_threshold] = 0
        img_sobel[img_sobel >= skipping_threshold] = 255
        return np.array(img_sobel, dtype=np.uint8)

    def calculate_th(self, e_bg):
        '''

        Input:
        ---
            e_bg: #edge pixel in the n-th background frame

        Output:
        ---
            Adaptive threshold (All the values used in paper)
        '''
        h, w = e_bg.shape[0], e_bg.shape[1]
        e_bg = np.sum(e_bg) / 255

        if e_bg == 0:
            return -1.0
        elif e_bg / (w * h) < 0.026 * 2:
            return 150 / e_bg
        elif 0.026*2 <= e_bg / (w * h) < 0.046 * 2:
            return 400 / e_bg
        else:
            return 1500 / e_bg

    def calculate_tamper_validation_count(self, tamper_validation_count, w, edr, aedr, th):
        '''
        To prevent wrong false alarms due to noise or temporary scene characteristic change\n
        
        Input:
        ---
            w: Size of the tamper validation window
            edr: Edge disappearance rate
            aedr: Average edge disappearance rate
            th: Adaptive threshold
        
        Output:
        ---
            tamper_validation_count
        '''
        return min(tamper_validation_count + 1, w) if edr > aedr + th else max(tamper_validation_count - 1, 0)