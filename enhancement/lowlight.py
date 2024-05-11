import cv2
import numpy as np
import matplotlib.pyplot as plt

class LowLightEnhancer:
    def enhance(self, image, fgmask, e_bg, clip_hist_percent=1):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate grayscale histogram
        mask = np.logical_and(fgmask != 255, e_bg != 255)
        hist = cv2.calcHist([gray[mask]], [0], None, [256], [0, 256])
        hist_size = len(hist)
        
        # Calculate cumulative distribution from the histogram
        accumulator = []
        accumulator.append(float(hist[0]))
        for index in range(1, hist_size):
            accumulator.append(accumulator[index -1] + float(hist[index]))
        
        # Locate points to clip
        maximum = accumulator[-1]
        clip_hist_percent *= (maximum/100.0)
        clip_hist_percent /= 2.0
        
        # Locate left cut
        minimum_gray = 0
        while accumulator[minimum_gray] < clip_hist_percent:
            minimum_gray += 1
        
        # Locate right cut
        maximum_gray = hist_size -1
        while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
            maximum_gray -= 1
        
        # Calculate alpha and beta values
        alpha = 255 / (maximum_gray - minimum_gray)
        beta = -minimum_gray * alpha
        
        '''
        # Calculate new histogram with desired range and show histogram 
        new_hist = cv2.calcHist([gray],[0],None,[256],[minimum_gray,maximum_gray])
        plt.plot(hist)
        plt.plot(new_hist)
        plt.xlim([0,256])
        plt.show()
        '''

        auto_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        auto_result = cv2.convertScaleAbs(auto_result, alpha=1.5, beta=50)
        auto_result = cv2.GaussianBlur(auto_result, (25, 25), 0, 0)
        return auto_result