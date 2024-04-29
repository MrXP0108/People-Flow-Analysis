import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from torchvision import transforms
from enhancement.loss.losses import *
from enhancement.net.CIDNet import CIDNet
from PIL import Image
import cv2
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Enhancer:
    def __init__(self, perc=False, lol=False, lol_v2_real=False, lol_v2_syn=False, \
                 best_GT_mean=False, best_PSNR=False, best_SSIM=False, alpha=1.0):
        self.perc = perc
        self.lol = lol
        self.lol_v2_real = lol_v2_real
        self.lol_v2_syn = lol_v2_syn
        self.best_GT_mean = best_GT_mean
        self.best_PSNR = best_PSNR
        self.best_SSIM = best_SSIM
        self.alpha = alpha
        """
        lol + perc (weights that trained with perceptual loss)
        lol        (weights that trained without perceptual loss)

        lol_v2_real + best_GT_mean (or best_PSNR or best_SSIM)

        lol_v2_syn + perc (weights that trained with perceptual loss)
        lol_v2_syn        (weights that trained without perceptual loss)
        """
        if self.lol:
            if self.perc:
                weight_path = 'enhancement/weights/LOLv1/w_perc.pth'
            else:
                weight_path = 'enhancement/weights/LOLv1/wo_perc.pth'
                
        elif self.lol_v2_real:
            if self.best_GT_mean:
                weight_path = 'enhancement/weights/LOLv2_real/w_perc.pth'
            elif self.best_PSNR:
                weight_path = 'enhancement/weights/LOLv2_real/best_PSNR.pth'
            elif self.best_SSIM:
                weight_path = 'enhancement/weights/LOLv2_real/best_SSIM.pth'
                
        elif self.lol_v2_syn:
            if self.perc:
                weight_path = 'enhancement/weights/LOLv2_syn/w_perc.pth'
            else:
                weight_path = 'enhancement/weights/LOLv2_syn/wo_perc.pth'

        self.eval_net = CIDNet()
        self.eval_net.to(device)
        self.eval_net.load_state_dict(torch.load(weight_path, map_location=lambda storage, loc: storage))
        self.eval_net.eval()

    def enhance(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        eval_data = Image.fromarray(img)
        eval_data = transforms.ToTensor()(eval_data).unsqueeze(0)
        torch.set_grad_enabled(False)
        if self.lol:
            self.eval_net.trans.gated = True
        else:
            self.eval_net.trans.gated2 = True
            self.eval_net.trans.alpha = self.alpha
        with torch.no_grad():
            input = eval_data.to(device)
            output = self.eval_net(input)
                
        output = torch.clamp(output.to(device),0,1).to(device)
        output_img = np.asarray(transforms.ToPILImage()(output.squeeze(0)))
        output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)

        torch.cuda.empty_cache()
        if self.lol:
            self.eval_net.trans.gated = False
        else:
            self.eval_net.trans.gated2 = False
        torch.set_grad_enabled(True)

        return output_img