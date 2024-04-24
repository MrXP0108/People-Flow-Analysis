import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse
from tqdm import tqdm
from enhancement.data.data import *
from torchvision import transforms
from torch.utils.data import DataLoader
from enhancement.loss.losses import *
from enhancement.net.CIDNet import CIDNet

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
                self.alpha = 0.84
            elif self.best_PSNR:
                weight_path = 'enhancement/weights/LOLv2_real/best_PSNR.pth'
                self.alpha = 0.8
            elif self.best_SSIM:
                weight_path = 'enhancement/weights/LOLv2_real/best_SSIM.pth'
                self.alpha = 0.82
                
        elif self.lol_v2_syn:
            if self.perc:
                weight_path = 'enhancement/weights/LOLv2_syn/w_perc.pth'
            else:
                weight_path = 'enhancement/weights/LOLv2_syn/DVCNet_epoch_320_best.pth'

        # eval_net = CIDNet().cuda()
        self.eval_net = CIDNet().cpu()
        self.eval_net.load_state_dict(torch.load(weight_path, map_location=lambda storage, loc: storage))
        self.eval_net.eval()

    def enhance(self):
        eval_data = DataLoader(dataset=get_eval_set('temp'), num_workers=1, batch_size=1, shuffle=False)
        torch.set_grad_enabled(False)
        if self.lol:
            self.eval_net.trans.gated = True
        else:
            self.eval_net.trans.gated2 = True
            self.eval_net.trans.alpha = self.alpha
        for batch in eval_data:
            with torch.no_grad():
                input = batch[0].cpu()
                # input = batch[0].cuda()
                output = self.eval_net(input)
                
            # output = torch.clamp(output.cuda(),0,1).cuda()
            output = torch.clamp(output.cpu(),0,1).cpu()
            output_img = transforms.ToPILImage()(output.squeeze(0))
            output_img.save('temp/temp.png')
            torch.cuda.empty_cache()
        if self.lol:
            self.eval_net.trans.gated = False
        else:
            self.eval_net.trans.gated2 = False
        torch.set_grad_enabled(True)