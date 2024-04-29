import torch
import torch.nn as nn
import numpy as np

pi = 3.141592653589793
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RGB_HVI(nn.Module):
    def __init__(self):
        super(RGB_HVI, self).__init__()
        self.density_k = torch.nn.Parameter(torch.full([1],0.2)) # k is reciprocal to the paper mentioned
        self.gated = False
        self.gated2= False
        self.alpha = 1.0
        self.this_k = 0
        
    def HVIT(self, img):
        eps = 1e-8
        device = img.device
        dtypes = img.dtype
        hue = torch.Tensor(img.shape[0], img.shape[2], img.shape[3]).to(device).to(dtypes)
        value = img.max(1)[0].to(dtypes)
        img_min = img.min(1)[0].to(dtypes)
        hue[img[:,2]==value] = 4.0 + ( (img[:,0]-img[:,1]) / (value - img_min + eps)) [img[:,2]==value]
        hue[img[:,1]==value] = 2.0 + ( (img[:,2]-img[:,0]) / (value - img_min + eps)) [img[:,1]==value]
        hue[img[:,0]==value] = (0.0 + ((img[:,1]-img[:,2]) / (value - img_min + eps)) [img[:,0]==value]) % 6

        hue[img.min(1)[0]==value] = 0.0
        hue = hue/6.0

        saturation = (value - img_min ) / (value + eps )
        saturation[value==0] = 0

        hue = hue.unsqueeze(1)
        saturation = saturation.unsqueeze(1)
        value = value.unsqueeze(1)
        
        k = self.density_k
        self.this_k = k.item()
        
        color_sensitive = ((value * 0.5 * pi).sin() + eps).pow(k)
        cx = (2.0 * pi * hue).cos()
        cy = (2.0 * pi * hue).sin()
        X = color_sensitive * saturation * cx
        Y = color_sensitive * saturation * cy
        Z = value
        xyz = torch.cat([X, Y, Z],dim=1)
        return xyz
    
    def PHVIT(self, img):
        eps = 1e-8
        H,V,I = img[:,0,:,:], img[:,1,:,:], img[:,2,:,:]
        
        # clip
        H = torch.clamp(H,-1,1)
        V = torch.clamp(V,-1,1)
        I = torch.clamp(I,0,1)
        
        v = I
        k = self.this_k
        color_sensitive = ((v * 0.5 * pi).sin() + eps).pow(k)
        H = (H) / (color_sensitive + eps)
        V = (V) / (color_sensitive + eps)
        H = torch.clamp(H,-1,1)
        V = torch.clamp(V,-1,1)
        h = torch.atan2(V,H) / (2*pi)
        h = h%1
        s = torch.sqrt(H**2 + V**2)
        
        if self.gated:
            s = s * 1.3
        
        s = torch.clamp(s,0,1)
        v = torch.clamp(v,0,1)
        
        hi = torch.floor(h * 6.0)
        f = h * 6.0 - hi
        p = v * (1. - s)
        q = v * (1. - (f * s))
        t = v * (1. - ((1. - f) * s))
        
        # Convert tensors to NumPy arrays
        hi_np = hi.cpu().numpy()
        r_np = np.zeros_like(hi_np)
        g_np = np.zeros_like(hi_np)
        b_np = np.zeros_like(hi_np)
        p_np = p.cpu().numpy()
        q_np = q.cpu().numpy()
        t_np = t.cpu().numpy()
        v_np = v.cpu().numpy()

        # Convert hi_np to boolean arrays
        hi0 = hi_np == 0
        hi1 = hi_np == 1
        hi2 = hi_np == 2
        hi3 = hi_np == 3
        hi4 = hi_np == 4
        hi5 = hi_np == 5

        # Perform computations with NumPy arrays
        r_np[hi0] = v_np[hi0]
        g_np[hi0] = t_np[hi0]
        b_np[hi0] = p_np[hi0]

        r_np[hi1] = q_np[hi1]
        g_np[hi1] = v_np[hi1]
        b_np[hi1] = p_np[hi1]

        r_np[hi2] = p_np[hi2]
        g_np[hi2] = v_np[hi2]
        b_np[hi2] = t_np[hi2]

        r_np[hi3] = p_np[hi3]
        g_np[hi3] = q_np[hi3]
        b_np[hi3] = v_np[hi3]

        r_np[hi4] = t_np[hi4]
        g_np[hi4] = p_np[hi4]
        b_np[hi4] = v_np[hi4]

        r_np[hi5] = v_np[hi5]
        g_np[hi5] = p_np[hi5]
        b_np[hi5] = q_np[hi5]

        # Convert NumPy arrays back to tensors
        r = torch.tensor(r_np, device=device).unsqueeze(1)
        g = torch.tensor(g_np, device=device).unsqueeze(1)
        b = torch.tensor(b_np, device=device).unsqueeze(1)
                
        rgb = torch.cat([r, g, b], dim=1)
        if self.gated2:
            rgb = rgb * self.alpha
        return rgb
