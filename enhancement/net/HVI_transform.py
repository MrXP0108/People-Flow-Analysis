import torch
import torch.nn as nn
import numpy as np
import cupy as cp

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
        
        if self.gated: s *= 1.3
        
        s = torch.clamp(s,0,1)
        v = torch.clamp(v,0,1)
        
        hi = torch.floor(h * 6.0)
        f = h * 6.0 - hi
        p = v * (1. - s)
        q = v * (1. - (f * s))
        t = v * (1. - ((1. - f) * s))
        r, g, b = None, None, None

        if device == 'cuda':
            # Convert tensors to CuPy arrays directly on GPU
            hi_cp = cp.asarray(hi)
            p_cp = cp.asarray(p)
            q_cp = cp.asarray(q)
            t_cp = cp.asarray(t)
            v_cp = cp.asarray(v)

            # Create boolean arrays directly on GPU
            hi_eq = (hi_cp == cp.arange(6)[:, None, None])

            # Perform computations using CuPy vectorized operations
            r_cp = cp.where(hi_eq[0], v_cp, cp.where(hi_eq[1], q_cp, cp.where(hi_eq[2], p_cp, cp.where(hi_eq[3], p_cp, cp.where(hi_eq[4], t_cp, v_cp)))))
            g_cp = cp.where(hi_eq[0], t_cp, cp.where(hi_eq[1], v_cp, cp.where(hi_eq[2], v_cp, cp.where(hi_eq[3], q_cp, cp.where(hi_eq[4], p_cp, p_cp)))))
            b_cp = cp.where(hi_eq[0], p_cp, cp.where(hi_eq[1], p_cp, cp.where(hi_eq[2], t_cp, cp.where(hi_eq[3], v_cp, cp.where(hi_eq[4], v_cp, q_cp)))))

            # Convert CuPy arrays back to tensors on GPU
            r = torch.tensor(cp.asnumpy(r_cp), device=device).unsqueeze(1)
            g = torch.tensor(cp.asnumpy(g_cp), device=device).unsqueeze(1)
            b = torch.tensor(cp.asnumpy(b_cp), device=device).unsqueeze(1)
        else:
            # Convert tensors to NumPy arrays
            hi_np = hi.cpu().numpy()
            p_np = p.cpu().numpy()
            q_np = q.cpu().numpy()
            t_np = t.cpu().numpy()
            v_np = v.cpu().numpy()

            # Create boolean arrays directly
            hi_eq = (hi_np == np.arange(6)[:, None, None])

            # Perform computations using NumPy vectorized operations
            r_np = np.where(hi_eq[0], v_np, np.where(hi_eq[1], q_np, np.where(hi_eq[2], p_np, np.where(hi_eq[3], p_np, np.where(hi_eq[4], t_np, v_np)))))
            g_np = np.where(hi_eq[0], t_np, np.where(hi_eq[1], v_np, np.where(hi_eq[2], v_np, np.where(hi_eq[3], q_np, np.where(hi_eq[4], p_np, p_np)))))
            b_np = np.where(hi_eq[0], p_np, np.where(hi_eq[1], p_np, np.where(hi_eq[2], t_np, np.where(hi_eq[3], v_np, np.where(hi_eq[4], v_np, q_np)))))

            # Convert NumPy arrays back to tensors
            r = torch.tensor(r_np, device=device).unsqueeze(1)
            g = torch.tensor(g_np, device=device).unsqueeze(1)
            b = torch.tensor(b_np, device=device).unsqueeze(1)
                
        rgb = torch.cat([r, g, b], dim=1)
        if self.gated2: rgb *= self.alpha
        return rgb
