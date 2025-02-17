import torch, math
import torch.nn as nn
from methods.loss import  CE_softlabel_loss, conf_loss, DiceLoss

def autopad(k, p=None, d=1): 
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

class ConvModule(nn.Sequential):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        self.c1 = c1
        self.c2 = c1
        self.k = k
        self.s = s
        self.p = p
        self.g = g
        self.d = d
        self.act = act
        modules = [
            nn.Conv3d(c1, c2, k, s, autopad(k, p, d), groups = g, dilation = d, bias=False), 
            nn.InstanceNorm3d(c2),
            nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        ]
        super(ConvModule, self).__init__(*modules)

class Bottleneck(nn.Module):
    def __init__(self, c_in, c_out, g, add = True):
        super(Bottleneck, self).__init__()
        if add: c_out = c_in
        self.add = add
        self.cv = nn.Sequential(
            ConvModule(c_in, c_out//2, k=3, s=1),
            ConvModule(c_out//2, c_out, k=3, s=1, g=g)   
        )

    def forward(self, x):
        return x + self.cv(x) if self.add else self.cv(x)
    
class SPPF(nn.Module):
    def __init__(self, c_in, k=5):
        super(SPPF, self).__init__()
        c_ = c_in//2
        self.conv1 = ConvModule(c_in, c_, k=1, s=1)
        self.m = nn.MaxPool3d(kernel_size=k, stride=1, padding=k // 2)
        self.conv2 = ConvModule(c_* 4, c_in, k=1, s=1)

    def forward(self, x):
        x = self.conv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.conv2(torch.cat([x, y1, y2, self.m(y2)], dim = 1))

class CSPBlock(nn.Module):
    def __init__(self, c_in, c_out, g, n = 3, add = True):
        super(CSPBlock, self).__init__()
        self.c_ = c_out//2
        self.in_conv = ConvModule(c_in, c_out, k=1, s=1)
        self.m = nn.ModuleList([Bottleneck(self.c_, self.c_, g=g, add = add) for _ in range(n)])
        self.out_conv = ConvModule(self.c_ * (n+2), c_out, k=1, s=1)

    def forward(self, x):
        y = list(self.in_conv(x).split((self.c_, self.c_), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.out_conv(torch.cat(y, 1))
    
class Encoder(nn.Module):
    def __init__(self, c_in, ngf, g, d, w, r):
        super(Encoder, self).__init__()
        self.Layer1 = nn.Sequential(
            ConvModule(int(c_in), int(ngf * w), k=3, s=2, g=g),
            CSPBlock(int(ngf * w), int(ngf * w), g, n = int(3 * d),add=True)
        )
        self.Layer2 = nn.Sequential(
            ConvModule(int(ngf * w), int(2 * ngf * w), k=3, s=2, g=g),
            CSPBlock(int(2 * ngf * w), int(2 * ngf * w), g, int(6 * d),add=True)
        )
        self.Layer3 = nn.Sequential(
            ConvModule(int(2 * ngf * w), int(4 * ngf * w), k=3, s=2, g=g),
            CSPBlock(int(4 * ngf * w), int(4 * ngf * w), g, int(6 * d),add=True)
        )
        self.Layer4 = nn.Sequential(
            ConvModule(int(4 * ngf * w), int(4 * ngf * w * r), k=3, s=2, g=g),
            CSPBlock(int(4 * ngf * w * r), int(4 * ngf * w * r), g, int(3 * d),add=True),
            SPPF(int(4 * ngf * w * r))
        )

    def forward(self, x):
        C0 = x
        C1 = self.Layer1(C0)
        C2 = self.Layer2(C1)
        C3 = self.Layer3(C2)
        C4 = self.Layer4(C3)
        return C1, C2, C3, C4

class Decoder(nn.Module):
    def __init__(self, ngf, g, d, w, r):
        super(Decoder, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')#, align_corners=True)
        self.Layer1 = CSPBlock(int(4 * ngf * w *(1+r)), int(4 * ngf * w), g, n = int(3*d), add=False)
        self.Layer0 = CSPBlock(int(6 * ngf * w), int(2 * ngf * w), g, n = int(3*d), add=False)
        self.Layer_m1 = CSPBlock(int(3 * ngf * w), int(1 * ngf * w), g, n = int(3*d), add=False)

    def forward(self, C1, C2, C3, C4):
        C3 = torch.cat((self.up(C4), C3), dim=1)
        C3  = self.Layer1(C3)
        C2 = torch.cat((self.up(C3), C2), dim=1)
        C2  = self.Layer0(C2)
        C1 = torch.cat((self.up(C2), C1), dim=1)
        C1  = self.Layer_m1(C1)
        return C1, C2, C3, C4

class Bottom_up(nn.Module):
    def __init__(self, ngf, g, d, w, r):
        super(Bottom_up, self).__init__()
        self.DownSample_0 = ConvModule(int(2 * ngf * w), int(2 * ngf * w), k=3, s=2, g = g)
        self.BottomUpLayer_0 = CSPBlock(int(6 * ngf * w ), int(4 * ngf * w), g, n = int(3*d), add=False)
        self.DownSample_1 = ConvModule(int(4 * ngf * w), int(4 * ngf * w), k=3, s=2, g = g)
        self.BottomUpLayer_1 = CSPBlock(int(4 * ngf * w * (1+r)), int(4 * ngf * w * r), g, n = int(3*d), add=False)

        self.DownSample_m1 = ConvModule(int(1 * ngf * w), int(1 * ngf * w), k=3, s=2, g = g)
        self.BottomUpLayer_m1 = CSPBlock(int(3 * ngf * w ), int(2 * ngf * w), g, n = int(3*d), add=False)

    def forward(self, C1, C2, C3, C4):
        C2 = self.BottomUpLayer_m1(torch.cat((self.DownSample_m1(C1), C2), dim=1))
        C3 = self.BottomUpLayer_0(torch.cat((self.DownSample_0(C2), C3), dim=1))
        C4 = self.BottomUpLayer_1(torch.cat((self.DownSample_1(C3), C4), dim=1))
        return C2, C3, C4

class Head(nn.Module):
    def __init__(self, in_channels, g, num_classes):
        super(Head, self).__init__()
        self.cls = nn.Sequential(
            ConvModule(in_channels, in_channels, k=3, s=1, g = g),
            ConvModule(in_channels, in_channels, k=3, s=1, g = g),
            nn.Conv3d(in_channels, num_classes, 1, 1, 0, bias=True)  
        )

    def forward(self, x):
        return self.cls(x)
  
class Proto(nn.Module):
    def __init__(self, c1, c_=256, c2=32):
        super().__init__()
        self.cv1 = ConvModule(c1, c_, k=3)
        self.upsample = nn.Upsample(scale_factor = 2, mode='trilinear', align_corners=False)
        self.cv2 = ConvModule(c_, c_, k=3)
        self.cv3 = nn.Conv3d(c_, c2, 1, 1, 0)

    def forward(self, x):
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))

class masked_multiscale_fusion(nn.Module):
    def __init__(self, ngf, g, d, w, r, num_classes):
        super().__init__()
        self.mp = nn.MaxPool3d(2)
        self.bottomup = Bottom_up(ngf, g, d, w, r)
        self.head_C2 = Head(int(2 * ngf * w), g, num_classes + 1)
        self.head_C3 = Head(int(4 * ngf * w), g, num_classes + 1)
        self.head_C4 = Head(int(4 * ngf * w * r), g, num_classes + 1)

    def forward(self, pred_mask, C1, C2, C3, C4):
        pred_mask_prob = pred_mask.softmax(1)[:,1,...].unsqueeze(1)
        C2, C3, C4 = self.bottomup(C1 * self.mp(pred_mask_prob), C2 * self.mp(self.mp(pred_mask_prob)), \
                                   C3 * self.mp(self.mp(self.mp(pred_mask_prob))), C4* self.mp(self.mp(self.mp(self.mp(pred_mask_prob)))))
        C2_cls = self.head_C2(C2)
        C3_cls = self.head_C3(C3)
        C4_cls = self.head_C4(C4)
        pred_label_list = [C2_cls, C3_cls, C4_cls]
        return pred_label_list

class GliomaNet(nn.Module):
    def __init__(self, c_in = 4, ngf = 64, g = 1, d=0.67, w = 0.75, r = 1.5, num_classes = 7, num_regions = 2):
        super(GliomaNet, self).__init__()
        self.encoder = Encoder(c_in, ngf, g, d, w, r)
        self.decoder = Decoder(ngf, g, d, w, r)
        self.segout = Proto(int(ngf * w), c_=int(0.25*ngf * w), c2=num_regions)
        self.mmsfm = masked_multiscale_fusion(ngf, g, d, w, r, num_classes)

    def forward(self, x):
        downs = self.encoder(x)
        C1, C2, C3, C4 = self.decoder(*downs)
        pred_mask = self.segout(C1)
        pred_label_list = self.mmsfm(pred_mask, C1, C2, C3, C4)

        return pred_label_list, pred_mask

class VAE(nn.Module):
    def __init__(self, ngf, g, d, w, r, out_channels, decoder_input_chwd = (16, 10, 10, 8)):
        super(VAE, self).__init__()
        self.decoder_input_chwd = decoder_input_chwd
        in_project = 1
        for ele in self.decoder_input_chwd:
            in_project *= ele
        self.VD = nn.Sequential(
            nn.InstanceNorm3d(int(4 * ngf * w * r)),
            nn.ReLU(),
            nn.Conv3d(int(4 * ngf * w * r), 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.Flatten(),
            nn.Linear(in_project, 256)
        )
        self.Project = nn.Linear(128, in_project)
        self.VU = nn.Sequential(
            nn.ReLU(),
            nn.Conv3d(16, int(4 * ngf * w * r), kernel_size=1, stride=1, padding=0, bias=False),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        )
        self.ups = nn.Sequential(nn.Conv3d(int(4 * ngf * w * r), int(4 * ngf * w), kernel_size =1, stride = 1, padding = 0),
                            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
                            CSPBlock(int(4 * ngf * w), int(4 * ngf * w), g, int(6 * d),add=True),
                            nn.Conv3d(int(4 * ngf * w), int(2 * ngf * w), kernel_size =1, stride = 1, padding = 0),
                            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
                            CSPBlock(int(2 * ngf * w), int(2 * ngf * w), g, int(6 * d),add=True),
                            nn.Conv3d(int(2 * ngf * w), int(ngf * w), kernel_size =1, stride = 1, padding = 0),
                            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
                            CSPBlock(int(ngf * w), int(ngf * w), g, n = int(3 * d),add=True),
                            )

        self.Vend = Proto(int(ngf * w), c_=int(0.25*ngf * w), c2=out_channels)

    def forward(self, x):
        mean, logvar = self.VD(x).chunk(2, -1)
        eps = torch.randn_like(logvar)
        std = torch.exp(logvar / 2)
        eps = torch.randn_like(std)
        z = eps * std + mean
        x = self.Project(z).reshape(-1, *self.decoder_input_chwd)
        decode = self.VU(x)
        decode = self.ups(decode)
        decode = self.Vend(decode)
        return decode, mean, std

class GliomaNet_VAE(nn.Module):
    def __init__(self, c_in = 4, ngf = 64, g = 1, d=0.67, w = 0.75, r = 1.5, num_regions = 2):
        super(GliomaNet_VAE, self).__init__()
        self.encoder = Encoder(c_in, ngf, g, d, w, r)
        self.decoder = Decoder(ngf, g, d, w, r)
        self.segout = Proto(int(ngf * w), c_=int(0.25*ngf * w), c2=num_regions)
        self.vae = VAE(ngf, g, d, w, r, c_in, decoder_input_chwd = (16,5,5,4))

    def forward(self, x):
        downs = self.encoder(x)
        C1, C2, C3, C4 = self.decoder(*downs)
        pred_mask = self.segout(C1)
        decode, mean, logvar = self.vae((C4))
        return pred_mask, decode, mean, logvar

class MappingLoss(nn.Module): 
    def __init__(self, device = 'cuda'):
        super().__init__()
        self.seg_ratio      = 1.
        self.cre            = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.1,0.9]))
        self.grade_weight   = torch.tensor([0.3,0.5,0.2]).to(device)
        self.idh_weight     = torch.tensor([0.33,0.67]).to(device)
        self.p19q_weight    = torch.tensor([0.25,0.75]).to(device)
        self.diceloss       = DiceLoss()
        self.mp             = nn.MaxPool3d(2)

    def forward(self, pred_label_list, pred_mask, gt_label, gt_bbox, gt_mask, train = True):
        loss_conf_list  = []
        loss_grade_list   = []
        loss_idh_list   = []
        loss_p19q_list  = []
        pred_conf_allch = []
        pred_cls_allch  = []
        pred_cls_prob_allch=[]

        pred_mask_prob_map = pred_mask.softmax(1)[:,1,...].unsqueeze(1).detach()
        for ch in range(len(pred_label_list)):
            conf = pred_label_list[ch][:,0,...] 
            pred_label_ch = pred_label_list[ch][:,1:,...]
            if train:
                loss_conf_list  += conf_loss(conf, gt_bbox)
                loss_grade_list += CE_softlabel_loss(pred_label_ch[:,:3], gt_label[...,0], gt_bbox,weight_tensor=self.grade_weight)
                loss_idh_list   += CE_softlabel_loss(pred_label_ch[:,3:5], gt_label[...,1], gt_bbox, weight_tensor = self.idh_weight)
                loss_p19q_list  += CE_softlabel_loss(pred_label_ch[:,5:7], gt_label[...,2], gt_bbox, weight_tensor = self.p19q_weight)

            i = int(math.log2(pred_mask_prob_map.shape[-1] // pred_label_list[ch].shape[-1]))
            mapping = pred_mask_prob_map
            for _ in range(i):
                mapping = self.mp(mapping)
            mapping /= mapping.sum(tuple(range(1,mapping.dim())))
            
            pred_conf = (conf.sigmoid() * mapping).sum((-1,-2,-3))

            pred_label_ch  = torch.cat((pred_label_ch[:,:3].softmax(1),\
                                            pred_label_ch[:,3:5].softmax(1),\
                                            pred_label_ch[:,5:7].softmax(1)), dim=1)
            pred_cls_prob  = (pred_label_ch * mapping).sum((-1,-2,-3))
            
            pred_cls       = torch.cat((torch.argmax(pred_cls_prob[:,:3], dim = 1, keepdim=True),\
                                            torch.argmax(pred_cls_prob[:,3:5], dim = 1, keepdim=True),\
                                            torch.argmax(pred_cls_prob[:,5:7], dim = 1, keepdim=True)), dim=1)
            
            pred_conf_allch.append(pred_conf.unsqueeze(-1))
            pred_cls_prob_allch.append(pred_cls_prob.unsqueeze(1))
            pred_cls_allch.append(pred_cls.unsqueeze(1))            
        
        pred_conf_allch= torch.cat(pred_conf_allch, dim=1)
        pred_cls_allch = torch.cat(pred_cls_allch, dim=1)
        pred_cls_prob_allch = torch.cat(pred_cls_prob_allch, dim=1)
        pred_conf, pos  = torch.max(pred_conf_allch, dim = 1)
        for batch_id in range(pos.shape[0]): 
            pred_cls[batch_id]       = pred_cls_allch[batch_id,pos[batch_id]]
            pred_cls_prob[batch_id]  = pred_cls_prob_allch[batch_id,pos[batch_id]]
        
        if train:
            loss_seg = self.diceloss(pred_mask.softmax(1), gt_mask) + self.cre(pred_mask, gt_mask)
            loss_conf = sum(loss_conf_list)/len(loss_conf_list)
            loss_QFL = 1.0720 * sum(loss_grade_list)/len(loss_grade_list) + 1.640 * sum(loss_idh_list)/len(loss_idh_list) + 2.891 *sum(loss_p19q_list)/len(loss_p19q_list)
            loss = 1*loss_conf + 1.*loss_QFL + self.seg_ratio *loss_seg
            return loss, loss_conf, loss_QFL, loss_seg, pred_conf, pred_cls
        else:
            return pred_cls_prob, pred_cls
