import torch
import torch.nn.functional as F

def vae_loss(y, y_hat, mean, logvar):
    recons_loss = F.mse_loss(y_hat, y)
    kl_loss = torch.mean(
        -1 * torch.sum(1 + logvar - mean**2 - torch.exp(logvar), 1), 0)
    loss = recons_loss + kl_loss
    return loss

def compute_dice(input, target):
    eps = 1E-3  
    inter = torch.sum(target * input) 
    union = torch.sum(input) + torch.sum(target) 
    t = (2 * inter.float()+ eps) /(union.float()+ eps)
    return t

def multi_dice(input, target):
    assert input.size() == target.size()
    dice = 0.
    for channel in range(input.shape[1]):
        dice += compute_dice(input[:, channel, ...], target[:, channel, ...])
    return dice / input.shape[1]

class DiceLoss(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    def forward(self, input, targets):
        return 1 - multi_dice(input, targets)

def box_xyzwhd_2_box_x1y1z1_x2y2z2(box: torch.tensor):
    b_xyz,b_whd = box.chunk(2, -1)
    b_whd_half  = b_whd / 2.
    b_mins      = b_xyz - b_whd_half
    b_maxes     = b_xyz + b_whd_half

    b_mins = torch.clamp(b_mins,0,0.995)
    b_maxes = torch.clamp(b_maxes,0,0.995)
    return torch.cat((b_mins, b_maxes), dim=-1)

def conf_loss(pred, gt_bbox, weight = 0.9):
    B, H, W, D = pred.shape
    epsilon = 1E-3
    gt_bbox = box_xyzwhd_2_box_x1y1z1_x2y2z2(gt_bbox)
    x1,y1,z1,x2,y2,z2 = gt_bbox.chunk(6, -1)
    x1,x2 = (x1*H).long(),(x2*H).long()
    y1,y2 = (y1*W).long(),(y2*W).long()
    z1,z2 = (z1*D).long(),(z2*D).long()
    gt_label_bbox = torch.zeros(B, H, W, D, dtype=torch.long).to(pred.device)
    for b in range(B):
        gt_label_bbox[b, x1[b]: x2[b], y1[b]: y2[b], z1[b]: z2[b]] = 1
    pred = pred.sigmoid().clip(epsilon, 1.0 - epsilon)
    loss  = - gt_label_bbox * torch.log(pred)* weight - (1.0 - gt_label_bbox) * torch.log(1.0 - pred)* (1-weight)
    return (loss.reshape(B,-1)).mean(dim=-1)

def CE_softlabel_loss(pred, gt_label, gt_bbox, weight_tensor=torch.tensor([0.3,0.6,0.1])):
    B, _, H, W, D       = pred.shape
    bbox_centers, diagonal_len = (gt_bbox * torch.tensor([H,W,D,H,W,D]).to(pred.device)).chunk(2, -1)
    diagonal_len = torch.norm(diagonal_len, dim=1).reshape(B,1,1,1)
    xx, yy, zz = torch.meshgrid(
        torch.arange(0, H),
        torch.arange(0, W),
        torch.arange(0, D),indexing='ij'
    )
    coordinates = torch.stack((xx, yy, zz), dim=0).expand(B, -1, -1, -1, -1).to(pred.device)

    distances = torch.norm(coordinates - bbox_centers.reshape(*bbox_centers.shape,1,1,1), dim=1)
    norm_distances = (distances/diagonal_len).clamp(0,1)
    gt_label_bbox = torch.zeros_like(pred).to(pred.device)
    for b in range(B):
        if gt_label[b] != -1:
            gt_label_bbox[b, gt_label[b],...] = (1 - norm_distances[b,...]) *4 
    gt_label_bbox = gt_label_bbox.softmax(dim=1)

    loss = F.cross_entropy(pred, gt_label_bbox, \
                           weight=weight_tensor,\
                              reduction="none")

    loss = loss.reshape(B,-1) * (gt_label.unsqueeze(-1) !=-1) 
    return torch.mean(loss, dim=-1)
