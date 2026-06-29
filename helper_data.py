import torch
import torch.nn.functional as F

def augment_ts(x, jitter_std=0.05, scale_std=0.1, prob=0.5):
    if torch.rand(1).item() > prob:
        return x 

    batch_size, channels, seq_len = x.shape

    noise = torch.randn_like(x) * jitter_std
    x_aug = x + noise

    scale = torch.normal(1.0, scale_std, size=(batch_size, 1, 1)).to(x.device)
    x_aug = x_aug * scale

    return x_aug

def augment_tsa(x, p_mask=0.2, p_crop=0.2, p_warp=0.2, p_noise=0.2, p_scale=0.2):
    batch_size, channels, seq_len = x.shape
    x_aug = x.clone()

    if torch.rand(1).item() < p_noise:
        noise = torch.randn_like(x_aug) * 0.05
        x_aug = x_aug + noise

    if torch.rand(1).item() < p_scale:
        scale = torch.normal(1.0, 0.1, size=(batch_size, 1, 1)).to(x_aug.device)
        x_aug = x_aug * scale

    if torch.rand(1).item() < p_mask:
        mask_ratio = 0.1 # masking 10% of data
        mask_len = max(1, int(seq_len * mask_ratio))
        start = torch.randint(0, seq_len - mask_len, (1,)).item()
        x_aug[:, :, start:start+mask_len] = 0.0

    # if torch.rand(1).item() < p_crop:
    #     crop_ratio = torch.FloatTensor(1).uniform_(0.7, 0.9).item() 
    #     crop_len = max(1, int(seq_len * crop_ratio))
    #     start = torch.randint(0, seq_len - crop_len, (1,)).item()
        
    #     cropped = x_aug[:, :, start:start+crop_len]
    #     x_aug = F.interpolate(cropped, size=seq_len, mode='linear', align_corners=False)

    if torch.rand(1).item() < p_warp:
        split_point = torch.randint(int(seq_len * 0.3), int(seq_len * 0.7), (1,)).item()
        
        part1 = x_aug[:, :, :split_point]
        part2 = x_aug[:, :, split_point:]
        new_len_1 = torch.randint(int(seq_len * 0.2), int(seq_len * 0.8), (1,)).item()
        new_len_2 = seq_len - new_len_1
        
        part1_warped = F.interpolate(part1, size=new_len_1, mode='linear', align_corners=False)
        part2_warped = F.interpolate(part2, size=new_len_2, mode='linear', align_corners=False)
        
        x_aug = torch.cat([part1_warped, part2_warped], dim=-1)

    return x_aug