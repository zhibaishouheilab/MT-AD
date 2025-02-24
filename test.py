import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from utils.MM_loader import *
from MultiTransAD.model.MMPC import *
from options import Test_Options

import cv2
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

# 初始化线程池执行器，根据需求调整max_workers
executor = ThreadPoolExecutor(max_workers=256)

opt = Test_Options().get_opt()
os.makedirs(opt.img_save_path, exist_ok=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
opt.data_root = './data/BraTS2021/test/'
opt.edge_root = './data/BraTS2021/test_edge/'
opt.checkpoint_path = './weight/BraTS2021/50_MAE.pth'
opt.img_save_path = './output/BraTS2021_test'
os.makedirs(opt.img_save_path, exist_ok=True)
opt.modality = ['t1', 't2', 't1ce', 'flair']
opt.batch_size = 256

def save_image_data(data, path):
    """将二进制图像数据保存到文件"""
    with open(path, 'wb') as f:
        f.write(data)

def save_npz_data(path, **kwargs):
    """异步保存NPZ数据"""
    np.savez(path, **kwargs)

def content_cosine_similarity(zA, zB, maskA=None, maskB=None):
    """
    对两个“内容编码” zA, zB (B, L, D) 做余弦相似度 => (B, L, L)。
    只对 valid_2d 范围内的值做 min-max 归一化(可包含负数)。
    valid_2d 范围外设为 0。
    """
    B, L, D = zA.shape
    # (B,L,1,D) vs (B,1,L,D) => (B,L,L)
    sim_matrix = F.cosine_similarity(zA.unsqueeze(2), zB.unsqueeze(1), dim=-1)  # (B,L,L)

    # 构造 valid_2d 掩码
    # 若 maskA, maskB 不为 None，则 (B,L)
    # valid_2d => (B,L,L)；只对其范围内元素做 min-max
    # 范围外 => 置为 0。
    if maskA is not None and maskB is not None:
        fA = maskA.unsqueeze(2)  # (B,L,1)
        fB = maskB.unsqueeze(1)  # (B,1,L)
        valid_2d = (fA * fB) > 0  # 布尔类型 (B,L,L)
    else:
        # 如果没给 mask，就当全是 valid
        valid_2d = torch.ones_like(sim_matrix, dtype=torch.bool)

    # 首先把范围外的元素设为0
    sim_matrix = sim_matrix.clone()
    sim_matrix[~valid_2d] = 0.0

    # 针对每个batch单独做 min-max
    sim_matrix_norm = torch.zeros_like(sim_matrix)  # (B,L,L)
    eps = 1e-8

    for b in range(B):
        sub_mat = sim_matrix[b]      # (L,L)
        valid_mask = valid_2d[b]     # (L,L) bool

        in_range_vals = sub_mat[valid_mask]  # 此batch下 valid 区域的所有值(可含负数)

        if in_range_vals.numel() > 0:
            min_val = in_range_vals.min()
            max_val = in_range_vals.max()
            denom = (max_val - min_val).clamp_min(eps)
            
            # 在 valid 区域做归一化
            tmp = sub_mat.clone()
            tmp_valid = (tmp[valid_mask] - min_val) / denom
            tmp[valid_mask] = tmp_valid

            sim_matrix_norm[b] = tmp
        else:
            # valid_2d 全 False，这张图没有前景 => 全0
            sim_matrix_norm[b] = 0.0

    return sim_matrix_norm



def get_diagonal_map(sim_matrix):
    """
    从 sim_matrix (B, L, L) 中对每张 (L,L) 的对角线做 min-max；逐图处理。
    取对角线后 reshape => (B, side, side)。
    """
    B, L, _ = sim_matrix.shape
    diag = torch.diagonal(sim_matrix, dim1=1, dim2=2)  # => (B, L)

    side = int(L**0.5)
    diag_map = diag.reshape(B, side, side)  # (B, side, side)
    #print(diag_map[diag_map>0].min())

    diag_map_norm = torch.zeros_like(diag_map)
    eps = 1e-8
    for b in range(B):
        sub = diag_map[b]  # (side, side)
        pos_vals = sub[sub > 0]
        if pos_vals.numel() > 0:
            min_val = pos_vals.min()
            max_val = pos_vals.max()
            denom = (max_val - min_val).clamp_min(eps)

            tmp = sub.clone()
            mask_pos = (sub > 0)
            tmp[mask_pos] = (sub[mask_pos] - min_val) / denom
            diag_map_norm[b] = tmp
        else:
            # 没有正值 => 原样 or 全0
            diag_map_norm[b] = sub

    return diag_map_norm


# 1) 初始化模型
mae = MultiModalPatchMAE(
    img_size=opt.img_size,
    patch_size=opt.patch_size,
    embed_dim=opt.dim_encoder,
    depth=opt.depth,
    num_heads=opt.num_heads,
    in_chans=1,
    decoder_embed_dim=opt.dim_decoder,
    decoder_depth=opt.decoder_depth,
    decoder_num_heads=opt.decoder_num_heads,
    mlp_ratio=opt.mlp_ratio,
    norm_layer=nn.LayerNorm
).to(device)

# 2) 读取训练好的权重
print('load checkpoint......', opt.checkpoint_path)
mae.load_state_dict(torch.load(opt.checkpoint_path, map_location=device), strict=False)

# 3) 构建 dataloader（或 test_loader）
test_loader = get_maeloader(
    batchsize=opt.batch_size, 
    shuffle=False, 
    pin_memory=True, 
    img_size=opt.img_size,
    img_root=opt.data_root, 
    num_workers=opt.num_workers, 
    augment=False,
    if_addlabel=True,
    modality=opt.modality,
    edge_root=opt.edge_root
)

#######################
# 两两翻译 + 可视化
#######################

for i, (M1, M2, M3, M4, M1_edge, M2_edge, M3_edge, M4_edge, label) in enumerate(test_loader):
    M1 = M1.to(device, dtype=torch.float)
    M2 = M2.to(device, dtype=torch.float)
    M3 = M3.to(device, dtype=torch.float)
    M4 = M4.to(device, dtype=torch.float)
    M1_edge = M1_edge.to(device, dtype=torch.float)
    M2_edge = M2_edge.to(device, dtype=torch.float)
    M3_edge = M3_edge.to(device, dtype=torch.float)
    M4_edge = M4_edge.to(device, dtype=torch.float)
    target_edge = [M1_edge, M2_edge, M3_edge, M4_edge]
    label = label.to(device, dtype=torch.long)
    B = M1.size(0)

    with torch.no_grad():
        ################################
        # 1) 分别计算各模态的 foreground mask
        #    以及 content、style
        ################################
        # -- 计算前景 mask --
        p_M1 = mae.patchify(M1)
        p_M2 = mae.patchify(M2)
        p_M3 = mae.patchify(M3)
        p_M4 = mae.patchify(M4)
        mask_M1 = (p_M1.var(dim=-1) > 0).float()  # (B, L)
        mask_M2 = (p_M2.var(dim=-1) > 0).float()
        mask_M3 = (p_M3.var(dim=-1) > 0).float()
        mask_M4 = (p_M4.var(dim=-1) > 0).float()
        masks = [mask_M1, mask_M2, mask_M3, mask_M4]

        # -- Content + Style --
        # 对应新版模型写法: patch_embed + CLS + pos_embed 后再 content_encoder、style_encoder
        # 这里手工拆解，让测试代码和训练保持一致
        #   => content_encoder_M1: 
        #   => style_encoder_M1:   (内部是 self.style_encoder_M1(...) => 2D -> chunk)
        
        # M1
        embed_M1 = mae.patch_embed(M1)
        embed_M2 = mae.patch_embed(M2)
        embed_M3 = mae.patch_embed(M3)
        embed_M4 = mae.patch_embed(M4)
        embed_M1 = torch.cat([mae.cls_token.expand(embed_M1.shape[0], -1, -1), embed_M1], dim=1)
        embed_M2 = torch.cat([mae.cls_token.expand(embed_M2.shape[0], -1, -1), embed_M2], dim=1)
        embed_M3 = torch.cat([mae.cls_token.expand(embed_M3.shape[0], -1, -1), embed_M3], dim=1)
        embed_M4 = torch.cat([mae.cls_token.expand(embed_M4.shape[0], -1, -1), embed_M4], dim=1)
        embed_M1 = embed_M1 + mae.pos_embed
        embed_M2 = embed_M2 + mae.pos_embed
        embed_M3 = embed_M3 + mae.pos_embed
        embed_M4 = embed_M4 + mae.pos_embed
        
        # 3) 计算风格信息 (从 embed 层计算 style)
        style_mean_M1, style_std_M1 = mae.compute_style_mean_std(embed_M1)
        style_mean_M2, style_std_M2 = mae.compute_style_mean_std(embed_M2)
        style_mean_M3, style_std_M3 = mae.compute_style_mean_std(embed_M3)
        style_mean_M4, style_std_M4 = mae.compute_style_mean_std(embed_M4)
        
        z_M1_c = embed_M1
        z_M2_c = embed_M2
        z_M3_c = embed_M3
        z_M4_c = embed_M4
        for blk in mae.content_encoder:
            z_M1_c = blk(z_M1_c)
            z_M2_c = blk(z_M2_c)
            z_M3_c = blk(z_M3_c)
            z_M4_c = blk(z_M4_c)

        contents = [z_M1_c, z_M2_c, z_M3_c, z_M4_c]
        style_means = [style_mean_M1, style_mean_M2, style_mean_M3, style_mean_M4]
        style_stds  = [style_std_M1,  style_std_M2,  style_std_M3,  style_std_M4]
        real_imgs = [M1, M2, M3, M4]

        ################################
        # 2) 两两翻译： s->t
        ################################
        translations = [[None]*4 for _ in range(4)]
        for t_idx in range(4):
            for s_idx in range(4):
                if s_idx == t_idx:
                    continue
                # 1) decoder_embed
                z_s_dec = mae.decoder_embed(contents[s_idx])
                cls_token = z_s_dec[:, :1, :]         # (B, 1, decoder_embed_dim)
                patch_tokens = z_s_dec[:, 1:, :] 
                edge_latent = mae.decoder_embed(mae.patch_embed_edge(target_edge[t_idx]))
                
                concat_tokens = torch.cat([patch_tokens, edge_latent], dim=-1)  # (B, N, 2*decoder_embed_dim)
                fused_patch = mae.edge_fuse(concat_tokens)
                fused_c = torch.cat([cls_token, fused_patch], dim=1)  
                # 2) + decoder_pos_embed
                fused_c = fused_c + mae.decoder_pos_embed[:, :fused_c.size(1), :]
                # 3) adain => style of t
                z_s_to_t = mae.adain(fused_c, style_means[t_idx], style_stds[t_idx])
                # 4) generator_t
                p_gen = mae.generator(z_s_to_t)
                # 5) 去掉CLS patch
                p_gen = p_gen[:, 1:, :]
                # 6) unpatchify
                M_s_to_t = mae.unpatchify(p_gen)
                translations[t_idx][s_idx] = M_s_to_t

        ################################
        # 3) 对每个 batch 的可视化和指标计算
        ################################
        for b_idx in range(B):
            global_idx = i * opt.batch_size + b_idx + 1
            #if global_idx != 159:
            #    continue
            
            # 新增：初始化字典保存每个翻译对的组合误差图
            err_mix_dict = {}
            err_mix_dict['Label'] = label[b_idx].squeeze().cpu().numpy()
            err_dict = {}
            err_dict['Label'] = label[b_idx].squeeze().cpu().numpy()


            err_avg = torch.zeros(opt.img_size, opt.img_size)

            for t_idx in range(4):
                # 第 (t_idx,0) 放 原图
                target_img = real_imgs[t_idx][b_idx]  # shape (1,H,W)      
                target_label = label[b_idx]  # (1,H,W)


                col = 0
                for s_idx in range(4):
                    if s_idx == t_idx:
                        continue                    
                    trans_img = translations[t_idx][s_idx][b_idx]  # shape (1, H, W)
                    mask_s = masks[s_idx][b_idx]  # (L,)

                    # 2) 绝对误差
                    abs_err = (trans_img - target_img).abs().cpu().numpy()

                    # 3) 相似度矩阵 
                    z_t = contents[t_idx][b_idx][1:,:] # (L,D)
                    z_s = contents[s_idx][b_idx][1:,:]  # (L,D)
                    #print(z_t.shape, z_s.shape)
                    # mask
                    mt = masks[t_idx][b_idx]      # (L,)
                    ms = masks[s_idx][b_idx]      # (L,)

                    z_t = z_t.unsqueeze(0)   # => (1,L,D)
                    z_s = z_s.unsqueeze(0)
                    mt = mt.unsqueeze(0)     # => (1,L)
                    ms = ms.unsqueeze(0)

                    sim_mat = content_cosine_similarity(z_t, z_s, mt, ms)  # => (1,L,L)
                    sim_mat_0 = sim_mat[0].cpu().numpy()

                    # 4) 相似映射 sim_map (对角线)
                    diag_map = get_diagonal_map(sim_mat)  # => (1, side, side)
                    diag_map_0 = diag_map[0].cpu().numpy()
                    err_mix_dict[f"diag_map_M{s_idx+1}_to_M{t_idx+1}"] = diag_map_0.squeeze()
                    
                    # 5) 组合误差
                    up_diag_map_0 = cv2.resize(diag_map_0, (256, 256), interpolation=cv2.INTER_LINEAR)
                    err_mix = abs_err.squeeze() * (1-up_diag_map_0)
                    err_avg += err_mix
                    
                    # 保存当前组合误差图到字典，key 表示“源模态_到_目标模态”
                    err_mix_dict[f"M{s_idx+1}_to_M{t_idx+1}"] = err_mix
                    err_dict[f"M{s_idx+1}_to_M{t_idx+1}"] = abs_err.squeeze()


                    col += 1

            err_avg = err_avg / 12
            err_avg_value = np.sum(err_avg.cpu().numpy())
            
            
            # 异步保存NPZ文件
            executor.submit(save_npz_data, 
                          os.path.join(opt.img_save_path, f"{global_idx}_mix_err.npz"),
                          **err_mix_dict)
            executor.submit(save_npz_data,
                          os.path.join(opt.img_save_path, f"{global_idx}_err.npz"),
                          **err_dict)
            

            print(f"[Sample {global_idx}] 文件保存任务已提交")
            
# 等待所有任务完成
executor.shutdown(wait=True)