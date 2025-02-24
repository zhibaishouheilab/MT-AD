#使用同一个patch编码、同一个风格编码器和同一个生成器
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from timm.models.vision_transformer import PatchEmbed, Block
from utils.pos_embed import get_2d_sincos_pos_embed
import random
from itertools import combinations

class MultiModalPatchMAE(nn.Module):
    def __init__(self, img_size=256, patch_size=16, in_chans=1, embed_dim=128, depth=12, num_heads=8,
                 decoder_embed_dim=64, decoder_depth=6, decoder_num_heads=8, mlp_ratio=4.,
                 norm_layer=nn.LayerNorm, style_dim=2):
        super().__init__()
        
        # 用于融合 patch token 与 edge latent 的线性层
        self.edge_fuse = nn.Linear(2 * decoder_embed_dim, decoder_embed_dim)
        
        # Patch embedding for 4模态（原始图像与边缘图均使用同一模块）
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        
        self.patch_embed_edge = PatchEmbed(img_size, patch_size, in_chans, embed_dim)

        num_patches = self.patch_embed.num_patches
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches+1, embed_dim), requires_grad=False)
        
        # Style encoders：对每个模态各自独立
        style_encoder = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(depth)])

        self.style_encoder = nn.Sequential(*style_encoder, nn.Linear(embed_dim, 2*decoder_embed_dim))

        # 内容编码器：所有模态统一使用同一内容编码器
        self.content_encoder = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(depth)])
        

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)
        
        self.sig = nn.Sigmoid()
        self.norm = norm_layer(embed_dim)
        
        self.generator = self.build_generator(decoder_embed_dim, decoder_embed_dim, decoder_depth, decoder_num_heads, norm_layer)

        
        # Initialize weights
        self.initialize_weights()

    def initialize_weights(self):
        # 使用 sin-cos 方式初始化位置编码
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**0.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**0.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        for w_ in [self.patch_embed.proj.weight.data,
                   self.patch_embed_edge.proj.weight.data]:
            torch.nn.init.xavier_uniform_(w_.view([w_.shape[0], -1]))

        torch.nn.init.normal_(self.cls_token, std=.02)
        
        torch.nn.init.normal_(self.mask_token, std=.02)
        
        # 初始化 Linear 和 LayerNorm 层
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def build_generator(self, input_dim, decoder_embed_dim, decoder_depth, decoder_num_heads, norm_layer):
        input_projection = nn.Linear(input_dim, decoder_embed_dim)
        generator_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio=4., qkv_bias=True, norm_layer=norm_layer)
            for _ in range(decoder_depth)
        ])
        generator_pred = nn.Linear(decoder_embed_dim, self.patch_size**2, bias=True)

        return nn.Sequential(input_projection, *generator_blocks, 
                             norm_layer(decoder_embed_dim), generator_pred, self.sig)

    def adain(self, content, style_mean, style_std, eps=1e-6):
        style_mean = style_mean.unsqueeze(1)  # (N, 1, D)
        style_std = style_std.unsqueeze(1)
        content_mean = content.mean(dim=-1, keepdim=True)  # (N, L, 1)
        content_std = content.std(dim=-1, keepdim=True) + eps
        return style_std * (content - content_mean) / content_std + style_mean

    def patchify(self, imgs):
        """
        imgs: (N, 1, H, W)
        返回: (N, L, patch_size**2)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 1, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2)
        返回: (N, 1, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**0.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 1))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 1, h * p, h * p))
        return imgs

    def compute_style_mean_std(self, embed_M, mod_id=1):

        style_features = self.style_encoder(embed_M)
        
        style_mean, style_std = torch.chunk(style_features, 2, dim=-1)
        style_mean = style_mean.mean(dim=1)
        style_std = style_std.mean(dim=1)
        return style_mean, style_std

    def compute_nce_loss(self, z_M1_s, z_M2_s, mask_i, mask_j):
        """
        计算 NCE 损失，原理与之前保持一致
        """
        B, N, C = z_M1_s.shape
        sim_matrix = F.cosine_similarity(z_M1_s.unsqueeze(2), z_M2_s.unsqueeze(1), dim=-1)

        # 构造前景掩码
        f_mask_i = mask_i.unsqueeze(2)   # (B, N, 1)
        f_mask_j = mask_j.unsqueeze(1)   # (B, 1, N)
        valid_mask_2d = f_mask_i * f_mask_j  # (B, N, N)

        # positives：对角线部分
        positives = torch.diagonal(sim_matrix, dim1=1, dim2=2)
        diag_mask = mask_i & mask_j    
        positives = positives * diag_mask  # (B, N)
        # negatives：非对角线部分
        mask = ~torch.eye(N, dtype=torch.bool, device=z_M1_s.device).unsqueeze(0).expand(B, N, N)
        negatives = sim_matrix[mask].view(B, N, -1)
        negatives = negatives * valid_mask_2d[mask].view(B, N, -1)

        temperature = 0.1
        positives_exp = torch.exp(positives / temperature)
        negatives_exp = torch.exp(negatives / temperature).sum(dim=-1)
        eps = 1e-8
        nce_loss = -torch.log((positives_exp+eps) / (positives_exp + negatives_exp + eps))
        nce_loss = nce_loss.sum() / diag_mask.sum()  # 前景区域平均
        return nce_loss
    
    def fuse_and_generate(self, candidate_modalities, style_mean, style_std, generator, target_edge):
        """
        生成目标模态的预测：
          - 从候选内容 latent 中随机选取一份
          - 经过 decoder_embed 得到初步 latent 表示 fused_c
          - 利用目标模态对应的边缘图 target_edge，通过对应的 patch embedding 和 decoder_embed 得到 edge_latent，
            并将其加到 fused_c 上，帮助补充细节
          - 加入位置编码后，采用 AdaIN 融合风格（style_mean, style_std）
          - 经过 generator 得到 patch 重构，去掉 CLS patch 后返回
        """
        selected = random.choice(candidate_modalities)
        fused_c = self.decoder_embed(selected)
        
        # 分离 CLS token 和 patch tokens
        cls_token = fused_c[:, :1, :]         # (B, 1, decoder_embed_dim)
        patch_tokens = fused_c[:, 1:, :]        # (B, N, decoder_embed_dim)
        
        # 利用目标模态的边缘图获得边缘 latent（注意：不同模态使用对应的 patch embedding）
        edge_latent = self.decoder_embed(self.patch_embed_edge(target_edge))
        # 注意：edge_latent 尺寸为 (B, N, decoder_embed_dim)

        # 融合 patch tokens 与 edge_latent：先拼接，再通过 self.edge_fuse 降维融合
        concat_tokens = torch.cat([patch_tokens, edge_latent], dim=-1)  # (B, N, 2*decoder_embed_dim)
        fused_patch = self.edge_fuse(concat_tokens)                     # (B, N, decoder_embed_dim)

        # 组合 CLS token 与融合后的 patch tokens
        fused_c = torch.cat([cls_token, fused_patch], dim=1)              # (B, N+1, decoder_embed_dim)

        # 加入位置编码
        fused_c = fused_c + self.decoder_pos_embed[:, :fused_c.size(1), :]
        # 自适应实例归一化
        fused = self.adain(fused_c, style_mean, style_std)
        # 生成 patch 重构，去掉 CLS token 后返回
        p_gen = generator(fused)
        p_gen = p_gen[:, 1:, :]
        return p_gen

    def forward(self, M1, M2, M3, M4, M1_edge, M2_edge, M3_edge, M4_edge):
        # 1) 对原始图像做 patchify，用于重构损失计算
        p_M1 = self.patchify(M1)  # (B, L, patch_size**2)
        p_M2 = self.patchify(M2)
        p_M3 = self.patchify(M3)
        p_M4 = self.patchify(M4)

        # foreground mask（基于原始图像 patch 计算）
        valid_mask_M1 = (p_M1.var(dim=-1) > 0)  # (B, L)
        valid_mask_M2 = (p_M2.var(dim=-1) > 0)
        valid_mask_M3 = (p_M3.var(dim=-1) > 0)
        valid_mask_M4 = (p_M4.var(dim=-1) > 0)
        valid_mask_arr = [valid_mask_M1, valid_mask_M2, valid_mask_M3, valid_mask_M4]

        # 2) 分别做 patch_embed + CLS + pos
        embed_M1 = self.patch_embed(M1)
        embed_M2 = self.patch_embed(M2)
        embed_M3 = self.patch_embed(M3)
        embed_M4 = self.patch_embed(M4)
        
        # 加入 CLS token
        embed_M1 = torch.cat([self.cls_token.expand(embed_M1.shape[0], -1, -1), embed_M1], dim=1)
        embed_M2 = torch.cat([self.cls_token.expand(embed_M2.shape[0], -1, -1), embed_M2], dim=1)
        embed_M3 = torch.cat([self.cls_token.expand(embed_M3.shape[0], -1, -1), embed_M3], dim=1)
        embed_M4 = torch.cat([self.cls_token.expand(embed_M4.shape[0], -1, -1), embed_M4], dim=1)

        # 加入位置编码
        embed_M1 = embed_M1 + self.pos_embed
        embed_M2 = embed_M2 + self.pos_embed
        embed_M3 = embed_M3 + self.pos_embed
        embed_M4 = embed_M4 + self.pos_embed
        
        # 5) 计算各模态的风格信息（均值与标准差）
        style_mean_M1, style_std_M1 = self.compute_style_mean_std(embed_M1, mod_id=1)
        style_mean_M2, style_std_M2 = self.compute_style_mean_std(embed_M2, mod_id=2)
        style_mean_M3, style_std_M3 = self.compute_style_mean_std(embed_M3, mod_id=3)
        style_mean_M4, style_std_M4 = self.compute_style_mean_std(embed_M4, mod_id=4)

        # 3) 内容编码：所有模态统一使用同一内容编码器
        z_M1_c, z_M2_c, z_M3_c, z_M4_c = embed_M1, embed_M2, embed_M3, embed_M4
        for blk in self.content_encoder:
            z_M1_c = blk(z_M1_c)
            z_M2_c = blk(z_M2_c)
            z_M3_c = blk(z_M3_c)
            z_M4_c = blk(z_M4_c)

        modalities = [z_M1_c, z_M2_c, z_M3_c, z_M4_c]
        num_modalities = len(modalities)

        # 4) 计算一致性损失（L1）和 NCE 损失
        l1_losses = []
        nce_losses = []
        for i in range(num_modalities):
            for j in range(num_modalities):
                if i == j:
                    continue
                z_i = modalities[i][:, 1:, :]  
                z_j = modalities[j][:, 1:, :]
                l1_ij = F.l1_loss(z_i, z_j)
                l1_losses.append(l1_ij)
                nce_ij = self.compute_nce_loss(z_i, z_j, valid_mask_arr[i], valid_mask_arr[j])
                nce_losses.append(nce_ij)

        consistent_loss = sum(l1_losses) / len(l1_losses)
        nce_loss = sum(nce_losses) / len(nce_losses)

        # 6) 针对每个目标模态生成对应的 patch 重构，其中在 fuse_and_generate 中加入了目标边缘图信息
        p_M1_gen = self.fuse_and_generate(modalities, style_mean_M1, style_std_M1, self.generator, M1_edge)
        p_M2_gen = self.fuse_and_generate(modalities, style_mean_M2, style_std_M2, self.generator, M2_edge)
        p_M3_gen = self.fuse_and_generate(modalities, style_mean_M3, style_std_M3, self.generator, M3_edge)
        p_M4_gen = self.fuse_and_generate(modalities, style_mean_M4, style_std_M4, self.generator, M4_edge)
        
        # 7) 重构损失（仅计算原始图像 patch 与生成 patch 的 L1 损失）
        rec_loss_M1 = F.l1_loss(p_M1_gen, p_M1)
        rec_loss_M2 = F.l1_loss(p_M2_gen, p_M2)
        rec_loss_M3 = F.l1_loss(p_M3_gen, p_M3)
        rec_loss_M4 = F.l1_loss(p_M4_gen, p_M4)
        rec_loss = rec_loss_M1 + rec_loss_M2 + rec_loss_M3 + rec_loss_M4

        # 8) 解码 patch 重构结果，恢复为完整图像
        M1_gen = self.unpatchify(p_M1_gen)
        M2_gen = self.unpatchify(p_M2_gen)
        M3_gen = self.unpatchify(p_M3_gen)
        M4_gen = self.unpatchify(p_M4_gen)

        return {
            'consistent_loss': consistent_loss,
            'nce_loss': nce_loss,
            'rec_loss_M1': rec_loss_M1,
            'rec_loss_M2': rec_loss_M2,
            'rec_loss_M3': rec_loss_M3,
            'rec_loss_M4': rec_loss_M4,
            'rec_loss': rec_loss,
            'M1_gen': M1_gen,
            'M2_gen': M2_gen,
            'M3_gen': M3_gen,
            'M4_gen': M4_gen
        }
        