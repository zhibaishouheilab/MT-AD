import torch
import logging
from utils.MM_loader import *
from MultiTransAD.model.MMPC import *
from utils.mae_visualize import *
from options import Pretrain_Options
import os
from tensorboardX import SummaryWriter
from torchvision.utils import save_image 

Task = 'BraTS2021'

def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = init_lr * decay


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

opt = Pretrain_Options().get_opt()
opt.img_save_path = f'./snapshot/{Task}/'
opt.weight_save_path = f'./weight/{Task}/'
opt.log_path = f'./log/{Task}.log'
opt.logs_path = f'./log/{Task}'
opt.data_root = './data/BraTS2021/train'
opt.edge_root = './data/BraTS2021/train_edge'  
opt.patch_size = 16
opt.batch_size = 12


mae = MultiModalPatchMAE(img_size=opt.img_size, patch_size=opt.patch_size, embed_dim=opt.dim_encoder, depth=opt.depth, num_heads=opt.num_heads, in_chans=1,
        decoder_embed_dim=opt.dim_decoder, decoder_depth=opt.decoder_depth, decoder_num_heads=opt.decoder_num_heads,
        mlp_ratio=opt.mlp_ratio, norm_layer=nn.LayerNorm)

os.makedirs(opt.img_save_path, exist_ok=True)
os.makedirs(opt.weight_save_path, exist_ok=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train_loader = get_maeloader(batchsize=opt.batch_size, shuffle=True, pin_memory=True, img_size=opt.img_size,
            img_root=opt.data_root, num_workers=opt.num_workers, augment=opt.augment, edge_root=opt.edge_root)

optimizer = torch.optim.Adam(mae.parameters(), lr=opt.lr, betas=(0.9, 0.95))
mae = mae.to(device)

if opt.use_checkpoints:
    print('load checkpoint......', opt.checkpoint_path)
    mae.load_state_dict(torch.load(opt.checkpoint_path, map_location=torch.device(device)), strict=False)

logging.basicConfig(filename=opt.log_path,
                    format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                    level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')

writer = SummaryWriter(log_dir=opt.logs_path)

for epoch in range(1, opt.epoch):
    for i, (M1, M2, M3, M4, M1_edge, M2_edge, M3_edge, M4_edge) in enumerate(train_loader):

        adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        
        optimizer.zero_grad()

        M1, M2, M3, M4, M1_edge, M2_edge, M3_edge, M4_edge = M1.to(device, dtype=torch.float), M2.to(device, dtype=torch.float), \
            M3.to(device, dtype=torch.float), M4.to(device, dtype=torch.float), M1_edge.to(device, dtype=torch.float), \
            M2_edge.to(device, dtype=torch.float), M3_edge.to(device, dtype=torch.float), M4_edge.to(device, dtype=torch.float)

        # Forward pass and calculate losses
        outputs = mae(M1, M2, M3, M4, M1_edge, M2_edge, M3_edge, M4_edge)
        #mask_loss = outputs['mask_loss']
        consistent_loss = outputs['consistent_loss']
        rec_loss_M1 = outputs['rec_loss_M1']
        rec_loss_M2 = outputs['rec_loss_M2']
        rec_loss_M3 = outputs['rec_loss_M3']
        rec_loss_M4 = outputs['rec_loss_M4']
        rec_loss = outputs['rec_loss']
        nce_loss = outputs['nce_loss']
        M1_gen = outputs['M1_gen']
        M2_gen = outputs['M2_gen']
        M3_gen = outputs['M3_gen']
        M4_gen = outputs['M4_gen']
        
        consistent_loss = consistent_loss * opt.consistent_weight
        nce_loss = nce_loss * opt.nce_weight
        rec_loss_M1 = rec_loss_M1 * opt.rec_weight
        rec_loss_M2 = rec_loss_M2 * opt.rec_weight
        rec_loss_M3 = rec_loss_M3 * opt.rec_weight
        rec_loss_M4 = rec_loss_M4 * opt.rec_weight     
        #print(consistent_loss, nce_loss, rec_loss_M1, rec_loss_M2, rec_loss_M3, rec_loss_M4)   

        total_loss = consistent_loss + rec_loss_M1 + rec_loss_M2 + rec_loss_M3 + rec_loss_M4 + nce_loss
        writer.add_scalar('Loss/total_loss', total_loss.item(), epoch * len(train_loader) + i)
        writer.add_scalar('Loss/nce_loss', nce_loss.item(), epoch * len(train_loader) + i)
        writer.add_scalar('Loss/consistent_loss', consistent_loss.item(), epoch * len(train_loader) + i)
        writer.add_scalar('Loss/rec_loss_M1', rec_loss_M1.item(), epoch * len(train_loader) + i)
        writer.add_scalar('Loss/rec_loss_M2', rec_loss_M2.item(), epoch * len(train_loader) + i)
        writer.add_scalar('Loss/rec_loss_M3', rec_loss_M3.item(), epoch * len(train_loader) + i)
        writer.add_scalar('Loss/rec_loss_M4', rec_loss_M4.item(), epoch * len(train_loader) + i)
        

        total_loss.backward()
        optimizer.step()

        print(
            f"[Epoch {epoch}/{opt.epoch}] [Batch {i}/{len(train_loader)}] [total_loss: {total_loss.item():.4f}] "
            f"[consistent_loss: {consistent_loss.item():.4f}] [nce loss: {nce_loss.item():4f}] "
            f"[rec_loss_M1: {rec_loss_M1.item():.4f}] [rec_loss_M2: {rec_loss_M2.item():.4f}] "
            f"[rec_loss_M3: {rec_loss_M3.item():.4f}] [rec_loss_M4: {rec_loss_M4.item():.4f}] "
            f"[lr: {get_lr(optimizer):.6f}]"
        )

        # Save generated images for visualization
        if i % opt.save_output == 0:
            save_image(
                [M1[0], M2[0], M3[0], M4[0], M1_gen[0], M2_gen[0], M3_gen[0], M4_gen[0]],
                opt.img_save_path + f"{epoch}_{i}.png",
                nrow=4,
                normalize=True,
            )
            logging.info(
                f"[Epoch {epoch}/{opt.epoch}] [Batch {i}/{len(train_loader)}] [total_loss: {total_loss.item():.4f}] "
                f"[consistent_loss: {consistent_loss.item():.4f}] [nce loss: {nce_loss.item():4f}] "
                f"[rec_loss_M1: {rec_loss_M1.item():.4f}] [rec_loss_M2: {rec_loss_M2.item():.4f}] "
                f"[rec_loss_M3: {rec_loss_M3.item():.4f}] [rec_loss_M4: {rec_loss_M4.item():.4f}] "
                f"[lr: {get_lr(optimizer):.6f}]"
            )

    # Save model checkpoints
    if epoch % opt.save_weight == 0:
        torch.save(mae.state_dict(), opt.weight_save_path + f"{epoch}_MAE.pth")

# Final save
torch.save(mae.state_dict(), opt.weight_save_path + '/MAE_final.pth')
