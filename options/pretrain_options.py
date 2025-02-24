import argparse
import os
TASK = 'MM_Brain_16_adain_mask_nce0.01_0208'

class Pretrain_Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        # Training hyperparameters
        self.parser.add_argument('--lr', type=float, default=3e-3, help='learning rate')
        self.parser.add_argument("--batch_size", default=12, type=int, help='batch size')
        self.parser.add_argument("--epoch", default=50, type=int, help='number of epochs')
        self.parser.add_argument("--decay_epoch", default=10, type=float, help='epochs to decay learning rate')
        self.parser.add_argument("--decay_rate", default=0.1, type=float, help='decay rate for learning rate')

        # Model architecture
        self.parser.add_argument("--patch_size", default=16, type=int, help='patch size for embedding')
        self.parser.add_argument("--img_size", default=256, type=int, help='input image size')
        self.parser.add_argument("--depth", default=12, type=int, help='depth of encoder')
        self.parser.add_argument("--decoder_depth", default=8, type=int, help='depth of decoder')
        self.parser.add_argument("--num_heads", default=16, type=int, help='number of attention heads in encoder')
        self.parser.add_argument("--decoder_num_heads", default=8, type=int, help='number of attention heads in decoder')
        self.parser.add_argument("--mlp_ratio", default=4, type=int, help='MLP ratio in transformer blocks')
        self.parser.add_argument("--dim_encoder", default=128, type=int, help='dimension of encoder embeddings')
        self.parser.add_argument("--dim_decoder", default=64, type=int, help='dimension of decoder embeddings')

        # Data augmentation
        self.parser.add_argument("--augment", default=True, type=bool, help='perform data augmentation')
        self.parser.add_argument("--rotation_range", default=3, type=int, help='range of rotations for augmentation (0-3)')
        self.parser.add_argument("--flip_prob", default=0.5, type=float, help='probability of applying random flip')
        
        # Data options
        self.parser.add_argument("--modality", default=['t1','t2','t1ce','flair'], help='modality used for training (all, t1, t2, etc.)') 
        self.parser.add_argument("--data_root", default='./data/BraTS2020/train/', help='path to training data')
        self.parser.add_argument("--num_workers", default=4, type=int, help='number of workers for data loader')

        # Loss weights
        self.parser.add_argument("--nce_weight", default=0.01, type=float, help='weight for nce_loss')
        self.parser.add_argument("--consistent_weight", default=0.0, type=float, help='weight for consistent_loss')
        self.parser.add_argument("--rec_weight", default=1.0, type=float, help='weight for reconstruction loss')

        # Logging and checkpoints
        self.parser.add_argument('--use_checkpoints', default=False, type=bool, help='resume training from checkpoints')
        self.parser.add_argument('--checkpoint_path', type=str, default=None, help='path to checkpoint file')
        self.parser.add_argument('--img_save_path', type=str, default=f'./snapshot/{TASK}/', help='path to save generated images')
        self.parser.add_argument('--weight_save_path', type=str, default=f'./weight/{TASK}/', help='path to save model weights')
        self.parser.add_argument("--save_output", default=200, type=int, help='save output every N batches')
        self.parser.add_argument("--save_weight", default=10, type=int, help='save weights every N epochs')
        self.parser.add_argument("--log_path", default=f'./log/{TASK}.log', help='path to log file')
        self.parser.add_argument("--logs_path", default=f'./log/{TASK}', help='folder to save summary writer logs')

    def get_opt(self):
        self.opt = self.parser.parse_args()
        return self.opt
