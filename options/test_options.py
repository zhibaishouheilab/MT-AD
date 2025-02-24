import argparse
import os

class Test_Options():
    def __init__(self):
        
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--batch_size", default=16, type=int)
        self.parser.add_argument("--img_size", default=256, type=int)
        self.parser.add_argument("--modality", default=['t1','t2','t1ce','flair'], help='modality used for training (all, t1, t2, etc.)') 
        self.parser.add_argument("--data_root", default='./data/BraTS2020/test/')
        self.parser.add_argument("--slice_num", default=60, type=int)
        self.parser.add_argument('--checkpoint_path', type=str, default='./weight/MM_Brain_16_adain_mask_nce_0206/20_MAE.pth',
                                 help='path to checkpoint file')
        self.parser.add_argument('--img_save_path', type=str, default=f'./snapshot/MM_Brain_16_adain_mask_nce_0206/BraTS2020_test_20/',
                                 help='path to save generated images')
        
        self.parser.add_argument("--depth", default=12,type=int)
        self.parser.add_argument("--num_workers", default=4, type=int)
        self.parser.add_argument("--data_rate", default=1, type=float)
        self.parser.add_argument("--mae_patch_size", default=16, type=int)
        self.parser.add_argument("--patch_size", default=16, type=int)
        self.parser.add_argument("--dim_encoder", default=128, type=int)
        self.parser.add_argument("--dim_decoder", default=64, type=int)
        self.parser.add_argument("--decoder_depth", default=8, type=int)
        self.parser.add_argument("--decoder_num_heads", default=8, type=int)
        self.parser.add_argument("--vit_dim", default=128, type=int)
        self.parser.add_argument("--window_size", default=8, type=int)
        self.parser.add_argument("--num_heads", default=16, type=int)
        self.parser.add_argument("--mlp_ratio", default=4, type=int)
        self.parser.add_argument("--random", default=False)
        self.parser.add_argument("--augment", default=False, type=bool, help='perform data augmentation')

    def get_opt(self):
        self.opt = self.parser.parse_args()
        return self.opt
        
