
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import numpy as np
from PIL import ImageEnhance, Image
import random
import os

def norm(img):
    img -= img.min(1, keepdim=True)[0]
    img /= img.max(1, keepdim=True)[0]
    return img

def cv_random_flip(img):
    # left right flip
    flip_flag = random.randint(0, 2)
    if flip_flag == 1:
        img = np.flip(img, 0).copy()
    if flip_flag == 2:
        img = np.flip(img, 1).copy()
    return img

def randomRotation(image):
    rotate_time = random.randint(0, 3)
    image = np.rot90(image, rotate_time).copy()
    return image

def colorEnhance(image):
    bright_intensity = random.randint(8, 12) / 10.0
    image = ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity = random.randint(8, 12) / 10.0
    image = ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity = random.randint(0, 20) / 10.0
    image = ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity = random.randint(0, 30) / 10.0
    image = ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image

def randomGaussian(img, mean=0.002, sigma=0.002):
    def gaussianNoisy(im, mean=mean, sigma=sigma):
        for _i in range(len(im)):
            im[_i] += random.gauss(mean, sigma)
        return im

    flag = random.randint(0, 3)
    if flag == 1:
        width, height = img.shape
        img = gaussianNoisy(img[:].flatten(), mean, sigma)
        img = img.reshape([width, height])
    return img

def randomPeper(img):
    flag = random.randint(0, 3)
    if flag == 1:
        noiseNum = int(0.0015 * img.shape[0] * img.shape[1])
        for i in range(noiseNum):
            randX = random.randint(0, img.shape[0] - 1)
            randY = random.randint(0, img.shape[1] - 1)
            if random.randint(0, 1) == 0:
                img[randX, randY] = 0
            else:
                img[randX, randY] = 1
    return img

from torchvision import transforms


class MultiModalDataset(data.Dataset):
    def __init__(self, 
                 img_size, 
                 image_root, 
                 modal, 
                 augment=False,
                 if_addlabel=False,
                 edge_root=None):
        """
        Args:
            img_size: 目标图像尺寸，resize 后的尺寸。
            image_root: 存放 .npy 文件的目录路径。
            modal: 模态名称列表 (例如 ['t1', 't2', 't1ce', 'flair'])。
            augment: 是否对数据做数据增强。
            if_addlabel: 如果为 True，则认为 npy 文件的最后一层为分割 label。
            edge_root: 如果不为 None，则从该目录中加载与 image_root 同名的 npy 文件，
                       每个 edge npy 文件中每一层存放与 image npy 文件对应层的 edge 图像，
                       image 和 edge 的最后一层均为 label。返回的数据顺序为：
                       [modal data, edge data, label] (if if_addlabel=True) 或 [modal data, edge data] (if if_addlabel=False)。
        """
        self.modal_list = modal
        self.image_root = image_root
        self.modal_indices = [i for i in range(len(self.modal_list))]  # 各模态对应的索引
        self.images = [os.path.join(self.image_root, f) 
                       for f in os.listdir(self.image_root) if f.endswith('.npy')]
        # sort 按文件名中数字排序
        self.images.sort(key=lambda x: int(os.path.basename(x).split(".npy")[0]))
        
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(img_size)
        ])
        
        # 对 label（以及 edge 图像）的 transform（通常采用最近邻插值）
        self.label_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(img_size, interpolation=transforms.InterpolationMode.NEAREST)
        ])

        self.Len = len(self.images)
        self.augment = augment
        self.if_addlabel = if_addlabel
        self.edge_root = edge_root
        print('Number of slices:', self.Len)

    def __getitem__(self, index):
        # 加载 image npy 文件
        npy_path = self.images[index]
        npy = np.load(npy_path)  # shape 可能为 (N, H, W)
        
        # 根据 if_addlabel 参数分离 modal data 和 label
        label_img = None
        if self.if_addlabel:
            label_img = npy[-1, :, :]  # 最后一层为 label
            modal_data = [npy[i, :, :] for i in self.modal_indices]
        else:
            modal_data = [npy[i, :, :] for i in self.modal_indices]
        
        # 如果 edge_root 非 None，则加载对应的 edge npy 文件
        edge_data = None
        if self.edge_root is not None:
            edge_npy_path = os.path.join(self.edge_root, os.path.basename(npy_path))
            edge_npy = np.load(edge_npy_path)
            # 这里 edge npy 文件中每一层对应 image npy 文件中的各模态，
            # 最后一层同样为 label，但此处只提取 modal 对应的 edge 数据。
            edge_data = [edge_npy[i, :, :] for i in self.modal_indices]

        # -------------- Data Augmentation --------------
        if self.augment:
            # 生成几何变换标志：翻转和旋转
            flip_flag = random.randint(0, 2)    # 0: 不翻转, 1: 垂直翻转, 2: 水平翻转
            rotate_flag = random.randint(0, 3)    # 0: 不旋转, 1: 90°, 2: 180°, 3: 270°
            # 对 modal_data 做翻转、旋转、colorEnhance（颜色增强）
            modal_data = self.apply_consistent_augmentation(modal_data, flip_flag, rotate_flag)
            # 对 label 做同样的几何变换，但不做颜色增强
            if self.if_addlabel and label_img is not None:
                label_img = self.apply_label_augmentation(label_img, flip_flag, rotate_flag)
            # 对 edge_data 每一通道做与 label 相同的几何变换
            if edge_data is not None:
                edge_data = [self.apply_label_augmentation(ed, flip_flag, rotate_flag) for ed in edge_data]

        # -------------- Transform to Tensor / Resize --------------
        # modal_data：若经过增强，此时为 PIL Image 列表；否则为 numpy 数组
        modal_data_tensors = []
        for modal in modal_data:
            modal_tensor = self.img_transform(modal)
            modal_data_tensors.append(modal_tensor)
        
        # 对 edge_data，每个通道使用与 label 相同的 transform
        edge_data_tensors = []
        if edge_data is not None:
            for ed in edge_data:
                ed_tensor = self.img_transform(ed)
                edge_data_tensors.append(ed_tensor)
        
        # 对 label
        if self.if_addlabel and label_img is not None:
            label_tensor = self.label_transform(label_img)
        
        # -------------- 返回数据 --------------
        # 如果 edge 数据存在，则返回 [modal, edge, label] 或 [modal, edge]
        if edge_data is not None:
            if self.if_addlabel and label_img is not None:
                return modal_data_tensors + edge_data_tensors + [label_tensor]
            else:
                return modal_data_tensors + edge_data_tensors
        else:
            # 若 edge_root 为 None，则与原来逻辑相同
            if self.if_addlabel and label_img is not None:
                return modal_data_tensors + [label_tensor]
            else:
                return modal_data_tensors

    def __len__(self):
        return self.Len

    def apply_consistent_augmentation(self, modal_data, flip_flag, rotate_flag):
        """
        对 modal 数据做翻转、旋转和颜色增强（colorEnhance）。
        """
        # 翻转
        if flip_flag == 1:  # 垂直翻转
            modal_data = [np.flip(modal, 0).copy() for modal in modal_data]
        elif flip_flag == 2:  # 水平翻转
            modal_data = [np.flip(modal, 1).copy() for modal in modal_data]

        # 旋转
        modal_data = [np.rot90(modal, rotate_flag).copy() for modal in modal_data]

        # 转换为 uint8
        modal_data = [np.clip(modal * 255, 0, 255).astype(np.uint8) for modal in modal_data]

        # 对每个 modal，先转为 PIL Image => 颜色增强 => 返回 PIL Image
        new_modals = []
        for modal in modal_data:
            pil_img = Image.fromarray(modal).convert('L')
            pil_img = colorEnhance(pil_img)  # 仅对 modal 做颜色增强
            new_modals.append(pil_img)
        return new_modals

    def apply_label_augmentation(self, label_img, flip_flag, rotate_flag):
        """
        对 label 或 edge 图像做与 modal 一致的翻转、旋转，
        不做颜色增强，保持原始数值信息。
        label_img: np.array, shape (H, W)
        """
        # 翻转
        if flip_flag == 1:  # 垂直翻转
            label_img = np.flip(label_img, 0).copy()
        elif flip_flag == 2:  # 水平翻转
            label_img = np.flip(label_img, 1).copy()

        # 旋转
        label_img = np.rot90(label_img, rotate_flag).copy()
        return label_img


def get_maeloader(batchsize, shuffle, 
                  modality=['t1', 't2', 't1ce', 'flair'], 
                  pin_memory=True,
                  img_size=256, 
                  img_root='../data/train/', 
                  num_workers=16, 
                  augment=False,
                  if_addlabel=False,
                  edge_root=None):
    """
    构建 DataLoader。
    若 if_addlabel=True，则 Dataset 会将 npy 的最后一层作为 label。
    若 edge_root 非 None，则会同时加载 edge 数据，返回的顺序为：
        [modal data, edge data, label] 或 [modal data, edge data]
    """
    dataset = MultiModalDataset(
        img_size=img_size, 
        image_root=img_root,
        modal=modality, 
        augment=augment,
        if_addlabel=if_addlabel,
        edge_root=edge_root
    )
    data_loader = data.DataLoader(
        dataset=dataset, 
        batch_size=batchsize, 
        shuffle=shuffle,
        pin_memory=pin_memory, 
        num_workers=num_workers
    )
    return data_loader
