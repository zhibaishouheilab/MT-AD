import numpy as np
from matplotlib import pylab as plt
import nibabel as nib
import random
import glob
import os
from PIL import Image
import imageio
import scipy.ndimage

def normalize(image, mask=None, percentile_lower=0.2, percentile_upper=99.8):

    if mask is None:
        mask = image != image[0, 0, 0]
    cut_off_lower = np.percentile(image[mask != 0].ravel(), percentile_lower)
    cut_off_upper = np.percentile(image[mask != 0].ravel(), percentile_upper)
    res = np.copy(image)
    res[(res < cut_off_lower) & (mask != 0)] = cut_off_lower
    res[(res > cut_off_upper) & (mask != 0)] = cut_off_upper
    res = res / res.max()  # 0-1

    return res

def resample_to_spacing(image, new_spacing=(1, 1, 1), order=3):
    original_spacing = image.header.get_zooms()[:3]
    zoom_factors = np.array(original_spacing) / np.array(new_spacing)
    new_image_data = scipy.ndimage.zoom(image.get_fdata(), zoom_factors, order=order)
    new_image = nib.Nifti1Image(new_image_data, affine=image.affine)
    return new_image

def center_crop_or_pad(img, target_size):
    h, w = img.shape[:2]
    target_h, target_w = target_size
    
    if h > target_h:
        start_h = h // 2 - target_h // 2
        img = img[start_h:start_h + target_h, :]
    if w > target_w:
        start_w = w // 2 - target_w // 2
        img = img[:, start_w:start_w + target_w]
    
    if h < target_h:
        pad_h = (target_h - h) // 2
        img = np.pad(img, ((pad_h, target_h - h - pad_h), (0, 0), (0, 0)), mode='constant')
    if w < target_w:
        pad_w = (target_w - w) // 2
        img = np.pad(img, ((0, 0), (pad_w, target_w - w - pad_w), (0, 0)), mode='constant')
    
    return img

def preprocess(image_path, spacing=(1,1), crop_size=(240,240), target_size=(256,256), rotate=None, order=3):
    image = nib.load(image_path)
    if spacing:
        image = resample_to_spacing(image, new_spacing=(spacing[0],spacing[1],image.header.get_zooms()[2]),order=order)
    image_data = image.get_fdata()
    if rotate:
        image_data = np.rot90(image_data,rotate)
    if crop_size:
        image_data = center_crop_or_pad(image_data, crop_size)
    zoom_array = [target_size[0]/image_data.shape[0], target_size[1]/image_data.shape[1], 1]
    image_data = scipy.ndimage.zoom(image_data, zoom_array, order=order)
    
    image_data = normalize(image_data)
    return image_data

    

if __name__ == '__main__':
    
    base_folder = '/home/ubuntu/Project/datasets/MSLUB/lesion/'
    train_path = '../data/MSLUB_public/train/'
    test_path = '../data/MSLUB_public/test/'

    # 获取每个患者的文件夹路径
    patient_folders = sorted(glob.glob(os.path.join(base_folder,  'patient*')))
    
    data_len = len(patient_folders) 
    train_len = int(data_len * 0.8)
    test_len = data_len - train_len

    os.makedirs(train_path,exist_ok=True)
    os.makedirs(test_path,exist_ok=True)
    
    count_number = 0
    
    for i, patient_folder in enumerate(patient_folders):
        print(f'Processing {i + 1}th patient:', patient_folder)
        t1_path = os.path.join(patient_folder, 'T1W_stripped_registered.nii.gz')
        t2_path = os.path.join(patient_folder, 'T2W_stripped_registered.nii.gz')
        flair_path = os.path.join(patient_folder, 'FLAIR_stripped_registered.nii.gz')
        gt_path = os.path.join(patient_folder, 'anomaly_segmentation.nii.gz')
        
        t1_img = nib.load(t1_path) 
        t2_img = nib.load(t2_path)
        flair_img = nib.load(flair_path)
        gt_img = nib.load(gt_path)
        
        t1_data = t1_img.get_fdata()
        t2_data = t2_img.get_fdata()
        flair_data = flair_img.get_fdata()
        gt_data = gt_img.get_fdata()
        gt_data = gt_data.astype(np.uint8)
        
        t1_data = normalize(t1_data)
        t2_data = normalize(t2_data)
        flair_data = normalize(flair_data)
        
        
        tensor = np.stack([t1_data, t2_data, flair_data, gt_data])  
        
        if i < train_len:
            for j in range(60):
                Tensor = tensor[:, 10:210, 25:225, 50 + j]
                np.save(train_path + str(60 * i + j + 1) + '.npy', Tensor)
        else:
            for j in range(60):
                Tensor = tensor[:, 10:210, 25:225, 50 + j]
                np.save(test_path + str(60 * (i - train_len) + j + 1) + '.npy', Tensor)
