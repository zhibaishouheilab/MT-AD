import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve
from torchvision.utils import make_grid, save_image
import torch.nn.functional as F
import re
import scipy.ndimage as ndimage
import cv2
from multiprocessing import Pool, cpu_count
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

#############################################
# 评价指标函数
#############################################
def compute_auroc(predictions, targets):
    """计算全图像像素级的 AUROC"""
    preds_flat = predictions.view(-1).detach().cpu().numpy()
    targets_flat = targets.view(-1).detach().cpu().numpy()
    if targets_flat.sum() == 0:
        print("Warning: No positive pixels in targets; returning AUROC=0.5")
        return 0.5
    return roc_auc_score(targets_flat, preds_flat)

def compute_aupr(predictions, targets):
    """计算全图像像素级的 AUPRC"""
    preds_flat = predictions.view(-1).detach().cpu().numpy()
    targets_flat = targets.view(-1).detach().cpu().numpy()
    precision, recall, _ = precision_recall_curve(targets_flat, preds_flat)
    return auc(recall, precision)

def compute_dice(pred_bin, targets):
    """计算 Dice 分数，输入必须为二值图"""
    pred_flat = pred_bin.view(-1).float()
    targ_flat = targets.view(-1).float()
    intersection = (pred_flat * targ_flat).sum()
    dice = (2 * intersection) / (pred_flat.sum() + targ_flat.sum() + 1e-8)
    return dice.item()

def apply_connected_component_analysis(binary_map,connected_component_size=10):
    """
    使用OpenCV加速连通组件分析
    """
    binary_map = binary_map.squeeze().astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_map, connectivity=8)
    if num_labels <= 1:
        return torch.zeros_like(torch.tensor(binary_map)).unsqueeze(0)
    
    # 向量化处理有效区域
    areas = stats[1:, cv2.CC_STAT_AREA]
    valid_labels = np.where(areas >= connected_component_size)[0] + 1
    processed_map = np.isin(labels, valid_labels).astype(np.uint8)
    return torch.tensor(processed_map).unsqueeze(0)

def compute_best_dice(predictions, targets, n_thresh=100,connected_component_size=10):
    """
    在 [min, max] 范围内等间隔采样 n_thresh 个阈值，计算二值化后的 Dice 分数，
    返回最佳 Dice 分数及对应阈值。
    """
    tmin = predictions.min().item()
    tmax = predictions.max().item()
    thresholds = np.linspace(tmin, tmax, n_thresh)
    best_dice = 0.0
    best_thresh = thresholds[0]
    for t in tqdm(thresholds, desc="Searching best threshold"):
        pred_bin = (predictions > t).float()
        processed_bin = torch.stack([apply_connected_component_analysis(pred_bin[i].cpu().numpy(),connected_component_size) for i in range(pred_bin.shape[0])]).to(pred_bin.device)
        dice = compute_dice(processed_bin, targets)
        if dice > best_dice:
            best_dice = dice
            best_thresh = t
    return best_dice, best_thresh

#############################################
# 可视化函数
#############################################
def plot_results(predictions, bin_maps, targets, method_name, n_images=5, save_path=None):
    """
    可视化前 n_images 个样本，分为三列显示：
      - 最左列：原始异常分数图
      - 中间列：二值化预测图
      - 最右列：原始标签图
    所有图像均采用 jet 色图显示。
    如果 save_path 不为 None，则将图像保存到指定路径。
    """
    # 创建 n_images 行, 3 列的子图
    fig, axs = plt.subplots(n_images, 3, figsize=(15, 5 * n_images))
    
    # 如果只有一个样本，确保 axs 以列表方式处理
    if n_images == 1:
        axs = [axs]
    
    for i in range(n_images):
        # 获取单个样本（去掉 channel 维度）
        pred_img = predictions[i].squeeze(0).detach().cpu().numpy()
        bin_img  = bin_maps[i].squeeze(0).detach().cpu().numpy()
        gt_img   = targets[i].squeeze(0).detach().cpu().numpy()
        
        # 左列：原始异常分数图
        axs[i][0].imshow(pred_img, cmap='jet')
        axs[i][0].axis("off")
        axs[i][0].set_title("Anomaly Score")
        
        # 中列：二值化预测图
        axs[i][1].imshow(bin_img, cmap='jet')
        axs[i][1].axis("off")
        axs[i][1].set_title("Binary Prediction")
        
        # 右列：原始标签图
        axs[i][2].imshow(gt_img, cmap='jet')
        axs[i][2].axis("off")
        axs[i][2].set_title("Ground Truth")
    
    fig.suptitle(f"Method: {method_name}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    plt.show()

#############################################
# 数据加载函数（单文件夹，多方法支持）
#############################################
import os
import numpy as np
import torch
import torch.nn.functional as F

def load_dataset_from_npz(data_dir, pattern,fixed_size = (64, 64), load_npz_number=None, load_npz_start=0, label_threshold=None):
    """
    加载存放在单一文件夹中的 npz 文件，每个 npz 文件中包含：
      - key 'Label'：对应标签（二维数组，非零区域为异常，转换为 int8）
      - 其他 key：对应不同方法的异常分数图（二维数组）
    对读入的数据在二维层面下采样到固定大小。
    返回:
      - ann_dict: dict, 键为文件名（不含扩展名），值为 torch.tensor，形状 [1, H, W]（标签）
      - scores_dict: dict, 键为方法名，每个值为 dict，键为文件名，值为 torch.tensor，形状 [1, H, W]（异常分数图）
    """
    files = sorted([f for f in os.listdir(data_dir) if pattern.match(f)])#[:2000]
    if load_npz_number is not None:
        files = files[load_npz_start:load_npz_start+load_npz_number]
    ann_dict = {}
    scores_dict = {}  # { method_name: { base: score_tensor } }
    
    for fname in files:
        base = os.path.splitext(fname)[0]
        file_path = os.path.join(data_dir, fname)
        data = np.load(file_path)
        # 检查标签
        if 'Label' not in data.files:
            print(f"Warning: 文件 {fname} 中未找到 'Label' 键，跳过。")
            continue
        label = data['Label']
        label = (label != 0).astype(np.int8)
        if label_threshold is not None:
            if np.sum(label) < label_threshold:
                continue
        #if np.sum(label) < 100:
        #    continue
        
        # 将 label 转为 tensor 并下采样
        # 原始 shape 为 [H, W]，先扩展为 [1, H, W]（channel 维度）
        label_tensor = torch.tensor(label).unsqueeze(0)
        # 为 F.interpolate 添加 batch 维度，变为 [1, 1, H, W]
        label_tensor = label_tensor.unsqueeze(0)
        label_tensor = F.interpolate(label_tensor.float(), size=fixed_size, mode="nearest")
        # 恢复到 [1, H, W]
        label_tensor = label_tensor.squeeze(0).long()
        ann_dict[base] = label_tensor

        # 处理其他 key（异常分数图）
        for key in data.files:
            if key == 'Label':
                continue
            score = data[key]
            # 转为 tensor，初始 shape 假定为 [H, W]，扩展为 [1, H, W]
            score_tensor = torch.tensor(score).unsqueeze(0)
            # 增加 batch 维度：[1, 1, H, W]
            score_tensor = score_tensor.unsqueeze(0)
            # 对异常分数图使用 bilinear 插值进行下采样（可根据实际情况选择模式）
            score_tensor = F.interpolate(score_tensor.float(), size=fixed_size, mode="bilinear", align_corners=False)
            # 恢复为 [1, H, W]
            score_tensor = score_tensor.squeeze(0)
            if key not in scores_dict:
                scores_dict[key] = {}
            scores_dict[key][base] = score_tensor

    return ann_dict, scores_dict


#############################################
# 主程序
#############################################
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate 2D anomaly detection for multiple methods from npz files (Label and scores in one file)")
    parser.add_argument("--data_dir", type=str, 
                        default="./outputs/BraTS2021",
                        help="存放 npz 文件的文件夹路径，每个文件中包含 'Label' 和其他方法的异常分数图")
    parser.add_argument("--task", type=str,default="BraTS2021", help="任务名称")
    parser.add_argument("--n_images", type=int, default=5,
                        help="可视化的样本数量")
    parser.add_argument("--save_dir", type=str, default="./results",
                        help="保存可视化图像的文件夹路径（可选）")
    parser.add_argument("--fixed_size", type=int, nargs=2, default=(64,64),
                        help="下采样到固定大小的目标尺寸")
    parser.add_argument("--pattern", type=str, default=r'^\d+_mix_err\.npz$',
                        help="文件名匹配模式")
    parser.add_argument("--median_filter_size", type=int, default=6, help="中值滤波核大小")
    parser.add_argument("--n_thresh", type=int, default=100, help="搜索最佳阈值的数量")
    parser.add_argument("--connected_component_size", type=int, default=55,
                        help="连通组件分析的最小像素数")
    parser.add_argument("--load_npz_number", type=int, default=None, help="加载的文件数量")
    parser.add_argument("--load_npz_start", type=int, default=0, help="加载的文件起始位置")
    parser.add_argument("--vis_indices", type=list, default=None, help="可视化的样本索引")
    parser.add_argument("--label_threshold", type=int, default=None, help="标签中异常的大小阈值")
    parser.add_argument("--method_name", type=str, default="all", help="指定方法名称")
    args = parser.parse_args()
    
    # 如果指定了保存目录，则确保该目录存在
    task_save_dir = os.path.join(args.save_dir, args.task)
    if task_save_dir is not None:
        os.makedirs(task_save_dir, exist_ok=True)
        
    result_output = os.path.join(task_save_dir,f"{args.median_filter_size}_{args.connected_component_size}_evaluation_result.txt")  # 指定输出文件名
    if result_output is not None:
        with open(result_output, "w") as file:
            file.write(f"Results for task: {args.task}\n")
    
    # 加载数据（标注和各方法的异常分数图），均来自同一文件夹
    pattern = re.compile(args.pattern)
    ann_dict, scores_dict = load_dataset_from_npz(args.data_dir, pattern,args.fixed_size, args.load_npz_number, args.load_npz_start, args.label_threshold)
    if len(ann_dict) == 0:
        print("未加载到任何标注样本，请检查文件夹路径和文件格式。")
        return
    
    all_keys = set(ann_dict.keys())
    for method_name in scores_dict.keys():
        all_keys = all_keys & set(scores_dict[method_name].keys())
    common_keys = sorted(all_keys)
    if len(common_keys) == 0:
        print("没有所有方法都共有的样本！")
        exit(1)

    # 预先生成共用的随机索引，基于 common_keys
    n_total = len(common_keys)
    n_vis = min(args.n_images, n_total)
    if args.vis_indices is not None:
        random_indices = args.vis_indices
    else:
        random_indices = np.random.choice(n_total, size=n_vis, replace=False)
    print("可视化样本索引:", random_indices)
    
    # 对于每个方法，取标注和异常分数文件名的交集，确保数量一致
    for method_name, method_scores in scores_dict.items():
        if (args.method_name != "all" and method_name != args.method_name) or ("diag" in method_name):
            continue
        common_keys = sorted(set(ann_dict.keys()) & set(method_scores.keys()))
        if len(common_keys) == 0:
            print(f"Method {method_name} 无共同文件，跳过。")
            continue
        print(f"\nEvaluating method: {method_name}，共有 {len(common_keys)} 个样本。")
        # 堆叠标注和预测
        ann_list = [ann_dict[k] for k in common_keys]
        score_list = [method_scores[k] for k in common_keys]
        targets = torch.stack(ann_list)  # shape [N, 1, H_label, W_label]
        predictions = torch.stack(score_list)  # shape [N, 1, H_pred, W_pred]
        
        # 若 label 尺寸与预测尺寸不一致，则使用最近邻插值调整 label 尺寸
        if targets.shape[2:] != predictions.shape[2:]:
            print(f"Resizing targets from {targets.shape[2:]} to {predictions.shape[2:]}")
            targets_resized = F.interpolate(targets.float(), size=predictions.shape[2:], mode="nearest")
            targets_resized = targets_resized.long()
        else:
            targets_resized = targets
        
        # 将 predictions 转为 CPU numpy 数组
        # 并行化中值滤波处理
        def parallel_median_filter(predictions_cpu, median_filter_size):
            with Pool(cpu_count()) as pool:
                smoothed_list = pool.starmap(ndimage.median_filter, 
                                        [(predictions_cpu[i], median_filter_size) for i in range(predictions_cpu.shape[0])])
            return np.stack(smoothed_list)
        
        # 在主循环中添加：
        predictions_cpu = predictions.cpu().numpy()
        smoothed_predictions = parallel_median_filter(predictions_cpu, args.median_filter_size)
        predictions = torch.tensor(smoothed_predictions).to(predictions.device)
        
        # 计算 AUROC 和 AUPRC
        auroc = compute_auroc(predictions, targets_resized)
        auprc = compute_aupr(predictions, targets_resized)
        print(f"AUROC: {auroc:.4f}")
        print(f"AUPRC: {auprc:.4f}")
        
        # 搜索最佳阈值（基于 Dice 分数）
        best_dice, best_thresh = compute_best_dice(predictions, targets_resized, n_thresh=100, connected_component_size=args.connected_component_size)
        print(f"Best Dice: {best_dice:.4f} at threshold: {best_thresh:.4f}")
        
        # 利用最佳阈值进行二值化预测
        bin_maps = (predictions > best_thresh).float()
        
        # 根据预先生成的 random_indices 选择需要可视化的样本
        vis_indices = random_indices  # 注意：这里 random_indices 的范围在 [0, len(common_keys))
        
        # 构造保存路径：在保存根目录下为当前方法创建子文件夹，并保存可视化图像
        save_path = None
        if args.save_dir is not None:
            method_save_dir = os.path.join(task_save_dir, method_name)
            os.makedirs(method_save_dir, exist_ok=True)
            save_path = os.path.join(method_save_dir, "visualization.png")

        # 调用可视化函数，使用相同的随机索引
        plot_results([predictions[i] for i in vis_indices],
                    [bin_maps[i] for i in vis_indices],
                    [targets_resized[i] for i in vis_indices],
                    method_name=method_name,
                    n_images=len(vis_indices),
                    save_path=save_path)
        
        # 输出该方法的评估结果
        print(f"Method {method_name} evaluation:")
        print(f"  AUROC: {auroc:.4f}")
        print(f"  AUPRC: {auprc:.4f}")
        print(f"  Best Dice: {best_dice:.4f}")
        print(f"  Best Threshold: {best_thresh:.4f}")
        

        with open(result_output, "a") as file:
            file.write(f"Method {method_name} evaluation:\n")
            file.write(f"  AUROC: {auroc:.4f}\n")
            file.write(f"  AUPRC: {auprc:.4f}\n")
            file.write(f"  Best Dice: {best_dice:.4f}\n")
            file.write(f"  Best Threshold: {best_thresh:.4f}\n")

if __name__ == "__main__":
    main()
