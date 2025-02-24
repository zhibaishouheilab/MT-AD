import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Sobel(nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        self.filter = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=0, bias=False)
        # 定义 Sobel 核
        Gx = torch.tensor([[-1.0, 0.0, 1.0],
                           [-2.0, 0.0, 2.0],
                           [-1.0, 0.0, 1.0]])
        Gy = torch.tensor([[ 1.0,  2.0,  1.0],
                           [ 0.0,  0.0,  0.0],
                           [-1.0, -2.0, -1.0]])
        G = torch.cat([Gx.unsqueeze(0), Gy.unsqueeze(0)], 0)  # shape: (2, 3, 3)
        G = G.unsqueeze(1)  # shape: (2, 1, 3, 3)
        self.filter.weight = nn.Parameter(G, requires_grad=requires_grad)
        self.Repad = nn.ReplicationPad2d(padding=(1, 1, 1, 1))
    
    def forward(self, img):
        # 1. 复制边界并卷积计算梯度
        x = self.Repad(img)
        x = self.filter(x)
        x = torch.mul(x, x)
        x = torch.sum(x, dim=1, keepdim=True)
        x = torch.sqrt(x)
        
        # 2. 根据边缘图像的分布计算下阈值和上阈值
        #    这里取 75% 和 95% 分位数（可根据需要修改）
        nonzero = x[x > 0]
        if nonzero.numel() > 0:
            lower = torch.quantile(nonzero, 0.75)
            upper = torch.quantile(nonzero, 0.95)
        else:
            lower, upper = 0, 1
        # 3. 将边缘图像的值剪裁到 [lower, upper] 区间
        x[x > 0] = torch.clamp(x[x > 0], lower, upper)
        # 4. 归一化到 [0,1]：先减去下阈值，再除以 (upper - lower)
        if upper - lower > 0:
            x[x > 0] = (x[x > 0] - lower) / (upper - lower)
        return x

# 定义处理单层图像的函数
def process_layer(layer, sobel):
    """
    对单个二维数组（layer）进行 Sobel 边缘检测处理。
    输入:
      layer: numpy 数组，二维，形状 [H, W]
      sobel: 已初始化的 Sobel 模块
    返回:
      edge_map: numpy 数组，二维，边缘图，值在 [0,1]
    """
    # 将 layer 转换为 float32 Tensor，并增加 batch 和 channel 维度
    tensor_img = torch.from_numpy(layer.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        edge_tensor = sobel(tensor_img)
    # 去除 batch 和 channel 维度，并转换为 numpy 数组
    edge_map = edge_tensor.squeeze().detach().cpu().numpy()
    return edge_map

def process_file(file_path, output_folder, sobel):
    """
    处理单个 npy 文件：对除最后一层外的每一层生成边缘图，
    最后一层保留原样，然后将处理后的数组保存到 output_folder 中（同名 npy 文件）。
    """
    data = np.load(file_path)
    if data.ndim != 3:
        raise ValueError(f"文件 {file_path} 数据维度不为 3，请检查数据格式。")
    num_layers, H, W = data.shape
    # 创建一个与原数据同样形状的输出数组
    out_data = np.empty_like(data)
    
    # 对每一层（除最后一层）生成边缘图
    for i in range(num_layers - 1):
        layer = data[i]
        edge_map = process_layer(layer, sobel)
        out_data[i] = edge_map
    # 保留最后一层原样
    out_data[-1] = data[-1]
    
    # 保存到输出文件夹，保持同名
    filename = os.path.basename(file_path)
    output_path = os.path.join(output_folder, filename)
    np.save(output_path, out_data)
    print(f"Processed {filename} and saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="对文件夹内所有 npy 文件进行边缘检测处理（除最后一层）")
    parser.add_argument("--input_folder", type=str, default="/home/ubuntu/Project/data/MSLUB_public/train_normal", help="输入 npy 文件所在文件夹")
    parser.add_argument("--output_folder", type=str, default="/home/ubuntu/Project/data/MSLUB_public/train_normal_edge", help="保存处理后 npy 文件的文件夹")
    args = parser.parse_args()
    
    # 确保输出文件夹存在
    os.makedirs(args.output_folder, exist_ok=True)
    
    # 初始化 Sobel 对象（可复用）
    sobel = Sobel()
    sobel.eval()  # 设置为评估模式
    
    # 遍历输入文件夹中所有 npy 文件
    files = sorted([f for f in os.listdir(args.input_folder) if f.endswith('.npy')])
    for fname in files:
        file_path = os.path.join(args.input_folder, fname)
        try:
            process_file(file_path, args.output_folder, sobel)
        except Exception as e:
            print(f"Error processing {fname}: {e}")

if __name__ == "__main__":
    main()
