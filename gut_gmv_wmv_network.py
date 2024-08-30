################################
#               written by Raven                #
# network build                                   #
#envornment: python3.9                     #
################################
# gut_sparcc;gmv+wmv_KL





# gut_sparcc
import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
# 读取CSV文件
data = pd.read_csv('G:\MDD_SZ_BD\GUT\genus_relative_focus_removalcor.csv', encoding='gb18030', encoding_errors='ignore')
num_people, num_features = data.shape
output_path = 'G:\MDD_SZ_BD\GUT\sparcc'
# 确保目标路径存在
if not os.path.exists(output_path):
    os.makedirs(output_path)
epsilon = 0.001
data_with_epsilon = data + epsilon  # 添加常数
# 计算每个特征的均值
mean = data_with_epsilon.mean(axis=0).values
# 初始化相关性矩阵
correlation_matrix = np.zeros((num_features, num_features))
# 定义方差到相似度的转换函数（未使用）
def variance_to_similarity(variance, variances=None):
    similarity = 1.0 - (variance / np.max(variances))
    return similarity
colors = [
    "#006837", "#1a9850", "#66bd63", "#a6d96a",
    "#d9ef8b", "#fee08b", "#fdae61", "#f46d43", "#d73027"]
custom_cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", colors)
for t in range(num_people):
    person_data = data_with_epsilon.iloc[t:t + 1].values
    for i in range(num_features):  # X
        for j in range(num_features):  # Y
            person_data_i = person_data[:, i]
            person_data_j = person_data[:, j]
            # 防止分母为零和对数无效操作
            with np.errstate(divide='ignore', invalid='ignore'):
                log = np.log(np.divide(person_data_i, person_data_j, out=np.zeros_like(person_data_i),
                                       where=(person_data_j != 0)))
                log = np.where(np.isfinite(log), log, 0)  # 将无效的对数计算结果设置为0
            log_mean = np.log(mean[i] / mean[j])
            num = np.sum((log - log_mean) ** 2)
            correlation_matrix[i, j] = num
    correlation_matrix = np.sqrt(correlation_matrix)
    max_value = np.max(correlation_matrix)
    min_value = np.min(correlation_matrix)
    normalized_data =  1-(correlation_matrix - min_value) / (max_value - min_value)
    print(f'sparcc{t+1}',normalized_data)
    plt.figure(figsize=(10, 8))
    sns.heatmap(normalized_data, cmap=custom_cmap, cbar=True)
    heatmap_output_path = os.path.join(output_path, f'heatmap\HeatMap of No. {t + 1} Participants.tiff')
    plt.savefig(heatmap_output_path, format='tiff', dpi=300)
    plt.show()
    plt.close()
    csv_file_name = f'sub_{t + 1:03}.csv'
    csv_file_path = os.path.join(output_path, csv_file_name)
    pd.DataFrame(normalized_data).to_csv(csv_file_path, index=False, header=False)





# GMV
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
def compute_kl_divergence(p, q):
    """计算 KL 散度"""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    epsilon = 1e-10
    p = np.clip(p, epsilon, 1.0)
    q = np.clip(q, epsilon, 1.0)
    return np.sum(p * np.log(p / q))
def process_nii_file(nii_file, template_file, index):
    # 读取 NIfTI 文件
    nii_img = nib.load(nii_file)
    template_img = nib.load(template_file)
    nii_data = nii_img.get_fdata()
    template_data = template_img.get_fdata()
    all_regions = np.unique(template_data)
    # 确保提取最后 246 个区域
    num_regions = 246
    if len(all_regions) < num_regions:
        raise ValueError("Template file has fewer than 246 regions.")
    # 选择最后 246 个区域
    regions = all_regions[-num_regions:]
    # 计算 KL 散度矩阵
    kl_matrix = np.zeros((num_regions, num_regions))
    for i in range(num_regions):
        region_i = (template_data == regions[i])
        prob_i = np.sum(nii_data[region_i]) / np.sum(nii_data)
        for j in range(num_regions):
            region_j = (template_data == regions[j])
            prob_j = np.sum(nii_data[region_j]) / np.sum(nii_data)
            # 计算 KL 散度
            kl_ij = compute_kl_divergence(prob_i, prob_j)
            kl_ji = compute_kl_divergence(prob_j, prob_i)
            # 使用对称 KL 散度
            kl_matrix[i, j] = (kl_ij + kl_ji) / 2
    # 确保矩阵是对称的
    kl_matrix = (kl_matrix + kl_matrix.T) / 2
    # 创建保存文件夹
    output_dir = 'G:/MDD_SZ_BD/MRI/GMV_SC'
    fig_dir = 'G:/MDD_SZ_BD/MRI/GMV_SC/Heatmap'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # 保存对称矩阵
    csv_filename = os.path.join(output_dir, f'GMV_sub_{index + 243}.csv')
    np.savetxt(csv_filename, kl_matrix, delimiter=',')
    print(f'GMV_sub_{index + 243}', kl_matrix)
    # 绘制热图
    colors = ["#2166AC", "#4393C3", "#92C5DE", "#D1E5F0", "white", "#F4A582", "#D6604D", "#B2182B", "#67001F"]
    custom_cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", colors)
    plt.figure(figsize=(10, 8))
    sns.heatmap(kl_matrix, cmap=custom_cmap, cbar=True)
    heatmap_filename = os.path.join(fig_dir, f'heatmap_{index + 243}.tiff')
    plt.savefig(heatmap_filename, format='tiff', dpi=300)
    plt.show()
    plt.close()
def batch_process_nii_files(nii_folder, template_file):
    # 查找所有 .nii 文件
    nii_files = glob.glob(os.path.join(nii_folder, '*.nii'))
    for index, nii_file in enumerate(nii_files):
        process_nii_file(nii_file, template_file, index)
# 示例用法
nii_folder = 'G:/MDD_SZ_BD/MRI/GMV/SZ'
template_file = 'D:/DPABI/DPABI_V6.2_220915/DPABI_V6.2_220915/Templates/Reslice_BrainnetomeAtlas_BNA_MPM_thr25_1.25mm.nii'
batch_process_nii_files(nii_folder, template_file)





# WMV
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
def compute_kl_divergence(p, q):
    """计算 KL 散度"""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    epsilon = 1e-10
    p = np.clip(p, epsilon, 1.0)
    q = np.clip(q, epsilon, 1.0)
    return np.sum(p * np.log(p / q))
def process_nii_file(nii_file, template_file, index):
    # 读取 NIfTI 文件
    nii_img = nib.load(nii_file)
    template_img = nib.load(template_file)
    nii_data = nii_img.get_fdata()
    template_data = template_img.get_fdata()
    all_regions = np.unique(template_data)
    # 确保提取最后 246 个区域
    num_regions = 246
    if len(all_regions) < num_regions:
        raise ValueError("Template file has fewer than 246 regions.")
    # 选择最后 246 个区域
    regions = all_regions[-num_regions:]
    # 计算 KL 散度矩阵
    kl_matrix = np.zeros((num_regions, num_regions))
    for i in range(num_regions):
        region_i = (template_data == regions[i])
        prob_i = np.sum(nii_data[region_i]) / np.sum(nii_data)
        for j in range(num_regions):
            region_j = (template_data == regions[j])
            prob_j = np.sum(nii_data[region_j]) / np.sum(nii_data)
            # 计算 KL 散度
            kl_ij = compute_kl_divergence(prob_i, prob_j)
            kl_ji = compute_kl_divergence(prob_j, prob_i)
            # 使用对称 KL 散度
            kl_matrix[i, j] = (kl_ij + kl_ji) / 2
    # 确保矩阵是对称的
    kl_matrix = (kl_matrix + kl_matrix.T) / 2
    # 创建保存文件夹
    output_dir = 'G:/MDD_SZ_BD/MRI/WMV_SC'
    fig_dir = 'G:/MDD_SZ_BD/MRI/WMV_SC/Heatmap'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # 保存对称矩阵
    csv_filename = os.path.join(output_dir, f'WMV_sub_{index + 243}.csv')
    np.savetxt(csv_filename, kl_matrix, delimiter=',')
    print(f'WMV_sub_{index + 243}', kl_matrix)
    # 绘制热图
    colors = ["#2166AC", "#4393C3", "#92C5DE", "#D1E5F0", "white", "#F4A582", "#D6604D", "#B2182B", "#67001F"]
    custom_cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", colors)
    plt.figure(figsize=(10, 8))
    sns.heatmap(kl_matrix, cmap=custom_cmap, cbar=True)
    heatmap_filename = os.path.join(fig_dir, f'heatmap_{index + 243}.tiff')
    plt.savefig(heatmap_filename, format='tiff', dpi=300)
    plt.show()
    plt.close()
def batch_process_nii_files(nii_folder, template_file):
    # 查找所有 .nii 文件
    nii_files = glob.glob(os.path.join(nii_folder, '*.nii'))
    for index, nii_file in enumerate(nii_files):
        process_nii_file(nii_file, template_file, index)
# 示例用法
nii_folder = 'G:/MDD_SZ_BD/MRI/WMV/SZ'
template_file = 'D:/DPABI/DPABI_V6.2_220915/DPABI_V6.2_220915/Templates/Reslice_BrainnetomeAtlas_BNA_MPM_thr25_1.25mm.nii'
batch_process_nii_files(nii_folder, template_file)







# 相关系数
import numpy as np
# 示例：三个 246x246 矩阵
matrix1 = np.random.rand(246, 246)
matrix2 = np.random.rand(246, 246)
matrix3 = np.random.rand(246, 246)
# 初始化存储结果的相关系数矩阵
correlation_matrix = np.zeros((246, 246))
# 逐元素计算相关系数
for i in range(246):
    for j in range(246):
        # 提取每个位置的三个值
        values = np.array([matrix1[i, j], matrix2[i, j], matrix3[i, j]])
        # 计算均值
        mean = np.mean(values)
        # 计算协方差矩阵
        cov_matrix = np.cov(values, rowvar=False, bias=True)
        # 计算相关系数矩阵
        corr_matrix = np.corrcoef(values, rowvar=False)
        # 存储相关系数（取上三角或下三角元素）
        # 这里使用均值作为示例
        correlation_matrix[i, j] = np.mean(corr_matrix)
print("相关系数矩阵:\n", correlation_matrix)
