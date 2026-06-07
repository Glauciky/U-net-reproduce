# U-net-reproduce
## 本科生复现15年U-net经典论文代码，欢迎交流指正
## U-Net 复现（PyTorch）

本项目为基于 PyTorch 的 U-Net 网络复现，用于医学图像分割任务（如息肉分割）。

---

## 📌 项目简介

U-Net 是经典医学图像分割模型（Ronneberger et al., 2015）。

本项目实现 U-Net，并用于 CVC-ClinicDB 等数据集训练与测试。

---

## 📁 项目结构

```text
U-Net/
├── data/
│   ├── train/        # CVC-ClinicDB (训练集)
│   ├── val/          # CVC-ClinicDB (验证集)
│   └── test/         # ETIS-LaribPolypDB (测试集)
├── unet_model.py     # U-Net 网络结构
├── data_loader       # 数据加载和增强
├── train_eval.py     # 训练和测试脚本
├── main.py           # 运行模型的主程序，内含可视化代码
├── best_model.pth    #保存最佳权重
├── vis/
│   └──epoch_1.png/   #可视化运行结果
└── README.md
```
---

## 📊 数据集说明

本仓库**不包含数据集**（体积较大 + 版权限制）。

### 📥 下载地址

- CVC-ClinicDB：
  https://polyp.grand-challenge.org/CVCClinicDB/

- ETIS-Larib：
  https://polyp.grand-challenge.org/ETISLarib/

---

### 📂 数据格式

下载后请整理为：

数据集名（例如：train）/
- images/
- masks/

要求：图像与标签文件名一一对应。
### 数据集划分

#### 训练/验证集（CVC-ClinicDB）
- 总样本数：约 612 张  
- 划分比例：Train : Val = 4 : 1  
- Train：约 489 张  
- Val：约 123 张  
- 划分方式：随机划分，可固定随机种子保证可复现

#### 测试集（可以删去，本模型并没有泛化学习能力（ETIS-LaribPolypDB）
- 用于模型最终评估  
- 跨数据集验证模型泛化能力  

---

---

## 🚀 训练

```bash
python main.py
```
训练过程中：
- 自动保存 best model（基于 val Dice）
- 记录 loss / dice 数值变化
- 保存预测可视化结果

## 📈 实验结果
|dataset|Loss|Val Dice|
|---|---|---|
|CVC-Val|0.1063|0.8747|

## 论文原文下载
- U-Net: Convolutional Networks for Biomedical Image Segmentation
  https://arxiv.org/pdf/1505.04597.pdf
