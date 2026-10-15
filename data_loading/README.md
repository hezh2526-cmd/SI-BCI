# BCI2020 PyTorch 数据适配与加载

本模块为三种实验协议提供同一个入口，公共 batch 始终为：

```text
X: torch.float32, (batch, 64, 512)
y: torch.int64,   (batch,)
```

模型自行调整维度：

```python
# EEGNet / ShallowConvNet
x = x.unsqueeze(1)       # (B, 1, 64, 512)

# LNN / LTC / CfC / LSTM
x = x.transpose(1, 2)    # (B, 512, 64)
```

## 防止重复归一化和数据泄漏

数据加载器根据协议自动选择字段：

- Within-subject：直接读取已经按该被试训练集标准化的 `X`；
- Mixed-subject：直接读取由各被试训练参数标准化后的 `X`；
- Cross-subject LOSO：读取 `X_volts`，再应用对应 fold 的
  `normalizer.npz`。该 normalizer 只由14名非留出训练被试拟合。

训练代码不要再次对返回的 batch 执行 Z-score。

## Python 用法

如果训练脚本从本项目根目录运行，可以直接使用完整模块路径：

```python
from lnns.data_loading import create_dataloaders

# 被试内
data = create_dataloaders("within", subject=1, batch_size=32, seed=42)

# 混合被试
data = create_dataloaders("mixed", batch_size=64, seed=42)

# LOSO fold 01：sub-01 仅作为 test
data = create_dataloaders("loso", fold=1, batch_size=64, seed=42)

for x, y in data.train:
    logits = model(x)
```

需要把预测追踪回被试和原始 trial 时：

```python
data = create_dataloaders(
    "loso", fold=1, batch_size=64, return_metadata=True
)
for x, y, metadata in data.test:
    print(metadata["subject_id"], metadata["original_epoch_index"])
```

`train` 使用固定 seed 打乱；`val/test` 不打乱，以便指标和预测结果稳定对应。
Windows 下默认 `num_workers=0`，避免多个 worker 分别复制数百 MB 的 EEG 数组。

## 命令行检查

单独查看某个协议：

```powershell
& 'C:\Users\He Zihan\anaconda3\envs\braindecode_env\python.exe' `
  'E:\PycharmProjects\EEGNet_Project\LNN\lnns\data_loading\bci2020_data.py' `
  --protocol within --subject 1 --batch-size 32
```

完整检查代表性的 Within、Mixed 和 LOSO 数据：

```powershell
& 'C:\Users\He Zihan\anaconda3\envs\braindecode_env\python.exe' `
  'E:\PycharmProjects\EEGNet_Project\LNN\lnns\data_loading\verify_data_loading.py'
```

验证内容包括形状、dtype、标签范围、NaN/Inf、归一化误差、协议输入字段以及
LOSO 留出被试隔离。
# Few-shot CBraMod-compatible loader

For the main small-sample experiment, use
`create_few_shot_dataloaders(subject, ratio_percent, subset_seed)`. It reads
the fixed indices from
`Data/BCI2020/processed/cbramod_compatible/preprocessed_data/few_shot_indices`
and fits each channel's z-score statistics only from that selected training
subset. The official 50-trial validation and test splits remain complete and
receive the same subset-derived statistics.

```python
from lnns.data_loading import create_few_shot_dataloaders

loaders = create_few_shot_dataloaders(
    subject="sub-01",
    ratio_percent=10,
    subset_seed=42,
    batch_size=16,
)
```
