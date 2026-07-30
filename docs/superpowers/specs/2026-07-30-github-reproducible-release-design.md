# GitHub 可下载复现版本设计

## 1. 目标

把当前只存在于本地 `resnet-complete-reproduction` 分支的完整工程发布为私有
GitHub 仓库的默认版本。仓库所有者或受邀协作者登录 GitHub 后，应当能够克隆
仓库、创建环境、自动下载 CIFAR-10、运行测试和短程训练，并可选择重新执行
64,000 iteration 正式实验。

仓库保持 Private。这里的“在 GitHub 上复现”是指从 GitHub 获取完整代码和
复现说明后在本地 CPU/GPU 环境运行；GitHub 网页本身不承担 CUDA 正式训练。

## 2. 分支与历史策略

- 以 `resnet-complete-reproduction` 的完整内容作为新版 `main`。
- 在移动 `main` 之前，为原来的 5-epoch baseline 提交保留
  `baseline-5epoch` 标签。
- 开发和验证在 `codex/github-reproducible-main` 隔离分支完成。
- 验证通过后把该分支合并或快进到 `main` 并推送。
- 不删除现有 `paper-64k-schedule`、`imagenet-resnet-architectures`、
  `cifar-results-analysis` 和 `resnet-complete-reproduction` 分支。

## 3. 克隆后的用户路径

README 首页提供以下顺序：

1. 登录有权限的 GitHub 账户；
2. 使用 SSH 或 HTTPS 克隆私有仓库；
3. 创建并激活 `resnet-paper` Conda 环境；
4. 验证 Python、PyTorch、CUDA 和 GPU；
5. 运行自动测试；
6. 执行 1 epoch smoke test；
7. 执行 PlainNet/ResNet 5 epoch 快速对照；
8. 按需执行四模型 64k 正式训练；
9. 使用 TensorBoard 查看日志；
10. 使用分析脚本汇总正式结果。

所有命令以仓库根目录为工作目录，不依赖作者电脑上的绝对路径。

## 4. 环境安装策略

提供两个层次的环境入口：

- `environment.yml`：适合现有 Windows + NVIDIA GPU 环境，移除作者电脑的
  `prefix`，避免还原到 `E:\Anaconda3\...`。
- README 手动安装路径：先创建 Python 3.11 环境，再从 PyTorch cu121 索引
  安装 `torch==2.5.1` 和 `torchvision==0.20.1`，最后安装其余依赖。

保留 `requirements-lock.txt` 作为当前环境快照，但不把它描述成对所有操作系统
都通用的唯一安装方式。

## 5. 自动化复现入口

新增轻量级验证脚本，串联以下只读或低成本检查：

- Python 和关键依赖可导入；
- PyTorch 能识别 CUDA（若用户选择 GPU 模式）；
- CIFAR PlainNet/ResNet 能完成前向传播；
- ImageNet ResNet-18/34/50/101/152 能完成结构级前向验证；
- 项目测试可以从仓库根目录运行。

正式训练继续由 `train.py` 提供，不增加隐藏的训练默认值。smoke、5 epoch 和
64k 三种入口必须明确区分，避免把短程验证写成论文结果复现。

## 6. 数据、结果与大文件策略

- CIFAR-10 由 `torchvision` 在首次运行时自动下载到仓库内 `data/`。
- `data/`、TensorBoard event、checkpoint 和大型本地训练目录不提交 Git。
- 保留体积较小的正式汇总 CSV 和结果图，让用户不重新训练也能查看已有证据。
- README 说明从头训练会生成 `outputs/` 和 `runs/`，并给出预计耗时与硬件差异。
- 不承诺不同 GPU、驱动、PyTorch 小版本或随机环境得到逐位相同的结果。

## 7. 报告改版

复现报告不再逐章套用 DQN 模板，改为项目自身结构：

1. 复现目标与学习收获；
2. ResNet 核心原理；
3. 项目实现过程；
4. 实验环境与训练配置；
5. PlainNet 与 ResNet 实验结果；
6. 结果分析与论文对比；
7. GitHub 下载与复现指南；
8. 局限性与后续工作；
9. 结论。

保留必要的结果表和两张核心图，减少重复的代码入口、证据锚点和模板化段落。
报告应明确：仓库是私有的，只有仓库所有者和受邀协作者能克隆。

## 8. 验证标准

发布前必须满足：

- 从干净工作区读取 README 的命令不存在作者电脑专属路径；
- `environment.yml` 不含本地 `prefix`；
- 全部自动测试通过；
- CIFAR 四个正式模型的 CPU 或 CUDA 前向测试通过；
- ImageNet 五个结构的前向测试通过；
- 1 epoch smoke test能生成配置、CSV、checkpoint 和 TensorBoard 日志；
- `.gitignore` 阻止数据集、checkpoint 和训练日志进入 Git；
- README 中的相对链接均可解析；
- Markdown 与 Word 报告包含 GitHub 复现章节，且版式重新检查；
- 推送后的 GitHub `main` 指向完整版本，原 baseline 标签可见。

## 9. 不在本次范围内

- 把仓库改为 Public；
- 在 GitHub Actions 上进行 CUDA 正式训练；
- 上传 CIFAR-10、ImageNet、checkpoint 或完整 TensorBoard 日志；
- 补跑 PlainNet/ResNet-32、44、多 seed、ResNet-110/1202；
- 训练 ImageNet 或提供预训练权重；
- 保证跨硬件逐位确定性。
