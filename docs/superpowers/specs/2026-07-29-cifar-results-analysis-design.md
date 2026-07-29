# CIFAR-10 正式实验结果分析设计

## 目标

将 PlainNet-20、ResNet-20、PlainNet-56、ResNet-56 的64k正式实验自动汇总成
可复查的CSV、论文口径错误率曲线和论文结果对照图，用数据验证普通深层网络的
退化问题以及残差连接带来的改善。

## 工作区与数据边界

- 分析代码位于 `E:\文档\resnet-results-analysis`。
- Git分支为 `cifar-results-analysis`，基于 `paper-64k-schedule`。
- 原始正式结果仍位于 `E:\文档\resnet复现\outputs`。
- 分析脚本通过命令行参数读取原始结果根目录，不复制或修改checkpoint。
- 脚本只读取 `history.csv` 和 `config.yaml`。
- 生成的汇总、图表和学习文档保存在分析Worktree。

## 输入实验

必须存在以下四个实验：

```text
plain20/seed42_paper64k
resnet20/seed42_paper64k
plain56/seed42_paper64k
resnet56/seed42_paper64k
```

每个实验必须满足：

- 最后一条记录的 `iteration` 为64000；
- 最后一条记录的 `is_partial_epoch` 为True；
- CSV包含训练/测试loss、accuracy、学习率、耗时等正式字段；
- config中的schedule为paper；
- batch size为128；
- seed为42；
- 最大更新数为64000；
- 学习率节点为32000和48000；
- 学习率序列为0.1、0.01、0.001。

任何实验缺失或配置不一致时，脚本应明确报错并停止，不能静默生成不可比图表。

## 指标口径

论文使用错误率，因此统一转换：

```text
train_error = 100 × (1 - train_accuracy)
test_error = 100 × (1 - test_accuracy)
```

每个模型汇总：

- 最佳测试准确率和错误率；
- 最佳结果对应的epoch和iteration；
- 最终训练准确率和错误率；
- 最终测试准确率和错误率；
- 总训练分钟数；
- 参数量；
- 是否完成64000 iterations。

同时保留最佳值和最终值，不混用二者。

## 生成文件

```text
analysis/results/cifar_summary.csv
analysis/results/cifar_error_curves.png
analysis/results/paper_comparison.png
```

### cifar_summary.csv

每个模型一行，使用固定列名，便于日报、论文表格和后续脚本继续读取。

### cifar_error_curves.png

使用上下两个子图：

1. 训练错误率随global iteration变化；
2. 测试错误率随global iteration变化。

四个模型使用固定颜色和线型。图中标注：

- 32000：学习率从0.1降到0.01；
- 48000：学习率从0.01降到0.001；
- 64000：训练结束。

横轴统一使用global iteration，不使用epoch，保证与论文日程一致。

### paper_comparison.png

使用分组柱状图对照：

| 模型 | 论文测试错误率 |
|---|---:|
| ResNet-20 | 8.75% |
| ResNet-56 | 6.97% |

复现实验使用最佳测试错误率，并在柱顶标注精确数值。PlainNet没有论文表格中的
精确最终数字，因此不伪造PlainNet论文柱状数据。

## 代码结构

新增：

```text
analysis/compare_cifar_results.py
tests/test_compare_cifar_results.py
```

分析脚本分成清晰单元：

- 读取并验证单个CSV；
- 读取并验证config；
- 汇总单个模型；
- 验证四个实验的公平比较条件；
- 写summary CSV；
- 绘制错误率曲线；
- 绘制论文对照图；
- 命令行入口。

测试使用临时合成CSV和YAML，不依赖电脑上的正式输出目录。实际结果生成作为
独立集成验证执行。

## 命令行接口

```bat
python analysis\compare_cifar_results.py ^
  --results-root "E:\文档\resnet复现\outputs" ^
  --output-dir "analysis\results"
```

脚本只允许显式指定输入根目录和输出目录，不写死用户名或绝对路径。

## 图表规范

- 使用Matplotlib；
- 使用清晰标题、坐标名称、图例和网格；
- 纵轴单位明确写成Error Rate (%)；
- 采用适合打印和色觉区分的颜色；
- PNG使用至少150 DPI；
- 不平滑或修改原始数据；
- 不把TensorBoard平滑曲线当作原始实验结果；
- 图中所有点直接来自history.csv。

## 论文结论验证

分析文档必须分别回答：

1. 同为20层时，ResNet是否优于PlainNet；
2. PlainNet从20层加深到56层时，训练错误率是否上升；
3. ResNet从20层加深到56层时，测试错误率是否下降；
4. 同为56层时，shortcut带来多大改善；
5. ResNet-20和ResNet-56与论文表格相差多少；
6. 为什么PlainNet-56训练误差更高证明这不是普通过拟合；
7. 单次seed结果能够支持什么，不能支持什么。

## 本阶段只完成

- 汇总四个已经完成的正式实验；
- 生成可复现图表和CSV；
- 对照论文分析20层与56层结果；
- 编写阶段十学习总结和日报材料；
- 将分析工作保存在独立分支。

## 本阶段暂不进行

- 重新训练任何模型；
- 删除或移动正式checkpoint；
- 训练32层或44层模型；
- 合并已有分支；
- 将训练结果上传GitHub；
- 修改论文中的基准数值。

## 完成标准

1. 合成数据单元测试覆盖缺失文件、缺失字段、未完成64k和配置不一致；
2. 正式四模型结果通过验证；
3. summary数值能从原始CSV独立复算；
4. 两张PNG成功生成且视觉检查清晰；
5. 结果明确显示PlainNet退化与ResNet改善；
6. 阶段十文档记录精确数值、计算方式和结论边界；
7. 分析分支提交完整，原始训练目录不发生修改。
