# ResNet Complete Reproduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在独立分支中整合正式 CIFAR-10 训练、ImageNet ResNet 结构和四模型结果分析，并形成可从根目录阅读和复查的最终复现报告。

**Architecture:** 以 `paper-64k-schedule` 为基础创建隔离 worktree，通过两个保留历史的 merge 引入 ImageNet 结构和 CIFAR 分析。训练代码、分析代码与文档保持分工：分析程序只读原始结果并生成版本化产物，README 和最终报告串联环境、命令、模型和结论。

**Tech Stack:** Git worktree、Python 3.11、PyTorch 2.5.1、Torchvision 0.20.1、PyYAML、Matplotlib、pytest、Markdown。

## Global Constraints

- 最终分支：`resnet-complete-reproduction`。
- 最终工作区：`E:\文档\resnet-complete复现`。
- 基础分支：`paper-64k-schedule`。
- 合并顺序：先 `imagenet-resnet-architectures`，再 `cifar-results-analysis`。
- 不修改 `main`、三个原始功能分支或 `E:\文档\resnet复现\outputs`。
- 不提交 checkpoint、TensorBoard event 或原始 `outputs`。
- 不重新训练，不下载 ImageNet 或预训练权重。
- CIFAR 结论只使用 PlainNet/ResNet-20、56 的 seed 42、64k 实验。
- 不声称完成 32/44、110/1202、多 seed 或 ImageNet 训练。
- 不执行 push、PR 或合并回其他分支。

## File Map

- Create worktree: `E:\文档\resnet-complete复现`
- Merge: `models/imagenet_resnet.py`
- Merge: `tests/test_imagenet_resnet.py`
- Merge: `analysis/compare_cifar_results.py`
- Merge: `tests/test_compare_cifar_results.py`
- Regenerate: `analysis/results/cifar_summary.csv`
- Regenerate: `analysis/results/cifar_error_curves.png`
- Regenerate: `analysis/results/paper_comparison.png`
- Create: `tests/test_project_documentation.py`
- Create: `docs/阶段十一-完整复现分支整合总结.md`
- Create: `docs/阶段十三-四模型最终结果分析总结.md`
- Create: `README.md`
- Create: `docs/ResNet论文复现完整报告.md`

---

### Task 1: 创建隔离的完整复现工作区

**Files:**
- Create worktree: `E:\文档\resnet-complete复现`
- Branch: `resnet-complete-reproduction`

**Interfaces:**
- Consumes: `paper-64k-schedule`.
- Produces: clean named-branch worktree.

- [ ] **Step 1: Verify target path and branch are absent**

```powershell
if (Test-Path -LiteralPath "E:\文档\resnet-complete复现") {
    throw "目标工作区已存在，停止以避免覆盖"
}
if (git branch --list resnet-complete-reproduction) {
    throw "目标分支已存在，停止以避免覆盖"
}
```

- [ ] **Step 2: Create the worktree**

```powershell
git worktree add `
  "E:\文档\resnet-complete复现" `
  -b resnet-complete-reproduction `
  paper-64k-schedule
```

- [ ] **Step 3: Verify branch and base commit**

```powershell
git branch --show-current
git status --short
git rev-parse HEAD
git rev-parse paper-64k-schedule
```

Expected: branch is `resnet-complete-reproduction`, status is empty, hashes match.

- [ ] **Step 4: Run pre-merge baseline tests**

```powershell
$baselineTemp = Join-Path $env:TEMP (
  "resnet-complete-baseline-" + [guid]::NewGuid().ToString("N")
)
E:\Anaconda3\envs\resnet-paper\python.exe `
  -m pytest -q --basetemp $baselineTemp
```

Expected: all baseline tests pass.

---

### Task 2: 合并 ImageNet ResNet 结构

**Files:**
- Merge: `models/imagenet_resnet.py`
- Merge: `tests/test_imagenet_resnet.py`
- Merge: `docs/阶段九-ImageNet-ResNet结构学习总结.md`
- Merge: ImageNet design and plan.

**Interfaces:**
- Consumes: `imagenet-resnet-architectures`.
- Produces: ImageNet ResNet-18/34/50/101/152 constructors and tests.

- [ ] **Step 1: Merge with history**

```powershell
git merge --no-ff imagenet-resnet-architectures `
  -m "merge: add ImageNet ResNet architectures"
```

Expected: clean merge. On conflict, inspect `git status --short`, preserve both CIFAR and ImageNet interfaces, test, then finish the merge.

- [ ] **Step 2: Run ImageNet tests**

```powershell
$imagenetTemp = Join-Path $env:TEMP (
  "resnet-imagenet-tests-" + [guid]::NewGuid().ToString("N")
)
E:\Anaconda3\envs\resnet-paper\python.exe `
  -m pytest tests\test_imagenet_resnet.py -q `
  --basetemp $imagenetTemp
```

- [ ] **Step 3: Run five CPU forwards**

```powershell
E:\Anaconda3\envs\resnet-paper\python.exe -c @'
import torch
from models.imagenet_resnet import (
    imagenet_resnet18, imagenet_resnet34, imagenet_resnet50,
    imagenet_resnet101, imagenet_resnet152,
)
for constructor in (
    imagenet_resnet18, imagenet_resnet34, imagenet_resnet50,
    imagenet_resnet101, imagenet_resnet152,
):
    model = constructor().eval()
    with torch.no_grad():
        output = model(torch.randn(1, 3, 224, 224))
    print(constructor.__name__, list(output.shape))
'@
```

Expected: each output is `[1, 1000]`.

- [ ] **Step 4: Verify clean merge state**

```powershell
git status --short
git log -3 --oneline
```

---

### Task 3: 合并 CIFAR 结果分析

**Files:**
- Merge: `analysis/`
- Merge: `tests/test_compare_cifar_results.py`
- Merge: stage ten and analysis design/plan documents.
- Merge: complete-reproduction design and this plan.

**Interfaces:**
- Consumes: `cifar-results-analysis`.
- Produces: `run_analysis(results_root, output_dir)` and analysis CLI.

- [ ] **Step 1: Merge with history**

```powershell
git merge --no-ff cifar-results-analysis `
  -m "merge: add CIFAR results analysis"
```

- [ ] **Step 2: Run analysis tests**

```powershell
$analysisTemp = Join-Path $env:TEMP (
  "resnet-analysis-tests-" + [guid]::NewGuid().ToString("N")
)
E:\Anaconda3\envs\resnet-paper\python.exe `
  -m pytest tests\test_compare_cifar_results.py -q `
  --basetemp $analysisTemp
```

- [ ] **Step 3: Verify CIFAR and ImageNet imports coexist**

```powershell
E:\Anaconda3\envs\resnet-paper\python.exe -c @'
from models import create_model
from models.imagenet_resnet import imagenet_resnet18
print(type(create_model("resnet20")).__name__)
print(type(imagenet_resnet18()).__name__)
'@
```

- [ ] **Step 4: Run combined tests**

```powershell
$combinedTemp = Join-Path $env:TEMP (
  "resnet-combined-tests-" + [guid]::NewGuid().ToString("N")
)
E:\Anaconda3\envs\resnet-paper\python.exe `
  -m pytest -q --basetemp $combinedTemp
```

Expected: zero failures and zero errors.

---

### Task 4: 编写阶段十一整合总结

**Files:**
- Create: `docs/阶段十一-完整复现分支整合总结.md`

**Interfaces:**
- Consumes: verified merge history.
- Produces: branch/worktree learning summary.

- [ ] **Step 1: Write fixed sections**

```markdown
# 阶段十一：完整复现分支整合总结
## 本阶段目标
## 三个原始分支分别保存什么
## 为什么创建新的工作区
## 合并顺序和文件流向
## CIFAR与ImageNet模型如何共存
## 原始训练结果为什么不进入Git
## 本阶段运行的验证
## 日常如何进入完整复现工作区
## 可直接用于日报的总结
```

Include:

```bat
conda activate resnet-paper
cd /d "E:\文档\resnet-complete复现"
git branch --show-current
```

- [ ] **Step 2: Verify required facts**

```powershell
$text = Get-Content -Raw "docs\阶段十一-完整复现分支整合总结.md"
foreach ($value in @(
  "paper-64k-schedule",
  "imagenet-resnet-architectures",
  "cifar-results-analysis",
  "resnet-complete-reproduction",
  "E:\文档\resnet复现\outputs"
)) {
  if (-not $text.Contains($value)) { throw "缺少：$value" }
}
```

- [ ] **Step 3: Commit**

```powershell
git add "docs\阶段十一-完整复现分支整合总结.md"
git commit -m "docs: explain complete branch integration"
```

---

### Task 5: 完成第十三阶段四模型分析

**Files:**
- Regenerate: `analysis/results/*`
- Create: `docs/阶段十三-四模型最终结果分析总结.md`

**Interfaces:**
- Consumes: `E:\文档\resnet复现\outputs` read-only.
- Produces: final four-model artifacts and conclusion.

- [ ] **Step 1: Regenerate artifacts**

```powershell
E:\Anaconda3\envs\resnet-paper\python.exe `
  analysis\compare_cifar_results.py `
  --results-root "E:\文档\resnet复现\outputs" `
  --output-dir "analysis\results"
```

Expected:

```text
plain20 best_test_error=9.30%
resnet20 best_test_error=8.07%
plain56 best_test_error=12.03%
resnet56 best_test_error=6.92%
```

- [ ] **Step 2: Verify exact CSV values**

```powershell
$rows = Import-Csv "analysis\results\cifar_summary.csv"
$expected = @{
  plain20="9.300000"; resnet20="8.070000"
  plain56="12.030000"; resnet56="6.920000"
}
if ($rows.Count -ne 4) { throw "必须恰好有四个模型" }
foreach ($row in $rows) {
  if ($row.best_test_error_percent -ne $expected[$row.model]) {
    throw "错误率不匹配：$($row.model)"
  }
  if ($row.completed_iterations -ne "64000") {
    throw "未完成64000 iterations：$($row.model)"
  }
}
```

- [ ] **Step 3: Visually inspect PNGs**

Open:

```text
analysis/results/cifar_error_curves.png
analysis/results/paper_comparison.png
```

Verify no clipping, four distinguishable curves, 32k/48k/64k markers, and bar labels 8.75%, 8.07%, 6.97%, 6.92%.

- [ ] **Step 4: Write stage thirteen summary**

Required sections:

```markdown
# 阶段十三：四模型最终结果分析总结
## 为什么只保留20层和56层
## 四组实验公平性
## 最佳值与最终值
## PlainNet退化证据
## ResNet加深后的改善
## shortcut在56层网络中的作用
## 与论文结果对比
## 当前结论边界
## 如何重新生成结果
## 可直接用于日报的总结
```

Must state 2.73, 1.15, and 5.11 percentage points and explicitly exclude 32/44, multi-seed, 110/1202, and ImageNet training.

- [ ] **Step 5: Commit**

```powershell
git add analysis\results "docs\阶段十三-四模型最终结果分析总结.md"
git commit -m "docs: finalize four-model CIFAR analysis"
```

---

### Task 6: 用测试定义最终文档入口

**Files:**
- Create: `tests/test_project_documentation.py`

**Interfaces:**
- Produces: documentation and artifact contract.

- [ ] **Step 1: Write failing tests**

```python
"""最终复现项目文档入口测试。"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
REPORT = ROOT / "docs" / "ResNet论文复现完整报告.md"


def test_final_documentation_entrypoints_exist() -> None:
    paths = [
        README,
        REPORT,
        ROOT / "docs" / "阶段十一-完整复现分支整合总结.md",
        ROOT / "docs" / "阶段十三-四模型最终结果分析总结.md",
        ROOT / "analysis" / "results" / "cifar_summary.csv",
        ROOT / "analysis" / "results" / "cifar_error_curves.png",
        ROOT / "analysis" / "results" / "paper_comparison.png",
    ]
    for path in paths:
        assert path.is_file(), f"缺少最终项目入口：{path}"
        assert path.stat().st_size > 0, f"最终项目入口为空：{path}"


def test_readme_exposes_workflows() -> None:
    text = README.read_text(encoding="utf-8")
    for value in [
        "python train.py --model plain20 --epochs 5",
        "python train.py --model resnet20 --schedule paper",
        "tensorboard --logdir",
        "analysis/compare_cifar_results.py",
        "models/imagenet_resnet.py",
        "ResNet论文复现完整报告.md",
    ]:
        assert value in text


def test_report_states_scope_and_results() -> None:
    text = REPORT.read_text(encoding="utf-8")
    for value in [
        "PlainNet-20", "ResNet-20", "PlainNet-56", "ResNet-56",
        "2.73", "1.15", "5.11", "ResNet-18/34/50/101/152",
        "单个随机种子", "未训练",
    ]:
        assert value in text
```

- [ ] **Step 2: Verify RED**

```powershell
$docsRedTemp = Join-Path $env:TEMP (
  "resnet-docs-red-" + [guid]::NewGuid().ToString("N")
)
E:\Anaconda3\envs\resnet-paper\python.exe `
  -m pytest tests\test_project_documentation.py -q `
  --basetemp $docsRedTemp
```

Expected: fail because README and final report do not exist.

---

### Task 7: 创建 README 和最终报告

**Files:**
- Create: `README.md`
- Create: `docs/ResNet论文复现完整报告.md`
- Test: `tests/test_project_documentation.py`

**Interfaces:**
- Consumes: verified code, CSV, PNG, and stage documents.
- Produces: root project entry and final scientific report.

- [ ] **Step 1: Write README**

Required sections:

```markdown
# ResNet论文复现：CIFAR-10训练、退化实验与ImageNet结构
## 项目目标
## 已完成范围
## 核心实验结论
## 项目目录
## 环境准备
## 5 epoch快速验证
## 64k正式训练
## TensorBoard查看训练日志
## 重新生成结果分析
## CIFAR与ImageNet模型入口
## 文档阅读顺序
## Git分支说明
## 未包含内容与结论边界
```

Commands must be copyable from Anaconda Prompt and use `E:\文档\resnet-complete复现`.

- [ ] **Step 2: Write final report**

Required sections:

```markdown
# ResNet论文复现完整报告
## 摘要
## 1. 复现目标与研究问题
## 2. PlainNet baseline
## 3. 残差学习与shortcut
## 4. CIFAR-10网络结构
## 5. ImageNet ResNet结构
## 6. 数据集和预处理
## 7. 参数初始化
## 8. 损失函数与优化策略
## 9. 训练、评估和日志闭环
## 10. 正式实验设置
## 11. 四模型实验结果
## 12. 退化问题分析
## 13. 与论文结果对比
## 14. 复现范围与限制
## 15. 学习收获
## 16. 复现命令和产物
## 结论
```

It must distinguish preprocessing from Kaiming initialization, augmentation from test preprocessing, forward evaluation from training, best from final metrics, and CIFAR models from ImageNet models.

- [ ] **Step 3: Verify GREEN**

```powershell
$docsGreenTemp = Join-Path $env:TEMP (
  "resnet-docs-green-" + [guid]::NewGuid().ToString("N")
)
E:\Anaconda3\envs\resnet-paper\python.exe `
  -m pytest tests\test_project_documentation.py -q `
  --basetemp $docsGreenTemp
```

- [ ] **Step 4: Commit**

```powershell
git add README.md "docs\ResNet论文复现完整报告.md" `
  tests\test_project_documentation.py
git commit -m "docs: publish complete ResNet reproduction report"
```

---

### Task 8: 最终验证和交付

**Files:**
- Verify all integrated source, tests, docs, CSV, and PNG files.

**Interfaces:**
- Produces: fresh completion evidence and clean branch.

- [ ] **Step 1: Compile changed Python entrypoints**

```powershell
E:\Anaconda3\envs\resnet-paper\python.exe -m py_compile `
  train.py models\imagenet_resnet.py `
  analysis\compare_cifar_results.py `
  tests\test_imagenet_resnet.py `
  tests\test_compare_cifar_results.py `
  tests\test_project_documentation.py
```

- [ ] **Step 2: Check whitespace**

```powershell
git diff --check paper-64k-schedule...HEAD
```

- [ ] **Step 3: Run full suite**

```powershell
$finalTemp = Join-Path $env:TEMP (
  "resnet-complete-final-" + [guid]::NewGuid().ToString("N")
)
E:\Anaconda3\envs\resnet-paper\python.exe `
  -m pytest -q --basetemp $finalTemp
```

- [ ] **Step 4: Run four CIFAR GPU forwards**

```powershell
E:\Anaconda3\envs\resnet-paper\python.exe -c @'
import torch
from models import create_model
device = torch.device("cuda")
for name in ("plain20", "resnet20", "plain56", "resnet56"):
    model = create_model(name).to(device).eval()
    with torch.no_grad():
        output = model(torch.randn(1, 3, 32, 32, device=device))
    print(name, list(output.shape), output.device)
'@
```

Expected: each prints `[1, 10] cuda:0`.

- [ ] **Step 5: Run five ImageNet GPU forwards sequentially**

```powershell
E:\Anaconda3\envs\resnet-paper\python.exe -c @'
import torch
from models.imagenet_resnet import (
    imagenet_resnet18, imagenet_resnet34, imagenet_resnet50,
    imagenet_resnet101, imagenet_resnet152,
)
device = torch.device("cuda")
for constructor in (
    imagenet_resnet18, imagenet_resnet34, imagenet_resnet50,
    imagenet_resnet101, imagenet_resnet152,
):
    model = constructor().to(device).eval()
    with torch.no_grad():
        output = model(torch.randn(1, 3, 224, 224, device=device))
    print(constructor.__name__, list(output.shape), output.device)
    del model, output
    torch.cuda.empty_cache()
'@
```

Expected: each prints `[1, 1000] cuda:0`.

- [ ] **Step 6: Verify original workspaces and final branch**

```powershell
git -C "E:\文档\resnet复现" status --short --branch
git -C "E:\文档\resnet-imagenet结构" status --short --branch
git -C "E:\文档\resnet-results-analysis" status --short --branch
git status --short
git branch --show-current
git log --oneline --decorate -12
git worktree list
```

Expected: original workspaces unchanged; final worktree clean on `resnet-complete-reproduction`.

- [ ] **Step 7: Stop before external integration**

Present:

```text
Implementation complete. What would you like to do?

1. Merge back to paper-64k-schedule locally
2. Push and create a Pull Request
3. Keep the branch as-is (I'll handle it later)

Which option?
```
