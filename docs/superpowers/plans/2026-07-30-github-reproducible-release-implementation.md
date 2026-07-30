# GitHub Reproducible Release Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish the complete private repository as the default GitHub version so an authorized user can clone it, install the environment, verify the implementation, rerun CIFAR-10 experiments, and read a project-specific reproduction report.

**Architecture:** Keep training behavior in the existing model/data/engine/train modules. Add one small repository verification CLI, make environment files machine-independent, reorganize README around a clean-clone workflow, and rebuild the report from a project-specific Markdown source. Preserve the old main commit with a tag before updating main.

**Tech Stack:** Python 3.11, PyTorch 2.5.1+cu121, Torchvision 0.20.1+cu121, pytest 8.3.3, Conda, Git, Markdown, python-docx.

## Global Constraints

- The GitHub repository remains Private.
- Only the repository owner and invited collaborators can clone it.
- The new default `main` contains the complete reproduction.
- Preserve the old 5-epoch main commit with tag `baseline-5epoch`.
- Do not upload CIFAR-10, ImageNet, checkpoints, TensorBoard event files, or full training output directories.
- CIFAR-10 must download automatically at first use.
- Distinguish 1-epoch smoke, 5-epoch learning validation, and 64,000-iteration formal training.
- Do not claim ImageNet training or pretrained weights.
- Do not claim multi-seed statistical superiority.
- Commands in tracked documentation must not depend on the author's absolute `E:\...` paths.

---

### Task 1: Clean-clone verification CLI

**Files:**
- Create: `verify_reproduction.py`
- Create: `tests/test_verify_reproduction.py`
- Modify: `models/__init__.py`

**Interfaces:**
- Consumes: `models.create_model(name, num_classes)` and ImageNet model factories.
- Produces: `run_checks(device_name: str, include_imagenet: bool) -> list[CheckResult]` and a CLI returning exit code 0 only when every requested check passes.

- [ ] **Step 1: Write failing unit tests**

Test that CPU verification covers `plain20`, `resnet20`, `plain56`, and `resnet56`, returns `[1, 10]`, optionally verifies all five ImageNet factories with `[1, 1000]`, and reports an invalid device clearly.

- [ ] **Step 2: Run the focused tests**

Run:

```bat
python -m pytest tests/test_verify_reproduction.py -q
```

Expected: FAIL because `verify_reproduction.py` does not exist.

- [ ] **Step 3: Implement the minimal CLI**

Use a frozen result record:

```python
@dataclass(frozen=True)
class CheckResult:
    name: str
    shape: tuple[int, ...]
    device: str
```

Expose:

```python
def resolve_device(device_name: str) -> torch.device: ...
def run_checks(device_name: str = "auto", include_imagenet: bool = False) -> list[CheckResult]: ...
def main() -> int: ...
```

Use `torch.inference_mode()`, batch size 1, CIFAR input `3x32x32`, and ImageNet input `3x224x224`.

- [ ] **Step 4: Run focused and regression tests**

```bat
python -m pytest tests/test_verify_reproduction.py -q
python -m pytest -q --basetemp .pytest-tmp\github-release
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bat
git add verify_reproduction.py tests/test_verify_reproduction.py models/__init__.py
git commit -m "feat: add clean-clone reproduction verifier"
```

### Task 2: Portable environment and repository hygiene

**Files:**
- Modify: `environment.yml`
- Modify: `requirements-lock.txt`
- Modify: `.gitignore`
- Create: `tests/test_repository_release.py`

**Interfaces:**
- Consumes: tracked repository files.
- Produces: static release tests proving no local Conda prefix, no tracked absolute author path in user-facing entry files, and expected ignore rules.

- [ ] **Step 1: Write failing static tests**

Tests must assert:

```python
assert "prefix:" not in Path("environment.yml").read_text(encoding="utf-8")
assert "data/" in Path(".gitignore").read_text(encoding="utf-8")
assert "runs/" in Path(".gitignore").read_text(encoding="utf-8")
assert "*.pt" in Path(".gitignore").read_text(encoding="utf-8")
```

Also assert the environment name is `resnet-paper` and the required Python/PyTorch/Torchvision versions are documented.

- [ ] **Step 2: Run the test and confirm failure**

```bat
python -m pytest tests/test_repository_release.py -q
```

Expected: FAIL because `environment.yml` contains a local `prefix`.

- [ ] **Step 3: Make environment files portable**

Remove the `prefix` field. Keep Python 3.11 and the verified package versions. Add comments or README guidance explaining that PyTorch CUDA wheels may be installed separately if Conda cannot resolve `+cu121`.

- [ ] **Step 4: Strengthen ignore rules**

Ignore all local training artifacts under `outputs/` except explicitly tracked lightweight summary artifacts under `analysis/results/`. Keep source, tests, configuration examples, CSV summaries, and figures trackable.

- [ ] **Step 5: Run focused and full tests**

```bat
python -m pytest tests/test_repository_release.py -q
python -m pytest -q --basetemp .pytest-tmp\github-release
```

- [ ] **Step 6: Commit**

```bat
git add environment.yml requirements-lock.txt .gitignore tests/test_repository_release.py
git commit -m "chore: make reproduction environment portable"
```

### Task 3: Rewrite README for an authorized clean clone

**Files:**
- Modify: `README.md`
- Modify: `tests/test_repository_release.py`

**Interfaces:**
- Consumes: commands implemented by Tasks 1 and 2.
- Produces: a single authoritative clean-clone guide with repository-relative commands.

- [ ] **Step 1: Extend static README tests**

Assert README includes:

- Private repository permission warning;
- SSH and HTTPS clone forms;
- `conda env create -f environment.yml`;
- CUDA verification;
- `python verify_reproduction.py`;
- `python -m pytest`;
- 1 epoch, 5 epoch, and `--schedule paper` commands;
- TensorBoard command using `runs`;
- ImageNet structure-only boundary;
- expected output folders;
- no `E:\文档` absolute path.

- [ ] **Step 2: Run test and confirm failure**

```bat
python -m pytest tests/test_repository_release.py -q
```

- [ ] **Step 3: Replace README with project-specific guide**

Use this top-level order:

1. project purpose and current results;
2. private-repository clone requirements;
3. quick start;
4. environment options;
5. verification levels;
6. formal training;
7. TensorBoard and outputs;
8. ImageNet structure verification;
9. result analysis;
10. repository structure;
11. limitations and reproducibility notes.

- [ ] **Step 4: Validate links and commands**

Parse Markdown links and assert every relative target exists. Run the CPU verifier and full test suite.

- [ ] **Step 5: Commit**

```bat
git add README.md tests/test_repository_release.py
git commit -m "docs: add private clean-clone reproduction guide"
```

### Task 4: Reorganize the reproduction report

**Files:**
- Create: `docs/ResNet复现报告.md`
- Modify: `README.md`
- Modify: `tests/test_repository_release.py`
- Create or modify locally: `tools/build_resnet_report_docx.py`
- Produce locally: `复现报告/2026-07-30_ResNet复现报告/2026-07-30_ResNet复现报告-飞书版.docx`

**Interfaces:**
- Consumes: `analysis/results/cifar_summary.csv`, `cifar_error_curves.png`, and `paper_comparison.png`.
- Produces: a concise project-specific Markdown report tracked in Git and a visually verified DOCX for Feishu import.

- [ ] **Step 1: Add report structure tests**

Assert the report contains exactly the project-specific major sections:

```text
复现目标与学习收获
ResNet 核心原理
项目实现过程
实验环境与训练配置
实验结果
结果分析与论文对比
从私有 GitHub 仓库复现
局限性与后续工作
结论
```

Assert it contains all four formal errors, the private-repository boundary, and no phrase saying it follows the DQN template.

- [ ] **Step 2: Run the static test and confirm failure**

```bat
python -m pytest tests/test_repository_release.py -q
```

- [ ] **Step 3: Write the new Markdown report**

Reuse verified numerical results but remove template-shaped duplication. Keep two figures and one primary results table. Use repository-relative links and commands.

- [ ] **Step 4: Build and visually verify DOCX**

Generate the Word document, sanitize title formatting, export to PDF, render every page, and inspect every page for clipping, overflow, broken tables, orphan headings, and unreadable figures.

- [ ] **Step 5: Run report and regression tests**

```bat
python -m pytest tests/test_repository_release.py -q
python -m pytest -q --basetemp .pytest-tmp\github-release
```

- [ ] **Step 6: Commit tracked report changes**

```bat
git add README.md docs/ResNet复现报告.md tests/test_repository_release.py
git commit -m "docs: publish project-specific reproduction report"
```

### Task 5: End-to-end release validation

**Files:**
- Modify if needed: files from Tasks 1–4.

**Interfaces:**
- Consumes: the complete clean-clone workflow.
- Produces: release evidence suitable for updating `main`.

- [ ] **Step 1: Run static and unit tests**

```bat
python -m pytest -q --basetemp .pytest-tmp\github-release-final
```

- [ ] **Step 2: Run CPU verification**

```bat
python verify_reproduction.py --device cpu
```

Expected: four CIFAR model checks pass.

- [ ] **Step 3: Run CUDA structure verification**

```bat
python verify_reproduction.py --device cuda --include-imagenet
```

Expected: four CIFAR and five ImageNet model checks pass without training.

- [ ] **Step 4: Run one-epoch smoke training**

```bat
python train.py --model resnet20 --epochs 1 --run-name github_release_smoke
```

Expected: exit code 0 and generated config, history, best/latest checkpoints, and TensorBoard event.

- [ ] **Step 5: Audit tracked files**

Confirm no `data/`, `.pt`, TensorBoard event, local absolute paths, or smoke output is staged or tracked.

- [ ] **Step 6: Commit any validation fixes**

Use an explicit file list and a terse message describing the fix.

### Task 6: Preserve baseline and publish the private GitHub main

**Files:**
- Git refs only.

**Interfaces:**
- Consumes: validated `codex/github-reproducible-main`.
- Produces: remote tag `baseline-5epoch` and remote default branch `main` containing the complete release.

- [ ] **Step 1: Verify release ancestry and remote state**

```bat
git merge-base --is-ancestor main codex/github-reproducible-main
git status -sb
git remote -v
```

- [ ] **Step 2: Create the baseline tag**

```bat
git tag -a baseline-5epoch 08d07ae -m "Preserve the original five-epoch baseline"
```

- [ ] **Step 3: Push release branch and tag**

```bat
git push -u origin codex/github-reproducible-main
git push origin baseline-5epoch
```

- [ ] **Step 4: Update and push main**

Fast-forward local `main` to the validated release commit and push `main`. Do not force-push.

- [ ] **Step 5: Verify remote refs**

```bat
git ls-remote --heads origin main codex/github-reproducible-main
git ls-remote --tags origin baseline-5epoch
```

Confirm `main` resolves to the release commit and the baseline tag resolves to the old main commit.

- [ ] **Step 6: Final handoff**

Report the private clone URL, default branch, preserved baseline tag, exact verification commands, final test count, and local report links.
