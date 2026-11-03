# SYE6308 街霸强化学习实验

本项目将老师提供的 Classroom notebook 整理为便于在 **Trae 中阅读、编辑，在 Kaggle 远程内核中执行**的本地目录。先完成原始基线，再逐项比较观察、动作、奖励和模型设计。

**从这里开始：**阅读本页 → 对照 [页面中文翻译](translate.md) → 在 Trae 打开 [工作 notebook](notebooks/classroom_work.ipynb)。

## 当前完成状态

| 项目 | 状态 |
|---|---|
| Trae 连接 Kaggle 并执行多个单元 | 已由用户的小型验证 notebook 确认 |
| 老师框架支持 CUDA GPU 训练 | 已核实源码：普通 SB3 PPO，训练配置默认 `device='auto'` |
| Kaggle GPU 可用性 | 用户已确认：完成身份（ID）及手机号验证后，GPU 验证成功 |
| 源码整理、中文说明与翻译 | 已提供，静态检查见 [核验报告](docs/validation_report.md) |
| 街霸环境、ROM、训练及竞技场运行 | 尚未执行 |

GPU 训练需要在 Kaggle 会话中分配 GPU，并使用支持 CUDA 的 PyTorch。模拟器与游戏采样仍使用 CPU；老师代码使用一个学习器 GPU，选择双 GPU 会话不会自动进行双卡训练。独立 Arena 的推理默认使用 CPU，这与训练使用 GPU 不冲突。本地整理阶段未安装模拟器、CUDA 或 WSL；远程 GPU 可用性已由用户确认，正式 PPO 学习器的运行结果仍需单独记录。

## 目录与用途

```text
sye6308-streetfighter/
├── README.md
├── translate.md
├── notebooks/
│   ├── classroom_original.ipynb   # 原始快照，保留用于比较
│   └── classroom_work.ipynb       # 当前唯一实验执行入口
├── src/
│   ├── framework/                # 22 个公共模块的初始源码快照
│   └── student/
│       ├── pve/                  # 对内置 CPU 训练的四份模板快照
│       └── pvp/                  # 对固定模型训练的四份模板快照
├── docs/                         # 来源、代码导航、规则与核验记录
├── experiments/                  # 本地实验配置、说明与比较记录
└── outputs/                      # 从 Kaggle 下载的模型、日志、录像
```

完整代码导航见 [code_map.md](docs/code_map.md)；版本及逐文件校验值见 [source_info.md](docs/source_info.md)。来源是 [Classroom 页面](https://www.kaggle.com/code/ikunjun/sye6308-sf2-classroom-2026)的已获取快照。独立 [Arena 页面](https://www.kaggle.com/code/ikunjun/sye6308-sf2-arena-2026)用于后续提交测试，本轮未另行下载到此项目。

## Trae 连接 Kaggle 远程内核：探索与验证记录

更新日期：2026-11-03。远程连接与 CPU 环境信息来自前期实际测试输出；GPU 成功结论来自用户随后确认。正式街霸训练尚无完成报告。

### 方案选择

目标是在本地 Trae 中编辑和保存代码，使用 Kaggle 的 Python 环境与 GPU 算力。讨论过本地复现环境、SSH 隧道和远程 Jupyter 三种方式后，采用 Kaggle 官方的实验性 Jupyter Server 外部连接功能。

实际使用的是 **远程 Jupyter 内核连接，不是 SSH 登录**。Trae 基于 VS Code 并不能单独证明所有功能兼容，因此先用独立小型 notebook 验证连接，而没有为这一步复制完整街霸项目或安装训练依赖。

```text
本地 Trae 编辑 notebook
        → Kaggle Jupyter Server / Python 内核
        → 在 Kaggle 上执行代码
        → 输出返回 Trae
```

参考：[Kaggle notebook 与远程 Jupyter 官方文档](https://www.kaggle.com/docs/notebooks)、[外部编辑器连接公告](https://www.kaggle.com/product-announcements/567740)。官方将该入口标为实验性，具体界面以实际版本为准。

### 连接步骤

1. 在 Kaggle 创建自己的 notebook 并启动会话。
2. 打开 `Run → Kaggle Jupyter Server`，取得 `VS Code Compatible URL`。
3. 在 Trae 打开本地 `.ipynb`，通过“选择内核”连接该 Jupyter Server。
4. 选择远程列表中的 `Python 3 (ipykernel)`。
5. 执行简单计算、环境检查和跨单元变量读取，确认代码确实运行在远端。

看到远程内核列表说明服务可访问；实际执行单元后才能确认运行链路。连接 URL 可能包含令牌，不应保存到仓库或公开日志。这里不记录任何实际连接凭据。

### 远程执行的实际结果

| 检查项 | 已观察结果 | 结论 |
|---|---|---|
| 简单计算 | `21 * 2` 输出 `42` | 可以执行代码 |
| 系统与 Python | Linux、Python 3.13.15、`/usr/bin/python3` | 使用远程 Linux Python 环境 |
| 工作目录 | `/kaggle/working`，且 `/kaggle` 存在 | 与 Kaggle 环境一致 |
| 会话类型 | `Interactive` | 使用交互会话 |
| 跨单元状态 | 两个单元显示相同临时标记，第二单元输出 `49` 和 `INTERACTIVE_EXECUTION_OK` | 同一远程内核中的变量可以跨单元使用 |
| 最初的 PyTorch 环境 | `2.11.0+cpu`、CUDA 为 False | 最初连接的是 CPU 环境，不能据此否定远程连接 |
| 后续 GPU 验证 | 用户确认身份（ID）和手机号验证完成后，GPU 验证成功 | 账户已能使用 GPU，继续采用 Kaggle 远程执行方案 |

最初测试没有记录把 7 改成 8 后输出 64 的复测，异常回传和中断测试也未开启。GPU 型号、CUDA/PyTorch 版本及具体数值输出尚未写入本项目，不作推测。一次性连接验证文件已删除；下面保留可复现的最小示例。

第一个单元：

```python
import sys
import platform
from pathlib import Path
from uuid import uuid4

connection_marker = uuid4().hex[:12]
print("计算结果:", 21 * 2)
print("操作系统:", platform.system())
print("Python:", sys.executable)
print("当前目录:", Path.cwd())
print("Kaggle 目录存在:", Path("/kaggle").is_dir())
print("内核标记:", connection_marker)
```

第二个单元：

```python
print("同一内核标记:", connection_marker)
test_value = 7  # 可改成 8 并重新运行，平方应变为 64
print("结果:", test_value ** 2)
```

### GPU 选项置灰：本次遇到的问题与解决

曾在 `Settings → Accelerator` 看到 `None` 被选中，`GPU T4 x2` 和 TPU 同时置灰。**用户完成身份（ID）和手机号验证后，确认 GPU 已验证成功。** 这是本账户的实际解决过程，不据此断言所有账号都必须经过相同的身份验证流程。Kaggle 官方支持回复明确说明了手机验证要求，具体其他验证要求以账户页面提示为准。[手机验证说明](https://www.kaggle.com/product-feedback/451678)

之后使用 GPU 的流程为：在 Kaggle 会话设置中选择可用 GPU，等待会话重启，再从 Trae 连接新会话的远程内核。若加速器再次不可选，可检查账户验证、剩余配额、活动会话占用和页面提示。[GPU 使用与配额管理](https://www.kaggle.com/docs/efficient-gpu-usage)

下面的单元可在每次新会话中重新核对环境：

```python
import torch

print("PyTorch:", torch.__version__)
print("CUDA 编译版本:", torch.version.cuda)
assert torch.version.cuda is not None, "当前为 CPU 版 PyTorch"
assert torch.cuda.is_available(), "当前会话没有可用 CUDA GPU"
print("GPU:", torch.cuda.get_device_name(0))
x = torch.ones((256, 256), device="cuda:0")
y = x @ x
print("结果设备:", y.device)
print("结果:", y[0, 0].item())  # 预期为 256.0
```

GPU 环境可用与实际学习器使用 GPU 是两个检查点。正式训练时，老师代码的 `ACTUAL TRAINING CONFIG` 会输出模型参数的 `devices`，应看到 `["cuda:0"]`；训练结束报告再检查 `learner_parameters_changed`。第 4 节临时环境检查模型、独立评估、竞技场推理和游戏模拟器使用 CPU 属于框架设计，不代表正式学习器回退到 CPU。

### 本地文件与远程执行的边界

- 连接 Jupyter 内核不会自动上传整个本地目录，也不会自动同步本地 `.py` 文件。
- 当前通过工作 notebook 中的源码单元和 `%%writefile` 在远端生成框架及学生文件。
- Kaggle 网页副本与本地 notebook 不自动双向同步，以本地工作副本作为当前编辑依据。
- 训练输出先生成在 Kaggle，再下载到本地 `outputs/`；远程 `/kaggle/working` 不是 Windows 本地路径。
- 切换加速器或重启会话后重新确认连接、变量和设备，提前保存需要保留的结果。

下一步默认 GPU 基线实验按 [运行计划v1](docs/运行计划v1.md) 执行。该文件是计划，不是训练完成报告。

## 代码怎样运行

```text
游戏画面 + 可选位置输入
        → PPO 策略选择动作编号
        → 模拟器执行六帧按键计划
        → 得到新画面、伤害及终局信息
        → 计算奖励、收集样本、更新学习器
```

默认输入为四帧灰度图 `(4, 168, 192)`，可附加六个位置值。动作表包含 15 个基本动作。默认奖励为 `0.5 × (damage_dealt − damage_taken) / 176 + outcome_bonus`，自然获胜/失败分别额外加 `+1/-1`。三层 CNN 提取图像特征，位置分支处理位置输入，最终产生策略和价值预测。

本地工作 notebook 中的单元由远程内核执行，所以 `/kaggle/working` 等路径指向 Kaggle。第 1 节写出公共框架，第 3 节的 `%%writefile` 单元写出远程学生文件，然后启动器调用这些文件执行任务。

**本轮的修改约定：**在工作 notebook 中修改配置及学生代码单元；`src/` 用于独立阅读、搜索和对照。修改本地 `src/` 不会自动更新 notebook，也不会上传到 Kaggle。若先在 `.py` 快照中编辑，需要将代码正文同步回对应 notebook 单元，保留该单元的 `%%writefile` 首行，重新执行该单元和相应环境检查后才会用于训练。以后若改为 `.py` 主导开发，需要另行实现明确的同步机制。

连接远程内核也不会自动同步 Kaggle 网页上的 notebook 副本。将本地工作副本作为编辑依据，保存本地文件；不要同时在网页和 Trae 中维护两份不同的实验代码。

## 在 Trae 中开始实验

1. 打开本项目文件夹及工作 notebook。
2. 在自己的 Kaggle notebook 会话中通过 `Run → Kaggle Jupyter Server` 获取连接地址，在 Trae 中选择对应远程 Python 内核。连接地址可能含访问凭据，勿写入源码或实验记录。
3. 正式训练前，在 Kaggle 的会话设置分配 GPU，再检查远程 `torch.cuda.is_available()` 和老师代码打印的实际设备。连接流程参考 [Kaggle 官方文档](https://www.kaggle.com/docs/notebooks)。
4. 逐节执行下面的路线。可选开关保持关闭，直到准备运行相应任务；第一次不要直接 Run All。

| 原章节 | 操作 | 完成标志或检查点 |
|---|---|---|
| 1 | 依次运行三个设置单元，准备依赖与游戏资源 | `Setup finished`；检查版本和实际设备 |
| 2 | 配置自己的组名、角色、CPU 对手和训练参数 | 角色需按课程正式要求选择 |
| 3 | 执行全部四份学生模板，初次先沿用默认实现 | 远程学生文件写入成功 |
| 4 | 运行 PvE 环境检查，查看原始画面和模型输入 | `ENVIRONMENT_CHECK_PASSED` |
| 5 | 将 `RUN_PVE_TRAINING=True`，开始基线训练 | `TRAINING_COMPLETE`；保存 best/last 路径 |
| 5.1、6 | 开启结果回顾、独立评估并检查录像 | `EVALUATION_COMPLETE`；比较胜负、血量差和行为 |
| 7 或 7.1 | 选择一个模型、设置组号并打包 | `SUBMISSION_PACKAGED` |
| 7.2 | 对自己的完整提交包做实际自检 | `ARENA_SELF_CHECK_PASSED` |
| 7.3 | 导出实验备份并下载所需文件 | 本地确认文件存在，记录对应实验 |

原始模板的角色是 **ChunLi**，本地整理没有擅自更改默认值。若按之前的项目清单执行 Ryu-only，请在第 2 节设置 `PVE_AGENT_CONFIG['character']='Ryu'` 和 `PVE_CONFIG['opponent_characters']=['Ryu']`；之后使用 PvP 时也要核对学习者和已有模型的角色身份。角色 ID 大小写按代码使用 `Ryu`。更改角色、对手或难度后必须重跑环境检查，并查看实际角色；定向选择 CPU 角色在框架中仍标为实验性。

第 5 节默认请求 `100_000` 次交互，完整 rollout 实际会达到 `102_400`。这些是智能体 transition 数，不是游戏帧数或对局数。默认训练量是开始实验的基线，不保证获胜能力。

### 游戏资源

项目没有内置 ROM、预训练游戏策略或外部 `.state` 文件。需要老师提供的匹配游戏资源；`ROM_FILE` 必须是远程可读的 ROM 路径，Windows 本地路径不能直接由 Kaggle 内核读取。设置代码会扫描 `/kaggle/input`，Classroom 原配置还开启了历史外部来源的备用获取方式，但其可用性没有在本轮验证。缺失资源时按原报错处理，`.state` 不能替代 ROM。

### 修改入口

| 学生文件 | 工作 notebook 位置 | 修改内容 |
|---|---|---|
| `student_observation.py` | 3.1 / 9.2a | 输入裁剪、颜色、分辨率 |
| `student_actions.py` | 3.2 / 9.2b | 合法六帧动作表及短按键序列 |
| `student_reward.py` | 3.3 / 9.2c | 学习反馈，不改变比赛裁判 |
| `student_model.py` | 3.4 / 9.2d | 普通 PPO 网络和构建参数 |

PvE 与 PvP 源码独立。重复运行 `%%writefile` 会覆盖远程对应文件，修改应保存在 notebook 中。已有模型的观察和网络源码会做精确兼容性检查，注释或空白变化也可能导致从旧包继续学习失败。改变动作含义、观察或网络结构通常应训练兼容的新模型。

先建立原始基线，再一次修改一项。原清单禁止以固定条件规则代替学习策略；动作表定义可选操作，具体选择由策略决定。

## 评估、PvP 与提交

- 第 6 节对内置 CPU 评估；第 8 节让两个保存的 agent 对战，均不更新模型。
- 第 9 节使用一个学习者对一个固定模型训练，只更新学习者。可以以自己的旧模型为固定对手，但不会自动更新对手构成持续自博弈。
- 同时训练 P1/P2 可在第 9.1 节选择 `learner_sides=[1,2]`，先运行独立的 PvP 环境检查。P1/P2 是控制器端口，不是永久的屏幕左右位置。
- 查看验证胜率、血量差、录像和动作使用。奖励上涨不等于胜率提升；`cutoff` 不计为获胜。重复相同确定性开局也不等于独立试验。
- 最后将同一个完整提交包放到独立 Arena 的新会话验证；接口通过不代表模型强，也不代表已完成老师的正式评分。

| 文件 | 用途 |
|---|---|
| `best_agent.zip` / `last_agent.zip` | 完整 v3 智能体包，可评估、对战或作为新训练阶段的输入 |
| `*_submission.zip` | 对一个 agent 再封装的提交包，用于第 7.2 节和独立 Arena |
| `*_handoff.zip` | 实验源码快照和报告的备份 |
| `interrupted_policy.zip` / 原始 `.pth` | 开发恢复用途，不能直接当完整 v3 agent 输入 |
| `.sha256`、`receipt.json`、`arena_diagnostics.zip` | 保存所测试提交包的身份及自检证据 |

加载完整模型包开始下一阶段会使用新的优化器，不是精确恢复被中断的优化器状态。中断后“最新已完成训练”可能仍指向旧实验；比较或提交时优先记录和使用明确文件路径。

## 实验与文件保存约定

每次实验在 `experiments/` 下建立易读的编号记录，例如 `001_pve_ryu_baseline.md`，记录：改变了什么、组名和角色、种子、对手和难度、观察/动作/奖励/模型版本、训练配置、实际设备与软件版本、运行目录、评价条件和结果、模型路径与校验值。来自少数固定开局的结果应如实注明，不能描述为广泛泛化能力。

模型和录像先产生在远程 `/kaggle/working/sf2_classroom_rc3/`，需要通过生成的下载链接或 Kaggle Output 面板保存到本地 `outputs/`，建议按相同实验编号分目录。本轮没有实现自动下载或自动同步。内联链接无法下载时使用文件面板，不要仅为下载文件选择 `Save & Run All`，因为它会重新从头执行启用的任务。

仅保存文件不等于提交作业；代码、结果说明、展示材料及比赛提交内容以老师最终要求为准。

## 正式规则与练习代码的差异

| 内容 | 此前项目清单 | 当前 Classroom / Arena 练习框架 |
|---|---|---|
| 角色 | Ryu-only | 支持多个角色，模板为 ChunLi |
| 决策时限 | 一个游戏帧内完成决策 | 六帧动作计划；Arena 两秒 watchdog 不等于单帧期限 |
| 提交 | `act(state)` 与 `.pth` | 完整 v3 agent 和外层 submission ZIP |
| 赛事组织 | 资格赛及前八淘汰赛 | 提供对战检查，不是完整赛事调度系统 |

原代码明确将规则标为草案。正式角色、时限、提交和评分要求需由老师确认，不通过修改规则校验来“兼容”提交包。原始练习规则已提取到 [arena_rules_v1.json](docs/arena_rules_v1.json)。
