# SYE6308：老师代码测试版

分支：`test/teacher-baseline-demo`。先把一次真实运行的流程接通，再做性能改进。

本分支没有附带训练好的模型、游戏 ROM 或正式比赛成绩。

实际完成的检查见 [TEST_RESULTS.md](TEST_RESULTS.md)：老师原版与演示配置的真实 PPO 导出/重载均已通过。真实游戏训练仍需到 Kaggle 完成。

| 路线 | 打开或运行 | 实际做什么 |
| --- | --- | --- |
| Kaggle 真实游戏示例 | [demo_teacher_classroom_2026.ipynb](demo_teacher_classroom_2026.ipynb) | 环境检查 → 8192 步 PvE 短训练 → 评估 → 打包 → 自检 |
| 本地 / GitHub Actions 接口检查 | [run_teacher_smoke.py](run_teacher_smoke.py) | 实际建立 PPO、输入合成图片、导出、重载并检查动作；没有游戏、没有学习 |

## 先看一次真实游戏流程

1. 下载本分支的 `demo_teacher_classroom_2026.ipynb`。
2. 登录 Kaggle，在新 Notebook 中使用 **Import Notebook** 上传它。训练优先选可用 GPU；安装依赖时开启 Internet。
3. 按老师说明附加课程授权的匹配 ROM；需要时填写 `ROM_FILE`。`.state` 文件不能代替 ROM。
4. 把演示组名 `Group_DEMO` 改成真实组号，然后按顺序运行。**Run All 会实际运行游戏与训练。**
5. 先看到 `ENVIRONMENT_CHECK_PASSED`，再进行 PvE 训练。失败就停止，保留错误；不要关闭检查强行继续。
6. 查看训练、评估视频和报告，再查看第 7 节生成的完整外层 `*_submission.zip`、`.sha256`，以及第 7.2 节自检记录。
7. 下载所选模型和结果；用同一个外层 ZIP 到[老师官方 Arena](https://www.kaggle.com/code/ikunjun/sye6308-sf2-arena-2026)的干净会话再次测试。

示例使用 Ryu 对 1 星电脑 Ryu、2 个环境、8192 transitions；原有普通 PPO、观察、动作、奖励和公共模块保持原样。这个预算只用于检查流程，不保证实力。第 8 节 playground、第 9 节 PvP 等额外任务保持关闭。

模型文件要区分：`best_agent.zip` / `last_agent.zip` 是完整 agent；独立 Arena 接收另行打包的 **外层 `*_submission.zip`**。静态通过、自检通过与模型实力是不同结果。不能把 `.pth` 改扩展名冒充提交包，也不能把老师页面的示例输出当作本组结果。

## 本地 / CI：先验证老师的模型接口

仅提取源码、检查语法时，无需安装机器学习依赖：

```bash
python3 run_teacher_smoke.py --prepare-only
```

这只输出 `TEACHER_SOURCE_PREPARED` 和 `PREPARE_ONLY_COMPLETE`，不会声称模型检查通过。

实际运行老师的合成接口检查：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-demo.txt
python run_teacher_smoke.py --output output/teacher-smoke
```

Windows 用户使用 `.venv\Scripts\activate` 激活虚拟环境。Ubuntu / GitHub Actions 可先用 PyTorch 官方 CPU 源安装 `torch`，避免下载 CUDA 组件；本分支的 workflow 已这样配置。

编排脚本使用标准库静态提取老师 Notebook 的 22 个公共模块及 4 个 PvE 学生文件，然后直接运行老师原样的 `input_contract_smoke.py`。只有真实报告出现 `SYNTHETIC_INPUT_CONTRACT_PASSED` 才算合成检查通过。缺少依赖、加载失败或不一致都会返回失败退出码。

默认产物在 `output/teacher-smoke/`：`prepare-report.json` 记录源码 SHA256 和语法准备结果，`smoke-report.json` 记录实际接口检查状态，`stdout.log` / `stderr.log` 记录实际日志；`code/` 和 `student/pve/` 保存提取的源码。这些产物已被 Git 忽略。

这个检查不需要 ROM 或 `stable-retro`，不会启动模拟器、调用 PPO 学习或留下可用的已训练模型；通过后仍要在 Kaggle 验证真实游戏。

GitHub **Actions → Teacher PPO synthetic interface check** 会在本测试分支推送时运行，也可手动启动。请查看该次报告，不要把合成通过当作真实对战成绩。

## 四人怎样接手

- A：准备匹配资源、跑环境检查、解释观察和动作。
- B：先跑本示例的基础训练，管理配置与模型版本。
- C：基线完成后再做奖励等单因素对照实验。
- D：做独立评估、打包、干净 Arena 自检并保存证据。

正式角色、推理时限、提交格式、渠道和截止日期以老师确认的课程要求为准。本测试版配置不是对正式规则的修改。`docs/street-fighter-project-analysis.md` 是早期分析，含旧版 17 动作等配置；当前技术入口以这里引用的 2026 Notebook 为准。

## 老师源码与授权

- [官方 Classroom 2026](https://www.kaggle.com/code/ikunjun/sye6308-sf2-classroom-2026)，Version 1，版本 ID `355509824`，Kaggle 页面标注 **Apache-2.0**。
- [官方 Arena 2026](https://www.kaggle.com/code/ikunjun/sye6308-sf2-arena-2026)，Version 2，版本 ID `355521542`。
- [teacher_classroom_2026.ipynb](teacher_classroom_2026.ipynb) 是老师原始 source 的干净副本：仅清空执行输出与执行编号，cell source 保留。
- 演示 Notebook 只增补中文说明、演示参数和运行开关；22 个公共模块和学生源码未改。
- 公共模块内部原有第三方来源与许可注释均保留。仓库原 `LICENSE` 不用于重新授权老师或其引用的第三方代码；请同时阅读本分支提供的老师代码许可和来源清单。
