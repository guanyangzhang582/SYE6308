# 实际测试记录

日期：2026-10-31。执行平台：macOS Apple Silicon，Python 3.12.14。

| 检查 | 结果 |
| --- | --- |
| 原始 Notebook 的 53 个 cell source | 与老师下载文件逐字一致 |
| 演示 Notebook 的共享代码与学生源码 | 22 个公共模块和 8 个学生源码格保持原样 |
| 原始 Notebook：真实 PPO 合成输入 / 导出 / 重载 | `SYNTHETIC_INPUT_CONTRACT_PASSED` |
| Ryu 演示配置：真实 PPO 合成输入 / 导出 / 重载 | `SYNTHETIC_INPUT_CONTRACT_PASSED` |
| 模型参数量 | `11229296` |
| 导出前后预测概率最大误差 | `0.0`（两次检查均为 0） |
| 输入形状 | `(4, 168, 192)`，另有可选 position6 |
| GitHub Actions | 等待分支上传后运行，本记录不声称远程 CI 已通过 |
| 真实模拟器 / 游戏训练 | 本机未完成，见下方环境错误；本记录不声称游戏或 8192 步训练已完成 |

## 本机真实游戏环境检查

已执行老师原样的 `setup_sf2.py --sb3 --legacy-fetch`。当前 Python 3.12 分支安装 `stable-retro==0.9.5`，但其标记为 macOS arm64 的 wheel 内 `_retro` 二进制实际为 x86_64，导入失败：

```text
ImportError: incompatible architecture (have x86_64, need arm64)
```

因此本机流程停在模拟器导入，未进行游戏、ROM 检验、环境检查、训练、评估或 Arena 自检。没有通过关闭检查或修改游戏规则来跳过错误。完整短训练示例保留为 Kaggle Notebook，仍需在匹配的 Linux/Kaggle 环境中实际完成。

## 本次合成检查的实际依赖

- `torch==2.15.0`
- `stable-baselines3==2.7.0`
- `gymnasium==1.2.3`
- `numpy==2.5.4`
- `opencv-python-headless==5.0.0.93`

合成检查使用真实老师模型代码和序列化流程，但输入是合成图像，未调用 PPO 学习；临时随机模型在检查结束后清理，不是训练成果。源码校验见 `teacher-source-manifest.json`。
