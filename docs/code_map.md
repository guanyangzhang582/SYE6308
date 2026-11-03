# 代码导航

先看工作 notebook 的第 2 节配置和第 3 节四份学生模板，再按需要阅读框架。下面的单元索引从 0 开始，与 `source_manifest.json` 一致，界面中第几个单元则从 1 开始。

## 主要入口

| notebook 单元索引 | 原章节 | 功能 |
|---|---|---|
| 2、3、4 | 1 | 建立目录、写出框架和规则、安装与准备资源 |
| 6 | 2 | PvE 配置 |
| 8、10、12、14 | 3 | 观察、动作、奖励、模型四份 PvE 文件 |
| 16 | 4 | PvE 环境检查 |
| 20、22、24 | 5、5.1、6 | PvE 训练、回顾和评估 |
| 26、28、30、32 | 7—7.3 | 打包、模型选择器、自检和备份 |
| 34 | 8 | 两个原始 agent 对战 |
| 36 | 9.1 | PvP 独立配置 |
| 38、40、42、44 | 9.2 | 四份 PvP 学生文件 |
| 46、48、50、51 | 9.3—9.5 | PvP 检查、训练、评估和回顾 |

## 公共模块

全部位于 `src/framework/`，是原始快照；正常实验从 notebook 调用框架，不逐一双击这些文件。

| 模块 | 主要作用 |
|---|---|
| `lab_notebook.py` | notebook 任务启动、源码快照、检查门槛、结果显示 |
| `run_lab.py` | 环境检查、训练、评估和原始 agent 对战的执行入口 |
| `lab_review.py` | 训练曲线、报告回顾与实验备份 |
| `setup_sf2.py` | 缺失依赖安装、ROM 查找与校验 |
| `sf2_env.py` | Gymnasium 环境，把游戏交互转换为 PPO 所需接口 |
| `sf2_game.py` | 模拟器管理与经过验证的开局缓存 |
| `sf2_boot.py` | 从游戏启动流程进入所需状态 |
| `sf2_menu.py` | 游戏菜单识别与操作 |
| `sf2_start_cache.py` | 验证并缓存比赛起始状态 |
| `sf2_observation.py` | 每个 agent 的图像处理协议、帧堆叠与位置输入 |
| `sf2_actions.py` | 六帧动作定义、校验和基础动作表 |
| `sf2_bundle.py` | agent 包导出与加载，绑定权重、动作、源码和元数据 |
| `sf2_common.py` | 公共协议、运行时版本、学生模块加载等工具 |
| `sf2_release.py` | 配置与交付辅助、PvP 初始化与兼容性检查 |
| `sf2_pvp_eval.py` | 保存策略对固定模型的评估 |
| `sf2_submission.py` | 选择一个 agent，制作外层提交包和静态检查 |
| `sf2_arena_notebook.py` | notebook 中自检与独立 Arena 共用的启动和显示接口 |
| `sf2_arena.py` | 练习竞技场执行、终局统计及回执 |
| `sf2_arena_actor.py` | 独立 CPU 进程中的单个模型推理 |
| `sf2_arena_rpc.py` | Linux/Kaggle 进程间 JSON 通信 |
| `sf2_video.py` | 将实际模拟器帧和按键指示写入录像 |
| `input_contract_smoke.py` | 合成图像上的输入与模型导出/重载检查，不代表真实游戏测试 |

## 学生代码与远程路径

`src/student/pve/` 与 `src/student/pvp/` 各有观察、动作、奖励和模型四份源码。首次内容相同不代表两条训练路线会共享后续编辑。

执行 notebook 后，它们对应的远程目录为 `/kaggle/working/sf2_classroom_rc3/student/pve/` 和 `student/pvp/`。本地快照不会自动同步到这些目录。当前执行来源是工作 notebook 的源码单元，详情见 [README](../README.md)。
