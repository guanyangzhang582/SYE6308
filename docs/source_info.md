# 来源与版本记录

- 来源：[SYE6308_SF2_Classroom_2026](https://www.kaggle.com/code/ikunjun/sye6308-sf2-classroom-2026)。
- 获取方式：此前聊天中通过 Kaggle 公共 kernels pull 接口取得的完整源码，本次使用该快照整理；未声称重新查询了线上最新版本。
- Kaggle 快照版本号：**1**。
- API 返回的最后运行时间：**2026-10-29T18:30:59.2728Z**（UTC；这是运行时间，不是下载时间）。
- 本地整理日期：**2026-11-02（Asia/Shanghai）**。
- 原始 notebook SHA256：`d045ed7f932183684f92f4789dd38485cc320905d6998ffbbb192aa60def14a8`。
- 原始副本：[classroom_original.ipynb](../notebooks/classroom_original.ipynb)。
- 工作副本：[classroom_work.ipynb](../notebooks/classroom_work.ipynb)，只清除了代码单元输出及执行计数。
- 原始 notebook 共 53 个单元，其中 26 个 Markdown 单元；提取 22 个公共模块、8 个学生文件及 1 份练习规则 JSON。

## 版本标记的含义

Kaggle 页面版本号与正文中的 RC3 / R1 / R2 / R3 属于不同标记。源码同时包含 `classroom-rc3-submission-v1`、R2 学生指南文字和 R3 运行时修复说明，均照原样保留，没有自行统一或推断其发布时间。

原快照元数据的 `enableGpu=false` 是原页面会话配置，不代表训练框架不支持 GPU。训练配置保留 `device='auto'`，实际使用的设备由运行会话决定。

## 提取与核对方法

公共模块使用 Python AST 对 `CORE_FILES` 字面量求值后写出；练习规则来自 `ARENA_RULES_TEXT`。学生源码只移除了首行 `%%writefile`，其余内容（包括注释和空白）保留。没有执行原 notebook 的任何单元。

完整元数据、Markdown 单元对应关系、提取位置（从 0 开始的单元索引）及逐文件校验值见 [source_manifest.json](source_manifest.json)。[translate.md](../translate.md) 以相同单元索引标记译文，便于逐项核对。

## 使用与更新

源码来自课程公开 notebook，本地整理不改变原作者的归属或许可条件；本项目未新增对原代码的授权。原有注释及来源说明均保留。更新时应另存新快照并对比，不覆盖自己的实验代码与模型。

Arena 本轮仅保留[原页面链接](https://www.kaggle.com/code/ikunjun/sye6308-sf2-arena-2026)。Classroom 自带的 Arena 模块属于原始框架，已随公共模块提取；没有额外复制独立 Arena notebook。
