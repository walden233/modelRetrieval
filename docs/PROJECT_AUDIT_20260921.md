# 项目现状核查（2026-09-21）

本次先读取 `tmp/唐艺通-中期报告.pdf`（23 页），再核对 `AGENTS_CN.md`、`src/`、`runs/`、`configs/`、`scripts/`、`docs/`、`artifacts/`、`dataset/` 与 `tmp/`。中期报告是研究进展的历史依据；现存文件用于判断目前哪些结果可追溯，二者不能自动视为同一实验版本。本轮未训练模型、调用标注 API、修改算法或覆盖历史产物。

论文结构及后续执行方案见 [论文与实验总计划](THESIS_AND_EXPERIMENT_PLAN.md)。

## 1. 总体判断

项目已具备三模态离线检索的代码与实验基础，后续不需要推倒重写。现阶段最优先的工作是统一实验协议、恢复结果证据链，再开展结构化语义、语义弱监督视频学习及轨迹表示改进。

现存数据和产物并未全部丢失，但不能据此认定中期报告数值已经恢复。已发现报告—指标 JSON—相似度矩阵之间的不一致，以及现存系统评估查询混入训练场景的问题。正式论文应区分“中期历史结果”和“新协议复现实验”，以可复算的新结果形成主结论。

## 2. 各模块实际状态

| 范围 | 已有基础 | 未完成或需要验证 |
|---|---|---|
| 数据 | RH20T 扫描、人机配对、轨迹提取、scene/task 划分；WHIRL 视频读取支持 | 跨模态统一划分、原始执行 ID、坐标/时间语义、独立人工相关性基准 |
| 视频 | V-JEPA/VideoMAE、双域头/共享/独立编码器、多正样本与一致性损失；训练、评估、原始骨干对照入口 | 结构化语义弱监督、跨任务正样本挖掘、弱标签噪声控制、严格视角/机器人泛化协议 |
| 轨迹 | 双分支 Transformer + 共享投影头；6/21 关键点；增强和两阶段训练 | 人机物理量可比性、全局运动信息、时间缺失处理、阶段级对齐 |
| 语义 | VLM 单视频描述/粗标签、解析与归一化、同步/Batch、缓存、文本编码 | 有时间边界的动作阶段、对象关系/状态变化、独立人工质量与相关性标注 |
| 融合 | 离线建库、逐模态点积、z-score、权重融合、缺失重归一、七种组合、scene 聚合 | 无泄漏的统一评估、新 qrels 相关性、验证集权重搜索、质量降权对照 |
| 模型仓库 | 样本级检索记录与查询入口 | 样本到真实模型 ID 的映射、能力/适用条件、模型级去重评价 |
| 服务 | `src/bise/service/query_pipeline.py` 是分数融合骨架 | 未形成接入编码器和检索库的完整服务；按本轮要求暂不安排工程建设 |
| 产物 | checkpoint、配置、划分、embedding、matrix、JSON、图像均有保留 | 文件不可变性、哈希、指标复算校验、图表来源清单、异地备份与恢复演练 |

`docs/` 中多份计划形成于 4—5 月，部分“未完成”状态已经过时。例如当前 cfg3 已存在语义标注。不能直接按旧计划的完成度开展新工作。

## 3. 数据和产物盘点

以下是当前本地目录计数，不代表所有样本均通过质量检查，也不代表中期报告当时的完整数据规模。

| 数据 | task 目录 | scene 目录 | MP4 文件 |
|---|---:|---:|---:|
| RH20T cfg2 | 109 | 434 | 2,604 |
| RH20T cfg3 | 64 | 256 | 1,536 |

- 报告正文记载 cfg2 为 173 个任务，与当前本地 109 个任务不一致，需找回子集抽取清单或原始映射后解释。
- 当前子集中未发现名称包含 `depth` 或 `calib` 的文件/目录；这只是文件名扫描结果，不能据此声称原始 RH20T 没有深度或标定数据。
- cfg2 语义 manifest、raw response、normalized annotation 各 436 条；cfg3 各 256 条；两者错误 JSONL 当前均为 0 行。
- 视频目录中 10 个子目录有 `params.json`，其中 8 个有 `best_model.pth`；轨迹分别为 6 个和 6 个。目录可能包含复制件，不能当成独立重复实验数。
- `artifacts/` 约 12 GB，`dataset/` 约 7.2 GB。`tmp/jsons/`、`tmp/heatmap/`、`tmp/restored_charts/` 当前为空。
- `.gitignore` 排除了 `artifacts/`、`dataset/` 和 `tmp/`，因此单纯 Git 提交不会保护实验数据。

## 4. 必须先解决的证据问题

### 4.1 中期数值与现存产物不一致

选取几个具有代表性的条目，均指 human-to-robot task R@1；系统项为系统 task R@1。

| 项目 | PDF 中期报告 | 现存 `metrics.json` | 由同目录 matrix + metadata 重算 |
|---|---:|---:|---:|
| 视频 E4，cfg2 | 0.375 | 0.440000 | 0.375000 |
| 轨迹 T2，cfg2 | 0.238 | 0.095630 | 0.075630 |
| cfg2 三模态均匀融合 | 0.470 | 0.615776 | 本轮未对该系统项重算 |

来源：

- 视频：`artifacts/runs/video/video_vjepa_rh20t_intra_20260429_202630/final_test/`。重算 MRR 为 0.540146，现存 JSON 为 0.610146；报告为 0.545。另有 `final_test3/`，不能只凭名称确定最终版本。
- 轨迹：`artifacts/runs/trajectory/T2_trajectory_augment_scene_20260430_225507/final_test/`。重算 MRR 为 0.130773，现存 JSON 为 0.156077。
- 系统：`artifacts/retrieval/rh20t_cfg2_v1/eval/ablation_uniform/video_trajectory_semantic_text/metrics.json`。

重算口径：按保存的 task ID 确定全部正样本，对相似度降序排序，计算首个正样本名次。两项单模态矩阵均与保存 embedding 的 `human @ robot.T` 完全一致（本次 NumPy 计算最大绝对差为 0）。因此不能把差异简单归因于浮点舍入。差异产生原因尚未确定，不能推断为任何人的主观行为。

处理原则：保留原文件，不补写、猜测或调整数据以匹配报告。建立每张表/图的来源映射；从可信 embedding/matrix 重新计算，无法追溯的报告数值仅作为历史记载。新结果使用独立 run/eval ID。

### 4.2 现存 cfg2 系统评估不是严格预留测试

按照系统 `require_modalities = [video, trajectory, semantic_text]` 筛选查询，再将查询 scene ID 与建库所用 checkpoint 的 `split_manifest.json` 对齐：

| 对照模型 | 完整三模态查询数 | 查询独立 scene 数 | 与 train 重叠的查询/scene | 与 val 重叠的查询/scene | 与 test 重叠的查询/scene |
|---|---:|---:|---:|---:|---:|
| 视频 E1 | 393 | 213 | 313 / 171 | 36 / 19 | 44 / 23 |
| 轨迹 T2 | 393 | 213 | 313 / 171 | 44 / 23 | 36 / 19 |

建库配置是 `configs/retrieval/library_rh20t_cfg2.json`；对照 checkpoint 目录分别为 `video_vjepa_rh20t_baseline_20260429_145608`、`T2_trajectory_augment_scene_20260430_225507`。查询来源为该库 `manifests/query_human_eval.jsonl`。

现存建库脚本按 task 采样 scene，没有用两个训练模型共同遵守的测试 manifest 过滤。系统 `run_info.json` 记载 393 条查询、327 个聚合后候选 scene。因此：

1. 这份产物可以用于旧系统功能检查，不可标为无泄漏的 cfg2 held-out 泛化成绩。
2. 视频和轨迹的 val/test 身份也不一致；新系统实验应先统一母清单再训练全部模态。
3. 上述结论针对现存文件，不证明它就是 PDF 数字对应的那次实验。

### 4.3 轨迹问题首先是信息语义不一致

`src/bise/data/rh20t/hand_pose_extractor.py` 保存 `multi_hand_world_landmarks`。MediaPipe 官方说明，这些坐标以手部近似几何中心为原点；它们不能直接作为手在场景中的全局平移轨迹。见 [MediaPipe 官方 Hands 文档](https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/hands.md?plain=1)。

另一方面，`trajectory_dataset.py` 直接载入机器人 `tcp_base.npy`，没有将其转换为与手局部坐标一致的物理表示。需要抽样确认实际历史 NPY 的生产版本，不能仅由当前提取器断言所有旧文件都来自它。

另外，dataset 直接删去无手检测帧，未把原 frame index / timestamp 传入模型，会压缩动作时间；读取失败时递归替换为下一个样本，可能改变记录与实际输入的对应关系。现有编码器本身已经包含输入投影、五层 Transformer（T2 配置）和输出投影头；“再加一层”不直接解决这些问题。

### 4.4 评价和系统表述需要统一

- `calculate_label_retrieval_metrics` 中名为 R@K 的指标实际是“Top-K 至少出现一个正样本”的查询命中率；多正样本场景应注明 Hit@K，不能与“召回了多少比例正样本”的 Recall@K 混称。
- 单模态存在 camera 级候选，系统评价先做 camera-to-scene max 聚合；不同候选规模和正样本密度下数值不能直接横比。
- 旧 `calculate_ndcg(ranks)` 只基于首个正样本 rank，不能代替新多相关候选、分级 relevance 的 NDCG；正式实验必须锁定函数/公式版本。
- 当前系统评分核心 `src/bise/retrieval/system/scoring.py` 逐候选计算向量点积。仓库有独立 FAISS 封装，但不能仅据此断言当前系统入口实际经过 FAISS；中期报告的索引描述需要按最终实现核实。
- 语义特征在建库时会在同 scene 的 camera 记录之间共享。做“未见相机”实验时需明确查询是否只使用自身视频生成的语义，避免借用了其他视角信息。
- 未发现正式的融合权重搜索入口；当前有固定权重与七组合均匀消融配置。

## 5. 当前可以继承和暂时不能继承的内容

可以继承：三模块结构、训练/标注/检索入口、历史消融的研究问题、存留模型作为候选基线、产物保存和绘图的基础能力。

不能直接继承：中期最优数字作为新主表、现存 cfg2 系统评估作为 held-out 证据、cfg3 指标自动解释为纯机器人型号泛化、轨迹“已经物理对齐”的说法、任意单一 loss/网络改造已具创新性的说法。

本轮保留了工作区已有的 `AGENTS_CN.md` 修改与 `PROJECT_STATUS_PLAN.md` 删除状态。未恢复或覆盖这些用户变更。未运行模型训练与全量测试；本轮验证是文件盘点、split 交集检查、两项指标复算与 embedding/matrix 一致性检查。
