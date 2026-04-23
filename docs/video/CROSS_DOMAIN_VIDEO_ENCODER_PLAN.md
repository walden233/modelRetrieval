# 跨领域视频编码器完整实现规划

更新时间：2026-04-22

本文档仅聚焦视频模态，不涉及视频与轨迹、语义等其他模态的联合训练或联合检索。目标是把当前仓库中已有的视频原型，收敛为一个可复现、可评估、可扩展的跨领域视频编码器研究与工程闭环。

## 1. 文档目标

本文档回答五个问题：

1. 当前仓库里视频侧已经做到什么程度。
2. 现有实现为什么还不足以支撑“跨领域视频编码器”主线。
3. 接下来应如何补齐数据、模型、训练、评估和产物沉淀。
4. 视频单模态实验应该如何设计，评估哪些指标。
5. 最终结果应该以什么形式沉淀到论文、汇报和仓库中。

## 2. 当前视频侧现状审计

基于当前代码，视频模态已经有最小可运行骨架，但离稳定研究主线还有明显距离。

### 2.1 已有基础

当前已存在以下模块：

- `src/bise/data/rh20t/video_dataset.py`
  负责从 RH20T 按 task -> scene -> camera 采样视频对。
- `src/bise/data/whirl/video_pair_dataset.py`
  负责从 CSV 读取人类视频与机器人视频配对样本。
- `src/bise/modalities/video/models/videomae_adapter.py`
  已有 VideoMAE backbone 适配器。
- `src/bise/modalities/video/models/vjepa_adapter.py`
  已有 V-JEPA backbone 适配器。
- `src/bise/modalities/video/losses.py`
  已有最基础的双向 InfoNCE 损失。
- `src/bise/modalities/video/trainer.py`
  已有单 epoch 训练循环。
- `src/bise/modalities/video/evaluator.py`
  已有基础检索评估入口。
- `tools/train_video.py`
  已有基于 JSON 配置的训练入口。
- `configs/video/vjepa_whirl.json`
  已有一个 V-JEPA + WHIRL 的最小配置样例。

结论：

- 当前视频侧不是空白。
- 当前视频侧更接近“backbone 试跑原型”，还不是“跨领域视频编码器完整主线”。

### 2.2 当前主要问题

| 问题 | 当前表现 | 影响 |
| :--- | :--- | :--- |
| 样本定义不统一 | RH20T 与 WHIRL 数据集返回字段不一致 | 训练器、评估器和导出工具难以共用 |
| batch 约定不一致 | `trainer.py` 按 dense tensor 处理，`evaluator.py` 按 list 处理 | 训练与评估逻辑不能共享，容易出现隐藏 bug |
| 正样本定义过弱 | 目前只有一对一 InfoNCE | 不能表达“同任务不同 scene/camera 也是正样本” |
| 没有验证集闭环 | `tools/train_video.py` 只看训练 loss | 无法选模型、无法比较配置、无法复现实验结论 |
| 没有场景级/任务级双视角评估 | 现有评估依赖固定 `group_size` | 不适合 RH20T 这种 task/scene 多层级结构 |
| 没有统一实验产物沉淀 | 训练曲线、指标、embedding、案例图没有完整输出 | 论文材料和调参过程无法系统复盘 |
| 没有视频侧测试 | 当前测试几乎都在轨迹和语义侧 | 视频模块重构风险高 |

### 2.3 对现有实现的具体判断

1. `src/bise/data/rh20t/video_dataset.py`

- 当前将 scene 抽样、camera 抽样、视频解码、processor 编码都混在一个 `Dataset` 里。
- 返回值是变长 list，且通过递归方式跳过空样本。
- 这会导致 batch 结构不稳定，也会引入隐式采样偏置。

2. `src/bise/data/whirl/video_pair_dataset.py`

- 当前只返回 `human_pixel_values` 与 `robot_pixel_values`。
- 没有 `sample_id`、`task_id`、`scene_id`、`pair_id`、`domain` 等关键标识。
- 这不利于评估、可视化、错误分析和 embedding 导出。

3. `src/bise/modalities/video/trainer.py`

- 只支持单一训练损失。
- 没有 AMP、梯度裁剪、梯度累积、冻结策略、日志细分。
- 没有 train/val 分离，也没有 best checkpoint 的评估依据。

4. `src/bise/modalities/video/evaluator.py`

- 当前逻辑和训练使用的 batch 格式不一致。
- 评估依赖固定 `group_size`，更适合均匀分组，不适合真实 task/scene 检索。
- 没有双方向评估，也没有任务级与 scene 级分开统计。

5. `tools/train_video.py`

- 当前只接了 WHIRL CSV 数据。
- 没有 split、seed、resume、checkpoint 选择、验证集评估、early stopping。
- 还没有形成 RH20T 主线实验入口。

## 3. 视频主线的目标定义

### 3.1 研究目标

训练一个跨领域视频编码器 `E_vid`，将：

- 人类演示视频 `V_h`
- 机器人执行视频 `V_r`

编码到同一向量空间，使得：

- 同一任务的人类视频和机器人视频距离更近。
- 不同任务的视频距离更远。
- 编码器对 scene、camera、背景和执行风格变化保持一定鲁棒性。

### 3.2 本阶段不做的事情

本计划明确不包括：

- 视频与轨迹联合训练。
- 视频与语义联合训练。
- 多模态融合排序。
- 在线检索 API。
- 生产部署。

### 3.3 本阶段交付物

本阶段必须产出：

- 稳定的视频数据读取与 batch 处理代码。
- 可配置的跨领域视频编码器训练与评估入口。
- 至少一条 RH20T 主训练链路和一条 WHIRL 小规模 sanity 链路。
- 视频单模态检索指标、消融实验和案例分析。
- 可复现的实验产物归档。

## 4. 推荐技术路线

### 4.1 总体原则

视频侧优先采用“共享预训练视频 backbone + 轻量跨领域适配层 + 对比学习”的路线。

原因：

- 当前仓库已经有 VideoMAE 与 V-JEPA 适配器，继续沿用成本最低。
- 数据规模相对有限，不适合一开始就上双塔全量独立 backbone。
- 跨领域问题主要是人类视频和机器人视频的视觉风格差异，先用轻量 domain-aware 头部即可。

### 4.2 推荐主架构

建议采用以下结构：

`video -> frame sampler -> processor -> shared backbone -> domain adapter -> shared projector -> normalized embedding`

说明：

- `shared backbone`
  使用同一个预训练视频 backbone 处理 human / robot 两个域的视频。
- `domain adapter`
  在 backbone 输出后插入轻量域适配层，允许 human / robot 在共享空间前做小幅校正。
- `shared projector`
  最终仍投影到统一 embedding 空间，保证检索一致性。

### 4.3 推荐 backbone 策略

建议分两条线并行，但主报告只保留最稳的一条：

- 主线：`V-JEPA`
  理由是对表观变化通常更稳，适合作为跨领域视频编码主模型。
- 对照线：`VideoMAE`
  作为强 baseline，用于验证 backbone 选择是否真的带来收益。

### 4.4 推荐编码器形式

建议提供三种模式，但只把前两种作为正式实验对象：

1. `shared`

- backbone 共享
- projector 共享
- 最简单，作为 baseline

2. `dual_head`

- backbone 共享
- human / robot 使用不同 adapter head
- 最终共享投影维度
- 这是推荐默认模式

3. `dual_encoder`

- human / robot 各自一套 backbone
- 参数量大，仅作为后续增强项

### 4.5 推荐投影头

推荐将现有两层 MLP 投影头升级为：

- `LayerNorm`
- `Linear(hidden, hidden)`
- `GELU`
- `Dropout`
- `Linear(hidden, feature_dim)`
- `L2 normalize`

原因：

- 比当前简单 `Linear + ReLU + Linear` 更稳。
- 便于做冻结 backbone 时的轻量微调。

## 5. 数据与样本组织方案

### 5.1 统一样本单位

训练与评估的基本单位应统一为“一个跨域视频对”，字段至少包括：

- `sample_id`
- `pair_id`
- `dataset_name`
- `task_id`
- `scene_id`
- `camera_id`
- `human_video_path`
- `robot_video_path`
- `query_domain`
- `gallery_domain`

说明：

- RH20T 与 WHIRL 都应映射到这套 schema。
- 这样训练、评估、导出、错误分析才能共用代码。

### 5.2 正样本定义

需要同时支持两种正样本定义：

1. `scene-positive`

- 同一 scene 的 human / robot 视频为正样本
- 更严格，适合 sanity check

2. `task-positive`

- 同一 task 下不同 scene / camera 的 human / robot 视频均视为正样本
- 更符合最终检索目标

推荐：

- 训练阶段默认使用 `task-positive`
- 评估阶段同时报告 `scene-level` 和 `task-level`

### 5.3 视频采样策略

建议独立出统一的采样模块，支持：

- `uniform`
  全视频均匀采样
- `segment_random`
  先分段，再每段随机取帧
- `stride`
  固定步长采样
- `center_clip`
  中心片段采样，主要用于评估

推荐默认：

- 训练：`segment_random`
- 验证 / 测试：`uniform`

### 5.4 数据增强策略

建议将增强限制在对机器人操作语义破坏较小的范围内：

- `RandomResizedCrop`
- `ColorJitter`，幅度轻
- `TemporalJitter`
- `GaussianBlur`，可选
- `HorizontalFlip` 默认关闭

原因：

- 左右翻转可能破坏手部朝向、相机布局和机械臂执行习惯，不宜默认启用。

## 6. 模型与损失设计

### 6.1 编码器设计

建议新增统一模型类：

- `CrossDomainVideoEncoder`

其职责：

- 封装 backbone registry
- 封装 `shared / dual_head / dual_encoder` 三种模式
- 管理冻结策略
- 输出 human / robot embedding
- 导出中间特征用于可视化

### 6.2 损失函数设计

视频侧建议分三级损失：

1. 基础损失：双向 InfoNCE

- 作为最小可行版本
- 对应当前 `InfoNCELoss` 的升级版

2. 主损失：多正样本对比损失

- 同 task 的多个 robot video 都可作为 human query 的正样本
- 同时支持 human -> robot 与 robot -> human
- 这是正式实验默认损失

3. 辅助损失：模态内增强一致性损失

- 同一个 human video 的两种增强视图保持接近
- 同一个 robot video 的两种增强视图保持接近
- 作为第二阶段增强项

推荐总损失：

`L = L_inter + lambda_intra * L_intra`

推荐默认：

- 第一阶段先只开 `L_inter`
- 第二阶段再加 `L_intra`

### 6.3 训练策略

建议采用三阶段推进：

1. `P0`

- 冻结 backbone
- 只训练 projector / adapter
- 目标是先验证数据与评估链路

2. `P1`

- 解冻最后 `N` 个 transformer block
- 训练 `dual_head + task-positive contrastive`
- 这是主实验阶段

3. `P2`

- 加入增强一致性损失
- 做 hard negative 与采样策略消融

### 6.4 冻结策略

配置中至少支持：

- `freeze_backbone = true/false`
- `unfreeze_last_n_blocks`
- `freeze_patch_embed`
- `freeze_norm_layers`

原因：

- 视频 backbone 参数量大，必须有清晰的冻结控制，否则调参成本过高。

## 7. 代码实现规划

本节给出文件级实现方案。优先复用现有结构，不另起一套目录体系。

### 7.1 需要新增的文件

建议新增：

- `src/bise/modalities/video/batch.py`
  统一视频 batch schema 与 `collate_video_pairs`
- `src/bise/modalities/video/frame_sampling.py`
  统一帧采样策略
- `src/bise/modalities/video/transforms.py`
  统一训练 / 评估视频增强
- `src/bise/modalities/video/models/cross_domain_video_encoder.py`
  统一封装 backbone、adapter、projector 和冻结策略
- `src/bise/modalities/video/models/backbone_registry.py`
  统一创建 VideoMAE / V-JEPA backbone 与 processor
- `tools/evaluate_video.py`
  单模态视频评估入口
- `tools/export_video_embeddings.py`
  导出 embedding、相似度矩阵和案例分析文件
- `tests/test_video_batch.py`
- `tests/test_video_losses.py`
- `tests/test_video_evaluator.py`
- `tests/test_video_train_config.py`

### 7.2 需要重构的现有文件

1. `src/bise/data/rh20t/video_dataset.py`

改造目标：

- 从“变长 list + 内嵌随机采样”改成“稳定 pair sample 输出”
- 显式返回 `task_id / scene_id / pair_id / camera_id`
- 支持 `scene-positive` 与 `task-positive`
- 支持按 split manifest 加载，而不是在 `Dataset` 内隐式随机切分

2. `src/bise/data/whirl/video_pair_dataset.py`

改造目标：

- 对齐 RH20T 的统一 sample schema
- 增加 `task_id / pair_id / dataset_name`
- 支持 debug 模式和 deterministic sampling

3. `src/bise/modalities/video/models/videomae_adapter.py`

改造目标：

- 改成只负责 backbone-specific encode 逻辑
- 把 projector 从 adapter 中剥离到统一模型类

4. `src/bise/modalities/video/models/vjepa_adapter.py`

改造目标：

- 同上，适配到统一模型接口

5. `src/bise/modalities/video/losses.py`

改造目标：

- 保留基础 InfoNCE
- 新增 `multi_positive_video_contrastive_loss`
- 新增 `intra_domain_consistency_loss`
- 支持 scene/task label mask

6. `src/bise/modalities/video/trainer.py`

改造目标：

- 增加 train / val 两套循环
- 支持 AMP、梯度裁剪、梯度累积
- 记录 `loss_inter`、`loss_intra`、`val_mrr`、`val_ndcg`
- 输出 best checkpoint 与 last checkpoint

7. `src/bise/modalities/video/evaluator.py`

改造目标：

- 基于显式 label 评估，而不是固定 `group_size`
- 支持 human -> robot 与 robot -> human 双方向
- 同时输出 scene-level 和 task-level 指标
- 导出 similarity matrix 与错误案例索引

8. `tools/train_video.py`

改造目标：

- 支持 RH20T / WHIRL 两类数据源
- 支持 train / val / test split
- 支持 resume
- 支持从 config 选择 backbone、loss、冻结策略和采样策略
- 训练结束后自动跑验证与测试

### 7.3 推荐配置文件

建议新增：

- `configs/video/vjepa_rh20t_baseline.json`
- `configs/video/vjepa_rh20t_task_positive.json`
- `configs/video/vjepa_rh20t_task_positive_intra.json`
- `configs/video/videomae_rh20t_baseline.json`
- `configs/video/vjepa_whirl_debug.json`

建议统一配置字段：

- `dataset.type`
- `dataset.root_dir`
- `dataset.csv_path`
- `dataset.split_manifest`
- `dataset.num_frames`
- `dataset.sampling_strategy`
- `dataset.train_augmentations`
- `model.backbone_name`
- `model.backbone_type`
- `model.encoder_mode`
- `model.feature_dim`
- `model.dropout`
- `optimization.learning_rate`
- `optimization.weight_decay`
- `optimization.scheduler`
- `optimization.warmup_ratio`
- `optimization.grad_clip_norm`
- `training.batch_size`
- `training.num_epochs`
- `training.seed`
- `training.amp`
- `training.gradient_accumulation_steps`
- `loss.positive_level`
- `loss.temperature`
- `loss.lambda_intra`
- `evaluation.metrics`
- `output_dir`

## 8. 分阶段实施顺序

### 8.1 第一阶段：打通稳定 baseline

目标：

- 让视频训练、验证、测试闭环先稳定运行

必须完成：

- 统一数据 schema
- 统一 collate
- 统一 train/eval batch 约定
- RH20T 和 WHIRL 两类数据集均可读取
- `evaluate_video.py` 输出基础指标

验收标准：

- 至少一条 `WHIRL debug` 配置可以完整训练并评估
- 至少一条 `RH20T baseline` 配置可以完整训练并评估

### 8.2 第二阶段：形成正式主实验模型

目标：

- 将主线切换到 `task-positive` 跨领域检索

必须完成：

- `dual_head` 模式
- 多正样本对比损失
- 双方向检索评估
- scene-level / task-level 双指标

验收标准：

- 训练日志和 `best_metrics.json` 能明确区分 scene/task 两套结果
- 主模型相对 frozen baseline 有稳定提升

### 8.3 第三阶段：做增强与消融

目标：

- 形成论文和汇报需要的关键对照实验

必须完成：

- backbone 对照
- 采样策略对照
- 冻结策略对照
- 损失项对照
- 增强策略对照

验收标准：

- 每个消融项都有独立配置和结果表
- 结果可直接汇总成论文表格

### 8.4 第四阶段：沉淀可视化与案例分析

目标：

- 输出最终图表与定性案例

必须完成：

- 相似度矩阵热图
- embedding 降维图
- Top-K 检索案例图
- 失败案例归因

验收标准：

- `artifacts/figures/video/` 下能生成论文可用图

## 9. 实验设计

### 9.1 数据集使用策略

建议分为两层：

1. `WHIRL`

- 用作 debug 和快速 sanity check
- 样本少，适合快速验证配置、损失和训练入口是否正常

2. `RH20T`

- 用作正式主实验
- 重点验证跨 task、跨 scene、跨 camera 的视频检索能力

### 9.2 数据划分建议

建议至少准备三套划分：

1. `Split-A: Scene-disjoint`

- 同一 task 下按 scene 划分 train / val / test
- 评估跨场景鲁棒性

2. `Split-B: Camera-disjoint`

- 训练和测试使用不同 camera
- 评估跨视角鲁棒性

3. `Split-C: Task-held-out`

- 留出部分 task 只在测试中出现
- 评估任务级泛化能力

推荐主结果使用：

- `Split-A` 作为主表
- `Split-B` 作为泛化补充表
- `Split-C` 作为难度更高的扩展实验

### 9.3 baseline 设置

建议至少包含以下 baseline：

1. `Frozen Shared Backbone + InfoNCE`

- 最小可行基线

2. `Shared Backbone Partial Finetune + InfoNCE`

- 验证微调是否有效

3. `Dual-Head + Multi-Positive Contrastive`

- 正式主模型

4. `Dual-Head + Multi-Positive + Intra Consistency`

- 主模型增强版

### 9.4 消融实验

建议做以下消融：

- backbone: `VideoMAE` vs `V-JEPA`
- encoder mode: `shared` vs `dual_head`
- positive level: `scene-positive` vs `task-positive`
- frame sampler: `uniform` vs `segment_random`
- num frames: `8` vs `16` vs `32`
- freeze strategy: `frozen` vs `unfreeze_last_2` vs `unfreeze_last_4`
- loss: `InfoNCE only` vs `InfoNCE + intra consistency`
- augmentation: `off` vs `light augment`

### 9.5 训练稳定性实验

建议记录但不一定放主文：

- 不同随机种子 `3` 次重复实验
- 指标均值与标准差

原因：

- 视频检索对 batch 采样和初始化较敏感，单次结果说服力不足。

## 10. 评价指标

### 10.1 主指标

正式报告建议至少包含：

- `Recall@1`
- `Recall@5`
- `Recall@10`
- `MRR`
- `Mean Rank`
- `Mean Percentage Rank`
- `NDCG@10`

### 10.2 评估维度

所有主指标都建议从四个维度报告：

- `human -> robot, scene-level`
- `human -> robot, task-level`
- `robot -> human, scene-level`
- `robot -> human, task-level`

说明：

- 若只报单方向，跨领域检索结论不完整。
- task-level 是主指标，scene-level 是更严格的补充指标。

### 10.3 辅助指标

建议补充：

- `intra-domain similarity mean`
- `cross-domain positive similarity mean`
- `cross-domain negative similarity mean`
- `positive-negative margin`
- 推理吞吐量
- 单卡显存占用
- 参数量

### 10.4 模型选择指标

建议以：

- `task-level human->robot MRR`

作为 best checkpoint 选择主指标，原因是：

- 比只看训练 loss 更可靠。
- 比 R@1 更平滑。
- 更符合检索任务整体排序目标。

## 11. 最终结果呈现方式

### 11.1 论文主表

建议主表结构如下：

| Model | Backbone | Encoder Mode | Positive Level | R@1 | R@5 | R@10 | MRR | NDCG@10 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |

主表仅放：

- `human -> robot, task-level`
- RH20T 主 split 结果

### 11.2 泛化表

建议单独一张表呈现：

- `Scene-disjoint`
- `Camera-disjoint`
- `Task-held-out`

用于说明模型是否真的学到任务语义，而不是记住场景外观。

### 11.3 消融表

建议把以下因素独立成表：

- backbone
- encoder mode
- positive definition
- loss terms
- sampling strategy

### 11.4 图形结果

建议输出四类图：

- 训练曲线图
- similarity heatmap
- embedding 的 UMAP / t-SNE 图
- Top-K 检索案例拼图

### 11.5 定性案例

每次正式实验至少选三类案例：

- 成功匹配案例
- 近义任务混淆案例
- 跨视角失败案例

分析内容建议包括：

- 查询视频关键帧
- Top-5 检索结果
- 相似度分数
- 失败原因归因

## 12. 产物归档规范

建议每次正式 run 输出到：

- `artifacts/runs/video/<run_name>/`

至少包括：

- `params.json`
- `best_metrics.json`
- `test_metrics.json`
- `curves.png`
- `similarity_matrix.npy`
- `retrieval_cases.json`
- `embeddings.json`

建议图形输出到：

- `artifacts/figures/video/`

## 13. 风险与应对

### 13.1 数据量不足导致过拟合

应对：

- 先冻结 backbone
- 增加轻量增强
- 做 split 更严格的验证

### 13.2 人类视频与机器人视频外观差异过大

应对：

- 默认采用 `dual_head`
- 优先做 task-positive 训练
- 保留 scene-level 指标做过拟合监控

### 13.3 训练成本过高

应对：

- 从 `8/16` 帧开始
- 先做 partial finetune
- 只在主模型上做高成本配置

### 13.4 结果不稳定

应对：

- 固定 split manifest
- 固定随机种子
- 记录 3 次重复实验均值和标准差

## 14. 推荐执行顺序

建议按以下顺序推进代码实现：

1. 先重构数据集、batch 和评估接口，统一 sample schema。
2. 再实现统一 `CrossDomainVideoEncoder` 和多配置训练入口。
3. 然后完成 task-positive 对比损失和双方向评估。
4. 最后补消融、可视化和案例导出工具。

原因：

- 如果没有统一数据与评估协议，后续所有实验都会反复返工。

## 15. 本阶段完成标准

满足以下条件，才算“跨领域视频编码器代码与实验主线完成”：

- RH20T 与 WHIRL 都能通过统一入口完成训练和评估。
- 至少有一条正式 RH20T 主实验配置可复现。
- 能输出 scene-level 与 task-level 双指标。
- 能输出 human -> robot 与 robot -> human 双方向结果。
- 至少完成一组 backbone 对照和一组损失消融。
- 能导出论文可用的曲线、热图和案例分析。
- 有对应的视频模块测试，覆盖数据、损失、评估和配置加载。

## 16. 建议的下一步

建议下一步直接进入代码实现，不再继续扩展方案范围。实现顺序建议从下面四项开始：

1. 重构 `RH20TVideoDataset` 和 `VideoPairDataset` 的统一 sample schema。
2. 新增 `batch.py`、`frame_sampling.py`、`cross_domain_video_encoder.py`。
3. 重写 `tools/train_video.py` 与新增 `tools/evaluate_video.py`。
4. 补上视频侧测试和第一组 `WHIRL debug + RH20T baseline` 配置。
