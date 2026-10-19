# 视频跨相机匹配：实验与论文叙事

记录日期：2026-10-10；本机V系列更新：2026-10-19。

本文提取讨论的关键结论。逐视频标注与教师生成见 [语义实现方案](../semantic/PLAN.md)。已实现组织多视角采样、batch内关系及文本库蒸馏；2026-10-19本机V1/V2改用跨batch视频关系队列，配置与命令见第4节。32GB配置不受本次调整影响。

2026-10-17 补充：scene 教师来源与导出接口以 [第三章实现方案](../semantic/PLAN.md#12-scene-级汇总与训练教师) 为准。复用第三章通过 --source 选定的一种逐视频语义表示（默认S5结构），再做只返回任务描述的scene级文本汇总；教师池与训练batch无关，读取时过滤实际train；当前默认每 scene 两个相机、四段视频，训练使用范围服从对应 run 的划分。

当前已实现的基础矩阵为 E0–E7，基线是 **E3：冻结 V-JEPA2＋双适配头＋task 多正样本＋模态内增强**。
s44 视频已按该编号迁移并复用原权重与评估结果，设置与映射见
[当前矩阵和迁移记录](../RETRAIN.md#当前实验矩阵)。2026-10-11 已将两模态的模态内损失
统一为自身增强 1、同 scene 不同相机 0.5、同 task 不同 scene 0.25，逐行归一化。
组织采样与语义监督通过显式配置启用；原有随机采样实验保持原行为。
历史基础矩阵以E3为基线；本轮跨相机补充实验按用户选择以 **E5** 为起点，
三组实验复用实际E5 run的划分，以其结构和优化参数为起点；本次统一增大batch到8，不把旧结果当成扩展方法结果。
当前检查的s43/s44 E5实际params.json均为解冻最后1个block，应以实际产物为准，
不能直接套用文档中“最后两个block”的概括。

## 1. 问题与研究目标

当前观察：视频跨相机后，即使属于同一个 task 或 scene，相似度也可能很低。论文允许报告部分改善和失败边界，不预设一定解决，也不预设一定失败。

研究问题：在有限数据和微调预算下，扩展模态内一致性，以及进一步引入语义弱监督，能否缓解跨视角人机视频匹配退化？改善发生在任务语义、操作实例，还是动作阶段？

| 层次 | 目标 | 主要困难 |
|---|---|---|
| 任务语义 | 不同视角的相同任务能够匹配 | 背景、外观和视角差异 |
| 操作实例 | 同一个 scene 的不同视角能够对应 | 遮挡、操作细节和可见性差异 |
| 动作阶段 | 不同视角的接近、抓取、移动、释放能够对应 | 时序错位、执行速度、阶段不可见 |

余弦相似度绝对值低不等于检索失败；必须同时检查正负样本间隔、排名和严格跨相机检索指标。

## 2. 当前监督已经覆盖什么

这里的“域”指人类域和机器人域，两者都属于视频模态。

设同一 scene 的两个相机视频为 `H_A、H_B、R_A、R_B`。当前按 task 定义的跨域多正样本损失，在相关视频同时进入 batch 时，已经直接约束：

- `H_A ↔ R_A`、`H_A ↔ R_B`。
- `H_B ↔ R_A`、`H_B ↔ R_B`。

因此，不能把后续改进描述为“首次加入跨视角监督”。现有监督已经包含跨人机、跨相机的正样本。

修正前的模态内损失仅匹配同一视频的增强，没有直接将 `H_A ↔ H_B` 和 `R_A ↔ R_B`
作为正样本。共同的跨域正样本只能间接促进同域视角一致。
2026-10-11 起，上述同 scene 跨相机关系在同 batch 内以 0.5 权重直接参与模态内监督；
同 task 不同 scene 以 0.25 权重参与，自身增强权重为 1，不同 task 仍为负样本。

原E5训练随机打乱样本，batch 为4，没有保证同一个scene的两个相机同时进入batch。本机新V0/V2显式启用组织采样，V1保持随机采样。数据集具有跨相机配对，不代表每一步都使用了这些关系；梯度累积也不会自动扩大对比损失的候选集合。

实现参考：[共用模态内损失](../../src/bise/common/losses.py)、[跨域损失](../../src/bise/modalities/video/losses.py)、[训练器](../../src/bise/modalities/video/trainer.py)、[训练采样](../../runs/video/train_video.py)。

## 3. 扩展模态内一致性的准确含义

扩展模态内一致性沿用现有思路，将正样本从“同一视频的两次增强”扩展到“同一次执行的真实不同相机视频及其增强”。不必另建一个独立训练模块。

| 设置 | 同视频的两次增强 | 同次执行的不同相机 |
|---|---|---|
| 修正前的模态内损失 | 拉近 | 同 batch 时作为负样本推远 |
| 仅屏蔽误负样本 | 拉近 | 忽略，不拉近也不推远 |
| 当前加权模态内一致性 | 权重 1 | 同 batch 时作为权重 0.5 的正样本 |

“屏蔽负样本”和“加入正样本”不同，但属于同一损失设计的消融，不应包装成两项方法创新。本轮最终不再单独新增“仅屏蔽”的训练实验。

本机V0/V2使用`scene_pairs`：每batch选择4个不同task，各取一个scene的两个相机，共8组人机配对（16段原始视频）。同一样本内的人机视频仍来自同一相机。
每epoch覆盖全部train scene；不足整batch时从其它train task补齐双相机scene，少量scene可能重复，不使用val/test。两组同seed时采样顺序一致；V1随机打乱全部train配对。
V0/V2将同scene跨相机正样本权重从0.5提高到1.0，与自身增强同权；同task不同scene仍为0.25，但本组织采样不会在同一batch中出现该关系。V1保留0.5和0.25，相关样本同batch时照常生效。

增加的直接关系为：

- 同一次人类执行：`H_A ↔ H_B`。
- 同一次机器人执行：`R_A ↔ R_B`。

不同相机的视频应对应同一执行及相应时间区间。人类示范与机器人执行不是天然逐帧同步，不能直接按相同帧序号做阶段对齐。

## 4. 新实验与完整入口

| 实验 | 采样 | 同scene模态内权重 | 语义监督 |
|---|---|---|---|
| V0 | 4 task × 1 scene × 2相机 | 1.0 | 无 |
| V1 | random | 0.5 | 跨batch语义关系KL，缓存64对视频 |
| V2 | 同V0 | 1.0 | 与V1相同 |

三组都从项目内V-JEPA2预训练权重从头训练，不读取已训练E5的best_model。共同设置为双适配头＋共享128维检索投影、微调最后1个block、16帧、训练/评估batch=8、5epoch、学习率1e-5、模态内权重0.2，按验证task MRR选择best。V1的“仅语义”指只额外增加语义监督，原有跨域和增强损失仍保留；不加入V0的组织采样和增权。
本机V1/V2不再增加128→1024语义头，关系损失直接作用于最终检索向量；三组模型结构和初始化相同。与旧E5比较时，batch扩大也是变量；若需严格分解增益，应补充同batch=8、随机采样、无语义的E5对照，不能把全部变化都归因于语义。

2026-10-19在本机RTX 4060 Laptop 8GB上使用真实预训练骨干、合成输入完成显存压力测试：V0两次优化器更新通过；V2九次更新通过，队列达到64对、候选达到72，PyTorch峰值allocated为6.59GiB、reserved为6.95GiB。该数值不包含桌面等外部占用，余量较紧；这是运行与显存验证，不是检索性能实验。未启动正式重训或覆盖现有run。

配置目录为`configs/retrain/views/`，默认新batch为`v44`，结果在`artifacts/runs/retrain/v44/video/V0|V1|V2/`。`manifest_batch: s44`复用现有清单与冻结相机，不生成第二份公共清单；三组train/val/test直接复用`s44/video/E5/split_manifest.json`。E5实际train/val与s44公共清单不同，但test相同，因此不能只引用公共清单重新划分。每个新run仍保存自己的划分快照。

```bash
conda activate torch2
# 完整准备、审计、三组从头训练、cfg2/cfg3评估与绘图
python runs/retrain.py --config configs/retrain/views/run.json --batch v44 --modality video

# 独立环节；train会先准备完所有选定run的元信息与教师快照，再依次训练
python runs/retrain.py --config configs/retrain/views/run.json --batch v44 --modality video --stage prepare
python runs/retrain.py --config configs/retrain/views/run.json --batch v44 --modality video --stage audit
python runs/retrain.py --config configs/retrain/views/run.json --batch v44 --modality video --stage train
python runs/retrain.py --config configs/retrain/views/run.json --batch v44 --modality video --stage eval
python runs/retrain.py --config configs/retrain/views/run.json --batch v44 --modality video --stage charts
python runs/redraw.py --batch v44

# 只重新训练、评估V1；旧V1会归档，V0/V2不重训
python runs/retrain.py --config configs/retrain/views/run.json --batch v44 --modality video --stage train --only V1
python runs/retrain.py --config configs/retrain/views/run.json --batch v44 --modality video --stage eval --only V1
# 续跑整条流水线：已完成且best存在的训练复用，选定评估仍重新计算
python runs/retrain.py --config configs/retrain/views/run.json --batch v44 --modality video --skip-existing
```

训练只存best，不存last；`--skip-existing`仅复用完成的run，未完成run会归档后从头训练，不是逐step断点恢复。纯重绘用`runs/redraw.py`，`--stage charts`还会从已有评估重新汇总比较数据。

V1/V2语义权重改为0.5、教师/学生温度均0.1、首个epoch逐step线性升权；这些是起始配置，可在各自JSON中修改。教师快照为每run的`semantic_teachers.npz`，仅包含有效train scene的文本、归一化向量、身份与来源信息。准备后`params.json:loss.semantic`记录快照路径和覆盖情况，不新增单独的统计JSON。

记现有跨域损失为 `L_inter`，扩展后的模态内损失为 `L_intra_view`：

```text
V0：L = L_inter + λ_intra × L_intra_view
V1：L = L_inter + λ_intra × L_intra_original + λ_sem × L_sem
V2：L = L_inter + λ_intra × L_intra_view + λ_sem × L_sem
```

三个实验的纯视频评估均自动生成主检索指标、相机一致性三联图、相机级和scene最大值热图，比较表与柱状图进入batch下的`video/charts/`。

## 5. 推荐的语义监督粒度

训练以scene级、视角无关的操作语义为主。同一scene的人类与机器人、不同相机共享教师。本轮V1/V2选用`artifacts/semantic/v2/teachers/S3/cfg2.npz`，来源为逐视频S3经文本模型汇总的共享描述，无需重新标视频。读取时只保留实际train scene；不可用目标屏蔽语义损失，不删除视频样本或回退task描述。2026-10-18检查：全池401/428有效，实际E5训练覆盖241/258，另17个scene只参与原有损失；以后覆盖量以准备时写入params的快照为准。

| 粒度 | 优点 | 局限 | 本轮建议 |
|---|---|---|---|
| task 级 | 成本低，所有视角目标一致 | 无法区分同 task 的 scene，与 task 标签部分重复 | 可选补充，需显式区分，不默认回退 |
| scene 级共享描述 | 保留具体操作信息，避免描述随视角变化 | 泛化描述可能退化为 task 级 | 主方案 |
| 每个视频独立描述 | 能记录该视角的可见细节 | 遮挡、措辞和幻觉可能重新引入视角偏差 | 复用第三章结果作为汇总来源，不直接当共享教师 |

“scene 级”描述动作、交互对象、状态变化、动作顺序和目标，不是描述桌面、背景颜色或相机位置。不可见细节不补造，也不能仅因为两个相机措辞不同就将其判为负样本。

### 已有描述的复用

以下为此前讨论时检查的 [cfg2 旧标注](../../artifacts/semantic/rh20t/cfg2/annotations/normalized_annotations.jsonl)，是历史覆盖情况，不代表第三章新流水线的完成量：

- 436 条记录，109 个 task，218 个 task–scene 组合。
- 每个组合分别有人类和机器人描述。
- 记录中的 `cam_id` 只有一种；这描述的是现有标注字段，不代表数据集只有一个相机。

现有标注粒度已经比“每个 task 一份描述”更细，可作为整理来源；不能据此认定覆盖了全部训练 scene。

```text
同 scene 的已有人类描述、机器人描述
                  ↓ 核验并统一动作、对象关系和目标
             scene 操作描述
                  ↓ 冻结文本编码器
             scene 文本向量
                  ↓
     同 scene 的不同相机视频共享使用
```

使用前必须核对标注的 task/scene 与当前训练清单的身份映射，不能按文件顺序配对。人机描述矛盾时先核验；训练教师只使用训练划分的内容，不能借用验证或测试视频补充训练描述。

## 6. 语义监督如何参与训练

### 本机V1/V2：直接约束检索空间的跨batch关系

缓存最近64组有效教师对应的人机检索向量及scene身份，约8个batch。保存的是detach后的128维向量，不保存旧视频、骨干中间激活或计算图。当前batch最多8个有效配对，加上队列最多得到72个候选；教师缺失时实际数量会减少。
队列每epoch清空，先进先出，限制旧特征随训练漂移；不会新增视频编码或调用文本大模型。

设当前有效查询为i，当前batch及队列候选为j，scene教师为t，归一化视频向量为h、r：

```text
Q_ij = softmax_j(cos(t_i, t_j) / 0.1)
P^HR_ij = softmax_j(cos(h_i, r_j) / 0.1)
P^RH_ij = softmax_j(cos(r_i, h_j) / 0.1)
L_sem = [KL(Q || P^HR) + KL(Q || P^RH)] / 2
L_total = L_inter + 0.2 × L_intra + ramp(epoch) × 0.5 × L_sem
```

梯度直接经过当前128维检索向量、共享投影、双适配头和最后一个骨干block；旧候选只作为固定参照。这样监督不会仅被额外的语义头拟合，也不局限于旧batch4中的两个scene。缓存是跨batch候选集合，不等价于对全部历史视频联合反向传播；梯度累积也不等价于此队列。
缺失教师的当前查询及候选均排除；队列和当前batch合起来不足两个不同scene时跳过语义损失，当前有效样本仍入队，原有视频损失照常计算。
方法借鉴[Cross-Batch Memory](https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_Cross-Batch_Memory_for_Embedding_Learning_CVPR_2020_paper.html)的特征缓存，以及[ReSSL](https://arxiv.org/abs/2107.09282)的关系分布匹配；这里用冻结scene文本提供教师关系，并非原论文方法的完整复现。

`curves.json`和epoch日志保留语义原始损失、覆盖率、教师熵、实际权重，新增：

- `semantic_candidates`：平均每次计算的有效候选数，检查队列是否扩展了关系范围。
- `semantic_weighted_loss`：加入总损失的语义项实际数值，包含预热权重。
- `semantic_gradient_ratio`：加权语义损失与跨域损失对当前检索向量的梯度范数比；这是作用强度诊断，不代表检索收益，且不包含模态内支路梯度。

若覆盖正常、候选数扩大且梯度非零，只能说明监督确实到达检索表示；效果仍须由验证集选择best及独立cfg2/cfg3纯视频评估确认，不保证增权一定改善指标。

### 保留的兼容模式

`mode=batch`仍为原双向batch内关系KL；`mode=bank`仍为共享线性语义头＋全部有效train文本库软关系KL。原有数学实现和默认参数不变，`video32`继续使用bank配置，本机V1/V2显式使用新`queue`模式。推理均不读取教师或队列，现有评估入口不变。

### 监督边界

- 真实配对决定同次执行跨相机的正样本，文本不能推翻该关系。
- 不同 scene 的语义接近只是软关联，不代表同一个实例。
- 语义损失不保证自动去除背景依赖，应通过实验检验。
- 同 scene 共享描述，并未新增“属于同次执行”的身份信息；扩展模态内一致性已经提供它。语义项新增的是不同操作间的动作和目标关系。
- 如果所有 scene 描述都退化为任务模板，语义项不能学习同 task 内的实例或阶段差异。
- 文本教师可能混淆细节，应保留原有损失，并通过验证集选择语义权重。

## 7. 与语义模态章节的分工

| 章节 | 使用方式 | 推理输入 |
|---|---|---|
| 语义模态 | 生成描述、编码文本并做语义检索，优化其质量 | 描述或文本特征 |
| 视频模态 | 复用语义结果作为训练教师，向视频表示迁移知识 | 视频 |

视频章节必须以纯视频检索结果验证收益。推理阶段融合文本获得的提升，不能归因于视频编码器自身改善。

## 8. 评估与论文证据

2026-10-18 已接入独立视频评估、原始骨干评估和重绘流程；不改变原有主检索指标。
`runs/video/evaluate_views.py`只读已有embedding和metadata，不加载视频模型、不重新解码。

| 呈现 | 定义和用途 |
|---|---|
| 同scene正样本余弦 | HH、RR只比较不同物理相机；HR分别报告同相机和跨相机 |
| 正负间隔 | 正样本余弦减去不同task、相同目标相机的负样本平均余弦；控制背景相似和整体坍缩 |
| 受控跨相机检索 | 按相同物理相机对分块，每scene一个候选；HR同/跨相机使用相同scene候选集。保存task/scene R@1、MRR；主诊断图显示task指标，与论文主检索目标一致 |
| 相机级热图 | 全部人类视频×全部机器人视频，按camera/task/scene排序，不再每scene只取一个相机，默认不截取120项 |
| scene级热图 | 每个格子为对应两scene的2×2人机视频余弦最大值，所有非对角线也同样聚合；只表示最佳视角匹配，不能证明视角不变性 |

不同相机对的检索指标按查询数加权。只有一个scene的相机分块不参与检索汇总，
避免单候选带来的必然R@1=1；没有不同task负样本的查询不参与负样本均值和间隔，
原始正样本相似度仍记录。`views.json`保存分块规模、计数、逐scene相似度及聚合协议。
热图统一使用余弦[-1,1]色标，避免每图自动缩放造成视觉上的虚假改善。

每个`evaluations/cfg2|cfg3/`新增`views.json`、`views.csv`、`views.png`、
`camera_heatmap.png`、`scene_heatmap.png`，后两张替换旧的两个首相机热图。
scene最大值聚合的两方向task/scene指标另存于`views.json:scene_max_metrics`，
不覆盖原有`metrics.json`，不与单视频检索结果混用。所有图片依赖进入批次`plots.json`。

```bash
# 复用E5已保存向量，重算cfg2/cfg3一致性并绘图
python runs/video/evaluate_views.py --run artifacts/runs/retrain/s44/video/E5
# 多个实验比较时，可重复--run；单独处理一个cfg可加--cfg cfg2
# 全批次重绘也会从视频向量补齐这些分析和图，不重新编码
python runs/redraw.py --batch s44
```

| 分析 | 目的 |
|---|---|
| 人↔人、机↔机，同次执行跨相机 | 隔离视角问题与人机域差异 |
| 人↔机，按相机组合统计 | 分析人机域差异与视角变化的叠加 |
| task 与 scene 指标分别报告 | 区分任务语义收益与实例匹配收益 |
| 严格跨相机候选库上的 R@1、MRR | 避免同相机候选掩盖退化 |
| 正负样本相似度分布、间隔与排名 | 排除只提高绝对相似度的表征坍缩 |
| 清晰、遮挡、操作区域较小等分组 | 分析适用条件与失败边界 |

同视角和跨视角比较时，应匹配候选库规模、任务分布和正样本数量。多视角融合、最佳视角选择如需分析应单独报告，不能用其收益证明单视频表示具有视角不变性。

可优先从已有 embedding 做问题诊断。推荐图表为跨相机检索对比、相机组合指标热图、可见性分组结果和成功/失败案例。

### 论文叙事

```text
发现跨视角退化
    → 说明已有跨域多正样本监督仍存在局限
    → 扩展模态内一致性，直接约束同域真实视角
    → 迁移已有语义知识，检验额外收益
    → 分析任务语义、实例对应、遮挡和时序方面的边界
```

结果解释示例，均需实际数据支持：

- task 提高、scene 不提高：语义联系改善，实例区分仍不足。
- 清晰视角提高、遮挡视角失败：收益依赖可观测操作信息。
- 已见视角提高、新视角不提高：有限视角适应，尚未充分泛化。
- 相似度提高、排序不提高：整体拉近，没有提升区分能力。
- 联合方法没有收益：检查教师质量、描述信息量、目标冲突及监督权重，不能仅归因于问题太难。

## 9. 相关工作与借鉴范围

| 工作 | 核心思路 | 本项目可借鉴部分 |
|---|---|---|
| [TCN，ICRA 2018](https://research.google/pubs/time-contrastive-networks-self-supervised-learning-from-video/) | 同步真实多视角观测对齐，并利用时间关系区分 | 真实跨相机正样本与时序对应 |
| [AE2，NeurIPS 2023](https://vision.cs.utexas.edu/projects/AlignEgoExo/) | 手和操作物体区域、非同步视频时序对齐 | 操作区域关注与细粒度阶段分析 |
| [Viewpoint Rosetta Stone，CVPR 2025](https://vision.cs.utexas.edu/projects/ViewpointRosetta/) | 配对数据、特征翻译、语义加权软对齐 | 语义软关系监督；不必复现完整扩散翻译 |
| [LangView，CVPR 2025](https://arxiv.org/abs/2411.08753) | 用语言描述匹配程度弱监督视角选择 | 视角信息质量分析；视角选择不同于视角不变表示 |

上述工作提供设计依据，不能直接作为本项目方法已有效或已获得跨视角不变性的证据。
