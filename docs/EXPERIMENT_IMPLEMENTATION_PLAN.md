# 论文实验与实现执行计划

版本：2026-09-21，按本轮确认的五章结构制定。

2026-10-10 同步说明：本文保留当时的章节编号、工作包和中期复现清单；
P02 中的旧 E1–E6 / T1–T5、`configs/thesis/` 目录建议不等同于当前已实现流水线。
现行矩阵为轨迹 T0–T5、N1–N5（T3 基线）、视频 E0–E7（E3 基线），
配置、预算、权重规则、划分和 s44 视频迁移以 [RETRAIN.md](RETRAIN.md) 为准。
后续视频跨相机实验只保留扩展模态内一致性、扩展模态内一致性＋语义监督两组，
具体范围见 [跨相机方案](video/CROSS_VIEW.md)，尚未实施。

配套正文：[THESIS_SKELETON.md](THESIS_SKELETON.md)。本文自包含必要的代码现状、协议、任务、实验与产物要求，不依赖旧总计划；删除旧总计划不影响执行。`PROJECT_AUDIT_20260921.md` 可作历史核查材料，但实施所需事实已在下文重述。

**当前只实施第三、四章。第三章重跑中期全部实验并补权重搜索；第四章升级语义、验证弱标签、混合cfg2/cfg3训练并与第三章基线比较。轨迹改进可选，第五章只保留论文骨架。总预算按两个半月、约10周安排。**

本文中的“新增”“拟新增路径”“拟新增参数”均是待实现内容，本轮只创建规划文档，没有实现这些功能。命令模板中的变量必须按步骤生成后再使用。

## 1. 先固定执行口径

### 1.1 第三章历史材料与实验原始数据

保留三个来源，不互相覆盖：

| 来源 | 用途 | 能否训练、选权重或复算检索 |
|---|---|---|
| `historical_report` | 将中期报告可读取数值转换成绘图数据，重建原图表；记录PDF页码/表图号 | 不能，汇总数值不能代替逐样本预测 |
| `existing_artifact` | 现存checkpoint/embedding/matrix/JSON，核验后用于追溯 | 通过一致性及划分检查后才能使用 |
| `rerun` | 本次重跑直接输出的原始数据，支撑新增实验与后续比较 | 可以，必须保存模型和数据来源 |

报告提取值允许转换成旧 exporter 的 JSON schema，用于历史图表重建；不能写入重跑的 `metrics.json` 冒充原始实验结果。曲线/热图没有逐epoch数据/矩阵时保留报告原图，或将图上数字化结果明确标记为近似；不从汇总指标合成曲线或矩阵。重跑不一致时保留差异说明。

第三章正文承接中期大部分内容、实验设计和历史材料，同时明确新增结果来源；最终基线模型和融合权重必须来自可复算的实际实验。

### 1.2 数据集与评价术语

- **第三章**：保留cfg2训练/验证/预留测试与cfg3外部评估，另把新增融合搜索的验证集和最终测试集合写清楚。
- **第四章**：cfg2/cfg3合并为一个数据池，保留cfg字段；按执行分组拆为train、dev、test，不再以cfg2训练/cfg3测试作为主协议。
- **dev**：开发验证集，选语义阈值、弱标签系数、checkpoint和融合权重。
- **test**：最终人工评价集，对应用户所说的人工筛选验证集；冻结后不再调参。
- **原始执行组**：同一scene的人机视频、所有相机、裁剪与派生轨迹属于同组。先划分组，再构造人机配对，禁止随机划分pair导致同视频跨集合。
- **人工标签与弱标签**：人工标签独立判断行为相关性；自动弱标签是训练信号。人工评价时不显示模型分数、task/cfg、生成描述。

第三章默认调优主指标：human-to-robot task MRR。第四章人工dev/test主指标：行为相关性NDCG@10，gain=`2^grade-1`；辅助Hit@1/5/10、MRR以grade≥2为正，并保留旧task/scene指标。中期R@K在代码中实际是Hit@K，表注说明，旧键名可保留兼容。相关性、候选ID与聚合方式必须随指标保存。

### 1.3 规模控制

必做：中期全部实验重跑、两章各自的权重搜索、四种核心语义设置、弱标签分布/阈值验证、六种视频训练设置、七种新模态组合和少量模块替换对照。

不纳入必做：精确时间边界标注、三维人机坐标重建、复杂域对抗/交叉注意力、新大型骨干、端到端三模态联合训练、动态融合、独立未见机器人基准。标签首版是“整体描述＋有序阶段”，不新增分段神经网络。

探索先单种子；优先把第四章同数据任务监督控制组和最终弱监督模型补到3个种子。全部中期模型不强制3种子，不对全部弱标签系数都训练网络。两个半月不够时先取消可选项，不减少必要的中期复现和公平比较。

## 2. 代码现状与最小修改清单

路径均相对于项目根目录 `/home/ttt/BISE`。下表“现有”内容已核对文件；“拟新增”项当前不存在。

| 范围 | 现有入口/实现 | 需要做什么 | 步骤 |
|---|---|---|---|
| 产物保存 | `src/bise/common/run_artifacts.py`；视频/轨迹evaluate脚本；`runs/restore_charts_from_json.py` | 补统一来源/哈希、ordered IDs、复算检查和图表索引；保留原文件 | P00—P01 |
| 一键训练 | `runs/video/run_all_video_experiments.sh`、`runs/trajectory/run_all_trajectory_experiments.sh` | 两者当前从`final_test/metrics.json`选best；改为同协议dev，task-held-out单列；支持冻结配置清单 | P02 |
| 视频checkpoint | `runs/video/train_video.py` | 默认选择分数含负train loss＋MRR＋NDCG；历史复现记录原规则，新基线显式设定选择规则 | P02 |
| 系统库 | `runs/retrieval/build_retrieval_library.py`、`src/bise/retrieval/system/library.py` | 建库/评估支持明确query/gallery白名单，不默认拿全库查询当预留测试；混合库正确保留cfg | P03、P05 |
| 融合与评价 | `src/bise/retrieval/system/scoring.py`、`evaluator.py`；`runs/retrieval/evaluate_retrieval_system.py` | 补权重搜索、人工qrels、完整逐查询指标与矩阵行列ID；复用原评分顺序 | P03、P06 |
| 语义标注 | `src/bise/modalities/semantic/{schemas,prompts,parser,normalizer,pipeline,batch,cache,paths}.py` | 增v2字段/提示词/规范文本，兼容v1；隔离缓存与输出；sync和Batch一致 | P07 |
| 语义默认配置 | `configs/semantic/pipeline_v1.json` | 当前默认cfg3、`manifest_start_index=2`；正式配置显式指定根目录、范围、视角和版本 | P02、P07 |
| 混合视频数据 | `src/bise/data/rh20t/video_dataset.py`；`src/bise/common/schemas.py`；视频factory/batch | 当前sample ID和scene ID不含cfg，单root；增多root与全局ID、统一task映射，不直接拼接局部整数label | P05 |
| 数据质量 | RH20T视频与轨迹dataset | 当前读取失败会递归换样本；正式运行前预过滤并记录排除项，遇残余坏数据显式报错/剔除，不改变ID对应 | P01、P05 |
| 视频损失 | `src/bise/modalities/video/losses.py`、`trainer.py` | 当前按task相等生成目标；新增软权重矩阵分支、mask和独立人机ID查询 | P09 |
| 视频评估 | `src/bise/modalities/video/evaluator.py`；训练脚本 | 接入人工dev评价以选择C4 checkpoint，另存task指标；不再硬编码只有task主指标 | P09 |
| 轨迹 | 现有模型、factory、训练器 | 默认原样沿用C3模型；混合ID/评估适配属于必做，网络升级属于可选 | P05、P11 |
| 工程服务 | `src/bise/service/`、`runs/retrieval/query_retrieval_system.py` | 当前仅保留接口与模型包信息，第五章后续单独安排 | 暂缓 |

已知证据风险：当前cfg2有109个task、434个scene，cfg3有64个task、256个scene；报告记载cfg2任务数与本地不同。现存cfg2系统完整模态393条查询中313条来自所用模型的训练scene。视频E4某目录JSON R@1为0.44、矩阵重算为0.375，轨迹T2也存在JSON与矩阵不一致。因此旧默认目录和`best_run`只能作为线索，不直接作为本轮正式结果。

## 3. 统一编号与保存位置

### 3.1 实验编号

- `C3-Vxx`：第三章视频；`C3-Txx`：轨迹；`C3-Sxx`：语义；`C3-Fxx`：融合。
- `C4-Sxx`：新语义；`C4-Wxx`：弱标签有效性；`C4-Vxx`：视频训练；`C4-Fxx`：融合；`C4-T01`：可选轨迹。
- `P00—P12`：实施步骤，同一步可产出多个实验。
- 每次真实运行另有唯一run ID，每次评价有唯一eval ID；实验编号不能替代运行版本。

### 3.2 推荐目录

```text
configs/thesis/                      拟新增：本轮冻结配置、实验注册表、数据协议
docs/thesis_records/                 拟新增：小型来源表、人工规则、图表索引
artifacts/thesis/historical_report/  报告提取数值、来源、重建图
artifacts/thesis/protocols/          母清单、划分、query/gallery ID、qrels
artifacts/thesis/ch3/                本次基线训练、语义评价、库与权重搜索
artifacts/thesis/ch4/                语义v2、弱配对、训练、库与最终评价
artifacts/thesis/figures/            按章输出论文图和绘图源数据
```

不搬动原`artifacts/runs/`和`artifacts/semantic/`；旧产物只读。新路径通过output配置传入；对于旧脚本自动生成的`video/trajectory`子目录，注册表记录真实路径，不为了目录整齐复制大文件。

所有正式run保存配置、命令、代码版本/未提交差异、环境、随机种子、输入清单哈希、split、checkpoint与训练曲线。每个eval保存有序query/gallery ID、metadata、embedding或其引用、完整score matrix、相关性定义/qrels、逐查询指标、汇总指标和案例。每张图保存CSV/JSON绘图数据与绘图命令。

## 4. 按顺序执行的工作包

### P00｜建立实验台账与恢复中期图表数据

**对应论文：3.2.3、3.3—3.6、附录A/C。前置：无。**

执行：

1. 只读盘点PDF、历史配置、checkpoint、split、metrics、matrix及图表JSON；记录大小、SHA256与来源。
2. 新建`docs/thesis_records/experiment_registry.csv`，字段至少为`experiment_id, chapter_section, config_path, run_id, eval_id, data_version, protocol_id, status, source_type`。
3. 从`tmp/唐艺通-中期报告.pdf`人工复核提取表2-1至表2-7，以及图2-8至图2-10中明确可读的指标；每项记录PDF物理页码、表/图号、方向、层级、有效小数位。不可读数值记缺失，不猜。
4. 输出`historical_report/reported_results.json`，增加`source_type=historical_report`及来源字段；另转换为旧exporter所需绘图JSON，保留原实验名、顺序、颜色/坐标轴设定。
5. 对图2-2至图2-7的曲线/热图逐项寻找源数据；找到后先复算和核对；未找到则保留PDF原图与缺失说明。只对有依据的数据承诺数值一致，版式是否一致单独记录。
6. 将报告值与现存文件、后续重跑结果放在`result_reconciliation.csv`三列中比较，不覆盖任一来源。

**拟新增工具**：`runs/audit_experiment_artifacts.py`，输出清单、缺失项与一致性报告；规模控制为一个本地脚本，不引入实验管理服务。图表优先复用`runs/restore_charts_from_json.py`。

**产物**：注册表、来源表、报告提取JSON、历史重建图/原图、缺失清单。

**完成条件**：中期每张图/表都有来源与处理状态，能区分哪些是原数据重绘、报告数值重建、原图保留；所有重跑任务有唯一编号。

### P01｜补最低保存与评估校验能力

**对应论文：3.2.3、附录C。前置：P00。**

实现：

1. 在`run_artifacts.py`及各evaluate输出中补数据/模型哈希引用、完整行列ID和协议ID；系统`per_query_results.jsonl`当前主要是Top-K案例，补真正的逐查询指标，二者不要混称。
2. 用P00工具复算`embedding → similarity → ranking → metrics`。保留旧函数用于历史复现，新评价明确使用哪个函数。保存误差、有效query数、异常项。
3. 扫描解码失败视频、空轨迹和ID重名，写`excluded_samples.jsonl`；正式manifest排除它们，dataset不再静默替换成另一执行。
4. 保存Python/NumPy/Torch/采样器种子；只给split设seed不等于训练可复现。保存最优/最后checkpoint；需要中断恢复的长训练再保存优化器/调度器/随机状态。
5. 每个正式实验完成后复制到用户实际可用的独立备份介质，校验文件hash；备份位置尚未指定，开跑前填写`backup_location`。同盘另建目录不算防磁盘丢失备份。

**必要检查**：矩阵行列顺序与ID一致；同分排序规则固定；无正样本query显式计数；figure JSON可单独重绘；随机恢复一个eval能复算指标。备份不要求建设平台。

### P02｜重跑中期全部单模态实验（历史清单）

**对应论文：3.2—3.5。前置：P00—P01。**

先从旧run的`params.json`及split恢复实际参数；原config用于比对和缺失补充。所有修复和差异写入注册表。冻结`configs/thesis/ch3/`副本后运行，历史产物不覆盖。

视频清单（报告中的全部设置）：

| 实验ID | 设置 | 现有配置/入口 | 评价与正文 |
|---|---|---|---|
| C3-V01 | E1 V-JEPA双域头、多正样本 | `configs/video/vjepa_rh20t_baseline.json` | cfg2 dev/test＋cfg3；3.3 |
| C3-V02 | E2 VideoMAE | `configs/video/videomae_rh20t_baseline.json` | 同上 |
| C3-V03 | E3 task-held-out | `configs/video/vjepa_rh20t_task_heldout.json` | 自己的task协议＋cfg3，单列解释 |
| C3-V04 | E4 intra=0.2 | `configs/video/vjepa_rh20t_intra.json` | cfg2 dev/test＋cfg3 |
| C3-V05 | E5一对一InfoNCE | `configs/video/vjepa_rh20t_info_nce.json` | 同上 |
| C3-V06 | E6独立双编码器 | `configs/video/vjepa_rh20t_dual_encoder.json` | 同上 |
| C3-V07 | Raw V-JEPA | `runs/video/evaluate_raw_video_backbone.py`＋E1配置 | 不训练，使用E1评价ID |
| C3-V08 | Raw VideoMAE | 同入口＋E2配置 | 不训练，固定与对照相同评价ID |

当前一键视频脚本还训练`shared`结构，可作为附录补充；`E6_dual_head`就是E1，无需重复训练。为本轮加配置清单/实验过滤支持，避免额外训练或遗漏报告实验。

轨迹清单：

| 实验ID | 设置 | 现有配置 |
|---|---|---|
| C3-T01 | T1基础 | `configs/trajectory/T1_trajectory_baseline_scene.json` |
| C3-T02 | T2增强 | `configs/trajectory/T2_trajectory_augment_scene.json` |
| C3-T03 | T3两阶段 | `configs/trajectory/T3_trajectory_two_stage_scene.json` |
| C3-T04 | T4 task-held-out | `configs/trajectory/T4_trajectory_task_heldout.json` |
| C3-T05 | T5 21关键点 | `configs/trajectory/T5_trajectory_21_keypoints_scene.json` |

全部轨迹run在本身cfg2 dev/test及cfg3上评估；T4不与scene划分结果直接做网络优劣排名。

语义：`C3-S01`旧描述、`C3-S02`旧标签、`C3-S03`旧描述＋标签，复用`runs/semantic/evaluate_semantic_retrieval.py`一次输出三模式。已有原响应、标注、embedding先核验后使用，以保留中期语义；缺失才以冻结v1配置补生成，并记录服务版本变化。不能用升级后的v2标注补进v1。cfg2必做报告复现，cfg3语义用于后续融合和对照。

**开跑前修正的选择逻辑**：两个一键脚本当前用test MRR选“best”，改为显式dev结果。视频epoch级默认选择还包含训练loss与NDCG，历史复现保留并记录；新增正式基线选择规则统一为dev task MRR，配置显式指定`loss_weight=0,mrr_weight=1,ndcg_weight=0`或对应新的选择接口。不要在不同split之间直接选一个总冠军；先限定同协议候选。

**已有命令模板**（激活torch2后，变量指向冻结配置及新目录）：

```bash
python runs/video/train_video.py --config "$VIDEO_CONFIG" --output-dir "$VIDEO_OUTPUT"
python runs/trajectory/train_trajectory.py --config "$TRAJECTORY_CONFIG" --output-dir "$TRAJECTORY_OUTPUT"
python runs/video/evaluate_video.py --config "$VIDEO_RUN/params.json" --checkpoint "$VIDEO_RUN/best_model.pth" --split-manifest "$VIDEO_RUN/split_manifest.json" --split val --output-dir "$VIDEO_RUN/eval_cfg2_dev"
python runs/video/evaluate_video.py --config "$VIDEO_RUN/params.json" --checkpoint "$VIDEO_RUN/best_model.pth" --split-manifest "$VIDEO_RUN/split_manifest.json" --split test --output-dir "$VIDEO_RUN/eval_cfg2_test"
```

轨迹评价用`runs/trajectory/evaluate_retrieval.py`同类参数。cfg3复用`scripts/evaluate_all_video_cfg3.sh`和`evaluate_all_trajectory_cfg3.sh`，显式设置新`RUNS_JSON`、`OUTPUT_SUBDIR`、`CHART_OUTPUT_DIR`；cfg3 all-test不要再传cfg2 split。确认脚本是否跳过缺checkpoint，正式清单中不允许静默缺项。

**产物**：每个训练run的checkpoint、params、split、曲线；每个数据协议的embedding、matrix、metadata、metrics、case；中期全部单模态表/图与复现差异表。

**完成条件**：8种视频设置、5种轨迹设置、3种旧语义模式全部有状态与可追溯评价；缺失项明确失败原因，不能用报告数字补成成功运行。

### P03｜统一系统评价并复现七种均匀组合

**对应论文：3.6。前置：P02的候选模型；实现可与训练交错。**

1. 核对报告对应的建库模型。当前`library_rh20t_cfg2.json`是E1视频＋T2轨迹，不自动替换为E4。历史系统复现使用历史组合；新增最终基线使用dev选出的模型，名称分别记录。
2. 增建库与评价的query/gallery白名单支持，参数设计可为`query_manifest`、`gallery_manifest`，属于待实现；旧命令目前不支持这些参数。过滤后重新统计可用模态、query/候选/scene数。
3. 历史单模态split保留原状。新增融合实验需要所有参与模型共同未用于训练的dev/test。先计算所选模型训练集合的并集及历史val/test交集；dev与test按执行分开，不能直接复用现存393条全库查询。
4. 若共同保留集合太少，或一个模型的test是另一个模型的val，**只对最终选定的视频与轨迹方法补一次共同split训练**，不再重跑整个实验矩阵；标记`C3-B-final`受控最终基线，保留前述历史复现结果。这个必要分支在P02结束前确定预算。
5. cfg2新增融合dev/test，cfg3固定外部评价；提前预留第四章人工test的执行，禁止第三章新增调优使用这些执行。历史cfg3结果已经被观察过，不能宣称第四章是完全未观察数据上的盲测。
6. 对历史库和受控最终库分别运行V、T、S、V+T、V+S、T+S、V+T+S；图表不能混合两个协议。

**复用**：`scripts/run_retrieval_system_ablation.sh`，显式设置`LIBRARY_DIR`、`OUTPUT_ROOT`、`LEVEL`和`REQUIRE_MODALITIES`。默认指向cfg3且level为mixed，不能依赖默认值。

**实验**：`C3-F01` cfg2历史/受控均匀组合，`C3-F02` cfg3均匀组合；每个子协议有独立eval ID。

**产物**：过滤后的库、统一query/gallery、模态覆盖、逐模态/融合分数、七组合指标/案例、图3-9/图3-10。

### P04｜实现权重搜索并冻结第三章最终基线

**对应论文：3.7。前置：P03。**

**拟新增**：`runs/retrieval/search_fusion_weights.py`，必要函数放`src/bise/retrieval/system/`；配置放`configs/thesis/ch3/fusion_search.json`。同一工具随后用于第四章。

算法及接口要求：

1. 输入库/分数缓存、dev query/gallery ID、指标口径、校准、相机聚合、权重步长和输出目录；拒绝未显式标记为dev的选择输入。
2. 枚举`w_V,w_T,w_S≥0`、和为1；步长0.1共66组，包含零权重边界。额外加入精确的`(1/3,1/3,1/3)`等权点，因为它不在0.1网格内。
3. **复用真实评分顺序：camera级打分 → 每模态校准 → 权重融合/缺失重归一 → camera-to-scene max → 排序。** 现有eval保存的单模态矩阵已按scene聚合，不能直接将这些矩阵加权来替代原系统；`max(sum)`通常不等于`sum(max)`。
4. 小库可对每个权重重复调用真实评分代码，网络特征只提取一次；如缓存，则保存camera级分数、可用性mask、ID、scene聚合索引和校准设置。
5. 主目标C3 dev task MRR；并列先比较NDCG@10，再选择最接近等权的方案，最后固定字典序。规则事先写入配置。
6. 保存全部权重得分和选中配置；用一个权重回放现有evaluator，验证排名/指标一致。选完只在cfg2 test和预定cfg3评价上计算，不使用测试结果重选。

**实验**：`C3-F03`搜索，`C3-F04`固定最优权重与等权的最终对照。

**产物**：`weight_search.csv/json`、`selected_weights.json`、验证权重图、固定测试结果；`ch3/baseline_bundle.json`记录视频/轨迹checkpoint及hash、语义v1版本、校准、聚合、权重、训练/dev/test ID引用。这里的“最优”是指定网格与dev主指标内最优，不是所有可能方法的全局最优。

### P05｜建立第四章混合数据池与安全划分

**对应论文：4.2.1—4.2.2。前置：P03保留集合决策；可在P04调权时实施。**

**拟新增入口**：`runs/retrieval/build_experiment_manifest.py`。复用RH20T scanner，不复制原视频。

母manifest至少包含：

```text
video_id, execution_group_id, dataset, cfg, task_id, semantic_task_id,
scene_id, camera_id, domain, video_path, trajectory_ref, original_pair_id,
video_hash, available_modalities, exclusion_reason, split
```

使用`dataset::cfg::task::scene::camera::domain`形式的全局视频ID，执行组不含camera/domain。保留旧ID映射，旧checkpoint仍可读取原样数据；训练task相等使用经核实的`semantic_task_id`，不能仅因不同cfg目录都叫task_0001就未经核实认定语义相同，也不能把cfg拼进task标签后禁止所有跨cfg同任务关系。

实现选择：给现有RH20T视频dataset增加多root/显式manifest分支，factory统一生成样本列表和task整数映射。不要简单ConcatDataset后继续使用各子集局部task_indices。轨迹、语义和检索库沿同一母ID映射读取。视频eval兼容不同人机列表的metadata，不能默认所有新配对仍共用一份标签。

划分步骤：

1. 对第三章冻结模型包的train/dev使用记录求并集，确定人工test可用执行；不能只排除第四章训练样本。
2. 先为最终人工test和独立dev预留人类查询及机器人候选执行，两个cfg均争取覆盖；剩余进入混合train。同执行的所有视角一起归属。
3. 若旧模型没有足够干净cfg2执行，走P03的共同split补训分支；不能为了混合比例把见过的执行放进test。仍不足则缩小基准并报告实际cfg比例，不能声称不存在的混合覆盖。
4. 自动弱标签只在每个split内部构建，训练器只能访问train pair文件。人工dev用于选择，test仅最终评价。
5. 大规模自动task评价与小型人工评价分别保存协议ID；旧新模型在各协议中使用完全相同的候选库。

**产物**：母清单、全局ID映射、`split_manifest_v1.json`、train/dev/test query/gallery、覆盖统计、`split_overlap_report.json`。

**完成条件**：无跨split执行重复，第三章baseline train与第四章test无交集；cfg字段存在但不决定train/test身份；缺失语义/轨迹显式列出。

### P06｜建立小型人工评价集及qrels评价

**对应论文：4.2.3、4.4.3。前置：P05，抽样可在P07初版标签生成后完善。**

规模先控制在：dev约20条独立人类查询、test约40条，分别固定40—60个机器人候选执行，即约800—1200和1600—2400对判断。先做10×20试标，记录人工耗时再冻结正式规模；多个相机不重复算独立query。规模不足不强行增加数千样本。

抽样包括同/异task、同/异cfg、高/中/低语义分数和随机参照；保存抽样层与抽样概率。高分但错误的弱标签必须保留用于评价，不能“人为筛选”后只留下正确样本。先预留执行池，再按标签分层选query/gallery，最终固定小库全量判断。

人工等级：3=目标、关键操作与结果兼容；2=核心能力相关但部分条件不同；1=仅部分动作相同；0=不相关或关键目标冲突；无法判断=unknown。标注依据是视频内容，不是task/cfg一致性。等级不是执行成功承诺。

主标注者完成全部配对，独立复核至少20%和所有不确定样本；报告复核一致率、分歧仲裁与覆盖，不宣称全部双人标注。全库NDCG所需判断未完成时，先解决unknown或冻结删除不可判定的完整query/候选并报告排除，不能把unknown直接当0。

**拟新增**：`src/bise/retrieval/qrels.py`和`runs/retrieval/evaluate_rankings.py`；统一读取显式ID及score matrix，既支持单模态也支持融合。qrels字段为`query_id,gallery_id,grade,review_status,annotator,reason`，dev/test分文件。与旧task指标并行输出，不改掉旧评价定义。

**必要检查**：候选全判定、标签与矩阵ID对齐、无正样本query数量、手工小例子的NDCG/MRR/Hit、同分规则。无相关候选查询主NDCG单列处理并报告数量，不静默删去。

### P07｜实现语义v2并完成语义对比

**对应论文：4.3。前置：P05；标签开发只使用train/dev。**

修改`semantic/schemas.py`、prompts/parser/normalizer，新增v2配置，兼容v1；使用相同文本编码器，尽量固定VLM版本，避免把换模型收益当语义结构收益。

首版schema：

```text
global_description
object_role, target_relation, final_state
action_sequence: [{order, action, object_role, target_relation, observable}]
uncertainty_reason
schema_version, prompt_version, provider_model, video_id, video_hash
```

时间戳为可空扩展字段，当前不要求阶段边界。生成阶段不向VLM提供task/cfg或成对另一侧视频信息；prompt里样本显示ID使用不含任务含义的opaque ID，元数据保存在外层。允许阶段重复、缺失、unknown，不强制所有视频都是六步。

复用`pipeline.py`和Batch入口完成raw response、normalized annotation、失败清单、缓存。v2输出前缀独立；cache key包含输入hash、帧采样、schema/prompt/模型版本，防止命中v1缓存。先用stub验证解析再调用真实服务；已有缓存不足的样本必须明确补标，不能把不同camera描述静默借给新视频。

首版表示采用固定文本编码器：全局文本单独编码；阶段按序号、动作、对象关系拼为规范文本；组合将全局与有序阶段拼为一段规范文本后编码。先不训练序列编码器。顺序拼接是否有效由对照验证，不能仅因文本带序号就认定模型理解顺序。

实验：

| 实验ID | 语义表示 | 目的 |
|---|---|---|
| C4-S00 | 第三章v1描述/粗标签/组合，在本章同库重评 | 保留旧语义对照 |
| C4-S01 | 新整体行为描述 | 区分描述改写收益 |
| C4-S02 | 新整体描述＋有序阶段 | 主方案 |
| C4-S03 | 新整体描述＋打乱阶段顺序 | 低成本顺序消融，固定打乱seed |

可选增加阶段单独编码，不新增训练。为第四章所有参与视频补齐v1与v2语义，两个版本都只看该视频；v1历史没有覆盖的新增标注记录为“旧方法补生成”。

**产物**：v2提示词/词表、raw/normalized JSONL、embedding及ID、语义人工抽检、同库旧新指标、图4-2/表4-3。测试语义在提示词冻结后一次生成，不根据test表现改提示词。

### P08｜量化弱标签有效性并构建三信号标签

**对应论文：4.4—4.5。前置：P06—P07。**

**拟新增**：`runs/semantic/analyze_weak_labels.py`、`runs/semantic/build_weak_labels.py`；共用函数放`src/bise/modalities/semantic/weak_labels.py`。

**W01：同任务相似度阈值验证。** 在train/dev独立采样配对，用新旧语义计算相同pair集合的余弦分数；分别统计原配对、同任务不同scene、同任务跨cfg及不同任务对。

```text
coverage_same_task(tau) = count(s_ij >= tau and same_task) / count(same_task)
cross_task_above(tau)  = count(s_ij >= tau and different_task) / count(different_task)
```

输出直方图、ECDF、分位数、阈值—覆盖率曲线、同/异cfg分层。开发集可先取同任务分数10%分位数作为“约90%覆盖”的候选阈值，再检查人工配对precision和跨任务误判；同时报告完整曲线，不能把按定义得到的90%当独立证据。阈值冻结后报告最终样本实际覆盖率。不同task超阈比例不直接命名为FPR，只有人工判定不相关的部分才是错误。

**W02：人工弱正样本质量。** 在train/dev侧分层审计约150—250对，报告高分pair的人工precision、各层数量、跨任务新增合理pair与错误案例；这是小成本pair审计，不与最终检索test混用。抽样分布不同的precision不直接代表全库precision。

**W03：三信号加权标签。** 固定分数变换后定义：

```text
u_ij = clip((cosine(z_sem_i, z_sem_j) + 1) / 2, 0, 1)
d_ij = 1[semantic_task_id_i == semantic_task_id_j]
c_ij = 1[cfg_i == cfg_j]
w_ij = alpha*u_ij + beta*d_ij + gamma*c_ij
alpha,beta,gamma >= 0; alpha+beta+gamma = 1
```

`alpha,beta,gamma`是弱标签系数，后续`w_V,w_T,w_S`是融合权重，两个搜索不能混淆。阈值统计若使用cosine，弱标签使用u，则阈值也必须明确变换。

先在dev pair审计上比较少量固定候选，如仅语义`(1,0,0)`、语义加task`(0.75,0.25,0)`、完整`(0.7,0.2,0.1)`；这些是起始候选，不是预定最优。另以较小gamma候选做廉价标签质量比较，选定后才训练，不对网格每点训练网络。

保留连续w作为软目标基础；另存positive/negative/uncertain mask。弱正样本需通过冻结语义门槛或可靠任务配对规则，cfg相同单独不足以判正；uncertain不作硬负样本。可信原配对可作为回退正样本，失败/冲突样本按预定质量规则排除。是否使用门槛、回退和系数必须同等应用到相应对照并单独记录。

pair文件至少包含`human_video_id,robot_video_id,semantic_score,same_task,same_cfg,weight,mask,source,split,label_version`。先只生成/保存train内pair；dev/test分别生成诊断文件，test qrels绝不进入训练pair。

**产物**：逐pair分数、阈值曲线与选取配置、人工质量统计、三信号系数、train弱配对文件、每类正样本数量/覆盖/不确定比例；图4-3/图4-4和表4-4/表4-5。

### P09｜接入弱标签训练并完成视频对比

**对应论文：4.6。前置：P05—P08。**

训练实现：

1. 保留现有双塔视频结构；扩展`batch.py`带独立human/robot全局ID、cfg、task映射。初版仍从执行配对项加载人机视频，但batch同时纳入挖掘关系两端所在项，在全部人机交叉位置取弱标签；不要只改batch对角线。
2. 拟新增`src/bise/modalities/video/sampler.py`中的弱关系batch sampler：锚点项＋弱相关项＋其他项，固定每batch视频数；去重sample，并记录跨任务关系实际进入batch的比例。对照使用相同采样预算与视频池，减少采样方式的混杂。
3. 在`losses.py`增soft-target分支，输入`weight_matrix`、`valid_pair_mask`、可用query mask。按行/按列分别归一化，双向交叉熵或KL；零正样本行/列跳过并统计，整批无有效项时正确返回可反传零值或按约定跳批。禁止NaN和不确定pair当负样本。
4. `trainer.py`根据显式loss名称读取矩阵；不认识的loss名称报错，不能像旧分支一样无声回退到task multi-positive。保留原loss路径兼容第三章。
5. 第一版总目标仅软标签跨域损失＋第三章已有intra项，不额外堆蒸馏、阶段对齐或域对抗。双向矩阵转置后单独归一，不能直接转置已归一化Q。
6. `train_video.py`为本章增加人工dev NDCG的checkpoint选择，保留dev task曲线；使用固定小dev降低开销。最终模型选择不读取test。

必要实验：

| 实验ID | 数据/初始化/监督 | 用途 |
|---|---|---|
| C4-V00 | 第三章最佳视频，冻结直接重评 | 用户要求的旧模型直接比较，不训练 |
| C4-V01 | 混合train，任务ID监督 | 控制混合数据与训练量变化 |
| C4-V02 | 混合train，旧v1语义＋task＋cfg | 控制语义升级本身 |
| C4-V03 | 混合train，新语义，alpha=1 | 仅语义监督 |
| C4-V04 | 混合train，新语义＋task，gamma=0 | task信号贡献 |
| C4-V05 | 混合train，新语义＋task＋cfg | 完整提出方案 |
| C4-V06 | 与V05相同但打乱语义对应，保留元数据关系 | 负控制：验证有效信息并非仅来自task/cfg |

V01—V06统一骨干、可训练层、更新步数、视频池、真实batch大小和intra设置。优先都从同一个C3视频checkpoint初始化，且该checkpoint没见过本章test；因此描述为继续训练。若改从预训练骨干开始，则整组一起改变，不能只给新方法额外预训练优势。V00直接迁移分数变化包含混合数据分布因素，V01才是弱监督因果比较的关键控制组。

先每项一个seed；dev筛选后V01和选定完整/最佳弱监督方案补到3个seed。V05若不如V04，如实报告cfg项无益，不将三信号公式强行宣称更优。所有方法冻结后同一人工test一次统一评价；视频单模态推理只使用视频。

**必要测试**：软权重可退化到任务多正样本；行/列和正确；batch重排与ID对齐；零正样本、mask、梯度有限；混合cfg无ID碰撞；sampler至少能将一条跨任务关系放入batch；测试视频不进入任何训练pair。

**产物**：每项checkpoint/曲线/弱标签版本及训练pair统计、dev选择记录、同库test分数/指标/案例、表4-6/表4-7/图4-5。

### P10｜第四章融合、权重搜索与第三章最终对照

**对应论文：4.8。前置：P04、P06、P09。**

构建同一混合query/gallery库的特征版本。每个scene候选和query必须保持同一ID；缺失模态主表使用固定完整子集，另报覆盖数。轨迹默认C3冻结模型，只做多cfg ID与数据读取适配。

实验顺序：

1. `C4-F01`：新视频＋旧轨迹＋新语义的七组合，固定等权；完整模态版本不必然优于V+S，边界也要报告。
2. `C4-F02`：逐模块替换四组——旧V/旧T/旧S、旧V/旧T/新S、新V/旧T/旧S、新V/旧T/新S；先固定同一套权重评价归因，再各自在dev调权作性能补表。
3. `C4-F03`：复用P04搜索器，以本章人工dev NDCG@10选融合权重，固定校准和聚合方法；保存精确等权参照和全部候选得分。
4. `C4-F04`：冻结方法与权重后，在同一人工test比较下表；同时报告旧task指标。所有ID、候选规模、人工qrels完全一致。

| 最终行 | 模型与语义 | 权重 | 回答什么 |
|---|---|---|---|
| B3-fixed | 第三章冻结模型与v1语义 | 第三章原最优权重 | 旧系统直接迁移表现 |
| B3-retuned | 与上一行相同 | 第四章dev重新搜索 | 排除仅重新调权的收益 |
| Mixed-task | C4-V01任务监督视频＋旧轨迹＋旧语义 | 第四章dev搜索 | 混合数据/继续训练的控制 |
| New-full | 选定弱监督视频＋旧轨迹＋v2语义 | 第四章dev搜索 | 总体方法改进 |

弱监督的独立收益以P09中V01对新模型为依据，不能只用B3-fixed对New-full作因果结论。七组合和权重搜索主要是缓存分数计算，不重训网络。

**产物**：七组合表、模块替换表、权重搜索数据、固定test主表、跨任务/跨cfg分组及Top-K案例、`ch4/final_bundle.json`。

### P11｜可选轨迹改进

**对应论文：4.7，可删除。前置：P09核心实验已跑通，且剩余至少一周机动时间。**

先抽查20条轨迹的坐标含义、frame index和缺失比例。旧提取器保存hand-centered world landmarks，不能当作全局掌心路径；加一层MLP不能恢复已缺失的位置。当前不启动深度/标定补采、三维重建或Soft-DTW全链路。

只选择一个可实现问题，例如有效帧/缺失质量过滤或明确的输入归一化，做`C4-T01`对照并检查融合贡献。2—3个工作日内没有清楚改善，则保留旧轨迹并删除正文可选节。数据可读性修复与ID适配是基础维护，不包装成独立创新。

### P12｜论文图表、复算与归档收尾

**对应论文：全篇、附录A/C。前置：P04、P10。**

1. 从实验注册表生成论文图表索引，所有标题、实验名、数据协议、数值来源一致。
2. 在空输出目录仅用绘图JSON/CSV重建第三、四章图，热图从矩阵重建；再次检查embedding/matrix/metrics一致性。
3. 对最终V01/新方法记录seed均值与波动；相同query的系统差值可按执行组做paired bootstrap，输出置信区间。只有一个seed的实验明确标记，不制作虚假的误差条。
4. 最终包中的checkpoint、权重、语义版本、qrels、ID和代码版本均能追溯；测试后不反向改阈值/权重并仍称盲测。
5. 整理复现差异附录，保留未成功实验和缺失原因；备份并抽样恢复一个完整run。

第五章目前只接收`final_bundle.json`、输入/输出schema和模型ID映射需求，工程实现另行安排。

## 5. 论文图表与实验产物索引

| 论文位置/图表 | 实验或步骤 | 必须保存的绘图/表格输入 |
|---|---|---|
| 3.1 图3-1 | P00＋实际框架 | 结构图源文件，实际调用流程说明 |
| 3.2 表3-1 | P00—P03 | 数据统计、split、排除清单、来源类型 |
| 3.3 表3-2至3-4，图3-2至3-5 | C3-V01—V08 | 配置、指标、逐epoch曲线、矩阵与metadata |
| 3.4 表3-5至3-7，图3-6至3-7 | C3-T01—T05 | 同上，另含输入质量统计 |
| 3.5 表3-8、图3-8 | C3-S01—S03 | v1词表/提示词、标注与三种embedding/指标 |
| 3.6 图3-9、图3-10 | C3-F01—F02 | 七组合指标、库/ID、各分数及协议 |
| 3.7 图3-11、表3-9至3-10 | C3-F03—F04 | 全部搜索点、等权点、选定配置、固定test指标 |
| 4.1 图4-1 | P05—P09 | 弱监督流程图源文件 |
| 4.2 表4-1至4-2 | P05—P06 | 混合清单、split交集报告、人工qrels与复核统计 |
| 4.3 图4-2、表4-3 | C4-S00—S03 | 旧新标注示例、同库分数、人工/任务指标 |
| 4.4 图4-3至4-4、表4-4 | C4-W01—W02 | 逐pair分数、阈值曲线、人工质量/案例 |
| 4.5 表4-5 | C4-W03 | 三信号系数、各类pair数量与质量统计 |
| 4.6 表4-6至4-7、图4-5 | C4-V00—V06 | 配置、checkpoint、训练曲线、同库评价 |
| 4.7 可选轨迹表 | C4-T01 | 输入修正版本、旧新轨迹及融合对照 |
| 4.8 表4-8至4-11、图4-6至4-7 | C4-F01—F04 | 七组合、替换消融、搜索全表、最终对比和Top-K |
| 第5章 | 当前暂缓 | 后续实际工程材料 |
| 附录A/B/C | P00—P12 | 全配置、提示词/标注规则、来源及重绘索引 |

若删除可选4.7导致编号变化，先按实验ID和章节标题查找，最后统一更新图表编号；不能仅靠数字编号关联原数据。

## 6. 十周排期与停止扩展规则

| 时间 | 主要任务 | 写作与交付 |
|---|---|---|
| 第1周 | P00—P01；P02冻结配置与短跑，启动全部中期实验队列 | 中期图表来源清单；第三章方法段落；保存/备份通路 |
| 第2—3周 | 完成P02；P03统一库；P04权重搜索与最终基线 | 第三章完整结果与baseline包；第1—2章初稿 |
| 第3—4周 | P05混合池；P06试标；P07语义v2 | 第四章数据协议和语义部分；冻结人工dev/test |
| 第4—5周 | P07语义对比；P08相似度阈值与弱标签生成 | 弱标签有效性图表、train pair文件 |
| 第5—7周 | P09六组训练、筛选与关键重复；人工test标注完成并封存 | 视频改进主表与第四章训练部分 |
| 第8周 | P10融合/搜索/最终对照；P11仅在时间允许时启动 | 第四章完整结果与最终模型包 |
| 第9—10周 | P12复算、图表重绘、论文整理与机动补实验 | 可追溯论文初稿/定稿材料；第五章实施另行排期 |

本表不承诺第五章工程在同一十周内自动完成。若两个半月还必须包括工程，最晚第4周明确接入范围并从机动时间中预留；不能边做前三四章边隐性增加完整平台任务。

第1周用一个视频epoch和一个完整轨迹短跑测GPU耗时，外推全部中期实验队列。若算力明显不足，先减少重复seed、可选shared对照、轨迹改进和额外超参探索；中期报告中的全部实验仍保留，必要时优化调度和评估缓存，训练预算变化必须记录。

## 7. 每步完成前的最小核对

- **P00完成**：每张中期图表都有来源；报告提取数据与实际实验原始数据分开。
- **P01完成**：一个eval可复算、一个图可重绘、一个run可从备份恢复。
- **P02完成**：中期全部实验有结果；模型选择不再读取test；不同split实验不混选。
- **P03—P04完成**：系统评价不含参与模型训练执行；权重搜索回放与真实评分一致；baseline包固定。
- **P05—P06完成**：混合cfg ID无碰撞、执行分组无泄漏、人工小库有完整qrels。
- **P07—P08完成**：语义v1/v2不混缓存，同任务阈值覆盖有参照，三信号与人工质量有数据。
- **P09完成**：训练器确实读取弱矩阵，batch包含新增关系，公平对照和人工dev选择跑通。
- **P10完成**：第三章原模型原权重在同一人工test重评，另有重新调权和混合task监督控制组。
- **P12完成**：论文每个结果均能追溯到数据文件，未完成第五章和可选轨迹不写成已完成成果。

新增实现的必要测试集中在ID/split、软标签矩阵、qrels和评分顺序；不为纯文档或绘图排版建立冗余测试。代码阶段遵循仓库要求在torch2执行相关测试，正式提交前运行`python -m pytest -q tests`。本轮仅编写这两份文档，未启动上述实现、训练或付费标注。
