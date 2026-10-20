# WiLoR 独立提取与质量检查

先在本机做小规模离线提取，所有产物位于`artifacts/hand/wilor/`，不写入原数据集，
不替换`human_hand_3d.npy`，不改已有训练、划分与评估。默认试验为三个scene、全部三视角及全部RGB帧：

- cfg2/task_0020/scene_4：操作手与键盘旁手。
- cfg2/task_0109/scene_4：持刀与扶持物体的双手协作。
- cfg3/task_0004/scene_4：球杆操作与键盘旁手。

## 环境与来源

使用现有torch2创建共享其Torch的隔离venv；新依赖只安装在`artifacts/tools/wilor-env`，不修改torch2。
无桌面训练机需补齐系统库`libegl1`、`libgl1`、`libglu1-mesa`供官方模块导入，不涉及CUDA驱动升级。
官方代码位于`artifacts/tools/WiLoR`，版本固定于setup脚本。权重与MANO模板来自作者公开的
[官方演示仓库](https://huggingface.co/spaces/rolpotamias/WiLoR/tree/main)，保存于`artifacts/models/hand/wilor`。
MANO的模型说明与条款见[官网](https://mano.is.tue.mpg.de/)，WiLoR条款见[代码仓库](https://github.com/rolpotamias/WiLoR)。
角色标注复用`configs/semantic/ch3.json`中的GLM提供方、密钥池与计价，不写入明文密钥。
推理使用完整网络、FP16、batch1，不使用官方fast模式的跳层和compile。

```bash
conda activate torch2
python scripts/setup_wilor.py

# 默认3个scene：提取→基础质检→GLM角色标注→初步深度定位→身份校验→几何检查/融合→轨迹质检→N1跨视角比较
# 已有提取与成功标注自动跳过；轨迹重建和绘图重新生成
artifacts/tools/wilor-env/bin/python runs/trajectory/hand_wilor.py

# 独立入口
artifacts/tools/wilor-env/bin/python runs/trajectory/hand_wilor.py --stage extract
artifacts/tools/wilor-env/bin/python runs/trajectory/hand_wilor.py --stage quality
artifacts/tools/wilor-env/bin/python runs/trajectory/hand_wilor.py --stage roles
artifacts/tools/wilor-env/bin/python runs/trajectory/hand_wilor.py --stage build
artifacts/tools/wilor-env/bin/python runs/trajectory/hand_wilor.py --stage compare

# 新选样使用独立输出目录；不修改默认试验
artifacts/tools/wilor-env/bin/python runs/trajectory/hand_wilor.py \
  --scene cfg2/task_0001/scene_1 --output-dir artifacts/hand/wilor/extra
```

配置为`configs/hand/wilor.json`。更换提取设置或scene集合使用新输出目录；同设置重跑用`--overwrite`。
执行批量参数可原目录续跑：`frame_batch_size=4`表示每次检测4帧，`batch_size=4`表示将这些帧的手部裁剪
跨帧合批、每次最多重建4只手；空帧和尾部不足4帧也会保存，跟踪、时间戳和输出按原始帧序处理。
仅改变这两个批量参数不会拒绝已有完整camera缓存，重启后只重做未完成camera；新NPZ记录
`frame_batch_size`和`hand_batch_size`，旧NPZ不改写。半精度合批可能产生微小数值差异。
前缀提取冒烟使用`--stage extract --max-frames 4 --output-dir artifacts/hand/wilor/smoke`，
明确不是完整轨迹，角色标注要求完整camera结果。强制重标注使用`--stage roles --overwrite`；
失败请求自动重试三次，仍失败则保留已成功记录，下次同命令只补缺失scene。
提示词、采样或模型配置改变时使用新的输出目录，防止旧标注与新条件混用。

训练机仅运行①提取，全部cfg2/cfg3 scene的三个视角，成功后自动打包：

```bash
cd /root/autodl-tmp/BISE
artifacts/tools/wilor-env/bin/python runs/trajectory/hand_wilor.py \
  --stage extract --all-scenes --cfg cfg2 cfg3 \
  --output-dir artifacts/hand/wilor/full --archive
```

当前子集为cfg2 434个scene/1302路视频、cfg3 256个scene/768路视频；
加`--list-only`只检查并显示范围，不加载模型、不写产物。正式提取已有完整camera自动跳过。
`--archive`只在提取或整条流水线成功结束后生成`artifacts/hand/wilor/full.tar.gz`，
先写临时包，完成后替换；失败时保留已有camera以供同命令续跑，不生成完成包。
①不调用GLM、不做深度重建。训练机使用共享现有Torch的隔离venv，依赖固定在requirements.txt，
不替换正在训练使用的Torch。CUDA模型前向仍需GPU，启动时间由使用者决定。
后续标注、重建、比较继续沿用相同的`--all-scenes --cfg cfg2 cfg3 --output-dir artifacts/hand/wilor/full`，
分别使用`--stage roles`、`build`、`compare`；重建会输出两套N1候选，比较会生成原始表和图。

## 产物与坐标

```text
artifacts/hand/wilor/pilot/
  params.json                 配置与官方代码版本
  cfg2|cfg3/task_*/scene_*/
    <camera>.npz              逐帧时间、全部手的观测、临时track ID、MANO参数
    <camera>.mp4              原视频叠加骨架、ID和源时间
    <camera>.jpg              六帧预览
  cameras.csv                 每相机检测统计、耗时及同帧MediaPipe检测覆盖率
  tracks.csv                  每条临时轨迹的覆盖率、骨长变化与跳点诊断
  views.csv                   基于原标定与源时间戳的相机对极线误差诊断
  quality.png                 从cameras.csv、tracks.csv、views.csv绘制
  report.md                   汇总表与解释边界
  roles/
    params.json               提示词、提供方、采样设置（不含密钥）
    requests.jsonl            原始API响应、失败、重试、token与费用记录
    annotations.jsonl         成功scene的角色、物理手身份与时间段
    summary.json              累计费用、延迟、成功率与执行耗时
  cfg*/task_*/scene_*/roles/
    <camera>.jpg              实际送给GLM的带编号采样帧接触图
  cfg*/task_*/scene_*/trajectory/
    <camera>.npz              角色筛选与深度定位结果；记录被选原始观测索引
    <camera>.mp4              被选物理手ID、角色与三维有效性叠加视频
    fused.npz                 择优/融合结果、来源、缺测掩码、初始掌部归一化结果
    summary.json              覆盖率、融合比例、相机对诊断与参数
    quality.png               腕点轨迹、有效关节比例、融合比例
  trajectory_params.json      本次重建阈值
  trajectories.csv            各scene的轨迹质量比较
  trajectories.md             处理结果和解释边界
  normalized.csv/.png/.md      N1归一化前后的跨视角三维比较
  candidates.png              两个正式N1候选在相同观测上的诊断比较，原始数据在normalized.csv
  cfg*/task_*/scene_*/trajectory/
    normalized.npz/.png        配对源帧、N1坐标、共同锚点变换及腕点位移曲线
```

NPZ无pickle：`frame_ptr`将每帧映射到展平的手观测，零检测帧也保留；`timestamp_ms`来自原始RGB时间戳，
不是用视频播放fps推算。`joints_image_px`为模型关节在原图的投影，`joints_root_m`为腕点相对MANO三维手形。
`joints_camera_estimate_m`使用WiLoR默认假设焦距，只用于保存原模型输出，**不能当作已标定的基坐标运动**。
`track_id`只在当前相机内成立，不代表解剖学左右手或操作角色，也不保证遮挡后的身份正确。

## 质量检查的含义

检测覆盖率与旧MediaPipe的原始`hand_count`在同一批帧上比较，不与21点深度完整率混比。
两者阈值不同；检测到更多手可能包含误检。空场也计入分母，因此覆盖率不是漏检率。
骨长CV与腕点跳变用于发现问题，不自动剔除真实快速动作。MANO解剖先验也可能使错误姿态看起来稳定。
极线误差使用原始内参按分辨率缩放、原外参和50ms以内最近时间匹配；按较小极线残差匹配手，
因此是偏乐观的相机对诊断，不能证明身份或三维精度正确。

## 后续处理路线

采用“三视角辅助选手与质检，以最佳单视角为基础，可靠观测才融合”，不要求三个视角强行融合。

1. WiLoR分别提取三视角的全部手，保留原始时间戳和观测。
2. 基础质检，处理明显跳点、身份断裂；将手编号、骨架和时间叠加到对应视频帧。
3. 第三方多模态API标注主操作手、辅助手、闲置手和操作时段，允许返回不确定。
4. 提取各视角的操作手观测；双手协作保留辅助手，不将它当噪声删除。
5. 初步三维恢复后，校验两个物理手的跨视角身份置换，再复核标定、极线与重投影误差。
6. 几何可靠则鲁棒融合；否则择优保留单视角，缺失与不确定部分明确标记。
7. 最后才做N系列坐标归一化、运动与手形分离。

API前只做基础质检，不因跨视角差异大直接删除整条轨迹，因为此时可能对应不同的手。
画面骨架标注对齐不等于三维几何对齐，API负责角色判断，不负责证明三维坐标准确。
WiLoR默认相机位置不能直接参与米制全局融合，需结合实际标定与可靠深度恢复。
操作角色规则及案例见[手选择记录](HAND_SELECTION.md)。

## 已实现的角色选择和重建

每scene一次请求联合查看三视角，每相机均匀16帧；为未覆盖的ID片段补代表帧，最多24帧。
使用原始视频读取对应帧，叠加当前WiLoR编号、骨架与**共同源时间**，不按导出mp4的播放时间配对。
GLM返回跨视角统一的物理手ID、main/support/idle/unknown及各相机片段时间段；
提示词为`configs/hand/roles.md`。允许没有主操作手，不按检测帧数替代角色判断。
相同物理手在同相机的重叠冲突、非法ID/时间会触发校验和自动修正重试。
低置信度、未指认、跟踪关联含糊的观测不进入重建；辅助手保留为独立轨迹。

使用原始深度及真实RGB/深度时间戳，在WiLoR投影关节邻域取稳健深度。
沿用子集已有的RGB/深度配准及分辨率缩放约定，用真实内参反投影；
固定WiLoR腕点相对手形，仅稳健拟合三维平移，至少5个深度锚点、其中2个掌部锚点，
逐次排除超过4cm残差的锚点。**这是深度锚定的模型手形，不是21个关节均有真实深度测量。**
原有外参仍带“同序列号人机固定相机共用外参”的假设，记录在产物里，不修改原标定。

各物理手选有效三维观测最多的相机作为首选视角。补缺测或融合前，
相机对至少5个同步有效样本，腕点距离中位数≤8cm、极线残差中位数≤4px。
逐时刻还要求时间差≤50ms；三角化采用相机对候选和重投影共识，检查正深度、
≥2°射线夹角、≤4px重投影、至少17个有效关节与掌部关键点，以及骨长和深度位置的一致性。
未通过则保留首选单视角，不能验证的另一视角也不拿来补洞。没有插值或强行平均融合。
双手分别处理，保持原始物理时间轴。

`fused.npz`的`points_base_m`为`[物理手, 时间, 21, xyz]`，`hands`/`roles`给身份与角色，
`main_hand`指明主手；`valid`为逐关节掩码，`method`为0缺测/1单视角/2三角化，
`source_camera`记录作为基础的相机，`reprojection_px`仅对三角化关节有值。
`verified_camera_pairs`及JSON文本字段`camera_pair_diagnostics`记录是否允许跨视角混用。
辅助字段`points_initial_palm`用第一帧可靠操作手的腕点/掌轴归一化，再除以初始食指MCP到小指MCP距离；
`initial_palm_in_base`和`palm_width_m`保存变换依据。未观测值始终为NaN，不能当零坐标使用。
该辅助字段带尺度归一化，**不等于训练N1**；严格N1比较见下一节。

## 正式N1候选：共同参考系与逐视角参考系

`build`在身份修正、米制三维重建之后同时生成候选，暂不自动选择胜者：

- `points_n1_shared`：从融合后主操作手的一段可靠窗口中，选一个掌心和朝向作为全scene唯一参考系。
  三视角的全部时刻、主手和辅手都使用同一个平移旋转，保留双手相对位置；不按掌宽缩放。
  优先用三角化支持的稳定窗口，没有时明确回退到单视角支持的窗口。
- `points_n1_view`：每相机、每只手在**共同参考时刻附近**估计自己的掌心和朝向，执行原N1。
  缺少可靠同时间观测时标为不可用，不悄悄改用另一个动作阶段；共同候选本身不可用时，
  此候选仍可用各自的可靠初始窗口独立生成。

`fused.npz`保存`points_n1_shared`及其掩码；各相机`trajectory/<camera>.npz`同时保存两套坐标与掩码。
共同变换为`n1_shared_pose_in_base`，共同时间为`n1_shared_anchor_timestamp_ms`，
来源`n1_shared_anchor_source`为multiview/single_view/unavailable；
逐视角变换和时间为`n1_view_pose_in_base`、`n1_view_anchor_timestamp_ms`。
无可靠主手参考系时共同输出保持NaN，不改变原始米制三维结果。
所有阈值沿用`comparison`的短窗、时间与稳定性设置，状态记入已有summary.json的`normalization_candidates`。

`compare`保留原来按相机对寻找共同时间窗的N1诊断，另外在**完全相同的源观测交集**上比较两个正式候选，
同时列出各自可用帧数与交集帧数，避免只看较少可用样本的低误差。
输出写入原`normalized.csv/.md`并绘制`candidates.png`，提取完成后的实际重建可直接复用。
共同刚体变换保留原有跨视角欧氏距离，因此不能仅靠更小的一致性误差判定最终检索性能更好。
旧`points_initial_palm`腕点/掌宽辅助字段与原N1诊断继续保留，不覆盖已有实验定义。

默认至少10个有效主手时刻、GLM标注的操作时段内覆盖率≥30%，记为`usable_single_view`。
这只是工程可用性筛选，不证明识别或姿态准确率。`main_coverage`仍以完整源时间轴为分母，
`main_active_coverage`以标注操作时段为分母。当前结果不自动接入旧轨迹训练，不覆盖原轨迹。

## 融合前的双手身份校验

位于`--stage build`内部：各相机初步三维恢复完成后、保存最终操作手与融合前执行。
配置为`trajectory.identity`，设`enabled=false`即可关闭。
按最多两个物理手身份建模；WiLoR可产生更多检测框或轨迹片段，不将它们直接当作更多真实手。
临时保留main/support/idle的低置信映射用于候选检验；检测质量、深度和跟踪歧义过滤仍生效。

每个相机对在**两只手都有效的相同源帧集合**上，比较原对应和A/B交换对应，
至少10个一对一同步样本（≤50ms）。两种假设不能使用不同覆盖范围来制造改善。
得分综合两只手腕点的三维距离与二维关节极线残差，两个手分别计分后等权汇总。
枚举全scene相机的保留/交换组合，要求全局得分至少降低50%、绝对降低至少0.5，
与其它相对对应方案有足够间隔，并且不能破坏已验证的原对应。
接受交换还要求对应参考相机上的两只手分别满足腕点距离中位数≤8cm、极线残差中位数≤8px。
这是身份校验门槛，最终三角化仍使用更严格的4px等原门槛，不因换身份而放宽。

三视角使用最少修改GLM身份的解来保留原有语义角色锚点；只有两个有效视角时，
几何本身不能决定哪一边的main/support错了，还需原GLM映射置信度差至少0.2来锚定一边。
证据不足或两种对应难以区分则不交换。左右类别只用于观察，不作为硬交换规则；
归一化后运动相似也不单独作为自动交换证据，防止双手协作同动造成误判。
主手/辅手的语义仍来自GLM，几何只能验证物理身份对应，不能替代语义识别。

接受时交换该相机观测与物理身份的映射，全scene统一的A/B及main/support角色标签保持固定。
低置信候选仅在逐帧也获得参考视角几何支持的时刻保留，不能凭整体交换而整段升级。
闲置手不进入最终操作轨迹，辅助手保留。原GLM回答与原始WiLoR文件不修改；
交换相机、前后得分、参考相机对与样本数记录在已有`trajectory/summary.json`的`identity`中，
根目录`trajectories.csv/.md`列出交换的相机，无额外标注请求。

切菜样本已自动交换037522062165：与另外两视角分别有27、37个双手同步样本，
原/交换假设使用同一批观测。腕点中位距离由约16–27cm降至2.7–4.2cm，
极线中位误差由约40–72px降至1.5–2.5px，综合得分由4.996降至0.335。
修正后主手三角化关节占完整输出时间轴关节数的约37.8%；这不是37.8%的准确率。
其它两组未交换。新增5项身份校验测试，与既有手提取/重建/N1测试合计17项通过。

## 共同时间锚点与严格N1比较

独立入口`--stage compare`仅读取每相机已生成的三维结果，不重新提取、不请求GLM，亦不修改融合门槛。
复用训练的`palm_pose_from_landmarks`和`hand_xyz_features(..., 'palm_relative')`：
初始掌心为原点，初始掌部朝向为坐标轴，全轨迹只做同一个平移旋转，**不缩放**。
比较的是深度锚定并经相机外参转换的三维点，不是二维像素轨迹。

按已有物理手ID和原始时间戳寻找同步有效观测，各源帧最多匹配一次，所有视角的时间跨度≤50ms。
优先选三视角共同短窗；不存在时分别寻找可比较相机对，不强求三视角齐全。
取最早的连续3个同步有效样本，逐视角相邻间隔≤250ms，掌心相对中间帧漂移≤3cm、
掌部转角≤15°，以中间帧各视角自己的掌心位姿为固定归一化参考。
各相机锚点位于同一物理时刻附近，不取各自第一帧。这个锚点是坐标系参考，**不必是任务语义上的开始**，
比较保留锚点前后的共同有效观测，无需再让GLM精确标注动作起点。
找不到稳定短窗、或共同有效帧不足5个，显式标为不可比较，不插值、不DTW、不逐帧旋转拟合。
短窗稳定只能筛除一部分抖动，不能证明锚点姿态与物理手身份正确。

报告包括：归一化前/后的21点欧氏RMSE、腕点相对锚点的运动RMSE和位移余弦、
腕点相对关节的RMSE、共同样本数、锚点时间和相机组范围。静止手的位移余弦未定义。
仅有5或14个配对样本的结果证据有限，不能代表完整操作。
归一化后相似说明运动或手形可能一致，**不能代替真实标定检查，不能据此放开三角化**。

身份修正前的诊断结果：cfg2/task_0020的一对视角21点RMSE由34.22cm降至2.44cm（14帧）；
切菜主手036422060215—104122061850由2.46cm降至2.00cm（115帧），
涉及037522062165的两对仍为15.79cm和10.91cm。
该视角所选主手的WiLoR左右类别与另外两视角相反，接触图也提示主手/辅手对应需复核；
因此不能直接将剩余差异归因为归一化无效或姿态模型误差，也不能凭左右类别自动交换角色。
cfg3/task_0004的一对视角由22.72cm降至8.95cm（41帧），仍有明显差异。
详细结果及叠加曲线见pilot/normalized.md和各scene/trajectory/normalized.png。
当前报告已重算为身份修正后的结果；切菜涉及037522062165的两对主手N1误差约4.85cm、1.57cm。
因低置信候选的逐帧几何验证改变了可用样本，不能把修正前后整表RMSE当作固定样本的受控比较；
身份交换的决策证据以summary.json中固定同批样本的前后得分为准。

## 本机首轮结果（2026-10-20）

RTX 4060 Laptop 8GB、Torch 2.7.0+cu126：9路视频共1305帧，提取总耗时108.73秒（含加载），
逐相机耗时合计约97.27秒，PyTorch峰值allocated为2.459GiB。日志保存在pilot/run.log。
这只是本批小样本推理测速，不代表全量速度或实时保证。
跨帧合批另在cfg2/task_0109/scene_4的036422060215相机前32帧测试：单帧/单手批量为11.78帧/秒，
双批量均为4时为22.08帧/秒（约1.87倍，不含视频解码和叠加写入）。两者均检出65只手，检测框一致，
二维关键点平均差异0.043像素、最大0.172像素；不能据此假定不同视频或与视频训练共享GPU时有相同加速比。

- cfg2/task_0109/scene_4：三个相机对的极线残差中位数约2.13/2.96/2.20像素，几何一致性相对较好。
- cfg2/task_0020/scene_4：约28.62/32.64/7.42像素；风扇区域可见疑似背景误检。高检测覆盖不等于准确。
- cfg3/task_0004/scene_4：约111.50/207.07/32.25像素；不能直接跨视角平均融合。
  需要区分标定适用性、时间配对、手身份与模型误差，不把全部残差归因于WiLoR。
- 总计51个临时track片段，不代表51只手。cfg3的104122063550源时间戳出现3次超过0.5秒的间隔，
  最大约0.737秒，触发当前保守跟踪器的重新编号；这不是模型骨架本身跳变的证明。

现已在上述三组样本完成真实GLM标注和全链路重建；质量报告见pilot/trajectories.md。
角色标注能区分闲置手与辅助手，但稀疏帧仍可能遗漏身份切换，需要复核叠加视频。
目前跨视角深度位置偏差明显，严格筛选下可能完全退回最佳单视角，不能宣称多视角融合已提升准确率。

身份修正前的完整链路验证：主手有效时刻分别为97/120、152/228、71/97；
标注操作时段内覆盖率约80.8%、86.4%、93.4%。三组均未通过严格融合门槛，融合比例为0，
切菜辅助手独立保留。GLM已知费用0.0686156元（按当前配置单价估算，含校验重试）；
另1次HTTP 500无usage，费用未知。9份原始WiLoR结果复用，无额外提取。
复跑验证确认90个源文件的大小/修改时间及9份原始NPZ未变，未增加API响应。
针对跟踪、身份冲突、手角色选择、深度离群、三角化、时间错位、拒绝错误补帧等，以及密钥管理，
27项测试通过；几何正确的合成样本能进入融合分支，真实样本未满足条件则保持单视角。
