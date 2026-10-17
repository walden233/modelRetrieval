# 第三章语义检索：实现与运行

更新：2026-10-17。本文件覆盖上一版实现约定。配置为 [ch3.json](../../configs/semantic/ch3.json)，入口为 [run.py](../../runs/semantic/run.py)。当前代码只使用 schema_version=3；上一版产物归档，不能混入新实验。原视频、原始数据、轨迹与视频训练配置不修改。

## 1. 实验设计

| 组别 | 任务表示 | 提取方式 |
|---|---|---|
| S0 | 一句主动作、操作对象和目标，不提颜色及无关外观 | 同次联合输出 baseline_description |
| S1 | 自然描述，通常 1–3 句，可保留有用对象外观 | 与 S0、S2–S5 同次提取 |
| S2 | 同次 S1 原文＋原四维标签 | 保存完整文本，不在使用时读取 S1 |
| S3 | 对象、操作、可选结果分离，保留对象关系 | 模型直接输出 |
| S4 | 动作优先、必要功能角色、弱化外观与类别 | 模型直接输出 |
| S5 | 对象、顺序操作、结果的原始语义 JSON | 固定序列化模型的 structured 子对象 |

S0 是用当前模型和输入重新运行的旧式提示策略，不是将历史 GLM-4.6V 的缓存当作控制基线。S0–S5 在同一次请求内基于相同视频帧生成；S0 单独字段采用简述约束。比较包含信息取舍与组织方式差异，不单独归因于句长。论文可只保留 S0/S1 之一。

## 2. 提示词与输出边界

[video.md](../../configs/semantic/prompts/video.md)、[scene.md](../../configs/semantic/prompts/scene.md) 统一只要求任务内容。

- 不要求理由、解释、不确定性说明、可见性报告、置信度或证据引用。
- 联合视频响应只有 baseline_description、description、labels、object_action_text、action_focus_text、structured。六组同次生成，S2由同次description与labels组织而成。
- S0 使用 baseline_description；scene 响应只有 description，可为 null。
- 无法确定的动作、结果直接省略；structured.result 设 null，不生成“无法确认是因为……”等文本。
- S3/S4 忽略无关颜色、花纹、品牌、装饰和样式。S4 还泛化无关材质与对象类别；任务本身必需的区分可以保留，无需输出理由。
- 人类关注演示者；机器人关注机械臂与末端执行器。无关人物、键盘和录制活动不进入描述，实际人机交互保留动作归属。

程序仍记录身份、来源、时间、费用、schema 错误和重试。这些工程记录不作为模型生成的任务描述，也不参与文本编码。原始 API 响应保留用于费用与失败追溯。

## 3. 范围与冻结清单

初始 sources 指向 s44 的 cfg2/cfg3 视频划分清单。第一次 prepare 将相机对 ID、顺序、划分和 data_root 存入语义自己的 scope.json；之后使用该快照，不再依赖 batch 名，也不重新读取上游划分来悄悄改变实验。

| 范围 | scene | 视频 |
|---|---:|---:|
| cfg2/train | 258 | 1,032 |
| cfg2/val | 50 | 200 |
| cfg2/test | 120 | 480 |
| cfg3/test | 128 | 512 |
| 合计 | 556 | 2,224 |

每 scene 冻结两个相机、四段人机视频。prepare 只检查文件存在与 stat，无解码、无视频哈希。身份包含 cfg/task/scene/camera/role，cfg2 与 cfg3 的相同数字 task 不自动合并。

## 4. 集中产物与独立分组

所有本轮语义产物集中在：

```text
artifacts/semantic/v2/
  scope.json                  # 划分快照、成本子集、增量来源
  videos.jsonl                # 视频索引，无 split/status/哈希
  params.json                 # 配置、执行记录、已移除组及当前展示组
  jobs/
    glm20[-rN]/               # S0–S5 联合提取，正常每视频一次请求
    glm32/ glm16/ glm8/        # 成本实验的联合提取
    scene-s4[-rN]/            # 选定S4作为来源的scene汇总（其他S同理）
  S0/ ... S5/
    params.json               # 本组请求协议与来源 job
    annotations.jsonl         # 本组完整任务文本；S2另存标签，S5另存结构
    embeddings/bge1.npz       # 本组向量与接受的标注修订
    evaluations/cfg2/         # 本组 metrics.json、rankings.jsonl
    evaluations/cfg3/
    analysis.json
    reviews.csv
  evaluations/cfg2/ cfg3/     # 从当前独立组汇总的比较数据
  cost/
    glm32[-rN]/S0/ ... S5/         # 成本变体也按表示分开
    glm16/ glm8/
    summary.json
  teachers/S4/cfg2.npz       # 按来源分组保存scene文本与向量，其他S/cfg同理
  charts/                    # 柱状图、视角图、案例板及绘图数据 JSON
  plots.json                 # 本目录图源索引，无哈希
  analysis.json reviews.csv  # 当前组汇总分析与人工核验
  results.csv report.html
  history/                   # 上一版产物和被新协议替代的组
```

没有 S1 目录时，S2 仍可编码、评估；S3/S4/S5 不读取 S0 或 S1。共享 job 只是请求账本与原始响应，不是组间的运行依赖。删除一组不会抹除当时已经发生的共同 API 费用。

## 5. 归档与协议变化

首次 prepare 自动将上一版根目录的 glm*/scene* job 移至 history/jobs，将 artifacts/runs/retrain/s44/semantic 移至 history/s44，并从 s44/plots.json 移除语义图源项，保留另外两模态项。上一版 scope 中的固定范围与成本子集继续沿用；旧响应、向量、评估仅作历史记录，不转成新格式冒充新实验。

当前模型或 prompt 改变时，创建可读 job 修订，例如 glm20-r2。同一表示改用不同协议时归档该组旧产物，防止旧评估残留。编码参数变化只生成新的向量修订，不触发 VLM 重标注。

## 6. 标注、恢复与工程统计

主实验 GLM-5.3-Flash，均匀 20 帧，覆盖首尾；记录实际索引、时间戳、尺寸、抽帧与编码耗时。主提取并发 4，响应即时写入 requests.jsonl，再验证和接受。六组同次生成，正常全量合计 2,224 次请求；修复另计。--only 不改变输出协议，仅决定哪些组保存、编码与评估。

缓存包含视频身份/stat、模型、prompt 原文、schema、完整请求参数、图像设置与帧数。已有正确响应可本地重解析；标注结束后将失败ID、接受数量写入job的summary.json及主实验params.json，单条失败仅警告，不阻断后续环节。全部失败仍明确报错。中断且计费结果未知的请求须显式 --retry-unknown。修复只要求返回完整任务 JSON，不要求解释错误。

每次请求记录 usage、请求时延、重试、已知折算费用和未知费用；每次执行记录新请求、缓存命中和墙钟耗时。推理 token 不重复计入输出费用，缺失字段保持 null。价格折算不等于平台对账账单，缓存读取也不算 API 延迟。

## 7. 文本编码

使用本地 BAAI/bge-m3，固定 revision、CLS 池化、L2 归一化，max_length=2048。各组独立 NPZ 保存 sample_id、标注 attempt、来源、长度、向量和编码设置。超长明确报错，不静默截断。

## 8. 评估与公平集合

默认各 cfg/test，人→机为主，同时输出反方向、task 与 scene 多正样本指标。cfg2 正式 query/gallery 各 240，cfg3 各 256。指标为 R@1/5/10、MRR、NDCG@10；MPR-all 保留在原始数据，不进入主图。

只重算 --only 指定组，其他兼容结果可合并。默认允许少量标注失败：保存成功项，继续编码、评估、分析、绘图和报告；六组使用共同可用集合，metrics保留partial、coverage和missing_ids，图表与报告标注覆盖率。--allow-partial 保留兼容旧命令，无需额外指定。某组缺失不影响另一组独立评估。合并时检查 ID、候选范围、文本编码、seed、split、模型／输入设置与联合prompt；不同协议或集合的旧组记录到excluded，不混入当前主图，也不阻止所选组独立评估。

## 9. 分析与论文展示

每组分别分析 HH、RR、同相机 HR、跨相机 HR，相同 task 不同 scene 与不同 task 为对照。S2 保存自己的标签，标签统计不依赖 S1。两者均存在且评估集合一致时再计算 S1→S2 排名变化。

人工核验表由程序生成空项，模型不输出核验理由。展示同视频采样帧和当前选中的表示，支持只展示 S0/S2/S3/S4/S5 或 S1/S2/S3/S4/S5。失败案例关联已有视频／轨迹时，只在真实公共 query/gallery 上重排，不将互补机会当作融合收益。

## 10. 成本实验

冻结 cfg2/test 的 55 个不同 task，每 task 一个 scene，共 220 段视频；各模型、帧数、表示使用共同集合。每 task 一个 scene，因此 task/scene 正样本相同。GLM32/16/8 共 660 次正常请求；显式加 Qwen16 后 880 次。GLM16 在帧数与厂商轴之间共享，主20帧不能冒充16帧响应。

成本实验保持与主实验相同的联合 S0–S5 输出要求，六组均可评估。即使只呈现 S4，也不能减少模型输出字段来人为降低费用。费用、时延、吞吐、成功和重试均记录。存在失败时，各模型／帧数与六组使用共同可用视频子集，显式保留partial和目标220段的覆盖率；某变体全部失败或没有可用人机集合时仍报错。--fresh-benchmark 会产生新的付费调用。

## 11. 独立移除与恢复

--stage remove --only S1 只移除 S1 的活动目录和成本变体下的 S1，更新比较、分析、图表和报告；同时在 params.json 标记停用，后续默认完整命令不会自动重新创建它。S2 完整文本与其他组保持不变，已导出的教师仍保留来源记录。费用账本继续保存。

显式 --only S1 --stage all 可以恢复 S1，同协议成功的联合响应可复用。若只想隐藏而不删除，使用 --stage plot --only S0,S2,S3,S4,S5，随后 --stage report。手动删除目录后可以直接重绘；如需后续默认完整命令也忽略该组，使用 remove 入口或修改配置的 representations。

## 12. scene 级汇总与训练教师

scene 阶段通过 --source S0…S5 选一种表示，默认读配置 scene.source=S5。只读取该组在同一 scene 四段视频的独立任务文本，不混合不同 S。额外调用一次文本模式模型生成共享任务描述；S5 的输入文本为语义 JSON。不同来源使用独立的 scene-sN 请求缓存与 teachers/SN/ 导出目录。模型只返回 description 或 null，不返回分歧、来源解释或不确定理由；程序保存四份来源和修订，并根据 null 生成 usable 标记。

默认覆盖当前冻结范围的 cfg2/cfg3 全部 scene，教师池与训练 batch 无关。--run 指定视频 run 时按其实际 train 划分导出专用 NPZ。训练只能使用对应 train scene，不能借用 val/test 教师；不可用目标只屏蔽语义项，不删除视频样本。当前只准备教师数据，没有新增视频训练语义损失。

## 13. 图源与重绘

绘图数据和 plots.json 在 PNG 渲染之前保存。重绘优先使用保存的图源，缺失时从各组评估、analysis 或 cost summary 恢复；移除组会同步从图源中过滤。所有图片入口为 runs/redraw.py --root artifacts/semantic/v2，无 API。原 runs/redraw.py --batch s44 继续负责原两模态。

## 14. 已实现的统一入口

从项目根目录，conda activate torch2 后执行。GLM key 读 ZHIPU_API_KEY 或 GLM_API_KEY，Qwen 仅在显式选择时读取 DASHSCOPE_API_KEY。无需 --batch。

```bash
# 范围与缓存准备，无 API；首次会归档上一版产物
python runs/semantic/run.py --stage prepare

# 完整 S0–S5 标注、编码、评估、分析、绘图、报告
python runs/semantic/run.py

# 成本实验，以及可选的 Qwen16
python runs/semantic/run.py --stage cost
python runs/semantic/run.py --stage cost --providers glm,qwen

# 各环节独立；可选定任意组
python runs/semantic/run.py --stage annotate --only S0
python runs/semantic/run.py --stage encode
python runs/semantic/run.py --stage eval
python runs/semantic/run.py --stage eval --only S2,S5
python runs/semantic/run.py --stage analyze
python runs/semantic/run.py --stage plot
python runs/semantic/run.py --stage report

# 论文只展示其中一个基线；保留其他组原始数据
python runs/semantic/run.py --stage plot --only S0,S2,S3,S4,S5
python runs/semantic/run.py --stage report

# 单独移除一组；今后默认命令也忽略该组
python runs/semantic/run.py --stage remove --only S1
# 如需恢复，显式选中该组
python runs/semantic/run.py --only S1

# scene 教师共享池：示例选S4；省略source则采用配置中的S5
python runs/semantic/run.py --stage scene --source S4
python runs/semantic/run.py --stage scene --source S4 --run artifacts/runs/retrain/s44/video/E3

# 增量视频列表或清单，不扩大固定主评估范围
python runs/semantic/run.py --stage annotate --videos extra_videos.txt
python runs/semantic/run.py --stage annotate \
  --add-manifest artifacts/manifests/retrain/s46/cfg2/split_manifest_video.json

# 现有两模态公共候选上的失败分析
python runs/semantic/run.py --stage analyze \
  --video-run artifacts/runs/retrain/s44/video/E3 \
  --trajectory-run artifacts/runs/retrain/s44/trajectory/T2

# 全部语义图片重绘，无 API
python runs/redraw.py --root artifacts/semantic/v2

# 小规模工程检查，不作为论文结果
python runs/semantic/run.py --limit 8 --allow-partial
```

主实验、成本、教师为三个独立入口，主入口不会运行另两项。原 --suite chapter 已移除。

正常续跑复用成功标注；--force 重标选中的成功视频并保留全部尝试。eval/analyze/plot/report 不会偷偷补 VLM 调用。正式实验不要带 --limit；存在缺失的结果不能当作全量结果，补齐后重跑会覆盖评估、图表和报告。

## 15. 协议切换与验收

本次将独立 S0＋五组联合改为六组联合，video_p5 与 joint_s0_s5_p1
共同区分旧缓存。旧独立 S0 已成功的 2,212 条、旧五组联合结果及其费用账本
保留在原 jobs 修订，不冒充六组联合结果。新请求使用新的 glm20-rN；投影
保存时自动归档被替换组的旧向量、评估，教师只归档对应来源组。
此前案例图与评估是旧协议小规模测试，主入口完成新协议评估后会重新生成。

本次用本地 stub 验证六组单请求、断点续跑、独立保存／移除、完整主流水线、
六种教师来源及实际 train 过滤、成本三帧数组合，专项30项通过。
全库检查286项通过，12项失败仍为此前的轨迹／划分相关问题。没有运行新的付费标注。
历史 API 测试与费用以保留的 jobs 账本为准，不能当作当前协议验收或正式论文结果。

schema 合格不等于事实正确。选定论文案例时，在对应 S 目录 reviews.csv 将
kind 设为 case 或 failure_case，再运行 analyze 和 plot，保留人工核验步骤。

2026-10-17：修复少量标注失败阻断后续环节。保留原提示词、schema与job签名，现有glm20-r3的成功标注可直接续用；不放宽任务内容校验。回归覆盖接口失败、schema失败、共同评估集合、失败补齐后的全量重评估与成本实验公平子集。

本轮专项33项通过；全库291项通过、12项失败，仍为此前的轨迹／划分相关失败。未调用付费API。
