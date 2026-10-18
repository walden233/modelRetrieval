# 第三章语义检索：实现与运行

更新：2026-10-17。本文件覆盖上一版实现约定。配置为 [ch3.json](../../configs/semantic/ch3.json)，入口为 [run.py](../../runs/semantic/run.py)。当前代码只使用 schema_version=3；上一版产物归档，不能混入新实验。原视频、原始数据、轨迹与视频训练配置不修改。

## 1. 实验设计

| 组别 | 任务表示 | 提取方式 |
|---|---|---|
| S1 | 原S0/S1合并，1–2句自然简述：对象、主要动作、可见结果 | 模型输出description，标签独立 |
| S2c | 同次S1原文＋原四维标签，拼接后一次编码 | 独立保存描述、标签、完整文本 |
| S2w | 同次描述、标签分别编码，0.7:0.3向量融合 | 独立保存两份文本、分量向量和融合向量 |
| S3 | 对象、操作、可选结果分离，保留对象关系 | 模型直接输出 |
| S4 | 功能角色与操作过程，保留任务机制、泛化具体对象名称 | 模型直接输出，与S3保持相同动作、顺序和结果 |
| S5 | 对象、顺序操作、结果的原始语义 JSON | 固定序列化模型的 structured 子对象 |

六组基于同一次响应。S1允许理解任务必要的外观信息，省略无用的颜色、背景、衣着和偶然动作。S2c/S2w使用完全相同的描述与标签，仅改变编码和融合方式；S3/S5提取设计保持原样；S4使用功能角色替代具体对象名称，保持与S3相同的动作、顺序和结果，仅为明确操作机制保留必要对象类别或结构。提示词附有“从笔筒取笔”的S3/S4对照示例，仅示范抽象方式，当前视频事实仍须来自当前帧。比较同时涉及信息取舍和组织方式，不将全部差异归因于序列化格式。旧S0/S2退役归档，不进入新主图。

## 2. 提示词与输出边界

[video.md](../../configs/semantic/prompts/video.md)、[scene.md](../../configs/semantic/prompts/scene.md) 统一只要求任务内容。

- 不要求理由、解释、不确定性说明、可见性报告、置信度或证据引用。
- 联合视频响应只有description、labels、object_action_text、action_focus_text、structured。标签只在labels输出，不附加到S1或其他描述。
- S2c/S2w在本地使用同次description与labels；scene响应只有description，可为null。
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

每scene冻结两个相机、四段人机视频。上述2,224段为当前完整标注目标，共556个scene。cfg2主评估合并train/val/test，限冻结清单内1712视频；cfg3仍按test清单。配置annotation.splits控制默认范围；显式annotate --split train等可单独补齐指定划分，不扩大冻结来源。

prepare只检查文件存在与stat，无解码、无视频哈希。身份包含cfg/task/scene/camera/role，cfg2与cfg3相同数字task不自动合并。

## 4. 集中产物与独立分组

所有本轮语义产物集中在：

```text
artifacts/semantic/v2/
  scope.json                  # 划分快照、成本子集、增量来源
  videos.jsonl                # 视频索引，无 split/status/哈希
  params.json                 # 配置、执行记录、已移除组及当前展示组
  jobs/
    glm16[-rN]/               # 当前联合提取，正常每视频一次请求
    glm32/ glm8/              # 成本实验；同协议16帧复用主job
    scene-s4[-rN]/            # 选定S4作为来源的scene汇总（其他S同理）
  S1/ S2c/ S2w/ S3/ S4/ S5/
    params.json               # 本组请求协议与来源 job
    annotations.jsonl         # 完整任务文本；S2c/w另存独立描述、标签
    embeddings/bge1.npz       # 本组向量；S2w含描述、标签分量向量
    evaluations/cfg2/         # 本组 metrics.json、rankings.jsonl
    evaluations/cfg3/
    analysis.json
    reviews.csv
  evaluations/cfg2/ cfg3/     # 从当前独立组汇总的比较数据
  cost/
    glm32[-rN]/S1/ ... S5/     # 同样含S2c、S2w独立目录
    glm16/ glm8/
    summary.json
  teachers/S4/cfg2.npz       # 按来源分组保存scene文本与向量，其他S/cfg同理
  charts/                    # 柱状图、视角图、案例板及绘图数据 JSON
  plots.json                 # 本目录图源索引，无哈希
  analysis.json reviews.csv  # 当前组汇总分析与人工核验
  results.csv report.html
  history/                   # 上一版产物和被新协议替代的组
```

没有S1目录时，S2c/S2w仍可独立编码、评估；所有组都不读取其他组的目录。共享 job 只是请求账本与原始响应，不是组间的运行依赖。删除一组不会抹除当时已经发生的共同 API 费用。

## 5. 归档与协议变化

首次 prepare 自动将上一版根目录的 glm*/scene* job 移至 history/jobs，将 artifacts/runs/retrain/s44/semantic 移至 history/s44，并从 s44/plots.json 移除语义图源项，保留另外两模态项。上一版 scope 中的固定范围与成本子集继续沿用；旧响应、向量、评估仅作历史记录，不转成新格式冒充新实验。

当前模型或prompt改变时，创建可读job修订，例如glm16-r2。旧S0/S2在prepare时移至history，其他组在新标注投影时归档旧产物。同一表示改用不同协议时归档该组旧产物，防止旧评估残留。编码参数变化只生成新的向量修订，不触发 VLM 重标注。

## 6. 标注、恢复与工程统计

主实验GLM-5.3-Flash，均匀16帧，覆盖首尾；记录实际索引、时间戳、尺寸、抽帧与编码耗时。主提取并发按配置（当前48），响应即时写入 requests.jsonl，再验证和接受。六组共用同次响应，当前完整清单正常合计2224次请求；修复另计。--only 不改变输出协议，仅决定哪些组保存、编码与评估。

缓存包含视频身份/stat、模型、prompt 原文、schema、完整请求参数、图像设置与帧数。已有正确响应可本地重解析；标注结束后将失败ID、接受数量写入job的summary.json及主实验params.json，单条失败仅警告，不阻断后续环节。全部失败仍明确报错。中断且计费结果未知的请求须显式 --retry-unknown。修复只要求返回完整任务 JSON，不要求解释错误。

主实验、成本变体和scene教师在首轮标注全部结束后，自动补重试一轮失败项，再进入下一阶段。成功项从缓存复用；补重试仍按单条max_attempts处理，保留两轮请求、修复反馈与费用记录，不无限重试。即使首轮全部失败也先补重试；最终全部失败才报错。

每次请求记录 usage、请求时延、重试、已知折算费用和未知费用；每次执行记录新请求、缓存命中和墙钟耗时。推理 token 不重复计入输出费用，缺失字段保持 null。价格折算不等于平台对账账单，缓存读取也不算 API 延迟。

## 7. 文本编码

使用本地BAAI/bge-m3，固定revision、CLS池化、L2归一化，max_length=2048。S2c将描述与规范标签文本拼接后编码。S2w分别编码描述与标签，先各自L2归一化，再计算`normalize(0.7 * description_vector + 0.3 * label_vector)`。这里融合的是向量，不是检索分数。

各组独立NPZ保存sample_id、标注attempt、来源、长度、向量与编码设置；S2w在同一文件保存S2w_description和S2w_labels两份分量向量。改变权重只生成新向量修订，不重标注；其他组不受影响。S2w长度记录两输入token数之和，每部分分别检查max_length；超长明确报错，不静默截断。

## 8. 评估与公平集合

默认cfg2/all、cfg3/test，人→机为主，同时输出反方向、task与scene多正样本指标；论文主体讨论task检索。cfg2完整清单109 task、428 scene，query/gallery各856；cfg3目标各256，失败时按共同可用集合并标partial。evaluation.split支持按cfg配置，--split显式覆盖。指标为 R@1/5/10、MRR、NDCG@10；MPR-all 保留在原始数据，不进入主图。

只重算 --only 指定组，其他兼容结果可合并。默认允许少量标注失败：保存成功项，继续编码、评估、分析、绘图和报告；六组使用共同可用集合，metrics保留partial、coverage和missing_ids，图表与报告标注覆盖率。--allow-partial 保留兼容旧命令，无需额外指定。某组缺失不影响另一组独立评估。合并时检查 ID、候选范围、文本编码、seed、split、模型／输入设置与联合prompt；不同协议或集合的旧组记录到excluded，不混入当前主图，也不阻止所选组独立评估。

## 9. 分析与论文展示

每组分别分析 HH、RR、同相机 HR、跨相机 HR，相同 task 不同 scene 与不同 task 为对照。S2c/S2w各自保存标签，标签统计不依赖S1。存在兼容评估时分别计算S1→S2c、S1→S2w排名变化。

人工核验表由程序生成空项，模型不输出核验理由。展示同视频采样帧和当前选中的表示，支持通过--only选择任意当前组。失败案例关联已有视频／轨迹时，只在真实公共 query/gallery 上重排，不将互补机会当作融合收益。

## 10. 成本实验

从cfg2完整清单冻结60个不同task，每task一个scene，共240段视频；各模型、帧数、表示使用共同集合。每 task 一个 scene，因此 task/scene 正样本相同。默认GLM32/16/8、Qwen16、DeepSeek16，缓存全空时共1200次正常请求（不含重试）。GLM16在帧数与厂商轴之间共享；当前主实验同为16帧，相同完整协议和视频可复用主job。复用不重复计费，缓存读取不作为新API延迟；需要重新实测时延时使用--fresh-benchmark。旧20帧响应不能复用。厂商独立密钥池、Qwen专属地址与DeepSeek分时估价见[配置说明](../../configs/semantic/README.md)。

成本实验保持与主实验相同的联合输出要求，六组均可评估。即使只呈现 S4，也不能减少模型输出字段来人为降低费用。费用、时延、吞吐、成功和重试均记录。存在失败时，各模型／帧数与六组使用共同可用视频子集，显式保留partial和目标240段的覆盖率；某变体全部失败或没有可用人机集合时仍报错。--fresh-benchmark 会产生新的付费调用。

## 11. 独立移除与恢复

--stage remove --only S1 只移除 S1 的活动目录和成本变体下的 S1，更新比较、分析、图表和报告；同时在 params.json 标记停用，后续默认完整命令不会自动重新创建它。S2c/S2w完整文本与其他组保持不变，已导出的教师仍保留来源记录。费用账本继续保存。

显式 --only S1 --stage all 可以恢复 S1，同协议成功的联合响应可复用。若只想隐藏而不删除，使用 --stage plot --only S1,S2c,S3,S4,S5，随后 --stage report。手动删除目录后可以直接重绘；如需后续默认完整命令也忽略该组，使用 remove 入口或修改配置的 representations。

## 12. scene 级汇总与训练教师

scene阶段通过--source选S1、S2c、S2w、S3、S4、S5中的一种表示，默认读配置 scene.source=S5。只读取该组在同一 scene 四段视频的独立任务文本，不混合不同 S。额外调用一次文本模式模型生成共享任务描述；S5 的输入文本为语义 JSON。不同来源使用独立的 scene-sN 请求缓存与 teachers/SN/ 导出目录。模型只返回 description 或 JSON null，不返回分歧、来源解释或不确定理由；程序保存四份来源和修订。导出的NPZ仅保留非null且通过校验的教师，usable均为true；被排除的scene保存在metadata.excluded_scenes，原始响应仍在job账本。--repair只重新生成违规、null及显式--scene-ids选定项，其余合规教师记录reused_from后复用到新版本，避免全量重标。

默认汇总当前完整清单范围的556个scene，教师池与训练batch无关。--run指定视频run时按其实际train划分导出专用NPZ；训练视频已纳入当前标注范围；失败项需先补齐。训练只能使用对应 train scene，不能借用 val/test 教师；不可用目标只屏蔽语义项，不删除视频样本。当前只准备教师数据，没有新增视频训练语义损失。

## 13. 图源与重绘

绘图数据和 plots.json 在 PNG 渲染之前保存。重绘优先使用保存的图源，缺失时从各组评估、analysis 或 cost summary 恢复；移除组会同步从图源中过滤。所有图片入口为 runs/redraw.py --root artifacts/semantic/v2，无 API。原 runs/redraw.py --batch s44 继续负责原两模态。

## 14. 已实现的统一入口

从项目根目录，conda activate torch2 后执行。GLM、Qwen、DeepSeek分别读取configs/semantic/keys.local.json、qwen.keys.local.json、deepseek.keys.local.json；文件及锁文件被Git忽略，权限600。明确欠费时永久转入对应overdue并切换，全部耗尽则停止，补新key后续跑。配置密钥池后不回退环境变量；未配置密钥池的provider仍兼容环境变量。无需--batch。

```bash
# 范围与缓存准备，无 API；首次会归档上一版产物
python runs/semantic/run.py --stage prepare

# 默认完整清单2224段，16帧；cfg2/all、cfg3/test评估
python runs/semantic/run.py

# 完整成本实验：GLM32/16/8、Qwen16、DeepSeek16
python runs/semantic/run.py --stage cost
# 只允许GLM补标，其余完成变体从缓存比较
python runs/semantic/run.py --stage cost --providers glm

# 各环节独立；可选定任意组
python runs/semantic/run.py --stage annotate --only S1
python runs/semantic/run.py --stage encode
python runs/semantic/run.py --stage eval
python runs/semantic/run.py --stage eval --only S2c,S2w,S5
python runs/semantic/run.py --stage analyze
python runs/semantic/run.py --stage plot
python runs/semantic/run.py --stage report

# 论文可只展示一种标签编码方式；保留其他组原始数据
python runs/semantic/run.py --stage plot --only S1,S2c,S3,S4,S5
python runs/semantic/run.py --stage report

# 单独移除一组；今后默认命令也忽略该组
python runs/semantic/run.py --stage remove --only S1
# 如需恢复，显式选中该组
python runs/semantic/run.py --only S1

# scene 教师共享池：示例选S4；省略source则采用配置中的S5
python runs/semantic/run.py --stage scene --source S4
python runs/semantic/run.py --stage scene --source S4 --run artifacts/runs/retrain/s44/video/E3

# 增量标注训练集，之后才能为训练run生成教师
python runs/semantic/run.py --stage annotate --cfg cfg2 --split train

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

当前提示video_p9与description_labels_p1校验策略取代旧video_p5。S0/S1合并为S1，S2拆成S2c/S2w；主实验改16帧；2026-10-18范围调整为完整清单2224段标注，主评估cfg2/all、cfg3/test，成本为60 task/240视频。旧glm20-r3成功项、请求原文和费用账本保留，不能据此跳过新协议调用。prepare归档退役S0/S2；新标注投影时归档被替换组的旧文本、向量和评估，以及对应来源的教师。

54项专项测试通过，覆盖默认范围与训练集增量、六组联合响应、恢复续跑、独立存储／移除、加权向量与权重改动缓存、成本公平集合、教师来源、部分失败后继续评估，以及密钥欠费切换、并发持久化、全部耗尽后的停止与续跑。本地真实BGE-M3也验证了两份1024维分量向量、7:3融合、归一化和缓存。全库312项通过，12项失败仍为此前的轨迹／划分相关问题。本轮未运行付费API；历史API记录不能当作当前16帧协议的正式结果。

schema合格不等于事实正确。选定论文案例时，在对应S目录reviews.csv将kind设为case或failure_case，再运行analyze和plot，保留人工核验。历史20帧案例见[失败分析](FAILURES.md)，不能与新结果直接混用。

成本抽样设置cfg/split/tasks/seed改变时自动更新scope.cost，旧cost产物与图表归档到history/cost-*，保留原始请求job缓存。厂商或帧数改变不会重抽子集。

仅续跑未完成的Qwen和DeepSeek：`python runs/semantic/run.py --stage cost --providers qwen,deepseek --retry-unknown`。显式providers只控制这些厂商的API请求，GLM仅从完整缓存合并。各厂商并发与Qwen实际限额见[配置说明](../../configs/semantic/README.md)；并发设置不改变缓存签名，原始API时间不含节流等待。
