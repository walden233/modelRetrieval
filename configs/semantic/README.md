# 第三章语义实验

配置为 [ch3.json](ch3.json)，当前 schema_version=3。全部产物统一在 `artifacts/semantic/v2`，无需指定 batch。

```bash
conda activate torch2
# S1/S2c/S2w/S3/S4/S5：标注、编码、评估、分析、绘图、报告
python runs/semantic/run.py
# 固定60 task、240视频：GLM32/16/8、Qwen16、DeepSeek16
python runs/semantic/run.py --stage cost
# 选定一种表示，汇总scene教师文本与向量
python runs/semantic/run.py --stage scene --source S4
# 修复教师：复用合规旧文本，仅重新生成违规、null和缺失项
python runs/semantic/run.py --stage scene --cfg cfg2 --source S3 --repair
# 单独移除S1，其他组保持独立可用
python runs/semantic/run.py --stage remove --only S1
# 只重绘语义图片，无API
python runs/redraw.py --root artifacts/semantic/v2
```

主模型 GLM-5.3-Flash，均匀16帧。默认标注冻结清单内cfg2全部1712段和cfg3的512段，共2224段。主评估默认cfg2合并train/val/test（109 task、428 scene，人机各856段），cfg3仍按test清单；evaluation.split按cfg配置，CLI --split可显式覆盖。论文主体讨论task检索，scene指标仍保留。S1合并原S0/S1为1–2句自然简述，保留必要外观信息，省略无用颜色、背景、衣着和偶然动作；标签单独输出。S2c将同次描述与标签拼接后编码；S2w分别编码、分别归一化，再按描述0.7＋标签0.3加权并归一化。S3/S5提取设计保持原样；S4改用功能角色描述对象，保持与S3相同的动作、顺序和结果，保留任务机制差异。

各组独立保存文本、向量和评估，S2w还保存两部分向量，不依赖S1目录。模型只输出任务内容；程序记录时间、费用、来源和失败。

教师使用scene_p6：只保留人机双方支持的具体操作，不输出视角对比、选择理由、推测或非共享结果。导出的`teachers/<S>/<cfg>.npz`仅含通过校验的非null教师，排除scene记录在metadata.excluded_scenes；训练仍需按实际run的train划分筛选。`--repair`在新prompt版本中复用合规旧文本，并逐条记录`reused_from`，旧job和账本保留；可配合`--scene-ids cfg2/task_XXXX/scene_Y,...`强制复核指定scene。每次显式repair会重新尝试null项，仍无可靠共享操作则继续排除，不强行补描述。

成本实验从cfg2完整冻结清单抽60个task、每task一个scene，共240视频（人机各120段），seed=44；默认启用GLM、Qwen、DeepSeek三家，输出在`artifacts/semantic/v2/cost`，图表在`charts/cost.png`，汇总为`cost/summary.json`、`results.csv`和`report.html`。仅允许GLM补标可用`--stage cost --providers glm`；其它已完成变体仍从缓存纳入比较。三个入口互相独立，主入口不运行成本或教师生成。教师默认来源为S5，可用 `--source S1/S2c/S2w/S3/S4/S5` 改选（实际传单个组名），保存在 teachers/S4/ 等独立目录。重复命令可以续跑。

GLM现在通过`providers.glm.key_pool`统一读取本地`configs/semantic/keys.local.json`，文件及锁文件已加入Git忽略，凭证文件权限为600。当前环境key和新增key按顺序放在`active`；只在收到官方欠费码1113时永久移入`overdue`，同一请求自动换下一个key。限流1302、过载1305、超时、鉴权错误均不会移入欠费区。配置了key_pool后不再回退环境变量，欠费key不会重新启用。

后续新key直接追加到本地文件的`active`数组，例如`{"id":"key3","key":"新key"}`；保留`overdue`。所有key耗尽时停止标注，补新key后原命令续跑。视频标注、成本实验及scene教师共用该机制；切换key不改变标注缓存协议。实验配置、请求日志和产物不保存明文key，本地配置不放在artifacts中。

Qwen使用`qwen3.7-plus-2026-05-26`和配置中的北京业务空间专属地址，读取`configs/semantic/qwen.keys.local.json`；DeepSeek使用`deepseek-flash`，读取`configs/semantic/deepseek.keys.local.json`，已有环境变量`DEEPSEEK_API_KEY`已导入。两份文件及锁文件均被Git忽略、权限600，不需要再次export。各家密钥池独立；Qwen仅在明确的Arrearage／PrepaidBillOverdue／PostpaidBillOverdue错误时隔离，DeepSeek仅在HTTP402时隔离，限流和鉴权错误不会隔离。

Qwen和DeepSeek成本配置关闭思考，GLM沿用主实验的开启思考、low设置；请求配置记录于各job，不将结果解释为相同推理预算下的能力排名。DeepSeek价格按响应时间戳（缺失时用请求开始时间）、北京时间时段和2026节假日表估算，并记录实际采用的单价和缓存token；未维护年份保守按高峰价并标记calendar_known=false。费用仍是usage估算，以厂商账单为准。模型别名可能更新，当前官方`deepseek-flash`对应DeepSeek-V4.1-Flash-1017，实际响应model保存在请求账本。

独立环节、论文选组、增量标注、scene教师和完整布局见 [实现与命令](../../docs/semantic/PLAN.md#14-已实现的统一入口)。本目录其他 v1 JSON 属于历史流水线，不用于当前第三章入口。

新提示词video_p9与16帧产生新的job修订。首次prepare归档退役S0/S2；其余组在新标注投影时归档旧格式产物，旧20帧标注不能跳过新协议请求。费用账本保留。`--only`只选择保存和评估的组，模型仍输出完整字段。原`--suite chapter`参数已移除。

增量标注cfg2训练集：`python runs/semantic/run.py --stage annotate --cfg cfg2 --split train`。不会改变默认评估范围；之后可按视频run实际train生成教师。默认已包含清单内训练视频；该入口也可单独补齐训练划分失败项。

标注首轮结束后，自动补重试一轮失败项，成功项不重发；主实验、成本变体和scene教师均启用。补重试仍沿用单条max_attempts上限，结束后才进入后续步骤，不循环补重试。

补重试后少量标注失败默认不阻断主流水线或成本评估。成功项继续编码；各组／成本变体使用共同可用集合，原始数据保存缺失ID与覆盖率，图表标为PARTIAL。直接重跑同一命令只补失败／缺失项，随后更新评估与图表，无需 --allow-partial。全部失败或没有可评估的人机集合仍会报错。

独立比较cfg2的scene数量：

```bash
# 完整cfg2清单（train/val/test合并）：全部、每task最多3/2/1个scene
python runs/semantic/eval_scenes.py --cfg cfg2 --split all
# 仅测试集上的同类比较
python runs/semantic/eval_scenes.py --cfg cfg2 --split test
# 独立选择scene上限和表示
python runs/semantic/eval_scenes.py --cfg cfg2 --split all --caps 3,2,1 --only S1
```

使用冻结scope中原有相机配对，同一seed下1⊆2⊆3⊆all；scene不足上限的task保留其全部scene，不剔除task。默认S1/S2c/S2w/S3/S4/S5均评估，原始数据含双向task/scene全部指标，图展示人→机五项主指标。结果自动保存到新的`artifacts/semantic/v2/scene_eval/cfg2-all-时间/`，包含`summary.json`、`summary.csv`、`README.md`和`all/s3/s2/s1`各自的图表、排名；向量缓存也在该次独立目录内。缺向量时从现有标注本地补编码，缺标注则明确报错，不调用VLM。现有主实验、成本实验和原始向量均不写入。可用`--output`指定新目录，目录已存在则拒绝覆盖。

成本cfg/split/tasks/seed变化时，prepare自动更新scope.cost，并将旧cost目录及成本图源、图片移入history/cost-*；原始jobs请求与成功标注保留，可续用。主评估范围改为全量后，使用eval、analyze、plot、report依次更新指标、分析、主图和报告；单纯redraw依然从已有图源恢复。

按厂商分别控制并发：GLM沿用48；Qwen固定快照`qwen3.7-plus-2026-05-26`使用64个工作线程、每秒2.5次平滑发起；DeepSeek Flash上限设为2500，实际线程数不超过待标注数（本成本集最多240）。本地抽帧解码单独限制为4，避免高API并发同时解码大量视频占满内存。`--concurrency`可显式覆盖所有厂商的工作线程数。并发与节流参数不改变标注缓存签名，账本记录实际配置；限速等待单独记为rate_wait_seconds，不计入API延迟。

仅续跑Qwen和DeepSeek、复用已完成GLM：

```bash
python runs/semantic/run.py --stage cost --providers qwen,deepseek --retry-unknown
```

`--providers`明确限定允许发起API请求的厂商，其它厂商仅在缓存完整时并入比较，不会补标；缓存不全则跳过并在summary中记录。Qwen成功项继续复用，不要加--force或--fresh-benchmark。--retry-unknown仅允许重试中断后没有收到响应的请求，这些请求原来的计费结果未知。

Qwen当前账号实查限额：60请求/6秒、1,000,000 token/60秒，无额外workspace限额；官方未公布该模型固定并发数，64是本地工作线程配置。2.5请求/秒按已有约5400 token/请求留出余量，遇账号共享额度或单请求token波动仍可能429并按原策略重试。[官方固定快照限额](https://help.aliyun.com/zh/model-studio/qwen3-7-plus)、[配额查询接口](https://help.aliyun.com/zh/model-studio/list-quotas)。DeepSeek官方Flash账号并发上限2500，[官方限速文档](https://api-docs.deepseek.com/zh-cn/quick_start/rate_limit/)。不同并发和节流下的吞吐作为部署效率记录，不完全归因于模型本身。

成本结果总览：直接打开`artifacts/semantic/v2/cost/index.html`，可切换表示方式、检索方向与热力图指标；`overview.png`汇总S1五项检索指标、六种表示MRR、已知费用和API延迟，`comparison.csv`提供完整比较数据（含MPR-all，但图片不画该指标）。费用来自当前账本，未知费用请求单列。独立重绘用`python runs/semantic/plot_cost.py`，只读取`cost/summary.json`，不调用API；成本流水线与常规plot阶段也会自动更新，PNG依赖记录在根目录`plots.json`。
