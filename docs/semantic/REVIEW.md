# 语义检索模块审计与改进报告

日期：2026-10-11。范围：当前代码、cfg2 实际标注与检索产物，以及截至本日期可查的相关工作。

后续确定的章节方案以 [CHAPTER.md](CHAPTER.md) 为准。本报告保留现状审计与测量；早期实验组合、验证集选优、迁移及复杂匹配建议不再作为最终计划。

本轮只新增报告。未修改生产代码、提示词、配置、标注、embedding 或数据集，未调用付费 VLM API。统计、CPU 文本重编码和少量视频抽帧仅用于审计，临时文件保存在 `/tmp`。

## 1. 主要结论

1. 语义模块已形成视频标注、归一化、文本编码、检索评估的基本链路，但实际覆盖与当前视频实验清单尚未统一。cfg2 的 109 个任务全部有描述，434 个 scene 只覆盖 218 个，且每个 scene 只标注一组相机的人机视频。
2. 当前结构主要保存粗能力集合、复杂度、环境和场景类别，缺少检索最需要的动作顺序、交互对象角色、来源/目标和状态变化。436 条标注只有 27 种标签文本，标签直接加入文本向量后，现有人→机任务级 R@1 从 31.19% 降到 24.31%。
3. 描述存在泛化、主体措辞偏差、对象误识别和动作遗漏。不能把低检索精度全部归因于文本编码器。部分跨 task 的文字完全相同，也可能涉及任务标签与语义等价关系不一致，需要人工核验。
4. BGE-M3 当前用了均值池化，与官方稠密表示的 CLS 提取方式不同。但本轮受控重编码显示，改为 CLS 后任务级 R@1 仍为 31.19%，scene 级仅从 15.14% 到 16.51%。这是需要规范的实现问题，不是已经找到的主要性能瓶颈。
5. 推荐章节主线为“面向人机跨域检索的任务语义表示与匹配”：以可核验的动作—对象—关系—状态变化为核心，进行分字段匹配，并检验困难候选下的区分能力。先用冻结模型，不以训练大型 VLM 为前提。
6. 用户直接输入任务文本与从人类视频提取任务信息，应区分输入来源和评估协议，共用表示、检索器和机器人候选库。前者表达请求，后者描述观测，两者不应强行要求携带相同的信息。
7. 前述跨相机讨论中的“同 scene 人机共享描述”仅适合训练集内的辅助教师设计，不能用于独立语义检索的查询与候选标注；后者必须分别生成，否则会泄露配对关系。

以下区分三种证据：代码事实、现有数据上的测量、尚待验证的改进建议。临时重编码属于探索性审计，不是完整固定测试集上的最终论文结果。

## 2. 当前链路与代码审计

### 2.1 实际链路

```text
数据目录
  → 每 task 选 scene、每 scene 选一组相机
  → 人类与机器人视频分别抽帧
  → VLM 单阶段联合输出描述和标签，或两次调用分别输出
  → 解析、taxonomy 归一化、非空校验
  → 描述向量与标签文本向量
  → text / label / combined 三种语义检索
```

cfg2 现有原始响应记录显示使用 `glm-4.6v`，436 次记录均为 `joint`。现有标注元数据为 `single_stage`。当前默认配置指向 cfg3，不能直接把默认配置当成 cfg2 旧实验的完整运行快照。

| 环节 | 当前实现 | 判断及建议 |
|---|---|---|
| scene 抽样 | 每 task 默认取排序后的前 2 个 scene | cfg2 因此覆盖 scene_1、scene_2；不能代表随机或固定测试覆盖 |
| 相机抽样 | 优先匹配配置相机；未命中则按相机标识排序取第一组 | 实际 436 条记录均为 `cam_036422060215`；未提供多相机语义稳健性证据 |
| 视频输入 | 抽取多帧，通过多张 `image_url` 输入 VLM | 当前配置 16 帧；未显式附带逐帧时间戳，采样与证据定位需补齐 |
| 输出格式 | 在提示中给出 JSON 形状，随后做解析 | 不是严格类型与语义约束的完整 JSON Schema；JSON 合法不等于事实正确 |
| 标签归一化 | 未落入词表的能力标签被过滤 | 对 cut、open、close 等缺失动作，容易丢失关键区分信息 |
| 非空校验 | 描述、能力、环境不能为空 | 有利于产物完整，但不能检查动作、对象、目标或状态的真实性 |
| 文本编码 | BGE-M3，所有 token 均值池化后归一化 | 应按模型官方方式提取；更换模型时不能继续共用同一种通用池化 |
| 文本与标签融合 | `normalize(text_vector + label_vector)` | 产生文本/标签之间的交叉项，不等同于两路独立相似度的可控加权 |
| 续跑 | 已完成记录直接跳过 | 改提示词或编码配置后，不会自动重新生成既有记录 |
| 缓存 | 包含样本、路径、模型名、提示/词表版本等 | 不包含抽帧数量和提示正文；修改内容却沿用版本会复用陈旧结果 |
| 可追溯性 | 标注存提示版本、词表版本、向量等 | 缺少编码模型、池化方式、抽帧策略的完整快照，难以追溯混合产物 |
| 真实查询入口 | 消费预生成语义向量 JSON | 当前没有直接 `--text` 文本编码入口；`--video-path` 只现场计算视频向量，不自动生成语义 |

代码位置：

- [manifest 构建](../../runs/semantic/build_semantic_manifest.py)、[相机选择](../../src/bise/modalities/semantic/sampler.py)。
- [提示词构建](../../src/bise/modalities/semantic/prompts.py)、[请求组装](../../src/bise/modalities/semantic/vlm_client.py)。
- [schema](../../src/bise/modalities/semantic/schemas.py)、[解析](../../src/bise/modalities/semantic/parser.py)、[归一化](../../src/bise/modalities/semantic/normalizer.py)。
- [流水线及缓存复用](../../src/bise/modalities/semantic/pipeline.py)、[文本编码](../../src/bise/modalities/semantic/embedder.py)。
- [语义检索评估](../../src/bise/modalities/semantic/evaluator.py)、[独立评估入口](../../runs/semantic/evaluate_semantic_retrieval.py)、[统一查询入口](../../runs/retrieval/query_retrieval_system.py)。

### 2.2 当前提示词的问题

[描述提示](../../configs/semantic/prompt_description_v1.json)与[联合提示](../../configs/semantic/prompt_joint_with_taxonomy_v1.json)要求一个简洁句子，只聚焦主要动作、物体和目标，并禁止所有颜色信息。

这种设计能抑制冗长背景描述，但也有不足：

- “一句简短描述”容易把打开、关闭、堆叠、切割等操作压缩成 grasp/place。
- 没有要求比较初始与结束状态，也没有要求保留多步顺序。
- 没有明确规定“看不见就未知”，容易用泛化常识补齐目标或结果。
- 人类描述常为祈使句，机器人描述常以 `Robot grasps...` 开头，形成与操作无关的主体差异。
- 全面禁止颜色不够灵活：跨人机时应弱化非功能性外观，但用户明确要求特定对象时，颜色可能是必要约束。应把操作核心与可选对象属性分开。
- 模型输入带样本标识，而检索表示不需要 task/scene 编号；建议从内容提示中移除身份信息，仅保留在元数据中。

### 2.3 当前 schema 为什么区分度有限

现有主要字段为：`task_description`、`capability_tags`、`task_complexity`、`environment_tags`、`scene_category`。

能力词表为 grasp、transport、place、insert、rotate、pour、press。描述中的 open、close、cut、stack、wipe 等没有相应的主要动作字段。词频规则检查发现，40 条包含 cut 词形的描述覆盖 10 个 task，其中 34 条能力集合为 grasp/transport/place；这只是文本规则统计，但能说明词表无法完整表达这些操作。

复杂度和环境类别更多用于管理、筛选和统计，不应被当作主要检索特征。人机复杂度不一致也不一定意味着标注错误：同任务在人类与机器人侧的执行难度可能不同。因此，应将其从任务等价关系中分离。

### 2.4 评估和工程细节

- 旧标注的 `scene_id` 为 scene_1、scene_2，当前评估通过 task 前缀消除冲突；跨 cfg 建库仍应保留 dataset/task/scene 身份，避免局部名称碰撞。
- 单个角色缺失时，`split_annotations_by_role` 会退回全体对全体；真实实验应拒绝这种隐式回退，防止把自检索当成跨人机检索。
- 缺少有效帧时，当前 `_load_frames` 可返回空列表；应在调用前拒绝或标记无视觉证据，不能让模型只根据提示生成“视频描述”。
- prompt_mode 的 two_stage 指描述与标签的两次调用，不能据此宣称做了基于视觉证据的第二阶段核验。
- 标签重复造成大量并列分数；当前排序没有专门的并列相关度协议，应记录确定性排序规则，或补充并列分数的敏感性分析。
- 提示、taxonomy、抽帧或编码变更应使用明确的新版本和独立输出目录。按用户既有约定，不建议再向数据清单新增哈希；版本与配置快照足以作为本轮建议。

## 3. cfg2 实际覆盖

数据来源：[标注](../../artifacts/semantic/rh20t/cfg2/annotations/normalized_annotations.jsonl)、[manifest](../../artifacts/semantic/rh20t/cfg2/manifests/semantic_manifest_v1.jsonl)、[原始响应](../../artifacts/semantic/rh20t/cfg2/annotations/raw_responses.jsonl)。完整逐任务覆盖见附录 A。

| 项目 | 实测 |
|---|---:|
| 子集 task | 109 |
| 子集 scene | 434 |
| 有描述 task | 109，100% |
| 有描述 scene | 218，50.23% |
| 缺描述 scene | 216 |
| 标注记录 | 436，人类 218＋机器人 218 |
| 标注状态 | 436 success，manifest 436 completed |
| 描述文件中的视频路径存在 | 436/436 |
| 每 task 已标注 scene | scene_1、scene_2 |
| 相机标识 | 只有 cam_036422060215 |

107 个 task 各有 4 个 scene；task_0032 和 task_0087 各有 3 个 scene。文件路径存在与解析成功不能证明标注事实准确。

### 3.1 与 s44 现有视频划分的交集

以下是 [s44 视频清单](../../artifacts/manifests/retrain/s44/cfg2/split_manifest_video.json)的实测快照；不是重新划分。该文件实际列出 428 个 scene，而子集目录有 434 个，六个未列入 scene 的原因不在本轮审计范围内。

| split | 清单 scene | 已有描述 scene | 缺描述 scene | 原相机描述与清单视频对精确匹配 |
|---|---:|---:|---:|---:|
| train | 258 | 135 | 123 | 127/516 |
| val | 50 | 18 | 32 | 14/100 |
| test | 120 | 61 | 59 | 57/240 |

“scene 有描述”不等于“该 split 选中的相机视频有独立描述”。scene 共享教师可以服务训练，但单视角语义检索需要规定能否使用额外相机信息。

当前语义评估直接使用 218 个人类描述查询和 218 个机器人描述候选，没有按 s44 的 120 个测试 scene 过滤。因此，旧语义结果不能直接与 s44 其他模态的测试结果横向比较。后续统一协议时，建议每 scene 一个语义候选实体，避免重复相机描述扩大正样本数量；若保留每相机候选，应明确去重/聚合规则。

## 4. 描述区分度与实际检索

### 4.1 描述和标签的分布

| 统计 | 实测 | 含义 |
|---|---:|---|
| 唯一完整描述 | 390/436 | 字面不同不等于语义可区分 |
| 平均描述长度 | 12.13 个空格分词 | 当前描述较短 |
| 包含泛化对象短语 | 73/436，16.74% | 按 small object/an object/the object/a small item/an item 规则统计，非人工错误率 |
| 包含颜色词 | 102/436，23.39% | 按常见颜色词规则统计；即使当前提示禁止，现有产物仍包含颜色 |
| 唯一标签文本 | 27/436 | 标签信息显著压缩 |
| 最常见标签文本 | 123 条，涉及 76 个 task | 标签不能区分大量候选 |
| 能力集合 grasp/place | 242/436，55.50% | 大量任务被同一粗操作集合覆盖 |
| 环境标签无障碍物 | 436/436 | 在本子集没有区分信息 |
| 场景类别工业 | 411/436，94.27% | 主要是共同背景类别 |
| 跨 task 完全相同描述 | 16 组，43 条记录 | 需要辨别粗描述碰撞与真正语义等价 |

人机对应的 218 对记录中：能力集合一致 104 对；复杂度一致 88 对；完整标签文本一致 27 对。不能将这些一致率当成人工标注准确率，但它们说明粗标签并未自然形成稳定的人机任务表示。

218 条机器人描述有 186 条包含 robot，而人类描述为 0 条。建议统一使用不带主体的操作短语，保留不同形态真正影响功能的差异。

### 4.2 现有结果复算

直接调用项目现有 `evaluate_annotation_retrieval`，使用标注元数据中的向量，复算结果与已有 summary 一致。候选库为 218 个机器人描述，查询为 218 个人类描述。

| 人→机表示 | task R@1 | task R@5 | task MRR | scene R@1 | scene MRR |
|---|---:|---:|---:|---:|---:|
| 描述文本 | 31.19% | 55.50% | 0.4176 | 15.14% | 0.2860 |
| 粗标签文本 | 4.59% | 10.55% | 0.0996 | 2.29% | 0.0616 |
| 文本向量＋标签向量 | 24.31% | 50.92% | 0.3639 | 12.39% | 0.2554 |

机器人→人类文本检索的 task R@1 为 31.19%、MRR 为 0.4366，scene R@1 为 15.60%、MRR 为 0.3110。反向结果主要用于诊断；本课题的人类查询→机器人候选应作为主要方向。

描述向量的同 scene 人机相似度均值为 0.8342，不同 task 均值仍为 0.7431。150/218 个查询的最高不同-task 候选分数严格超过最高同-task 候选。绝对相似度普遍偏高不代表检索匹配准确。

### 4.3 BGE-M3 池化受控对照

本轮用本地已缓存同一 BGE-M3 权重，在 CPU 上对 390 种描述和 27 种标签文本重编码；相同 tokenizer、max_length=256、FP32、L2 归一化，只改变 mean/CLS。既有向量与重算 mean 的平均余弦为 1.0000，最大逐元素差约 1.01e-6，确认旧产物来自该均值池化路径。

官方说明稠密向量使用首 token 的 CLS 表示：[作者说明](https://huggingface.co/BAAI/bge-m3/discussions/80)。

| 人→机文本单独检索 | task R@1 | task R@5 | task MRR | scene R@1 | scene MRR |
|---|---:|---:|---:|---:|---:|
| mean，旧实现 | 31.19% | 55.50% | 0.4176 | 15.14% | 0.2860 |
| CLS，官方稠密提取方式 | 31.19% | 53.21% | 0.4189 | 16.51% | 0.2924 |

结论：应建立符合模型规范的基线，但本数据上并未出现整体大幅提升；R@5 还下降。不能将池化修正包装为主要方法创新，也不能承诺只改池化就解决区分度。

### 4.4 典型冲突与抽样视觉核查

| task/scene | 人类描述要点 | 机器人描述要点 | 诊断 |
|---|---|---|---|
| task_0020/scene_1 | 按台灯按钮使其亮起 | 抓取并移动图案表面的小物体 | 动作和对象语义不一致；抽帧可见人机两侧台灯，机器人描述明显遗漏任务关键对象和交互 |
| task_0074/scene_1 | 打开微波炉门 | 抓住并打开笔记本电脑 | 抽帧可见金属箱体、通风孔和门结构，与 laptop 描述不符；具体设备类别仍可人工核验 |
| task_0095/scene_2 | 按键盘按键 | 抓取小物体并移向容器 | 人类帧可见键盘，机器人帧中未清楚呈现相同对象；需要看完整视频、其他视角和来源映射，不能仅凭文字断言全是 VLM 错误 |
| task_0031/scene_2 | 堆叠木块 | 将方块放到有其他方块的表面 | 关键关系 stack/on 丢失，容易混淆普通放置 |
| task_0092/scene_1 | 从夹具取出笔并放到桌面 | 抓工具，在织物上进行动作 | 来源关系和操作结果未保留；尚未逐帧核验 |
| task_0006 与 task_0008 | 方块放到纸板表面 | 存在跨 task 完全同文描述 | 可能是描述过粗，也可能标签区分了语言未表达的因素；需确认相关性定义 |

视觉抽查仅覆盖上述前三组、六段视频，每段取五个时间位置；它是定性诊断，不是随机抽样准确率，也不能据此估计全部 436 条的错误比例。

## 5. 相关工作与前沿路线

以下只使用论文、作者项目和官方模型资料。借鉴表示与评估设计，不将通用榜单成绩直接视为本数据集性能保证。

| 工作与时间 | 方法要点 | 本课题可借鉴部分及边界 |
|---|---|---|
| [BGE-M3，2024](https://arxiv.org/abs/2402.03216) | 同时支持 dense、sparse、multi-vector 检索 | 先规范编码；可用词法和稠密互补，而不必立即训练新模型 |
| [VideoComp，CVPR 2025](https://arxiv.org/abs/2504.03970) | 通过事件重排、动作替换、部分描述等困难负样本检验组合与时序理解 | 建立开/关、取出/放入、顺序互换等最小差异评估；不能只看一般 R@K |
| [CaReBench，2024/2025 预印本](https://arxiv.org/abs/2501.00513) | 将详细描述的空间与时间信息分开，联合评估 caption 与 retrieval | 对象/关系与动作/状态分字段，分别测精度和检索收益；不照搬长篇背景描述 |
| [Robo2VLM，2025](https://arxiv.org/abs/2505.15517) | 利用末端位姿、夹爪开合和力等信号分阶段，生成具身空间与交互问题 | 状态变化和阶段应有证据；可将机器人夹爪信号用于离线抽帧辅助，但须说明辅助信息，不混称纯视频方法 |
| [EGOILLUSION，EMNLP 2025](https://aclanthology.org/2025.emnlp-main.1446/) | 对视频动作、对象等幻觉进行专门评估 | 结构合法与流畅不是精准度，需人工事实核查和未知机制 |
| [Qwen3 Embedding/Reranker，2025](https://arxiv.org/abs/2506.05176) | 指令感知的文本向量与重排，提供 0.6B、4B、8B 等规模 | 0.6B 可作为低成本编码/重排对照；按官方 last-token 与查询指令规范接入，不能直接套现有 mean pooling |
| [TARA，作者项目标注 ECCV 2026](https://www.robots.ox.ac.uk/~vgg/research/tara/) | 用文本三元组和针对时间反义、否定、组合查询的困难负样本适配多模态表示 | 可先借鉴困难文本构造与小型编码器适配，不必复现其大型多模态训练 |
| [Qwen3-VL-Embedding/Reranker，2026 技术报告](https://arxiv.org/abs/2601.04720) | 文本、图像、视频统一编码，再用交叉编码器重排 | 适合强外部 text→video 对照，但属于直接多模态路线，不是“描述→文本匹配”主方法 |
| [Qwen-RobotWorld，2026 技术报告](https://arxiv.org/html/2606.17030v1) | 任务目标、动作细节、物理结果、长短描述分层标注，结合质量过滤 | 借鉴目标/动作/结果分离及少量定向重标；不用复现世界模型和巨量数据流程 |
| [STaR，RA-L 2026](https://trailab.github.io/STaR-website/) | 带时间的 caption、空间原语和视觉记忆联合检索 | 借鉴结构化、去冗余记忆和证据回溯；其导航记忆任务与本课题执行检索不同 |
| [MARS，2026，论文标注 EMNLP 主会接收](https://arxiv.org/abs/2609.02565) | 多层、多槽位表示与困难负样本，缓解单向量细粒度信息压缩 | 支持“分字段/多向量”方向；本规模先显式分字段，不复制整套网络 |
| [RoboTok，2026-09 预印本](https://arxiv.org/abs/2609.03199) | 基于三维人手运动表示检索互联网操作演示 | 与课题整体的多模态检索定位相关；属于轨迹方向，不作为本章语义方法的直接替代 |

相关趋势是保留操作组合、时序、对象状态与困难候选，并用证据控制描述质量。单纯加长 caption、换更大模型或增加泛化标签，不能自动改善机器人任务检索。

## 6. 推荐的语义表示

### 6.1 以操作事件为中心

保留简洁 summary，新增实体及有序事件，记录来源/目标和可观察状态变化。复杂度、场景类别等可保留用于管理和统计，但不默认参与主检索分数。

下面是提议的逻辑格式，不是已实现的新 schema：

```json
{
  "summary": "Press the lamp switch to turn on the lamp.",
  "goal": "lamp on",
  "entities": [
    {"id": "o1", "name": "lamp", "role": "affected_object"},
    {"id": "o2", "name": "switch", "role": "interaction_part", "part_of": "o1"}
  ],
  "steps": [
    {
      "action": "press",
      "object": "o2",
      "source": null,
      "target": "o1",
      "state_before": "lamp off",
      "state_after": "lamp on",
      "evidence_frames": [3, 8, 12]
    }
  ],
  "uncertain_fields": []
}
```

示例中只有确实观察到灯光变化，才能填写 off→on；否则 state_before/state_after 应为 null，并说明不确定。用户只说“开灯”时可以保存请求目标，不能补写已观察到的状态和帧证据。

metadata 单独保存输入来源 `user_text` 或 `video_caption`、schema/prompt 版本、相机、编码模型/池化、采样帧索引等；task/scene ID 仅用于身份与评估，不进入语义特征。

最低限度的校验应包括字段类型、实体引用、步骤顺序、可空规则、视频证据帧合法性。动作使用规范化词汇，但允许 `other` 或保留原始动作，不把词表外动作静默改成 grasp/place。未知状态与负状态必须分开。

### 6.2 提示词改进

视频描述提示可围绕以下英文规则设计，具体措辞在开发集上比较：

```text
Describe the observable manipulation in the chronological frames.
Use an actor-neutral representation: do not distinguish a human hand
from a robot gripper unless that difference changes the task.
Identify the interacted object, its relevant part, source, destination,
ordered actions, and visible state changes. Compare the beginning and
the end. Do not label every contact as a grasp or every motion as a
transport. Preserve distinctions such as open/close, insert/remove,
stack/place, press/grasp, and cut/move.
For each event, cite supporting frame indices. Use null when an object,
relation, state, or result cannot be established. Do not infer successful
completion from an attempted action. Ignore unrelated background and
camera-relative directions. Keep object attributes separately when they
are necessary to identify the requested object. Return the specified JSON.
```

给模型带索引、按时间排序的帧；先改内容约束和证据要求，不默认增加多轮 API。可仅对缺对象、状态矛盾、低可见性或开发集发现的系统性错误做第二次视觉核验。

直接任务文本采用另一套轻量解析提示：忠实提取用户给出的动作、对象、目标、顺序和显式约束；不补充未提供的物体颜色、来源、初始状态或执行成功。信息已足够时可直接编码原文，不必强制调用 LLM 扩写。

### 6.3 抽帧与可见性

在固定成本下，先比较 uniform-16 与事件覆盖的 16 帧：保留开始、结束和交互前后，减少长时间静止画面的重复。也可用 uniform-32 作为单独成本对照，但不要同时更换 prompt、模型与采样。

机器人夹爪数据可辅助选择接触/释放附近帧，人类侧可用已有关键点或图像变化；如果用了这些信息，必须在实验中披露，并保留纯视频采样对照。状态和细节无法从输入观察到时，应保持未知，不靠更长提示补全。

## 7. 推荐的检索方法

第一阶段采用冻结文本编码器，保留可复现基线：

1. BM25/TF-IDF 词法匹配，检查是否已有简单强基线。
2. 规范 BGE-M3 稠密表示；选一个 Qwen3-Embedding-0.6B 对照，避免大规模模型遍历。
3. 同一 VLM、相同输入预算下比较旧简述与新结构化操作表示。

主方法建议先采用分字段匹配：summary、对象/角色、动作顺序、目标/状态分别评分，缺失字段不计入分母。示意为：

```text
S(q, r) = Σ[w_f × c_f × s_f(q, r)] / Σ[w_f × c_f] - contradiction_penalty
```

其中 `c_f` 表示两侧该字段的可用性或经过核验的可靠程度，不直接把 VLM 自报置信度当作校准概率。动作序列与实体角色不能退化成无序词袋；未知字段为中性，明确且可靠的相反状态才施加冲突惩罚。

例如 open box 与 close box 的名词相同，但状态方向相反；stack block on block 与 place block on table 的放置目标不同。字段级匹配有助于定位是对象识别错、关系丢失还是动作方向混淆。

在候选库较小时，可先做全库分字段匹配。若加入 reranker，采用固定 top-K 候选并报告候选召回率、重排后指标及延迟；重排无法找回第一阶段未召回的目标。优先考虑小型文本交叉编码器，不默认让大 VLM 对全库逐个判断。

仅当冻结方案仍明确卡在动作反义和顺序辨别时，才考虑在训练划分上对小文本编码器做对比适配。困难负样本包括 open/close、insert/remove、动作正确但目标错误、对象正确但动作错误、步骤交换。必须核验负例确实不相关，避免把同义操作或同 task 的有效变体当负样本。生成困难三元组的思想可参考 [VideoComp](https://arxiv.org/abs/2504.03970) 与 [TARA](https://www.robots.ox.ac.uk/~vgg/research/tara/)，不意味着其训练收益会自动迁移。

## 8. 两种文本来源是否区分

推荐区分。投入主要是来源元数据和一组查询评估，不需要两套模型，也不需要拆成两章。

| 来源 | 输入含义 | 处理 | 主要评价 |
|---|---|---|---|
| 用户直接文本 | 想让机器人完成什么，可包含要求或否定 | 原文编码或保守结构化解析 | 相关任务与显式约束是否匹配 |
| 人类视频提取文本 | 视频可观察到做了什么，可能看不全 | VLM 视觉提取、证据与未知机制 | 提取事实质量＋端到端机器人检索 |

共同路径：

```text
用户任务文本 ─→ 保守解析或直接编码 ─┐
                                  ├→ 统一任务语义表示 → 同一机器人语义库
人类演示视频 ─→ 独立视觉提取 ─────┘
```

低成本评估可选 30–50 个任务，人工独立撰写 60–100 条任务请求，包含完整描述、简短请求、同义改写，以及适量目标/顺序/否定约束。具体数量是建议，不是已完成数据。

撰写者可以看人类视频或任务说明，但不看机器人候选描述和检索结果；不能把候选 caption 复制或同义改写后当成人类独立输入。纯 LLM 生成的改写可作补充稳健性集，应明确标为合成查询。

任务级文本往往无法唯一指定某个 scene。此时应把全部语义有效候选视为相关，不应因没命中一个特定 scene 就判错。若要评价实例检索，查询必须携带足以区分实例的条件。

在系统章节中说明：原视频与其派生语义共享输入来源，并非三个完全独立传感器。直接文本是真正独立的输入路径；从视频提取的文本是视频到语义的派生路径。三种检索表示可以并用，但不能据此宣称输入信息彼此独立。

当前统一入口尚缺直接文本处理，需要后续新增明确的文本查询入口；这属于产品输入完整性，不需要训练新模型。

## 9. 独立语义章节的评估设计

### 9.1 先统一有效候选与固定划分

使用同 batch 的固定 scene 清单；补齐对应查询和机器人候选语义。若仅评估覆盖交集，报告缺失数量和固定交集，不允许不同方法各自丢掉不同失败样本。

人类查询描述从给定人类视频独立提取，机器人候选描述从对应机器人输入独立提取。不能把人机描述先联合成同一文本再评估检索，也不能用已知 task ID 在测试时选取预先写好的答案描述。

可在同一机器人 scene 内聚合允许使用的多个相机以改善离线建库，但所有方法应使用相同可用输入。单相机查询不能暗中使用另一相机或机器人视频补齐信息。相机聚合与单视角分别报告。

### 9.2 双层相关性与困难集

- 保留 task/scene ID 指标，服务与其他模态的统一比较。
- 对可能跨 task 同义或同 task 多义的样本，另外建立小规模人工语义相关性判断：精确满足请求、功能相关但细节不符、不相关。
- 同对象不同动作、同动作不同目标、相反状态、顺序差异构成困难候选集；其相关性应经过人工确认。
- 人工相关性指标与 ID 指标分别报告，不能为了提高成绩在看过测试结果后修改唯一主协议。

### 9.3 提取质量与检索质量分开

建议在固定的 60–100 对人机样本上盲评并抽取部分双人标注，检查：

- 主动作、对象类别、来源/目标、状态变化、步骤顺序的正确率或字段 F1。
- 无依据细节比例、可观察信息遗漏率、未知输出比例与有效信息覆盖。
- 条件允许时按可见性分组，并报告人工标注分歧。

大量输出 unknown 可以降低幻觉，但会损失信息；应同时报告正确率和覆盖率。不能只把人机描述一致率或 VLM 自评当作事实准确率。

检索主指标为人→机 task R@1、R@5、R@10、MRR；scene 作为细粒度辅助。若有人工分级相关性，报告对应 NDCG。沿用项目现有 mpr-all 约定作为原始数据指标，不因本报告改动绘图规则。补充任务宏平均与按 task 聚类的置信区间，避免多个相机/scene 当作完全独立观测。

### 9.4 最小实验安排

以下是独立语义检索章节的实验，不改变 [跨相机计划](../video/CROSS_VIEW.md)已经确定的两个视频训练实验；不另外新增跨相机训练组。

| 实验 | 固定内容 | 唯一主要变化 | 目的 |
|---|---|---|---|
| S0 词法基线 | 相同描述、候选、划分 | BM25/TF-IDF | 是否超过简单词匹配 |
| S1 规范稠密基线 | 旧简述、规范 BGE-M3 | 冻结 dense 编码 | 建立实现正确的基线 |
| S2 结构化描述 | 同 VLM、帧预算、编码器 | 旧简述→事件与状态表示，先统一序列化编码 | 内容细化本身的收益 |
| S3 分字段匹配 | 复用 S2 相同提取结果 | 整体编码→分字段与缺失/冲突处理 | 匹配设计的收益 |
| S4 小型重排，可选 | 固定 S3 候选 top-K | 增加文本 reranker | 精度与延迟的收益 |

编码器对照只选 S1/S3 的一个固定设置换 Qwen3-Embedding-0.6B，不把更换模型、提示、采样和匹配同时算作一个方法提升。当前粗标签融合保留为失败分析，不必继续把它作为主方案。

同一批方法分别评价用户文本和视频派生文本，不需要为两种来源各训练一套。另用小型人工校正描述做诊断上限，观察抽取误差与匹配误差谁更大；人工校正描述不能混入自动方法的正式结果。

## 10. 投入产出与实施顺序

| 顺序 | 工作 | 成本 | 价值 |
|---|---|---|---|
| P0 | 统一清单、候选实体和输入来源；补齐覆盖 | 中，主要为必要标注 | 保证模态比较和语义评估可解释 |
| P0 | 规范池化、显式版本、空帧拒绝、重编码与重标分离 | 低 | 排除实现与陈旧产物干扰，不计作主要创新 |
| P1 | 在开发集比较事件/状态 schema 和证据提示 | 中 | 当前最有针对性的内容改进 |
| P1 | 固定小型人工事实评估与困难候选集 | 中 | 支撑独立章节，定位性能瓶颈 |
| P1 | 直接文本查询和轻量来源对比 | 低到中 | 完成三种用户输入的课题定义，叙事清晰 |
| P2 | 分字段匹配＋一个小型重排对照 | 低到中 | 利用结构化信息，解释区分度来源 |
| P3 | 小文本编码器的困难负样本适配 | 中 | 仅在明确发现匹配瓶颈后考虑 |
| 暂缓 | 大型 VLM 全量微调、复杂图网络、完整世界模型 | 高 | 当前数据规模下不优先 |

粗略规模：cfg2 全量 scene、单相机、人机各一条，共 868 条描述；已有 436 条，因此仅补缺覆盖为 432 条。若替换 schema，应在独立版本中重标全部相关条目，不能把旧、新字段混合比较。多相机独立标注的量按实际可用相机另算，不能默认所有 scene 都有相同相机集合。

成本优化：先开发集验证提示，再批量离线建库；仅对失败样本做定向复核。语义重编码不需要重新调用 VLM。不要先全量标注多个大模型再比较，其成本高且难定位收益来源。

## 11. 推荐论文组织与边界

建议章节题目：**面向人机跨域检索的结构化任务语义表示与匹配**。

```text
任务请求与视觉观测的统一语义接口
  → 现有粗描述/粗标签的信息损失与跨域误识别
  → 可验证动作、对象关系与状态变化的提取
  → 缺失感知的分字段语义匹配
  → 两种查询来源、困难候选、cfg2/cfg3 泛化评估
  → 遮挡、语义歧义和状态不可见的失败边界
```

可以主张的贡献，需要实验支持：检索导向的结构化表示、未知与证据约束、动作/状态冲突感知的匹配、针对人机任务检索的评估协议。通用 VLM/API 调用、换模型、修池化和普通 JSON 解析本身不是足够的方法贡献。

与视频章节的区别：本章研究“提取什么、是否正确、怎样做语义匹配”；视频章节复用合格训练描述作教师，研究“语义结构是否迁移到纯视频表示”。两章共享成果，但分别评估，不重复计入同一贡献。

已完成验证：11 个相关语义测试文件，32 项测试通过；重算既有检索结果；本地 CPU mean/CLS 对照；三个 scene 的定性视频抽查。尚未完成：完整人工事实标注、全量多相机核验、新 schema 提取与改进方法训练/评估。因此本报告不承诺新方案的提升幅度。

## 附录 A：全部 task 的覆盖与代表性描述

下表来自本轮审计统计。描述为原始标注的 scene_1 人类与机器人文本，不是人工确认的任务定义；scene_2 的完整记录见原始 JSONL。不能把本表中的对象识别直接当作真值。

| task | 目录 scene 数 | 已标注 | 缺失 | scene_1 人类描述 | scene_1 机器人描述 |
|---|---:|---|---|---|---|
| task_0001 | 4 | scene_1, scene_2 | scene_3, scene_4 | Grasp the small object on the table and place it on the target area of the table | Robot grasps a small object on a black surface and transports it. |
| task_0002 | 4 | scene_1, scene_2 | scene_3, scene_4 | Grasp a tissue from a tissue box and place it on a table | Grasp a tissue from a tissue box and pull it out |
| task_0003 | 4 | scene_1, scene_2 | scene_3, scene_4 | Press the red button on the table | Robot grasps and presses a button on a table with multiple buttons |
| task_0004 | 4 | scene_1, scene_2 | scene_3, scene_4 | Grasp a small cube and place it on a patterned surface | Robot grasps a small cube and places it on a patterned surface |
| task_0005 | 4 | scene_1, scene_2 | scene_3, scene_4 | Grasp a small cube and place it on a flat surface with a space-themed design | Grasp a cube on a table |
| task_0006 | 4 | scene_1, scene_2 | scene_3, scene_4 | Grasp a small cube and place it on a cardboard surface | Robot grasps a small cube and places it on a flat surface |
| task_0007 | 4 | scene_1, scene_2 | scene_3, scene_4 | Grasp a cube and place it on a surface | Robot grasps a cube and places it on a patterned surface |
| task_0008 | 4 | scene_1, scene_2 | scene_3, scene_4 | Place a small cube on a flat surface | Robot grasps a small cube and places it on a cardboard surface |
| task_0009 | 4 | scene_1, scene_2 | scene_3, scene_4 | Grasp a small object from a red base and place it on the table surface | Robot grasps a small object on a red base and places it on the table |
| task_0010 | 4 | scene_1, scene_2 | scene_3, scene_4 | Pick up the telephone handset and place it on the telephone base | Robot grasps a telephone on a table |
| task_0011 | 4 | scene_1, scene_2 | scene_3, scene_4 | Grasp the water pot and pour water into the plant on the table | Robot grasps a watering can and moves it towards a potted plant on a table |
| task_0012 | 4 | scene_1, scene_2 | scene_3, scene_4 | Place a small ball into a small container on the table | Robot grasps a small ball and places it into a container on the table |
| task_0013 | 4 | scene_1, scene_2 | scene_3, scene_4 | Grasp a cube and place it on a square platform | Grasp a small cube and place it on a square base on the table |
| task_0014 | 4 | scene_1, scene_2 | scene_3, scene_4 | Pick up a small box from a scale and place it on the table | Robot grasps a small rectangular object and places it on a flat surface on the table |
| task_0015 | 4 | scene_1, scene_2 | scene_3, scene_4 | Use a drumstick to strike a drum placed on the table | Robot grasps a drumstick and strikes a drum on a wooden table |
| task_0016 | 4 | scene_1, scene_2 | scene_3, scene_4 | Use a cue stick to hit a ball on a pool table | Robot grasps a cue stick from a pool table and places it on the table |
| task_0017 | 4 | scene_1, scene_2 | scene_3, scene_4 | Grasp a pen from a pen holder and place it on a sheet of paper | Robot grasps a pen and places it into a cup on a table |
| task_0018 | 4 | scene_1, scene_2 | scene_3, scene_4 | Grasp a wooden block and place it on a table surface | Grasp a wooden block and place it on a table |
| task_0019 | 4 | scene_1, scene_2 | scene_3, scene_4 | Place a game piece on the grid board | Robot grasps a small object and places it on a grid board |
| task_0020 | 4 | scene_1, scene_2 | scene_3, scene_4 | Press the button on the lamp to turn it on | Robot grasps and transports a small object on a patterned surface in an industrial setting. |
| task_0021 | 4 | scene_1, scene_2 | scene_3, scene_4 | Press the button on the desk lamp | Robot grasps and places a small object on a blue and white lamp base on a table |
| task_0022 | 4 | scene_1, scene_2 | scene_3, scene_4 | Use a stick to press a button on a box, causing a green sheet to lift. | Robot grasps a small object and places it on a table surface |
| task_0023 | 4 | scene_1, scene_2 | scene_3, scene_4 | Press the power strip on the table | Robot grasps a power strip on a wooden table |
| task_0024 | 4 | scene_1, scene_2 | scene_3, scene_4 | Press the power strip on the table | Robot grasps a power strip on a wooden table |
| task_0025 | 4 | scene_1, scene_2 | scene_3, scene_4 | Place a phone into a white storage box on the table | Robot grasps a small object and places it into a white box on a table |
| task_0026 | 4 | scene_1, scene_2 | scene_3, scene_4 | Grasp a white box on the table and place it down | Robot grasps a white container and places it into a cardboard box on a table |
| task_0027 | 4 | scene_1, scene_2 | scene_3, scene_4 | Grasp the white box on the table and place it on the table | Robot grasps a small rectangular object and places it into a white box on the table |
| task_0028 | 4 | scene_1, scene_2 | scene_3, scene_4 | Place a small object into a cardboard box on a table | Robot grasps and transports a cardboard box on a table. |
| task_0029 | 4 | scene_1, scene_2 | scene_3, scene_4 | Close the open box on the table | Robot grasps a box lid and places it on the box |
| task_0030 | 4 | scene_1, scene_2 | scene_3, scene_4 | Place a small box onto a cardboard structure on the table | Robot grasps a cube and places it on a cardboard surface |
| task_0031 | 4 | scene_1, scene_2 | scene_3, scene_4 | Stack wooden blocks into a tower on the table | Robot grasps and places cubes to build a tower on a surface |
| task_0032 | 3 | scene_1, scene_2 | scene_4 | Grasp the white package on the blue mat and place it on the table | Robot grasps a small box and places it on a table surface |
| task_0033 | 4 | scene_1, scene_2 | scene_3, scene_4 | Place a cube on a tissue box | Robot grasps and places small cubic objects on a flat surface |
| task_0034 | 4 | scene_1, scene_2 | scene_3, scene_4 | Grasp a transparent cup from a patterned surface and transport it to a target location | Robot grasps a glass and places it on a table |
| task_0035 | 4 | scene_1, scene_2 | scene_3, scene_4 | Grasp a glass and place it on a small cube on a table | Grasp a glass and transport it to a target location on the table |
| task_0036 | 4 | scene_1, scene_2 | scene_3, scene_4 | Grasp the white rectangular object on the cardboard and place it on the blue surface | Robot grasps a white rectangular object and places it on a cardboard surface with a small cube nearby. |
| task_0037 | 4 | scene_1, scene_2 | scene_3, scene_4 | Grasp a small bottle on a wooden table and place it down | Grasp a bottle on a table and transport it |
| task_0038 | 4 | scene_1, scene_2 | scene_3, scene_4 | Grasp a bottle and place it on the table | Grasp a small cube and place it on a table |
| task_0039 | 4 | scene_1, scene_2 | scene_3, scene_4 | Grasp a plastic bag on a wooden table and lift it. | Robot grasps a plastic bag on a wooden table and transports it. |
| task_0040 | 4 | scene_1, scene_2 | scene_3, scene_4 | Grasp a pen and place it on a stand | Robot grasps a pen from the table and transports it to a target location |
| task_0041 | 4 | scene_1, scene_2 | scene_3, scene_4 | Grasp a glass from a stand and place it on the table | Robot grasps a transparent object and places it on the table |
| task_0042 | 4 | scene_1, scene_2 | scene_3, scene_4 | Grasp a transparent glass and place it on the table | Robot grasps a glass and places it on a stand on the table |
| task_0043 | 4 | scene_1, scene_2 | scene_3, scene_4 | Rotate the steering wheel on the table | Robot grasps a small object and transports it to a target location on the table |
| task_0044 | 4 | scene_1, scene_2 | scene_3, scene_4 | Rotate the black steering wheel controller on the table | Robot grasps a steering wheel on a wooden table and performs a rotation action. |
| task_0045 | 4 | scene_1, scene_2 | scene_3, scene_4 | Pick up the plate from the table and place it on the dish rack | Robot grasps a plate and transports it to a new location on the table. |
| task_0046 | 4 | scene_1, scene_2 | scene_3, scene_4 | Grasp a basket from a small wooden stand and place it on the table | Robot grasps a basket from a stand and places it on a table |
| task_0047 | 4 | scene_1, scene_2 | scene_3, scene_4 | Grasp a small cube and place it on a table with a picture frame | Robot grasps a small object and transports it to a target location on the table |
| task_0048 | 4 | scene_1, scene_2 | scene_3, scene_4 | Place a small object into a basket on a table | Robot grasps a small object and places it into a basket on a table |
| task_0049 | 4 | scene_1, scene_2 | scene_3, scene_4 | Place a small toy on a large printed mat | Robot grasps a small object and places it on a fabric surface |
| task_0050 | 4 | scene_1, scene_2 | scene_3, scene_4 | Grasp a small object and place it into a glass container on the table | Robot grasps a red pen from a pen holder and places it into a glass on the table |
| task_0051 | 4 | scene_1, scene_2 | scene_3, scene_4 | Place a small object on the table | Robot grasps a small object and places it on a flat surface |
| task_0052 | 4 | scene_1, scene_2 | scene_3, scene_4 | Press the white switch on the table | Robot grasps and inserts a plug into a power strip on a wooden table |
| task_0053 | 4 | scene_1, scene_2 | scene_3, scene_4 | Press the white power strip on the table | Robot grasps a power strip on a wooden table and transports it to a new position. |
| task_0054 | 4 | scene_1, scene_2 | scene_3, scene_4 | Press a black rectangular object on a wooden table | Grasp a small ball on a wooden table |
| task_0055 | 4 | scene_1, scene_2 | scene_3, scene_4 | Press the button on the black rectangular device on the wooden table | Robot grasps a black rectangular object on a wooden table and places it on a black base |
| task_0056 | 4 | scene_1, scene_2 | scene_3, scene_4 | Place a small cube into a pink box on a table | Robot grasps a small cube from a pink box and places it on the table |
| task_0057 | 4 | scene_1, scene_2 | scene_3, scene_4 | Place a roll of paper on a holder on the table | Grasp a roll of paper and place it on a metal holder on a table |
| task_0058 | 4 | scene_1, scene_2 | scene_3, scene_4 | Use a shovel to transport a box from one position to another on the table | Robot grasps a tool and places it on a tissue box on a table |
| task_0059 | 4 | scene_1, scene_2 | scene_3, scene_4 | Grasp the roll of paper from the dispenser and place it on the table | Grasp a roll of paper and place it on the table |
| task_0060 | 4 | scene_1, scene_2 | scene_3, scene_4 | Place small blocks onto a baseplate | Robot grasps a small object and places it on a designated area of a base plate on the table |
| task_0061 | 4 | scene_1, scene_2 | scene_3, scene_4 | Place a block onto a green baseplate | Robot grasps a block and places it onto a green baseplate |
| task_0062 | 4 | scene_1, scene_2 | scene_3, scene_4 | Press the green button on the yellow and green cube | Robot grasps a green button on a yellow and white base and presses it |
| task_0063 | 4 | scene_1, scene_2 | scene_3, scene_4 | Grasp the orange cube and place it on the table | Robot grasps a dark cube and transports it to a white box with an orange button |
| task_0064 | 4 | scene_1, scene_2 | scene_3, scene_4 | Grasp a small rectangular object and place it on a table | Robot grasps a small square object on a wooden table and moves it slightly. |
| task_0065 | 4 | scene_1, scene_2 | scene_3, scene_4 | Grasp a small wooden object and place it into a black clamp on a table | Robot grasps a small wooden frame on a workbench and places it into a clamp. |
| task_0066 | 4 | scene_1, scene_2 | scene_3, scene_4 | Grasp the clamp and place it on the table | Robot grasps a small wooden object on a workbench and places it into a clamp. |
| task_0067 | 4 | scene_1, scene_2 | scene_3, scene_4 | Press the button on the small box next to the piggy bank | Robot grasps a small object and places it on a table near a piggy bank and a button |
| task_0068 | 4 | scene_1, scene_2 | scene_3, scene_4 | Place a small cube into a white container | Grasp a small box and place it into a white container on a table |
| task_0069 | 4 | scene_1, scene_2 | scene_3, scene_4 | Grasp a small cube and place it into a white box on a table | Robot grasps and transports a white box with a yellow component on a blue surface. |
| task_0070 | 4 | scene_1, scene_2 | scene_3, scene_4 | Move a chess piece on a chessboard | Robot grasps and moves a chess piece on a chessboard |
| task_0071 | 4 | scene_1, scene_2 | scene_3, scene_4 | Place a small block onto a grid base | Robot grasps a small block and places it on a baseplate |
| task_0072 | 4 | scene_1, scene_2 | scene_3, scene_4 | Place a small block onto a gray baseplate | Robot grasps a small block and places it on a gray baseplate |
| task_0073 | 4 | scene_1, scene_2 | scene_3, scene_4 | Grasp a metal box and place it on a table | Robot grasps a rectangular object and places it into a slot on a metallic box |
| task_0074 | 4 | scene_1, scene_2 | scene_3, scene_4 | Open the microwave door | Robot grasps and opens a laptop on a table |
| task_0075 | 4 | scene_1, scene_2 | scene_3, scene_4 | Pick up a sheet of paper from the table and place it back on the table | Robot grasps a white paper on a table and transports it. |
| task_0076 | 4 | scene_1, scene_2 | scene_3, scene_4 | Grasp the glasses from the table and place them on the mouse pad | Robot grasps a small object on a wooden table and transports it to a target location. |
| task_0077 | 4 | scene_1, scene_2 | scene_3, scene_4 | Grasp a black object from a metal pot and place it into a white bowl on a wooden table | Robot grasps a ladle from a bowl and places it on the table |
| task_0078 | 4 | scene_1, scene_2 | scene_3, scene_4 | Pick up the fly swatter and place it on the cardboard | Robot grasps a fly swatter and places it on a cardboard surface |
| task_0079 | 4 | scene_1, scene_2 | scene_3, scene_4 | Grasp a small object on the table and place it on the table | Grasp a small object on a table and transport it |
| task_0080 | 4 | scene_1, scene_2 | scene_3, scene_4 | Grasp a small ball on a table and place it into a yellow ring. | Grasp a small object on a table and transport it |
| task_0081 | 4 | scene_1, scene_2 | scene_3, scene_4 | Grasp a small stick and place it on a vintage telephone | Robot grasps a small object and places it into a decorative container on a table. |
| task_0082 | 4 | scene_1, scene_2 | scene_3, scene_4 | Pick up a small object from a pink box and place it into a light blue box on the table | Robot grasps a small object from a pink box and places it into a gray box on the table |
| task_0083 | 4 | scene_1, scene_2 | scene_3, scene_4 | Place an object into a box on the table | Robot grasps a small object and places it into a cardboard box on a table |
| task_0084 | 4 | scene_1, scene_2 | scene_3, scene_4 | Open the drawer of a white storage box and place a small cube on the table | Robot grasps a small cube from a white container and places it on a table |
| task_0085 | 4 | scene_1, scene_2 | scene_3, scene_4 | Grasp a glass on the table and place it on the table | Robot grasps a glass and transports it to a target location on the table |
| task_0086 | 4 | scene_1, scene_2 | scene_3, scene_4 | Wipe the table surface with a cloth | Robot grasps a cloth on a table and places it on the table |
| task_0087 | 3 | scene_1, scene_2 | scene_4 | Grasp a small yellow object with a green stem and place it on a white surface | Robot grasps a brush and transports it to a target location on a flat surface. |
| task_0088 | 4 | scene_1, scene_2 | scene_3, scene_4 | Grasp a bowl and place it on a cardboard surface | Grasp a small bowl and transport it to a target location on a cardboard surface |
| task_0089 | 4 | scene_1, scene_2 | scene_3, scene_4 | Place a bowl on a cardboard surface | Robot grasps a small object from a container and places it on a board |
| task_0090 | 4 | scene_1, scene_2 | scene_3, scene_4 | Grasp a tool and place it on a table | Robot grasps and places a small object onto a phone on a table |
| task_0091 | 4 | scene_1, scene_2 | scene_3, scene_4 | Grasp a small object from a black holder and place it on a table | Robot grasps a pen and places it into a pen holder on a fabric surface |
| task_0092 | 4 | scene_1, scene_2 | scene_3, scene_4 | Grasp a red pen from a clamp and place it on the table | Robot grasps a tool and performs an action on a fabric surface |
| task_0093 | 4 | scene_1, scene_2 | scene_3, scene_4 | Plug a white power strip into a socket on the table | Robot grasps a power strip on a table and performs a pressing action on it |
| task_0094 | 4 | scene_1, scene_2 | scene_3, scene_4 | Plug the lamp cord into the power strip | Robot grasps a power strip and places it on a table |
| task_0095 | 4 | scene_1, scene_2 | scene_3, scene_4 | Human presses a keyboard on a desk | Robot grasps a small object and places it on a circular target on the table |
| task_0096 | 4 | scene_1, scene_2 | scene_3, scene_4 | Grasp a pipette from a container and place it on the table | Robot grasps a small object and places it into a container on the table |
| task_0097 | 4 | scene_1, scene_2 | scene_3, scene_4 | Grasp a phone and place it on the table | Robot grasps a small container on a table |
| task_0098 | 4 | scene_1, scene_2 | scene_3, scene_4 | Grasp a blue box on the table and place it on a transparent surface | Robot grasps a small object on a table and transports it |
| task_0099 | 4 | scene_1, scene_2 | scene_3, scene_4 | Grasp a small object from a container and place it into another container on the table | Robot grasps a small object from a container and places it into another container on the table |
| task_0100 | 4 | scene_1, scene_2 | scene_3, scene_4 | Use a knife to cut a vegetable on a cutting board | Robot uses knife to cut vegetable on cutting board |
| task_0101 | 4 | scene_1, scene_2 | scene_3, scene_4 | Cut green vegetables on a wooden cutting board using a knife | Robot grasps a knife to cut green onions on a cutting board |
| task_0102 | 4 | scene_1, scene_2 | scene_3, scene_4 | Use a knife to cut a green pepper on a wooden cutting board | Robot grasps a knife and cuts a green pepper on a cutting board |
| task_0103 | 4 | scene_1, scene_2 | scene_3, scene_4 | Use a knife to cut an object on a cutting board | Robot grasps a knife and cuts a piece of food on a cutting board |
| task_0104 | 4 | scene_1, scene_2 | scene_3, scene_4 | Grasp knife and cut vegetable on cutting board | Robot grasps a knife to cut a white object on a wooden cutting board |
| task_0105 | 4 | scene_1, scene_2 | scene_3, scene_4 | Grasp a knife and cut an onion on a cutting board | Robot grasps an object and uses a knife to cut it on a cutting board |
| task_0106 | 4 | scene_1, scene_2 | scene_3, scene_4 | Cut an orange on a cutting board using a knife | Robot grasps a knife and cuts orange slices on a cutting board |
| task_0107 | 4 | scene_1, scene_2 | scene_3, scene_4 | Use a knife to cut a vegetable on a cutting board | Robot grasps a vegetable and uses a knife to cut it on a cutting board |
| task_0108 | 4 | scene_1, scene_2 | scene_3, scene_4 | Use a knife to cut an object on a cutting board and place the cut pieces on a rack | Robot grasps a knife and cuts a piece of food on a cutting board |
| task_0109 | 4 | scene_1, scene_2 | scene_3, scene_4 | Use a knife to cut a cucumber on a cutting board | Robot grasps a knife and cuts a vegetable on a cutting board |
