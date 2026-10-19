# 27个null教师scene的视觉复核

日期：2026-10-19。范围为`jobs/scene-s3-r3/annotations.jsonl`中
`description=null`的27个cfg2 scene，不是全数据集的配对质量审计。

本次没有发现足以按“明确错配才排除”规则删除的scene：27个保留，0个排除。
这表示证据不足以排除，不表示已经证明所有片段的动作、时段和标签完全正确。
上次列出的礼盒移动/合盖、夹具取笔/接触等属于文本冲突候选，不能写成已经确认的原数据错配。
用户明确接受的“人类按插排开关、机器人插入插头”保留。

## 核查方法与证据

- 按原标注的`source.observations.sample_id`定位子集实际视频，没有另换相机。
- 27个scene共108段视频，每段在完整时长上均匀读取8帧（包含首尾）。
- 对`task_0026/scene_4`、`task_0029/scene_4`、`task_0046/scene_4`、
  `task_0092/scene_1`、`task_0101/scene_4`的20段视频进一步各读取24帧。
- 这是抽帧视觉核查，不是逐帧播放或与D盘原始数据重新核对映射。
  不能用它断定瞬时动作没有发生、机器人一定成功或一定失败。
- 排除依据必须是明确错误的场景/任务内容对应，不能只依据null教师、模型描述差异、
  操作同一对象的不同阶段、背景外观变化、人员协助、机器人未完成动作或检索失败。

完整逐条结论在[核查清单](SCENE_REVIEW.csv)，四视频对照图在
[`artifacts/semantic/v2/scene_review/`](../../artifacts/semantic/v2/scene_review/)。
每张`cfg2_task_xxxx_scene_x.jpg`依次为两个人类视角、两个机器人视角；
`*_human_detail.jpg`、`*_robot_detail.jpg`为各相机24帧的补充图，时间标注按各视频独立计算。

例如，台灯场景的机器人描述可能把灯座附近物体当作主对象，但实际画面仍有台灯；
方向盘背面被描述为提箱，另一视角能确认方向盘；复古电话、夹具、蔬菜和砧板也能在人机两侧找到对应。
这些都不足以确认配对错误。相关证据：
[台灯](../../artifacts/semantic/v2/scene_review/cfg2_task_0020_scene_4.jpg)、
[方向盘](../../artifacts/semantic/v2/scene_review/cfg2_task_0044_scene_2.jpg)、
[夹具](../../artifacts/semantic/v2/scene_review/cfg2_task_0092_scene_1_robot_detail.jpg)。

## 最小改动入口

新增独立工具`runs/filter_scenes.py`，只读取清单中明确标为`exclude`的scene；
`keep`行不会参与排除，也不会从教师null或检索指标自动推断排除项。
目前没有exclude行，因此不需要改任何训练或评估命令。

只检查影响、不写文件：

```bash
python runs/filter_scenes.py --cfg cfg2 \
  --source artifacts/manifests/retrain/s44/cfg2/split_manifest_video.json \
  --dry-run
```

显式生成一份清单副本：

```bash
python runs/filter_scenes.py --cfg cfg2 \
  --source artifacts/manifests/retrain/s44/cfg2/split_manifest_video.json \
  --output /tmp/cfg2_video_filtered.json
```

工具兼容run格式的轨迹scene ID和视频sample ID，按cfg隔离，过滤train/val/test的全部相机，
保持其它样本和排序，不补抽样本。拒绝覆盖输入或已有输出。
它不修改数据、公共协议、run、params、教师、指标，也不自动接入现有流水线。
因此现有本机训练、评估指纹和历史评估复用规则均不改变。

将来有明确证据并改为exclude后，生成的副本可通过现有训练配置的
`split.manifest_path`或独立评估入口的`--split-manifest`显式使用。
不要把副本替换为公共`metadata.json`，也不要改写公共协议的120/50划分。
task划分实验必须过滤自己的原清单，不能换成普通scene划分清单；cfg3使用其自己的cfg和清单。
所有对比run需应用同一排除策略。仅重评不会移除旧模型已见过的训练样本影响。

## 本次实际处理

没有排除任何训练、验证、测试样本，没有修改任何既有配置或划分。
27个null教师继续保持原状：它们不会进入语义教师库，但其视频/轨迹仍按原清单参与实验。
没有付费重标注、没有重新训练或覆盖评估结果。
