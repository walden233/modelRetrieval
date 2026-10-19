# 32GB视频主实验

默认batch名`v32`表示本组32GB配置，随机种子仍为44。产物在
`artifacts/runs/retrain/v32/video/`，与`s44`及跨相机`v44`分开。

| 实验 | 骨干 | 适配头 | 骨干训练范围 | 跨域损失 | 模态内增强 | 划分 |
|---|---|---|---|---|---|---|
| E0 | V-JEPA2 ViT-L | 双头 | 最后2个block | 单正样本 | 无 | scene |
| E1 | 同上 | 双头 | 最后2个block | task多正样本 | 无 | scene |
| E2 | 同上 | 双头 | 最后2个block | 单正样本 | 有 | scene |
| E3 | 同上 | 双头 | 最后2个block | task多正样本 | 有 | scene |
| E4 | 同上 | 双头 | 全部冻结 | task多正样本 | 有 | scene |
| E5 | 同上，24帧、batch16 | 双头 | 最后2个block | task多正样本 | 有 | scene |
| E6 | 同上 | 双头 | 最后2个block | task多正样本 | 有 | task |
| E7 | VideoMAE-Base | 双头 | 最后2个block | task多正样本 | 有 | scene |
| RawJ / RawM | 原始V-JEPA2 / VideoMAE | 不用于编码 | 原始预训练模型，仅评估 | 无训练 | 无 | scene |

跨相机扩展全部继承E3（微调最后2层、双头、task多正样本、模态内增强），形成跨视角训练策略与语义监督的两因素对照：

| 实验 | batch采样 | scene正样本权重 | 语义监督 |
|---|---|---|---|
| E3 | 随机 | 0.5 | 无 |
| V0 | 8个不同task，各取一个scene的两个相机 | 1.0 | 无 |
| V1 | 随机 | 0.5 | 训练集文本库监督 |
| V2 | 与V0相同 | 1.0 | 与V1相同 |

V0/V2的`loss.scene_positive_weight=1.0`同时用于人域和机器人域的模态内损失，
使同一scene另一个相机与自身增强的原始权重均为1，归一化后各占1/2。
跨视角策略包含组织采样和更强的跨相机约束，不能把E3→V0的变化单独归因于采样。
同task不同scene的权重仍为0.25，整体`lambda_intra`仍为0.2。
其它配置缺省scene权重为0.5，原本机视频和轨迹实验不变。

V1/V2复用S3的scene教师`artifacts/semantic/v2/teachers/S3/cfg2.npz`。
只把当前train中非null的有效教师纳入文本库，快照保存到各run的`semantic_teachers.npz`；缺失教师的scene继续参与原视频损失。
共享辅助投影为128→1024维，语义KL权重0.1、文本/视频温度均0.1、1epoch线性预热；辅助投影不参与最终检索。
语义参数沿用已有文本库方案，本配置不包含旧的batch内语义监督实验。

E3为完整方法。E4与E3比较冻结/微调；E5沿用E3架构和batch16，仅将采样帧数由16改为24，用于比较增加帧数的影响。双头指人类和机器人分别使用残差适配头，最终检索投影共享。
E6的cfg2测试集按任务重新划分，约120个test、50个val scene；各task完全隔离，实际数量以run内清单为准。其余实验严格共用新batch的划分；cfg3均使用每task两个scene的独立测试清单。
新batch独立生成划分，可能与历史s44不同；损失和结构消融在新batch内部比较，不把新旧指标直接当作同清单对照。

## 训练设置

- 全部配置的训练和评估batch均为16组人机配对，即每个训练batch包含32段原始视频；E5同样为batch16。均不使用梯度累积。
- E0–E7及V1使用`random`：每epoch随机打乱全部train样本，不强制task或scene重复；尾batch按现有训练入口保留。
- V0/V2显式启用新选项`scene_pairs`：每batch为8个不同task，每task一个scene的两个相机，共16对。同一个样本内人类与机器人视频仍来自同相机。
- 组织采样每epoch覆盖全部train scene；不足整batch时从其它train task补充完整双相机scene，不从val/test取样。同seed下V0/V2的batch顺序一致。
- 旧配置的`random`、`task_pairs`、batch4的`scene_views`行为均不改变；`configs/retrain/s43`及`configs/retrain/views`保持原样。
- 默认16帧，E5为24帧，使用各骨干原有processor；V-JEPA2输入为256分辨率，VideoMAE使用原有预处理和特征接口。
- 7epoch，AdamW学习率1e-5、weight_decay=0.01、线性调度、5%warmup、AMP。模态内权重0.2。
- 516个train配对时随机batch16每epoch33步，7epoch共231步，尾batch为4对；E5步数和尾batch相同。组织采样按scene覆盖及task数量补齐，实际步数由清单决定；E6按自身train规模计算。
- 按cfg2验证task MRR选择best，仅保存`best_model.pth`，不保存last。

按最后两层微调＋增强的小batch实测趋势外推，16帧batch16的张量峰值约19.8GiB；E5的24帧batch16显存需求更高。仍需为缓存和CUDA运行时留出余量，两种设置均需在目标机器实测。

## 入口

```bash
conda activate torch2
# 准备、审计、全部元信息预先生成、训练、cfg2/cfg3评估与绘图
python runs/retrain.py --config configs/retrain/video32/run.json --batch v32 --modality video

# 独立训练、重新评估、重绘
python runs/retrain.py --config configs/retrain/video32/run.json --batch v32 --modality video --stage train
python runs/retrain.py --config configs/retrain/video32/run.json --batch v32 --modality video --stage eval
python runs/redraw.py --batch v32

# 选定实验；正式测试前可只做准备和审计
python runs/retrain.py --config configs/retrain/video32/run.json --batch v32 --modality video --stage prepare
python runs/retrain.py --config configs/retrain/video32/run.json --batch v32 --modality video --stage audit
python runs/retrain.py --config configs/retrain/video32/run.json --batch v32 --modality video --only E0 E3

# 仅训练和评估跨相机扩展；已有公共清单时可跳过prepare阶段
python runs/retrain.py --config configs/retrain/video32/run.json --batch v32 --modality video --only V0 V1 V2
```

独立`--stage train`需先完成prepare；新机器运行完整入口会自行准备清单，配置不引用旧E0–E7的run。数据放在`dataset/RH20T_subset`。
新配置的`split.require_disjoint=true`在审计、训练准备和训练入口检查scene互斥，E6还检查task互斥；发现旧的重叠清单会报错，不能继续训练。
迁移到32GB机器时，除数据和本目录外，还需同步代码及S3教师npz；无需运行大模型标注接口。

## 离线骨干目录

本组全部配置显式设置`model.backbone_dir`，训练、已训练模型评估和RawJ/RawM均从项目内加载骨干及processor：

```text
artifacts/models/video/
  facebook--vjepa2-vitl-fpc64-256/
    config.json
    model.safetensors
    video_preprocessor_config.json
  MCG-NJU--videomae-base/
    config.json
    model.safetensors
    preprocessor_config.json
```

复制这两个完整目录到训练机的同一相对位置即可，不需要复制系统HF缓存。
目录中的文件是实际文件，不是指向本机缓存的符号链接；约1.214GiB和0.351GiB。
加载时显式使用`local_files_only=true`，目录缺失会报错，不会退回系统缓存或自动下载。
`backbone_name`保留原HF模型ID，用于模型身份和历史原始骨干pth定位。
已有的两份同名`.pth`也可一并迁移供Raw评估复用；仅复制两个完整目录时，首次Raw评估会自动生成对应公共`.pth`。
旧配置未设置`backbone_dir`时仍按原方式从HF缓存/模型ID加载，其配置不变。

## AutoDL迁移

硬件推荐型号明确的RTX 5090 32GB；vGPU-32GB也可候选，但须确认实例底层型号和实测耗时。
AutoDL官方首页将部分vGPU-32G标为RTX 4080S 32G，不能仅凭vGPU名称认定其为共享切片。
[官方GPU信息](https://www.autodl.com/home)。

5090选Python 3.12、PyTorch 2.7.0、CUDA 12.8镜像。
本机torch2是PyTorch 2.7.0＋CUDA 12.6，不直接复制Conda目录到5090。
[PyTorch 2.7的Blackwell/CUDA 12.8支持](https://pytorch.org/blog/pytorch-2-7/)，
[AutoDL镜像列表](https://api.autodl.com/docs/base_config/)。

### 文件与空间

同步最新项目代码及配置、上节两个完整骨干目录、`artifacts/semantic/v2/teachers/S3/cfg2.npz`、
`dataset/RH20T_subset/RH20T_cfg2`和`RH20T_cfg3`。
视频新batch的公共划分仍依据手轨迹有效性选择scene，所以各scene的
`human_hand_3d.npy`和`tcp_base.npy`也必须上传。单独上传MP4不足以准备清单。
只做视频时无需上传深度视频、夹爪缓存、系统HF缓存或历史训练run。

完整两份子集当前约25.0GiB，完整离线骨干目录约1.57GiB，教师约1.6MB。
后续11个训练run的best权重约需十几GiB，Raw评估还会生成两份公共pth。
若同时保留上传压缩包、解压数据及重跑归档，建议至少80–100GB数据盘；以实际剩余空间为准。
将项目及数据放在`/root/autodl-tmp/BISE`等本地数据盘位置。

新机从源码配置运行并现场生成`v32`清单，不复制本机旧`pipeline.json`、run快照或整份旧清单目录：
旧metadata/params里有本机绝对路径。数据完整且seed一致时，新机的划分规则和样本选择可复现。

### 环境

`requirements.txt`是视频流水线所需的依赖集，固定本机Transformers开发版对应Git提交，
不直接升级到最新版。Torch/CUDA由镜像提供，依赖文件不替换其CUDA构建。

```bash
cd /root/autodl-tmp/BISE
python -c 'import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.version.cuda); print(torch.cuda.get_device_name(0))'
# 应为torch 2.7.0、torchvision 0.22.0、CUDA 12.8
python -m pip install -r configs/retrain/video32/requirements.txt
python -m pip install -e . --no-deps

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MPLBACKEND=Agg
```

依赖安装需要联网，训练和评估不需要HF网络或语义API key。
若镜像Torch版本不符，先按[官方安装指令](https://pytorch.org/get-started/previous-versions/)安装匹配的cu128版：

```bash
python -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
```

### 先验证再全跑

```bash
python -m pytest -q tests/test_video_local_models.py tests/test_video_losses.py
python runs/retrain.py --config configs/retrain/video32/run.json --batch v32 --modality video --stage prepare
python runs/retrain.py --config configs/retrain/video32/run.json --batch v32 --modality video --stage audit

# 单独smoke batch：验证两种骨干、24帧及语义分支，同时实测显存；不改正式7epoch配置
python runs/retrain.py --config configs/retrain/video32/run.json --batch v32smoke --modality video --only E3 E5 E7 V2 --video-epochs 1
```

这是一轮小epoch完整验证，仍会训练和评估，不是仅导入检查。
必须先确认目标机器上的16帧batch16、E5的24帧batch16能完成优化器更新；本机8GB测量不能代替该实测。
建议另开终端用`nvidia-smi`观察显存。若OOM，先统一调整受控比较组的batch，再正式训练；
普通梯度累积不等价于扩大对比损失的候选池。

验证通过后，在tmux会话中运行上节完整v32入口，避免SSH断线终止训练。
中断后续跑使用同一入口加`--skip-existing`：已完成run跳过训练，未完成run归档后从头训练，
已有选定评估仍重新生成。这不是epoch/step级断点恢复。

回传时使用`python scripts/pack_retrain.py`导出不含权重的产物；best权重另行备份。
该压缩包适合保留结果与重绘；在不同项目路径下重新评估还需要重新绑定params及协议中的绝对路径，
不能把“可看结果”当作“搬回本机后直接续训/重评”。

默认命令会归档已有选定run后从头训练。加`--skip-existing`可复用已完成训练；未完成run仍归档重训，不是step级恢复。旧`s43`配置和`s44/v44`产物不作迁移。
