# 第三章语义实验

配置为 [ch3.json](ch3.json)，当前 schema_version=3。全部产物统一在 `artifacts/semantic/v2`，无需指定 batch。

```bash
conda activate torch2
# S0–S5：标注、编码、cfg2/cfg3 评估、分析、绘图、报告
python runs/semantic/run.py
# 固定55 task、220视频，GLM32/16/8成本实验
python runs/semantic/run.py --stage cost
# 选定一种表示，汇总scene教师文本与向量
python runs/semantic/run.py --stage scene --source S4
# 单独移除S1，S2–S5保持独立可用
python runs/semantic/run.py --stage remove --only S1
# 只重绘语义图片，无API
python runs/redraw.py --root artifacts/semantic/v2
```

主模型 GLM-5.3-Flash，20帧。S0–S5 一次联合提取，各组单独保存完整文本、向量和评估。模型只输出任务内容，不输出理由、可见性／不确定性说明或证据引用。程序记录时间、费用、来源和失败。

GLM key 读 `ZHIPU_API_KEY` 或 `GLM_API_KEY`。可显式 `--stage cost --providers glm,qwen` 启用 Qwen16，需要 `DASHSCOPE_API_KEY`。三个入口互相独立，主入口不运行成本或教师生成。教师默认来源为配置中的 S5，可用 --source S0…S5 改选，产物在 teachers/S4/ 等独立目录。重复命令可以续跑。首次 prepare 归档上次格式的产物，旧结果不会混入当前实验。

独立环节、论文选组、增量标注、scene教师和完整布局见 [实现与命令](../../docs/semantic/PLAN.md#14-已实现的统一入口)。本目录其他 v1 JSON 属于历史流水线，不用于当前第三章入口。

六组联合协议会创建新的 job 修订；旧的独立 S0 和五组联合结果及费用账本保留，不能用来跳过新协议请求。--only 只选择保存和评估的组，模型仍一次输出完整六组。原 --suite chapter 参数已移除。

少量标注失败默认不阻断主流水线或成本评估。成功项继续编码；各组／成本变体使用共同可用集合，原始数据保存缺失ID与覆盖率，图表标为PARTIAL。直接重跑同一命令只补失败／缺失项，随后更新评估与图表，无需 --allow-partial。全部失败或没有可评估的人机集合仍会报错。
