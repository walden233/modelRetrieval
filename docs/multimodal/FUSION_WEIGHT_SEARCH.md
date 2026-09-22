# 第三章融合权重搜索

本次只使用已有cfg2/cfg3检索库，不训练模型、不重新提取特征。

```bash
conda activate torch2
bash scripts/search_fusion_weights.sh
```

默认使用`configs/retrieval/system_default.json`的校准、缺失模态策略和场景聚合设置。搜索固定为video、trajectory、semantic_text三个模态，覆盖0.1步长的66组权重及精确等权参照，共67组。按cfg2调权集合的task MRR选择；并列依次比较NDCG@10、距离等权的远近和权重字典序。cfg3只计算等权与选定权重，不参与选择。

cfg2查询自动根据建库checkpoint同目录的`split_manifest.json`排除任一模型训练过的scene，且要求三个模态齐全、scene在两个模型的划分中都有记录。调权集合包含现有非训练数据，不声称是一份新建立的独立验证集。候选库保持不变，其中允许包含已知训练执行；cfg3不应用cfg2的scene名称过滤。

评分直接复用`evaluate_retrieval_system`，沿用camera打分、校准、融合、再聚合scene的顺序，没有把已聚合的模态矩阵直接相加。历史均匀融合指标可能与重新计算不同，正式对比使用本次同一次评价输出。

默认输出：`artifacts/retrieval/weight_search_cfg2_to_cfg3/`。

| 文件 | 内容 |
|---|---|
| `weight_search.csv/json` | 全部67组权重及cfg2指标，也是搜索图原始数据 |
| `weight_search.png` | 权重分布与MRR |
| `best_weights.json` | 最优权重和选择时的cfg2指标 |
| `best_config.json` | 可直接交给现有检索/评估入口的完整配置 |
| `final_comparison.csv/json` | cfg3等权与最优权重的task指标对照 |
| `query_filter.json` | 调权查询ID、训练划分来源和排除原因 |
| `run_info.json` | 搜索配置、两个检索库的模型来源 |
| `cfg2_tuning/{uniform,selected}/` | 调权指标、案例、矩阵与行列ID |
| `cfg3_test/{uniform,selected}/` | 最终scene/task/mixed指标、案例、矩阵与行列ID |

输出目录已有内容时会退出，避免覆盖结果；重跑指定新目录：

```bash
bash scripts/search_fusion_weights.sh --output-dir artifacts/retrieval/weight_search_cfg2_to_cfg3_run2
```

也可直接执行Python入口并指定全部参数：

```bash
python runs/retrieval/search_fusion_weights.py \
  --search-library artifacts/retrieval/rh20t_cfg2_v1 \
  --test-library artifacts/retrieval/rh20t_cfg3_v1 \
  --config configs/retrieval/system_default.json \
  --step 0.1 \
  --output-dir artifacts/retrieval/weight_search_cfg2_to_cfg3
```

两个库必须声明不同cfg并使用同一视频、轨迹checkpoint路径。脚本不修改模型、检索库或旧实验文件。论文只需补“搜索设置＋权重图＋cfg3最终对比表”；这里的最优是指定搜索空间及cfg2主指标内最优，不保证cfg3全部指标均优于等权。
