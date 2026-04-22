语义模态配置目录。

关键字段：

- `execution_mode`
  - `sync`：走现有同步语义提取链路，使用 `tools/run_semantic_annotation.py`
  - `batch`：走批处理链路，顺序使用：
    1. `tools/build_semantic_batch_requests.py`
    2. `tools/submit_semantic_batch.py`
    3. `tools/sync_semantic_batch_jobs.py`
    4. `tools/ingest_semantic_batch_results.py`

- `prompt_mode`
  - `two_stage`：同步链路支持
  - `single_stage`：同步链路和 batch 链路都支持；当前 batch 只支持这一种

- `batch_endpoint`
  Batch 请求文件中每条请求的目标端点，默认 `/v4/chat/completions`

- `batch_max_requests_per_file`
  单个 batch request 文件的最大请求数

- `batch_max_file_bytes`
  单个 batch request 文件的最大字节数。视频多帧 base64 输入体积较大时，实际分片通常由这个字段决定。
