# 水质传感器异常复核 Agent

给定站点、绝对时间范围和问题，程序从 SQLite 复核 pH 突降，将同站现场日志与 LightRAG 检索到的手册片段带入 Markdown 或 JSON 报告。可直接运行随附演示，也可从现场 CSV 导入观测、基线与运维记录，使用自己的 Markdown 手册复核。随附观测和运维资料均为模拟，基线为预设值。

本示例基于 [HKUDS/LightRAG](https://github.com/HKUDS/LightRAG) 的 commit [`ce3aa6bfe0982003c0684b408995945582681f05`](https://github.com/HKUDS/LightRAG/commit/ce3aa6bfe0982003c0684b408995945582681f05) 开发，沿用官方示例的 `initialize_storages()`、`ainsert()` 和 `finalize_storages()`，通过 `aquery_data()` 获取片段及来源。仓库保留上游 [MIT 许可证和 LightRAG Team 版权声明](../../LICENSE)。

## Windows PowerShell

在 LightRAG checkout 根目录执行。使用 Python 3.11，真实运行会在首次使用时下载本地向量模型和分词文件。

```powershell
py -3.11 -m venv .venv
$python = '.\.venv\Scripts\python.exe'
$env:PYTHONUTF8 = '1'
& $python -m pip install -e . -r .\examples\sensor_anomaly_rag\requirements.txt
Copy-Item .\examples\sensor_anomaly_rag\.env.example .\examples\sensor_anomaly_rag\.env
```

在示例目录的 `.env` 设置 `SENSOR_LLM_API_KEY`。默认 DeepSeek 端点也可直接使用 PowerShell 进程中已有的 `DEEPSEEK_API_KEY`。`SENSOR_LLM_BASE_URL` 和 `SENSOR_LLM_MODEL` 用于配置 OpenAI 兼容对话端点，`SENSOR_EMBEDDING_MODEL` 指定 FastEmbed 本地向量模型。默认模型是 `deepseek-flash` 和 `BAAI/bge-small-zh-v1.5`。同名环境变量优先于 `.env`。

### 导入自己的数据并生成报告

在仓库已忽略的 `data/` 目录中准备 UTF-8 CSV，支持带 BOM 的 PowerShell 导出。时间使用带偏移的 ISO 8601 整秒，pH 和基线允许 `0` 到 `14` 的百分位数值。站点标识区分大小写，导入会去掉标识首尾空格。日志 `id` 在数据库中全局唯一，`kind` 填 `维护` 或 `校准`。观测和日志可分别导入。

`observations.csv`

```csv
site,timestamp,observed_ph,baseline_ph
East,2026-11-23T21:30:00+08:00,7.19,7.20
East,2026-11-23T22:00:00+08:00,5.84,7.20
```

`logs.csv`

```csv
id,site,timestamp,kind,note
E-001,East,2026-11-23T21:45:00+08:00,维护,值班员清洗探头
W-001,West,2026-11-23T22:00:00+08:00,校准,另站校准失败
```

```powershell
& $python -m examples.sensor_anomaly_rag import-data --database .\data\field.sqlite3 `
  --observations .\data\observations.csv --logs .\data\logs.csv

& $python -m examples.sensor_anomaly_rag review --database .\data\field.sqlite3 --site East `
  --start '2026-11-23T21:00:00+08:00' --end '2026-11-23T23:00:00+08:00' `
  --logs-only --format json --output .\data\east-review.json
```

重复导入相同数据会计入 `already_present`。如果同一站点和时刻的观测，或同一日志 ID 的内容冲突，命令会指出 CSV 行号并回滚整批。现场记录模式只需 Python 标准库。省略 `--format json` 可生成 Markdown，省略 `--output` 可在终端查看结果。报告文件使用新路径保存。JSON 中的 `schema_version`、`status`、事件、日志和手册引用可直接供下游程序读取。

需要使用现场手册时，去掉 `--logs-only`，加上 `--manual-dir .\data\site-manuals`。目录内放 UTF-8 `.md` 文件。建索引时 LightRAG 会把手册内容送到配置的对话模型做抽取。现场数据保存在本机 SQLite，`naive` 检索由本地向量模型完成。

首次索引成功后可重复使用，后续复核沿用相同语料与模型配置时无需对话 API 凭据。Windows 默认索引目录为 `%LOCALAPPDATA%\SensorAnomalyRAG\indexes`，也可通过 `SENSOR_INDEX_DIR` 或 `--index-dir` 指定其他目录。手册或模型配置变化会使用新索引。构建失败时重新运行可重试，冷启动遇到服务故障时可用 `--logs-only` 完成现场复核。使用新索引目录可以重建本地索引。

现场使用前需准备站点认可的基线、pH 判定阈值和运维手册，并用本地历史案例核对输出。应用以批次 CSV 和单机 CLI 为使用入口。

### 固定时间演示

```powershell
& $python -m examples.sensor_anomaly_rag review --site A `
  --start '2026-01-15T21:00:00+08:00' --end '2026-01-15T23:00:00+08:00' `
  --question 'A 站昨晚 pH 突降，有什么记录可以解释？'
```

需要无凭据演示时，在 `review` 后加 `--offline`。CLI 报告会注明替身检索。时间范围始终取 `--start` 和 `--end`，因此问题中的“昨晚”不会随运行日期移动。起点包含、终点排除。默认 `--threshold 0.80` 标记 `观测值 − 基线值 ≤ −0.80` 的观测。`--window-minutes 90` 设置日志查询的前后窗口，含两端。

### 离线测试与对照评测

只运行替身演示或替身评测时，Python 标准库即可。测试需要 pytest、上游 pytest 配置使用的插件和索引锁依赖。三个命令均无需模型凭据。

```powershell
& $python -m pip install 'pytest>=8.4,<10' 'pytest-asyncio>=1.2,<2' 'filelock>=3.13,<5'
& $python -m pytest .\examples\sensor_anomaly_rag\tests -q
& $python -m examples.sensor_anomaly_rag evaluate --offline
```

已配置模型凭据时，运行真实 LightRAG 对照评测。可选 `--output` 将逐案例的来源、片段 ID、日志 ID 和指标写成 UTF-8 JSON。

```powershell
& $python -m examples.sensor_anomaly_rag evaluate --output .\sensor-evaluation.json
```

## 模拟数据

[`fixtures/schema.sql`](fixtures/schema.sql) 定义导入与演示共用的表，[`fixtures/demo.sql`](fixtures/demo.sql) 写入固定的五条观测和三条运维日志，无随机数。随示例提供 [`fixtures/demo.sqlite3`](fixtures/demo.sqlite3)。库内时间为 UTC Unix 整秒，输入及下表用 `+08:00`。pH 用百分位整数存储，Python 转成 `Decimal` 后相减。

| 案例 | 数据及预期行为 |
| --- | --- |
| A 站，1 月 15 日 22:00 | pH `5.85`，基线 `7.20`，偏差 `−1.35`。命中当晚 21:45 的清洗工单 `A-0115-clean`。工单记录清洗，缺少独立复测和探头响应检查。 |
| B 站，同日 22:00 | `B-0115-cal` 记录校准失败。复核 A 时按站点排除该日志。B 的观测 `7.16` 相对基线 `7.21` 未触发演示阈值。 |
| A 站，1 月 8 日 22:00 | `A-0108-cal` 记录缓冲液过期。复核 1 月 15 日事件时按日志窗口排除。 |
| C 站，1 月 15 日 22:00 | pH `6.12`，基线 `7.05`，偏差 `−0.93`。没有同站现场日志，有手册建议，报告显示原因未确定。 |
| A 站，1 月 16 日 21:00–23:00 | 没有观测，返回 `no_observations`，直接结束复核。 |

演示观测还包括 A 站突降前后的两条普通读数。需要重建时，用下列命令另存一份，并可通过 `review --database .\demo-rebuilt.sqlite3` 使用它。

```powershell
& $python -m examples.sensor_anomaly_rag build-db --output .\demo-rebuilt.sqlite3
```

## 对照设计与实测

三组案例为 A 站有现场日志、C 站只有手册、A 站空时间段。两个流程共用一个索引，语料包括两篇手册和由 SQLite 三条日志导出的 Markdown 文本。设置相同的 `naive` 检索、`top_k=5`、`chunk_top_k=5` 和关闭重排。真实运行时也共用同一组模型配置。

- **固定检索**以问题、站点和请求时间为固定查询模板，引用所有召回片段。匹配到日志出处时记录其站点和时间，供评测判分。
- **事件复核**先查观测和同站窗口日志，再用包含实测 pH、基线和偏差的问题检索。报告使用 SQLite 取到的现场日志，以及已知手册来源的召回片段。常规 `review` 命令的向量索引只收录手册。

错站和超窗各按含错误引用的案例数统计。同一案例可以同时记入两项。对于无观测案例，评测用请求时间段判定日志是否超窗。两起突降案例的预期原因状态均为证据不足，两流程使用同一保守判定。

在 Windows PowerShell 和 Python 3.11.9 下运行，**14 项离线测试通过**。真实运行使用 `deepseek-flash` 做文档入库时的抽取、`BAAI/bge-small-zh-v1.5` 做向量化，成功执行三案例评测及 A 站 CLI 演示。另用上面的 East 站 CSV 和自写站点手册完成导入、重复导入、Markdown／JSON 导出和真实检索。报告核验出偏差 `−1.36`，现场仅引用 `E-001`。A 站建索引后，移除凭据重跑也成功复用了索引。[离线逐案例结果](results/offline.json) · [真实 LightRAG 逐案例结果](results/lightrag.json)

| 指标 | 离线固定检索 | 离线事件复核 | 真实固定检索 | 真实事件复核 |
| --- | ---: | ---: | ---: | ---: |
| 引用错站日志的案例 | 3/3 | 0/3 | 3/3 | 0/3 |
| 引用超窗日志的案例 | 3/3 | 0/3 | 3/3 | 0/3 |
| 日志集合与预期完全匹配 | 0/3 | 3/3 | 0/3 | 3/3 |
| 正确核验观测、基线、偏差及事件时间 | 未核验 | 2/2 | 未核验 | 2/2 |
| 对突降事件正确标记证据不足 | 2/2 | 2/2 | 2/2 | 2/2 |
| C 站案例引用手册 | 是 | 是 | 是 | 是 |
| A 站空时段返回 `no_observations` | 否 | 是 | 否 | 是 |

这批语料每份文档只有一个片段，真实检索也召回了全部五个片段。错引减少来自 SQLite 的站点和时间约束。该评测衡量证据归属及空数据处理，样本覆盖的原因状态只有“证据不足”。

## 从演示问题跟进代码

1. [`__main__.py`](./__main__.py) 的 `main()` 接收问题和 `--site`、`--start`、`--end`。`run_review()` 生成请求并打开 SQLite。
2. [`review.py`](./review.py) 的 `collect_events()` 调用 [`database.py`](./database.py) 的 `read_observations()` 查 A 站观测。观测 `5.85` 减基线 `7.20` 得 `−1.35`，触发阈值。接着 `find_site_logs()` 用精确站点和事件前后 90 分钟找出 `A-0115-clean`。
3. [`retrieval.py`](./retrieval.py) 的 `open_retriever()` 复用现有索引，首次使用时通过上游 `ainsert(..., file_paths=...)` 建索引。`review_events()` 组织异常查询，`Retriever.retrieve()` 调用上游 `aquery_data()` 并读取 `data.chunks`、`data.references`，核对片段和文件来源。
4. [`review.py`](./review.py) 的 `render_report()` 输出传感器事实、命中日志、带来源的手册原文和证据限制。A 站工单支持“清洗了探头”这一事实，复核结论为“当前证据不足以确定原因”。
