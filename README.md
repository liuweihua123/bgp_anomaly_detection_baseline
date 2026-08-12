# BGP-Baseline

BGP 路由异常检测与劫持检测基线集合。本仓库汇总了多种具有代表性的研究方法及其复现代码，覆盖实时监控、语义感知检测、图神经网络、AS 表示学习、弱监督学习和多尺度时序建模等方向，便于进行方法复现、横向比较与二次开发。

> [!IMPORTANT]
> 本仓库是多个独立研究工程的集合，而不是一个具有统一入口和统一依赖的 Python 软件包。各工程的数据格式、运行环境和训练流程不同，建议为每个工程分别创建虚拟环境，并先阅读对应说明。

## 工程总览

| 工程 | 主要任务 | 方法/代码形态 | 文档与入口 | 使用前注意 |
| --- | --- | --- | --- | --- |
| [Artemis](Artemis/) | BGP 劫持实时检测与缓解 | ARTEMIS Helm Chart | [部署说明](Artemis/README.md)、[`values.yaml`](Artemis/values.yaml) | 当前目录主要是 Kubernetes/Helm 部署文件，不是完整的 Docker Compose 源码树 |
| [BEAM](BEAM/) | 路由异常检测与告警聚合 | 语义感知的 AS 路径差异建模 | [完整流程](BEAM/readme.md)、[`BEAM_engine/train.py`](BEAM/BEAM_engine/train.py) | 训练依赖 CAIDA AS 关系数据和 CUDA；监控流程需要 RouteViews 数据 |
| [BGPgraph](BGPgraph/) | 异常检测、类型识别与异常 AS 定位 | TIFS 2025；带节点/边属性的 GCN 与自编码器 | [工程说明](BGPgraph/README.md)、[`DetectionLocation/`](BGPgraph/DetectionLocation/) | 部分路径和本地模块需要补齐后才能运行，详见下文 |
| [BGPvector](BGPvector/) | AS 表示学习、AS 关系/类型分类 | 基于 BGP 路径的 BGP2Vec/Word2Vec 嵌入 | [方法与数据说明](BGPvector/BGP2Vec/README.md)、[`bgp2vec.py`](BGPvector/BGP2Vec/bgp2vec.py)、[`train_test.py`](BGPvector/BGP2Vec/train_test.py) | 包含样例数据和预训练模型；依赖版本较旧，建议使用独立环境 |
| [ISP-Operated](ISP-Operated/) | ISP 自运营异常检测 | 滑动窗口、SA-LSTM、弱监督分类 | [`dataset.py`](ISP-Operated/dataset.py) → [`classification.py`](ISP-Operated/classification.py) → [`predict.py`](ISP-Operated/predict.py) | 脚本含原实验环境的绝对路径，需要先改为本地数据、日志和模型路径 |
| [MSLSTM](MSLSTM/) | BGP 异常分类 | 多尺度小波特征与 LSTM | [`feature_extraction.py`](MSLSTM/sendXiaohui/feature_extraction.py) → [`classification.py`](MSLSTM/sendXiaohui/classification.py) → [`test_model.py`](MSLSTM/sendXiaohui/test_model.py) | 脚本含绝对路径，训练默认使用 GPU；运行前需准备 JSON 特征和模型目录 |

## 目录结构

```text
BGP-Baseline/
├── Artemis/          # ARTEMIS 的 Helm 部署配置
├── BEAM/             # 语义感知的路由异常检测流水线
├── BGPgraph/         # 图构建、GCN 检测及异常定位
├── BGPvector/        # BGP2Vec 实现、实验 Notebook、数据与预训练模型
├── ISP-Operated/     # ISP 自运营弱监督检测代码
├── MSLSTM/           # 多尺度 LSTM 特征、训练与测试代码
└── README.md
```

## 开始使用

### 1. 选择工程并创建独立环境

这些工程来自不同年份，PyTorch、PyTorch Geometric、PyTorch Lightning、Gensim 等依赖版本可能互不兼容。不要在同一个环境中一次性安装所有工程的依赖。推荐的基本方式是：

```bash
python -m venv .venv

# Linux/macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

随后进入目标子目录，根据其 README、导入项及实际硬件安装依赖。尤其需要注意：

- `BGPvector/BGP2Vec/requirements.txt` 固定了较旧的完整 Anaconda 环境，其中包含平台相关包；优先在隔离的旧版 Python 环境中按需安装核心依赖。
- `BGPgraph` 的 README 给出的参考版本包括 `networkx==2.8.8`、`torch==1.13.1`、`torch_geometric==2.4.0`、`torch-scatter==2.1.2` 和 `pytorch_lightning==2.1.3`。
- `ISP-Operated` 与 `MSLSTM` 主要依赖 PyTorch、PyTorch Lightning、NumPy 和 scikit-learn；`MSLSTM` 还使用 PyWavelets。
- `Artemis` 应在 Linux/Kubernetes 环境中通过 Helm 部署，具体要求以其[部署说明](Artemis/README.md)为准。

### 2. 准备数据

不同方法使用的数据并不相同，常见来源包括：

- [RouteViews](https://www.routeviews.org/) 的 BGP RIB/UPDATE 数据；
- [CAIDA AS Relationships](https://publicdata.caida.org/datasets/as-relationships/) 的 AS 关系数据；
- 各方法预处理后的 Pickle、NumPy、PyTorch 或 JSON 数据集；
- 用户自行构造的正常/异常事件标签及训练、验证、测试划分。

大型原始数据和多数实验产物未统一存放在仓库中。请先确认目标脚本需要的文件格式，再修改输入、输出和模型路径。

## 各工程运行入口

### Artemis

`Artemis/` 保存的是 ARTEMIS Helm Chart，核心配置入口为 [`values.yaml`](Artemis/values.yaml) 和 [`files/configmaps/config.yaml`](Artemis/files/configmaps/config.yaml)。部署前需要配置密钥、域名、证书以及监控前缀等信息。

本目录未包含 `docker-compose.yml`；若选择 Docker Compose 方式，请按照 ARTEMIS 子目录 README 中链接的上游项目准备完整源码。不要直接在当前 `Artemis/` 目录运行 `docker-compose up`。

### BEAM

BEAM 是当前仓库中流程较完整的检测工程，主要步骤为：

1. `BEAM_engine/train.py`：训练语义感知的 BEAM 嵌入模型；
2. `routing_monitor/detect_route_change_routeviews.py`：从 RouteViews 数据中提取路由变化；
3. `anomaly_detector/BEAM_diff_evaluator_routeviews.py`：计算路径差异；
4. `anomaly_detector/report_anomaly_routeviews.py`：检测并聚合异常；
5. `post_processor/`：生成告警和汇总报告。

训练示例：

```bash
cd BEAM
python BEAM_engine/train.py \
  --serial 2 \
  --time 20240801 \
  --Q 10 \
  --dimension 128 \
  --epoches 1000 \
  --device 0 \
  --num-workers 10
```

其余命令和参数见 [BEAM 使用说明](BEAM/readme.md)。

### BGPvector（BGP2Vec）

该工程的实际实现位于 `BGPvector/BGP2Vec/`，包含 BGP2Vec 模型、AS 关系分类 Notebook、CAIDA AS 关系样例、预处理数据、预训练 Word2Vec 模型和向量查询脚本。此前重复的根目录 `BGP2Vec/` 已移除。

`BGP2VEC` 通过 Python 类调用，没有统一的命令行入口。输入通常为 RouteViews OIX 快照，模型创建与加载逻辑位于 [`bgp2vec.py`](BGPvector/BGP2Vec/bgp2vec.py)。如需验证仓库中的预训练向量，可进入 `BGPvector/BGP2Vec/`，确认 Gensim 版本兼容后运行：

```bash
cd BGPvector/BGP2Vec
python train_test.py
```

子目录 README 还介绍了 AP2Vec 研究，但当前仓库代码主要是 BGP2Vec 训练、AS 嵌入和 AS 关系分类，并不包含完整、独立的 AP2Vec 检测实现。

### BGPgraph

BGPgraph 对应 IEEE Transactions on Information Forensics and Security（TIFS）2025 论文，面向动态 BGP 路由场景中的异常检测、异常类型识别和异常 AS 定位。工程由三个阶段组成：

- `TopoBuild/`：从 BGP 更新增量构建 AS 级拓扑并生成图数据；
- `DetectionLocation/`：训练 GCN 与自编码器，执行异常检测和定位；
- `OnlineDetection/`：面向在线检测的数据加载和推理代码。

当前代码仍包含较多原实验环境绝对路径，并引用了未随目录提供的本地模块，例如 `Edge_conv`、`pyg_test`、`Mydataset3`、`Data_generator_random_sampling` 等。因此该工程目前应视为研究代码快照；在补齐这些模块、数据库配置和数据路径之前，README 中的训练脚本不能直接完整运行。

### ISP-Operated

建议按以下顺序适配：

1. 在 `dataset.py` 中设置事件数据目录并生成滑动窗口数据集；
2. 在 `classification.py` 中设置 `.pt` 数据路径、日志目录、特征维度和类别数；
3. 训练后在 `predict.py` 中设置 checkpoint 与测试集路径，再执行评估。

代码默认面向研究环境，运行前还应检查 `WINDOW_SIZE`、`INPUT_SIZE` 和 checkpoint 中保存的模型参数是否一致。

### MSLSTM

建议按以下顺序适配：

1. `feature_extraction.py`：读取 `fea.json`，清洗数据并生成多尺度小波特征；
2. `classification.py`：加载生成的 `.pt` 数据集并训练分类模型；
3. `test_model.py`：加载 checkpoint 和测试集，输出 FPR、FNR、F1 与准确率。

运行前请把脚本中的 `/home/...` 路径替换为本地路径，并确认模型 checkpoint 与测试数据不是同一个文件。

## 复现实验前检查

- [ ] 已为目标工程创建独立环境，并记录 Python/CUDA/依赖版本；
- [ ] 已准备相应时间段、采集器和格式的 BGP 数据；
- [ ] 已将脚本中的绝对路径改成当前机器上的有效路径；
- [ ] 已核对输入特征数、滑动窗口大小、类别标签和 checkpoint 参数；
- [ ] 已为随机种子、数据划分和评价指标保存配置；
- [ ] 已确认输出目录存在，并避免覆盖已有模型和实验结果；
- [ ] 使用 `BGPgraph` 时，已补齐缺失模块及 MySQL/数据源配置。

## 对应研究

| 工程 | 论文/系统 |
| --- | --- |
| Artemis | Sermpezis et al., *ARTEMIS: Neutralizing BGP Hijacking within a Minute*, IEEE/ACM Transactions on Networking, 2018 |
| BEAM | Chen et al., *Learning with Semantics: Towards a Semantics-Aware Routing Anomaly Detection System*, USENIX Security, 2024 |
| BGPgraph | *GraphBGP: BGP Anomaly Detection and Anomaly Source Localization with Graph Neural Networks*, IEEE Transactions on Information Forensics and Security (TIFS), 2025 |
| BGPvector | Shapira and Shavitt, *BGP2Vec: Unveiling the Latent Characteristics of Autonomous Systems*, IEEE TNSM, 2022 |
| ISP-Operated | Dong et al., *ISP Self-Operated BGP Anomaly Detection Based on Weakly Supervised Learning*, IEEE ICNP, 2021 |
| MSLSTM | Cheng et al., *Multi-scale LSTM Model for BGP Anomaly Classification*, IEEE Transactions on Services Computing, 2021 |

## 说明

本仓库中的实现主要用于学术研究和基线复现。各子工程可能保留其原始作者的代码风格、依赖和许可证；使用、修改或分发前，请分别检查子目录中的许可证与上游项目条款。实验结果应结合所用数据时间范围、采集点、预处理过程和评价协议进行解释。
