# SCOPE

SCOPE 在冻结的文生图模型上训练一个提示词改写器。改写器是 [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)，渲染器是 [Qwen/Qwen-Image](https://huggingface.co/Qwen/Qwen-Image)，图像对齐分数来自 [GenEval](https://github.com/djghosh13/geneval) 的 Mask2Former 与 OpenAI CLIP。策略优化沿用 TRL 0.24 的 GRPO，信用分配改成按语义片段加权。

改写器把用户提示词写成：

```text
<answer>phrase 1 | phrase 2 | phrase 3</answer>
```

`|` 只是片段边界。送进渲染器之前会去掉标签和 `|`，把片段用空格拼回去。

## 算法流程

一组大小为 `N`（默认 8）。前 `N-1` 条由当前策略 `π_θ` 采样，第 `N` 条是原始提示词，不参与参数更新。同一组共用一个扩散种子。

1. 清洗改写结果，用 Qwen-Image 生成 512×512 图片（20 步，CFG 4.0）。GenEval 给出 0 或 1。
2. 改写的格式奖励是 `+0.25` 或 `-0.25`，检查是否正好有一段可切分的 `<answer>`。原始提示词这条的格式奖励固定为 `+0.25`。
3. 混合奖励 `r = w_t2i * r_image + w_fmt * r_fmt`，两项权重默认都是 1。
4. 混合优势：原始提示词的优势为 0。它的奖励低于阈值 `τ`（默认 1）时，改写的优势是 `clip(r - r_N, -c, c)`。否则用这 `N-1` 条的组内 z-score，再裁到 `[-c, c]`。`c` 默认 3，标准差分母上的 `ε` 默认 `1e-6`。
5. `π_old` 是上一轮优化器更新后的参数拷贝。第一步它和当前策略相同。token 对数比是 `log π_θ - log π_old`。
6. 每个语义片段里对对数比取平均，得到 `clip(exp(β * 平均), 0.8, 1.2)`，再把片段内 token 的权重归一化到均值 1。`β` 默认 1。`<answer>` 标签和 `|` 的权重保持 1。格式不合法、被截断，或某个 token 同时跨过片段和 `|` 时，整条退回全 1 权重。
7. 权重从计算图里断开。PPO 裁剪只作用在概率比上。TRL 0.24 未改配置时 `epsilon` 为 0.2，概率比限制在 0.8 到 1.2。一条 completion 先对 token 求和，再只对 `N-1` 条改写取平均。KL 指向训练开始时冻结的参考模型，系数 `beta=0.01`。`π_old` 是上一轮更新后的滞后拷贝，两者分开保存。

默认学习率 `1e-6`，补全最长 512 token。多卡训练使用 `configs/ds_zero2_bf16.json` 的 DeepSpeed ZeRO-2。

## 代码

| 路径 | 作用 |
|---|---|
| `src/train.py` | `SCOPETrainer`。采样 `N-1` 条改写并插入原始提示词，调用奖励服务，计算混合优势和 span 权重，再做裁剪更新。 |
| `src/scope_credit.py` | 纯 Python 的清洗、片段切分、格式奖励、混合优势、span 权重。不依赖 PyTorch。 |
| `src/reward_client.py` | 训练进程访问奖励服务的 HTTP 客户端。组内种子由 `stable_group_seed` 生成后随请求发送。 |
| `src/reward_server.py` | 奖励协调。按组把清洗后的文本发给图像服务，再把图片发给 GenEval，返回 0/1。 |
| `src/image_server.py` | Qwen-Image 的 FastAPI 服务。启动时从本地目录加载，不在线下载。 |
| `src/geneval_reward_server.py` | Mask2Former 检测加 CLIP 颜色分类，按 GenEval 元数据打 0/1。 |
| `src/geneval_backend.py` | 把 MMCV 多尺度变形注意力换成 PyTorch 实现，供新 GPU 使用。 |
| `src/service_auth.py` | 图像和评分服务共用的 Bearer token。启动器为每次作业生成 token。 |
| `src/model_paths.py` | 把环境变量或 `MODEL_ROOT` 解析成本地模型目录和权重文件。 |
| `src/launch.py` | 按节点拉起环境、图像服务、GenEval、奖励服务和训练，并在作业结束时清理本作业的进程组。 |
| `configs/ds_zero2_bf16.json` | ZeRO-2、bf16。 |
| `eval/geneval/` | GenEval 上游评测代码和 `prompts/evaluation_metadata.jsonl`。检测权重改由 `src/model_paths.py` 解析。 |
| `eval/T2I-CompBench/` | T2I-CompBench / T2I-CompBench++ 上游仓库。提示词在 `examples/dataset/`，评测脚本在 `BLIPvqa_eval/`、`CLIPScore_eval/`、`UniDet_eval/`、`3_in_1_eval/` 和 `MLLM_eval/`。 |
| `data/geneval/train_metadata.jsonl` | 默认训练数据。每行包含 `prompt`、`tag`、`include`，计数类样本另有 `exclude`。 |
| `data/geneval/t2i_compbench_train_metadata.jsonl` | 补充训练记录，字段与上一文件相同，并带 `source`、`source_index`。评分同样走 GenEval。 |
| `data/T2I-CompBench/` | 从该 benchmark 抽出的训练提示词文本，内容对应 `eval/T2I-CompBench/examples/dataset/` 里的 `*_train.txt`。 |
| `scripts/start.sh` | 8 机完整训练的唯一入口。每台机器都执行它。 |

`src/launch.py` 的机器分工是固定的。每台机器需要 8 张可见 GPU。节点 0 和 1 训练，共 16 张 GPU。节点 2 到 6 每张卡一个图像服务。节点 7 每张卡一个 GenEval 服务，并在本机再起一个奖励协调服务。

## 环境复现

当前安装入口是 `env/roles/`。三个角色各自使用一个虚拟环境。

| 文件 | 内容 |
|---|---|
| `env/roles/setup_env.sh` | 用 `uv` 创建 `.venvs/train`、`.venvs/image`、`.venvs/geneval`。 |
| `env/roles/requirements-train.txt` | PyTorch 2.8、transformers 4.57.3、trl 0.24.0、DeepSpeed、PEFT。 |
| `env/roles/requirements-image.txt` | PyTorch 2.8、diffusers 0.36、FastAPI。 |
| `env/roles/requirements-geneval.txt` | PyTorch 2.0.1+cu118、mmcv-full 1.7.2、mmdet 2.28.2、open_clip。 |
| `env/roles/setup_native_libs.sh` | 为 OpenCV 下载 `libGL`，解压到 `.native`。 |
| `env/roles/README.md` | 锁、wheel 索引和注意力后端的说明。 |
| `env/env.example.txt` | 启动 `scripts/start.sh` 时可复制的环境变量。 |

在 Linux x86_64、带 NVIDIA GPU 的机器上，从仓库根目录执行：

```bash
bash env/roles/setup_env.sh train
bash env/roles/setup_env.sh image
bash env/roles/setup_env.sh geneval
```

默认 Python 3.10，包索引是 PyPI。需要换解释器或镜像时：

```bash
export SCOPE_PYTHON=/path/to/python3.10
export SCOPE_PYPI_INDEX=https://pypi.org/simple
```

GenEval 的 `mmcv-full` 从 OpenMMLab 的 cu118 / torch2.0 预编译索引安装。换地址时设置 `MMCV_WHEEL_INDEX`。`requirements-geneval.txt` 里的 PyTorch 轮子指向 `download.pytorch.org`；访问不到时，把这两行改成同一个版本的镜像 URL。

`src/launch.py` 在作业开始时也会调用 `env/roles/setup_env.sh`。依赖没变且 `.ready` 已写好时，安装会直接跳过。

## 模型

权重放在 `MODEL_ROOT` 下，默认是仓库里的 `models/`。训练启动后只读本地文件。按下面的相对路径自行放好：

| 环境变量 | 默认相对路径 | 来源 |
|---|---|---|
| `MODEL_PATH` | `Qwen/Qwen3-8B/`（目录，内含 `config.json`） | Hugging Face `Qwen/Qwen3-8B` |
| `MODEL_NAME` | `Qwen/Qwen-Image/`（目录，内含 `model_index.json`） | Hugging Face `Qwen/Qwen-Image` |
| `GENEVAL_MODEL_PATH` | `mmdet/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco_20220504_001756-743b7d99.pth` | OpenMMLab MMDetection |
| `CLIP_LOCAL_PATH` | `openai/clip-vit-large-patch14/ViT-L-14.pt` | OpenAI CLIP ViT-L/14 |

目录模型也可以放在该路径下唯一的一个版本子目录里。有多个版本时，用对应环境变量指向要使用的那一份。训练进程只读取已经放好的本地文件。`GENEVAL_MODEL_PATH` 和 `CLIP_LOCAL_PATH` 指向权重文件；`MODEL_PATH` 和 `MODEL_NAME` 指向模型目录。

两个 Qwen 目录来自 Hugging Face 的 `Qwen/Qwen3-8B` 和 `Qwen/Qwen-Image`。CLIP 权重是 OpenAI 的 `ViT-L-14.pt`：

```text
https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt
```

Mask2Former 权重用下面的命令下载到上表的路径。配置文件来自已安装的 `mmdet` 包；自动找不到时设置 `GENEVAL_MODEL_CONFIG`。

```bash
bash eval/geneval/evaluation/download_models.sh \
  models/mmdet/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco_20220504_001756-743b7d99.pth
```

放好之后检查解析结果：

```bash
python3 src/model_paths.py
```

## 启动

先把四个模型放到 `MODEL_ROOT`。参数示例见 `env/env.example.txt`。每台机器使用同一个 `SCOPE_JOB_ID` 和同一份 8 个地址的 `SCOPE_NODE_IPS`，并用 `SCOPE_NODE_RANK` 标明自己是第几台（0 到 7）：

```bash
export SCOPE_JOB_ID=SCOPE-run1
export SCOPE_NODE_RANK=0
export SCOPE_NODE_IPS=10.0.0.1,10.0.0.2,10.0.0.3,10.0.0.4,10.0.0.5,10.0.0.6,10.0.0.7,10.0.0.8
export SCOPE_OUTPUT_ROOT=/path/to/outputs
bash scripts/start.sh
```

常用覆盖项：`DATA_PATH`、`NUM_GEN`、`PER_DEVICE_TRAIN_BATCH_SIZE`、`GRAD_ACC_STEPS`、`REWARD_TIMEOUT`、`SCOPE_STARTUP_TIMEOUT`。图像、GenEval、奖励端口默认从 18000、17000、15000 起，分别用 `SCOPE_IMAGE_PORT`、`SCOPE_GENEVAL_PORT`、`SCOPE_REWARD_PORT` 修改。

也可以在奖励服务已经就绪时单独启动训练进程。未设置 `REWARD_ENDPOINTS` 时，它访问 `http://127.0.0.1:5000`。8 机启动器会把它设成节点 7 上的奖励端口，默认是 `15000`。

```bash
.venvs/train/bin/python src/train.py \
  --data data/geneval/train_metadata.jsonl \
  --model /path/to/Qwen3-8B \
  --outdir outputs/checkpoints \
  --num_gen 8
```

这条命令的每卡 batch 使用 `train.py` 的默认值 1。`scripts/start.sh` 使用 `PER_DEVICE_TRAIN_BATCH_SIZE`，默认 8。`--num_gen` 至少为 2。`--lora` 使用 r=8、alpha=16 的 LoRA。`--no_deepspeed` 关闭 ZeRO。`--max_steps` 和 `--max_samples` 用来截断试跑。

## 输出

一次作业写到 `SCOPE_OUTPUT_ROOT/<SCOPE_JOB_ID>/`，未设置时就是 `outputs/<SCOPE_JOB_ID>/`。

- `logs/node-<rank>/`：安装、服务和训练日志
- `generated_images/`：奖励服务保存的图片
- `checkpoints/`：训练输出
- `service.token`：仅本次作业使用，权限 0600，不要提交

服务通过鉴权健康检查后才开始训练。图像生成、评分或存盘失败会使作业失败，并写一个 `FAILED` 标记。
