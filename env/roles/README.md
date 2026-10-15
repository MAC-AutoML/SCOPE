# 角色环境

训练、图像生成、GenEval 评分使用三个互不混装的虚拟环境，安装到项目 `.venvs/<role>`。
需要 Bash 和可用的 Python。默认 Python 3.10，用 `SCOPE_PYTHON` 指定已有解释器。
包索引默认是 `https://pypi.org/simple`，用 `SCOPE_PYPI_INDEX` 换成镜像。
缺少 `uv` 时，脚本只在项目 `.bootstrap` 里安装它。

```bash
bash env/roles/setup_env.sh train
bash env/roles/setup_env.sh image
bash env/roles/setup_env.sh geneval
```

同一角色的安装用目录锁串行。锁等待 30 分钟后报错。异常断电留下的锁，要先确认没有安装进程，再手工删掉。
依赖文件摘要没变、并且已经写过 `.ready` 时，脚本直接返回。
安装结束前会执行 `uv pip check`，并导入该角色的核心模块。

训练和图像使用 PyTorch 2.8。GenEval 使用 PyTorch 2.0.1+cu118 和 NumPy 1.26，因为 MMDetection 2.28 需要这套组合。
`mmcv-full` 使用与 CUDA 11.8、PyTorch 2.0、Python 3.10 匹配的预编译轮子，默认索引是
`https://download.openmmlab.com/mmcv/dist/cu118/torch2.0.0/index.html`。
用 `MMCV_WHEEL_INDEX` 指向镜像或本地 wheel 目录。

预编译 MMCV 1.7.2 在部分新 GPU 上没有对应 kernel。`src/geneval_backend.py` 默认把多尺度变形注意力换成 MMCV 自带的 PyTorch 实现，只改当前进程。
GPU 上会对照 CPU 结果做一次矩阵检查。已有兼容的 MMCV CUDA 算子时，设置 `GENEVAL_ATTN_BACKEND=mmcv`。

GenEval 的 OpenCV 需要 `libGL`。`setup_native_libs.sh` 用系统 APT 源下载 `libgl1`、`libglx0`、`libglvnd0`，解压到项目 `.native`，不安装到系统。
单独跑 GenEval 脚本时，在项目根目录设置：

```bash
export LD_LIBRARY_PATH="$PWD/.native/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

完整训练由 `scripts/start.sh` 按节点调用 `setup_env.sh`。当前安装以本目录的三个 requirements 文件为准。
