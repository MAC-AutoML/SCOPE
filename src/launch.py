"""Supervise only this pipeline job's process groups."""
import json
import os
from pathlib import Path
import re
import secrets
import signal
import subprocess
import time
import urllib.request

from model_paths import MODELS, resolve_model

ROOT = Path(__file__).resolve().parent.parent


def cleanup_processes(children, grace=10):
    for process in children:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + grace
    while any(p.poll() is None for p in children) and time.monotonic() < deadline:
        time.sleep(0.2)
    for process in children:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def main():
    job = os.environ.get("SCOPE_JOB_ID", "")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", job):
        raise ValueError("Set SCOPE_JOB_ID to a name shared by every node; use letters, digits, _ and -")
    rank = int(os.environ.get("SCOPE_NODE_RANK", "0"))
    ips = [item for item in re.split(r"[,;\s]+", os.environ.get("SCOPE_NODE_IPS", "").strip()) if item]
    if len(ips) != 8 or rank not in range(8):
        raise ValueError("Set SCOPE_NODE_IPS to exactly eight addresses and SCOPE_NODE_RANK in 0..7")
    run = Path(os.environ.get("SCOPE_OUTPUT_ROOT", str(ROOT / "outputs"))).expanduser() / job
    run.mkdir(parents=True, exist_ok=True)
    logs = run / "logs" / f"node-{rank}"
    logs.mkdir(parents=True, exist_ok=True)
    children, streams = [], []

    def interrupted(signum, frame):
        raise RuntimeError(f"Job interrupted by signal {signum}")

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, interrupted)

    def start(command, name, env):
        stream = open(logs / f"{name}.log", "a", buffering=1)
        streams.append(stream)
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        children.append(process)
        return process

    def check():
        if (run / "FAILED").exists():
            raise RuntimeError("Another worker failed; see per-node logs")
        for process in children:
            if process.poll() is not None:
                raise RuntimeError(f"Child {process.pid} exited unexpectedly ({process.returncode})")

    def ready(urls):
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        deadline = time.monotonic() + int(os.environ.get("SCOPE_STARTUP_TIMEOUT", "1800"))
        pending = set(urls)
        while pending:
            check()
            for url in list(pending):
                try:
                    request = urllib.request.Request(url, headers={"Authorization": "Bearer " + token})
                    with opener.open(request, timeout=2) as response:
                        if json.load(response).get("ok"):
                            pending.remove(url)
                except (OSError, ValueError):
                    pass
            if pending and time.monotonic() > deadline:
                raise TimeoutError(f"Services not ready: {sorted(pending)}")
            if pending:
                time.sleep(2)

    try:
        gpus = subprocess.check_output(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                                       text=True, timeout=15).strip().splitlines()
        if len(gpus) < 8:
            raise RuntimeError("Insufficient visible GPUs for this job")
        token_path = run / "service.token"
        if rank == 0:
            with open(token_path, "x", opener=lambda path, flags: os.open(path, flags, 0o600)) as stream:
                stream.write(secrets.token_hex(32))
        deadline = time.monotonic() + 1800
        while not token_path.is_file() or token_path.stat().st_size < 64:
            check()
            if time.monotonic() > deadline:
                raise TimeoutError("Waiting for job token")
            time.sleep(1)
        token = token_path.read_text().strip()
        env = dict(os.environ, SCOPE_SERVICE_TOKEN=token, WANDB_MODE="disabled", HF_HUB_OFFLINE="1",
                   SAVE_DIR=str(run / "generated_images"))
        env["LD_LIBRARY_PATH"] = str(ROOT / ".native/usr/lib/x86_64-linux-gnu") + (
            ":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
        for name, value in {"NCCL_IB_DISABLE": "0", "NCCL_SOCKET_IFNAME": "eth0", "GLOO_SOCKET_IFNAME": "eth0",
                            "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1"}.items():
            env.setdefault(name, value)
        env.update({name: resolve_model(name) for name in MODELS})
        roles = ["train" if rank < 2 else "image" if rank < 7 else "geneval"]
        for role in roles:
            installer = start(["bash", str(ROOT / "env/roles/setup_env.sh"), role], f"install-{role}", env)
            rc = installer.wait(timeout=3600)
            if rc:
                raise RuntimeError(f"{role} environment failed ({rc})")
            children.remove(installer)

        def python(role):
            return str(ROOT / ".venvs" / role / "bin/python")

        image_port = int(env.get("SCOPE_IMAGE_PORT", "18000"))
        geneval_port = int(env.get("SCOPE_GENEVAL_PORT", "17000"))
        reward_port = int(env.get("SCOPE_REWARD_PORT", "15000"))
        reward_ip = ips[7]
        image_hosts = ips[2:7]
        image_urls = [f"http://{host}:{image_port + gpu}" for host in image_hosts for gpu in range(8)]
        env["REWARD_ENDPOINTS"] = f"http://{reward_ip}:{reward_port}"
        if 2 <= rank <= 6:
            for gpu in range(8):
                start([python("image"), "-m", "uvicorn", "image_server:app", "--app-dir", str(ROOT / "src"),
                       "--host", "0.0.0.0", "--port", str(image_port + gpu)], f"image-{gpu}",
                      dict(env, CUDA_VISIBLE_DEVICES=str(gpu)))
        if rank == 7:
            for gpu in range(8):
                start([python("geneval"), str(ROOT / "src/geneval_reward_server.py")], f"geneval-{gpu}",
                      dict(env, CUDA_VISIBLE_DEVICES=str(gpu), GENEVAL_PORT=str(geneval_port + gpu)))
            ready([f"http://127.0.0.1:{geneval_port + gpu}/health" for gpu in range(8)])
            start([python("geneval"), str(ROOT / "src/reward_server.py")], "reward",
                  dict(env, IMG_URLS=",".join(url + "/generate_b64_batch" for url in image_urls),
                       GENEVAL_URL=",".join(f"http://127.0.0.1:{geneval_port + gpu}/score" for gpu in range(8)),
                       REWARD_PORT=str(reward_port)))
        if rank < 2:
            ready([url + "/health" for url in image_urls] + [env["REWARD_ENDPOINTS"] + "/health"])
            command = [python("train"), "-m", "torch.distributed.run", "--nnodes=2", "--nproc_per_node=8",
                       f"--node_rank={rank}", f"--master_addr={ips[0]}",
                       f"--master_port={env.get('MASTER_PORT', '29500')}",
                       str(ROOT / "src/train.py"),
                       "--data", env.get("DATA_PATH", str(ROOT / "data/geneval/train_metadata.jsonl")),
                       "--model", env["MODEL_PATH"], "--outdir", str(run / "checkpoints"),
                       "--num_gen", env.get("NUM_GEN", "8"),
                       "--per_device_train_batch_size", env.get("PER_DEVICE_TRAIN_BATCH_SIZE", "8"),
                       "--gradient_accumulation_steps", env.get("GRAD_ACC_STEPS", "1")]
            train = start(command, "train", env)
            while train.poll() is None:
                for process in children:
                    if process is not train and process.poll() is not None:
                        raise RuntimeError("Service exited during training")
                if (run / "FAILED").exists():
                    raise RuntimeError("Another worker failed")
                time.sleep(2)
            if train.returncode:
                raise RuntimeError(f"Training failed ({train.returncode})")
            children.remove(train)
            (run / f"train-{rank}.done").touch()
        while not all((run / f"train-{n}.done").exists() for n in range(2)):
            check()
            time.sleep(2)
    except BaseException:
        (run / "FAILED").touch()
        raise
    finally:
        cleanup_processes(children)
        for stream in streams:
            stream.close()


if __name__ == "__main__":
    main()
