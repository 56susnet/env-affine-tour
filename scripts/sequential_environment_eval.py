import docker
import time
import requests
import random
import threading
from datetime import datetime
from huggingface_hub import snapshot_download

# --- Configuration ---
BASE_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
BASE_MODEL_REVISION = None
LORA_MODEL_NAME = None
# LORA_MODEL_NAME = None
LORA_MODEL_REVISION = None
SGLANG_IMAGE = "lmsysorg/sglang:latest"
AGENTGYM_IMAGE = "diagonalge/openspiel:latest"
NETWORK_NAME = "agent_eval_net"
SGLANG_PORT = 30000
HF_CACHE_DIR = "/mnt/hf_cache"

# Evaluation Params
NUM_EVALS = 500
NUM_RUNS = 1
task_id_range = (0, 99999999)
task_id_min, task_id_max = task_id_range
DATA_LEN_RANGE = task_id_max
TEMPERATURE = 0.0
RANDOM_SEED = 42


client = docker.from_env()

def run_random_eval_suite():
    containers = {}
    all_results = []
    avg_score = 0.0

    try:
        # 1. Infrastructure Setup
        networks = client.networks.list(names=[NETWORK_NAME])
        if not networks: client.networks.create(NETWORK_NAME, driver="bridge")

        lora_dir = None
        if LORA_MODEL_NAME:
            print(f"🚀 Starting SGLang: {BASE_MODEL_NAME} w/ lora {LORA_MODEL_NAME}")
            safe_lora_name = LORA_MODEL_NAME.replace("/", "_")
            lora_dir = f"/tmp/sglang_lora/{safe_lora_name}"
            print(f"⬇️  Downloading LoRA to {lora_dir}...")
            snapshot_download(
                repo_id=LORA_MODEL_NAME,
                revision=LORA_MODEL_REVISION,
                local_dir=lora_dir,
                local_dir_use_symlinks=False,
            )
            sglang_command = (
                f"python3 -m sglang.launch_server --model-path {BASE_MODEL_NAME} "
                "--enable-lora --lora-paths trained_lora=/lora/trained_lora "
                "--lora-backend triton "
                "--host 0.0.0.0 --port 30000 --tensor-parallel-size 1 --dtype float16 --enable-deterministic-inference "
                f"--random-seed {RANDOM_SEED}"
            )
        else:
            print(f"🚀 Starting SGLang: {BASE_MODEL_NAME}")
            sglang_command = (
                f"python3 -m sglang.launch_server --model-path {BASE_MODEL_NAME} "
                f"{'--revision ' + BASE_MODEL_REVISION if BASE_MODEL_REVISION else ''} "
                "--host 0.0.0.0 --port 30000 --tensor-parallel-size 1 --dtype float16 --enable-deterministic-inference "
                f"--random-seed {RANDOM_SEED}"
            )

        sglang = client.containers.run(
            SGLANG_IMAGE,
            command=sglang_command,
            name="sglang-server",
            detach=True,
            network=NETWORK_NAME,
            ports={f"{SGLANG_PORT}/tcp": SGLANG_PORT},
            device_requests=[docker.types.DeviceRequest(count=-1, capabilities=[['gpu']])],
            environment={
                "HF_HOME": "/hf",
                "TRANSFORMERS_CACHE": "/hf",
                "HUGGINGFACE_HUB_CACHE": "/hf",
                "HF_HUB_ENABLE_HF_TRANSFER": "1",
                "PYTHONHASHSEED": str(RANDOM_SEED),
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                "NVIDIA_TF32_OVERRIDE": "0",
            },
            volumes={
                HF_CACHE_DIR: {"bind": "/hf", "mode": "rw"},
                **({lora_dir: {"bind": "/lora/trained_lora", "mode": "ro"}} if lora_dir else {}),
            },
            ipc_mode="host",
        )
        containers['sglang'] = sglang

        print("🚀 Starting AgentGym Server...")
        agent = client.containers.run(
            AGENTGYM_IMAGE,
            name="agentgym-server",
            detach=True,
            network=NETWORK_NAME,
            ports={'8000/tcp': 8001} 
        )
        containers['agent'] = agent

        # Stream AgentGym logs to a file
        agent_log_path = "agentgym-server.log"
        def _stream_agent_logs():
            with open(agent_log_path, "ab") as log_file:
                for line in agent.logs(stream=True, follow=True):
                    log_file.write(line)
                    log_file.flush()

        threading.Thread(target=_stream_agent_logs, daemon=True).start()
        print(f"📄 AgentGym logs streaming to {agent_log_path}")

        # 2. Wait for Readiness
        print("⏳ Waiting for SGLang health check...")
        while True:
            try:
                if requests.get(f"http://localhost:{SGLANG_PORT}/v1/models", timeout=2).status_code == 200:
                    break
            except:
                time.sleep(5)
        print("✅ SGLang Ready.\n")

        # 3. Evaluation Loop
        random.seed(RANDOM_SEED)
        eval_list = random.sample(range(1, DATA_LEN_RANGE + 1), NUM_EVALS)
        total_score = 0.0
        total_time = 0.0

        if LORA_MODEL_NAME:
            # For OpenAI-compatible API, use base-model:adapter-name format per SGLang docs
            # Format: model_path:adapter_name (e.g., "Qwen/Qwen2.5-3B-Instruct:trained_lora")
            inference_model_name = f"{BASE_MODEL_NAME}:trained_lora"
        else:
            inference_model_name = BASE_MODEL_NAME

        for i, task_id in enumerate(eval_list):
            print(f"🔄 [{i+1}/{NUM_EVALS}] Task ID: {task_id}...", end="", flush=True)

            payload = {
                "model": inference_model_name,
                "base_url": f"http://sglang-server:{SGLANG_PORT}/v1",
                "task_id": task_id,
                "temperature": TEMPERATURE,
                "seed": RANDOM_SEED,
                "opponent": "mcts",
                "api_key": "test"
            }

            try:
                start_ts = time.time()
                response = requests.post("http://localhost:8001/evaluate", json=payload, timeout=2500)
                result = response.json()

                # AgentGym wraps actual payload under "result" in MethodResponse
                result_payload = result.get("result") if isinstance(result, dict) else None
                if isinstance(result_payload, dict):
                    data = result_payload
                else:
                    data = result if isinstance(result, dict) else {}

                latency = data.get('time_taken', time.time() - start_ts)
                score = data.get('score', 0.0)

                total_score += score
                total_time += latency

                all_results.append({
                    "task_id": task_id,
                    "task_name": data.get('task_name', 'unknown'),
                    "score": score,
                    "success": data.get('success', False),
                    "time": latency,
                    "error": data.get('error')
                })
                print(f" Done (Score: {score})")
            except Exception as e:
                print(f" Failed: {e}")

        # 4. Final Aggregation & File Writing
        avg_score = total_score / len(all_results) if all_results else 0
        avg_time = total_time / len(all_results) if all_results else 0


        safe_model_name = BASE_MODEL_NAME.split("/")[1]

        if LORA_MODEL_NAME:
            safe_lora_name = LORA_MODEL_NAME.split("/")[1]
            filename = f"eval_results_{safe_model_name}_{safe_lora_name}.txt"
        else:
            filename = f"eval_results_{safe_model_name}.txt"

        with open(filename, "w") as f:
            f.write("="*40 + "\n")
            f.write(f"EVALUATION REPORT - {datetime.now()}\n")
            f.write(f"Model: {BASE_MODEL_NAME}\n")
            f.write("="*40 + "\n\n")
            f.write(f"SUMMARY STATS:\n")
            f.write(f"- Total Tasks: {len(all_results)}\n")
            f.write(f"- Average Score: {avg_score:.4f}\n")
            f.write(f"- Average Time Per Episode: {avg_time:.2f}s\n\n")
            f.write("DETAILED RESULTS:\n")
            f.write(f"{'Task ID':<10} | {'Name':<15} | {'Score':<7} | {'Success':<8} | {'Time':<7}\n")
            f.write("-" * 60 + "\n")
            for res in all_results:
                f.write(f"{res['task_id']:<10} | {res['task_name']:<15} | {res['score']:<7} | {str(res['success']):<8} | {res['time']:<7.2f}s\n")
                if res['error']:
                    f.write(f"   └─ Error: {res['error']}\n")

        print(f"\n✅ Evaluation complete. Results saved to: {filename}")
        print(f"Average score: {avg_score:.4f}")
        return avg_score

    finally:
        print("🧹 Cleaning up containers...")
        for c in containers.values():
            try: c.remove(force=True)
            except: pass

    return avg_score


def run_multiple_evals():
    scores = []
    for run_idx in range(1, NUM_RUNS + 1):
        print(f"\n=== Run {run_idx}/{NUM_RUNS} ===")
        score = run_random_eval_suite()
        scores.append(score)
    print("\n✅ All runs complete. Run scores:")
    for idx, score in enumerate(scores, start=1):
        print(f"Run {idx}: {score:.4f}")

if __name__ == "__main__":
    run_multiple_evals()
