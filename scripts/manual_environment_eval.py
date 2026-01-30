import docker
import time
import requests
import random
from datetime import datetime

# --- Configuration ---
BASE_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
LORA_MODEL_NAME = None # Place the name of your HuggingFace repo with the trained LORA here.
VLLM_IMAGE = "vllm/vllm-openai:latest"
ENV_IMAGE = "diagonalge/openspiel:latest"
NETWORK_NAME = "agent_eval_net"

# Evaluation Params
NUM_EVALS = 500
TEMPERATURE = 0.0
RANDOM_SEED = 42

games_to_task_id_range = {
    "liars_dice": (100000000, 199999999),
    "leduc_poker": (200000000, 299999999),
    "gin_rummy": (300000000, 399999999),
    "othello": (400000000, 499999999),
    "backgammon": (500000000, 599999999),
    "hex": (600000000, 699999999),
    "clobber": (700000000, 799999999),
    "hearts": (800000000, 899999999),
    "euchre": (900000000, 999999999),
    "goofspiel": (0, 99999999)
}

selected_game = "goofspiel"

client = docker.from_env()

def run_random_eval_suite():
    containers = {}
    all_results = []

    try:
        # 1. Infrastructure Setup
        networks = client.networks.list(names=[NETWORK_NAME])
        if not networks: client.networks.create(NETWORK_NAME, driver="bridge")

        if LORA_MODEL_NAME:
            print(f"🚀 Starting vLLM: {BASE_MODEL_NAME} w/ lora {LORA_MODEL_NAME}")
            vllm_command = f"--model {BASE_MODEL_NAME} --enable-lora --lora-modules trained_lora={LORA_MODEL_NAME} --max-lora-rank 64 --port 8000 --trust-remote-code"

        else:
            print(f"🚀 Starting vLLM: {BASE_MODEL_NAME}")
            vllm_command = f"--model {BASE_MODEL_NAME} --port 8000 --trust-remote-code"

        vllm = client.containers.run(
            VLLM_IMAGE,
            command=vllm_command,
            name="vllm-server",
            detach=True,
            network=NETWORK_NAME,
            ports={'8000/tcp': 50000},
            device_requests=[docker.types.DeviceRequest(count=-1, capabilities=[['gpu']])],
        )
        containers['vllm'] = vllm

        print("🚀 Starting AgentGym Server...")
        agent = client.containers.run(
            ENV_IMAGE,
            name="agentgym-server",
            detach=True,
            network=NETWORK_NAME,
            ports={'8000/tcp': 50001}
        )
        containers['agent'] = agent

        # 2. Wait for Readiness
        print("⏳ Waiting for vLLM health check...")
        while True:
            try:
                # vLLM is mapped to 50000 on the host
                if requests.get("http://localhost:50000/v1/models", timeout=2).status_code == 200:
                    break
            except:
                time.sleep(5)
        print("✅ vLLM Ready.\n")

        # 3. Evaluation Loop
        random.seed(RANDOM_SEED)
        eval_list = random.sample(range(games_to_task_id_range[selected_game][0], games_to_task_id_range[selected_game][1]), NUM_EVALS)
        total_score = 0.0
        total_time = 0.0

        if LORA_MODEL_NAME:
            inference_model_name = "trained_lora"
        else:
            inference_model_name = BASE_MODEL_NAME

        for i, task_id in enumerate(eval_list):
            print(f"🔄 [{i+1}/{NUM_EVALS}] Task ID: {task_id}...", end="", flush=True)

            # Prepare the payload for the generic dispatcher
            payload = {
                "model": inference_model_name,
                "base_url": "http://vllm-server:8000/v1",
                "task_id": task_id,
                "temperature": TEMPERATURE,
            }

            try:
                start_ts = time.time()
                # AgentGym is mapped to 50001 on the host
                response = requests.post("http://localhost:50001/evaluate", json=payload, timeout=2500)
                response_data = response.json()

                # The dispatcher wraps the actual result in a 'result' field
                # Format: {"status": "success", "result": {...}}
                result = response_data.get('result', {})

                # Extract metrics from the inner result returned by _run_evaluation
                latency = result.get('time_taken', time.time() - start_ts)
                score = result.get('score', 0.0)

                total_score += score
                total_time += latency

                all_results.append({
                    "task_id": task_id,
                    "task_name": result.get('task_name', 'unknown'),
                    "score": score,
                    "success": response_data.get('status') == "success",
                    "time": latency,
                    "error": response_data.get('detail') if response.status_code != 200 else None
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

    finally:
        print("🧹 Cleaning up containers...")
        for c in containers.values():
            try: c.remove(force=True)
            except: pass

if __name__ == "__main__":
    run_random_eval_suite()