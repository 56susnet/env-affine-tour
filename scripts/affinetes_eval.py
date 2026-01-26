import asyncio
import docker
import time
import random
import os
import requests
import affinetes as af
from datetime import datetime
from docker.errors import NotFound

# --- Configuration ---
# Model settings
BASE_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
LORA_MODEL_NAME = None # e.g., "your-username/your-repo-name"
VLLM_IMAGE = "vllm/vllm-openai:latest"

# Environment settings (Affinetes)
ENV_PATH = "environments/openspiel" # Ensure this path exists locally
ENV_IMAGE_TAG = "openspiel:v1"

# Evaluation Params
NUM_EVALS = 100
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
    "euchre": (900000000, 999999999)
}

selected_game = "gin_rummy"

# Local Hardware limits
MAX_CONCURRENT_EVALS = 1  # Adjust based on your GPU VRAM. Too high = OOM or Timeouts.
VLLM_PORT = 9000

client = docker.from_env()

async def wait_for_vllm(port):
    """Polls the vLLM server until it provides a 200 OK response."""
    url = f"http://127.0.0.1:{port}/v1/models"
    print(f"⏳ Waiting for vLLM on port {port}...")
    while True:
        try:
            response = await asyncio.to_thread(requests.get, url, timeout=2)
            if response.status_code == 200:
                print("✅ vLLM Ready.")
                return
        except requests.exceptions.RequestException:
            await asyncio.sleep(5)

def save_report(results, base_model, lora_model):
    """Generates the detailed text report matching the original format."""
    total_score = sum(r['score'] for r in results)
    total_time = sum(r['time'] for r in results)
    avg_score = total_score / len(results) if results else 0
    avg_time = total_time / len(results) if results else 0

    safe_model_name = base_model.split("/")[-1]
    if lora_model:
        safe_lora_name = lora_model.split("/")[-1]
        filename = f"eval_results_{safe_model_name}_{safe_lora_name}.txt"
    else:
        filename = f"eval_results_{safe_model_name}.txt"

    with open(filename, "w") as f:
        f.write("="*40 + "\n")
        f.write(f"EVALUATION REPORT - {datetime.now()}\n")
        f.write(f"Model: {base_model}\n")
        if lora_model:
            f.write(f"LoRA: {lora_model}\n")
        f.write("="*40 + "\n\n")
        f.write(f"SUMMARY STATS:\n")
        f.write(f"- Total Tasks: {len(results)}\n")
        f.write(f"- Average Score: {avg_score:.4f}\n")
        f.write(f"- Average Time Per Episode: {avg_time:.2f}s\n\n")
        f.write("DETAILED RESULTS:\n")
        f.write(f"{'Task ID':<10} | {'Score':<7} | {'Time':<7}\n")
        f.write("-" * 60 + "\n")
        for res in results:
            f.write(f"{res['task_id']:<10} | {res['score']:<7} | {res['time']:<7.2f}s\n")
            if 'error' in res and res['error']:
                f.write(f"   └─ Error: {res['error']}\n")
    
    print(f"\n✅ Evaluation complete. Results saved to: {filename}")

async def run_single_eval(env, task_id, model_name, semaphore):
    """Runs a single evaluation guarded by a semaphore."""
    async with semaphore:
        start_ts = time.time()
        try:
            # env.evaluate acts as the client talking to vLLM
            result = await env.evaluate(
                model=model_name,
                base_url=f"http://127.0.0.1:{VLLM_PORT}/v1",
                task_id=task_id,
                seed=42, # Task specific seed if needed, or global
                temperature=TEMPERATURE,
            )
            latency = time.time() - start_ts
            
            # Affinetes return structure normalization
            # Assuming result contains 'score' or is the score itself depending on env implementation
            score = result.get('score', 0.0) if isinstance(result, dict) else 0.0
            
            print(f"✅ Task {task_id} | Score: {score}")
            return {
                "task_id": task_id,
                "score": score,
                "time": latency,
                "raw_result": result
            }
        except Exception as e:
            print(f"❌ Task {task_id} Failed: {e}")
            return {
                "task_id": task_id,
                "score": 0.0,
                "time": time.time() - start_ts,
                "error": str(e)
            }

async def main():
    vllm_container = None
    
    try:
        # 1. Start vLLM Container (Infrastructure)
        print(f"🚀 Initializing vLLM Container...")
        
        # Check if network exists or is needed, though with host binding we might not strictly need it
        # for localhost access, but good for container-to-container if needed later.
        
        vllm_cmd = f"--model {BASE_MODEL_NAME} --port {VLLM_PORT} --trust-remote-code"
        if LORA_MODEL_NAME:
            print(f"   ...with LoRA: {LORA_MODEL_NAME}")
            vllm_cmd = f"--model {BASE_MODEL_NAME} --enable-lora --lora-modules trained_lora={LORA_MODEL_NAME} --port {VLLM_PORT} --trust-remote-code"

        # Check if already running to avoid conflict
        try:
            old_c = client.containers.get("vllm-server")
            old_c.remove(force=True)
        except NotFound:
            pass

        vllm_container = client.containers.run(
            VLLM_IMAGE,
            command=vllm_cmd,
            name="vllm-server",
            network_mode="host",
            detach=True,
            device_requests=[docker.types.DeviceRequest(count=-1, capabilities=[['gpu']])],
            shm_size='4g' 
        )

        # 2. Prepare Affinetes Environment
        print("🛠️  Building/Loading Agent Environment via Affinetes...")
        
        env = af.load_env(
            image="openspiel:v1",
            mode="docker",
            cleanup=False, # We clean up manually or let affinetes handle it
            force_recreate=True,
            host_network=True,
        )

        # 3. Wait for Infrastructure
        await wait_for_vllm(VLLM_PORT)

        # 4. Generate Task List
        random.seed(RANDOM_SEED)
        eval_list = random.sample(range(games_to_task_id_range[selected_game][0], games_to_task_id_range[selected_game][1]), NUM_EVALS)
        
        inference_model_name = "trained_lora" if LORA_MODEL_NAME else BASE_MODEL_NAME

        # 5. Run Evaluation Loop (Parallelized with Semaphore)
        print(f"🔄 Starting {NUM_EVALS} evaluations (Max concurrent: {MAX_CONCURRENT_EVALS})...")
        
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_EVALS)
        tasks = [
            run_single_eval(env, task_id, inference_model_name, semaphore) 
            for task_id in eval_list
        ]
        
        results = await asyncio.gather(*tasks)

        # 6. Save Results
        save_report(results, BASE_MODEL_NAME, LORA_MODEL_NAME)

    finally:
        print("🧹 Cleaning up vLLM container...")
        if vllm_container:
            try:
                vllm_container.remove(force=True)
            except:
                pass
        print("🧹 Cleaning up Env container...")
        await env.cleanup()

if __name__ == "__main__":
    asyncio.run(main())