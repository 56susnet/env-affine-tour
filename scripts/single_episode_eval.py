import asyncio
import docker
import time
import requests
import random
from datetime import datetime
import affinetes as af


async def main():
    # --- Configuration ---
    BASE_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
    LORA_MODEL_NAME = None # Place the name of your HuggingFace repo with the trained LORA here.

    # Evaluation Params
    GAME_TO_EVAL = "gin_rummy"
    NUM_EVALS = 500
    TEMPERATURE = 0.0
    TASK_TO_EVAL = 300000000
    RANDOM_SEED = 42


    ENV_SERVER_IMAGE = "openspiel:v1"

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


    client = docker.from_env()
    containers = {}

    if LORA_MODEL_NAME:
        print(f"🚀 Starting vLLM: {BASE_MODEL_NAME} w/ lora {LORA_MODEL_NAME}")
        vllm_command = f"--model {BASE_MODEL_NAME} --enable-lora --lora-modules trained_lora={LORA_MODEL_NAME} --port 9000 --trust-remote-code"

    else:
        print(f"🚀 Starting vLLM: {BASE_MODEL_NAME}")
        vllm_command = f"--model {BASE_MODEL_NAME} --port 9000 --trust-remote-code"

    vllm = client.containers.run(
        "vllm/vllm-openai:latest",
        command=vllm_command,
        name="vllm-server",
        detach=True,
        network_mode="host",
        device_requests=[docker.types.DeviceRequest(count=-1, capabilities=[['gpu']])],
    )
    containers['vllm'] = vllm



    print("🚀 Starting Affine GAME Server...")
    env = af.load_env(
        image="openspiel:v1",
        mode="docker",
        cleanup=False, # We clean up manually or let affinetes handle it
        force_recreate=True,
        host_network=True,
    )


    print("⏳ Waiting for vLLM health check...")
    while True:
        try:
            if requests.get("http://localhost:9000/v1/models", timeout=2).status_code == 200:
                break
        except:
            time.sleep(5)
    print("✅ vLLM Ready.\n")



    if LORA_MODEL_NAME:
        inference_model_name = "trained_lora"
    else:
        inference_model_name = BASE_MODEL_NAME

    result = await env.evaluate(
        model=inference_model_name,
        base_url=f"http://localhost:9000/v1",
        task_id=TASK_TO_EVAL,
        seed=42, # Task specific seed if needed, or global
        temperature=TEMPERATURE,
    )

    print(f" Done (Result: {result})")

    print("🧹 Cleaning up containers...")
    for c in containers.values():
        try: c.remove(force=True)
        except: pass


if __name__ == "__main__":
    asyncio.run(main())
