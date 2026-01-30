FROM axolotlai/axolotl:main-py3.11-cu128-2.9.1
COPY --from=ghcr.io/astral-sh/uv:0.9.14 /uv /uvx /bin/

ENV UV_SYSTEM_PYTHON=1 \
    AXOLOTL_DO_NOT_TRACK=1

# Core deps
RUN uv pip install packaging setuptools wheel awscli pydantic \
      mlflow>=2.10.0 wandb>=0.16.0 huggingface_hub aiohttp requests toml fastapi \
      uvicorn httpx loguru python-dotenv scipy numpy datasets \
      tenacity minio pandas tiktoken sentencepiece peft Pillow \
      PyYAML textstat langcheck detoxify protobuf==3.20.3 \
      git+https://github.com/rayonlabs/fiber@2.4.0 \
      trl==0.8.6

RUN uv pip install --no-build-isolation vllm==0.10.2

WORKDIR /workspace/axolotl
RUN mkdir -p /workspace/axolotl/configs \
    /workspace/axolotl/outputs \
    /workspace/axolotl/data \
    /workspace/input_data 

COPY dockerfiles/patches/axolotl_grpo_rollout_fix.py /workspace/axolotl/src/axolotl/core/trainers/grpo/__init__.py
COPY dockerfiles/environment_functions/ /workspace/axolotl/src
COPY core /workspace/core
COPY miner /workspace/miner
COPY trainer /workspace/trainer
COPY scripts /workspace/scripts
COPY core/config/base.yml /workspace/axolotl/base.yml
COPY core/config/base_grpo.yml /workspace/axolotl/base_grpo.yml
COPY core/config/base_environment.yml /workspace/axolotl/base_environment.yml

RUN chmod +x /workspace/scripts/run_text_trainer.sh /workspace/scripts/text_trainer.py

ENTRYPOINT ["/workspace/scripts/run_text_trainer.sh"]