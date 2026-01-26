#!/bin/bash

# Configuration
TASK_ID="1"
MODEL="Qwen/Qwen2.5-3B-Instruct"
DATASET="https://huggingface.co/datasets/TuringEnterprises/Turing-Open-Reasoning/resolve/main/Computational_STEM_QA_Dataset.json?download=true"
DATASET_TYPE='{"environment_name": "game"}'
FILE_FORMAT="s3"
HOURS_TO_COMPLETE=12
EXPECTED_REPO_NAME="environment_test"

# Directory Setup
CHECKPOINTS_DIR="$(pwd)/secure_checkpoints"
OUTPUTS_DIR="$(pwd)/outputs"
mkdir -p "$CHECKPOINTS_DIR" "$OUTPUTS_DIR"
chmod 777 "$CHECKPOINTS_DIR" "$OUTPUTS_DIR"

# 1. Create a dedicated network so containers can see each other by name
NETWORK_NAME="trainer-net"
docker network inspect $NETWORK_NAME >/dev/null 2>&1 || docker network create $NETWORK_NAME

# 2. Build images
docker build -t trainer-downloader -f dockerfiles/trainer-downloader.dockerfile .
docker build -t standalone-text-trainer -f dockerfiles/standalone-text-trainer.dockerfile .

# 3. Download model and dataset
echo "Downloading model and dataset..."
docker run --rm \
  --network $NETWORK_NAME \
  --volume "$CHECKPOINTS_DIR:/cache:rw" \
  --name downloader-image \
  trainer-downloader \
  --task-id "$TASK_ID" \
  --model "$MODEL" \
  --dataset "$DATASET" \
  --file-format "$FILE_FORMAT" \
  --task-type "EnvTask"

# 4. Start the Environment Server in the background (-d)
# We name it 'env-server' so the trainer can find it at that hostname
echo "Starting environment server..."
docker run -d --rm \
  --name env-server \
  --network $NETWORK_NAME \
  --security-opt=no-new-privileges \
  openspiel:v1

# 5. Run the Trainer
# ENVIRONMENT_SERVER_URLS points to the container name 'env-server'
echo "Starting trainer..."
docker run --rm --gpus all \
  --network $NETWORK_NAME \
  --security-opt=no-new-privileges \
  --cap-drop=ALL \
  --volume "$CHECKPOINTS_DIR:/cache:rw" \
  --volume "$OUTPUTS_DIR:/app/checkpoints/:rw" \
  --env ENVIRONMENT_SERVER_URLS="http://env-server:8000" \
  --name grpo-text-trainer-example \
  standalone-text-trainer \
  --task-id "$TASK_ID" \
  --model "$MODEL" \
  --dataset "$DATASET" \
  --dataset-type "$DATASET_TYPE" \
  --task-type "EnvTask" \
  --file-format "$FILE_FORMAT" \
  --hours-to-complete "$HOURS_TO_COMPLETE" \
  --expected-repo-name "$EXPECTED_REPO_NAME"

# 6. Cleanup: Stop the background server when finished
docker stop env-server