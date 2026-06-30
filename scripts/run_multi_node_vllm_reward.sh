# export NCCL_DEBUG=INFO
# export NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_IB_DISABLE=1
# export NCCL_SOCKET_IFNAME=eth01

set -e

# ========== Environment variables from the platform ==========
# MASTER_IP=MASTER_IP
# LOCAL_IP=${LOCAL_IP}
# MASTER_PORT=${MASTER_PORT:-29500}
# MACHINE_RANK=$1
# NUM_NODES=4
# GPUS_PER_NODE=8
# TOTAL_PROCS=$((NUM_NODES * GPUS_PER_NODE))

MASTER_IP=${CHIEF_IP}
LOCAL_IP=${LOCAL_IP}
MASTER_PORT=${MASTER_PORT:-29500}
MACHINE_RANK=${INDEX:-0}
NUM_NODES=${HOST_NUM:-4}
GPUS_PER_NODE=${HOST_GPU_NUM:-8}
TOTAL_PROCS=$((NUM_NODES * GPUS_PER_NODE))

# Repository root (absolute path). Override via env or edit here.
ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"

# ========== vLLM Reward Server Assignment ==========
# One dedicated vLLM (VLM reward) server and one LLM (judge) server per node,
# indexed by MACHINE_RANK. Servers must be running before launching training.
# Replace the placeholder host:port entries with your own server addresses,
# one per node (the array length must be >= NUM_NODES).

VLLM_SERVERS=(
    "http://SERVER_0:8000"
    "http://SERVER_1:8000"
    "http://SERVER_2:8000"
    "http://SERVER_3:8000"
)

LLM_SERVERS=(
    "http://SERVER_0:7000"
    "http://SERVER_1:7000"
    "http://SERVER_2:7000"
    "http://SERVER_3:7000"
)


if [ "${MACHINE_RANK}" -ge "${#VLLM_SERVERS[@]}" ]; then
    echo "[ERROR] MACHINE_RANK=${MACHINE_RANK} exceeds number of configured vLLM servers (${#VLLM_SERVERS[@]}). Abort."
    exit 1
fi

if [ "${MACHINE_RANK}" -ge "${#LLM_SERVERS[@]}" ]; then
    echo "[ERROR] MACHINE_RANK=${MACHINE_RANK} exceeds number of configured LLM servers (${#LLM_SERVERS[@]}). Abort."
    exit 1
fi

# export VLLM_BASE_URL="http://SERVER:8000"
# export VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
export LLM_BASE_URL="${LLM_SERVERS[$MACHINE_RANK]}"
export LLM_MODEL="${LLM_MODEL:-Qwen/Qwen3-8B}"
export LLM_API_KEY="${LLM_API_KEY:-EMPTY}"


export VLLM_BASE_URL="${VLLM_SERVERS[$MACHINE_RANK]}"
export VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
export VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"

# ========== NCCL ==========
export NCCL_DEBUG=INFO

# ========== Logging ==========
echo "=== Multi-Node Launch (vLLM Reward) ==="
echo "Master:          ${MASTER_IP}:${MASTER_PORT}"
echo "Num nodes:       ${NUM_NODES}"
echo "GPUs per node:   ${GPUS_PER_NODE}"
echo "Total processes: ${TOTAL_PROCS}"
echo "This node IP:    ${LOCAL_IP}"
echo "This node rank:  ${MACHINE_RANK} / ${NUM_NODES}"
echo "Script dir:      ${ROOT_DIR}"
echo "VLLM server:     ${VLLM_BASE_URL}  (rank ${MACHINE_RANK})"
echo "VLLM model:      ${VLLM_MODEL}"
echo "LLM server:     ${LLM_BASE_URL}  (rank ${MACHINE_RANK})"
echo "LLM model:      ${LLM_MODEL}"

# python ./tools/eval_qwen3.py
# ========== Launch ==========
# Self-Forcing training (default). For LongLive training, comment this block
# and uncomment the LongLive block below.
accelerate launch \
  --config_file ${ROOT_DIR}/scripts/accelerate_files/accelerate_config.yaml \
  --machine_rank ${MACHINE_RANK} \
  --main_process_ip ${MASTER_IP} \
  --main_process_port ${MASTER_PORT} \
  --num_processes ${TOTAL_PROCS} \
  --num_machines ${NUM_NODES} \
  ${ROOT_DIR}/scripts/train_flow_grpo_llm_diffusion_mix_acc.py \
  --config config/self_forcing.py:self_forcing_llm_diffusion_rl_mix

# LongLive training (alternative)
# accelerate launch \
#   --config_file ${ROOT_DIR}/scripts/accelerate_files/accelerate_config.yaml \
#   --machine_rank ${MACHINE_RANK} \
#   --main_process_ip ${MASTER_IP} \
#   --main_process_port ${MASTER_PORT} \
#   --num_processes ${TOTAL_PROCS} \
#   --num_machines ${NUM_NODES} \
#   ${ROOT_DIR}/scripts/train_longlive_flow_grpo_llm_diffusion_mix_acc.py \
#   --config config/longlive.py:longlive_llm_diffusion_rl_mix
