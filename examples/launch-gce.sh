#!/usr/bin/env bash
#
# HighPerfASR — one-command GCE evaluation server.
#
# Usage:
#   ./launch-gce.sh                                   # Deploy with current gcloud auth
#   ./launch-gce.sh --key service-account.json        # Deploy with service account
#   ./launch-gce.sh --mode batch --gpu t4             # Options
#   ./launch-gce.sh teardown                          # Delete evaluation VM + firewall
#
# Prerequisites: gcloud CLI (authenticated), GPU quota in target zone.
# Runs in Google Cloud Shell or any terminal with gcloud installed.

set -euo pipefail

VERSION="0.3.0"
MODE="stream"
ZONE="us-central1-a"
ZONE_EXPLICIT=false
GPU="l4"
TTL_HOURS=4
KEY_FILE=""
ACTION="deploy"

L4_FALLBACK_ZONES=("us-central1-a" "us-east1-b" "us-west1-b" "us-east4-c" "europe-west4-b")
T4_FALLBACK_ZONES=("us-central1-a" "us-east1-b" "us-west1-b" "europe-west4-b" "asia-east1-b")

VM_PREFIX="hpfasr-eval"
FW_RULE="allow-highperfasr-eval"
LABEL_APP="highperfasr"
LABEL_CREATOR="launch-gce"

usage() {
  local exit_code="${1:-0}"
  cat <<EOF
Deploy a HighPerfASR evaluation server on Google Cloud.

Usage: $(basename "$0") [OPTIONS] [teardown]

Options:
  --key FILE        GCP service account JSON key file
  --zone ZONE       GCE zone (default: auto-selects from US zones)
  --mode MODE       batch or stream (default: stream)
  --gpu GPU         l4 (default) or t4
  --version VER     GHCR image version (default: 0.3.0)
  --ttl HOURS       Auto-shutdown after N hours (default: 4)
  -h, --help        Show this help

Examples:
  $(basename "$0")                                  # Stream on L4, current gcloud auth
  $(basename "$0") --key sa.json                    # Stream on L4, service account
  $(basename "$0") --mode batch --gpu t4            # Batch on T4
  $(basename "$0") teardown                         # Delete evaluation VM

Required IAM permissions:
  compute.instances.create, compute.instances.delete,
  compute.instances.get, compute.instances.list,
  compute.regions.get, compute.zones.get,
  compute.firewalls.create, compute.firewalls.delete,
  compute.firewalls.get, serviceusage.services.list
  Suggested roles: compute.instanceAdmin.v1, compute.securityAdmin,
  serviceusage.serviceUsageViewer
EOF
  exit "$exit_code"
}

require_value() {
  local option="$1"
  local value="${2-}"

  if [[ -z "$value" || "$value" == --* ]]; then
    echo "ERROR: $option requires a value." >&2
    echo "Run '$(basename "$0") --help' for usage." >&2
    exit 1
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --key)         require_value "$1" "${2-}"; KEY_FILE="$2"; shift 2 ;;
    --zone)        require_value "$1" "${2-}"; ZONE="$2"; ZONE_EXPLICIT=true; shift 2 ;;
    --mode)        require_value "$1" "${2-}"; MODE="$2"; shift 2 ;;
    --gpu)         require_value "$1" "${2-}"; GPU="$2"; shift 2 ;;
    --version)     require_value "$1" "${2-}"; VERSION="$2"; shift 2 ;;
    --ttl)         require_value "$1" "${2-}"; TTL_HOURS="$2"; shift 2 ;;
    --no-public-ip)
      echo "ERROR: --no-public-ip is not supported by this one-click evaluator." >&2
      echo "Private VMs need project-specific Cloud NAT and IAP firewall setup for Docker/GHCR egress." >&2
      exit 1
      ;;
    teardown)      ACTION="teardown"; shift ;;
    -h|--help)     usage 0 ;;
    *)
      echo "ERROR: unknown option: $1" >&2
      echo "Run '$(basename "$0") --help' for usage." >&2
      exit 1
      ;;
  esac
done

validate_inputs() {
  case "$MODE" in
    batch|stream) ;;
    *) echo "ERROR: unsupported mode '$MODE' (use batch or stream)" >&2; exit 1 ;;
  esac

  if [[ ! "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+(-[0-9A-Za-z][0-9A-Za-z.-]*)?$ ]]; then
    echo "ERROR: unsupported image version '$VERSION'." >&2
    echo "Use a pinned GHCR semver Docker tag like 0.3.0 or 0.3.0-rc.1." >&2
    exit 1
  fi

  if [[ ! "$TTL_HOURS" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: --ttl must be a positive integer number of hours." >&2
    exit 1
  fi
}

# --- Resolve GPU config ---
case "$GPU" in
  l4)  MACHINE="g2-standard-4"; ACCEL_FLAGS=(); GPU_QUOTA_METRIC="NVIDIA_L4_GPUS" ;;
  t4)  MACHINE="n1-standard-4"; ACCEL_FLAGS=("--accelerator=type=nvidia-tesla-t4,count=1"); GPU_QUOTA_METRIC="NVIDIA_T4_GPUS" ;;
  *)   echo "ERROR: unsupported GPU '$GPU' (use l4 or t4)" >&2; exit 1 ;;
esac

check_gpu_quota() {
  local region quota_check quota_limit quota_usage quota_available
  region="${ZONE%-*}"

  if ! quota_check=$(gcloud compute regions describe "$region" \
    --project="$PROJECT" \
    --format=json 2>/dev/null | python3 -c '
import json
import sys

metric = sys.argv[1]
region = json.load(sys.stdin)
for quota in region.get("quotas", []):
    if quota.get("metric") == metric:
        limit = float(quota.get("limit", 0))
        usage = float(quota.get("usage", 0))
        print(f"{limit:g} {usage:g} {limit - usage:g}")
        sys.exit(0)
sys.exit(2)
' "$GPU_QUOTA_METRIC"); then
    echo "    WARNING: cannot read $GPU GPU quota ($GPU_QUOTA_METRIC) in region $region." >&2
    echo "    (will attempt deployment anyway — zone fallback handles stockouts)" >&2
    return 0
  fi

  read -r quota_limit quota_usage quota_available <<< "$quota_check"
  if ! python3 -c 'import sys; sys.exit(0 if float(sys.argv[1]) >= 1 else 1)' "$quota_available"; then
    echo "    WARNING: $GPU GPU quota may be insufficient in $region (${quota_usage}/${quota_limit} used)." >&2
    echo "    Will attempt deployment — zone fallback may find capacity in another region." >&2
    echo "    Request more quota at https://console.cloud.google.com/iam-admin/quotas?project=$PROJECT" >&2
    return 0
  fi

  echo "    GPU quota: ${quota_usage}/${quota_limit} $GPU_QUOTA_METRIC used (${quota_available} available)"
}

# --- Preflight checks ---
preflight() {
  echo "==> Preflight checks"

  if ! command -v gcloud &>/dev/null; then
    echo "ERROR: gcloud CLI not found. Install: https://cloud.google.com/sdk/install" >&2
    exit 1
  fi

  if [[ -n "$KEY_FILE" ]]; then
    if [[ ! -f "$KEY_FILE" ]]; then
      echo "ERROR: key file not found: $KEY_FILE" >&2
      exit 1
    fi
    PROJECT=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["project_id"])' "$KEY_FILE" 2>/dev/null) \
      || PROJECT=$(jq -r .project_id "$KEY_FILE" 2>/dev/null) \
      || true
    if [[ -z "$PROJECT" || "$PROJECT" == "null" ]]; then
      echo "ERROR: cannot read project_id from $KEY_FILE" >&2
      echo "  The file must be a valid GCP service account JSON key." >&2
      exit 1
    fi
    if ! gcloud auth activate-service-account --key-file="$KEY_FILE" --project="$PROJECT"; then
      echo "" >&2
      echo "ERROR: failed to authenticate with service account key." >&2
      echo "  Check that the key is valid and the service account is not disabled." >&2
      echo "  Key file: $KEY_FILE" >&2
      echo "  Project:  $PROJECT" >&2
      exit 1
    fi
    echo "    Auth: service account from $KEY_FILE"
  else
    ACCOUNT=$(gcloud auth list --filter=status:ACTIVE --format='value(account)' 2>/dev/null | head -1)
    if [[ -z "$ACCOUNT" ]]; then
      echo "ERROR: no active gcloud auth. Run: gcloud auth login" >&2
      exit 1
    fi
    PROJECT=$(gcloud config get-value project 2>/dev/null)
    if [[ -z "$PROJECT" || "$PROJECT" == "(unset)" ]]; then
      echo "ERROR: no project set. Run: gcloud config set project PROJECT_ID" >&2
      exit 1
    fi
    echo "    Auth: $ACCOUNT"
  fi

  BILLING_INFO=$(gcloud billing projects describe "$PROJECT" --format='value(billingEnabled)' 2>/dev/null) || BILLING_INFO="UNKNOWN"
  if [[ "$BILLING_INFO" == "False" ]]; then
    echo "ERROR: billing is not enabled for project $PROJECT." >&2
    echo "  GPU VMs require an active billing account." >&2
    echo "  Link billing at: https://console.cloud.google.com/billing/linkedaccount?project=$PROJECT" >&2
    exit 1
  elif [[ "$BILLING_INFO" == "UNKNOWN" ]]; then
    echo "    Billing: could not verify (missing cloudbilling.projects.get permission)"
    echo "             VM creation will fail if billing is not enabled."
  else
    echo "    Billing: enabled"
  fi

  SVC_OUTPUT=$(gcloud services list --enabled --project="$PROJECT" --format='value(name)' 2>&1) || {
    echo "ERROR: cannot list enabled APIs for project $PROJECT." >&2
    echo "  This usually means the account lacks serviceusage.services.list permission." >&2
    echo "  Ensure the account has roles/serviceusage.serviceUsageViewer on the project." >&2
    exit 1
  }
  if ! echo "$SVC_OUTPUT" | grep -q compute.googleapis.com; then
    echo "ERROR: Compute Engine API not enabled. Run:" >&2
    echo "  gcloud services enable compute.googleapis.com --project=$PROJECT" >&2
    echo "  (If this fails, check that billing is enabled for the project)" >&2
    exit 1
  fi

  if ! gcloud compute zones describe "$ZONE" --project="$PROJECT" &>/dev/null; then
    echo "ERROR: zone '$ZONE' not found in project $PROJECT" >&2
    exit 1
  fi

  if [[ "$ACTION" == "deploy" ]]; then
    check_gpu_quota
  fi

  echo "    Project: $PROJECT"
  echo "    Zone:    $ZONE"
  echo "    GPU:     $GPU ($MACHINE)"
  echo "    Mode:    $MODE"
  echo "    Version: $VERSION"
  echo "    TTL:     ${TTL_HOURS}h"
  echo ""
}

# --- Find existing eval VMs ---
find_vms() {
  gcloud compute instances list \
    --project="$PROJECT" \
    --filter="labels.app=$LABEL_APP AND (labels.created-by=$LABEL_CREATOR OR labels.created-by=launch-gce-web)" \
    --format="value(name,zone.basename(),networkInterfaces[0].accessConfigs[0].natIP,status)" \
    2>/dev/null
}

# --- Teardown ---
do_teardown() {
  echo "==> Looking for evaluation VMs..."
  VMS=$(find_vms)
  if [[ -z "$VMS" ]]; then
    echo "    No evaluation VMs found."
  else
    while IFS=$'\t' read -r name zone _ip status; do
      echo "    Deleting $name ($zone, $status)..."
      gcloud compute instances delete "$name" \
        --project="$PROJECT" --zone="$zone" --quiet 2>/dev/null || true
    done <<< "$VMS"
  fi

  echo "==> Checking firewall rule..."
  if gcloud compute firewall-rules describe "$FW_RULE" --project="$PROJECT" &>/dev/null; then
    echo "    Deleting firewall rule $FW_RULE..."
    gcloud compute firewall-rules delete "$FW_RULE" \
      --project="$PROJECT" --quiet 2>/dev/null || true
  else
    echo "    No firewall rule found."
  fi

  echo "==> Teardown complete."
}

# --- Deploy ---
do_deploy() {
  VM_NAME="${VM_PREFIX}-$(date +%s | tail -c 7)"

  if [[ "$MODE" == "batch" ]]; then
    IMAGE="ghcr.io/beastoin/highperfasr-batch:${VERSION}"
    PORT=8000
    CONTAINER_NAME="highperfasr-batch"
  else
    IMAGE="ghcr.io/beastoin/highperfasr-stream:${VERSION}"
    PORT=8001
    CONTAINER_NAME="highperfasr-stream"
  fi

  TTL_MINUTES=$((TTL_HOURS * 60))
  STARTUP_IMAGE=$(printf '%q' "$IMAGE")
  STARTUP_CONTAINER_NAME=$(printf '%q' "$CONTAINER_NAME")

  STARTUP_SCRIPT=$(cat <<SCRIPT
#!/bin/bash
set -ex

IMAGE=${STARTUP_IMAGE}
CONTAINER_NAME=${STARTUP_CONTAINER_NAME}
PORT=${PORT}

shutdown -h +${TTL_MINUTES} "Auto-shutdown: evaluation TTL expired" &

if ! command -v docker &>/dev/null; then
  curl -fsSL https://get.docker.com | sh
fi

if ! command -v nvidia-ctk &>/dev/null; then
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \\
    gpg --batch --yes --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  ARCH=\$(dpkg --print-architecture)
  echo "deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://nvidia.github.io/libnvidia-container/stable/deb/\$ARCH /" | \\
    tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
  apt-get update && apt-get install -y nvidia-container-toolkit
fi
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker

docker pull "\$IMAGE"
docker run -d --gpus all -p "\${PORT}:8000" \\
  -v hf-cache:/app/.cache/huggingface \\
  -e NVIDIA_VISIBLE_DEVICES=all \\
  -e HF_HOME=/app/.cache/huggingface \\
  -e NUMBA_CACHE_DIR=/tmp/numba_cache \\
  -e HPFASR_CORS_ORIGINS='*' \\
  --name "\$CONTAINER_NAME" \\
  --restart unless-stopped \\
  "\$IMAGE"
SCRIPT
)

  # Check for existing eval VMs
  EXISTING_VMS=$(find_vms)
  if [[ -n "$EXISTING_VMS" ]]; then
    echo "WARNING: existing evaluation VM(s) found:" >&2
    while IFS=$'\t' read -r name zone _ip status; do
      echo "    $name ($zone, $status)" >&2
    done <<< "$EXISTING_VMS"
    echo "  Run '$(basename "$0") teardown' first, or this will create an additional VM." >&2
    for _countdown in 5 4 3 2 1; do
      printf "\r  Continuing in %ds... (Ctrl-C to cancel)  " "$_countdown" >&2
      sleep 1
    done
    echo "" >&2
  fi

  # Firewall rule (idempotent)
  echo "==> Creating firewall rule..."
  if gcloud compute firewall-rules describe "$FW_RULE" --project="$PROJECT" &>/dev/null; then
    echo "    Firewall rule already exists."
  else
    if ! firewall_error=$(gcloud compute firewall-rules create "$FW_RULE" \
      --project="$PROJECT" \
      --allow=tcp:8000,tcp:8001 \
      --source-ranges=0.0.0.0/0 \
      --target-tags=highperfasr-eval \
      --description="HighPerfASR evaluation — auto-created by launch-gce.sh" \
      --quiet 2>&1); then
      echo "" >&2
      echo "ERROR: firewall rule creation failed." >&2
      echo "$firewall_error" >&2
      echo "Common causes:" >&2
      echo "  - Insufficient permissions: need roles/compute.securityAdmin or compute.firewalls.create" >&2
      echo "  - Organization policy blocks public ingress firewall rules" >&2
      echo "  - A conflicting firewall rule name exists in another evaluation flow" >&2
      echo "" >&2
      echo "Cleanup: $(basename "$0") teardown" >&2
      exit 1
    fi
    echo "    Created $FW_RULE (ports 8000-8001, tag: highperfasr-eval)."
  fi

  # Create VM
  echo "==> Creating VM: $VM_NAME..."
  CREATE_FLAGS=(
    --project="$PROJECT"
    --machine-type="$MACHINE"
    --maintenance-policy=TERMINATE
    --no-restart-on-failure
    --image-family=common-cu129-ubuntu-2204-nvidia-580
    --image-project=deeplearning-platform-release
    --boot-disk-size=50GB
    --boot-disk-type=pd-balanced
    --tags=highperfasr-eval
    "--labels=app=$LABEL_APP,created-by=$LABEL_CREATOR,mode=$MODE,image-tag=${VERSION//./-}"
    "--metadata=startup-script=$STARTUP_SCRIPT,install-nvidia-driver=True"
    --no-service-account
    --no-scopes
    ${ACCEL_FLAGS[@]+"${ACCEL_FLAGS[@]}"}
  )

  if [[ "$ZONE_EXPLICIT" == "true" ]]; then
    ZONES_TO_TRY=("$ZONE")
  else
    if [[ "$GPU" == "l4" ]]; then
      ZONES_TO_TRY=("${L4_FALLBACK_ZONES[@]}")
    else
      ZONES_TO_TRY=("${T4_FALLBACK_ZONES[@]}")
    fi
  fi

  VM_CREATED=false
  for try_zone in "${ZONES_TO_TRY[@]}"; do
    if [[ "${#ZONES_TO_TRY[@]}" -gt 1 ]]; then
      echo "    Trying zone $try_zone..."
    fi
    if gcloud compute instances create "$VM_NAME" \
      --zone="$try_zone" "${CREATE_FLAGS[@]}" 2>&1; then
      ZONE="$try_zone"
      VM_CREATED=true
      break
    else
      if [[ "${#ZONES_TO_TRY[@]}" -gt 1 ]]; then
        echo "    No capacity in $try_zone, trying next..."
      fi
    fi
  done

  if [[ "$VM_CREATED" != "true" ]]; then
    echo "" >&2
    echo "ERROR: VM creation failed in all zones tried: ${ZONES_TO_TRY[*]}" >&2
    echo "" >&2
    echo "Common causes:" >&2
    echo "  - GPU capacity exhausted: all zones are out of $GPU GPUs right now" >&2
    echo "  - GPU quota insufficient: request at https://console.cloud.google.com/iam-admin/quotas" >&2
    echo "  - Insufficient permissions: need roles/compute.instanceAdmin.v1" >&2
    echo "" >&2
    echo "Cleanup: $(basename "$0") teardown" >&2
    exit 1
  fi

  # Get IP
  IP=$(gcloud compute instances describe "$VM_NAME" \
    --project="$PROJECT" --zone="$ZONE" \
    --format="value(networkInterfaces[0].accessConfigs[0].natIP)")
  SERVER_URL="http://${IP}:${PORT}"
  echo "    External IP: $IP"

  # Wait for health
  echo ""
  echo "==> Waiting for server (typically 5-10 minutes)..."
  echo ""
  _start_wait=$SECONDS
  for _attempt in $(seq 1 120); do
    if curl -sf "${SERVER_URL}/health" >/dev/null 2>&1; then
      _elapsed=$(( SECONDS - _start_wait ))
      echo ""
      echo ""
      echo "============================================================"
      echo ""
      echo "  HighPerfASR server is ready!  (${_elapsed}s)"
      echo ""
      echo "  Server    ${SERVER_URL}"
      echo "  Mode      ${MODE}"
      echo "  GPU       ${GPU}"
      echo "  Image     ghcr.io/beastoin/highperfasr-${MODE}:${VERSION}"
      echo "  TTL       ${TTL_HOURS}h auto-shutdown"
      echo ""
      echo "  ---"
      echo ""
      if [[ "$MODE" == "batch" ]]; then
        echo "  Quick test:"
        echo "    curl -F 'file=@audio.wav' ${SERVER_URL}/v1/transcriptions"
        echo ""
        echo "  Benchmark:"
        echo "    python3 benchmarks/scripts/bench_batch.py \\"
        echo "      --server ${SERVER_URL} \\"
        echo "      --concurrency 1,8,16,32,64 \\"
        echo "      --image-tag ${VERSION} \\"
        echo "      --output /tmp/bench_batch.json"
      else
        echo "  Quick test:"
        echo "    python3 -c \"import asyncio, websockets; asyncio.run(websockets.connect('ws://${IP}:${PORT}/v1/stream'))\""
        echo ""
        echo "  Benchmark:"
        echo "    python3 benchmarks/scripts/bench_stream.py \\"
        echo "      --server ws://${IP}:${PORT} \\"
        echo "      --endpoint /v1/stream \\"
        echo "      --concurrency 1,16,32 \\"
        echo "      --image-tag ${VERSION} \\"
        echo "      --output /tmp/bench_stream.json"
      fi
      echo ""
      echo "  Teardown:"
      echo "    $(basename "$0") teardown"
      echo ""
      echo "============================================================"
      exit 0
    fi

    _elapsed=$(( SECONDS - _start_wait ))
    if [[ $_elapsed -lt 60 ]]; then
      printf "\r    [%3ds] Installing GPU drivers...          " "$_elapsed"
    elif [[ $_elapsed -lt 180 ]]; then
      printf "\r    [%3ds] Pulling container image...         " "$_elapsed"
    elif [[ $_elapsed -lt 420 ]]; then
      printf "\r    [%3ds] Loading ASR model into GPU...      " "$_elapsed"
    else
      printf "\r    [%3ds] Waiting for /health...             " "$_elapsed"
    fi
    sleep 5
  done

  echo ""
  echo ""
  echo "WARNING: server not healthy after 10 minutes." >&2
  echo "The VM ($VM_NAME) is running but the container may still be starting." >&2
  echo "" >&2
  echo "Keep waiting (check every 5s until healthy):" >&2
  echo "  until curl -sf ${SERVER_URL}/health; do sleep 5; done && echo 'Ready!'" >&2
  echo "" >&2
  echo "Check container logs:" >&2
  echo "  gcloud compute ssh $VM_NAME --zone=$ZONE --project=$PROJECT -- docker logs $CONTAINER_NAME" >&2
  echo "" >&2
  echo "Check if container is running:" >&2
  echo "  gcloud compute ssh $VM_NAME --zone=$ZONE --project=$PROJECT -- docker ps" >&2
  echo "" >&2
  echo "Cleanup: $(basename "$0") teardown" >&2
  exit 1
}

# --- Main ---
validate_inputs
preflight

case "$ACTION" in
  teardown) do_teardown ;;
  deploy)   do_deploy ;;
esac
