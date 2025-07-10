#!/usr/bin/env bash
set -euo pipefail

# ───── ① Key: Calculate the root directory of the project ─────
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="${SCRIPT_DIR}/project"         # If your folder name is not 'project', modify here
PROJECTS=(python-service forecasting-platform streamlit-ui)

# ───── Other variables (remain unchanged, can be overridden as needed) ─────
ACCOUNT_ID="890742606479"
REGION="ap-southeast-2"
TAG="latest"
PUSH=false

# ───── ② Command-line argument parsing (omitted details) ─────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --push)    PUSH=true ;;
    --region)  REGION="$2";  shift ;;
    --account) ACCOUNT_ID="$2"; shift ;;
    --project-dir) PROJECT_DIR="$2"; shift ;;   # Use this to specify manually if needed
    *) echo "Unknown arg $1"; exit 1 ;;
  esac
  shift
done

ECR_BASE="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"

# ───── ③ If push is required, login to ECR first (same as before) ─────
if $PUSH; then
  aws ecr get-login-password --region "$REGION" |
    docker login --username AWS --password-stdin "$ECR_BASE"
fi

# ───── ④ Loop for build/push ─────
for PROJ in "${PROJECTS[@]}"; do
  SRC_PATH="${PROJECT_DIR}/${PROJ}"      # ★ The path has been changed to absolute
  IMAGE_LOCAL="${PROJ}-app:${TAG}"

  echo "🚧 Building $SRC_PATH -> $IMAGE_LOCAL"
  docker build -t "$IMAGE_LOCAL" "$SRC_PATH"

  if $PUSH; then
    REPO_URI="${ECR_BASE}/${PROJ}-app"
    aws ecr describe-repositories --repository-names "${PROJ}-app" \
          --region "$REGION" > /dev/null 2>&1 ||
      aws ecr create-repository --repository-name "${PROJ}-app" \
          --region "$REGION" > /dev/null
    docker tag "$IMAGE_LOCAL" "${REPO_URI}:${TAG}"
    docker push "${REPO_URI}:${TAG}"
  fi
done

echo "🎉 All done!"
