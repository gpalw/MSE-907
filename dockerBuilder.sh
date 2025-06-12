#!/usr/bin/env bash
set -euo pipefail

# ───── ① 关键：计算 project 根目录 ─────
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="${SCRIPT_DIR}/project"         # 如果你的目录名不是 project，请改这里
PROJECTS=(python-service forecasting-platform streamlit-ui)

# ───── 其他变量（保持不变，可按需覆盖）─────
ACCOUNT_ID="890742606479"
REGION="ap-southeast-2"
TAG="latest"
PUSH=false

# ───── ② 命令行参数解析（略）─────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --push)    PUSH=true ;;
    --region)  REGION="$2";  shift ;;
    --account) ACCOUNT_ID="$2"; shift ;;
    --project-dir) PROJECT_DIR="$2"; shift ;;   # 如需手动指定
    *) echo "Unknown arg $1"; exit 1 ;;
  esac
  shift
done

ECR_BASE="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"

# ───── ③ 如需推送，先登录 ECR（同前）─────
if $PUSH; then
  aws ecr get-login-password --region "$REGION" |
    docker login --username AWS --password-stdin "$ECR_BASE"
fi

# ───── ④ 循环构建/推送 ─────
for PROJ in "${PROJECTS[@]}"; do
  SRC_PATH="${PROJECT_DIR}/${PROJ}"      # ★ 路径已改为绝对
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
