#!/usr/bin/env bash
set -euo pipefail

# ===== 1. Detect deploy and project root directories =====
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DEPLOY_DIR="$SCRIPT_DIR"
PROJECT_DIR="$SCRIPT_DIR/../project"     # project is one level up from deploy
PROJECTS=(python-service forecasting-platform streamlit-ui)

# ===== 2. Load environment variables from .env if present =====
if [ -f "$DEPLOY_DIR/.env" ]; then
  # Export variables defined in .env
  export $(grep -v '^#' "$DEPLOY_DIR/.env" | xargs)
fi

# Use values from .env or fallback to default
ACCOUNT_ID="${ACCOUNT_ID:-890742606479}"
REGION="${REGION:-ap-southeast-2}"
TAG="latest"
PUSH=false

# ===== 3. Parse command line arguments (optional) =====
while [[ $# -gt 0 ]]; do
  case "$1" in
    --push)    PUSH=true ;;
    --region)  REGION="$2";  shift ;;
    --account) ACCOUNT_ID="$2"; shift ;;
    --project-dir) PROJECT_DIR="$2"; shift ;;
    *) echo "Unknown arg $1"; exit 1 ;;
  esac
  shift
done

ECR_BASE="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"

# ===== 4. Log in to ECR if pushing =====
if $PUSH; then
  aws ecr get-login-password --region "$REGION" |
    docker login --username AWS --password-stdin "$ECR_BASE"
fi

# ===== 5. Loop to build and optionally push images =====
for PROJ in "${PROJECTS[@]}"; do
  SRC_PATH="${PROJECT_DIR}/${PROJ}"
  IMAGE_LOCAL="${PROJ}-app:${TAG}"

  echo "🚧 Building $SRC_PATH -> $IMAGE_LOCAL"
  docker build -t "$IMAGE_LOCAL" "$SRC_PATH"

  if $PUSH; then
    REPO_URI="${ECR_BASE}/${PROJ}-app"
    # Create ECR repo if not exists
    aws ecr describe-repositories --repository-names "${PROJ}-app" \
          --region "$REGION" > /dev/null 2>&1 ||
      aws ecr create-repository --repository-name "${PROJ}-app" \
          --region "$REGION" > /dev/null
    docker tag "$IMAGE_LOCAL" "${REPO_URI}:${TAG}"
    docker push "${REPO_URI}:${TAG}"
  fi
done

echo "🎉 All done!"