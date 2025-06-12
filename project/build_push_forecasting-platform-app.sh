#!/bin/bash
# Automatically detect image name to build and push based on script filename

AWS_REGION="ap-southeast-2"
ACCOUNT_ID="890742606479"

# Get the script's own filename
SCRIPT_NAME="${0##*/}"

# Extract IMAGE_NAME from the script filename (remove prefix and suffix)
IMAGE_NAME="${SCRIPT_NAME#build_push_}"
IMAGE_NAME="${IMAGE_NAME%.sh}"

ECR_REPO="$ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$IMAGE_NAME"

echo "Building and pushing image: $IMAGE_NAME"

# Login to ECR
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

# Build Docker image
docker build -t $IMAGE_NAME:latest .

# Tag the image
docker tag $IMAGE_NAME:latest $ECR_REPO:latest

# Push to ECR
docker push $ECR_REPO:latest

echo "✅ $IMAGE_NAME has been pushed to $ECR_REPO:latest"
