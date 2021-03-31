PROJECT_NAME=mnist

# Build and run the Docker image for model training.
docker build -t $PROJECT_NAME:train ./examples/mnist/
docker run -it --rm \
    --name $PROJECT_NAME \
    --gpus all \
    --ipc host \
    -v ~/.config/gcloud:/root/.config/gcloud \
    -v $(pwd)/data/mnist:/app/data/mnist \
    $PROJECT_NAME:train
