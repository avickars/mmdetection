SHELL := /bin/bash

push-torch: registry-login
	docker push ghcr.io/avickars/mmdet-serve:v1

build-torch:
	docker build --tag ghcr.io/avickars/mmdet-serve:v1 docker/serve/

registry-login:
	@/bin/sh -c 'cat ghcr.io-token.txt | docker login ghcr.io -u avickars --password-stdin'

run-torch:
    docker run --rm \
    --cpus 4 \
    --gpus device=0 \
    -p8080:8080 -p8081:8081 -p8082:8082 \
    --mount type=bind,source=$(pwd)/model-store,target=/home/model-server/model-store \
    ghcr.io/avickars/mmdet-serve:v1

test-torch:
    curl http://127.0.0.1:8080/predictions/seg -T 3dogs.jpg

