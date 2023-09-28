## Triton-Pipeline

### Installing
- Create a image that wraps triton image
```
docker build -t triton_pineline .
```

### Prepare model
- 

### Run Server

```
docker run --gpus=1 -itd --add-host=host.docker.internal:host-gateway -p 8050-8052:8000-8002 -v ${PWD}/model_repository:/models --name triton_test triton_test:latest tritonserver --model-repository=/models
```


```
docker run --gpus=1 -itd -p 8050-8052:8000-8002 -v ${PWD}/model_repository:/models --name triton_test nvcr.io/nvidia/tritonserver:21.10-py3 tritonserver --model-repository=/models

```

```
docker run --gpus=1 --rm --net=host -v ${PWD}/model_repository:/models nvcr.io/nvidia/tritonserver:23.08-py3 tritonserver --model-repository=/models

```

# NOW TEST
```
docker run --gpus=1 -itd --add-host=host.docker.internal:host-gateway -p 8050-8052:8000-8002 -v ${PWD}/model_repository:/models --name triton_test triton_test:latest tritonserver --model-repository=/models
```

```
docker logs -f triton_test
```