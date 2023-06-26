# FP8 mixed precision

[NVIDIA's Transformer Engine](https://github.com/NVIDIA/TransformerEngine) (TE) is a library for accelerating Transformer models on NVIDIA GPUs, including using 8-bit floating point (FP8) precision on Hopper GPUs, to provide better performance with lower memory utilization in both training and inference

To use it, you'll need access to an H100 machine. If you don't have it, you can use CoreWeave's cloud.

## Access CoreWeave's H100 machines

If you haven't accessed CoreWeave's Kubernetes before, please follow [these instruction](https://docs.coreweave.com/coreweave-kubernetes/getting-started) to set it up.

Create this `manifest.yaml` file. Feel free to adapt it to your requirements:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: h100
spec:
  containers:
  - name: h100
    # Default base image. See section below to build your own
    image: "ghcr.io/coreweave/ml-containers/torch:ceeb8c2-nccl-cuda12.0.1-nccl2.18.1-1-torch2.0.1-vision0.15.2-audio2.0.2"
    command:  ["tail", "-f", "/dev/null"]  # keep it running indefinitely
    resources:
      limits:
        memory: 960Gi
        nvidia.com/gpu: 8
      requests:
        cpu: 110
        memory: 960Gi
  # if you set up a persistent volume (named h100-data) via the CoreWeave UI
  # uncomment the following. docs: https://docs.coreweave.com/storage/storage/using-storage-kubectl
  #  volumeMounts:
  #  - mountPath: /storage
  #    name: "h100-data"
  #volumes:
  #- name: "h100-data"
  #  persistentVolumeClaim:
  #    claimName: h100-data
  affinity:
    nodeAffinity:
      requiredDuringSchedulingIgnoredDuringExecution:
        nodeSelectorTerms:
        - matchExpressions:
          - key: gpu.nvidia.com/class
            operator: In
            values:
            - H100_NVLINK_80GB
          - key: topology.kubernetes.io/region
            operator: In
            values:
            - LAS1
```

Use `kubectl` to launch and interact with your pod:

```shell
kubectl apply -f manifest.yaml  # launch h100 pod
kubectl exec -it pod/h100 -- /bin/bash  # SSH into it
kubectl delete pod/h100  # Delete the pod once you are done
```

You can find a list of useful commands [here](https://docs.coreweave.com/coreweave-kubernetes/useful-commands). The `pod_id` is `pod/h100`.

### Dockerfile

The configuration file above uses one of CoreWeave's base images. You might want to create your own Dockerfile to
avoid having to set up the machine every time you use it. Here's an example that prepares the steps to use Lit-Parrot:

```dockerfile
FROM ghcr.io/coreweave/ml-containers/torch:ceeb8c2-nccl-cuda12.0.1-nccl2.18.1-1-torch2.0.1-vision0.15.2-audio2.0.2

RUN pip install --index-url https://download.pytorch.org/whl/nightly/cu121 --pre 'torch>=2.1.0dev' \
    && pip install git+https://github.com/Lightning-AI/lightning.git@carmocca/transformer-engine \
    && pip install -U 'setuptools>=49.4.0' \
    && pip install flash-attn --no-build-isolation \
    && NVTE_FRAMEWORK=pytorch pip install git+https://github.com/NVIDIA/TransformerEngine.git@main --no-deps \
    && git clone https://github.com/Lightning-AI/lit-gpt && cd lit-gpt && git checkout carmocca/h100 \
    && pip install -r requirements.txt
```

Then you'll need to push your image, here's on guide using DockerHub:

1. Install Docker: If you haven't already, [install Docker](https://docs.docker.com/engine/install/) on your local machine.
2. Build the Docker image: Open a terminal, navigate to the directory containing your Dockerfile, and run the following command to build the Docker image: 
    ```bash
    docker build -t lit-gpt-h100:v1 .
    ```
    Docker will read the Dockerfile and execute the instructions to create the image. It may take some time, depending on the complexity of your image and the internet speed to fetch dependencies.

3. Log in to Docker Hub: If you want to upload the image to Docker Hub, create an account on the [Docker Hub](hub.docker.com).
    ```bash
    docker login
    ```

4. Tag the Docker image: Before pushing the image, you need to tag it with the repository information.
    ```bash
    docker tag lit-gpt-h100:v1 username/repository:v1
    ```
    Replace username/repository:tag with your Docker Hub username and the desired repository name.

5. Push the Docker image
    ```bash
    docker push username/repository:v1
    ```

### Alternatively, set up your environment

TransformerEngine requires some specific installation steps:

```shell
# you'll want CUDA 12.1
pip install --index-url https://download.pytorch.org/whl/nightly/cu121 --pre 'torch>=2.1.0dev'
pip install git+https://github.com/Lightning-AI/lightning.git@carmocca/transformer-engine
pip install -U 'setuptools>=49.4.0'
# needs to be installed separately until https://github.com/HazyResearch/flash-attention/issues/246 is resolved
pip install flash-attn --no-build-isolation
NVTE_FRAMEWORK=pytorch pip install git+https://github.com/NVIDIA/TransformerEngine.git@main --no-deps
```
