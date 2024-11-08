# MLOps Workflow: Recognizing Digits with Kubeflow

The [MNIST database of handwritten digits](http://yann.lecun.com/exdb/mnist/) is the Hello-World of deep learning and therefore the best example to focus not on the ML model itself, but on creating the ML pipeline. The goal here is to create an automated ML pipeline for getting the data, data pre-processing, and creating and serving the ML model. You can see an overview of the digits recognizer application below.

![](images/app-overview.jpg)

**You need to follow these steps**:

1. Deploy a Kubernetes Cluster and install Kubeflow
2. Access the Kubeflow Central Dashboard
3. Setup Jupyter Notebooks
4. Setup MinIO for Object Storage
5. Setting up Kserve
6. Create a ML pipeline with Kubeflow Pipelines
7. Test the model inference

**Used Components**:

* Kubeflow 2.9.0 - Notebook, Pipelines, Kserve
* MinIO
* Kubernetes

[![youtube](images/youtube.png)](https://youtu.be/6wWdNg0GMV4)
*Check out the [Walk-through Video](https://youtu.be/6wWdNg0GMV4)!*

## Prerequisite: Setup Development Environment
```
# Install Minikube
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube && rm minikube-linux-amd64
minikube version

# Install Docker
sudo apt-get update && sudo apt-get -y install docker.io
sudo usermod -aG docker $USER && newgrp docker

# Install Kubectl
sudo snap install kubectl --classic

# Install kustomize
sudo snap install kustomize 

# Clone Kubeflow manifest
git clone https://github.com/kubeflow/manifests.git

# Start minikube
minikube start --cpus 4 --memory 8096 --disk-size=40g
```

## 1. Deploy a Kubernetes Cluster and install Kubeflow

Install Kubeflow on your Kubernetes cluster. You can find more information in the [Kubeflow docs](https://www.kubeflow.org/docs/started/installing-kubeflow/).

```
cd Kubeflow/manifests
while ! kustomize build example | kubectl apply -f -; do echo "Retrying to apply resources"; sleep 10; done
- or -
while ! kustomize build example | kubectl apply --server-side --force-conflicts -f -; do echo "Retrying to apply resources"; sleep 20; done
- or -
while ! kustomize build example | awk '!/well-defined/' | kubectl apply -f -; do echo "Retrying to apply resources"; sleep 10; done
```

You can check with kubectl if all pods are coming up successfully: 

```
kubectl get pods --all-namespaces
NAMESPACE                   NAME                                                              READY   STATUS      RESTARTS   AGE
auth                        dex-5ddf47d88d-cksfj                                              1/1     Running     1          3h7m
cert-manager                cert-manager-7b8c77d4bd-m4zht                                     1/1     Running     0          3h7m
cert-manager                cert-manager-cainjector-7c744f57b5-nzfb4                          1/1     Running     0          3h7m
cert-manager                cert-manager-webhook-fcd445bc4-7fkj4                              1/1     Running     0          3h7m
istio-system                authservice-0                                                     1/1     Running     0          3h7m
istio-system                cluster-local-gateway-64f58f66cb-ncnkd                            1/1     Running     0          3h7m
istio-system                istio-ingressgateway-8577c57fb6-c8t9p                             1/1     Running     0          3h7m
istio-system                istiod-6c86784695-bvgqs                                           1/1     Running     0          3h7m
knative-eventing            eventing-controller-79895f9c56-2zpmv                              1/1     Running     0          3h7m
knative-eventing            eventing-webhook-78f897666-n5m5q                                  1/1     Running     0          3h7m
knative-eventing            imc-controller-688df5bdb4-66gvz                                   1/1     Running     0          3h7m
knative-eventing            imc-dispatcher-646978d797-2z2b2                                   1/1     Running     0          3h7m
knative-eventing            mt-broker-controller-67c977497-mgtmc                              1/1     Running     0          3h7m
knative-eventing            mt-broker-filter-66d4d77c8b-gjrhc                                 1/1     Running     0          3h7m
knative-eventing            mt-broker-ingress-5c8dc4b5d7-tgh6l                                1/1     Running     0          3h7m
knative-serving             activator-7476cc56d4-lwtqq                                        2/2     Running     2          3h7m
knative-serving             autoscaler-5c648f7465-wzmzl                                       2/2     Running     0          3h7m
knative-serving             controller-57c545cbfb-nnjcm                                       2/2     Running     0          3h6m
knative-serving             istio-webhook-578b6b7654-s445x                                    2/2     Running     0          3h7m
knative-serving             networking-istio-6b88f745c-887mz                                  2/2     Running     0          3h7m
knative-serving             webhook-6fffdc4d78-ml2mn                                          2/2     Running     0          3h7m
kserve                      kserve-controller-manager-0                                       2/2     Running     0          3h7m
kubeflow-user-example-com   ml-pipeline-ui-artifact-d57bd98d7-s84t4                           2/2     Running     0          174m
kubeflow-user-example-com   ml-pipeline-visualizationserver-65f5bfb4bf-bmtg8                  2/2     Running     0          174m
kubeflow                    admission-webhook-deployment-7df7558c67-d7mfm                     1/1     Running     0          3h7m
kubeflow                    cache-deployer-deployment-6f4bcc969-zh9vx                         2/2     Running     1          3h7m
kubeflow                    cache-server-575d97c95-jc4nw                                      2/2     Running     0          3h7m
kubeflow                    centraldashboard-79f489b55-cr7hn                                  2/2     Running     0          3h7m
kubeflow                    jupyter-web-app-deployment-5886974887-m96wv                       1/1     Running     0          3h7m
kubeflow                    katib-controller-58ddb4b856-9zjtj                                 1/1     Running     0          3h7m
kubeflow                    katib-db-manager-d77c6757f-jt9b6                                  1/1     Running     4          3h7m
kubeflow                    katib-mysql-7894994f88-zzwrz                                      1/1     Running     0          3h7m
kubeflow                    katib-ui-f787b9d88-cwg9l                                          1/1     Running     0          3h7m
kubeflow                    kfserving-controller-manager-0                                    2/2     Running     0          3h6m
kubeflow                    kfserving-models-web-app-7884f597cf-8vg4b                         2/2     Running     0          3h7m
kubeflow                    kserve-models-web-app-5c64c8d8bb-sqtzs                            2/2     Running     0          3h7m
kubeflow                    kubeflow-pipelines-profile-controller-84bcbdb899-rddgd            1/1     Running     0          3h7m
kubeflow                    metacontroller-0                                                  1/1     Running     0          3h6m
kubeflow                    metadata-envoy-deployment-7b847ff6c5-cqvkw                        1/1     Running     0          3h7m
kubeflow                    metadata-grpc-deployment-f8d68f687-kqlgq                          2/2     Running     4          3h7m
kubeflow                    metadata-writer-78fc7d5bb8-t5xp7                                  2/2     Running     0          3h7m
kubeflow                    minio-5b65df66c9-sx8kb                                            2/2     Running     0          3h7m
kubeflow                    ml-pipeline-7bb5966955-87jbv                                      2/2     Running     6          3h7m
kubeflow                    ml-pipeline-persistenceagent-87b6888c4-n9tgd                      2/2     Running     0          3h7m
kubeflow                    ml-pipeline-scheduledworkflow-665847bb9-2mpz6                     2/2     Running     0          3h7m
kubeflow                    ml-pipeline-ui-554ffbd6cd-8sswm                                   2/2     Running     0          3h7m
kubeflow                    ml-pipeline-viewer-crd-68777557fb-k65lr                           2/2     Running     1          3h7m
kubeflow                    ml-pipeline-visualizationserver-66c54744c-cp2np                   2/2     Running     0          3h7m
kubeflow                    mysql-f7b9b7dd4-56gjd                                             2/2     Running     0          3h7m
kubeflow                    notebook-controller-deployment-7474fbff66-26fzm                   2/2     Running     1          3h7m
kubeflow                    profiles-deployment-5cc86bc965-vjfv6                              3/3     Running     1          3h7m
kubeflow                    tensorboard-controller-controller-manager-5cbddb7fb5-cglzr        3/3     Running     1          3h7m
kubeflow                    tensorboards-web-app-deployment-7c5db448d7-84pjw                  1/1     Running     0          3h7m
kubeflow                    training-operator-6bfc7b8d86-l59l8                                1/1     Running     0          3h7m
kubeflow                    volumes-web-app-deployment-87484c848-rl4rl                        1/1     Running     0          3h7m
kubeflow                    workflow-controller-5cb67bb9db-7bfqc                              2/2     Running     2          3h7m
```

## 2. Access the Kubeflow Central Dashboard

Once you have everything deployed, you can do a port-forward with the following command:

```
kubectl port-forward svc/istio-ingressgateway -n istio-system 8080:80
```

and access the Kubeflow Central Dashboard remotely at [http://localhost:8080](http://localhost:8080).

![](images/kf_central_dashboard.png)

## 3. Setup Jupyter Notebooks

### Allow access to Kubeflow Pipelines from Jupyter Notebooks

In this demo you will access the Kubeflow Pipeline via the Python SDK in a Jupyter notebook. Therefore, one additional setting is required to allow this.

At first insert your Kubeflow username in this Kubernetes manifest (your Kubeflow username is also the name of a Kubernetes namespace where all your user-specific containers will be spun up): [kubeflow_config/access_kfp_from_jupyter_notebook.yaml](kubeflow_config/access_kfp_from_jupyter_notebook.yaml). You can the extract namespace name under the **Manage Contributers** menu.

Once done, apply it with this command:

```
kubectl apply -f kubeflow_configs/access_kfp_from_jupyter_notebook.yaml
```

### Spinning up a new Notebook Instance

Now, you need to spin a up new Jupyter notebook instance. For the container image select **jupyter-tensorflow-full:v1.9.2**. This can take several minutes depending on your download speed.

![](images/kf_notebook_1.9.2.png)

Don't forget to enable this configuration:

![](images/kf_kfp_config.png)

### Access Jupyter Notebooks & Cloning the code from Github

Go to **Notebooks** and click on **CONNECT** to start the Jupyter Notebook container.

With Juypter Lab you have access to a terminal and Python notebook in your web browser. This is where your data science team and you can collaborate on exploring that dataset and also create your Kubeflow Pipeline.

At first, let's clone this repository so you have access to the code. You can use the terminal or directly do that in the browser.

```
git clone https://github.com/flopach/digits-recognizer-kubeflow-intersight
```

Then open `digits_recognizer_notebook.ipynb` to get a feeling of the [dataset](http://yann.lecun.com/exdb/mnist/) and its format.

### Update Python Packages

Once started, double check if the latest versions of the Kubeflow python packages are installed within the Jupyter notebook container:

`pip list` should list versions above these::

```
Package                      Version
---------------------------- --------------
absl-py                      2.1.0
annotated-types              0.7.0
anyio                        4.6.0
archspec                     0.2.3
argon2-cffi                  23.1.0
argon2-cffi-bindings         21.2.0
arrow                        1.3.0
asttokens                    2.4.1
astunparse                   1.6.3
async-lru                    2.0.4
attrs                        24.2.0
Babel                        2.14.0
beautifulsoup4               4.12.3
bleach                       6.1.0
bokeh                        3.3.4
boltons                      24.0.0
Brotli                       1.1.0
cached-property              1.5.2
cachetools                   5.5.0
certifi                      2024.8.30
cffi                         1.17.1
charset-normalizer           3.3.2
click                        8.1.7
cloudevents                  1.11.0
cloudpickle                  2.2.1
colorama                     0.4.6
comm                         0.2.2
conda                        24.9.0
conda-libmamba-solver        24.9.0
conda-package-handling       2.3.0
conda_package_streaming      0.10.0
contourpy                    1.3.0
cycler                       0.12.1
debugpy                      1.8.6
decorator                    5.1.1
defusedxml                   0.7.1
deprecation                  2.1.0
dill                         0.3.8
distro                       1.9.0
docstring_parser             0.16
entrypoints                  0.4
exceptiongroup               1.2.2
executing                    2.1.0
fastapi                      0.115.4
fastjsonschema               2.20.0
flatbuffers                  24.3.25
fonttools                    4.54.1
fqdn                         1.5.1
frozendict                   2.4.4
gast                         0.6.0
gitdb                        4.0.11
GitPython                    3.1.43
google-api-core              2.20.0
google-auth                  2.35.0
google-auth-oauthlib         1.2.1
google-cloud-core            2.4.1
google-cloud-storage         2.18.2
google-crc32c                1.6.0
google-pasta                 0.2.0
google-resumable-media       2.7.2
googleapis-common-protos     1.65.0
grpcio                       1.66.2
h11                          0.14.0
h2                           4.1.0
h5py                         3.11.0
hpack                        4.0.0
httpcore                     1.0.6
httptools                    0.6.4
httpx                        0.27.2
hyperframe                   6.0.1
idna                         3.10
imagecodecs                  2024.6.1
imageio                      2.35.1
importlib_metadata           8.5.0
importlib_resources          6.4.5
ipykernel                    6.29.5
ipympl                       0.9.4
ipython                      8.27.0
ipython_genutils             0.2.0
ipywidgets                   8.1.5
isoduration                  20.11.0
jedi                         0.19.1
Jinja2                       3.1.4
joblib                       1.4.2
json5                        0.9.25
jsonpatch                    1.33
jsonpointer                  3.0.0
jsonschema                   4.23.0
jsonschema-specifications    2023.12.1
jupyter_client               8.6.3
jupyter_core                 5.7.2
jupyter-events               0.10.0
jupyter-lsp                  2.2.5
jupyter_server               2.14.2
jupyter-server-mathjax       0.2.6
jupyter_server_terminals     0.5.3
jupyterlab                   4.2.5
jupyterlab_git               0.50.1
jupyterlab_pygments          0.3.0
jupyterlab_server            2.27.3
jupyterlab_widgets           3.0.13
keras                        2.15.0
kfp                          2.9.0
kfp-pipeline-spec            0.4.0
kfp-server-api               2.3.0
kiwisolver                   1.4.7
kserve                       0.14.0
kubernetes                   30.1.0
lazy_loader                  0.4
libclang                     18.1.1
libmambapy                   1.5.10
mamba                        1.5.10
Markdown                     3.7
MarkupSafe                   2.1.5
matplotlib                   3.8.4
matplotlib-inline            0.1.7
menuinst                     2.1.2
minio                        7.2.10
mistune                      3.0.2
ml-dtypes                    0.3.2
munkres                      1.1.4
nbclient                     0.10.0
nbconvert                    7.16.4
nbdime                       4.0.2
nbformat                     5.10.4
nest_asyncio                 1.6.0
networkx                     3.3
notebook                     7.2.2
notebook_shim                0.2.4
numpy                        1.23.5
oauthlib                     3.2.2
opt_einsum                   3.4.0
orjson                       3.10.11
overrides                    7.7.0
packaging                    24.1
pandas                       2.2.3
pandocfilters                1.5.0
parso                        0.8.4
patsy                        0.5.6
pexpect                      4.9.0
pickleshare                  0.7.5
pillow                       10.4.0
pip                          24.3.1
pkgutil_resolve_name         1.3.10
platformdirs                 4.3.6
pluggy                       1.5.0
ply                          3.11
prometheus_client            0.20.0
prompt_toolkit               3.0.48
proto-plus                   1.24.0
protobuf                     4.25.5
psutil                       5.9.8
ptyprocess                   0.7.0
pure_eval                    0.2.3
pyasn1                       0.6.1
pyasn1_modules               0.4.1
pycosat                      0.6.6
pycparser                    2.22
pycryptodome                 3.21.0
pydantic                     2.9.2
pydantic_core                2.23.4
Pygments                     2.18.0
pyparsing                    3.1.4
PyQt5                        5.15.9
PyQt5-sip                    12.12.2
PySocks                      1.7.1
python-dateutil              2.9.0
python-dotenv                1.0.1
python-json-logger           2.0.7
pytz                         2024.2
PyWavelets                   1.7.0
PyYAML                       6.0.2
pyzmq                        26.2.0
referencing                  0.35.1
requests                     2.32.3
requests-oauthlib            2.0.0
requests-toolbelt            0.10.1
rfc3339-validator            0.1.4
rfc3986-validator            0.1.1
rpds-py                      0.20.0
rsa                          4.9
ruamel.yaml                  0.18.6
ruamel.yaml.clib             0.2.8
scikit-image                 0.22.0
scikit-learn                 1.3.2
scipy                        1.9.3
seaborn                      0.13.2
Send2Trash                   1.8.3
setuptools                   75.1.0
sip                          6.7.12
six                          1.16.0
smmap                        5.0.1
sniffio                      1.3.1
soupsieve                    2.5
stack-data                   0.6.2
starlette                    0.41.2
statsmodels                  0.14.3
tabulate                     0.9.0
tensorboard                  2.15.2
tensorboard-data-server      0.7.2
tensorflow                   2.15.1
tensorflow-estimator         2.15.0
tensorflow-io-gcs-filesystem 0.37.1
termcolor                    2.4.0
terminado                    0.18.1
threadpoolctl                3.5.0
tifffile                     2024.9.20
timing-asgi                  0.3.1
tinycss2                     1.3.0
toml                         0.10.2
tomli                        2.0.1
tornado                      6.4.1
tqdm                         4.66.5
traitlets                    5.14.3
truststore                   0.9.2
types-python-dateutil        2.9.0.20240906
typing_extensions            4.12.2
typing-utils                 0.1.0
tzdata                       2024.2
uri-template                 1.3.0
urllib3                      1.26.20
uvicorn                      0.30.6
uvloop                       0.21.0
watchfiles                   0.24.0
wcwidth                      0.2.13
webcolors                    24.8.0
webencodings                 0.5.1
websocket-client             1.8.0
websockets                   13.1
Werkzeug                     3.0.4
wheel                        0.44.0
widgetsnbextension           4.0.13
wrapt                        1.14.1
xgboost                      1.7.6
xyzservices                  2024.9.0
zipp                         3.20.2
zstandard                    0.23.0
```

Install missing packages

```bash
pip install kserve
pip install minio
...
```

### Behind a Proxy fix (optional)

If you are behind a proxy, apply the [kubeflow_configs/proxy-fix-notebooks.yaml](kubeflow_configs/proxy-fix-notebooks.yaml) fix to your kubernetes cluster.

## 4. Setup MinIO for Object Storage

In order to provide a single source of truth where all your working data (training and testing data, saved ML models etc.) is available to all your components, using an object storage is a recommended way. For our app, we will setup [MinIO](https://min.io).

Since Kubeflow has already setup a MinIO tenant, we will leverage the **mlpipeline bucket**. But you can also deploy your own MinIO tenant.

### Get credentials from Kubeflow's integrated MinIO

1. Obtain the accesskey and secretkey for MinIO with these commands:

```
kubectl get secret mlpipeline-minio-artifact -n kubeflow -o jsonpath="{.data.accesskey}" | base64 --decode
```

```
kubectl get secret mlpipeline-minio-artifact -n kubeflow -o jsonpath="{.data.secretkey}" | base64 --decode
```

2. In order to get access to MinIO from outside of your Kubernetes cluster and check the bucket, do a port-forward:

```
kubectl port-forward -n kubeflow svc/minio-service 9000:9000
```

3. Then you can access the MinIO dashboard at [http://localhost:9000](http://localhost:9000) and check the bucket name or create your own bucket. Alternatively, you can use the [MinIO CLI Client](https://docs.min.io/docs/minio-client-quickstart-guide.html)

**Default** values should be (already in the code and no action on your end):

* accesskey: **minio**
* secretkey: **minio123**
* bucket: **mlpipeline**

## 5. Setting up Kserve

In this step we are setting up Kserve for model inference serving. The Kserve container will be created when we are executing our ML pipeline which will happen in the next step.

### Install Kserve (Optional: if its already installed or running)

```
kubectl apply -f https://github.com/kserve/kserve/releases/download/v0.14.0-rc0/kserve_kubeflow.yaml
or
kubectl apply -f kubeflow_configs/create_kserve_inference.yaml
```

### Install Model Registry (Optional: if its already installed or running)
```
kubectl apply -k "https://github.com/kubeflow/model-registry/manifests/kustomize/overlays/db?ref=v0.2.3-alpha"
```

### Set minIO secret for kserve

We need to apply this yaml file so that the created model which is saved on minIO can be accessed by Kserve. Kserve will copy the saved model in the newly created inference container.

```
kubectl apply -f kubeflow_configs/set-minio-kserve-secret.yaml
```

### Troubleshooting: Can't fetch docker image

If kserve can't fetch the docker image at container startup, you need to edit the configuration:

`kubectl -n knative-serving edit configmap config-deployment`

Add the key-value *registriesSkippingTagResolving* directly below data and apply:

```
apiVersion: v1
data:
registriesSkippingTagResolving: "index.docker.io"
  _example: |
    ################################
    #                              #
    #    EXAMPLE CONFIGURATION     #
    #                              #
    ################################
...
```

Find more troubleshooting information: [https://kserve.github.io/website/developer/debug/](https://kserve.github.io/website/developer/debug/)

## 6. Create a ML pipeline with Kubeflow Pipelines

Kubeflow Pipelines (KFP) is the most used component of Kubeflow. It allows you to create for every step or function in your ML project a reusable containerized pipeline component which can be chained together as a ML pipeline.

For the digits recognizer application, the pipeline is already created with the Python SDK. You can find the code in the file `digits_recognizer_pipeline.ipynb`

![](images/pipeline_new.png)

![](images/kserve.png)

## 7. Test the model inference

Now you can test the model inference. The simplest way is to use a Python script directly in the Jupyter Notebook:

![](images/test-inference.png)

Alternatively, you can use the web application which you can find in the `web_app` folder. Be aware that some configuration needs to be done if you want to access the inference service from outside of the cluster.

## Versioning

**1.0** - Sample ML workflow with Kubeflow 1.5.1

**2.0** - Sample ML workflow with Kubeflow 2.9.0

## Authors

* **Flo Pachinger** - [flopach](https://github.com/flopach)

* **Ronila Sanjula** - [Roni-Boiz](https://github.com/Roni-Boiz)

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE.md](LICENSE.md) file for details