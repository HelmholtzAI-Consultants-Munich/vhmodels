# vhmodels on Kubernetes (experimental)

Runs the [Ray runtime](../docs/ray.md) on Kubernetes, in namespace `chc`: a
Ray head pod and, per model, a Ray worker pod started from the model's
[image](../docs/images.md). Scripts and Serve configs are the same as on
Slurm.

The Ray and Serve ports have no authentication; reach them only through
`kubectl port-forward`.

```text
k8s/
├── head.yaml     Ray head: dashboard, Ray Jobs, Ray Serve
├── models/       Ray worker per model: <model id>.yaml
├── serve.yaml    Serve config, one app per model, deployed by the head on every start
└── run.sh        end-to-end test
```

## Models

| Model | Serve app | Variant | Resources | Cache volume |
| --- | --- | --- | --- | --- |
| MolE | `/mole` | | 2 CPUs, 6Gi | 5Gi |
| DinoBloom | `/dinobloom` | `s` | 2 CPUs, 6Gi | 10Gi |
| H-Optimus-0 | `/hoptimus0` | | 2 CPUs, 16Gi, 1 GPU | 10Gi |
| Hyformer | `/hyformer` | `hyformer_molecules_8M` | 2 CPUs, 4Gi | 5Gi |
| Nicheformer | `/nicheformer` | | 2 CPUs, 4Gi | 5Gi |
| ProtTrans | `/prottrans` | `prot_bert` | 2 CPUs, 8Gi | 20Gi |

- **GPU.** H-Optimus-0 runs on the namespace's GPU; its app and one Ray job share it. The other models run on CPU.
- **Memory** fits the app plus one Ray job of a model of similar size. DinoBloom `g` and the ProtTrans T5 variants need more; see [Add a model](#add-a-model).
- **All six** together with the head take 13 CPUs, 48Gi of memory and the GPU. Start only the workers you need; apps without a worker wait in `DEPLOYING`.
- **H-Optimus-0** is gated; its worker needs the token Secret from [Setup](#setup-once).

## Setup (once)

```bash
brew install kubectl int128/kubelogin/kubelogin
export KUBECONFIG=<path to the namespace's kubeconfig>
kubectl auth whoami

uv venv --managed-python --python 3.11 .venv
uv pip install --python .venv/bin/python -e ".[cli,ray]"
```

Put the `export` in `~/.zshrc`; `kubectl auth whoami` opens the browser login.
The `.venv` has the cluster's Ray version (2.59.0); use `.venv/bin/ray` for
`ray` commands.

**H-Optimus-0:** accept its conditions on
<https://huggingface.co/bioptimus/H-optimus-0> and create a token of type Read
at <https://huggingface.co/settings/tokens>. The commands below store it in the
Secret that `models/hoptimus0.yaml` reads. Paste the token at the prompt of the
first line and press Enter; it stays invisible and out of the shell history.
The `curl` line prints `200` if the token can download the model (`401`: wrong
token, `403`: conditions not accepted):

```bash
printf 'Hugging Face token: '; read -rs HF_TOKEN; echo
curl -sI -o /dev/null -w '%{http_code}\n' -H "Authorization: Bearer $HF_TOKEN" https://huggingface.co/bioptimus/H-optimus-0/resolve/main/config.json
kubectl -n chc create secret generic vhmodels-hf-token --from-literal=HF_TOKEN="$HF_TOKEN"
unset HF_TOKEN
```

To replace the token, run `kubectl -n chc delete secret vhmodels-hf-token`, the
commands above, and `kubectl -n chc rollout restart statefulset/vhmodels-hoptimus0`
if the worker is running: it reads the Secret only when it starts.

## How to use

**Run the test** from the repo root. [run.sh](run.sh) starts the head, the
workers and the Serve config. Then, per model, it sends an example input to
its app and runs an example as a Ray job. Afterwards it deletes what
it started; `KEEP=1` leaves it running. Without arguments it tests every
model; pass model ids to test only those. The first start takes several
minutes (image pulls and model downloads).

```bash
k8s/run.sh
k8s/run.sh mole hyformer
KEEP=1 k8s/run.sh
```

**Start the service** without the test, with all workers or only some:

```bash
kubectl -n chc apply -f k8s/serve.yaml -f k8s/head.yaml -f k8s/models/
kubectl -n chc apply -f k8s/serve.yaml -f k8s/head.yaml -f k8s/models/mole.yaml
```

**Use a running service.** Open the tunnels in one terminal and leave it open:

```bash
kubectl -n chc port-forward svc/vhmodels-head 8265:8265 8000:8000
```

Then, in another terminal:

```bash
curl -F input=@example_data/MolE/examples_molecules.tsv localhost:8000/mole/predict/files
curl -F input=@example_data/DinoBloom/001.bmp localhost:8000/dinobloom/embed/files
curl -H 'Content-Type: application/json' -d '{"input": ["PRTEINO", "SEQWENCE"]}' localhost:8000/prottrans/embed
curl localhost:8000/mole/info
.venv/bin/ray job submit --address http://localhost:8265 --working-dir <dir> -- python <script>.py
.venv/bin/ray job logs -f --address http://localhost:8265 <job id>
```

The dashboard is at <http://localhost:8265>: jobs, logs, actors and Serve apps.
See [Ray Serve](../docs/ray.md#ray-serve) for the other routes and
[File uploads](../docs/ray.md#file-uploads) for each model's form fields.
`GET <app>/info` shows a model's input.

**Change the Serve config:** apply it, wait up to a minute until the head sees
the change, then deploy it:

```bash
kubectl -n chc apply -f k8s/serve.yaml
kubectl -n chc exec deploy/vhmodels-head -- serve deploy /etc/vhmodels/serve.yaml
```

**Watch the pods.** Worker pods are named `vhmodels-<model id>-0`:

```bash
kubectl -n chc get pods -w
kubectl -n chc logs -f vhmodels-mole-0
```

**Stop everything.** The second command also deletes the models' downloaded
files; `kubectl -n chc delete pvc cache-vhmodels-<model id>-0` deletes one
model's:

```bash
kubectl -n chc delete -f k8s/serve.yaml -f k8s/head.yaml -f k8s/models/
kubectl -n chc delete pvc -l app.kubernetes.io/part-of=vhmodels-inference
```

## Add a model

Copy a file in `models/` to `models/<model id>.yaml` with the model's image,
name, label (`vhmodels/project=<model id>`), resources and volume size. Then add
its app to `serve.yaml` and its test to `test_spec` in `run.sh`.

- **More memory** for bigger variants: raise `requests` and `limits` together.
- **GPU workers** request `nvidia.com/gpu: 1`, tolerate the GPU nodes' taint and start Ray with `--num-gpus=1`, like `models/hoptimus0.yaml`. Their Serve app sets `device: cuda:0`, and `ray_options: {num_gpus: 0.5}` to leave half of the GPU to Ray jobs.
- **Gated models** read `HF_TOKEN` from a Secret, like `models/hoptimus0.yaml`.
