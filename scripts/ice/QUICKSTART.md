# PACE ICE quickstart

Commands for finding out what you can use on GT's PACE ICE cluster and for watching your jobs.
The repository's own scripts sit next to this file in `scripts/ice/`; the README section "GT PACE ICE"
explains the training pipeline. Everything here runs on a login node, from the repository root.

```bash
ssh <gt-username>@login-ice.pace.gatech.edu      # GT VPN when off campus
```

In a browser, ICE OnDemand > Clusters > ICE Shell Access opens the same terminal.

## 1. Where can I run? (partitions, QOS, accounts)

Start with the repository's summary. It lists every partition with its free nodes, GPU models and
time limit, and asks Slurm whether it would accept a job from you there (`sbatch --test-only`, so
nothing is submitted):

```bash
bash scripts/ice/partitions.sh                            # every partition
bash scripts/ice/partitions.sh coc-gpu ice-gpu pace-gpu   # only these
GPU_TYPE=h100 bash scripts/ice/partitions.sh coc-gpu ice-gpu   # can I get an H100 there?
```

The QOS column names the quality of service that worked, which is what you pass as `QOS=...`.
The same facts from plain Slurm:

| Question | Command |
| --- | --- |
| Which partitions exist, how full are they? | `sinfo -s` |
| GPUs, limits and node states per partition | `sinfo -p coc-gpu,ice-gpu,pace-gpu -o "%.12P %.5a %.10l %.5D %.8t %G"` |
| My accounts and QOS (empty partition = any) | `sacctmgr show assoc user=$USER format=account%15,partition%15,qos%40` |
| Who may use a partition, its max time | `scontrol show partition coc-gpu` (AllowAccounts, AllowQos, MaxTime) |
| Limits of a QOS (GPUs per user, wall time) | `sacctmgr show qos name=coc-ice format=name%12,maxwall,maxtrespu%30,maxjobspu,maxsubmitpu` |
| Would this exact job be accepted, and when would it start? | `sbatch --test-only -p coc-gpu --qos=coc-ice --gres=gpu:1 -t 1:00:00 --wrap true` |
| PACE's own view of a queue | `pace-check-queue coc-gpu` (a PACE tool; `sinfo` above when it is missing) |

`sbatch --test-only` is the definitive check: it fails with Slurm's reason (for example
"Invalid qos specification" or "Invalid account or account/partition combination") when you may
not use that partition, QOS, account or GPU model.

## 2. Nodes and GPUs

| Question | Command |
| --- | --- |
| Every node of a partition: state, GPUs, GPUs in use, CPUs, memory | `sinfo -N -p coc-gpu -O "NodeList:24,StateCompact:8,Gres:28,GresUsed:34,CPUsState:16,Memory:9"` |
| Which GPU models exist where | `sinfo -o "%.12P %G" \| sort -u` |
| Idle nodes only | `sinfo -p coc-gpu,ice-gpu,pace-gpu -t idle,mix -N -o "%N %P %t %G"` |
| One node in detail (GPUs, CPUs, memory, features) | `scontrol show node <node-name>` |
| What runs on a node | `squeue -w <node-name>` |

`CPUsState` reads allocated/idle/other/total. A node in state `mix` still has free CPUs and maybe
free GPUs: `GresUsed` shows how many GPUs are taken. GPU model names in `Gres` (`gpu:h100:4` means
four H100s per node) are what `GPU_TYPE` and `--gres=gpu:<model>:1` take.

## 3. Choosing partitions for this repository's jobs

`plan.sh` reads these settings (defaults in `scripts/ice/slurm.sh`) and checks every listed
partition with `sbatch --test-only` before submitting. A partition that refuses the jobs is left
out, with Slurm's reason printed.

| Setting | Default | Meaning |
| --- | --- | --- |
| `PARTITION` | `coc-gpu` | GPU partition(s) for training and the optional smoke test; a comma list runs each job on whichever listed partition has a free GPU first |
| `QOS` | `coc-ice` | QOS of the GPU jobs |
| `CPU_PARTITION` | `ice-cpu` | partition(s) of the data job |
| `CPU_QOS` | same as `QOS` | QOS of the data job |
| `ACCOUNT` | your default | `sbatch -A` |
| `GPU_TYPE` | any | one GPU model, e.g. `h100`, `h200`, `a100` |
| `SMOKE` | `0` | set to `1` to queue a GPU smoke test before training |

```bash
NAME=swin_daily bash scripts/ice/plan.sh                                      # coc-gpu, coc-ice
NAME=swin_daily PARTITION=coc-gpu,ice-gpu bash scripts/ice/plan.sh            # first free GPU of two
NAME=swin_h100 PARTITION=coc-gpu,ice-gpu GPU_TYPE=h100 bash scripts/ice/plan.sh
NAME=swin_pace PARTITION=pace-gpu QOS=<its qos> CPU_QOS=coc-ice bash scripts/ice/plan.sh
```

One job carries one QOS. Partitions that need different QOS (check the QOS column of
`partitions.sh`) therefore cannot share one `PARTITION` list; run a separate plan for each.
Single jobs take the same choices on the `sbatch` command line, which overrides the script's
`#SBATCH` lines:

```bash
NAME=swin_daily sbatch -p coc-gpu,ice-gpu,pace-gpu --qos=coc-ice scripts/ice/train.sbatch
sbatch -p ice-cpu --qos=coc-ice scripts/ice/data.sbatch
```

## 4. Watching jobs

| Question | Command |
| --- | --- |
| Everything at once (jobs, run states, latest logs) | `bash scripts/ice/status.sh` |
| My jobs | `squeue -u $USER -o "%.10i %.20j %.9P %.8T %.10M %.12l %.24R"` |
| When will my pending jobs start? | `squeue -u $USER --start` |
| Why is a job pending, and where did it run? | `scontrol show job <jobid>` (Reason, Partition, NodeList) |
| How busy is a partition? | `squeue -p coc-gpu -t PENDING \| wc -l` and `squeue -p coc-gpu -t RUNNING \| wc -l` |
| Finished jobs today: state, time, memory, exit code | `sacct -u $USER -S today --format=JobID,JobName%20,Partition,State,Elapsed,MaxRSS,ExitCode` |
| How efficiently a finished job used its CPUs and memory | `seff <jobid>` |
| Follow a log | `tail -f logs/fp-<name>-<jobid>.out` |
| My scheduling priority and fair share | `sprio -u $USER` and `sshare -u $USER` |

Common pending reasons in the last column of `squeue`:

| Reason | Meaning |
| --- | --- |
| `Priority`, `Resources` | waiting its turn, or for a free node; nothing to fix |
| `Dependency` | waiting for an earlier job of the chain (`plan.sh`) |
| `DependencyNeverSatisfied` | the job it waits for failed, e.g. the smoke test; `scancel` it |
| `QOSMaxGRESPerUser`, `AssocGrpGRES` | you already use the most GPUs your QOS allows; it starts when one ends |
| `ReqNodeNotAvail` | the nodes are reserved, often for maintenance |
| `PartitionTimeLimit` | `-t` is longer than the partition allows (`scontrol show partition`) |

## 5. Cancelling

```bash
scancel <jobid>              # one job
scancel -n fp-swin_daily     # every job with that name (a whole training chain)
scancel -u $USER -t PENDING  # all of my pending jobs
scancel -u $USER             # all of my jobs
```

## 6. Interactive GPU session

For trying code by hand on a GPU node:

```bash
salloc -p coc-gpu --qos=coc-ice --gres=gpu:1 -c 8 --mem=32G -t 1:00:00
hostname                         # a compute node; if it is still a login node, run: srun --pty bash
nvidia-smi
source scripts/ice/env.sh        # this repository's Python environment
exit                             # give the node back
```

## 7. Python environment and tests

`setup.sh` builds the environment in `~/scratch/envs/firepred`. By default it is a venv from ICE's
own `python3` (Python 3.9) with the paper's torch 2.3.1 and MONAI 1.3.2, which the golden tests were
captured with. Setup checks library imports and does not run pytest. Training plans skip the
separate GPU smoke test by default, so training can start as soon as its data is ready.

```bash
bash scripts/ice/setup.sh                        # venv from python3, paper library versions
PYTHON=/path/to/python3.12 bash scripts/ice/setup.sh
CONDA_PYTHON=3.12 bash scripts/ice/setup.sh      # conda environment from ICE's mamba/anaconda3 module
STACK=latest bash scripts/ice/setup.sh           # newest torch and MONAI for that Python
NAME=swin_daily SMOKE=1 bash scripts/ice/plan.sh  # optional GPU smoke test before training
sbatch scripts/ice/test.sbatch                   # optional: the whole test suite on a CPU node
SLOW=1 sbatch scripts/ice/test.sbatch            # also the slow end-to-end runs
```

To rebuild from scratch, delete `~/scratch/envs/firepred` first; `setup.sh` otherwise reuses it.
`python -V` and `pip list` inside `source scripts/ice/env.sh` show what is installed.

## 8. Storage

| Question | Command |
| --- | --- |
| My quotas and usage (home is small, scratch is large) | `pace-quota` |
| What takes the space | `du -sh ~/scratch/* \| sort -h` |

Data, environments and runs live in `~/scratch` (`scripts/ice/env.sh`).
