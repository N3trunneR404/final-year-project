# Fabric Digital Twin Simulator

A self-contained digital-twin and planning sandbox for experimenting with job placement, scheduling policies, failure scenarios, and performance analytics. The repository bundles a Flask API, console planner, browser dashboard, Monte-Carlo tooling, chaos engine, and optional Docker launcher so you can drive closed-loop simulations end-to-end on a laptop or workstation.

## Table of contents
1. [Fresh machine setup](#fresh-machine-setup)
2. [Repository layout](#repository-layout)
3. [Core runtime services](#core-runtime-services)
4. [Digital twin lifecycle and telemetry ingestion](#digital-twin-lifecycle-and-telemetry-ingestion)
5. [Job descriptors and submission formats](#job-descriptors-and-submission-formats)
6. [Synthetic nodes and topology assets](#synthetic-nodes-and-topology-assets)
7. [Workload planning workflows](#workload-planning-workflows)
8. [Chaos and fault-injection experiments](#chaos-and-fault-injection-experiments)
9. [Monte-Carlo and policy evaluation](#monte-carlo-and-policy-evaluation)
10. [Experiment reproduction and digital twin validation](#experiment-reproduction-and-digital-twin-validation)
11. [Docker-based fabric emulation](#docker-based-fabric-emulation)
12. [Makefile shortcuts](#makefile-shortcuts)
13. [Running tests and quality checks](#running-tests-and-quality-checks)
14. [Troubleshooting](#troubleshooting)

## Fresh machine setup
Follow these steps on a clean Ubuntu, Debian, or macOS host. Replace `apt` commands with your platform's package manager when necessary.

1. **Install system dependencies**
   ```bash
   # Ubuntu/Debian example
   sudo apt update
   sudo apt install -y python3 python3-venv python3-pip git
   # Optional: Docker engine if you intend to run container-based fabrics
   curl -fsSL https://get.docker.com | sudo sh
   ```

2. **Clone the repository**
   ```bash
   git clone https://github.com/<your-org>/final-year-project.git
   cd final-year-project
   ```

3. **Create an isolated Python environment and install requirements**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

   The `requirements.txt` file bundles Flask, PyYAML, requests, rich, jsonschema, numpy/pandas/matplotlib, Docker SDK (optional), and developer tools such as black, ruff, and pytest.

4. **(Optional) Use the Makefile wrapper**
   The command below creates a virtual environment in `.venv`, installs dependencies, and prints reminders for activating the environment.
   ```bash
   make install
   ```

## Repository layout
```
.
├── dt/            # Core state machine, cost model, reservation engine, policies, API
├── planner/       # CLI clients for submitting jobs locally or to a remote API
├── sim/           # Synthetic node generator, chaos engine, Monte-Carlo runner
├── tools/         # Validation, reporting, and export utilities
├── ui/            # Flask dashboard for monitoring and manual control
├── nodes/         # Sample node descriptors (YAML) produced by sim/gen_nodes.py
├── jobs/          # Example job definitions (YAML/JSON)
├── schemas/       # JSON Schemas shared by validators and generators
├── fabric_docker/ # Optional helper for launching containerised testbeds
└── Makefile       # Convenience targets (install, run-api, plan, chaos, ...)
```

## Core runtime services
Once dependencies are installed and the virtual environment is active, you can launch the core components individually or together.

### Digital Twin API (`dt/api.py`)
Exposes `/snapshot`, `/plan`, `/plan_batch`, `/release`, and `/observe` endpoints while keeping node/link state, reservations, and overrides in sync.
```bash
python -m dt.api --host 0.0.0.0 --port 8080 --debug
```
**Flags**
- `--host`: Bind address (default `127.0.0.1`). Use `0.0.0.0` for LAN access.
- `--port`: HTTP port (default `8080`).
- `--debug`: Enables Flask debug/auto-reload.

### Dashboard UI (`ui/dashboard.py`)
A single-file Flask app that renders topology maps, job history, chaos overrides, and provides a JSON editor for submitting plans.
```bash
python -m ui.dashboard --host 0.0.0.0 --port 8090 --remote http://127.0.0.1:8080
```
**Flags**
- `--remote`: Base URL for a running API (`http://127.0.0.1:8080` by default when using `make run-ui`). Omit the value (e.g., `FABRIC_DT_REMOTE=`) to use the dashboard's embedded state.
- `--remote-timeout`: Seconds to wait for API responses (default `5`).
- `--debug`: Enable Flask debug mode.

Once the dashboard loads you can:
- Inspect the **Overview** card for total capacity, energy and SLO hit summaries sourced from `/snapshot`.
- Drill into the **Nodes** and **Links** tables to confirm predictive fields such as reliability, thermal derate, and network loss rates that the API attaches under `dyn` and `effective`.
- Use the **Strategy** dropdown (greedy, resilient/federated, RL/Markov, bandit) to run the same job against multiple planners and compare returned latency/energy/risk metrics side-by-side in the **Recent plans** panel.
- Paste or edit JSON descriptors in the **Job composer**, toggle `dry run` vs. `commit`, and submit to either the embedded planner or a remote API.
- Trigger **Demo job** runs or **Apply observation** actions to inject failures/derates, which call the `/plan_demo` and `/observe` endpoints respectively.
- Click **Refresh snapshot** whenever you update descriptors on disk so that the latest watcher state is rendered without restarting the UI process.

### Planner CLI (`planner/run_plan.py`)
Runs the scheduling engine locally or against a remote API. Results are printed to the terminal and optionally persisted to disk.
```bash
python -m planner.run_plan --job jobs/jobs_10.yaml --strategy resilient --remote http://127.0.0.1:8080
```
**Key flags**
- `--job PATH`: Required. Accepts a single job file, a list of files, or a JSON/YAML document with a top-level `jobs` array.
- `--dry-run`: Evaluate the plan without reserving capacity.
- `--remote URL`: Call a Digital Twin API instead of running the local planner.
- `--strategy`: Choose a placement policy. Options include `greedy`, `cheapest-energy`, `bandit`, `resilient`, `network-aware`, `federated`, `balanced`, `fault-tolerant`, `rl`, `rl-markov`, and `mdp`.
- `--repeat N`: Re-run the local planner N times (useful for the bandit learner).
- `--out PATH`: Persist plan responses as JSON.

## Digital twin lifecycle and telemetry ingestion
The `dt.state.DTState` class powers both the API and dashboard. It continuously merges static descriptors, live overrides, and predictive analytics so that every plan call reflects the most recent view of the fabric.

### Loading assets and hot reloading
- Node descriptors are read from `nodes/*.yaml`; topology defaults come from `sim/topology.yaml`; runtime overrides are pulled from `sim/overrides.json`.
- File watchers poll every ~0.5s by default. Touch or replace a YAML file (for example, `cp nodes/sample.yaml nodes/sample_lab.yaml && touch nodes/sample_lab.yaml`) and the API/UI will reload it automatically—no restart is required.
- To disable background threads in automation, instantiate the state manually: `python - <<'PY'` with `DTState(auto_start_watchers=False)`. You can then call `state.start()` when ready.

### Streaming telemetry and overrides
- Push live measurements through the `/observe` endpoint. Example:
  ```bash
  curl -X POST http://127.0.0.1:8080/observe \
       -H 'Content-Type: application/json' \
       -d '{
             "action": "apply",
             "payload": {
               "type": "node",
               "node": "edge-node-3",
               "changes": {
                 "down": false,
                 "thermal_derate": 0.25,
                 "reliability": 0.82,
                 "used_cpu_cores": 3.5
               }
             }
           }'
  ```
  The same structure works for links using `{ "type": "link", "key": "node-a|node-b", "changes": {"loss_pct": 1.2} }`.
- `sim/chaos.py` writes the identical payload shape to `sim/overrides.json`. The watcher merges those deltas automatically, so long-running simulations keep parity with injected incidents.
- Retrieve the latest merged view at any time with `curl http://127.0.0.1:8080/snapshot` or by checking the **Overview** card in the dashboard.

### Self-healing, guard rails, and event history
- The API enables two background controllers by default: `SelfHealingController` (monitors reliability/availability) and `ResourceGuardian` (cleans up expired reservations). Tune their behaviour via environment variables such as `FABRIC_SELF_HEAL_INTERVAL`, `FABRIC_SELF_HEAL_RELIABILITY`, `FABRIC_GUARDIAN_INTERVAL`, and `FABRIC_GUARDIAN_UTIL` before launching `dt.api` or the dashboard.
- Lightweight noise injection can be toggled with `FABRIC_ENABLE_NOISE=1` and tuned using `FABRIC_NOISE_INTERVAL`, `FABRIC_NOISE_NODE_JITTER`, and `FABRIC_NOISE_LINK_JITTER` to emulate background variance.
- Inspect recent CloudEvents emitted by the twin—plan commits, overrides, self-heal actions—by running a short script:
  ```bash
  python - <<'PY'
  from dt.state import DTState
  state = DTState(auto_start_watchers=False)
  for evt in state.recent_events(limit=10):
      print(evt["type"], evt.get("subject"), evt["data"])
  PY
  ```

### Predictive analytics and exporters
- Every `/plan` response returns a `predictive` block populated by `dt.predict.PredictiveAnalyzer`, exposing per-node utilisation forecasts, failure windows, and link variability.
- Use the exporters in `dt/exporters.py` to share the current twin snapshot:
  ```bash
  python - <<'PY'
  import json
  from dt.state import DTState
  from dt.exporters import as_dtdl, as_k8s_crds

  state = DTState(auto_start_watchers=False)
  print(json.dumps(as_dtdl(state), indent=2))
  print(json.dumps(as_k8s_crds(state), indent=2))
  PY
  ```
- The JSON emitted above can seed Azure Digital Twins (DTDL) or bootstrap Kubernetes CRDs when mirroring the fabric into other control planes.

## Job descriptors and submission formats
Jobs can be written in YAML or JSON and validated against `schemas/job.schema.yaml`. Each document may contain shared `defaults` and a `jobs` array. The schema includes an optional top-level `version` string (recommended when exchanging descriptors between teams). A minimal JSON payload that can be POSTed to `/plan` looks like:
```json
{
  "version": "0.1.0",
  "jobs": [
    {
      "id": "job-demo",
      "deadline_ms": 4500,
      "stages": [
        {
          "id": "ingest",
          "type": "io",
          "size_mb": 40,
          "resources": { "cpu_cores": 1, "ram_gb": 1 },
          "allowed_formats": ["native", "wasm"]
        },
        {
          "id": "cnn",
          "type": "cv",
          "size_mb": 120,
          "resources": { "cpu_cores": 4, "ram_gb": 4, "vram_gb": 2 },
          "allowed_formats": ["cuda", "native"]
        }
      ],
      "placement": {
        "prefer_region": ["campus"],
        "required_fabric": ["infiniband"],
        "min_trust": 0.7
      },
      "fault_tolerance": { "replication_factor": 2 }
    }
  ]
}
```
**Field highlights**
- `defaults`: Optional job-wide settings (`qos`, `deadline_ms`, retry policy, placement hints).
- `stages`: At least one stage with `id`, `type`, and resource hints. Use `resources.cpu_cores`, `resources.ram_gb`, `resources.vram_gb`, architecture constraints, and accelerator preferences to steer placement.
- `edges`: Optional explicit DAG edges. When omitted, stages run sequentially.
- `datasets`: Name and locality metadata for inputs referenced by the job.
- `placement`: Site/region preferences, fabric requirements, trust thresholds, and architecture constraints.
- `fault_tolerance`: Replication factor, checkpointing style, retry/backoff limits.
- `policy`: Optimisation target (`latency`, `cost`, `energy`, `carbon`, `balanced`) and associated limits.

**Submitting to the API**
```bash
curl -X POST http://127.0.0.1:8080/plan \
     -H 'Content-Type: application/json' \
     -d @jobs/example_job.json
```
The API returns a JSON plan with per-stage placements, predicted utilisation, reliability, fallback recommendations, and any admission errors. Use `/plan_batch` to send a list of job documents in one request.

## Synthetic nodes and topology assets
Sample node descriptors live under `nodes/`. Generate new ones with:
```bash
python -m sim.gen_nodes --count 50 --seed 1234 --outdir nodes/generated
```
**Flags**
- `--count`: Number of node YAMLs to create.
- `--seed`: Seed for deterministic generation.
- `--outdir`: Output directory (will be created if missing).

Topologies (link, latency, and chaos schedules) are stored in `sim/topology.yaml` and validated by `schemas/topology.schema.yaml`. Update or extend these files before running chaos or Monte-Carlo experiments.

## Workload planning workflows
Common flows combine the planner, API, and dashboard:
1. Start the API: `make run-api` (or `python -m dt.api`).
2. Optionally launch the dashboard: `make run-ui`.
3. Dry-run plans locally: `python -m planner.run_plan --job jobs/jobs_10.yaml --dry-run`.
4. Commit reservations through the API: `python -m planner.run_plan --remote http://127.0.0.1:8080 --job jobs/jobs_10.yaml`.
5. Generate synthetic workloads: `python -m sim.montecarlo --synthetic 100 --synthetic-profiles cpu,gpu --out reports/montecarlo.csv`.

The planner supports live strategy switching via `--strategy` or through the dashboard selector. The resilient and federated planners compute redundant placements, network-aware scoring, and fallback coverage, while the `rl-markov` policy taps into `dt/policy/mdp.py` for reinforcement-learned decisions.

## Chaos and fault-injection experiments
Drive deterministic or accelerated failure scenarios against the digital twin.
```bash
python -m sim.chaos --topology sim/topology.yaml --scenario link-fail --speed 20 --run
```
**Flags**
- `--topology`: Topology YAML with named scenarios.
- `--scenario`: Scenario to execute (`campus_edge_failover`, `multi_federation_surge`, etc.).
- `--speed`: Time compression factor (e.g., `20` runs the schedule at 20×).
- `--dt`: Optional observe endpoint (`http://127.0.0.1:8080/observe`) for posting overrides directly.
- `--overrides`: Where to write `sim/overrides.json` when running offline.
- `--nodes`: Alternative node descriptor directory.
- `--dry-run`: Print the schedule without applying it.
- `--run`: Apply chaos events (writes overrides and optionally hits `/observe`).

The API automatically watches `sim/overrides.json` and merges failures, latency shifts, and derates into the live state so subsequent plans reflect the injected conditions.

## Monte-Carlo and policy evaluation
Use the simulation tooling under `sim/` and `tools/` to reproduce the experiments referenced in project reports.

### Monte-Carlo planner sweeps (`sim/montecarlo.py`)
```bash
python -m sim.montecarlo --nodes nodes/ --jobs jobs/jobs_10.yaml --trials 100 --seed 42 --out reports/montecarlo.csv
```
**Flags**
- `--nodes`: Node descriptor directory.
- `--jobs`: Job YAML/JSON file.
- `--synthetic N`: Generate N random jobs instead of reading a file (`--synthetic-max-stages` and `--synthetic-profiles` control shape and resource mix).
- `--topology`: Topology YAML for link perturbations.
- `--trials`: Number of Monte-Carlo runs.
- `--seed`: RNG seed.
- `--dt`: Remote `/plan` endpoint to offload planning to the API.
- `--out`: CSV output path (includes per-trial latency, energy, risk, reliability, and infeasibility statistics).

### Policy benchmark and reporting (`tools/policy_benchmark.py`)
```bash
python -m tools.policy_benchmark --jobs jobs/jobs_10.yaml --strategies greedy resilient rl-markov --limit 10 --out reports/policy_metrics.png --json-out reports/policy_metrics.json
```
The script samples jobs (up to `--limit`), runs each strategy, and emits a plot plus an optional JSON dump. Ensure `matplotlib` is installed (it is included in `requirements.txt`). Additional utilities include:
- `python -m tools.export_csv --inputs reports/montecarlo.csv --outdir reports/summary`
- `python -m tools.summarize_nodes --dir nodes/ --md reports/nodes.md`
- `python -m tools.validate_nodes --dir nodes/ --apply-defaults`

## Experiment reproduction and digital twin validation
Recreate the evaluation suite from the project report and quantify how closely the digital twin mirrors injected scenarios.

### 1. Prepare nodes, jobs, and topology assets
- Generate fresh hardware descriptors if needed:
  ```bash
  python -m sim.gen_nodes --count 64 --seed 2024 --outdir nodes/exp_run
  cp -r nodes/exp_run/*.yaml nodes/
  ```
- Review `sim/topology.yaml` to ensure link speeds and chaos scenarios match the experiment you want to reproduce.
- Curate the workload set (JSON or YAML) under `jobs/`. For synthetic mixes, run `python -m sim.montecarlo --synthetic 0` once to confirm schema expectations and reuse the generated catalog.

### 2. Start the digital twin stack
- Launch the API: `FABRIC_SELF_HEAL_INTERVAL=5 FABRIC_ENABLE_NOISE=0 make run-api`.
- Optionally open the dashboard with `FABRIC_DT_REMOTE=http://127.0.0.1:8080 make run-ui` so you can watch node health, overrides, and plan outcomes in real time.
- Verify the environment is consistent by grabbing an initial baseline snapshot:
  ```bash
  curl -s http://127.0.0.1:8080/snapshot | jq '.nodes | length'
  ```

### 3. Capture baseline planner outputs
- Dry-run the target strategies and persist the JSON responses (these include predictive telemetry and scoring rationales):
  ```bash
  mkdir -p reports/baselines
  python -m planner.run_plan --job jobs/jobs_10.yaml --strategy greedy --out reports/baselines/plan_greedy.json --dry-run
  python -m planner.run_plan --job jobs/jobs_10.yaml --strategy resilient --out reports/baselines/plan_resilient.json --dry-run
  python -m planner.run_plan --job jobs/jobs_10.yaml --strategy rl-markov --out reports/baselines/plan_rl.json --dry-run
  ```
- Each saved file contains a `predictive` block along with assigned nodes. Keep these artefacts for side-by-side comparison with Monte-Carlo ground truth.

### 4. Run Monte-Carlo sweeps with and without the twin
- Digital twin–driven placement:
  ```bash
  python -m sim.montecarlo --nodes nodes/ --jobs jobs/jobs_10.yaml \
         --topology sim/topology.yaml --trials 200 --seed 42 \
         --dt http://127.0.0.1:8080/plan --out reports/montecarlo_dt.csv
  ```
- Fallback planner only (no `/plan` calls) to produce a control run:
  ```bash
  python -m sim.montecarlo --nodes nodes/ --jobs jobs/jobs_10.yaml \
         --topology sim/topology.yaml --trials 200 --seed 42 \
         --out reports/montecarlo_greedy.csv
  ```
- If you are stress-testing chaos scenarios, pass `--synthetic N` or point `--topology` at the scenario you injected with `sim.chaos` so perturbations align.

### 5. Summarise metrics and compute deltas
- Convert Monte-Carlo CSVs into comparable summaries:
  ```bash
  python -m tools.export_csv \
         --inputs reports/montecarlo_dt.csv reports/montecarlo_greedy.csv \
         --labels dt greedy --outdir reports/summary --write-merged
  ```
- Inspect `reports/summary/summary.csv` for mean, p95/p99 latencies, and SLO hit-rate per run. The `merged.csv` file is convenient for plotting custom charts in notebooks.
- Combine the planner JSON artefacts with Monte-Carlo results to quantify drift. For example, calculate the difference between predicted latency in `plan_resilient.json` and the empirical p95 in the exported summary.

### 6. Validate interactively in the dashboard
- In the dashboard, load the same job under **Job composer**, submit it with each strategy, and confirm the **Recent plans** table mirrors the JSON stored in `reports/baselines/`.
- Apply a failure using **Apply observation** or run `python -m sim.chaos --scenario ... --run` and watch the predictive scores change (higher projected derate, lower reliability) before repeating the Monte-Carlo sweep.
- Export the twin state with the snippet in [Predictive analytics and exporters](#predictive-analytics-and-exporters) to archive the exact topology used for the experiment.

## Docker-based fabric emulation
Launch lightweight containers that mirror the generated node descriptors.
```bash
python -m fabric_docker.launch_fabric --nodes nodes/ --topology sim/topology.yaml --network fabric-net --image ghcr.io/your-org/fabric-node:latest --prefix fabric
```
**Flags**
- `--image`: Base container image.
- `--image-arch ARCH=REF`: Override images for specific architectures (`arm64=ghcr.io/...`). Repeatable.
- `--network`: Docker bridge network (created if missing).
- `--tc`: Traffic shaping mode (`none` or `container` via `tc` commands).
- `--force-arch`: Ignore host vs node architecture mismatches (use with caution).
- `--out`: Write the container mapping to a JSON file for teardown scripts.

## Makefile shortcuts
The Makefile wraps the most common flows while ensuring the virtual environment is active.
```bash
make install          # create venv and install requirements
make run-api          # start the Digital Twin API on :8080
make run-ui           # launch the dashboard on :8090 and point it at the API
make plan             # dry-run greedy planner for jobs/jobs_10.yaml
make demo NUM=25      # fire 25 random jobs with configurable worker/QPS knobs
make chaos            # execute the default chaos scenario from sim/topology.yaml
make policy-benchmark # render reports/policy_metrics.png comparing planners
```
All targets honour optional variables. Example: `make demo NUM=50 WORKERS=8 QPS=5.0 DRY=0` to send live reservations.

## Running tests and quality checks
Activate the virtual environment before running tests or linters.
```bash
pytest                 # run unit tests under tests/
ruff check .           # static analysis
black --check .        # formatting guard
isort --check-only .   # import ordering
```
The `pytest` suite exercises the API snapshot and planning endpoints to confirm end-to-end behaviour.

## Troubleshooting
- **Missing modules (e.g., matplotlib, numpy)**: Ensure the virtual environment is activated and `pip install -r requirements.txt` has completed successfully.
- **Port collisions**: Pass `--port` flags to `dt.api` or `ui.dashboard` to avoid conflicts.
- **Dashboard cannot reach the API**: Verify `FABRIC_DT_REMOTE` or the `--remote` flag matches the API host/port and that CORS/network firewalls allow access.
- **Chaos overrides not applied**: Confirm `sim/overrides.json` exists and the API process logs file watch events. Use `--dt http://127.0.0.1:8080/observe` when running the chaos engine to push updates directly.

You now have a full-stack digital-twin environment capable of simulating heterogeneous fabrics, validating scheduling policies, injecting failures, and exporting analytics for reports or publications.
