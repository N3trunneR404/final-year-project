# Fabric Digital Twin Simulator

A self-contained digital-twin and planning sandbox for experimentation with job
placement, scheduling policies, and chaos scenarios. The repository bundles a
Flask API, a console planner, a browser dashboard, Monte-Carlo tooling, and a
chaos engine so you can drive closed-loop simulations end-to-end.

## Project layout

```
.
├── dt/            # Core state machine, cost model, reservation engine, policies
├── planner/       # CLI clients for submitting jobs locally or to a remote API
├── sim/           # Synthetic node generator, chaos engine, Monte-Carlo runner
├── tools/         # Validation, reporting, and export utilities
├── ui/            # Flask dashboard for monitoring and manual control
├── nodes/         # Sample node descriptors (YAML) produced by sim/gen_nodes.py
├── jobs/          # Example job definitions
├── schemas/       # JSON Schemas shared by validators and generators
└── Makefile       # Convenience targets (install, run-api, plan, chaos, ...)
```

## Prerequisites

* Python 3.10 or newer
* (Optional) Docker daemon if you want to experiment with `fabric_docker/launch_fabric.py`

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The Makefile wraps these steps under `make install` and keeps all commands in the
virtual environment for you.

## Running the closed-loop simulator

1. **Start the Digital Twin API**
   ```bash
   make run-api            # exposes http://127.0.0.1:8080
   ```
   The API loads the node YAMLs, watches for chaos overrides, and exposes
   endpoints such as `/snapshot`, `/plan`, `/plan_batch`, `/release`, and `/observe`.

2. **(Optional) Launch the dashboard**
   ```bash
   make run-ui             # exposes http://127.0.0.1:8090
   ```
   The dashboard can submit demo jobs, inspect snapshots, and apply ad-hoc
   observations to the running simulation. The “Fabric Topology” card renders a
   live D3 map of nodes and links so you can watch reservations, thermal
   derates, and chaos-induced failures propagate in real time. Federation tables
   summarise per-fabric load, down hosts, and cross-federation link health while
   the plan history surfaces fallback coverage so you can spot weak failover
   paths. Updates now stream in every two seconds with animated node/link
   transitions, the job library lists every YAML under `jobs/`, and the planner
   form preloads whichever job you select from the catalog before you tweak it
   in the JSON editor. When launched via `make run-ui` it now points at the API
   from step 1 (`http://127.0.0.1:8080`) by default so the UI reflects real
   reservations and chaos events. To fall back to the dashboard's embedded
   state, explicitly run `FABRIC_DT_REMOTE= python -m ui.dashboard` (note the
   empty value before the command).

   To point the dashboard at a remote API instance instead of its embedded
   state, export `FABRIC_DT_REMOTE=http://127.0.0.1:8080` (or pass
   `--remote http://127.0.0.1:8080` to `python -m ui.dashboard`). The UI will
   proxy snapshot, plan, and observe requests to the API and surface the live
   plan history collected on the server.

3. **Submit workload plans**
   * Local greedy planner:
     ```bash
     make plan                                 # dry-run plan for jobs/jobs_10.yaml
     python -m planner.run_plan --job jobs/jobs_10.yaml --dry-run
     ```
     The dry-run variant never reserves capacity, so the dashboard snapshot
     remains unchanged. Omit `--dry-run` (or run `make demo DRY=0`) when you
     want to see the DT consume resources.
   * Remote planner talking to the API:
     ```bash
     python -m planner.run_plan --remote http://127.0.0.1:8080 --job jobs/jobs_10.yaml --dry-run
     python -m planner.run_plan --remote http://127.0.0.1:8080 --job jobs/jobs_10.yaml
     ```
     The first example performs a dry run; the second commits reservations so
     the topology view glows cyan around the nodes assigned to the job.
   * Fire randomized demo jobs (records JSON/CSV under `plans/`):
     ```bash
     make demo NUM=25 WORKERS=4 QPS=2.0
     ```
  Planner strategies can be swapped at runtime via `--strategy` or the
  dashboard selector. In addition to the original greedy and
  `cheapest-energy` modes, the CLI and UI surface `bandit` (greedy placement
  with a UCB1 format bandit), `rl-markov` (the Markov decision-process
  planner backed by the reinforcement learner), and the network-aware
  `resilient`/`federated` family from
  `dt/policy/resilient.FederatedPlanner`. Every plan response now carries the
  predictive snapshot (per-node utilisation trends, availability windows, and
  reliability), the average reliability observed across stages, and explicit
  fallback recommendations when redundancy is requested. The dashboard’s job
  catalog includes a “Plan entire catalog” shortcut so you can replay every
  YAML with whichever policy you are comparing and immediately inspect the
  resulting reliability plots in the UI.

4. **Inject chaos and observe feedback**
   ```bash
   make chaos                     # executes the schedule in sim/topology.yaml
   make chaos SCENARIO=link-fail  # choose a named scenario from topology.yaml
   make chaos SCENARIO=campus_edge_failover
   make chaos SCENARIO=multi_federation_surge
   ```
   The chaos engine writes overrides to `sim/overrides.json` (and optionally
   posts them to `/observe`). Group events such as `zone_blackout` and
   `federation_partition` will coordinate outages across sets of nodes or
   federations so you can validate the federation-aware planner. The DT watcher
   thread merges the overrides into the live state so subsequent plans reflect
   the new conditions while the predictive model updates reliability, link
   latency P95 estimates, and availability windows that flow straight into the
   planners and dashboard.

5. **Analyse outcomes**
   * `sim/montecarlo.py` perturbs nodes/links and repeatedly plans jobs.
   * `tools/export_csv.py` summarises Monte-Carlo CSVs (requires numpy/pandas).
   * `tools/policy_benchmark.py` evaluates one or more planner strategies over
     a job corpus (sampling the first few jobs by default for speed) and emits
     publication-ready plots comparing latency, energy, risk, reliability, and
     infeasibility. Run `make policy-benchmark` to generate
     `reports/policy_metrics.png` and a matching JSON dump for your paper.
   * `tools/summarize_nodes.py` and `tools/validate_nodes.py` help curate inputs.

## Key components

* `dt/state.py` – thread-safe runtime that keeps nodes, links, and reservations
  in sync while watching for filesystem overrides.
* `dt/cost_model.py` – deterministic latency, energy, risk, and reliability
  estimators used by both the API and the planner.
* `dt/predict.py` – lightweight predictive telemetry (EWMA + trends) for node
  availability, reliability, and link variability.
* `dt/events.py` – in-memory CloudEvents bus surfaced via the API.
* `dt/exporters.py` – renders DTDL models and Kubernetes CRDs for external
  tooling.
* `dt/policy/greedy.py` – baseline planner that scans feasible nodes, optionally
  collaborates with a bandit format selector, and performs reservations.
* `dt/policy/resilient.py` – network- and federation-aware planner that scores
  candidates for risk, load, link loss, and redundancy to emit primary and
  fallback placements.
* `dt/policy/mdp.py` – Markov decision process planner with an embedded
  reinforcement learner that scores stages using a weighted Tchebycheff
  objective and returns fallback-aware assignments plus learned Q-values.
* `planner/run_plan.py` – CLI wrapper that can call the local planner or a
  remote API, summarising results in the terminal.
* `sim/chaos.py` – builds and executes chaos schedules against the DT by writing
  overrides or posting to `/observe`.
* `ui/dashboard.py` – single-file Flask UI for visualising state and submitting
  jobs interactively.

## Validation & tooling

* `make validate-nodes` – JSON Schema validation for every file under `nodes/`.
* `make summarize-nodes` – prints aggregates and optionally exports CSV/JSON.
* `make montecarlo` – runs Monte-Carlo simulations (`TRIALS=500` to override).
* `python -m dt.validators` – standalone schema validation CLI.

## Development tips

* Use `make format` / `make lint` to run Black, isort, and Ruff (optional tools
  from `requirements.txt`).
* Planner and dashboard persist learning state under `sim/bandit_state.json` and
  `sim/rl_state.json`. Delete them to reset exploration.
* To extend schemas or cost modelling, update the files under `schemas/` and the
  relevant modules under `dt/`.

With these pieces in place you can loop: generate nodes, schedule workloads,
perturb the environment, and inspect outcomes – all inside one repository.

## Real-time telemetry and standards exports

* `GET /events?limit=100` streams recent CloudEvents (node updates, link
  changes, reservations) so you can feed dashboards or tracing systems.
* `GET /standards/dtdl` emits a Digital Twin Definition Language model of the
  current fleet, suitable for importing into Azure Digital Twins or other DTDL
  aware tooling.
* `GET /standards/crds` exports Kubernetes `NodeTwin`/`LinkTwin` custom
  resources so you can mirror the simulated state in a cluster.
* Every `/plan` response includes `predictive` telemetry (utilisation trends,
  reliability, availability) and fallback assignments, letting you reason about
  proactive failover and churn-resilience directly from the API.
