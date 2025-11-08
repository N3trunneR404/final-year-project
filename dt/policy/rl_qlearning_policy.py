#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dt/policy/qlearning.py — Q-Learning based planner for Fabric DT.

What it does
------------
- Learns optimal node assignments through Q-learning
- Maintains Q-table: Q(state, action) = expected future reward
- State: encodes stage requirements, node capacities, previous placement
- Action: choosing a node for current stage
- Reward: based on latency reduction, energy efficiency, successful completion

Key API
-------
planner = QLearningPlanner(state, cost_model, cfg=None)
result  = planner.plan_job(job, dry_run=False, train=True)
planner.save_model(path)
planner.load_model(path)

Configuration
-------------
- learning_rate: α for Q-value updates (default: 0.1)
- discount_factor: γ for future rewards (default: 0.9)
- epsilon: exploration rate (default: 0.1)
- epsilon_decay: decay rate per episode (default: 0.995)
- epsilon_min: minimum exploration (default: 0.01)
"""
from __future__ import annotations

import json
import os
import hashlib
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
import random

try:
    from dt.state import DTState, safe_float
    from dt.cost_model import CostModel, merge_stage_details
except Exception:
    DTState = object  # type: ignore
    CostModel = object  # type: ignore
    def safe_float(x: Any, d: float = 0.0) -> float:
        try:
            return float(x)
        except Exception:
            return d
    def merge_stage_details(primary, cost):  # type: ignore
        return (cost or []) or (primary or [])


DEFAULT_CFG = {
    # Q-Learning parameters
    "learning_rate": 0.1,        # alpha - how much to update Q-values
    "discount_factor": 0.9,      # gamma - importance of future rewards
    "epsilon": 0.1,              # epsilon - exploration rate
    "epsilon_decay": 0.995,      # decay epsilon after each episode
    "epsilon_min": 0.01,         # minimum exploration rate
    
    # Reward shaping
    "latency_penalty_weight": 1.0,    # penalty per ms
    "energy_penalty_weight": 0.1,     # penalty per kJ
    "risk_penalty_weight": 100.0,     # penalty for risk
    "success_reward": 1000.0,         # bonus for feasible plan
    "failure_penalty": -500.0,        # penalty for infeasible plan
    
    # Fallback to greedy
    "greedy_fallback": True,          # use greedy if no valid Q-action
    "risk_weight": 10.0,              # for greedy fallback
    "energy_weight": 0.0,             # for greedy fallback
}


def _hash_state(state_features: Dict[str, Any]) -> str:
    """It like creates a hash key for Q-table from state features."""
    # Discretize continuous values for stable state representation
    discrete = {
        "stage_cpu": round(state_features.get("stage_cpu", 0), 1),
        "stage_mem": round(state_features.get("stage_mem", 0), 1),
        "stage_gpu": round(state_features.get("stage_gpu", 0), 1),
        "stage_size": round(state_features.get("stage_size", 0), 0),
        "prev_node": state_features.get("prev_node", "none"),
        "stage_idx": state_features.get("stage_idx", 0),
        "total_stages": state_features.get("total_stages", 1),
    }
    s = json.dumps(discrete, sort_keys=True)
    return hashlib.md5(s.encode()).hexdigest()[:16]


def _extract_state_features(
    stage: Dict[str, Any],
    stage_idx: int,
    total_stages: int,
    prev_node: Optional[str],
) -> Dict[str, Any]:
    """Does woodoo magecraft to extract features from current stage for state representation."""
    res = stage.get("resources") or {}
    return {
        "stage_cpu": safe_float(res.get("cpu_cores"), 0.0),
        "stage_mem": safe_float(res.get("mem_gb"), 0.0),
        "stage_gpu": safe_float(res.get("gpu_vram_gb"), 0.0),
        "stage_size": safe_float(stage.get("size_mb"), 10.0),
        "prev_node": prev_node or "none",
        "stage_idx": stage_idx,
        "total_stages": total_stages,
    }


def _fits(state: DTState, node: Dict[str, Any], stage: Dict[str, Any]) -> bool:
    """something something checks if stage can fit on node (capacity check)."""
    if (node.get("dyn") or {}).get("down", False):
        return False
    caps = state._effective_caps(node)
    res = stage.get("resources") or {}
    need_cpu  = safe_float(res.get("cpu_cores"), 0.0)
    need_mem  = safe_float(res.get("mem_gb"), 0.0)
    need_vram = safe_float(res.get("gpu_vram_gb"), 0.0)
    if caps["free_cpu_cores"] + 1e-9 < need_cpu:  return False
    if caps["free_mem_gb"]   + 1e-9 < need_mem:   return False
    if caps["free_gpu_vram_gb"] + 1e-9 < need_vram: return False
    return True


class QLearningPlanner:
    """Main sauce: Q-Learning based job planner with exploration/exploitation."""
    
    def __init__(
        self,
        state: DTState,
        cost_model: CostModel,
        cfg: Optional[Dict[str, Any]] = None,
        model_path: Optional[str] = None,
    ):
        self.state = state
        self.cm = cost_model
        self.cfg = {**DEFAULT_CFG, **(cfg or {})}
        
        # Q-table: Q[state_hash][node_name] = expected reward
        self.q_table: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        
        # Training statistics
        self.epsilon = float(self.cfg["epsilon"])
        self.episodes_trained = 0
        self.total_reward_history: List[float] = []
        
        # Load existing model if provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def _get_feasible_nodes(
        self,
        stage: Dict[str, Any]
    ) -> List[str]:
        """Gets list of nodes that can fit the stage."""
        feasible = []
        for name, node in self.state.nodes_by_name.items():
            if _fits(self.state, node, stage):
                feasible.append(name)
        return feasible
    
    def _select_action(
        self,
        state_hash: str,
        feasible_nodes: List[str],
        explore: bool = True,
    ) -> str:
        """Selects node using epsilom-greedy policy."""
        if not feasible_nodes:
            return None
        
        # Exploration: random choice
        if explore and random.random() < self.epsilon:
            return random.choice(feasible_nodes)
        
        # Exploitation: choose best Q-value
        q_values = self.q_table[state_hash]
        
        # Get Q-values for feasible nodes
        feasible_q = [(node, q_values.get(node, 0.0)) for node in feasible_nodes]
        
        if not feasible_q:
            return random.choice(feasible_nodes)
        
        # Choose node with highest Q-value (break ties randomly)
        max_q = max(q for _, q in feasible_q)
        best_nodes = [node for node, q in feasible_q if q == max_q]
        return random.choice(best_nodes)
    
    def _compute_reward(
        self,
        stage: Dict[str, Any],
        node_name: str,
        prev_node: Optional[str],
        success: bool,
    ) -> float:
        """Computes immediate reward for assigning stage to node."""
        if not success:
            return float(self.cfg["failure_penalty"])
        
        node = self.state.nodes_by_name[node_name]
        
        # Computes costs
        comp_ms = self.cm.compute_time_ms(stage, node)
        xfer_ms = 0.0 if prev_node in (None, node_name) else self.cm.transfer_time_ms(
            prev_node, node_name, safe_float(stage.get("size_mb"), 10.0)
        )
        energy = self.cm.energy_kj(stage, node, comp_ms)
        risk = self.cm.risk_score(stage, node)
        
        # Reward is negative cost (want to minimize)
        reward = (
            -comp_ms * float(self.cfg["latency_penalty_weight"])
            - xfer_ms * float(self.cfg["latency_penalty_weight"])
            - energy * float(self.cfg["energy_penalty_weight"])
            - risk * float(self.cfg["risk_penalty_weight"])
        )
        
        return reward
    
    def _update_q_value(
        self,
        state_hash: str,
        action: str,
        reward: float,
        next_state_hash: Optional[str],
        next_feasible_nodes: List[str],
    ):
        """Updates Q-value using Q-learning update rule."""
        alpha = float(self.cfg["learning_rate"])
        gamma = float(self.cfg["discount_factor"])
        
        # Current Q-value
        current_q = self.q_table[state_hash][action]
        
        # Max future Q-value
        if next_state_hash and next_feasible_nodes:
            next_q_values = [self.q_table[next_state_hash].get(n, 0.0) 
                           for n in next_feasible_nodes]
            max_next_q = max(next_q_values) if next_q_values else 0.0
        else:
            max_next_q = 0.0
        
        # Q-learning update: Q(s,a) ← Q(s,a) + alpha[r + gamma max Q(s',a') - Q(s,a)]
        new_q = current_q + alpha * (reward + gamma * max_next_q - current_q)
        self.q_table[state_hash][action] = new_q
    
    def _greedy_fallback(
        self,
        stage: Dict[str, Any],
        feasible_nodes: List[str],
        prev_node: Optional[str],
    ) -> Optional[str]:
        """Fallback to greedy selection if Q-learning has no learned values."""
        if not feasible_nodes:
            return None
        
        best_node = None
        best_score = float("inf")
        
        for node_name in feasible_nodes:
            node = self.state.nodes_by_name[node_name]
            comp_ms = self.cm.compute_time_ms(stage, node)
            xfer_ms = 0.0 if prev_node in (None, node_name) else self.cm.transfer_time_ms(
                prev_node, node_name, safe_float(stage.get("size_mb"), 10.0)
            )
            energy = self.cm.energy_kj(stage, node, comp_ms)
            risk = self.cm.risk_score(stage, node)
            
            score = (
                comp_ms + xfer_ms +
                float(self.cfg["risk_weight"]) * risk +
                float(self.cfg["energy_weight"]) * energy
            )
            
            if score < best_score:
                best_score = score
                best_node = node_name
        
        return best_node
    
    def plan_job(
        self,
        job: Dict[str, Any],
        dry_run: bool = False,
        train: bool = False,
    ) -> Dict[str, Any]:
        """Plans job using Q-learning (with optional training)."""
        stages: List[Dict[str, Any]] = job.get("stages") or []
        
        if not stages:
            return {
                "job_id": job.get("id"),
                "assignments": {},
                "per_stage": [],
                "reservations": [],
                "latency_ms": 0.0,
                "energy_kj": 0.0,
                "risk": 0.0,
                "infeasible": True,
                "reason": "no_stages",
            }
        
        assignments: Dict[str, str] = {}
        per_stage: List[Dict[str, Any]] = []
        reservations: List[Dict[str, str]] = []
        
        prev_node: Optional[str] = None
        infeasible = False
        episode_reward = 0.0
        
        # Episode trajectory for batch updates
        trajectory: List[Tuple[str, str, float, Optional[str], List[str]]] = []
        
        for idx, st in enumerate(stages):
            sid = st.get("id")
            if not sid:
                per_stage.append({"infeasible": True, "reason": "missing_stage_id"})
                infeasible = True
                prev_node = None
                continue
            
            # Extracts state features
            state_features = _extract_state_features(st, idx, len(stages), prev_node)
            state_hash = _hash_state(state_features)
            
            # Gets feasible nodes
            feasible = self._get_feasible_nodes(st)
            
            if not feasible:
                per_stage.append({
                    "id": sid, "node": None, "infeasible": True,
                    "reason": "no_feasible_node"
                })
                infeasible = True
                
                if train:
                    reward = float(self.cfg["failure_penalty"])
                    episode_reward += reward
                
                prev_node = None
                continue
            
            # Selects action (node)
            selected_node = self._select_action(state_hash, feasible, explore=train)
            
            # Fallback to greedy if Q-table is empty and configured
            if selected_node is None and self.cfg.get("greedy_fallback"):
                selected_node = self._greedy_fallback(st, feasible, prev_node)
            
            if selected_node is None:
                per_stage.append({
                    "id": sid, "node": None, "infeasible": True,
                    "reason": "selection_failed"
                })
                infeasible = True
                prev_node = None
                continue
            
            # Tries reservation
            res_id = None
            success = True
            
            if not dry_run:
                res = st.get("resources") or {}
                req = {
                    "node": selected_node,
                    "cpu_cores": safe_float(res.get("cpu_cores"), 0.0),
                    "mem_gb": safe_float(res.get("mem_gb"), 0.0),
                    "gpu_vram_gb": safe_float(res.get("gpu_vram_gb"), 0.0),
                }
                res_id = self.state.reserve(req)
                
                if res_id is None:
                    success = False
                    per_stage.append({
                        "id": sid, "node": selected_node, "infeasible": True,
                        "reason": "reservation_failed"
                    })
                    infeasible = True
                    prev_node = None
                    continue
                
                reservations.append({"node": selected_node, "reservation_id": res_id})
            
            # Computes reward
            reward = self._compute_reward(st, selected_node, prev_node, success)
            episode_reward += reward
            
            # Stores trajectory for learning
            next_state_features = None
            next_feasible = []
            if idx + 1 < len(stages):
                next_stage = stages[idx + 1]
                next_state_features = _extract_state_features(
                    next_stage, idx + 1, len(stages), selected_node
                )
                next_feasible = self._get_feasible_nodes(next_stage)
            
            next_state_hash = _hash_state(next_state_features) if next_state_features else None
            trajectory.append((state_hash, selected_node, reward, next_state_hash, next_feasible))
            
            # Computes metrics for reporting
            node = self.state.nodes_by_name[selected_node]
            comp_ms = self.cm.compute_time_ms(st, node)
            xfer_ms = 0.0 if prev_node in (None, selected_node) else self.cm.transfer_time_ms(
                prev_node, selected_node, safe_float(st.get("size_mb"), 10.0)
            )
            energy = self.cm.energy_kj(st, node, comp_ms)
            risk = self.cm.risk_score(st, node)
            
            rec = {
                "id": sid,
                "node": selected_node,
                "reservation_id": res_id,
                "compute_ms": round(comp_ms, 3),
                "xfer_ms": round(xfer_ms, 3),
                "energy_kj": round(energy, 5),
                "risk": round(risk, 4),
            }
            per_stage.append(rec)
            assignments[sid] = selected_node
            prev_node = selected_node
        
        # Q-learning updates (if training)
        if train:
            for state_h, action, rew, next_state_h, next_feas in trajectory:
                self._update_q_value(state_h, action, rew, next_state_h, next_feas)
            
            # Update exploration rate
            self.epsilon = max(
                float(self.cfg["epsilon_min"]),
                self.epsilon * float(self.cfg["epsilon_decay"])
            )
            self.episodes_trained += 1
            self.total_reward_history.append(episode_reward)
        
        # Computes final job cost
        job_cost = self.cm.job_cost(job, assignments)
        merged_per_stage = merge_stage_details(per_stage, job_cost.get("per_stage") or [])
        
        # Adds success reward if feasible
        if train and not infeasible:
            episode_reward += float(self.cfg["success_reward"])
        
        result = {
            "job_id": job.get("id"),
            "assignments": assignments,
            "per_stage": merged_per_stage,
            "reservations": reservations,
            "latency_ms": job_cost.get("latency_ms", float("inf")),
            "energy_kj": job_cost.get("energy_kj", 0.0),
            "risk": job_cost.get("risk", 1.0),
            "infeasible": infeasible or (job_cost.get("latency_ms") == float("inf")),
        }
        
        if train:
            result["episode_reward"] = round(episode_reward, 2)
            result["epsilon"] = round(self.epsilon, 4)
        
        return result
    
    def save_model(self, path: str):
        """Saves Q-table and training state to disk."""
        # Converts defaultdict to regular dict for JSON serialization
        q_table_serializable = {
            state: dict(actions) for state, actions in self.q_table.items()
        }
        
        model_data = {
            "q_table": q_table_serializable,
            "epsilon": self.epsilon,
            "episodes_trained": self.episodes_trained,
            "total_reward_history": self.total_reward_history[-1000:],  # keep last 1000
            "config": self.cfg,
        }
        
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(model_data, f, indent=2)
    
    def load_model(self, path: str):
        """Loads Q-table and training state from disk."""
        with open(path, "r") as f:
            model_data = json.load(f)
        
        # Restore Q-table
        self.q_table = defaultdict(lambda: defaultdict(float))
        for state, actions in model_data.get("q_table", {}).items():
            for action, value in actions.items():
                self.q_table[state][action] = value
        
        self.epsilon = model_data.get("epsilon", self.cfg["epsilon"])
        self.episodes_trained = model_data.get("episodes_trained", 0)
        self.total_reward_history = model_data.get("total_reward_history", [])
    
    def get_stats(self) -> Dict[str, Any]:
        """Gets training statistics."""
        recent_rewards = self.total_reward_history[-100:] if self.total_reward_history else []
        
        return {
            "episodes_trained": self.episodes_trained,
            "current_epsilon": round(self.epsilon, 4),
            "q_table_size": len(self.q_table),
            "avg_recent_reward": round(sum(recent_rewards) / len(recent_rewards), 2) 
                                if recent_rewards else 0.0,
            "total_reward_history": self.total_reward_history[-20:],  # last 20
        }
