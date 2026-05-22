import os
import glob
import time
import datetime
import json
import copy
import shutil
import gymnasium as gym
import numpy as np
import requests
import pandas as pd
import torch as th
import torch.nn as nn
from tqdm import tqdm

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.logger import configure
import sys

# ==============================================================================
# CONSOLE MIRROR LOGGER (Tee Utility)
# ==============================================================================
class TeeLogger:
    def __init__(self, filename, mode="a", max_mb=50):
        self.filename = filename
        self.max_bytes = max_mb * 1024 * 1024
        self.terminal = sys.stdout
        self.log_file = open(filename, mode, encoding="utf-8")

    def _rotate_if_needed(self):
        if os.path.exists(self.filename) and os.path.getsize(self.filename) > self.max_bytes:
            self.log_file.close()
            bak_file = self.filename + ".bak"
            if os.path.exists(bak_file):
                try: os.remove(bak_file)
                except: pass
            try: os.rename(self.filename, bak_file)
            except: pass
            self.log_file = open(self.filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self._rotate_if_needed()
        self.log_file.write(message)
        self.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

class TeeErrorLogger(TeeLogger):
    def __init__(self, filename, mode="a", max_mb=50):
        super().__init__(filename, mode, max_mb)
        self.terminal = sys.stderr
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize, unwrap_vec_normalize

from sts2_env import SlayTheSpire2Env, FLAT_ACTIONS

CHECKPOINT_DIR = "./checkpoints/"
LOG_DIR = "./logs/"
MODEL_DIR = "./models/"
ULTIMATE_BEST_DIR = os.path.join(MODEL_DIR, "ultimate_best")
TOP_MODELS_DIR = os.path.join(MODEL_DIR, "top_3")
HOF_DIR = os.path.join(MODEL_DIR, "hall_of_fame")
VICTORY_SEEDS_FILE = os.path.join(LOG_DIR, "victory_seeds.json")

SESSION_ID = int(time.time())

TRAIN_PORTS = [15526, 15527, 15528]
EVAL_PORT = 15529

BUFFER_STAGES = [
    (1024, 256, 18_432), 
    (3072, 512, 100_000_000)
]

PERF_STATE_FILE = os.path.join(LOG_DIR, "all_time_perf.json")

class SynergyCNNExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space):
        super().__init__(observation_space, features_dim=1152)
        # 1D-CNN branch for hand synergy analysis
        # Input shape: (batch, 30, 10) -> (batch, hand_features, hand_size)
        self.hand_cnn = nn.Sequential(
            nn.Conv1d(in_channels=30, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.relic_cnn = nn.Sequential(
            nn.Conv1d(in_channels=5, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.meta_processor = nn.Sequential(nn.Linear(1136, 512), nn.ReLU())
        self.final_dense = nn.Sequential(nn.Linear(1280 + 640 + 512, 1152), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        meta_1 = observations[:, :320]
        hand_flat = observations[:, 320:620]
        meta_2 = observations[:, 620:770]
        relic_flat = observations[:, 770:870]
        meta_3 = observations[:, 870:]
        metadata = th.cat([meta_1, meta_2, meta_3], dim=1)
        hand_spatial = hand_flat.view(-1, 10, 30).transpose(1, 2)
        relic_spatial = relic_flat.view(-1, 20, 5).transpose(1, 2)
        return self.final_dense(th.cat([self.hand_cnn(hand_spatial), self.relic_cnn(relic_spatial), self.meta_processor(metadata)], dim=1))

class GlobalPerformanceTracker:
    def __init__(self):
        self.best_reward, self.best_floor, self.hof_entries = -1000.0, 0, []
        self.load()
    def load(self):
        if os.path.exists(PERF_STATE_FILE):
            try:
                with open(PERF_STATE_FILE, "r") as f:
                    data = json.load(f)
                    self.best_reward, self.best_floor, self.hof_entries = data.get("best_reward", -1000.0), data.get("best_floor", 0), data.get("hof_entries", [])
            except: pass
    def update(self, reward, floor):
        if reward > self.best_reward: self.best_reward = reward
        if floor > self.best_floor: self.best_floor = floor
        self.save()
    def add_hof(self, reward, model_path):
        self.hof_entries.append({"reward": float(reward), "path": model_path, "date": str(datetime.datetime.now())})
        self.hof_entries = sorted(self.hof_entries, key=lambda x: x["reward"], reverse=True)
        
        if len(self.hof_entries) > 5:
            to_delete = self.hof_entries[5:]
            self.hof_entries = self.hof_entries[:5]
            for entry in to_delete:
                old_path = entry["path"]
                old_vec = old_path.replace(".zip", ".pkl")
                try:
                    if os.path.exists(old_path): os.remove(old_path)
                    if os.path.exists(old_vec): os.remove(old_vec)
                except: pass
        
        self.save()
    def save(self):
        try:
            with open(PERF_STATE_FILE, "w") as f: json.dump({"best_reward": float(self.best_reward), "best_floor": int(self.best_floor), "hof_entries": self.hof_entries}, f)
        except: pass

def synchronize_logs(current_step, checkpoint_path=None):
    """Merges fragmented logs and truncates ghost steps to ensure 100% data integrity."""
    print(f"[INTEGRITY] Synchronizing logs with current brain step: {current_step:,}...")
    
    tech_log_dir = os.path.join(LOG_DIR, "sb3_tech")
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint_time = os.path.getmtime(checkpoint_path)
        tech_files = glob.glob(os.path.join(tech_log_dir, "*"))
        purged_count = 0
        for path in tech_files:
            if os.path.getmtime(path) > checkpoint_time:
                try:
                    os.remove(path)
                    purged_count += 1
                except: pass
        if purged_count > 0:
            print(f"  > sb3_tech: Purged {purged_count} future/ghost session files.")

    log_files = {os.path.join(tech_log_dir, "progress.csv"): "timesteps", "exam_progression_log.csv": "step"}
    for path, step_col in log_files.items():
        if os.path.exists(path) and os.path.getsize(path) > 0:
            try:
                df = pd.read_csv(path)
                if step_col in df.columns:
                    original_len = len(df)
                    df = df[df[step_col] <= current_step]
                    if len(df) < original_len:
                        df.to_csv(path, index=False)
                        print(f"  > {os.path.basename(path)}: Truncated {original_len - len(df)} rows.")
            except: pass

    target_node_steps = current_step // len(TRAIN_PORTS)
    for port in TRAIN_PORTS + [EVAL_PORT]:
        fragments = glob.glob(os.path.join(LOG_DIR, f"*monitor*{port}*"))
        master_path = os.path.join(LOG_DIR, f"monitor_{port}_master.csv")
        
        if not fragments: continue
        
        combined_data = []
        header_to_keep = f'#{{ "t_start": {time.time()}, "env_id": "None" }}\n'
        
        for f_path in fragments:
            if "master" in f_path: continue
            try:
                with open(f_path, 'r') as f:
                    line = f.readline()
                    if line.startswith("#"): header_to_keep = line
                    df = pd.read_csv(f)
                    if not df.empty: combined_data.append(df)
            except: pass
            
        if os.path.exists(master_path):
            try:
                with open(master_path, 'r') as f:
                    f.readline() 
                    df = pd.read_csv(f)
                    if not df.empty: combined_data.insert(0, df)
            except: pass

        if combined_data:
            master_df = pd.concat(combined_data, ignore_index=True).sort_values(by='t').drop_duplicates()
            
            if port in TRAIN_PORTS:
                master_df['cumsum_l'] = master_df['l'].cumsum()
                original_len = len(master_df)
                master_df = master_df[master_df['cumsum_l'] <= target_node_steps]
                cleaned_count = original_len - len(master_df)
                master_df = master_df.drop(columns=['cumsum_l'])
            else:
                cleaned_count = 0

            with open(master_path, 'w', newline='') as f:
                f.write(header_to_keep)
                master_df.to_csv(f, index=False)
            
            for f_path in fragments:
                if "master" not in f_path:
                    try: os.remove(f_path)
                    except: pass
            
            status = f"Merged {len(combined_data)} runs"
            if cleaned_count > 0: status += f" and cleaned {cleaned_count} ghost runs"
            print(f"  > Port {port}: {status}. Master log established.")

    all_junk = glob.glob(os.path.join(LOG_DIR, "*tfevents*")) + glob.glob(os.path.join(LOG_DIR, "*monitor*"))
    for path in all_junk:
        if "master" in path or os.path.isdir(path): continue
        try:
            if os.path.getsize(path) < 150:
                os.remove(path)
        except: pass

class TensorboardMetricsCallback(BaseCallback):
    def __init__(self, tracker, verbose=0):
        super().__init__(verbose)
        self.tracker, self.counter_file = tracker, os.path.join(LOG_DIR, "run_counter.json")
        self.episode_count = self._load_counter()
    def _load_counter(self):
        if os.path.exists(self.counter_file):
            try:
                with open(self.counter_file, "r") as f: return json.load(f).get("count", 0)
            except: return 0
        return 0
    def _save_counter(self):
        try:
            with open(self.counter_file, "w") as f: json.dump({"count": self.episode_count}, f)
        except: pass
    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_count += 1
                self._save_counter()
                floor = info.get("floor", 0)
                reward = info["episode"]["r"]
                
                if reward > self.tracker.best_reward:
                    hof_path = os.path.join(HOF_DIR, f"hof_score_{int(reward)}_step_{self.model.num_timesteps}.zip")
                    hof_vec_path = hof_path.replace(".zip", ".pkl")
                    self.model.save(hof_path)
                    norm_env = unwrap_vec_normalize(self.training_env)
                    if norm_env: norm_env.save(hof_vec_path)
                    self.tracker.add_hof(reward, hof_path)
                    print(f"[HALL OF FAME] New Record! Score: {reward:.1f} saved to gallery.")

                self.tracker.update(reward, floor)
                timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                print(f"[{timestamp}] RUN #{self.episode_count} | Floor: {floor} | Rew: {reward:.1f} (Best: {self.tracker.best_floor})")
        return True

class CustomProgressBarCallback(BaseCallback):
    def __init__(self, total_timesteps, phase_manager, tracker, verbose=0):
        super().__init__(verbose)
        self.total_timesteps, self.pm, self.tracker, self.pbar = total_timesteps, phase_manager, tracker, None
    def _on_training_start(self) -> None:
        self.pbar = tqdm(total=self.total_timesteps, initial=self.model.num_timesteps, desc=f"Phase {self.pm.phase_idx}", unit="steps")
    def _on_step(self) -> bool:
        self.pbar.n = self.model.num_timesteps
        steps_to_eval = (self.pm.last_eval_step + self.pm.eval_freq_global) - self.model.num_timesteps
        self.pbar.set_description(f"Phase {self.pm.phase_idx} | Exam in: {max(0, steps_to_eval):,} | B.Floor: {self.tracker.best_floor} | B.Rew: {self.tracker.best_reward:.1f}")
        self.pbar.refresh()
        return True
    def _on_training_end(self) -> None:
        if self.pbar: self.pbar.close()

class PhaseManagerCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq_global, tracker, n_eval_episodes=10, state_file=None, verbose=1):
        super().__init__(verbose)
        self.eval_env, self.eval_freq_global, self.tracker, self.n_eval_episodes = eval_env, eval_freq_global, tracker, n_eval_episodes
        self.state_file = state_file or os.path.join(CHECKPOINT_DIR, "cluster_state.json")
        self.phase_idx, self.last_eval_step, self.best_mean_reward, self.consecutive_regressions = 1, 0, -1000.0, 0
        self.consecutive_failures = 0 
        self._load_state()

    def set_eval_freq(self, new_freq):
        self.eval_freq_global = new_freq

    def _load_state(self):
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r") as f:
                    state = json.load(f)
                    self.phase_idx = state.get("phase_idx", 1)
                    self.last_eval_step = state.get("last_eval_step", 0)
                    self.best_mean_reward = state.get("best_mean_reward", -1000.0)
            except: pass

    def save_state(self):
        try:
            with open(self.state_file, "w") as f:
                json.dump({
                    "phase_idx": int(self.phase_idx),
                    "last_eval_step": int(self.last_eval_step),
                    "best_mean_reward": float(self.best_mean_reward)
                }, f)
        except: pass

    def _on_training_start(self): 
        self._apply_phase()
        self._on_step()

    def _apply_phase(self):
        self.training_env.env_method("set_training_phase", self.phase_idx)
        self.save_state()
        # Dynamic Entropy: Base decay + "Shock" boost if failing
        base_entropy = max(0.01, 0.05 * (0.7 ** (self.phase_idx - 1)))
        self.model.ent_coef = min(0.05, base_entropy + (self.consecutive_failures * 0.01))
        lr = 1e-4 if self.phase_idx < 3 else 5e-5
        for p in self.model.policy.optimizer.param_groups: p['lr'] = lr

    def _on_step(self) -> bool:
        current_step = self.model.num_timesteps
        self.training_env.env_method("set_global_step", current_step)
        return True

    def _on_rollout_start(self) -> None:
        current_step = self.model.num_timesteps
        if current_step >= self.last_eval_step + self.eval_freq_global:
            self._run_exam()

    def _run_exam(self):
        current_step = self.model.num_timesteps
        self.last_eval_step = (current_step // self.eval_freq_global) * self.eval_freq_global
        self.save_state()
        
        model_path = os.path.join(CHECKPOINT_DIR, f"sts2_mid_run_step_{current_step}.zip")
        vec_path = model_path.replace(".zip", ".pkl")
        self.model.save(model_path)
        norm_env = unwrap_vec_normalize(self.training_env)
        if norm_env: norm_env.save(vec_path)
        print(f"[CHECKPOINT] Updated Brain Secured at Step {current_step:,}")
        
        all_zips = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "sts2_mid_run_step_*.zip")), key=os.path.getmtime)
        while len(all_zips) > 3:
            oldest_zip = all_zips.pop(0)
            oldest_vec = oldest_zip.replace(".zip", ".pkl")
            try: 
                os.remove(oldest_zip)
                if os.path.exists(oldest_vec): os.remove(oldest_vec)
            except: pass
        
        print(f"\n{'='*60}")
        print(f"[EXAM] MILESTONE REACHED: {current_step:,} steps")
        print(f"[EXAM] Starting Deterministic Skills Test on Port {EVAL_PORT}...")
        print(f"{'='*60}\n")

        total_rewards, total_floors, total_lengths = [], [], []
        eval_pbar = tqdm(total=self.n_eval_episodes, desc="[EXAM] Progress", unit="ep", ascii=True)
        
        for ep in range(self.n_eval_episodes):
            obs = self.eval_env.reset()
            done, ep_rew, ep_len, floor = [False], 0.0, 0, 0
            while not done[0]:
                masks = self.eval_env.env_method("action_masks")[0]
                action, _ = self.model.predict(obs, action_masks=masks, deterministic=True)
                obs, reward, done, info = self.eval_env.step(action)
                ep_rew += float(reward[0])
                ep_len += 1
                floor = info[0].get("floor", 0)
            
            total_rewards.append(ep_rew)
            total_floors.append(floor)
            total_lengths.append(ep_len)
            print(f"  > Ep {ep+1}: Floor {floor} | Reward {ep_rew:.1f}")
            eval_pbar.update(1)
        
        eval_pbar.close()
        
        mean_reward_final = np.mean(total_rewards)
        mean_floor = np.mean(total_floors)
        best_floor = np.max(total_floors)
        worst_floor = np.min(total_floors)
        mean_len = np.mean(total_lengths)
        consistency = np.std(total_floors)
        
        # Mastery Thresholds
        promo_reward_targets = {1: 100.0, 2: 325.0, 3: 550.0, 4: 500.0, 5: 450.0, 6: 400.0}
        promo_floor_targets = {1: 16.0, 2: 33.0, 3: 50.0, 4: 51.0, 5: 51.0, 6: 51.0}
        exam_win_floors = {1: 16, 2: 33, 3: 50, 4: 51, 5: 51, 6: 51}
        
        target_win_floor = exam_win_floors.get(self.phase_idx, 51)
        wins = sum(1 for f in total_floors if f >= target_win_floor)
        
        print(f"\n[EXAM] Mean Reward: {mean_reward_final:.2f} | Mean Floor: {mean_floor:.1f}")
        print(f"[EXAM] Successes: {wins}/{self.n_eval_episodes} hit target Floor {target_win_floor}")
        
        latest_stats_path = os.path.join(LOG_DIR, f"latest_exam_stats_step_{current_step}.txt")
        with open(latest_stats_path, "w") as f:
            f.write(f"{'='*40}\n")
            f.write(f"LATEST EXAM REPORT - STEP {current_step}\n")
            f.write(f"{'='*40}\n")
            f.write(f"Phase: {self.phase_idx}\n")
            f.write(f"Total Training Steps: {current_step}\n\n")
            f.write(f"--- FLOOR METRICS (10-Game Exam) ---\n")
            f.write(f"Mean Floor Reached: {mean_floor:.1f}\n")
            f.write(f"Best Floor Reached: {best_floor}\n")
            f.write(f"Worst Floor Reached: {worst_floor}\n\n")
            f.write(f"--- REWARD METRICS (10-Game Exam) ---\n")
            f.write(f"Mean Score: {mean_reward_final:.2f}\n")
            f.write(f"Highest Score: {np.max(total_rewards):.2f}\n")
            f.write(f"Lowest Score: {np.min(total_rewards):.2f}\n")
            f.write(f"{'='*40}\n")
        
        all_latest = sorted(glob.glob(os.path.join(LOG_DIR, "latest_exam_stats_step_*.txt")), key=os.path.getmtime)
        while len(all_latest) > 3:
            try: os.remove(all_latest.pop(0))
            except: pass

        best_bio_path = os.path.join(LOG_DIR, "best_model_details.json")
        with open(best_bio_path, "w") as bio_f:
            json.dump({
                "global_step": int(current_step),
                "phase": int(self.phase_idx),
                "date": str(datetime.datetime.now()),
                "mean_reward": float(mean_reward_final),
                "best_reward": float(np.max(total_rewards)),
                "worst_reward": float(np.min(total_rewards)),
                "mean_floor": float(mean_floor),
                "best_floor": int(best_floor),
                "worst_floor": int(worst_floor),
                "mean_length": float(mean_len),
                "consistency_stddev": float(consistency),
                "success_rate": f"{wins}/{self.n_eval_episodes}"
            }, bio_f, indent=4)
        
        prog_log_path = os.path.join(LOG_DIR, "exam_progression_log.csv")
        prog_data = {
            "step": [current_step],
            "phase": [self.phase_idx],
            "mean_reward": [mean_reward_final],
            "mean_floor": [mean_floor],
            "success_rate": [wins / self.n_eval_episodes],
            "timestamp": [str(datetime.datetime.now())]
        }
        df_new = pd.DataFrame(prog_data)
        if not os.path.exists(prog_log_path):
            df_new.to_csv(prog_log_path, index=False)
        else:
            df_new.to_csv(prog_log_path, mode='a', header=False, index=False)
        
        top_log_path = os.path.join(TOP_MODELS_DIR, "top_models_log.json")
        top_entries = []
        if os.path.exists(top_log_path):
            try:
                with open(top_log_path, "r") as f: top_entries = json.load(f)
            except: pass
        
        top_entries.append({
            "mean_reward": float(mean_reward_final),
            "mean_floor": float(mean_floor),
            "step": int(current_step),
            "phase": int(self.phase_idx),
            "success_rate": f"{wins}/{self.n_eval_episodes}",
            "date": str(datetime.datetime.now()),
            "id": f"step_{current_step}_phase_{self.phase_idx}"
        })
        top_entries = sorted(top_entries, key=lambda x: x["mean_reward"], reverse=True)[:3]
        
        with open(top_log_path, "w") as f: json.dump(top_entries, f, indent=4)
        
        for rank, entry in enumerate(top_entries, 1):
            if entry["id"] == f"step_{current_step}_phase_{self.phase_idx}":
                rank_model_path = os.path.join(TOP_MODELS_DIR, f"model_rank_{rank}_step_{current_step}_phase_{self.phase_idx}.zip")
                rank_vec_path = rank_model_path.replace(".zip", ".pkl")
                rank_report_path = os.path.join(TOP_MODELS_DIR, f"report_rank_{rank}_step_{current_step}_phase_{self.phase_idx}.txt")
                
                self.model.save(rank_model_path)
                norm_env = unwrap_vec_normalize(self.training_env)
                if norm_env: norm_env.save(rank_vec_path)
                
                with open(rank_report_path, "w") as f:
                    f.write(f"{'='*40}\n")
                    f.write(f"TOP MODEL RANK #{rank} - VERIFIED EXAM\n")
                    f.write(f"{'='*40}\n")
                    f.write(f"Phase: {self.phase_idx} | Step: {current_step}\n")
                    f.write(f"Mean Reward: {mean_reward_final:.2f}\n")
                    f.write(f"Mean Floor:  {mean_floor:.1f}\n")
                    f.write(f"Success Rate: {wins}/{self.n_eval_episodes}\n")
                    f.write(f"Consistency: {consistency:.2f} (Floor StdDev)\n")
                    f.write(f"Date: {entry['date']}\n\n")
                    f.write(f"--- EPISODE BREAKDOWN ---\n")
                    for i, (r, f_val) in enumerate(zip(total_rewards, total_floors)):
                        f.write(f"Run {i+1}: Floor {f_val} | Reward {r:.1f}\n")
                    f.write(f"{'='*40}\n")
                
                for f in glob.glob(os.path.join(TOP_MODELS_DIR, f"*_rank_{rank}_*")):
                    if entry["id"] not in f:
                        try: os.remove(f)
                        except: pass

        if mean_reward_final > self.best_mean_reward:
            self.best_mean_reward = mean_reward_final
            self.consecutive_regressions = 0
            best_model_path = os.path.join(ULTIMATE_BEST_DIR, f"ultimate_best_model_Phase_{self.phase_idx}.zip")
            best_vec_path = best_model_path.replace(".zip", ".pkl")
            
            self.model.save(best_model_path)
            norm_env = unwrap_vec_normalize(self.training_env)
            if norm_env: norm_env.save(best_vec_path)
            
            best_stats_path = os.path.join(ULTIMATE_BEST_DIR, f"ultimate_best_stats_Phase_{self.phase_idx}.txt")
            with open(best_stats_path, "w") as f:
                f.write(f"{'='*40}\n")
                f.write(f"CHAMPION MODEL - PHASE {self.phase_idx}\n")
                f.write(f"{'='*40}\n")
                f.write(f"Total Training Steps: {current_step}\n")
                f.write(f"Total Exam Runs: {self.n_calls}\n\n")
                f.write(f"--- FLOOR METRICS (10-Game Exam) ---\n")
                f.write(f"Mean Floor Reached: {mean_floor:.1f}\n")
                f.write(f"Best Floor Reached: {best_floor}\n")
                f.write(f"Worst Floor Reached: {worst_floor}\n\n")
                f.write(f"--- REWARD METRICS (10-Game Exam) ---\n")
                f.write(f"Mean Score: {mean_reward_final:.2f}\n")
                f.write(f"Highest Score: {np.max(total_rewards):.2f}\n")
                f.write(f"Lowest Score: {np.min(total_rewards):.2f}\n")
                f.write(f"{'='*40}\n")
            print(f"  [+] New Personal Best for Phase {self.phase_idx}!")
        else:
            self.consecutive_regressions += 1

        r_target = promo_reward_targets.get(self.phase_idx, 999)
        f_target = promo_floor_targets.get(self.phase_idx, 999)
        
        if mean_reward_final >= r_target and mean_floor >= f_target and self.phase_idx < 6:
            if self.phase_idx == 3:
                grad_dir = os.path.join(MODEL_DIR, "phase_3_graduates")
                os.makedirs(grad_dir, exist_ok=True)
                grad_reg_file = os.path.join(grad_dir, "graduates.json")
                graduates = []
                if os.path.exists(grad_reg_file):
                    try:
                        with open(grad_reg_file, "r") as f: graduates = json.load(f)
                    except: pass
                
                new_grad_id = f"grad_step_{current_step}"
                graduates.append({"reward": float(mean_reward_final), "step": int(current_step), "id": new_grad_id, "date": str(datetime.datetime.now())})
                graduates = sorted(graduates, key=lambda x: x["reward"], reverse=True)
                
                if len(graduates) > 5:
                    to_delete = graduates[5:]
                    graduates = graduates[:5]
                    for entry in to_delete:
                        old_id = entry["id"]
                        for f in glob.glob(os.path.join(grad_dir, f"*{old_id}*")):
                            try: os.remove(f)
                            except: pass
                
                with open(grad_reg_file, "w") as f: json.dump(graduates, f, indent=4)
                
                if any(g["id"] == new_grad_id for g in graduates):
                    self.model.save(os.path.join(grad_dir, f"{new_grad_id}.zip"))
                    norm_env = unwrap_vec_normalize(self.training_env)
                    if norm_env: norm_env.save(os.path.join(grad_dir, f"{new_grad_id}.pkl"))
                    print(f"  [+] Phase 3 Graduation Backup Secured: {new_grad_id}")

            self.phase_idx += 1
            print(f"\n[PROMOTION] Mastery achieved! Advancing to Phase {self.phase_idx}!\n")
            print(f"  > Req: Reward {r_target:.1f}, Floor {f_target:.1f}")
            print(f"  > Got: Reward {mean_reward_final:.1f}, Floor {mean_floor:.1f}")
            self.consecutive_failures = 0
            self._apply_phase()
        else:
            if self.phase_idx > 1:
                if wins == 0: self.consecutive_failures += 1
                else: self.consecutive_failures = 0

                if self.consecutive_failures >= 3:
                    self.phase_idx -= 1
                    print(f"\n[DEMOTION] Performance Stall Detected. Stepping down to Phase {self.phase_idx}...")
                    self.consecutive_failures = 0
                    self.best_mean_reward = promo_reward_targets.get(self.phase_idx, 0.0) - 5.0 
                    self._apply_phase()
        
        self.save_state()
        print(f"{'='*60}\n")

def get_newest_model_and_vec():
    all_zips = []
    for root, _, files in os.walk(MODEL_DIR):
        for f in files:
            if f.endswith(".zip"):
                all_zips.append(os.path.join(root, f))
    for root, _, files in os.walk(CHECKPOINT_DIR):
        for f in files:
            if f.endswith(".zip"):
                all_zips.append(os.path.join(root, f))
    
    if not all_zips: return None, None
    
    latest_model = max(all_zips, key=os.path.getmtime)
    print(f"[SWEEPER] Resuming from absolute newest file: {latest_model}")
    
    vec_path = latest_model.replace(".zip", ".pkl")
    if os.path.exists(vec_path):
        return latest_model, vec_path
    
    dirname = os.path.dirname(latest_model)
    emergency_pkl = os.path.join(dirname, "sts2_emergency_save.pkl")
    best_pkl = os.path.join(dirname, "vec_normalize_best.pkl")
    
    if os.path.exists(emergency_pkl): return latest_model, emergency_pkl
    if os.path.exists(best_pkl): return latest_model, best_pkl
    
    return latest_model, None

def create_fresh_model(env, n_steps, batch_size, weights_path=None):
    policy_kwargs = dict(features_extractor_class=SynergyCNNExtractor, net_arch=dict(pi=[512, 256], vf=[512, 256]))
    model_params = {
        "policy": "MlpPolicy", "env": env, "verbose": 1, "tensorboard_log": "./tensorboard/",
        "device": "auto", "learning_rate": 1e-4, "n_steps": n_steps, "batch_size": batch_size,
        "n_epochs": 10, "gamma": 0.999, "gae_lambda": 0.98, "clip_range": 0.2, "ent_coef": 0.05,
        "vf_coef": 0.7, "policy_kwargs": policy_kwargs
    }
    
    if weights_path and os.path.exists(weights_path):
        try:
            print(f"[LOAD] Attempting to resume full model state from {weights_path}...")
            model = MaskablePPO.load(weights_path, env=env, n_steps=n_steps, batch_size=batch_size)
            print(f"[LOAD] Success! Full model state restored.")
            return model
        except Exception as e:
            print(f"[LOAD] Partial load fallback (Parameters mismatch or error): {e}")
            new_model = MaskablePPO(**model_params)
            old = MaskablePPO.load(weights_path)
            new_model.set_parameters(old.get_parameters())
            new_model.num_timesteps = old.num_timesteps
            return new_model
            
    return MaskablePPO(**model_params)

def make_env(port, is_eval=False):
    def _init():
        node_id = {15526:1, 15527:2, 15528:3, 15529:4}.get(port, 1)
        base_path = f"C:\\Program Files (x86)\\Steam\\steamapps\\common\\STS2_Node_{node_id}"
        data_dir = f"Node{node_id}_Data"
        base_env = SlayTheSpire2Env(port=port, game_path=base_path, user_data_dir=data_dir, force_fresh=False, is_eval=is_eval)
        env = ActionMasker(base_env, lambda e: e.action_masks())
        return Monitor(env, os.path.join(LOG_DIR, f"monitor_{port}_{SESSION_ID}.csv"), info_keywords=("floor",))
    return _init

def setup_vec_env(ports, vec_path=None, is_eval=False):
    vec_env = SubprocVecEnv([make_env(p, is_eval) for p in ports], start_method="spawn") if not is_eval else DummyVecEnv([make_env(p, is_eval) for p in ports])
    if vec_path and os.path.exists(vec_path):
        env = VecNormalize.load(vec_path, vec_env)
    else:
        env = VecNormalize(vec_env, norm_obs=False, norm_reward=True, clip_obs=10.0)
    if is_eval:
        env.training = False
        env.norm_reward = False
    return env

def main():
    for d in [CHECKPOINT_DIR, MODEL_DIR, LOG_DIR, ULTIMATE_BEST_DIR, HOF_DIR, TOP_MODELS_DIR]: os.makedirs(d, exist_ok=True)
    
    # Global Log Nuke (STS2 engine logs)
    appdata = os.getenv('APPDATA')
    if appdata:
        game_logs_path = os.path.join(appdata, "SlayTheSpire2", "logs")
        if os.path.exists(game_logs_path):
            print(f"[SYSTEM] Nuking engine logs: {game_logs_path}")
            for f in glob.glob(os.path.join(game_logs_path, "*")):
                try: 
                    if os.path.isfile(f): os.remove(f)
                except: pass

    # Initialize Console Mirror
    console_log_path = os.path.join(LOG_DIR, "train_console.log")
    sys.stdout = TeeLogger(console_log_path, mode="a")
    sys.stderr = TeeErrorLogger(console_log_path, mode="a")
    
    print(f"\n{'='*50}")
    print(f"--- NEW TRAINING SESSION STARTED: {datetime.datetime.now()} ---")
    print(f"{'='*50}\n")
    
    tech_log_dir = os.path.join(LOG_DIR, "sb3_tech")
    os.makedirs(tech_log_dir, exist_ok=True)
    new_logger = configure(tech_log_dir, ["stdout", "csv", "tensorboard"])
    current_model_file, latest_vec_path = get_newest_model_and_vec()
    train_env = setup_vec_env(ports=TRAIN_PORTS, vec_path=latest_vec_path, is_eval=False)
    eval_env = setup_vec_env(ports=[EVAL_PORT], vec_path=latest_vec_path, is_eval=True)
    tracker = GlobalPerformanceTracker()
    metrics_cb = TensorboardMetricsCallback(tracker)
    start_n_steps, start_batch_size = BUFFER_STAGES[0][0], BUFFER_STAGES[0][1]
    if current_model_file:
        try:
            temp_model = MaskablePPO.load(current_model_file)
            current_steps = temp_model.num_timesteps
            del temp_model
            for n_steps, batch_size, stage_end in BUFFER_STAGES:
                if current_steps < stage_end:
                    start_n_steps, start_batch_size = n_steps, batch_size
                    break
        except: pass

    model = None
    try:
        for n_steps, batch_size, stage_end in BUFFER_STAGES:
            eval_freq_global = n_steps * len(TRAIN_PORTS) 
            if stage_end == 18_432: 
                eval_freq_global *= 2
            
            phase_cb = PhaseManagerCallback(eval_env, eval_freq_global, tracker, n_eval_episodes=10)
            
            if model is None: 
                model = create_fresh_model(train_env, start_n_steps, start_batch_size, current_model_file)
                model.set_logger(new_logger)
                synchronize_logs(model.num_timesteps)
            
            if model.num_timesteps >= stage_end: 
                continue

            if model.n_steps != n_steps:
                tmp_path = os.path.join(CHECKPOINT_DIR, "transition_temp.zip")
                tmp_vec = os.path.join(CHECKPOINT_DIR, "transition_temp.pkl")
                model.save(tmp_path)
                tn = unwrap_vec_normalize(train_env)
                if tn: tn.save(tmp_vec)
                model = create_fresh_model(train_env, n_steps, batch_size, tmp_path)
                model.set_logger(new_logger)
                synchronize_logs(model.num_timesteps)
                
                try:
                    if os.path.exists(tmp_path): os.remove(tmp_path)
                    if os.path.exists(tmp_vec): os.remove(tmp_vec)
                    print(f"[SYSTEM] Stage Handoff Successful. Transition files purged.")
                except: pass
            
            model.learn(
                total_timesteps=(stage_end - model.num_timesteps), 
                callback=CallbackList([metrics_cb, phase_cb, CustomProgressBarCallback(stage_end, phase_cb, tracker)]), 
                reset_num_timesteps=False, 
                progress_bar=False
            )

            if model.num_timesteps >= phase_cb.last_eval_step + phase_cb.eval_freq_global:
                phase_cb._run_exam()
    except (KeyboardInterrupt, Exception) as e:
        print(f"\n[HALT] Cluster Failure or Stop Detected ({type(e).__name__}): {e}")
        import traceback
        traceback.print_exc()
        if 'phase_cb' in locals():
            phase_cb.save_state()
        print(f"[SAVE] Final state secured. Manual intervention required.")
    finally:
        try: 
            print("[SYSTEM] Closing environment clusters...")
            train_env.close()
            eval_env.close()
        except: pass

if __name__ == "__main__":
    main()
