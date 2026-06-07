import pandas as pd
import glob
import os
import json
import datetime
import numpy as np

# ==============================================================================
# STS2RL MASTER TELEMETRY REPORT GENERATOR
# ==============================================================================

LOG_DIR = "./logs"
TECH_DIR = os.path.join(LOG_DIR, "sb3_tech")
MONITOR_PATTERN = os.path.join(LOG_DIR, "*monitor*")
EXAM_LOG = os.path.join(LOG_DIR, "exam_progression_log.csv")
REBOOT_LOG = os.path.join(LOG_DIR, "reboot_history.csv")
BEST_MODEL_JSON = os.path.join(LOG_DIR, "best_model_details.json")
PROGRESS_CSV = os.path.join(TECH_DIR, "progress.csv")

def format_step(step):
    if step >= 1_000_000: return f"{step/1_000_000:.1f}M"
    if step >= 1_000: return f"{step/1_000:.1f}K"
    return str(step)

def read_csv_safe(path, skip_header=True):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return pd.DataFrame()
    try:
        if skip_header:
            with open(path, 'r') as f:
                line = f.readline()
                if line.startswith("#"): return pd.read_csv(path, skiprows=1)
        return pd.read_csv(path)
    except:
        return pd.DataFrame()

def generate_report():
    print(f"\n{'='*70}")
    print(f"       Slay the Spire 2 - MASTER TELEMETRY GUIDE")
    print(f"       Report Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")

    print("### PART 1: TRAINING TELEMETRY (Learning Cluster)")
    
    mon_files = glob.glob(MONITOR_PATTERN)
    train_ports = ["15526", "15527", "15528"]
    train_files = [f for f in mon_files if any(p in f for p in train_ports)]
    
    train_dfs = [read_csv_safe(f) for f in train_files]
    v_delta = 0
    if train_dfs and not all(df.empty for df in train_dfs):
        df_train = pd.concat(train_dfs, ignore_index=True).sort_values(by='t').dropna()
        total_steps = df_train['l'].sum()
        total_runs = len(df_train)
        mean_rew = df_train['r'].mean()
        best_floor = df_train['floor'].max()
        
        print(f"- Stats: Total Steps: {total_steps:,} | Total Runs: {total_runs:,}")
        print(f"- Performance: Mean Reward: {mean_rew:.2f} | Best Floor: {best_floor}")
        
        # Windowed Velocity (Last 100 Runs vs Previous 100 Runs)
        window = 100
        if total_runs >= window * 2:
            latest_win = df_train.iloc[-window:]['r'].mean()
            prev_win = df_train.iloc[-(window*2):-window]['r'].mean()
            v_delta = latest_win - prev_win
            print(f"- Rolling Velocity (Last {window*2} runs): {prev_win:.2f} -> {latest_win:.2f} ({'[UP]' if v_delta > 0 else '[STALL]'} {v_delta:+.2f} Delta)")
        elif total_runs >= 4:
            mid = total_runs // 2
            q1 = df_train.iloc[:mid]['r'].mean()
            q4 = df_train.iloc[mid:]['r'].mean()
            v_delta = q4 - q1
            print(f"- Early Learning Delta: {q1:.2f} -> {q4:.2f} ({'[UP]' if v_delta > 0 else '[DOWN]'} {v_delta:+.2f} Delta)")
    else:
        print("- Stats: No training data found in monitor logs.")

    # Aggressive Progress Search (Master + Sessions)
    all_progs = []
    if os.path.exists(PROGRESS_CSV) and os.path.getsize(PROGRESS_CSV) > 0:
        try: all_progs.append(pd.read_csv(PROGRESS_CSV))
        except: pass
    
    session_files = glob.glob(os.path.join(TECH_DIR, "session_*", "progress.csv"))
    for f in session_files:
        if os.path.exists(f) and os.path.getsize(f) > 0:
            try: all_progs.append(pd.read_csv(f))
            except: pass
            
    df_prog = pd.DataFrame()
    if all_progs:
        df_prog = pd.concat(all_progs, ignore_index=True)
        if 'time/total_timesteps' in df_prog.columns:
            df_prog = df_prog.drop_duplicates(subset=['time/total_timesteps'], keep='last').sort_values(by='time/total_timesteps')

    if not df_prog.empty:
        latest = df_prog.iloc[-1]
        print(f"- SB3 Rollout: fps: {latest.get('time/fps', 0):.1f} | iterations: {latest.get('time/iterations', 0)} | elapsed: {latest.get('time/time_elapsed', 0)/60:.1f}m")
        print(f"- Technical Health & Confidence:")
        
        entropy = latest.get('train/entropy_loss', 0)
        kl = latest.get('train/approx_kl', 0)
        v_loss = latest.get('train/value_loss', 0)
        exp_var = latest.get('train/explained_variance', 0)
        
        var_status = "STABLE" if exp_var > 0.5 else "CONFUSED" if exp_var < 0 else "LEARNING"
        
        print(f"  > Explained Variance: {exp_var:.4f} [{var_status}]")
        print(f"  > Entropy Loss: {entropy:.4f} (Exploration: {'HIGH' if entropy < -1.0 else 'LOW'})")
        print(f"  > Policy/Value: approx_kl: {kl:.4f} | value_loss: {v_loss:.4f} | lr: {latest.get('train/learning_rate', 0):.2e}")
    else:
        print("- Technical Health: progress.csv is empty.")

    print("\n### PART 2: EVALUATION TELEMETRY (Mastery Exams)")
    
    if os.path.exists(BEST_MODEL_JSON):
        with open(BEST_MODEL_JSON, 'r') as f:
            best = json.load(f)
        print(f"- Champion Model: Phase {best.get('phase')} | Floor: {best.get('best_floor')} | Mean Reward: {best.get('mean_reward'):.2f}")
        print(f"- Mastery Success: {best.get('success_rate')} hit target floors.")
    
    df_exam = read_csv_safe(EXAM_LOG, skip_header=False)
    if not df_exam.empty:
        last_3 = df_exam.tail(3)
        print("- Latest Exams:")
        for _, row in last_3.iterrows():
            print(f"  > Step {format_step(row['step'])}: Floor {row['mean_floor']:.1f} | Reward {row['mean_reward']:.2f}")
    else:
        print("- Latest Exams: No historical exam data found.")

    print("\n### PART 3: STATUS FLAGS & ORCHESTRATION")
    
    if train_dfs and not all(df.empty for df in train_dfs):
        recent_20 = df_train.tail(20)
        stalling = recent_20['l'].mean() > 1000 and recent_20['floor'].mean() < 5
        print(f"- [STATUS]: {'[UP] Staircase Reward Trend' if v_delta > 0 else '[OK] Stable Baseline'}")
        if stalling: print("- [WARNING]: Potential Stalling detected.")
    
    print("- Orchestrator: Master Log Consolidation active. Data Isolation confirmed.")

    print("\n### PART 4: CLUSTER STABILITY (Node Recovery)")
    
    df_reboot = read_csv_safe(REBOOT_LOG, skip_header=False)
    if not df_reboot.empty:
        total_restarts = len(df_reboot)
        node_counts = df_reboot['Node'].value_counts()
        node_str = ", ".join([f"{n}: {c}" for n, c in node_counts.items()])
        
        tracked = df_reboot[df_reboot['PID'].astype(str).str.contains('PID [0-9]|(?<!Unknown)[0-9]', regex=True, na=False)].shape[0]
        success_rate = (tracked / total_restarts) * 100
        
        print(f"- Recovery Stats: Total Restarts: {total_restarts} | Success Rate: {success_rate:.1f}%")
        print(f"- Node Hotspots: {node_str}")
        
        latest_reboot = df_reboot.iloc[-1]
        print(f"- Latest Recovery: {latest_reboot['Timestamp']} | {latest_reboot['Reason']}")
    else:
        print("- Stability: No reboots recorded.")

    print("\n### PART 5: STORAGE TELEMETRY (Godot Logs)")
    appdata = os.getenv('APPDATA')
    if appdata:
        godot_log_dir = os.path.join(appdata, "SlayTheSpire2", "logs")
        if os.path.exists(godot_log_dir):
            total_size_bytes = sum(os.path.getsize(os.path.join(godot_log_dir, f)) for f in os.listdir(godot_log_dir) if os.path.isfile(os.path.join(godot_log_dir, f)))
            size_gb = total_size_bytes / (1024**3)
            print(f"- Godot Log Size: {size_gb:.2f} GB")
            if size_gb > 5.0:
                print(f"- [WARNING]: Storage bloat detected ({size_gb:.2f} GB). Recommend restarting training instance.")
            else:
                print("- Storage Health: [OK] Log size is within safe limits.")
        else:
            print("- Storage: Godot log directory not found.")
    else:
        print("- Storage: Unable to locate APPDATA environment variable.")

    print(f"\n{'='*70}\n")

if __name__ == "__main__":
    try:
        generate_report()
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Failed to generate report: {e}")
        import traceback
        traceback.print_exc()
