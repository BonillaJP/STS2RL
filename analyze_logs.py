import pandas as pd
import glob
import os
import json

def analyze_subset(df, name, is_eval=False):
    if df.empty:
        print(f"\n--- {name} ---")
        print("No data found.")
        return

    df['r'] = pd.to_numeric(df['r'], errors='coerce')
    df['floor'] = pd.to_numeric(df['floor'], errors='coerce')
    df['l'] = pd.to_numeric(df['l'], errors='coerce')
    df['t'] = pd.to_numeric(df['t'], errors='coerce')
    df = df.sort_values(by='t').dropna()

    count = len(df)
    
    print(f"\n--- {name} ---")
    
    if is_eval:
        best_bio_path = './logs/best_model_details.json'
        if os.path.exists(best_bio_path):
            try:
                with open(best_bio_path, 'r') as f:
                    best_data = json.load(f)
                print(f"[ALL-TIME BEST MODEL] Floor: {best_data.get('best_floor', 0)} | Mean Reward: {best_data.get('mean_reward', 0):.2f} | Success: {best_data.get('success_rate', 'N/A')}")
            except: pass
    
    print(f"Total Steps: {df['l'].sum():,}")
    print(f"Total Runs: {count}")
    print(f"Highest Reward: {df['r'].max():.2f} | Lowest Reward: {df['r'].min():.2f}")
    print(f"Mean Reward: {df['r'].mean():.2f} | Best Floor: {df['floor'].max()}")
    
    if count >= 20:
        l20_df = df.iloc[-20:]
        print(f"[CURRENT PERFORMANCE - LAST 20] Reward: {l20_df['r'].mean():.2f} | Floor: {l20_df['floor'].mean():.2f}")

    if count >= 4:
        q_size = max(1, int(count/4))
        q1_df = df.iloc[:q_size]
        q4_df = df.iloc[-q_size:]
        
        print(f"[Q1 Trend] Floor: {q1_df['floor'].mean():.2f} | Reward: {q1_df['r'].mean():.2f} | Length: {q1_df['l'].mean():.1f}")
        print(f"[Q4 Trend] Floor: {q4_df['floor'].mean():.2f} | Reward: {q4_df['r'].mean():.2f} | Length: {q4_df['l'].mean():.1f}")
        
        rew_diff = q4_df['r'].mean() - q1_df['r'].mean()
        print(f"Overall Progress: {'📈' if rew_diff > 0 else '📉'} {rew_diff:+.2f} Reward Delta")
    else:
        print(f"Mean Floor: {df['floor'].mean():.2f}")

def analyze_reboots():
    reboot_file = './logs/reboot_history.csv'
    if not os.path.exists(reboot_file):
        print("\n--- CLUSTER STABILITY ---")
        print("No reboot history found.")
        return

    try:
        df = pd.read_csv(reboot_file)
        if df.empty: return
        
        print("\n--- CLUSTER STABILITY ---")
        print(f"Total Autonomous Restarts: {len(df)}")
        
        # Count by node
        node_counts = df['Node'].value_counts()
        node_str = ", ".join([f"{n}: {c}" for n, c in node_counts.items()])
        print(f"Restarts by Node: {node_str}")
        
        # Red Flag detection:
        non_standard = df[df['Reason'] != "API Startup Timeout"]
        if not non_standard.empty:
            print(f"[RED FLAG] {len(non_standard)} non-standard reboots detected!")
            for _, row in non_standard.iterrows():
                print(f"  > {row['Timestamp']} | {row['Reason']}")
        
        # Recent activity
        latest = df.iloc[-1]
        print(f"Latest Recovery: {latest['Timestamp']} | {latest['Node']} | {latest['Reason']}")
        
        if 'PID' in df.columns:
            tracked = df[df['PID'].astype(str).str.contains('PID [0-9]|(?<!Unknown)[0-9]', regex=True, na=False)].shape[0]
            print(f"Recovery Success Rate: {(tracked/len(df))*100:.1f}% (PID Tracked)")
    except Exception as e:
        print(f"\n[ERROR] Analyzing reboots: {e}")

log_files = glob.glob('./logs/monitor*.csv')
train_logs = [f for f in log_files if any(p in f for p in ["15526", "15527", "15528"])]
eval_logs = [f for f in log_files if "15529" in f]

train_data = []
for f in train_logs:
    try:
        df = pd.read_csv(f, skiprows=1)
        train_data.append(df)
    except: pass

eval_data = []
for f in eval_logs:
    try:
        df = pd.read_csv(f, skiprows=1)
        eval_data.append(df)
    except: pass

if train_data:
    analyze_subset(pd.concat(train_data, ignore_index=True), "TRAINING CLUSTER (Nodes 15526-15528)")

if eval_data:
    analyze_subset(pd.concat(eval_data, ignore_index=True), "EVALUATION NODE (Node 15529)", is_eval=True)

analyze_reboots()
