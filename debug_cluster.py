import os
import json
import glob
from datetime import datetime

def analyze_node_trace(trace_file):
    if not os.path.exists(trace_file):
        return None
        
    total_steps = 0
    rejections = 0
    auto_kicks = 0
    fatalities = 0
    
    # We can track average time between attempts
    last_time = None
    time_diffs = []
    
    with open(trace_file, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                msg = data.get('msg', '')
                ts_str = data.get('ts')
                
                if ts_str:
                    try:
                        current_time = datetime.strptime(ts_str, "%H:%M:%S")
                        if "Attempting:" in msg or "SMART-KICK" in msg:
                            if last_time:
                                diff = (current_time - last_time).total_seconds()
                                if diff >= 0 and diff < 60: # Ignore massive gaps or negative wraps
                                    time_diffs.append(diff)
                            last_time = current_time
                    except: pass
                
                if "ACCEPTED" in msg:
                    total_steps += 1
                elif "REJECTED" in msg:
                    rejections += 1
                elif "SMART-KICK" in msg:
                    auto_kicks += 1
                elif "FATAL" in msg:
                    fatalities += 1
            except: pass
            
    avg_step_time = sum(time_diffs) / len(time_diffs) if time_diffs else 0.0
    
    return {
        "steps": total_steps,
        "rejections": rejections,
        "auto_kicks": auto_kicks,
        "fatalities": fatalities,
        "avg_step_time_sec": avg_step_time
    }

def main():
    print(f"{'='*60}")
    print(f"       STS2 CLUSTER DIAGNOSTIC & TIMING DASHBOARD")
    print(f"       Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")
    
    trace_files = glob.glob("logs/node_trace_*.jsonl")
    if not trace_files:
        print("[!] No node trace logs found. Training might not have started yet.")
    
    total_rejections = 0
    total_kicks = 0
    
    for tf in sorted(trace_files):
        port = tf.split("_")[-1].replace(".jsonl", "")
        stats = analyze_node_trace(tf)
        if stats:
            print(f"--- Node Port: {port} ---")
            print(f"  Valid Steps Logged: {stats['steps']}")
            print(f"  Masked Rejections : {stats['rejections']}")
            print(f"  Auto-Kicks (Stuck): {stats['auto_kicks']}")
            print(f"  Game Over Deaths  : {stats['fatalities']}")
            print(f"  Avg Cycle Time    : {stats['avg_step_time_sec']:.3f} seconds/step")
            print()
            total_rejections += stats['rejections']
            total_kicks += stats['auto_kicks']
            
    print(f"--- Cluster Reboot History ---")
    reboot_file = "logs/reboot_history.csv"
    if os.path.exists(reboot_file):
        with open(reboot_file, 'r') as f:
            lines = f.readlines()
            if len(lines) > 1:
                # Show last 5 reboots
                print("  Recent Crash/Deadlock Events:")
                for line in lines[-5:]:
                    print(f"  > {line.strip()}")
            else:
                print("  [OK] No cluster reboots recorded.")
    else:
        print("  [OK] No cluster reboots recorded.")
        
    print(f"\n{'='*60}")
    print(f"HEALTH SUMMARY:")
    if total_rejections > 500:
        print(f"[!] HIGH REJECTION COUNT ({total_rejections}). Action mask might be desynced.")
    else:
        print(f"[OK] Rejection count is healthy ({total_rejections}).")
        
    if total_kicks > 50:
        print(f"[!] HIGH AUTO-KICK COUNT ({total_kicks}). Agent is getting stuck frequently.")
    else:
        print(f"[OK] Stagnation Auto-kicks are healthy ({total_kicks}).")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
