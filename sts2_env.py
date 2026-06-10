import gymnasium as gym
from gymnasium import spaces
import numpy as np
import requests
import time
import hashlib
import json
import subprocess
import os
import random
import re
import datetime
import glob

def stable_hash(name, max_val=10000):
    if not name: return 0
    return (int(hashlib.md5(str(name).encode()).hexdigest(), 16) % max_val) + 1

# Maps the neural network's 342 discrete outputs to specific game commands.
FLAT_ACTIONS = []
for c in range(10): 
    for t in range(5): FLAT_ACTIONS.append({"action": "play_card", "card_index": c, "target_idx": t})
for p in range(5): 
    for t in range(5): FLAT_ACTIONS.append({"action": "use_potion", "slot": p, "target_idx": t})
for p in range(5): FLAT_ACTIONS.append({"action": "discard_potion", "slot": p})
FLAT_ACTIONS.append({"action": "end_turn"})
for c in range(10): FLAT_ACTIONS.append({"action": "combat_select_card", "card_index": c})
FLAT_ACTIONS.append({"action": "combat_confirm_selection"})
for r in range(15): FLAT_ACTIONS.append({"action": "claim_reward", "index": r})
for c in range(15): FLAT_ACTIONS.append({"action": "select_card_reward", "card_index": c})
FLAT_ACTIONS.append({"action": "skip_card_reward"})
FLAT_ACTIONS.append({"action": "proceed"})
for o in range(10): FLAT_ACTIONS.append({"action": "choose_event_option", "index": o})
FLAT_ACTIONS.append({"action": "advance_dialogue"})
for o in range(5): FLAT_ACTIONS.append({"action": "choose_rest_option", "index": o})
for i in range(15): FLAT_ACTIONS.append({"action": "shop_purchase", "index": i})
for n in range(5): FLAT_ACTIONS.append({"action": "choose_map_node", "index": n})
for c in range(50): FLAT_ACTIONS.append({"action": "select_card", "index": c}) 
FLAT_ACTIONS.extend([{"action": "confirm_selection"}, {"action": "cancel_selection"}])
for b in range(5): FLAT_ACTIONS.append({"action": "select_bundle", "index": b})
FLAT_ACTIONS.extend([{"action": "confirm_bundle_selection"}, {"action": "cancel_bundle_selection"}])
for r in range(5): FLAT_ACTIONS.append({"action": "select_relic", "index": r})
FLAT_ACTIONS.append({"action": "skip_relic_selection"})
for r in range(5): FLAT_ACTIONS.append({"action": "claim_treasure_relic", "index": r})

MENU_OPTS = ["main_menu", "singleplayer", "standard", "embark", "continue", "yes", "no", "back", "settings", "ironclad", "silent", "defect", "regent", "necrobinder", "random"]
for m in MENU_OPTS: FLAT_ACTIONS.append({"action": "menu_select", "option": m})

FLAT_ACTIONS.append({"action": "crystal_sphere_set_tool", "tool": "big"})
FLAT_ACTIONS.append({"action": "crystal_sphere_set_tool", "tool": "small"})
for y in range(11):
    for x in range(9): FLAT_ACTIONS.append({"action": "crystal_sphere_click_cell", "x": x, "y": y})
FLAT_ACTIONS.append({"action": "crystal_sphere_proceed"})

# ==============================================================================
# GLOBAL MODIFIERS & CONFIGURATION
# ==============================================================================
GLOBAL_CONFIG = {
    # The base path to the game node folders. 
    "BASE_PATH_TEMPLATE": r"D:\Games\Steam\steamapps\common\STS2_Node_",
    
    # Target Character selection for singleplayer mode.
    # Options: IRONCLAD, SILENT, DEFECT, REGENT, NECROBINDER, RANDOM
    "TARGET_CHARACTER": "IRONCLAD",
    
    # Global stability throttle: adds a small delay (seconds) between actions.
    "STABILITY_THROTTLE": 0.05,
    
    # Base penalty for every action taken.
    "STEP_TAX": -0.05,
    
    # Reward Bounties for achieving milestones.
    "BOUNTIES": {
        "FLOOR_CLIMB": 10.0,
        "BOSS_KILL": 100.0,
        "BOSS_REACH": 50.0,
        "ELITE_KILL": 15.0,
        "ELITE_SIGHT": 5.0,
        "SMITH_BASE": 15.0,
        "CARD_REMOVAL": 15.0,
        "DRAFT_CARD": 2.0,
        "POTION_USE": 2.0,
        "NEW_RELIC": 25.0,
    },
    
    "STAGNATION": {
        "LOW_PENALTY_STEP": 10,
        "HIGH_PENALTY_STEP": 20,
        "CRITICAL_STEP": 50,
        "REBOOT_STEP": 200,
    },

    # 4096-float vector dimensions.
    "OBS_SIZE": 4096
}

ENCHANTMENT_KEYWORDS = ["adroit", "corrupted", "glam", "goopy", "imbued", "instinct", "momentum", "nimble", "perfect fit", "royally approved", "sharp", "slither", "soul's power", "sown", "swift", "vigorous"]
OVERLAY_KEYS = ["card_select", "enchanting", "enchant", "transform", "simple_select", "choose", "upgrade_select", "remove"]

class SlayTheSpire2Env(gym.Env):
    def __init__(self, port=15526, game_path=None, user_data_dir=None, force_fresh=False, is_eval=False):
        super().__init__()
        self.port = port
        
        node_id = {15526: 1, 15527: 2, 15528: 3, 15529: 4}.get(port, 1)
        
        if not game_path:
            base_node_path = f"{GLOBAL_CONFIG['BASE_PATH_TEMPLATE']}{node_id}"
            game_path = os.path.join(base_node_path, f"Node {node_id}.exe")
            if not os.path.exists(game_path):
                game_path = os.path.join(base_node_path, "SlayTheSpire2.exe")
        
        self.game_path = game_path
        self.user_data_dir = user_data_dir
        self.force_fresh = force_fresh
        self.is_eval = is_eval
        self.game_url = f"http://127.0.0.1:{port}/api/v1/singleplayer"
        self.action_space = spaces.Discrete(len(FLAT_ACTIONS))
        
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(GLOBAL_CONFIG["OBS_SIZE"],), dtype=np.float32)

        self.session = None
        self.current_state = {}
        self.previous_floor, self.previous_hp = 0, 0
        self.training_phase = 1
        self.lowest_enemy_hp_seen = {}
        self.combat_step_count = 0
        self.total_episode_steps = 0
        self.stagnant_steps = 0
        self.needs_reboot = False
        self.reboot_reason = "Stall/Deadlock Recovery"
        self.last_prog_screen, self.last_prog_floor, self.last_prog_hp_sum = "none", -1, -1
        self.last_prog_gold = -1
        self.last_prog_energy = -1
        self.last_prog_deck_size = -1
        self.last_prog_relic_count = -1
        self.last_prog_basic_card_count = -1
        self.last_prog_enemy_block_sum = -1
        self.last_prog_player_block = -1
        self.last_prog_selected_cards = []
        self.internal_selection_history = []
        self.sel_prog_steps = 0
        self.last_sel_count = -1
        self.combat_turn_count = 0
        
        self.pending_elite_bounty = False
        self.pending_boss_bounty = False
        self.pending_smith_bounty = False
        self.global_step = 0
        self.last_state_hash = ""
        self.state_action_counts = {} 
        self.action_cooldowns = {}
        self.state_rejection_count = 0
        self.ghost_save_cleanup_pending = False
        self.full_purge_cleanup_pending = False
        
        if self.port == 15526:
            self._vacuum_disk(full_nuke=True)
        
        self.trace_path = f"logs/node_trace_{self.port}.jsonl"
        try: 
            os.makedirs("logs", exist_ok=True)
            with open(self.trace_path, "a") as f: 
                f.write(json.dumps({"ts": str(datetime.datetime.now()), "msg": "--- Trace Session Started ---"}) + "\n")
        except: pass

    def _log_action(self, msg, vision=None):
        try:
            if os.path.exists(self.trace_path) and os.path.getsize(self.trace_path) > 1024 * 1024 * 1024:
                with open(self.trace_path, "w") as f: 
                    f.write(json.dumps({"ts": str(datetime.datetime.now()), "msg": "--- Trace Rotated ---"}) + "\n")
            
            with open(self.trace_path, "a") as f:
                ts = datetime.datetime.now().strftime("%H:%M:%S")
                log_obj = {"ts": ts, "msg": msg}
                if vision: log_obj["vision"] = vision
                f.write(json.dumps(log_obj) + "\n")
        except: pass

    def set_training_phase(self, phase_int):
        self.training_phase = phase_int

    def set_global_step(self, step):
        self.global_step = step

    def _get_session(self, reset=False):
        if reset and self.session:
            try: self.session.close()
            except: pass
            self.session = None
        if self.session is None: self.session = requests.Session()
        return self.session

    def _log_reboot(self, reason, pid="Unknown"):
        log_file = "logs/reboot_history.csv"
        try:
            os.makedirs("logs", exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            msg = f"{timestamp},Port {self.port},PID {pid},GlobalStep {self.global_step},LocalSteps {self.total_episode_steps},{reason}\n"
            
            lines = []
            if os.path.exists(log_file):
                with open(log_file, "r") as f:
                    lines = f.readlines()
            
            if not lines:
                lines = ["Timestamp,Node,PID,GlobalStep,LocalSteps,Reason\n"]
            
            lines.append(msg)
            if len(lines) > 500: lines = [lines[0]] + lines[-499:]
            
            with open(log_file, "w") as f:
                f.writelines(lines)
        except: pass

    def _vacuum_disk(self, full_nuke=True):
        appdata = os.getenv('APPDATA')
        if appdata:
            game_data_path = os.path.join(appdata, "SlayTheSpire2")
            for sub in [os.path.join(game_data_path, "logs"), os.path.join(game_data_path, "sentry", "reports")]:
                if os.path.exists(sub):
                    for f in glob.glob(os.path.join(sub, "*")):
                        try:
                            if os.path.isfile(f):
                                if full_nuke or os.path.getsize(f) > 1024 * 1024 * 1024:
                                    os.remove(f)
                                elif not full_nuke and os.path.basename(f).lower() == "godot.log":
                                    continue
                                else:
                                    os.remove(f)
                        except: pass

    def _nuke_current_run_save(self, target_id=None):
        appdata = os.getenv('APPDATA')
        if not appdata: return
        
        steam_path = os.path.join(appdata, "SlayTheSpire2", "steam")
        if not os.path.exists(steam_path): return
        
        user_dirs = [d for d in os.listdir(steam_path) if d.isdigit()]
        if not user_dirs: return
        
        profile_to_nuke = target_id if target_id else {15526: 1, 15527: 2, 15528: 3, 15529: 1}.get(self.port, 1)
        
        for user_id in user_dirs:
            save_dir = os.path.join(steam_path, user_id, "modded", f"profile{profile_to_nuke}", "saves")
            if os.path.exists(save_dir):
                for f in glob.glob(os.path.join(save_dir, "*")):
                    fname = os.path.basename(f).lower()
                    if "progress" in fname or "prefs" in fname: continue
                    if "run" in fname:
                        try: 
                            os.remove(f)
                            print(f"[CLEANUP] Deleted Run File: {os.path.basename(f)} (Profile {profile_to_nuke})")
                        except: pass

    def _reboot_game_client(self, reason=None):
        log_reason = reason if reason else self.reboot_reason
        node_id = {15526: 1, 15527: 2, 15528: 3, 15529: 4}.get(self.port, 1)

        self._log_reboot(log_reason)
        self.reboot_reason = "Stall/Deadlock Recovery"

        if "Storage Bloat" in str(log_reason):
            print(f"\n[PURGE] Storage bloat detected. Shutting down entire cluster...")
            ps_kill_all = "Get-Process | Where-Object { $_.Name -eq 'Node 1' -or $_.Name -eq 'Node 2' -or $_.Name -eq 'Node 3' -or $_.Name -eq 'Node 4' -or $_.Name -eq 'SlayTheSpire2' } | Stop-Process -Force"
            subprocess.run(["powershell", "-NoProfile", "-Command", ps_kill_all], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            time.sleep(3.0)
            self._vacuum_disk(full_nuke=True)
            self.full_purge_cleanup_pending = True
            
            print(f"[PURGE] Relaunching all 4 game nodes...")
            for i in range(1, 5):
                try:
                    target_exe = re.sub(r'Node \d', f'Node {i}', self.game_path)
                    target_exe = re.sub(r'Node_\d', f'Node_{i}', target_exe)
                    game_dir = os.path.dirname(target_exe)
                    game_exe = os.path.basename(target_exe)
                    cmd = f'cmd /c start "" "{game_exe}"'
                    subprocess.Popen(cmd, cwd=game_dir, shell=True)
                except: pass
        else:
            print(f"\n[REBOOT] Node {node_id} failure. Performing surgical restart...")
            target_name = f"Node {node_id}"
            ps_kill_node = f"Get-Process | Where-Object {{ $_.Name -eq '{target_name}' }} | Stop-Process -Force"
            subprocess.run(["powershell", "-NoProfile", "-Command", ps_kill_node], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            time.sleep(2.0)
            self._vacuum_disk(full_nuke=False)
            
            try:
                game_dir = os.path.dirname(self.game_path)
                game_exe = os.path.basename(self.game_path)
                cmd = f'cmd /c start "" "{game_exe}"'
                subprocess.Popen(cmd, cwd=game_dir, shell=True)
            except Exception as e: print(f"[REBOOT] relaunch failed: {e}")

        self._get_session(reset=True)

        for _ in range(30):
            try:
                cmd = f'netstat -ano | findstr LISTENING | findstr :{self.port}'
                output = subprocess.check_output(cmd, shell=True).decode('utf-8')
                if not output.strip(): break
            except: break
            time.sleep(1.0)
        
        for i in range(60):
            try:
                resp = requests.get(f"http://127.0.0.1:{self.port}/api/v1/profiles", timeout=1.0)
                if resp.status_code == 200: 
                    print(f"[REBOOT] Node {node_id} Online after {i}s."); break
            except: pass
            time.sleep(1.0)

    def action_masks(self):
        state = self.current_state
        mask = np.zeros(len(FLAT_ACTIONS), dtype=bool)
        if not state: return mask

        raw_screen = state.get("state_type", "unknown")
        player = state.get("player", {})
        battle = state.get("battle", {})
        enemies = battle.get("enemies", [])
        
        if "hand_select" in state: screen = "hand_select"
        elif any(k in state for k in OVERLAY_KEYS): screen = "card_select_overlay"
        elif len(enemies) > 0 or battle.get("is_play_phase"): screen = "combat"
        elif "treasure" in state: screen = "treasure"
        elif "card_reward" in state: screen = "card_reward"
        elif "rewards" in state: screen = "rewards"
        elif "event" in state or "neow" in state: screen = "event"
        elif "bundle_select" in state: screen = "bundle_select"
        elif "relic_select" in state: screen = "relic_select"
        elif "shop" in state or "fake_merchant" in state: screen = "shop"
        elif "rest_site" in state: screen = "rest_site"
        elif "crystal_sphere" in state: screen = "crystal_sphere"
        elif raw_screen == "map" and (state.get("map", {}).get("next_options") or state.get("map", {}).get("can_proceed")): screen = "map"
        else: screen = raw_screen

        def set_mask(start_idx, count, valid_indices=None):
            if valid_indices is None: mask[start_idx:start_idx+count] = True
            else:
                for idx in valid_indices:
                    if idx is not None and idx < count: mask[start_idx + idx] = True

        potions = player.get("potions", [])
        if len(potions) >= player.get("max_potion_slots", 3):
            if screen in ["combat", "rewards", "shop", "fake_merchant", "event", "neow", "treasure", "rest_site"]:
                for pot in potions:
                    slot_idx = pot.get("slot")
                    if slot_idx is not None and slot_idx < 5: mask[75 + slot_idx] = True

        if screen == "combat":
            if battle.get("is_play_phase"):
                mask[80] = True 
                for c_idx, card in enumerate(player.get("hand", [])[:10]):
                    can_play_flag = card.get("can_play")
                    if can_play_flag is None: can_play_flag = card.get("can_afford", True)
                    if can_play_flag:
                        if card.get("target_type") in ["AnyEnemy", "SingleTarget", "Target"]:
                            for t_idx in range(min(5, len(enemies))): mask[c_idx * 5 + t_idx] = True
                        else: mask[c_idx * 5] = True
                for p_idx, potion in enumerate(potions[:5]):
                    p_slot = potion.get("slot")
                    if p_slot is None or p_slot >= 5: continue
                    can_use_flag = potion.get("can_use_in_combat")
                    if can_use_flag is None: can_use_flag = potion.get("can_afford", True)
                    if can_use_flag:
                        if potion.get("target_type") in ["AnyEnemy", "SingleTarget", "Target"]:
                            for t_idx in range(min(5, len(enemies))): mask[50 + p_slot * 5 + t_idx] = True
                        else: mask[50 + p_slot * 5] = True 
        elif screen == "hand_select":
            hs = state.get("hand_select", {})
            match = re.search(r'\b(\d+)\b', hs.get("prompt", ""))
            max_select = int(match.group(1)) if match else 1
            can_confirm = hs.get("can_confirm", False)
            raw_sel = hs.get("selected_cards", []) or hs.get("selected", [])
            api_sel = set(str(c.get("index", c)) for c in raw_sel if c is not None)
            if not api_sel:
                api_sel = set(str(c.get("index")) for c in hs.get("cards", []) if (c.get("is_selected") or c.get("isSelected")))
            merged_selected = api_sel | set(map(str, self.internal_selection_history))
            eff_sel = 1 if (can_confirm and max_select == 1) else len(merged_selected)
            if eff_sel < max_select:
                set_mask(81, 10, [c.get("index") for c in hs.get("cards", []) if str(c.get("index")) not in merged_selected])
            if can_confirm or eff_sel >= max_select: mask[91] = True
        elif screen == "rewards":
            rew = state.get("rewards", {})
            set_mask(92, 15, [i.get("index") for i in rew.get("items", [])])
            if rew.get("can_proceed") or not rew.get("items"): mask[123] = True
        elif screen == "card_reward":
            cr = state.get("card_reward", {})
            set_mask(107, 15, [c.get("index") for c in cr.get("cards", [])])
            if cr.get("can_skip"): mask[122] = True
        elif screen in ["event", "neow"]:
            ev = state.get("event") or state.get("neow") or {}
            opts = ev.get("options", [])
            if ev.get("in_dialogue"): mask[134] = True 
            else:
                hp_ratio = player.get("hp", 0) / max(player.get("max_hp", 1), 1)
                selectable = [o.get("index") for o in opts if not (o.get("is_locked") or o.get("was_chosen") or (o.get("title", "").lower() == "linger" and hp_ratio < 0.3))]
                if selectable: set_mask(124, 10, selectable)
                if not selectable and not ev.get("in_dialogue"): mask[123] = True
        elif screen in ["room", "unknown"]: mask[123] = True
        elif screen == "rest_site":
            rs = state.get("rest_site", {})
            set_mask(135, 5, [o.get("index") for o in rs.get("options", []) if o.get("is_enabled")])
            if rs.get("can_proceed"): mask[123] = True
        elif screen == "shop" or screen == "fake_merchant":
            shop = state.get("fake_merchant", {}).get("shop", {}) if raw_screen == "fake_merchant" else state.get("shop", {})
            gold = player.get("gold", 0)
            set_mask(140, 15, [i.get("index") for i in shop.get("items", []) if i.get("price", i.get("cost", 9999)) <= gold and i.get("is_stocked")])
            mask[123] = True 
        elif screen == "map":
            map_data = state.get("map", {})
            next_nodes = map_data.get("next_options", [])
            selectable_nodes = [o.get("index") for o in next_nodes if o.get("is_selectable", True)]
            set_mask(155, 5, selectable_nodes)
            is_boss_floor = state.get("run", {}).get("floor", 0) in [17, 33, 48]
            if not selectable_nodes and map_data.get("can_proceed") and is_boss_floor: mask[123] = True
            else: mask[123] = False
        elif screen == "card_select_overlay":
            cs = next((state.get(k, {}) for k in OVERLAY_KEYS if k in state), {})
            prompt = cs.get("prompt", "")
            match = re.search(r'(?:choose|select|pick|to)\s+(\d+)', prompt, re.I)
            max_sel = int(match.group(1)) if match else 1
            raw_sel = cs.get("selected_cards", []) or cs.get("selected", []) or [{"index": i} for i in cs.get("selected_card_indices", [])]
            selected_indices = [str(c.get("index", c)) for c in raw_sel if c is not None]
            if not selected_indices: 
                selected_indices = [str(c.get("index")) for c in cs.get("cards", []) if (c.get("is_selected") or c.get("isSelected"))]
            num_selected = len(selected_indices)
            is_grid = cs.get("screen_type") in ["transform", "upgrade", "select", "simple_select"] or "can_confirm" in cs
            if is_grid:
                can_confirm = cs.get("can_confirm", False)
                if num_selected < max_sel:
                    merged_selected = set(selected_indices) | set(map(str, self.internal_selection_history))
                    if len(merged_selected) < max_sel:
                        set_mask(160, 50, [c.get("index") for c in cs.get("cards", []) if str(c.get("index")) not in merged_selected])
                if can_confirm or num_selected >= max_sel: 
                    mask[210] = True
                    mask[211] = False
                else:
                    if num_selected > 0: mask[211] = True
                    elif cs.get("can_cancel"): mask[211] = True
            else:
                set_mask(160, 50, [c.get("index") for c in cs.get("cards", [])])
                if cs.get("can_cancel"): mask[211] = True
            if cs.get("can_proceed"): mask[123] = True
        elif screen == "bundle_select":
            bs = state.get("bundle_select", {})
            set_mask(212, 5, [b.get("index") for b in bs.get("bundles", [])])
            if bs.get("can_confirm"): mask[217] = True
            if bs.get("can_cancel"): mask[218] = True
        elif screen == "relic_select":
            rs = state.get("relic_select", {})
            set_mask(219, 5, [r.get("index") for r in rs.get("relics", [])])
            if rs.get("can_skip"): mask[224] = True
        elif screen == "treasure":
            set_mask(225, 5, [r.get("index") for r in state.get("treasure", {}).get("relics", [])])
            if state.get("treasure", {}).get("can_proceed"): mask[123] = True
        elif screen == "crystal_sphere":
            cs = state.get("crystal_sphere", {})
            if cs.get("can_use_big_tool"): mask[240] = True
            if cs.get("can_use_small_tool"): mask[241] = True
            for cell in cs.get("clickable_cells", []):
                x, y = cell.get("x", 0), cell.get("y", 0)
                if 0 <= x < 9 and 0 <= y < 11:
                    mask[242 + (y * 9) + x] = True
            if cs.get("can_proceed"): mask[341] = True

        opts = [o.get("name", "").lower() if isinstance(o, dict) else o.lower() for o in state.get("options", [])]
        for i, m in enumerate(MENU_OPTS):
            if m.lower() in opts and m.lower() != "settings": mask[230 + i] = True

        for act_idx, count in self.state_action_counts.items():
            if count >= 20: mask[act_idx] = False 
        if screen != "map": mask[155:160] = False

        if not mask.any() or getattr(self, 'state_rejection_count', 0) >= 20:
            if screen in ["monster", "elite", "boss", "combat"]: mask[80] = True
            elif screen == "map":
                map_data = state.get("map", {})
                next_nodes = map_data.get("next_options", [])
                selectable_nodes = [o.get("index") for o in next_nodes if o.get("is_selectable", True)]
                if selectable_nodes: mask[155 + selectable_nodes[0]] = True
                else: mask[123] = True
            else: mask[123] = True
            
        for act_idx, cooldown in self.action_cooldowns.items():
            if cooldown > 0: mask[act_idx] = False
            
        return mask

    def _post(self, payload):
        try:
            resp = self._get_session().post(self.game_url, json=payload, timeout=5.0)
            if resp.status_code != 200: return False
            try:
                if resp.json().get("status") == "error": return False
            except: pass
            return True
        except: return False

    def _raw_state(self):
        for _ in range(5):
            try:
                resp = self._get_session().get(f"{self.game_url}?format=json", timeout=0.5)
                if resp.status_code == 200:
                    state = resp.json()
                    if state: return state
            except requests.exceptions.ConnectionError:
                return {} 
            except: pass
            time.sleep(0.01)
        return {}

    def _ensure_fresh_run(self):
        while True:
            consecutive_failures = 0
            old_hash = ""
            for i in range(1200):
                state = self._raw_state()
                if not state:
                    consecutive_failures += 1
                    if consecutive_failures >= 300:
                        self._reboot_game_client(reason="API Connection Lost")
                        break 
                    time.sleep(0.1); continue
                consecutive_failures = 0
                
                new_hash = hashlib.md5(json.dumps(state, sort_keys=True).encode()).hexdigest()
                if new_hash == old_hash and state.get("state_type") not in ["monster", "elite", "boss"]:
                    time.sleep(0.01); continue
                
                screen, menu = state.get("state_type", ""), state.get("menu_screen", "")
                ACTIVE = ["monster", "elite", "boss", "map", "event", "rest_site", "rewards", "card_reward", "shop", "neow", "hand_select", "treasure", "card_select", "bundle_select", "relic_select", "fake_merchant", "crystal_sphere"]
                
                if screen in ACTIVE: 
                    while state and state.get("state_type") in ["monster", "elite", "boss"] and state.get("battle", {}).get("is_play_phase") == False:
                        time.sleep(0.05)
                        state = self._raw_state()
                        if not state: break
                    self.current_state = state; return
                
                old_hash = new_hash
                
                opts = [o.get("name", "").lower() if isinstance(o, dict) else o.lower() for o in state.get("options", [])]
                if menu == "main":
                    target_profile = {15526: 1, 15527: 2, 15528: 3, 15529: 1}.get(self.port, 1)
                    
                    if getattr(self, 'full_purge_cleanup_pending', False):
                        print(f"[PURGE] Performing multi-profile save nuke to restore cluster stability...")
                        for p_id in [1, 2, 3]:
                            try:
                                profile_url = f"http://127.0.0.1:{self.port}/api/v1/profiles"
                                self._get_session().post(profile_url, json={"action": "switch", "profile_id": p_id}, timeout=5.0)
                                time.sleep(1.0)
                                self._nuke_current_run_save(target_id=p_id)
                                alt_p = 2 if p_id == 1 else 1
                                self._get_session().post(profile_url, json={"action": "switch", "profile_id": alt_p}, timeout=5.0)
                                time.sleep(1.0)
                                self._get_session().post(profile_url, json={"action": "switch", "profile_id": p_id}, timeout=5.0)
                            except: pass
                        
                        self.full_purge_cleanup_pending = False
                        self._get_session().post(profile_url, json={"action": "switch", "profile_id": target_profile}, timeout=5.0)
                        continue

                    if getattr(self, 'ghost_save_cleanup_pending', False):
                        self._nuke_current_run_save(target_id=target_profile)
                        
                        alt_profile = {1: 2, 2: 3, 3: 1}.get(target_profile, 2)
                        profile_url = f"http://127.0.0.1:{self.port}/api/v1/profiles"
                        try:
                            print(f"[CLEANUP] Refreshing profile {target_profile} (Switching {target_profile} -> {alt_profile} -> {target_profile}) to clear ghost saves...")
                            self._get_session().post(profile_url, json={"action": "switch", "profile_id": alt_profile}, timeout=5.0)
                            time.sleep(1.0)
                            self._get_session().post(profile_url, json={"action": "switch", "profile_id": target_profile}, timeout=5.0)
                            time.sleep(1.0)
                            self.ghost_save_cleanup_pending = False
                            continue 
                        except: pass

                    try:
                        profile_url = f"http://127.0.0.1:{self.port}/api/v1/profiles"
                        resp = self._get_session().get(profile_url, timeout=5.0).json()
                        if resp.get("current_profile_id") != target_profile:
                            self._get_session().post(profile_url, json={"action": "switch", "profile_id": target_profile}, timeout=5.0)
                            self._post({"action": "proceed"}); continue
                    except: pass

                    if "continue" in opts: self._post({"action": "menu_select", "option": "continue"})
                    elif "singleplayer" in opts: self._post({"action": "menu_select", "option": "singleplayer"})
                    else: self._post({"action": "menu_select", "option": "main_menu"})
                elif screen == "game_over": self._post({"action": "menu_select", "option": "main_menu"})
                elif menu == "singleplayer": self._post({"action": "menu_select", "option": "standard"})
                elif menu == "character_select":
                    target_char = GLOBAL_CONFIG.get("TARGET_CHARACTER", "IRONCLAD").lower()
                    if target_char in opts: self._post({"action": "menu_select", "option": target_char})
                    self._post({"action": "menu_select", "option": "embark"})
                elif menu in ["profiles", "profile_select", "profile"]:
                    target_profile = {15526: 1, 15527: 2, 15528: 3, 15529: 1}.get(self.port, 1)
                    profile_url = f"http://127.0.0.1:{self.port}/api/v1/profiles"
                    try: self._get_session().post(profile_url, json={"action": "switch", "profile_id": target_profile}, timeout=5.0)
                    except: pass
                    self._post({"action": "proceed"})
                elif menu == "popup":
                    if "confirm" in opts: self._post({"action": "menu_select", "option": "confirm"})
                    elif "yes" in opts: self._post({"action": "menu_select", "option": "yes"})
                    elif "ignore" in opts: self._post({"action": "menu_select", "option": "ignore"})
                    else: self._post({"action": "proceed"})
                elif screen == "neow": self._post({"action": "choose_event_option", "index": 0})
                else: self._post({"action": "proceed"})
                
                time.sleep(0.05)
            else:
                self._reboot_game_client(reason="API Startup Timeout")
                continue
            continue 

    def _flatten_state(self, state):
        obs = np.zeros(GLOBAL_CONFIG["OBS_SIZE"], dtype=np.float32)
        if not state: return obs
        
        screen_map = {"monster": 1, "elite": 2, "boss": 3, "map": 4, "event": 5, "rest_site": 6, "rewards": 7, "card_reward": 8, "shop": 9, "neow": 10, "game_over": 11, "treasure": 12, "hand_select": 13, "card_select": 14, "enchanting": 14, "enchant": 14, "transform": 14, "simple_select": 14, "choose": 14, "upgrade_select": 14, "remove": 14, "menu": 18, "crystal_sphere": 19}
        st_type = state.get("state_type", "")
        obs[0] = screen_map.get(st_type, 0) / 20.0
        
        run, player = state.get("run", {}), state.get("player", {})
        obs[1], obs[2], obs[3] = run.get("act", 1)/5.0, run.get("floor", 0)/100.0, run.get("ascension", 0)/20.0
        max_hp = max(player.get("max_hp", 1), 1)
        obs[4], obs[5], obs[6], obs[7] = player.get("hp", 0)/max_hp, max_hp/150.0, player.get("block", 0)/100.0, player.get("gold", 0)/1000.0
        obs[8], obs[9] = player.get("energy", 0)/10.0, player.get("max_energy", 3)/10.0
        obs[10], obs[11], obs[12] = player.get("draw_pile_count", 0)/30.0, player.get("discard_pile_count", 0)/30.0, player.get("exhaust_pile_count", 0)/30.0
        
        status = {s.get("id"): s.get("amount", 0) for s in player.get("status", [])}
        core_statuses = ["Strength", "Dexterity", "Vulnerable", "Weak", "Frail", "No-Draw", "Entangled", "Artifact", "Barricade", "Dark Embrace", "Feel No Pain", "Corruption", "Evolve", "Fire Breathing", "Juggernaut", "Rupture"]
        for i, s_id in enumerate(core_statuses): obs[20 + i] = status.get(s_id, 0) / 10.0
        
        obs[50], obs[51] = player.get("stars", 0) / 10.0, player.get("orb_slots", 0) / 10.0
        for i, orb in enumerate(player.get("orbs", [])[:10]): obs[55 + i] = stable_hash(orb.get("id")) / 10000.0
        
        battle = state.get("battle", {})
        enemies = battle.get("enemies", [])
        for i, enemy in enumerate(enemies[:5]):
            base = 120 + (i * 36)
            obs[base], obs[base+1], obs[base+2] = stable_hash(enemy.get("entity_id"))/10000.0, enemy.get("hp", 0)/max(enemy.get("max_hp", 1), 1), enemy.get("block", 0)/100.0
            if (intents := enemy.get("intents", [])):
                obs[base+3] = stable_hash(intents[0].get("type"))/10000.0 
                try: 
                    label = intents[0].get("label", "0")
                    dmg = int(label.split("x")[0])*int(label.split("x")[1]) if "x" in label else int(label)
                    obs[base+4] = dmg / 50.0
                except: pass
            
            e_status = {s.get("id"): s.get("amount", 0) for s in enemy.get("status", [])}
            e_core = ["Strength", "Ritual", "Vulnerable", "Weak", "Artifact", "Metallicize", "Malleable", "Regrow", "Angry", "Curiosity"]
            for j, s_id in enumerate(e_core): obs[base + 5 + j] = e_status.get(s_id, 0) / 10.0

        def encode_card(card, base_idx):
            if not card: return
            c_cost_raw = card.get("cost") if card.get("cost") is not None else card.get("card_cost")
            desc = (card.get("description") or card.get("card_description") or "").lower()
            
            is_aoe = (card.get("target_type") in ["AllEnemies", "All"]) or ("all enemies" in desc) or ("all enemy" in desc)
            is_multi = ("times" in desc)
            is_scaling = any(kw in desc for kw in ["strength", "dexterity", "focus", "artifact", "plating"])
            
            if c_cost_raw == "X": cost_val = -0.2
            elif isinstance(c_cost_raw, (int, float)): cost_val = float(c_cost_raw) / 5.0
            elif isinstance(c_cost_raw, str) and (c_cost_raw.isdigit() or (c_cost_raw.startswith("-") and c_cost_raw[1:].isdigit())): cost_val = float(c_cost_raw) / 5.0
            else: cost_val = 0.0
            
            can_play_flag = card.get("can_play")
            if can_play_flag is None: can_play_flag = card.get("can_afford", True)
            
            obs[base_idx], obs[base_idx+1] = stable_hash(card.get("id") or card.get("card_id"))/10000.0, cost_val
            obs[base_idx+2], obs[base_idx+3] = (1.0 if card.get("is_upgraded") or card.get("card_upgraded") or "+" in (card.get("name") or "") else 0.0), (1.0 if can_play_flag else 0.0)
            obs[base_idx+4], obs[base_idx+5], obs[base_idx+6] = (1.0 if is_aoe else 0.0), (1.0 if is_multi else 0.0), (1.0 if is_scaling else 0.0)
            obs[base_idx+7], obs[base_idx+8], obs[base_idx+9] = (1.0 if "exhaust" in desc else 0.0), (1.0 if "block" in desc else 0.0), (1.0 if "damage" in desc else 0.0)

            found_ench = next((e for e in ENCHANTMENT_KEYWORDS if e in desc), None)
            obs[base_idx+10], obs[base_idx+11] = (stable_hash(found_ench)/10000.0 if found_ench else 0.0), (1.0 if found_ench else 0.0)
            obs[base_idx+12] = stable_hash(card.get("type") or card.get("card_type"))/10000.0
            obs[base_idx+13] = {"Common":0.1, "Uncommon":0.3, "Rare":0.5, "Basic":0.0, "Special":0.7, "Curse":0.9}.get(card.get("rarity") or card.get("card_rarity"), 0.0)
            obs[base_idx+14] = 1.0 if (card.get("is_selected") or card.get("is_chosen") or card.get("selected")) else 0.0
            
            dmg_match = re.search(r'deal (\d+) damage', desc)
            obs[base_idx+15] = (int(dmg_match.group(1)) / 100.0) if dmg_match else 0.0
            
            blk_match = re.search(r'gain (\d+) block', desc)
            obs[base_idx+16] = (int(blk_match.group(1)) / 100.0) if blk_match else 0.0

        hand = player.get("hand", [])
        if st_type=="hand_select": hand=state.get("hand_select", {}).get("cards", [])
        elif st_type=="card_reward": hand=state.get("card_reward", {}).get("cards", [])
        elif st_type in OVERLAY_KEYS: hand=state.get(st_type, {}).get("cards", [])
        for i, card in enumerate(hand[:15]): encode_card(card, 300 + (i * 40))
        
        draw_pile = player.get("draw_pile", [])
        for i, card in enumerate(draw_pile[:15]): encode_card(card, 900 + (i * 40))
        
        discard_pile = player.get("discard_pile", [])
        for i, card in enumerate(discard_pile[:15]): encode_card(card, 1500 + (i * 40))

        for i, node in enumerate(state.get("map", {}).get("nodes", [])[:100]):
            base = 2100 + (i * 10)
            obs[base], obs[base+1], obs[base+2] = node.get("col", 0)/7.0, node.get("row", 0)/15.0, stable_hash(node.get("type"))/10000.0

        relics = player.get("relics", [])
        if st_type=="relic_select": relics=state.get("relic_select", {}).get("relics", [])
        elif st_type=="treasure": relics=state.get("treasure", {}).get("relics", [])
        for i, relic in enumerate(relics[:40]):
            base = 3100 + (i * 10)
            obs[base], obs[base+1] = stable_hash(relic.get("id") or relic.get("relic_id"))/10000.0, (relic.get("counter")/10.0) if relic.get("counter") is not None else 0.0

        if st_type=="hand_select": obs[3500], obs[3501] = 1.0, len(state.get("hand_select", {}).get("selected_cards", []))/10.0
        elif st_type=="rewards":
            obs[3510], obs[3511] = 1.0, len(state.get("rewards", {}).get("items", []))/15.0
            rew = state.get("rewards", {})
            obs[3512] = 1.0 if rew.get("can_proceed") else 0.0
            for i, item in enumerate(rew.get("items", [])[:15]):
                base = 3520 + (i * 2)
                obs[base] = stable_hash(item.get("type")) / 10000.0
                if item.get("type") == "gold": obs[base+1] = item.get("gold_amount", 0) / 500.0
                else: obs[base+1] = stable_hash(item.get("potion_id") or item.get("relic_id") or item.get("card_id")) / 10000.0
        
        if st_type == "shop" or st_type == "fake_merchant":
            shop = state.get(st_type, {}).get("shop", {}) if st_type == "fake_merchant" else state.get("shop", {})
            obs[3512] = 1.0 if shop.get("can_proceed") else 0.0
            for i, item in enumerate(shop.get("items", [])[:15]):
                base = 3550 + (i * 4)
                obs[base], obs[base+1] = stable_hash(item.get("category"))/10000.0, stable_hash(item.get("card_id") or item.get("relic_id") or item.get("potion_id") or "card_removal")/10000.0
                obs[base+2], obs[base+3] = (item.get("price", item.get("cost", 0))/500.0), (1.0 if item.get("can_afford") and item.get("is_stocked") else 0.0)
        
        exhaust_pile = player.get("exhaust_pile", [])
        obs[3620] = len(exhaust_pile) / 50.0
        ex_types = {"Attack": 0, "Skill": 0, "Power": 0, "Status": 0, "Curse": 0}
        for c in exhaust_pile:
            c_type = c.get("type") or c.get("card_type", "Unknown")
            if c_type in ex_types: ex_types[c_type] += 1
        for i, (k, v) in enumerate(ex_types.items()): obs[3621 + i] = v / 30.0

        if st_type == "crystal_sphere":
            cs = state.get("crystal_sphere", {})
            for cell in cs.get("cells", []):
                idx = 3800 + (cell.get("y", 0) * 9) + cell.get("x", 0)
                if idx < 3900:
                    val = 0.0
                    if not cell.get("is_hidden"):
                        val = 1.0 if cell.get("is_good") else -1.0
                        if item := cell.get("item_type"): val += (stable_hash(item) / 20000.0)
                    obs[idx] = val

        for i, pet in enumerate(player.get("pets", [])[:5]):
            base = 3900 + (i * 20)
            obs[base], obs[base+1], obs[base+2] = stable_hash(pet.get("id"))/10000.0, pet.get("hp",0)/max(pet.get("max_hp",1),1), pet.get("block",0)/100.0
            for j, s in enumerate(pet.get("status", [])[:5]): obs[base+3+(j*2)], obs[base+4+(j*2)] = stable_hash(s.get("id"))/10000.0, s.get("amount",0)/10.0

        for i, pot in enumerate(player.get("potions", [])[:5]):
            base = 4000 + (i * 10)
            obs[base], obs[base+1] = stable_hash(pot.get("id")) / 10000.0, (1.0 if pot.get("can_use_in_combat") else 0.0)

        obs[4050] = self.training_phase / 6.0
        obs[4051] = self.stagnant_steps / 200.0
        obs[4052] = self.combat_turn_count / 20.0
        
        return obs

    def step(self, action_idx):
        self.total_episode_steps += 1
        for k in list(self.action_cooldowns.keys()):
            self.action_cooldowns[k] -= 1
            if self.action_cooldowns[k] <= 0: del self.action_cooldowns[k]
        
        old_state_hash = hashlib.md5(json.dumps(self.current_state, sort_keys=True).encode()).hexdigest()

        fresh_state = self._raw_state()
        if fresh_state: self.current_state = fresh_state
        if self.current_state.get("state_type") == "menu" and self.current_state.get("menu_screen") == "popup":
            msg = self.current_state.get("message", "").lower()
            if "corrupt" in msg or "not able to load" in msg:
                self.reboot_reason = f"Engine Bug: {msg}"; self.needs_reboot = True
                return self._flatten_state(self.current_state), 0.0, True, False, {"floor": self.previous_floor, "engine_bug": True}

        if self.stagnant_steps >= GLOBAL_CONFIG["STAGNATION"]["REBOOT_STEP"]:
            self.reboot_reason = f"Deadlock ({GLOBAL_CONFIG['STAGNATION']['REBOOT_STEP']} Steps)"; self.needs_reboot = True
            return self._flatten_state(self.current_state), 0.0, True, False, {"floor": self.previous_floor, "reboot_void": True}
        
        payload = FLAT_ACTIONS[int(action_idx)].copy()
        raw_screen = self.current_state.get("state_type", "unknown")
        player = self.current_state.get("player", {})
        battle = self.current_state.get("battle", {})
        enemies_data = battle.get("enemies", [])
        
        if "hand_select" in self.current_state: screen = "hand_select"
        elif any(k in self.current_state for k in OVERLAY_KEYS): screen = "card_select_overlay"
        elif enemies_data or battle.get("is_play_phase"): screen = "combat"
        else: screen = raw_screen

        vision = {"hp": f"{player.get('hp', 0)}/{player.get('max_hp', 1)}", "energy": f"{player.get('energy', 0)}/{player.get('max_energy', 3)}", "block": player.get("block", 0)}
        if enemies_data: vision["enemies"] = [f"{e.get('id', 'unk')} (HP:{e.get('hp',0)} B:{e.get('block',0)})" for e in enemies_data]
        
        is_autokick, rejected = False, not self.action_masks()[action_idx]
        if rejected: self.state_rejection_count += 1
        if self.state_rejection_count >= 20 or self.stagnant_steps >= GLOBAL_CONFIG["STAGNATION"]["REBOOT_STEP"]:
            is_autokick = True
            m = self.action_masks()
            valid_actions = [i for i, val in enumerate(m) if val]
            action_idx = valid_actions[0] if valid_actions else (80 if screen in ["monster", "elite", "boss", "combat"] else 123)
            payload = FLAT_ACTIONS[action_idx].copy()
            self._log_action(f"-> SMART-KICK: {payload.get('action')}")
        elif rejected:
            self._log_action(f"-> REJECTED: Masked.")
            if self.total_episode_steps >= 1500: self.needs_reboot = True; return self._flatten_state(self.current_state), 0.0, True, False, {"floor": self.previous_floor, "reboot_void": True}
            self.stagnant_steps += 1; return self._flatten_state(self.current_state), -1.0, False, False, {"floor": self.previous_floor}
        else: payload = FLAT_ACTIONS[int(action_idx)].copy()

        if is_autokick: self.state_rejection_count, self.stagnant_steps = 0, 0
        else: self._log_action(f"[Floor {self.current_state.get('run',{}).get('floor',0)} | {screen}] Attempting: {payload}", vision=vision)

        if payload.get("action") == "play_card":
            hand = player.get("hand", [])
            if (c_idx := payload.get("card_index")) is not None and c_idx < len(hand):
                if hand[c_idx].get("target_type") in ["AnyEnemy", "SingleTarget", "Target"]:
                    if (t_idx := payload.get("target_idx")) is not None and t_idx < len(enemies_data):
                        payload["target"] = enemies_data[t_idx].get("entity_id")
            if "target_idx" in payload: del payload["target_idx"]
        elif payload.get("action") == "use_potion":
            potion = next((p for p in player.get("potions", []) if p.get("slot") == payload.get("slot")), None)
            if potion and potion.get("target_type") in ["AnyEnemy", "SingleTarget", "Target"]:
                if (t_idx := payload.get("target_idx")) is not None and t_idx < len(enemies_data):
                        payload["target"] = enemies_data[t_idx].get("entity_id")
            if "target_idx" in payload: del payload["target_idx"]
        
        valid = self._post(payload)
        
        if payload.get("action") in ["choose_map_node", "proceed"]:
            new_state = self._raw_state()
            old_screen = self.current_state.get("state_type")
            for _ in range(40): 
                if not new_state: break
                if new_state.get("state_type") != old_screen:
                    st = new_state.get("state_type")
                    if st == "map" and (new_state.get("map", {}).get("next_options") or new_state.get("map", {}).get("can_proceed")): break
                    if st in ["event", "neow"] and (new_state.get("event", {}).get("options") or new_state.get("neow", {}).get("options") or new_state.get("event", {}).get("in_dialogue") or new_state.get("neow", {}).get("in_dialogue")): break
                    if st == "rewards" and (new_state.get("rewards", {}).get("items") or new_state.get("rewards", {}).get("can_proceed")): break
                    if st == "treasure" and (new_state.get("treasure", {}).get("items") or new_state.get("treasure", {}).get("can_proceed")): break
                    if st not in ["map", "event", "neow", "rewards", "treasure"]: break
                time.sleep(GLOBAL_CONFIG["STABILITY_THROTTLE"]); new_state = self._raw_state()
        elif payload.get("action") in ["select_card", "combat_select_card", "select_card_reward", "claim_reward", "end_turn", "confirm_selection", "combat_confirm_selection", "cancel_selection", "skip_card_reward", "skip_relic_selection", "claim_treasure_relic", "select_bundle", "confirm_bundle_selection", "cancel_bundle_selection", "select_relic", "shop_purchase", "choose_event_option", "advance_dialogue", "choose_rest_option", "menu_select", "crystal_sphere_set_tool", "crystal_sphere_click_cell"]:
            new_state = self._raw_state()
            for _ in range(40):
                if not new_state: break
                st = new_state.get("state_type")
                if payload.get("action") == "select_card_reward" and st != "card_reward": break
                if payload.get("action") == "skip_card_reward" and st != "card_reward": break
                if payload.get("action") == "select_bundle" and new_state.get("bundle_select", {}).get("can_confirm"): break
                if payload.get("action") == "confirm_bundle_selection" and st != "bundle_select": break
                if payload.get("action") == "claim_reward" and st == "rewards":
                    if len(new_state.get("rewards", {}).get("items", [])) != len(self.current_state.get("rewards", {}).get("items", [])): break
                if payload.get("action") == "claim_treasure_relic" and st == "treasure":
                    if len(new_state.get("treasure", {}).get("relics", [])) != len(self.current_state.get("treasure", {}).get("relics", [])): break
                if payload.get("action") in ["select_relic", "skip_relic_selection"] and st != "relic_select": break
                if payload.get("action") == "shop_purchase" and st in ["shop", "fake_merchant"]:
                    if new_state.get("player", {}).get("gold", 0) != self.current_state.get("player", {}).get("gold", 0): break

                new_hash = hashlib.md5(json.dumps(new_state, sort_keys=True).encode()).hexdigest()
                
                if payload.get("action") == "advance_dialogue" and st in ["event", "neow"]:
                    if new_hash != old_state_hash: break

                if new_hash != old_state_hash: break
                time.sleep(0.01); new_state = self._raw_state()
        else:
            time.sleep(GLOBAL_CONFIG["STABILITY_THROTTLE"])
            new_state = self._raw_state()

        if not valid or not new_state:
            time.sleep(GLOBAL_CONFIG["STABILITY_THROTTLE"]); new_state = self._raw_state()
            if not new_state:
                self.reboot_reason = "API Invalid"; self.needs_reboot = True
                return self._flatten_state(self.current_state), 0.0, True, False, {"floor": self.previous_floor, "reboot_void": True}

        if payload.get("action") in ["play_card", "use_potion"]: self.action_cooldowns[action_idx] = 5
        elif payload.get("action") in ["shop_purchase", "choose_rest_option"]: self.action_cooldowns[action_idx] = 10
        self.state_action_counts[action_idx] = self.state_action_counts.get(action_idx, 0) + 1
        if payload.get("action") == "select_card": self.internal_selection_history.append(payload.get("index"))
        elif payload.get("action") in ["confirm_selection", "proceed", "cancel_selection"]: self.internal_selection_history = []

        if new_state and new_state.get("state_type") in ["monster", "elite", "boss"] and new_state.get("battle", {}).get("is_play_phase") == False:
            for _ in range(100):
                time.sleep(0.05); new_state = self._raw_state()
                if not new_state or new_state.get("state_type") not in ["monster", "elite", "boss"] or new_state.get("battle", {}).get("is_play_phase") or any(k in new_state for k in OVERLAY_KEYS) or "hand_select" in new_state: break
        
        b_breakdown = {"floor": 0.0, "dmg": 0.0, "hp": 0.0, "bounty": 0.0, "tax": 0.0}
        new_screen, floor_now = new_state.get("state_type", ""), new_state.get("run", {}).get("floor", 0)
        if new_state.get("run", {}).get("is_victory", False): floor_now = 51
        player_now, reward, terminated = new_state.get("player", {}), 0.0, False
        hp_m = {1: 0.1, 2: 0.15, 3: 0.2, 4: 0.4, 5: 0.6}.get(self.training_phase, 0.8)
        max_hp = max(player_now.get("max_hp", 1), 1)
        hp_ratio = self.previous_hp / max_hp
        hp_delta = player_now.get("hp", 0) - self.previous_hp

        current_basic_count = sum(1 for c in player_now.get("deck", []) if (c.get("rarity") or c.get("card_rarity")) == "Basic")
        if current_basic_count < self.last_prog_basic_card_count and self.last_prog_basic_card_count > 0:
            b_breakdown["bounty"] += GLOBAL_CONFIG["BOUNTIES"]["CARD_REMOVAL"]
        self.last_prog_basic_card_count = current_basic_count

        if payload.get("action") == "use_potion" and self.last_prog_screen in ["elite", "boss"]:
            b_breakdown["bounty"] += GLOBAL_CONFIG["BOUNTIES"]["POTION_USE"]

        current_relic_count = len(player_now.get("relics", []))
        if current_relic_count > self.last_prog_relic_count and self.last_prog_relic_count >= 0:
            b_breakdown["bounty"] += GLOBAL_CONFIG["BOUNTIES"]["NEW_RELIC"]

        if self.last_prog_screen == "elite" and new_screen == "rewards": self.pending_elite_bounty = True
        if self.last_prog_screen == "boss" and new_screen == "rewards": self.pending_boss_bounty = True
        
        last_choice_id = "none"
        if payload.get("action") == "choose_rest_option":
            idx = payload.get("index", 0)
            rs_opts = self.current_state.get("rest_site", {}).get("options", [])
            if idx < len(rs_opts):
                last_choice_id = str(rs_opts[idx].get("id", "none")).lower()

        is_upgrade_screen = new_screen in ["card_select", "enchanting", "enchant", "upgrade_select"]
        if is_upgrade_screen or last_choice_id in ["smith", "enchant", "hatch", "dig", "toke"]:
            is_upgrade = (new_state.get("card_select", {}).get("screen_type") == "upgrade")
            is_enchant = (new_screen in ["enchanting", "enchant", "upgrade_select"])
            if is_upgrade or is_enchant or last_choice_id in ["smith", "enchant", "hatch", "dig", "toke"]:
                self.pending_smith_bounty = True

        if last_choice_id == "rest":
            potential_heal = max_hp * 0.3
            overflow = (self.previous_hp + potential_heal) - max_hp
            if overflow <= 5:
                b_breakdown["bounty"] += GLOBAL_CONFIG["BOUNTIES"]["SMITH_BASE"]
            else:
                waste_penalty = min(GLOBAL_CONFIG["BOUNTIES"]["SMITH_BASE"], (overflow / potential_heal) * (GLOBAL_CONFIG["BOUNTIES"]["SMITH_BASE"] * 1.33))
                b_breakdown["bounty"] -= waste_penalty
                self._log_action(f"-> WASTEFUL HEAL: Overflew by {overflow:.1f} HP.")

        if new_screen == "elite" and self.last_prog_screen != "elite":
            if hp_ratio > 0.5:
                b_breakdown["bounty"] += GLOBAL_CONFIG["BOUNTIES"]["ELITE_SIGHT"]
            elif hp_ratio < 0.3:
                e_penalty = -5.0 if self.training_phase <= 1 else -10.0 if self.training_phase == 2 else -15.0
                b_breakdown["bounty"] += e_penalty

        if payload.get("action") == "end_turn":
            self.combat_turn_count += 1
        if new_screen in ["monster", "elite", "boss"] and self.last_prog_screen not in ["monster", "elite", "boss"]: 
            self.combat_turn_count = 0

        base_tax = abs(GLOBAL_CONFIG["STEP_TAX"])
        if self.stagnant_steps > GLOBAL_CONFIG["STAGNATION"]["HIGH_PENALTY_STEP"]: base_tax = 0.20
        b_breakdown["tax"] -= base_tax

        if new_screen != self.last_prog_screen:
            b_breakdown["tax"] += abs(GLOBAL_CONFIG["STEP_TAX"])

        if self.combat_turn_count > 20: b_breakdown["tax"] -= (self.combat_turn_count - 20) * 0.1
        
        if self.stagnant_steps > GLOBAL_CONFIG["STAGNATION"]["CRITICAL_STEP"]: b_breakdown["tax"] -= 20.0
        elif self.stagnant_steps > GLOBAL_CONFIG["STAGNATION"]["HIGH_PENALTY_STEP"]: b_breakdown["tax"] -= 10.0 
        elif self.stagnant_steps > GLOBAL_CONFIG["STAGNATION"]["LOW_PENALTY_STEP"]: b_breakdown["tax"] -= 2.0  
        if floor_now > self.previous_floor:
            b_breakdown["floor"] += GLOBAL_CONFIG["BOUNTIES"]["FLOOR_CLIMB"]
            
            if floor_now in [17, 33, 48]:
                b_breakdown["bounty"] += GLOBAL_CONFIG["BOUNTIES"]["BOSS_REACH"]

            if self.pending_boss_bounty: b_breakdown["bounty"] += GLOBAL_CONFIG["BOUNTIES"]["BOSS_KILL"]; self.pending_boss_bounty = False
            if self.pending_elite_bounty: b_breakdown["bounty"] += GLOBAL_CONFIG["BOUNTIES"]["ELITE_KILL"]; self.pending_elite_bounty = False
            if self.pending_smith_bounty:
                potential_heal = max_hp * 0.3
                would_overflow = (self.previous_hp + potential_heal) > (max_hp + 5)
                
                if would_overflow:
                    s_m = 1.5 if hp_ratio > 0.8 else 1.2 if hp_ratio > 0.5 else 0.8
                    b_breakdown["bounty"] += 20.0
                else:
                    s_m = 1.0 if hp_ratio > 0.6 else 0.5 if hp_ratio > 0.4 else -1.0 if hp_ratio > 0.25 else -3.0
                
                b_breakdown["bounty"] += (GLOBAL_CONFIG["BOUNTIES"]["SMITH_BASE"] * s_m); self.pending_smith_bounty = False
            
            if len(player_now.get("deck", [])) > self.last_prog_deck_size and floor_now < 16 and self.last_prog_deck_size > 0:
                b_breakdown["bounty"] += GLOBAL_CONFIG["BOUNTIES"]["DRAFT_CARD"]
            
            self.previous_floor = floor_now
            
        if hp_delta > 0:
            if hp_ratio > 0.9: b_breakdown["hp"] += hp_delta * -0.5
            elif hp_ratio > 0.8: pass
            elif hp_ratio > 0.5: b_breakdown["hp"] += hp_delta * 0.5
            elif hp_ratio > 0.3: b_breakdown["hp"] += hp_delta * 1.0
            else: b_breakdown["hp"] += hp_delta * 2.0
        elif hp_delta < 0:
            b_breakdown["hp"] += hp_delta * hp_m

        self.previous_hp = player_now.get("hp", 0)
        enemies_now = new_state.get("battle", {}).get("enemies", [])
        for i, e in enumerate(enemies_now):
            e_key = f"{e.get('entity_id')}_{i}"
            e_hp = e.get("hp", 0)
            if e_key not in self.lowest_enemy_hp_seen: self.lowest_enemy_hp_seen[e_key] = e_hp
            elif e_hp < self.lowest_enemy_hp_seen[e_key]: 
                b_breakdown["dmg"] += (self.lowest_enemy_hp_seen[e_key] - e_hp) * 0.1
                self.lowest_enemy_hp_seen[e_key] = e_hp
        if not enemies_now: self.lowest_enemy_hp_seen = {}

        curr_hp_sum = sum(e.get("hp", 0) for e in enemies_now) + player_now.get("hp", 0)
        hard_prog = (floor_now != self.last_prog_floor or 
                     curr_hp_sum != self.last_prog_hp_sum or 
                     player_now.get("gold", 0) != self.last_prog_gold or
                     player_now.get("block", 0) != self.last_prog_player_block or
                     player_now.get("energy", 0) != self.last_prog_energy or
                     len(player_now.get("deck", [])) != self.last_prog_deck_size or
                     len(player_now.get("relics", [])) != self.last_prog_relic_count)
        screen_prog = (new_screen != self.last_prog_screen)

        sel_count = len(self.internal_selection_history)
        sel_prog = (sel_count > 0 and sel_count != getattr(self, 'last_sel_count', -1))

        if hard_prog:
            self.stagnant_steps, self.state_action_counts, self.state_rejection_count = 0, {}, 0
            self.sel_prog_steps = 0
        elif screen_prog:
            self.stagnant_steps, self.state_rejection_count = 0, 0
            self.sel_prog_steps = 0
        elif sel_prog and getattr(self, 'sel_prog_steps', 0) < 20:
            self.state_rejection_count = 0
            self.sel_prog_steps = getattr(self, 'sel_prog_steps', 0) + 1
        else: self.stagnant_steps += 1
        self.last_sel_count = sel_count

        if new_screen != self.last_prog_screen:
            self.internal_selection_history = []
            self.sel_prog_steps = 0

        self.last_prog_screen, self.last_prog_floor, self.last_prog_hp_sum, self.last_prog_gold = new_screen, floor_now, curr_hp_sum, player_now.get("gold", 0)
        self.last_prog_player_block, self.last_prog_energy, self.last_prog_deck_size = player_now.get("block", 0), player_now.get("energy", 0), len(player_now.get("deck", []))
        self.last_prog_relic_count = len(player_now.get("relics", []))

        if self.total_episode_steps >= 1500:
            self.reboot_reason = "Timeout (1500 Steps)"; self.needs_reboot = True
            return self._flatten_state(self.current_state), 0.0, True, False, {"floor": self.previous_floor, "reboot_void": True}

        reward = sum(b_breakdown.values())
        if new_screen == "game_over":
            if self.training_phase <= 3:
                phase_targets = {1: 17, 2: 33, 3: 48}
                target = phase_targets.get(self.training_phase, 48)
                death_penalty = -50.0 + (min(floor_now / target, 1.0) * 40.0)
            else:
                death_penalty = -50.0
            reward += death_penalty; terminated = True
            self._log_action(f"-> FATAL: Game Over. Penalty: {death_penalty:.1f}")
        elif new_state.get("run", {}).get("is_victory", False): 
            terminated = True
        self.current_state = new_state

        obs = self._flatten_state(new_state)
        final_reward = reward * (2.0 if floor_now >= 45 else 1.0)
        if not terminated and new_screen != "game_over":
            sources = ", ".join([f"{k.capitalize()}: {v:+.1f}" for k, v in b_breakdown.items() if v != 0])
            self._log_action(f"-> ACCEPTED. Net: {final_reward:+.2f} ({sources})")
        return obs, final_reward, terminated, False, {"floor": floor_now}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.needs_reboot: 
            is_bloat = "Storage Bloat" in str(self.reboot_reason)
            self._reboot_game_client()
            self.needs_reboot = False
            if is_bloat: self.full_purge_cleanup_pending = True
            else: self.ghost_save_cleanup_pending = True
        self._ensure_fresh_run()
        state = self.current_state
        self.previous_floor, self.previous_hp = state.get("run", {}).get("floor", 0), state.get("player", {}).get("hp", 0)
        self.combat_step_count, self.total_episode_steps, self.stagnant_steps, self.combat_turn_count = 0, 0, 0, 0
        self.action_cooldowns = {}
        self.last_prog_screen, self.last_prog_floor, self.last_prog_hp_sum, self.last_prog_gold = "none", -1, -1, -1
        self.last_prog_energy, self.last_prog_player_block, self.last_prog_deck_size = -1, -1, -1
        self.last_prog_relic_count = -1
        self.pending_boss_bounty, self.pending_elite_bounty, self.pending_smith_bounty = False, False, False
        self.lowest_enemy_hp_seen = {}; return self._flatten_state(state), {}