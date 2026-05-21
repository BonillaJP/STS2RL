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

# Hash space compressed to 10000 for dense MLP representations while avoiding collisions
def stable_hash(name, max_val=10000):
    if not name: return 0
    return (int(hashlib.md5(str(name).encode()).hexdigest(), 16) % max_val) + 1

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
for c in range(25): FLAT_ACTIONS.append({"action": "select_card", "index": c}) 
FLAT_ACTIONS.extend([{"action": "confirm_selection"}, {"action": "cancel_selection"}])
for b in range(5): FLAT_ACTIONS.append({"action": "select_bundle", "index": b})
FLAT_ACTIONS.extend([{"action": "confirm_bundle_selection"}, {"action": "cancel_bundle_selection"}])
for r in range(5): FLAT_ACTIONS.append({"action": "select_relic", "index": r})
FLAT_ACTIONS.append({"action": "skip_relic_selection"})
for r in range(5): FLAT_ACTIONS.append({"action": "claim_treasure_relic", "index": r})
MENU_OPTS = ["main_menu", "singleplayer", "standard", "IRONCLAD", "embark", "continue", "yes", "no", "abandon", "settings"]
for m in MENU_OPTS: FLAT_ACTIONS.append({"action": "menu_select", "option": m})
FLAT_ACTIONS.append({"action": "crystal_sphere_set_tool", "tool": "big"})
FLAT_ACTIONS.append({"action": "crystal_sphere_set_tool", "tool": "small"})
for y in range(11):
    for x in range(9): FLAT_ACTIONS.append({"action": "crystal_sphere_click_cell", "x": x, "y": y})
FLAT_ACTIONS.append({"action": "crystal_sphere_proceed"})

ENCHANTMENT_KEYWORDS = ["adroit", "corrupted", "glam", "goopy", "imbued", "instinct", "momentum", "nimble", "perfect fit", "royally approved", "sharp", "slither", "soul's power", "sown", "swift", "vigorous"]

class SlayTheSpire2Env(gym.Env):
    def __init__(self, port=15526, game_path=None, user_data_dir=None, force_fresh=False, is_eval=False):
        super().__init__()
        self.port = port
        
        # Game path detection
        if game_path and os.path.isdir(game_path):
            node_id = {15526:1, 15527:2, 15528:3, 15529:4}.get(port, 1)
            game_path = os.path.join(game_path, f"Node {node_id}.exe")
        
        if not game_path or not os.path.exists(game_path):
            node_id = {15526:1, 15527:2, 15528:3, 15529:4}.get(port, 1)
            base_node_path = f"C:\\Program Files (x86)\\Steam\\steamapps\\common\\STS2_Node_{node_id}"
            game_path = os.path.join(base_node_path, f"Node {node_id}.exe")
            if not os.path.exists(game_path):
                game_path = os.path.join(base_node_path, "SlayTheSpire2.exe")
        
        self.game_path = game_path
        self.user_data_dir = user_data_dir
        self.force_fresh = force_fresh
        self.is_eval = is_eval
        self.game_url = f"http://127.0.0.1:{port}/api/v1/singleplayer"
        self.action_space = spaces.Discrete(len(FLAT_ACTIONS))
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(1536,), dtype=np.float32)

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
        self.last_prog_player_block = -1
        self.last_prog_enemy_block_sum = -1
        self.last_prog_selected_cards = []
        
        # Bounty Flags
        self.pending_elite_bounty = False
        self.pending_boss_bounty = False
        self.pending_smith_bounty = False
        self.global_step = 0
        self.last_state_hash = ""
        self.state_action_counts = {} # Maps action_idx -> consecutive failure count
        self._vacuum_disk()
        
        # Fresh Debug Trace for this run
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
                    f.write(json.dumps({"ts": str(datetime.datetime.now()), "msg": "--- Trace Rotated (1GB Cap) ---"}) + "\n")
            
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

    def _get_session(self):
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

    def _vacuum_disk(self):
        # Disk Vacuum: Clean Godot logs and Sentry reports to preserve disk space.
        appdata = os.getenv('APPDATA')
        if appdata:
            game_data_path = os.path.join(appdata, "SlayTheSpire2")
            log_dir = os.path.join(game_data_path, "logs")
            try:
                if os.path.exists(log_dir):
                    for log_file in glob.glob(os.path.join(log_dir, "godot*.log")):
                        if "godot.log" not in log_file:
                            os.remove(log_file)
            except: pass
            
            sentry_dir = os.path.join(game_data_path, "sentry", "reports")
            try:
                if os.path.exists(sentry_dir):
                    for dmp in glob.glob(os.path.join(sentry_dir, "*.dmp")):
                        os.remove(dmp)
            except: pass

    def _reboot_game_client(self, reason=None):
        print(f"\n[REBOOT] Port {self.port} is frozen. Physically restarting Node...")
        target_pid = "Unknown"
        killed = False
        try:
            cmd = f'netstat -ano | findstr LISTENING | findstr :{self.port}'
            output = subprocess.check_output(cmd, shell=True).decode('utf-8')
            for line in output.strip().split('\n'):
                parts = line.split()
                if len(parts) >= 5:
                    pid = parts[-1]
                    if pid != "4": 
                        target_pid = pid
                        subprocess.run(["taskkill", "/F", "/PID", pid], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        print(f"[REBOOT] Killed PID {pid} via Port {self.port}")
                        killed = True
        except: pass
        
        if not killed:
            try:
                folder_fragment = os.path.basename(os.path.dirname(self.game_path))
                ps_get = f"Get-Process -ErrorAction SilentlyContinue | Where-Object {{ $_.Path -like '*{folder_fragment}*' }} | Sort-Object WS -Descending | Select-Object -ExpandProperty Id"
                pid_out = subprocess.check_output(["powershell", "-NoProfile", "-Command", ps_get]).decode('utf-8').strip()
                if pid_out: 
                    target_pid = pid_out.split()[0]
                
                ps_kill = f"Get-Process -ErrorAction SilentlyContinue | Where-Object {{ $_.Path -like '*{folder_fragment}*' }} | Stop-Process -Force"
                subprocess.run(["powershell", "-NoProfile", "-Command", ps_kill], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print(f"[REBOOT] Attempted PowerShell Force-Kill on folder fragment '{folder_fragment}' (PID {target_pid})")
                killed = True
            except Exception as e:
                print(f"[REBOOT] PowerShell kill failed: {e}")

        log_reason = reason if reason else self.reboot_reason
        self._log_reboot(log_reason, pid=target_pid)
        self.reboot_reason = "Stall/Deadlock Recovery"
        self._vacuum_disk()

        print(f"[REBOOT] Waiting for Port {self.port} to be released...")
        for _ in range(30):
            try:
                cmd = f'netstat -ano | findstr LISTENING | findstr :{self.port}'
                output = subprocess.check_output(cmd, shell=True).decode('utf-8')
                if not output.strip(): break
            except: break
            time.sleep(1.0)

        time.sleep(2.0) 
        try:
            game_dir = os.path.dirname(self.game_path)
            game_exe = os.path.basename(self.game_path)
            cmd = f'cmd /c start "" "{game_exe}"'
            subprocess.Popen(cmd, cwd=game_dir, shell=True)
            print(f"[REBOOT] Relaunched Independent (via start): {self.game_path}")
        except Exception as e: print(f"[REBOOT] FAILED to relaunch: {e}")
        print(f"[REBOOT] Waiting 30s for API to warm up...")
        time.sleep(30.0)

    def action_masks(self):
        # --- STATE-FIRST ARCHITECTURE ---
        state = self.current_state
        mask = np.zeros(len(FLAT_ACTIONS), dtype=bool)
        if not state: return mask
        
        # --- SCREEN DETECTION ---
        raw_screen = state.get("state_type", "unknown")
        player = state.get("player", {})
        battle = state.get("battle", {})
        enemies = battle.get("enemies", [])
        
        OVERLAY_KEYS = ["card_select", "enchanting", "enchant", "transform", "simple_select", "choose", "upgrade_select", "remove"]
        if raw_screen == "hand_select" or "hand_select" in state: screen = "hand_select"
        elif raw_screen in OVERLAY_KEYS or any(k in state for k in OVERLAY_KEYS): screen = "card_select_overlay"
        elif raw_screen == "bundle_select" or "bundle_select" in state: screen = "bundle_select"
        elif raw_screen == "relic_select" or "relic_select" in state: screen = "relic_select"
        elif raw_screen == "card_reward" or "card_reward" in state: screen = "card_reward"
        elif raw_screen == "rewards" or "rewards" in state: screen = "rewards"
        elif raw_screen == "treasure" or "treasure" in state: screen = "treasure"
        elif raw_screen == "crystal_sphere" or "crystal_sphere" in state: screen = "crystal_sphere"
        elif len(enemies) > 0: screen = "combat"
        else: screen = raw_screen

        def set_mask(start_idx, count, valid_indices=None):
            if valid_indices is None: mask[start_idx:start_idx+count] = True
            else:
                for idx in valid_indices:
                    if idx is not None and idx < count: mask[start_idx + idx] = True
        
        # --- POTION DISCARD MASKING ---
        potions = player.get("potions", [])
        if len(potions) >= player.get("max_potion_slots", 3):
            if screen in ["combat", "rewards", "shop", "fake_merchant", "event", "neow", "treasure", "rest_site"]:
                for pot in potions:
                    slot_idx = pot.get("slot")
                    if slot_idx is not None and slot_idx < 5:
                        mask[75 + slot_idx] = True

        if screen == "combat":
            if battle.get("is_play_phase"):
                mask[80] = True # End Turn
                for c_idx, card in enumerate(player.get("hand", [])[:10]):
                    can_play_flag = card.get("can_play")
                    if can_play_flag is None: can_play_flag = card.get("can_afford", True)
                    if can_play_flag:
                        if card.get("target_type") in ["AnyEnemy", "SingleTarget", "Target"]:
                            for t_idx in range(min(5, len(enemies))): mask[c_idx * 5 + t_idx] = True
                        else: mask[c_idx * 5] = True
                
                for p_idx, potion in enumerate(potions[:5]):
                    can_use_flag = potion.get("can_use_in_combat")
                    if can_use_flag is None: can_use_flag = potion.get("can_afford", True)
                    if can_use_flag:
                        if potion.get("target_type") in ["AnyEnemy", "SingleTarget", "Target"]:
                            for t_idx in range(min(5, len(enemies))): mask[50 + p_idx * 5 + t_idx] = True
                        else: mask[50 + p_idx * 5] = True 
        elif screen == "hand_select":
            hs = state.get("hand_select", {})
            match = re.search(r'\b(\d+)\b', hs.get("prompt", ""))
            max_select = int(match.group(1)) if match else 1
            can_confirm = hs.get("can_confirm", False)
            
            selected_indices = [c.get("index") for c in hs.get("selected_cards", [])]
            if not selected_indices: selected_indices = [c.get("index") for c in hs.get("selected", [])]
            if not selected_indices: selected_indices = [c.get("index") for c in hs.get("cards", []) if c.get("is_selected", False)]
            
            effectively_selected = len(selected_indices)
            if can_confirm and max_select == 1: effectively_selected = 1
            
            if effectively_selected < max_select:
                selectable = [c.get("index") for c in hs.get("cards", []) if c.get("index") not in selected_indices]
                set_mask(81, 10, selectable)
            if can_confirm: mask[91] = True
        elif screen == "rewards":
            rew = state.get("rewards", {})
            set_mask(92, 15, [i.get("index") for i in rew.get("items", [])])
            if rew.get("can_proceed") or not rew.get("items"): mask[123] = True # proceed
        elif screen == "card_reward":
            cr = state.get("card_reward", {})
            set_mask(107, 15, [c.get("index") for c in cr.get("cards", [])])
            if cr.get("can_skip"): mask[122] = True
        elif screen in ["event", "neow"]:
            ev = state.get("event") or state.get("neow") or {}
            opts = ev.get("options", [])
            if ev.get("in_dialogue"):
                mask[134] = True # advance_dialogue
            else:
                selectable = [o.get("index") for o in opts if not o.get("is_locked") and not o.get("was_chosen")]
                if selectable: set_mask(124, 10, selectable)
                if not selectable and not ev.get("in_dialogue"): mask[123] = True
        elif screen in ["room", "unknown"]: mask[123] = True
        elif screen == "rest_site":
            rs = state.get("rest_site", {})
            set_mask(135, 5, [o.get("index") for o in rs.get("options", []) if o.get("is_enabled")])
            if rs.get("can_proceed"): mask[123] = True
        elif screen == "shop" or screen == "fake_merchant":
            shop = state.get(screen, {}).get("shop", {}) if screen == "fake_merchant" else state.get("shop", {})
            gold = self.current_state.get("player", {}).get("gold", 0)
            set_mask(140, 15, [i.get("index") for i in shop.get("items", []) if i.get("price", i.get("cost", 9999)) <= gold and i.get("is_stocked")])
            mask[123] = True 
        elif screen == "map":
            set_mask(155, 5, [o.get("index") for o in state.get("map", {}).get("next_options", []) if o.get("is_selectable", True)])
        elif screen == "card_select_overlay":
            cs = next((state.get(k, {}) for k in OVERLAY_KEYS if k in state), {})
            prompt = cs.get("prompt", "")
            match = re.search(r'\b(\d+)\b', prompt)
            max_sel = int(match.group(1)) if match else 1
            
            # --- ADAPTIVE SELECTION LIMIT ---
            selected_indices = [c.get("index") for c in cs.get("selected_cards", [])]
            if not selected_indices: selected_indices = [c.get("index") for c in cs.get("selected", [])]
            if not selected_indices: selected_indices = [c.get("index") for c in cs.get("selected_card_indices", [])]
            if not selected_indices: selected_indices = [c.get("index") for c in cs.get("cards", []) if (c.get("is_selected", False) or c.get("is_chosen", False) or c.get("selected", False))]
            
            can_confirm = cs.get("can_confirm", False)
            if not can_confirm:
                valid_grid_indices = [c.get("index") for c in cs.get("cards", []) if c.get("index") not in selected_indices]
                if cs.get("screen_type") == "choose":
                    if len(selected_indices) < max_sel:
                        set_mask(160, 25, valid_grid_indices)
                else:
                    if len(selected_indices) < max_sel:
                        set_mask(160, 25, valid_grid_indices)
            
            is_mid_selection = (can_confirm or len(selected_indices) > 0)
            proceed_exhausted = self.state_action_counts.get(123, 0) >= 10
            confirm_exhausted = self.state_action_counts.get(185, 0) >= 10
            
            if can_confirm: mask[185] = True
            if cs.get("can_proceed"): mask[123] = True
            if cs.get("can_cancel"):
                if not is_mid_selection or (proceed_exhausted and confirm_exhausted):
                    mask[186] = True

        elif screen == "bundle_select":
            bs = state.get("bundle_select", {})
            set_mask(187, 5, [b.get("index") for b in bs.get("bundles", [])])
            if bs.get("can_confirm"): mask[192] = True
            if bs.get("can_cancel"): mask[193] = True
        elif screen == "relic_select":
            rs = state.get("relic_select", {})
            set_mask(194, 5, [r.get("index") for r in rs.get("relics", [])])
            if rs.get("can_skip"): mask[199] = True
        elif screen == "treasure":
            set_mask(200, 5, [r.get("index") for r in state.get("treasure", {}).get("relics", [])])
            if state.get("treasure", {}).get("can_proceed"): mask[123] = True
        elif screen == "menu":
            opts = [o.get("name", "").lower() if isinstance(o, dict) else o.lower() for o in state.get("options", [])]
            for i, m in enumerate(MENU_OPTS):
                if m.lower() in opts and m.lower() not in ["settings", "abandon"]: mask[205 + i] = True
        elif screen == "game_over": mask[205] = True
        elif screen == "crystal_sphere":
            cs = state.get("crystal_sphere", {})
            if cs.get("can_use_big_tool"): mask[215] = True
            if cs.get("can_use_small_tool"): mask[216] = True
            for cell in cs.get("clickable_cells", []): mask[217 + (cell.get("y", 0) * 9) + cell.get("x", 0)] = True
            if cs.get("can_proceed"): mask[316] = True
        
        # --- GLOBAL STAGNATION SUPPRESSION ---
        current_hash = stable_hash(json.dumps(state, sort_keys=True))
        if current_hash == self.last_state_hash:
            for act_idx, count in self.state_action_counts.items():
                if count >= 10:
                    mask[act_idx] = False 

        if not mask.any():
            if screen in ["monster", "elite", "boss"]: mask[80] = True
            elif screen not in ["map", "event", "hand_select", "neow"]: mask[123] = True
        return mask

    def _post(self, payload):
        try:
            resp = self._get_session().post(self.game_url, json=payload, timeout=30.0)
            if resp.status_code != 200: return False
            try:
                if resp.json().get("status") == "error": return False
            except: pass
            return True
        except: return False

    def _raw_state(self):
        try: return self._get_session().get(f"{self.game_url}?format=json", timeout=30.0).json()
        except: return {}

    def _ensure_fresh_run(self):
        for i in range(100):
            state = self._raw_state()
            if not state: time.sleep(1.0); continue
            screen, menu = state.get("state_type", ""), state.get("menu_screen", "")
            ACTIVE = ["monster", "elite", "boss", "map", "event", "rest_site", "rewards", "card_reward", "shop", "neow", "hand_select", "treasure", "card_select", "bundle_select", "relic_select", "fake_merchant", "crystal_sphere"]
            if screen in ACTIVE: 
                while state and state.get("state_type") in ["monster", "elite", "boss"] and state.get("battle", {}).get("is_play_phase") == False:
                    time.sleep(0.05)
                    state = self._raw_state()
                    if not state: break
                self.current_state = state; return
            
            opts = [o.get("name", "").lower() if isinstance(o, dict) else o.lower() for o in state.get("options", [])]
            
            if menu == "main":
                if "continue" in opts: self._post({"action": "menu_select", "option": "continue"})
                elif "singleplayer" in opts: self._post({"action": "menu_select", "option": "singleplayer"})
                else: self._post({"action": "menu_select", "option": "main_menu"})
            elif screen == "game_over": self._post({"action": "menu_select", "option": "main_menu"})
            elif menu == "singleplayer": self._post({"action": "menu_select", "option": "standard"})
            elif menu == "character_select":
                if "ironclad" in opts: self._post({"action": "menu_select", "option": "ironclad"})
                self._post({"action": "menu_select", "option": "embark"})
            elif menu == "popup":
                if "confirm" in opts: self._post({"action": "menu_select", "option": "confirm"})
                elif "yes" in opts: self._post({"action": "menu_select", "option": "yes"})
                elif "ignore" in opts: self._post({"action": "menu_select", "option": "ignore"})
                else: self._post({"action": "proceed"})
            elif screen == "neow": self._post({"action": "choose_event_option", "index": 0})
            else: self._post({"action": "proceed"})
            time.sleep(0.5)
        self._reboot_game_client(reason="API Startup Timeout"); return self._ensure_fresh_run()

    def _flatten_state(self, state):
        obs = np.zeros(1536, dtype=np.float32)
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
        for i, pot in enumerate(player.get("potions", [])[:5]):
            base = 70 + (i * 10)
            obs[base], obs[base+1] = stable_hash(pot.get("id")) / 10000.0, (1.0 if pot.get("can_use_in_combat") else 0.0)
        enemies = state.get("battle", {}).get("enemies", [])
        for i, enemy in enumerate(enemies[:5]):
            base = 120 + (i * 40)
            obs[base], obs[base+1], obs[base+2] = stable_hash(enemy.get("entity_id"))/10000.0, enemy.get("hp", 0)/max(enemy.get("max_hp", 1), 1), enemy.get("block", 0)/100.0
            if (intents := enemy.get("intents", [])):
                obs[base+3] = stable_hash(intents[0].get("type"))/10000.0 
                try: 
                    label = intents[0].get("label", "0")
                    dmg = int(label.split("x")[0])*int(label.split("x")[1]) if "x" in label else int(label)
                    obs[base+4] = dmg / 50.0
                except: pass
        def encode_card(card, base_idx):
            if not card: return
            c_id = card.get("id") or card.get("card_id")
            c_cost_raw = card.get("cost") if card.get("cost") is not None else card.get("card_cost")
            desc = (card.get("description") or card.get("card_description") or "").lower()
            if c_cost_raw == "X": cost_val = -0.2
            elif isinstance(c_cost_raw, (int, float)): cost_val = float(c_cost_raw) / 5.0
            elif isinstance(c_cost_raw, str) and (c_cost_raw.isdigit() or (c_cost_raw.startswith("-") and c_cost_raw[1:].isdigit())):
                cost_val = float(c_cost_raw) / 5.0
            else: cost_val = 0.0
            
            can_play_flag = card.get("can_play")
            if can_play_flag is None: can_play_flag = card.get("can_afford", True)
            
            obs[base_idx], obs[base_idx+1] = stable_hash(c_id)/10000.0, cost_val
            obs[base_idx+2], obs[base_idx+3] = (1.0 if card.get("is_upgraded") or card.get("card_upgraded") or "+" in (card.get("name") or "") else 0.0), (1.0 if can_play_flag else 0.0)
            obs[base_idx+4], obs[base_idx+5], obs[base_idx+6], obs[base_idx+7] = (1.0 if "exhaust" in desc else 0.0), (1.0 if "strength" in desc else 0.0), (1.0 if "block" in desc else 0.0), (1.0 if "damage" in desc else 0.0)
            
            found_ench = next((e for e in ENCHANTMENT_KEYWORDS if e in desc), None)
            obs[base_idx+8] = stable_hash(found_ench) / 10000.0 if found_ench else 0.0
            obs[base_idx+9] = 1.0 if found_ench else 0.0
            
            obs[base_idx+10] = stable_hash(card.get("type") or card.get("card_type"))/10000.0
            obs[base_idx+11] = {"Common":0.1, "Uncommon":0.3, "Rare":0.5, "Basic":0.0, "Special":0.7, "Curse":0.9}.get(card.get("rarity") or card.get("card_rarity"), 0.0)
            obs[base_idx+12] = 1.0 if (card.get("is_selected") or card.get("is_chosen") or card.get("selected")) else 0.0
        hand = player.get("hand", [])
        if st_type=="hand_select": hand=state.get("hand_select", {}).get("cards", [])
        elif st_type=="card_reward": hand=state.get("card_reward", {}).get("cards", [])
        elif st_type in OVERLAY_KEYS: hand=state.get(st_type, {}).get("cards", [])
        for i, card in enumerate(hand[:10]): encode_card(card, 320 + (i * 30))
        if st_type=="hand_select": obs[620], obs[621] = 1.0, len(state.get("hand_select", {}).get("selected_cards", []))/10.0
        elif st_type=="rewards":
            obs[630], obs[631] = 1.0, len(state.get("rewards", {}).get("items", []))/15.0
            rew = state.get("rewards", {})
            obs[632] = 1.0 if rew.get("can_proceed") else 0.0
            for i, item in enumerate(rew.get("items", [])[:15]):
                base = 640 + (i * 2)
                obs[base] = stable_hash(item.get("type")) / 10000.0
                if item.get("type") == "gold": obs[base+1] = item.get("gold_amount", 0) / 500.0
                else: 
                    v_id = item.get("potion_id") or item.get("relic_id") or item.get("card_id")
                    obs[base+1] = stable_hash(v_id) / 10000.0
        if st_type == "shop" or st_type == "fake_merchant":
            shop = state.get(st_type, {}).get("shop", {}) if st_type == "fake_merchant" else state.get("shop", {})
            obs[632] = 1.0 if shop.get("can_proceed") else 0.0
            for i, item in enumerate(shop.get("items", [])[:15]):
                base = 671 + (i * 4)
                obs[base] = stable_hash(item.get("category")) / 10000.0
                v_id = item.get("card_id") or item.get("relic_id") or item.get("potion_id") or "card_removal"
                obs[base+1] = stable_hash(v_id) / 10000.0
                obs[base+2] = item.get("price", item.get("cost", 0)) / 500.0
                obs[base+3] = 1.0 if item.get("can_afford") and item.get("is_stocked") else 0.0
        elif st_type in ["rest_site", "treasure"]:
            obs[632] = 1.0 if state.get(st_type, {}).get("can_proceed") else 0.0
        relics = player.get("relics", [])
        if st_type=="relic_select": relics=state.get("relic_select", {}).get("relics", [])
        elif st_type=="treasure": relics=state.get("treasure", {}).get("relics", [])
        for i, relic in enumerate(relics[:20]):
            base = 770 + (i * 5)
            obs[base], obs[base+1] = stable_hash(relic.get("id") or relic.get("relic_id"))/10000.0, (relic.get("counter")/10.0) if relic.counter is not None else 0.0
        for i, node in enumerate(state.get("map", {}).get("nodes", [])[:50]):
            base = 870 + (i * 10)
            if base+2 < 1536: obs[base], obs[base+1], obs[base+2] = node.get("col", 0)/7.0, node.get("row", 0)/15.0, stable_hash(node.get("type"))/10000.0
        if st_type == "crystal_sphere":
            cs = state.get("crystal_sphere", {})
            for cell in cs.get("cells", []):
                idx = 1370 + (cell.get("y", 0) * 9) + cell.get("x", 0)
                if idx < 1469:
                    val = 0.0
                    if not cell.get("is_hidden"):
                        val = 1.0 if cell.get("is_good") else -1.0
                        if item := cell.get("item_type"): val += (stable_hash(item) / 20000.0)
                    obs[idx] = val
        for i, pet in enumerate(player.get("pets", [])[:3]):
            base = 1469 + (i * 14)
            obs[base] = stable_hash(pet.get("id")) / 10000.0
            obs[base+1] = pet.get("hp", 0) / max(pet.get("max_hp", 1), 1)
            obs[base+2] = pet.get("block", 0) / 100.0
            for j, s in enumerate(pet.get("status", [])[:5]):
                obs[base+3 + (j*2)] = stable_hash(s.get("id")) / 10000.0
                obs[base+4 + (j*2)] = s.get("amount", 0) / 10.0
        if st_type in ["event", "neow"]:
            ev = state.get("event") or state.get("neow") or {}
            for i, opt in enumerate(ev.get("options", [])[:10]):
                base = 1511 + (i * 2)
                if base + 1 < 1536:
                    obs[base] = stable_hash(opt.get("title")) / 10000.0
                    obs[base+1] = 0.0 if opt.get("is_locked") else 1.0
        return obs

    def step(self, action_idx):
        self.total_episode_steps += 1
        fresh_state = self._raw_state()
        if fresh_state: self.current_state = fresh_state

        if self.current_state.get("state_type") == "menu" and self.current_state.get("menu_screen") == "popup":
            msg = self.current_state.get("message", "").lower()
            if "corrupt" in msg or "not able to load" in msg:
                self._log_action(f"Neutral Abort: {msg}")
                self.reboot_reason = f"Engine Bug (Entry): {msg}"
                self.needs_reboot = True
                return self._flatten_state(self.current_state), 0.0, True, False, {"floor": self.previous_floor, "engine_bug": True}
        if self.stagnant_steps >= 100:
            self._log_action("Deadlock Triggered (100 Steps)")
            if self.current_state.get("state_type") == "menu" and self.current_state.get("menu_screen") == "popup":
                msg = self.current_state.get("message", "").lower()
                if "corrupt" in msg or "not able to load" in msg:
                    self.reboot_reason = f"Engine Bug (Stall): {msg}"
                    self.needs_reboot = True
                    return self._flatten_state(self.current_state), 0.0, True, False, {"floor": self.previous_floor, "engine_bug": True}
            self.reboot_reason = "Deadlock (100 Steps)"
            self.needs_reboot = True
            return self._flatten_state(self.current_state), -2000.0, True, False, {"floor": self.previous_floor, "deadlock": True}
        
        payload = FLAT_ACTIONS[int(action_idx)].copy()
        raw_screen = self.current_state.get("state_type", "unknown")
        OVERLAY_KEYS = ["card_select", "enchanting", "enchant", "transform", "simple_select", "choose", "upgrade_select", "remove"]
        if raw_screen == "hand_select" or "hand_select" in self.current_state: screen = "hand_select"
        elif raw_screen in OVERLAY_KEYS or any(k in self.current_state for k in OVERLAY_KEYS): screen = "card_select_overlay"
        elif raw_screen == "bundle_select" or "bundle_select" in self.current_state: screen = "bundle_select"
        elif raw_screen == "relic_select" or "relic_select" in self.current_state: screen = "relic_select"
        elif raw_screen == "card_reward" or "card_reward" in self.current_state: screen = "card_reward"
        elif raw_screen == "rewards" or "rewards" in self.current_state: screen = "rewards"
        elif raw_screen == "treasure" or "treasure" in self.current_state: screen = "treasure"
        elif raw_screen == "crystal_sphere" or "crystal_sphere" in self.current_state: screen = "crystal_sphere"
        elif len(self.current_state.get("battle", {}).get("enemies", [])) > 0: screen = "combat"
        else: screen = raw_screen

        player = self.current_state.get("player", {})
        battle = self.current_state.get("battle", {})
        vision_summary = {"hp": f"{player.get('hp', 0)}/{player.get('max_hp', 1)}", "energy": f"{player.get('energy', 0)}/{player.get('max_energy', 3)}", "block": player.get("block", 0)}
        enemies_data = battle.get("enemies", [])
        if enemies_data: vision_summary["enemies"] = [f"{e.get('id', 'unk')} (HP:{e.get('hp',0)} B:{e.get('block',0)})" for e in enemies_data]
        
        floor = self.current_state.get("run", {}).get("floor", 0)
        self._log_action(f"[Floor {floor} | {screen}] Attempting: {payload}", vision=vision_summary)

        if not self.action_masks()[action_idx]:
            self._log_action(f"-> REJECTED: Action masked (Invalid).")
            if self.total_episode_steps >= 1500: 
                self.reboot_reason = "Timeout (Invalid Action Loop)"
                self.needs_reboot = True
                return self._flatten_state(self.current_state), -2000.0, True, False, {"floor": self.previous_floor}
            self.stagnant_steps += 1; time.sleep(0.05); return self._flatten_state(self.current_state), -0.1, False, False, {"floor": self.previous_floor}
        
        if payload.get("action") == "play_card":
            card_idx = payload.get("card_index")
            hand = player.get("hand", [])
            if card_idx is not None and card_idx < len(hand):
                card = hand[card_idx]
                if card.get("target_type") in ["AnyEnemy", "SingleTarget", "Target"]:
                    enemies = battle.get("enemies", [])
                    if (t_idx := payload.get("target_idx")) is not None and t_idx < len(enemies):
                        payload["target"] = enemies[t_idx].get("entity_id")
            if "target_idx" in payload: del payload["target_idx"]
        elif payload.get("action") == "use_potion":
            slot_idx = payload.get("slot")
            potions = player.get("potions", [])
            potion = next((p for p in potions if p.get("slot") == slot_idx), None)
            if potion and potion.get("target_type") in ["AnyEnemy", "SingleTarget", "Target"]:
                enemies = battle.get("enemies", [])
                if (t_idx := payload.get("target_idx")) is not None and t_idx < len(enemies):
                    payload["target"] = enemies[t_idx].get("entity_id")
            if "target_idx" in payload: del payload["target_idx"]
        
        valid = self._post(payload)
        new_state = self._raw_state()
        if not valid or not new_state: 
            time.sleep(0.05); new_state = self._raw_state()
            if not new_state: 
                self.reboot_reason = "API/State Invalid"
                self.needs_reboot = True; return self._flatten_state(self.current_state), -2000.0, True, False, {"floor": self.previous_floor}

        new_hash = stable_hash(json.dumps(new_state, sort_keys=True))
        if new_hash == self.last_state_hash:
            self.state_action_counts[action_idx] = self.state_action_counts.get(action_idx, 0) + 1
        else:
            self.state_action_counts = {}
            self.last_state_hash = new_hash

        # --- ENEMY TURN WAIT STATE ---
        if new_state and new_state.get("state_type") in ["monster", "elite", "boss"] and new_state.get("battle", {}).get("is_play_phase") == False:
            while new_state and new_state.get("state_type") in ["monster", "elite", "boss"] and new_state.get("battle", {}).get("is_play_phase") == False:
                time.sleep(0.05)
                new_state = self._raw_state()
                if not new_state: break
        
        # --- NEUTRAL ABORT FAILSAFE ---
        if new_state.get("state_type") == "menu" and new_state.get("menu_screen") == "popup":
            msg = new_state.get("message", "").lower()
            if "corrupt" in msg or "not able to load" in msg:
                self.reboot_reason = f"Engine Bug (Mid-Step): {msg}"
                self.needs_reboot = True
                return self._flatten_state(self.current_state), 0.0, True, False, {"floor": self.previous_floor, "engine_bug": True}
        
        obs = self._flatten_state(new_state)
        new_screen, floor_now = new_state.get("state_type", ""), new_state.get("run", {}).get("floor", 0)
        if new_state.get("run", {}).get("is_victory", False): floor_now = 51
        player_now, reward, terminated = new_state.get("player", {}), 0.0, False
        p = self.training_phase
        f_b, b_b, e_b, s_b, hp_m = {1:(2.5,100.0,25.0,10.0,0.1), 2:(5.0,150.0,45.0,20.0,0.2), 3:(7.5,200.0,70.0,30.0,0.3)}.get(p, (10.0,300.0,100.0,50.0,0.5))
        
        if self.last_prog_screen == "elite" and new_screen == "rewards": self.pending_elite_bounty = True
        if self.last_prog_screen == "boss" and new_screen == "rewards": self.pending_boss_bounty = True
        if new_screen == "card_select" and new_state.get("card_select", {}).get("screen_type") == "upgrade": self.pending_smith_bounty = True

        reward -= 0.10 # Neutral Step Tax
        if floor_now > self.previous_floor:
            reward += f_b
            if self.pending_boss_bounty: reward += b_b; self.pending_boss_bounty = False
            if self.pending_elite_bounty: reward += e_b; self.pending_elite_bounty = False
            if self.pending_smith_bounty: reward += s_b; self.pending_smith_bounty = False
            self.previous_floor = floor_now
            
        reward += (player_now.get("hp", 0) - self.previous_hp) * hp_m; self.previous_hp = player_now.get("hp", 0)
        enemies_now = new_state.get("battle", {}).get("enemies", [])
        for e in enemies_now:
            e_id, e_hp = e.get("entity_id"), e.get("hp", 0)
            if e_id not in self.lowest_enemy_hp_seen: self.lowest_enemy_hp_seen[e_id] = e_hp
            elif e_hp < self.lowest_enemy_hp_seen[e_id]: reward += (self.lowest_enemy_hp_seen[e_id] - e_hp) * 0.05; self.lowest_enemy_hp_seen[e_id] = e_hp
        
        curr_hp_sum = sum(e.get("hp", 0) for e in enemies_now) + player_now.get("hp", 0)
        prog = (floor_now != self.last_prog_floor or curr_hp_sum != self.last_prog_hp_sum or player_now.get("gold", 0) != self.last_prog_gold)
        if prog: self.stagnant_steps = 0
        else: self.stagnant_steps += 1
        
        self.last_prog_screen, self.last_prog_floor, self.last_prog_hp_sum, self.last_prog_gold = new_screen, floor_now, curr_hp_sum, player_now.get("gold", 0)

        if self.total_episode_steps >= 1500:
            self.reboot_reason = "Timeout (1500 Steps)"; self.needs_reboot = True
            return self._flatten_state(self.current_state), -2000.0, True, False, {"floor": self.previous_floor}
        
        if new_screen == "game_over" or floor_now == 51: terminated = True
        self.current_state = new_state
        return obs, reward * (2.0 if floor_now >= 45 else 1.0), terminated, False, {"floor": floor_now}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.needs_reboot: self._reboot_game_client(); self.needs_reboot = False
        self._ensure_fresh_run()
        state = self.current_state
        self.previous_floor, self.previous_hp = state.get("run", {}).get("floor", 0), state.get("player", {}).get("hp", 0)
        self.combat_step_count, self.total_episode_steps, self.stagnant_steps = 0, 0, 0
        self.last_prog_screen, self.last_prog_floor, self.last_prog_hp_sum, self.last_prog_gold = "none", -1, -1, -1
        self.pending_boss_bounty, self.pending_elite_bounty, self.pending_smith_bounty = False, False, False
        self.lowest_enemy_hp_seen = {}; return self._flatten_state(state), {}
