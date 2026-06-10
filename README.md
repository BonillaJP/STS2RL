# STS2RL: Slay the Spire 2 Reinforcement Learning Agent

## 📜 Statement of Intent & Disclaimer
This project is developed strictly for **educational and research purposes**. It is an independent implementation designed to explore the application of specialized Reinforcement Learning (RL) in complex strategy games.

⚠️ **WIP EXPERIMENTAL STATUS:** The neural architectures, reward systems, and environment mechanics within this repository are currently undergoing rapid, live experimentation. Code, parameters, and documentation are **not finalized** and will be updated frequently as the model's behavior is analyzed.

**Primary Objectives:**
* **Specialized Logic:** While the base STS2MCP API allows for LLM-based interactions, LLMs often struggle with the intricate mathematical scaling of Slay the Spire 2. This project aims to build a dedicated, low-latency neural model capable of consistent, high-level tactical play.
* **Future Interoperability:** A long-term goal is to leverage the API's cooperative multiplayer capabilities to facilitate multi-agent benchmarking and collaborative AI-human gameplay.
* **Integrity:** This project is not intended for exploitation, cheating, or gaining an unfair advantage in a commercial or competitive environment. It is a sandbox for advancement in game-centric AI.

An advanced Reinforcement Learning (RL) agent designed to play **Slay the Spire 2** autonomously. Powered by Stable Baselines3 (Maskable PPO) and a custom Gymnasium environment, this project implements a phased training curriculum capable of navigating complex combat, map routing, and menu interactions.

**SESSION STATUS:**
* **Training Phase:** Warmup Stage (Phase 1).
* **Progress:** 12,288-step rollout buffer.
* **Cluster Health:** 3 Training Nodes + 1 Eval Node active and synchronized.
* **Stability:** Zero-intervention recovery active; 100% uptime.

## 🚀 Key Features

* **Performance Monitoring:** Implements a high-performance console mirroring system that captures all training telemetry and errors to disk (`logs/train_console.log`) with automatic file rotation, ensuring 100% auditability without slowing down the GPU rollout.
* **Architectural Handoff:** Supports seamless transitions between curriculum stages. The orchestrator automatically detects required buffer shifts (e.g., 1024 -> 4096 steps) and performs a "hot-swap" by saving the model, re-instantiating the object, and resuming training mid-flight.
* **High-Res Deck Features:** The agent processes its Hand, Draw Pile (top 15), and Discard Pile (top 15) using dedicated 1D-Convolutional branches. This allows it to scan for card combos and strategic deck cycles with high precision.
* **Combat Intelligence CNN:** A specialized CNN branch scans up to 5 enemies in the room, tracking their IDs, Intents, and 10 detailed statuses (Strength, Ritual, Artifact, etc.). This enables the agent to recognize AOE opportunities and tactical power windows.
* **Semantic Card Features:** Each card is encoded with 40 detailed sensors that detect AOE (target_type), Multi-Hit ("times"), Scaling (Strength/Dex/Focus) capabilities, and precise normalized raw damage and block integers extracted from API descriptors.
* **State-First Architecture:** Ensures perfect synchronization between the game's internal state and the agent's logic. The environment performs a **Fresh API Fetch** at the start of every Step cycle and utilizes a high-performance cache for masking, preventing desyncs without sacrificing training throughput (FPS).
* **Dynamic Action Masking:** Implements rigorous action masking (exactly 342 discrete actions) with a **Smart Commitment Gate** selection system.
    * **Action-use cap:** Automatically masks any action used more than **20 times** on the same screen to prevent policy loops and deadlocks.
    * **Smart Commitment Rule:** In multi-card selection overlays, the `cancel_selection` action is only disabled **after** the selection requirement is met. This stops menu flickering exploits while allowing the agent to swap choices if the requirement isn't satisfied yet.
    * **One-Way Combat Overlays:** For single-card combat overlays (like Scry or Exhaust), the system enforces a one-way selection to ensure deterministic progression.
    * **Zero-Reboot Grade Stability:**
        * **Patient API Fetching (Level 1):** Performs **100 micro-retries** (100ms intervals) to wait for the API port to wake up, allowing up to **10 seconds** for basic room transitions.
        * **60s Startup Safety (Level 2):** Provides a massive **60-second window** (1200 iterations) for the engine to reach the Main Menu during reboots, physically preventing the "Murder Loop" where the script kills the game while it is still preloading assets.
        * **Ghost Save Recovery (Level 3):** Automatically detects and clears cached "Continue" states by toggling profiles upon reboot. Node 1 physically switches its UI `1 -> 2 -> 1`, Node 2 switches `2 -> 3 -> 2`, Node 3 switches `3 -> 1 -> 3`. This forces the Godot engine to refresh its metadata and realize the run is completely gone.
        * **Lightning Reboot Sequence:** Replaces hardcoded warmup delays with a high-speed Port Polling loop. The orchestrator checks the game's API port every 1s, resuming training the absolute millisecond the game becomes alive.
        * **Surgical Reboot Protocol:** For standard failures (deadlocks or API stutters), the system identifies and kills **ONLY the specific frozen node**. All other nodes continue training uninterrupted.
        * **Tiered Recovery & Amnesty:** To maintain training integrity and ensure "just" rewards, the system implements a three-tier recovery hierarchy:
            * **Tier 1: Instant Technical Amnesty:** API crashes or "Corrupt Save" errors trigger an **instant reboot**. The agent receives **Partial Floor Credit** (`Floor / 48 * 10.0`) to reward progress made before the failure.
            * **Tier 2: Glitch-Specific Pardon:** Detects the **Map-Combat Overlay** bug. The system waits **10 stagnant steps** to allow for self-recovery, then reboots with **Partial Floor Credit**.
            * **Tier 3: Behavioral Correction:** For general deadlocks (stalling/looping), the system reboots after **60 stagnant steps**. A corrective **-100.0 penalty** is applied to discourage unproductive behavior, alongside the accumulated step taxes.
        * **Poisoned Save Recovery:** Every reboot surgically deletes the `current_run.save` and `current_run.save.backup` for the specific profile. This clears any "poisoned" game states (like bugged events) and ensures the node always starts at a fresh Main Menu, achieving 100% zero-intervention stability.
        * **Full Cluster Purge:** Triggered ONLY if `godot.log` exceeds 1GB. The system force-kills ALL nodes simultaneously to release file locks and autonomously deletes the bloated log.
        * **Deterministic Loop Protection:** Differentiates between **Hard Progress** (changes in Floor, HP, Gold, or Deck Size) and **Screen Progress**. Action-use counters are strictly preserved across screen transitions and only reset when Hard Progress is detected, preventing infinite menu loops. **Smart Kick Reset:** Autokicks no longer reset the stagnation timer, ensuring deadlocks eventually trigger a reboot even if the agent attempts invalid actions.
        * **Fast Mode Dynamic Polling:** Utilizes a high-speed micro-polling loop. During screen transitions, selection actions, shop purchases, event choices, and turn ends, the environment polls the game state every **10ms** and breaks early the absolute millisecond the state visually updates, ensuring near-instant responses and 100% fresh data without sacrificing training SPS. Max wait for synchronization is capped at **300ms** to maintain high throughput.
        * **Zero-Latency Restarts:** The `_ensure_fresh_run` loop utilizes a **Patient Iterative Architecture** with MD5 state hashing to navigate Game Over screens and main menus at 20Hz (50ms safety throttle). Upon reaching the main menu, it actively queries the API to ensure the exact profile matches the node ID, forcibly re-selecting the correct profile before embarking, thus avoiding file lock collisions.
        * **Action Cooldown Enforcement:** Applies a strict RL mask cooldown to rapid actions to prevent the agent from spamming the engine. Card and potion plays receive a **5-step cooldown**, while merchant and campfire choices are restricted by a **10-step cooldown**.
        * **Smart Progress Tracking:** Only **Hard Progress** (changes in HP, Gold, Floor, or Deck Size) fully resets the stagnation counter. Selection history progress only grants a small buffer (up to **20 steps**) rather than a full reset, preventing the "flicker-loop" exploit where the agent toggles cards to avoid death penalties.
            * **Compounding Step Tax:** Every action costs a base **-0.05**. If more than 20 steps are taken on a single screen without progress, the tax scales to **-0.20**. This ensures that stalling is always more expensive than the -100.0 death penalty.
            * **Absolute path-based Force-Kill:** The reboot logic performs an exhaustive path-based cleanup of the node's specific folder before relaunching to prevent duplicate game instances.

    * **Overlay-First Priority:** Selection overlays (Exhaust, Scry, Choose) are prioritized over combat actions, forcing the agent to resolve choices before fighting to prevent deadlocks.
    * **Safe Throttle:** Implements a global **50ms input delay** (0.05s) to provide a safety buffer for the Godot engine's C# animation handlers, preventing "Thread Collisions" and hard crashes.
    * **Boss-Only Map Proceed:** The map `proceed` button is strictly disabled unless the agent is on a Boss Floor (17, 33, 48), preventing the "Fake Proceed" trap.
    * **Real-Map Verification:** The environment only identifies the screen as "map" if actual navigation nodes are present in the API response.
* **Stagnation Detection & Failsafes:** Built-in orchestrator that detects engine bugs, infinite loops, or **API/State Invalid** errors. It automatically reboots the specific game node and triggers **Technical Amnesty** to protect the model's policy.
* **Deadlock Recovery:** Lowers the technical reboot timer to **60 stagnant steps**, ensuring high cluster uptime and fast self-healing from rare engine freezes.
* **Closed-Loop Telemetry:** Implements a "verify-then-delete" log consolidation system. Upon reboot, the orchestrator automatically merges fragmented CSVs into master files, truncates "ghost steps" to match the brain checkpoint, and purges technical telemetry created after the checkpoint timestamp.

## 💻 Cluster Setup & Game Cloning

To maximize training throughput, this project utilizes a **4-Node Cluster** running simultaneously.

* **Training Nodes:** 3 Instances (Ports 15526, 15527, 15528)
* **Evaluation Node:** 1 Instance (Port 15529)

**Hardware Performance & Scaling:**
This cluster is benchmarked on a high-end workstation (**RTX 4070 GPU, 32 GB RAM, i7-12700 CPU**) to achieve maximum training throughput.

* **The RAM Bottleneck:** Even with 32 GB of RAM, memory remains the primary constraint. A 4-node cluster typically utilizes ~75% of available memory (approx. 24 GB). This is due to the Godot engine's overhead per instance and the large shared rollout buffers in Python.
* **CPU Utilization:** CPU usage fluctuates between 30% and 90% depending on the current game screens. Combat animations and room transitions are the primary drivers of CPU spikes.
* **GPU Load:** The RTX 4070 maintains a steady 40-50% utilization, handling both the game renders and the neural updates with significant headroom.
* **Target Throughput:** This hardware configuration is capable of maintaining a massive aggregate throughput of **5+ FPS/SPS**, ensuring rapid policy convergence across the 6-phase curriculum.

You can scale the cluster based on your own hardware; however, decreasing memory below 32 GB may require reducing the number of active training nodes to prevent system instability.

### Scaling the Cluster (Adding/Removing Nodes)
If you wish to run more (or fewer) nodes, you must perform the following:
1.  **Clone more game folders** and assign them unique ports via the STS2MCP mod config.
2.  Update the `TRAIN_PORTS` list in `train.py` to include the new ports.
3.  **Crucial:** Adjust `n_steps` and `batch_size` in the `BUFFER_STAGES` matrix in `train.py`. Because the total rollout buffer is `n_steps * number_of_nodes`, adding nodes means you should lower `n_steps` to maintain an optimal update frequency.
4.  *Note: You **DO NOT** need to change the Box Dimensions (4096) or Action Space (342). These define a single agent's interface and remain identical regardless of the cluster size.*

### ⚠️ Crucial Mod Credits (STS2MCP)
**This project would be absolutely impossible without the incredible STS2MCP API Mod.** The mod acts as the core bridge, exposing the internal Slay the Spire 2 game state as a JSON payload and accepting external HTTP POST commands. This project relies entirely on this mod to provide the Python RL environment with visibility and control inside the game.
**Huge thanks to the creator:** [https://github.com/Gennadiyev/STS2MCP](https://github.com/Gennadiyev/STS2MCP)  

### How to Replicate the Cluster for Your Machine
Since this project runs on local hardware with physical node clones, every new user must perform the following manual setup:

1.  **Clone the Game Node Folders:**
    * Navigate to your Steam installation: `...\steamapps\common\SlayTheSpire2`.
    * Create 4 identical copies of the folder, naming them `STS2_Node_1`, `STS2_Node_2`, `STS2_Node_3`, and `STS2_Node_4`.
2.  **Bypass Steam Instance Lock:**
    * Inside each cloned folder, create a text file named `steam_appid.txt` and paste the Slay the Spire 2 Steam App ID (`1932700`) inside it. This allows the clones to run independently.
3.  **Assign Unique Ports:**
    * Install the **STS2MCP Mod** into every node folder.
    * Open the mod's configuration file (`STS2MCP.conf`) in each node and assign a unique port. The defaults in `train.py` are:
        * Node 1: **15526**
        * Node 2: **15527**
        * Node 3: **15528**
        * Node 4: **15529** (Evaluation Node)
4. **Configure Local Paths in Code:**
    * **Centralized Configuration:** Open `sts2_env.py` and locate the **`GLOBAL_CONFIG`** dictionary. Update the `BASE_PATH_TEMPLATE` to point to your new node folders (e.g., `D:\Games\Steam\steamapps\common\STS2_Node_`). The system will automatically handle the mapping for all 4 nodes.
5.  **Initialize Profiles:**
    * Manually open the game for each node at least once to ensure **3 profiles** are created. The orchestrator uses these to refresh memory and clear ghost saves.

### 🛡️ Automated Profile Isolation (API Stability)
To avoid the "Corrupt Save File" API startup error and ensure state integrity during parallel rollouts, the environment **automatically enforces** a unique profile mapping for each node. Slay the Spire 2 supports exactly 3 profiles:
* **Node 1 (Port 15526):** Automatically switches to **Profile 1**.
* **Node 2 (Port 15527):** Automatically switches to **Profile 2**.
* **Node 3 (Port 15528):** Automatically switches to **Profile 3**.
* **Node 4 (Evaluation - Port 15529):** Defaults to **Profile 1**.

This automated enforcement prevents the game engine from attempting to write/load from the same save metadata simultaneously. The orchestrator detects the current profile at the main menu and programmatically switches it if a mismatch is found.

### 🚨 CRITICAL CAUTION: Storage Bloat Bug (Godot Engine Logs)
Slay the Spire 2 is powered by the Godot engine, which generates detailed debug logs in the system's `AppData` directory (`%APPDATA%\SlayTheSpire2\logs\`). Under normal conditions, log utilization is remarkably low—typically remaining below a few megabytes or a gigabyte at most throughout extensive training flights.

⚠️ **KNOWN ENGINE BUG LOOP:** A massive storage consumption spike can happen if the evaluation or training nodes encounter a deterministic preloading/rendering exception. Specifically, when the model triggers the `PunchOff` event (most commonly observed on **Floor 14**), a native C# NullReferenceException inside the game's asynchronous animation thread causes an infinite error dump. Because this failure loops endlessly on every single frame tick (`_Process`), the log file can rapidly balloon by tens of gigabytes within minutes.

**High-Stability Hierarchy (Zero-Intervention Recovery):**
The environment features a five-layer **Self-Healing** system to handle engine crashes, lag spikes, and mod bugs autonomously:

1.  **Patient API Handshake (Level 1):** Performs **100 micro-retries** (100ms intervals) to wait for basic room transitions and API wake-up.
2.  **60s Startup Safety (Level 2):** Provides a massive **60-second window** for the engine to reach the Main Menu during reboots, physically preventing the "Murder Loop."
3.  **Surgical Reboot Protocol (Level 3):** Port Polling detects deadlocks and kills **ONLY the specific frozen node** by its absolute process name.
4.  **Ghost Save Recovery (Level 4):** Automatic profile-toggling upon reboot clears engine metadata corruption.
5.  **Full Cluster Purge (Level 5):** Triggered ONLY if `godot.log` exceeds 1GB. The system force-kills ALL nodes simultaneously to release file locks and release the disk buffer.

**Additional Precision Layers:**
* **Poisoned Save Recovery:** Every reboot surgically deletes the `current_run.save` and `current_run.save.backup` for the specific profile. This clears bugged game states and ensures the node always starts at a fresh Main Menu.
* **Safe Throttle:** Implements a global **50ms input delay** (0.05s) to provide a safety buffer for the Godot engine's C# animation handlers, preventing "Thread Collisions" and hard crashes during heavy enemy turns.        
* **Storage Bloat Failsafe:** The environment proactively monitors the Godot engine's log size. If `godot.log` exceeds **1 GB**, the system automatically triggers a Level 5 Purge reset.
* **Deadlock Recovery:** Lowers the technical reboot timer to **60 stagnant steps**, ensuring high cluster uptime and fast self-healing from rare engine freezes.

## ⚙️ Technical Deep Dive

### 1. High-Fidelity Sensory Framework (4096 Floats)
The agent perceives the Spire through a massive, flattened observation vector routed through specialized convolutional branches:
* **Observation Space:** `spaces.Box(low=-1.0, high=1.0, shape=(4096,), dtype=np.float32)`
    * **[0-120]: Core Meta State:** Tracks navigation, HP Ratio, Gold, Pile sizes, and Status Effects.
    * **[120-300]: Enemy Intelligence CNN:** Detailed data for up to 5 enemies simultaneously (ID, HP, Block, Intent, 10 Statuses each mapped out to 36 points).
    * **[300-2100]: High-Res Deck Vision (CNN):** 15 cards each in Hand [300-900], Draw Pile [900-1500], and Discard Pile [1500-2100] across 40 semantic points:
        * Identity, Cost, Keywords, AOE, Multi-hit flags, and **High-Resolution Power sensors** for raw Damage and Block.
    * **[2100-3100]: Spire Navigation Intelligence (CNN):** A massive mapping of 100 navigation nodes (Col/Row, Room Type, Accessibility).
    * **[3100-3500]: Expanded Relic Inventory:** Tracking up to 40 relics with unique ID hashes and counters mapped to 10 points per slot.
    * **[3500-3800]: Tactical Overlays:** Selective card/relic selection, Exhaust Radar, and reward item parsing.
    * **[3800-3900]: Crystal Sphere Grid:** 9x11 spatial mapping for the STS2 minigame.
    * **[3900-4000]: Summoned Pets & Minions:** Detailed scanning for up to 5 companions.
    * **[4000-4050]: Potion Inventory Scan:** Detailed parsing of 5 potion slots.
    * **[4050-4095]: High-Fidelity Combat Telemetry:** Real-time tracking of turns, stagnant steps, and curriculum phase.

### 2. Action Space (342 Discrete Actions)
A fully flattened discrete space representing exactly 342 possible interactions:
* **[0-49]: `play_card`**: 10 card slots, 5 target resolution slots each.
* **[50-74]: `use_potion`**: 5 potion slots, 5 potential targets.
* **[75-79]: `discard_potion`**: Safely discarding potions.
* **[80]: `end_turn`**.
* **[81-90]: `combat_select_card`**.
* **[91]: `combat_confirm_selection`**.
* **[92-106]: `claim_reward`**: Collecting up to 15 loot items.
* **[107-121]: `select_card_reward`**: Choosing 1 of 15 rewards.
* **[122-123]: `skip_card_reward` / `proceed`**.
* **[124-133]: `choose_event_option`**.
* **[134]: `advance_dialogue`**.
* **[135-139]: `choose_rest_option`**.
* **[140-154]: `shop_purchase`**: 15 merchant slots.
* **[155-159]: `choose_map_node`**.
* **[160-209]: `select_card`**: 50-slot selection grid.
* **[210-211]: `confirm_selection` / `cancel_selection`**.
* **[212-218]: `select_bundle`** (indices + confirm/cancel).
* **[219-224]: `select_relic`** (indices + skip).
* **[225-229]: `claim_treasure_relic`**.
* **[230-239]: `menu_select`**.
* [240-241]: **Crystal Sphere Tool Selection** (Big/Small).
* [242-340]: **Crystal Sphere Grid**: 9x11 grid interaction (99 cells).
* [341]: `crystal_sphere_proceed`.

### Synchronous Training Architecture (Stable Baselines3)
This project utilizes **Stable Baselines3 (SB3)** to orchestrate the training loop using vectorized environments:
* **SubprocVecEnv (Training):** Used for the training nodes. SB3's neural network update mechanism is entirely **synchronous**—it waits for the `n_steps` buffer to fill across *all* subprocesses before freezing to compute gradients.
* **DummyVecEnv (Evaluation):** Used for the single Evaluation node. It executes sequentially on the main thread.
* **The Bottleneck:** Because game APIs can sometimes stall, the synchronous nature means the fastest node is always waiting for the slowest node.
* **Experimentation Encouraged!** Experimentation with **asynchronous architectures** (such as Ray RLlib, SampleFactory, or async forks of SB3) is highly encouraged to solve the synchronous bottleneck!

### Neural Framework (MaskablePPO)
The agent relies on a specialized neural network architecture tailored for Slay the Spire:
* **Maskable PPO (Proximal Policy Optimization):** Standard PPO wastes millions of steps trying to click disabled buttons. By utilizing `sb3-contrib`'s MaskablePPO, the environment passes a boolean mask array of exactly **342 bits** alongside every observation. This guarantees that 100% of the gradient updates are spent evaluating valid tactical choices.
* **Safe Commitment (Multi-Card Selection):** To prevent infinite "toggle loops" where the agent unselects its first choice, the masking logic employs a **One-Way Gate**. Once a card is selected, its action slot is immediately blocked and the **Cancel** button is disabled until the menu is closed.
* **Repetition Protection:** A built-in blocker tracks the number of times an identical action is taken within the same game state. If any action is attempted **20 times** on a single screen without producing a physical change in the game state, that action is strictly disabled.
* **Multi-branch CNN Features:** Rather than treating the player's hand as a flat array of numbers, the environment extracts state features and passes them through 1D-Convolutional branches (`nn.Conv1d`). This allows the network to natively detect spatial synergies—such as a zero-cost card sitting next to a high-damage card.      

### Hyperparameters
* **Model Loader:** The training script performs a recursive search across all sub-directories and automatically resumes from the newest `.zip` file based on disk timestamp.
* **Intelligent Stage Selection (Zip-Peeking):** Upon resuming, the orchestrator "peeks" inside the model zip to identify the current internal step count and automatically selects the correct training parameters.
* **Rollout Buffer:** The main training stage uses `n_steps=4096` per node. With 3 training nodes, this results in exactly **12,288 steps** in the buffer before every neural weight update.
* **Checkpoints:** The model is saved to the `checkpoints/` directory immediately after every update, securely prior to executing an exam.     
* **Hall of Fame:** If a high score is achieved during a rollout, the updated model is saved to `hall_of_fame/` at the start of the next update iteration.
* **Phase Best:** Reserved for **Phase Promotions**. When an agent successfully completes a Phase, its graduating model is secured in the `models/best_phase_*/` directory for historical reference.
* **Curiosity Defibrillator:** To prevent behavioral "ruts", the orchestrator implements a dynamic curiosity spike.
    * **Phases 1-3 Baseline:** Strictly enforced at **0.10** to ensure maximum Act 1 discovery.
    * **Defibrillation:** If the agent fails to reach the Act Boss for **5 consecutive exams**, the `ent_coef` is automatically increased (+0.05) up to a max of **0.25** to force exploration of new synergies.
    * **Refinement:** Upon success, entropy decays (0.90x) to quickly stabilize the winning policy, and resets to baseline upon Act promotion.

* **Strike Persistence:** Stagnation strikes are recorded in `cluster_state.json`, ensuring state is preserved across transitions and cluster reboots.

### 📊 Persistent Data & Session Logging
To guarantee 100% data integrity across cluster reboots, the project implements a **Fragment-and-Merge** logging architecture:
* **Session Isolation:** Every time `train.py` starts, it creates a unique session folder (`logs/sb3_tech/session_{ID}/`). All SB3 metrics and TensorBoard events are isolated here to prevent overwriting historical data.   
* **Automatic Consolidation:** Upon startup, the orchestrator scans all session folders and automatically merges fragmented `progress.csv` files into a single combined log.
* **Ghost Step Truncation:** During synchronization, the system identifies and purges "ghost steps"—data recorded after the last checkpoint—to ensure your telemetry exactly matches your model's neural state.     

### The Reward System (Phased Mastery Telemetry)
A specialized, multi-layered reward system was designed to balance short-term survival with long-term scaling:

* **Step Tax (`-0.05`):** A minimal flat penalty applied to every single action. Uncapped: forces the agent to act efficiently and prioritize decisive combat.
* **Transition Reward:** Any action that physically changes the game screen grants a small **+0.05 reward**, reinforcing decisiveness.
* **Tapered Death Penalty:** Ranges from **-100.0 to -50.0 absolute floor**. Formula: `Penalty = -100.0 + (min(Floor / Phase_Target, 1.0) * 50.0)`. Dying early is heavily penalized (-100), but dying deep in an Act near the target Boss is forgiven (-50), naturally pulling the agent deeper into the Spire while sharpening the gradient for survival.
* **Combat Stall Penalty:** If a fight exceeds 20 turns, a compounding penalty (`-0.1` per extra turn) is applied. This prevents "infinite defense" loops.
* **Stagnation Tax Ceiling:** Heavy, compounding stagnation penalties are strictly capped at a cumulative **-30.0 per episode**. This prevents UI loops or technical errors from destroying the reward signal of an otherwise solid run.
* **Stagnation Tax Scaling:** * **10+ steps:** `-2.0` milestone penalty.
    * **20+ steps:** `-10.0` milestone penalty.
    * **50+ steps:** `-18.0` milestone penalty.
    * **200+ steps:** **Autonomous Surgical Reboot.**

* **Endgame Amplification:** A **2.0x global multiplier** is applied to all net rewards earned on or after **Floor 45**, sharpening the gradient for Act 3 mastery.

* **Indecision Tax (Masked Actions):** To prevent the agent from spamming invalid inputs, every masked action attempt results in a light **`-1.0` penalty**.
* **Reward Telemetry:** Every action reward is logged in real-time within the `node_trace_{port}.jsonl` files, featuring a breakdown of components.
* **Universal Healing Efficiency:** Implements a resource management model. Choice detection is powered by direct API IDs (`rest`, `smith`, etc.):
    * **Clean Heal Bounty (+15.0):** Rewarded if the 30% HP gain fits efficiently within the Max HP bar.
    * **Wasteful Heal Penalty:** If healing overflows Max HP, a penalty is applied to discourage resource waste.
    * **Smart Upgrade Bounty (+20.0):** Awarded if the agent correctly skips a wasteful heal in favor of camp options when at high HP.
    * **Reckless Skip Penalty:** Aggressively penalizes skipping a "Clean Heal" when in the danger zone (< 30% HP).
* **HP Delta:** Dynamic rewards (**`+0.1`** per enemy HP damage dealt).
* **Bounty Matrix:** Standardized rewards for major milestones:
    * *Standard Bounties:* **Floor (+10.0)**, **Boss Kill (+100.0)**, **Elite Kill (+15.0)**, **Smith (+15.0 Base)**.
    * **Boss Sight Bounty (+50.0):** Immediate reward for reaching Floor 17, 33, or 48.
    * **New Relic Bounty (+25.0):** Reward for acquiring a new relic.
    * **Card Removal (+15.0):** Granted when a **Basic card** is removed from the deck.
    * **Elite Ambition (Pathing):** Grants **+5.0** for entering an elite room while healthy (>50% HP) and a phased penalty (**-5.0 to -15.0**) for entering while critical (<30% HP).
    * **Early Foundation (+2.0):** Rewards adding any card during Act 1 (Floors 1-15).
    * **Tactical Potion Utility (+2.0):** Rewards using a potion in Elite or Boss encounters.

**Design Rationale:**
The system is designed to prevent "superstitious learning" or simply surviving without getting stronger. Implementing a strict Step Tax forces the agent to act efficiently.

### Phased Mastery Training (6-Phase Curriculum)
The agent progresses through a dynamic curriculum (Phase 1 to Phase 6). Progression requires passing a **20-episode** deterministic exam on the dedicated Evaluation Node.
Note: Stage 1 (Warmup Buffer) automatically transitions into Stage 2 (Mastery Buffer) during Phase 1 based on total accumulated steps. The orchestrator is explicitly configured to perform an exam evaluation exactly at the end of Stage 1 before initiating the buffer transition.

**Promotion Requirements:**
The curriculum is split into two distinct learning philosophies:
- **Experience-First (Phases 1-3):** Reward scores are ignored for promotion. The agent promotes if it successfully reaches the target Act Boss floor with a **30% success rate** (`wins >= 6/20`).
- **Mastery-Based (Phases 4-6):** The agent must meet BOTH a strict Mean Floor (48.0) AND a high Mean Reward target to prove Endgame Mastery.

* **Phase 1 (Act 1 - Floor 17):** Requires Wins >= 6/20.
* **Phase 2 (Act 2 - Floor 33):** Requires Wins >= 6/20.
* **Phase 3 (Act 3 - Floor 48):** Requires Wins >= 6/20.
    * *Upon graduation, a locked archival model `best_phase_3_GRADUATE.zip` is created.*
* **Phase 4 (Mastery Level 1):** Requires Mean Floor >= 48.0 AND Mean Reward >= **900.0**.
* **Phase 5 (Mastery Level 2):** Requires Mean Floor >= 48.0 AND Mean Reward >= **1,000.0**.
* **Phase 6 (Mastery Level 3):** Requires Mean Floor >= 48.0 AND Mean Reward >= **1,100.0**.

**Graduation Mandates:**
1. **The Demotion Floor:** Once the agent reaches Phase 4, it is locked into the Mastery curriculum.
2. **Ascension 0 Mastery:** Reward targets are set to strict levels (900-1100) to ensure the agent is battle-ready for higher Ascension difficulties.

**Demotion (Step-Down Recovery):**
If performance stalls, the system implements a recovery step-down. If the agent fails 20 consecutive exams, it drops down exactly 1 Phase.

## 🛠️ Prerequisites

Before running this project, ensure the following are installed:
* **Windows PC:** (Required to run the game client natively)
* **Python 3.11.15:** (Tested using **Miniconda**)
* **Slay the Spire 2:** Version `0.103.2 (2026.04.16)`
* **STS2MCP API Mod:** Version `0.4.0`

## 📦 Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/BonillaJP/STS2RL.git](https://github.com/BonillaJP/STS2RL.git)
    cd STS2RL
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 🎮 Usage

Initialize the training environment and connect to the local game nodes:      

```bash
python train.py

## 📊 Telemetry & Debugging

The project includes built-in diagnostic tools to monitor cluster health, timings, and training progression in real-time.

1.  **Log Analysis (`analyze_logs.py`)**
    Provides a comprehensive overview of the training telemetry, including early learning deltas, SB3 rollout metrics (FPS, explained variance, entropy loss), and storage telemetry.
    ```bash
    python analyze_logs.py
    ```

2.  **Cluster Diagnostic Dashboard (`debug_cluster.py`)**
    Monitors live node timings, action rejections, and auto-kicks to ensure the environment is correctly handling Fast Mode animations without deadlocking. It outputs the true average cycle time (seconds/step) for each active node.
    ```bash
    python debug_cluster.py
    ```

---

## 🛠️ Developer Tuning Guide

To modify the core behavior of the agent or optimize for different hardware, refer to the **`GLOBAL_CONFIG`** block at the top of **`sts2_env.py`**.

### 1. Global Modifiers (`sts2_env.py`)
*   **`BASE_PATH_TEMPLATE`**: The centralized path to your STS2 node folders.
*   **`TARGET_CHARACTER`**: Allows for rapid character switching. Options include: "IRONCLAD", "SILENT", "DEFECT", "REGENT", "NECROBINDER", "RANDOM".
*   **`STABILITY_THROTTLE`**: Controls the input delay (default 0.05s). Increase this if the game crashes during heavy animations.
*   **`STEP_TAX`**: Base penalty per action. Adjust this to make the agent more or less "impatient."
*   **`BOUNTIES`**: Fine-tune the rewards for Floor climbs, Boss kills, Elites, and Smithing.

### 2. Training Dynamics (`train.py`)
*   **`BUFFER_STAGES`**: A curriculum list where each entry is `(Batch Size, Steps per Update, Max Step)`. 
    *   *Tip:* If your GPU memory is low, decrease the **Batch Size** (e.g., 3072 -> 1024).
*   **`ent_coef`**: Initial exploration factor. The **Smart Entropy Defibrillator** in `PhaseManagerCallback` will automatically scale this if progress stalls.
*   **`learning_rate`**: Standard training rate. If the policy collapses (High `approx_kl`), lower this value in `PhaseManagerCallback._apply_phase`.

### 3. Masking & Logic
*   **`rejection_penalty`**: Currently set to **-1.0** in `GLOBAL_CONFIG`. 
    *   *Tip:* If the agent is "lazily" spamming end-turn to avoid fighting, increase this penalty to **-5.0** to force better mask adherence.


