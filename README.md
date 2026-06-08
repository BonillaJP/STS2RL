# STS2RL: Slay the Spire 2 Reinforcement Learning Agent

## 📜 Statement of Intent & Disclaimer
This project is developed strictly for **educational and research purposes**. It is an independent implementation designed to explore the application of specialized Reinforcement Learning (RL) in complex strategy games.

⚠️ **WIP EXPERIMENTAL STATUS:** The neural architectures, reward systems, and environment mechanics within this repository are currently undergoing rapid, live experimentation. Code, parameters, and documentation are **not finalized** and will be updated frequently as the model's behavior is analyzed.

**Primary Objectives:**
* **Specialized Logic:** While the base STS2MCP API allows for LLM-based interactions, LLMs often struggle with the intricate mathematical scaling of Slay the Spire 2. This project aims to build a dedicated, low-latency neural model capable of consistent, high-level tactical play.
* **Future Interoperability:** A long-term goal is to leverage the API's cooperative multiplayer capabilities to facilitate multi-agent synergy benchmarking and collaborative AI-human gameplay.
* **Integrity:** This project is not intended for exploitation, cheating, or gaining an unfair advantage in a commercial or competitive environment. It is a sandbox for advancement in game-centric AI.

An advanced Reinforcement Learning (RL) agent designed to play **Slay the Spire 2** autonomously. Powered by Stable Baselines3 (Maskable PPO) and a custom Gymnasium environment, this project implements a "Master Class" architecture capable of navigating complex combat, map routing, and menu interactions.

## 🚀 Key Features

* **Synergy-CNN Vision:** The agent processes its hand using a 1D-Convolutional branch, allowing it to "scan" for card combos rather than just reading raw scalar numbers.
* **State-First Architecture:** Ensures perfect synchronization between the game's internal state and the agent's logic. The environment performs a **Fresh API Fetch** at the start of every Step cycle and utilizes a high-performance cache for masking, preventing desyncs without sacrificing training throughput (FPS).
* **Dynamic Action Masking:** Implements rigorous action masking (exactly 317 discrete actions) with a **One-Way Gate** selection system.
    * **Zero-Reboot Grade Stability:**
        * **Patient API Fetching (Level 1):** Implements a robust state retriever that performs **20 micro-retries** (10ms intervals) before declaring an API failure. This allows the Godot engine up to **10 seconds** to finish room transitions and loading spikes without triggering technical reboots.
        * **60s Startup Safety (Level 2):** The orchestrator provides a generous **60-second window** for the game to reach the Main Menu during reboots, physically preventing the "Murder Loop" where the script kills the game while it is still booting.
        * **Ghost Save Recovery (Level 3):** Automatically detects and clears cached "Continue" states by toggling profiles (e.g., 1 -> 2 -> 1) upon reboot. This forces the Godot engine to refresh its metadata and realize the run is gone.
        * **Lightning Reboot Sequence:** Replaces hardcoded warmup delays with a high-speed Port Polling loop. The orchestrator checks the game's API port every 1s, resuming training the absolute millisecond the game becomes alive.
        * **Surgical Reboot Protocol:** For standard failures (deadlocks or API stutters), the system identifies and kills **ONLY the specific frozen node**. All other nodes continue training uninterrupted.
        * **Poisoned Save Recovery:** Every reboot surgically deletes the `current_run.save` and `current_run.save.backup` for the specific profile. This clears any "poisoned" game states (like bugged events) and ensures the node always starts at a fresh Main Menu, achieving 100% zero-intervention stability.
        * **Full Cluster Purge:** Triggered ONLY if `godot.log` exceeds 1GB. The system force-kills ALL nodes simultaneously to release file locks and autonomously deletes the bloated log.
        * **Void Reward System:** To maintain training integrity, any run interrupted by a technical reboot (Deadlock, Bloat, or Engine Bug) is automatically **Voided**. The environment returns a **0.0 reward** (Neutral Abort), ensuring that technical failures do not influence the neural policy.
        * **Deterministic Loop Protection:** Differentiates between **Hard Progress** (changes in Floor, HP, Gold, or Deck Size) and **Screen Progress**. Action-use counters are strictly preserved across screen transitions and only reset when Hard Progress is detected, preventing infinite menu loops.
        * **Fast Mode Dynamic Polling:** Replaces hardcoded thread sleeps with a high-speed polling loop. During screen transitions, the environment polls the game state every 50ms, instantly resuming training the absolute millisecond the transition finishes.
        * **Animation Cooldown Enforcement:** Applies a strict 5-step RL mask cooldown to rapid actions (like playing cards) to prevent the agent from spamming actions faster than the engine can process them.
        * **Smart Progress Tracking:** Stagnation counters are automatically reset if the system detects changes in **HP, Gold, Floor, Screen Type, Energy, Block, Deck Size, or Card Selection history.** This ensures the 100-step reboot failsafe only triggers for true engine freezes.
    * **Absolute path-based Force-Kill:** The reboot logic performs an exhaustive path-based cleanup of the node's specific folder before relaunching to prevent duplicate game instances.
    * **Overlay-First Priority:** Selection overlays (Exhaust, Scry, Choose) are prioritized over combat actions, forcing the agent to resolve choices before fighting to prevent deadlocks.
    * **Safe Throttle:** Implements a global **50ms input delay** (0.05s) to provide a safety buffer for the Godot engine's C# animation handlers, preventing "Thread Collisions" and hard crashes.
    * **Boss-Only Map Proceed:** The map `proceed` button is strictly disabled unless the agent is on a Boss Floor (16, 33, 51), preventing the "Fake Proceed" trap.
    * **Real-Map Verification:** The environment only identifies the screen as "map" if actual navigation nodes are present in the API response.
* **Stagnation Detection & Failsafes:** Built-in orchestrator that detects engine bugs, infinite loops, or **API/State Invalid** errors. It automatically reboots the specific game node and triggers a **Neutral Abort (0.0 reward)** to protect the model's policy.
* **Deadlock Recovery:** Lowers the technical reboot timer to **100 stagnant steps**, ensuring high cluster uptime and fast self-healing from rare engine freezes.
* **Closed-Loop Telemetry:** Implements a "verify-then-delete" log consolidation system. Upon reboot, the orchestrator automatically merges fragmented CSVs into master files, truncates "ghost steps" to match the brain checkpoint, and purges technical telemetry created after the checkpoint timestamp.

## 🖥️ Cluster Setup & Game Cloning

To maximize training throughput, this project utilizes a **4-Node Cluster** running simultaneously. 

* **Training Nodes:** 3 Instances (Ports 15526, 15527, 15528)
* **Evaluation Node:** 1 Instance (Port 15529)

**Hardware Performance & Scaling:**
This cluster is benchmarked on a high-end workstation (**RTX 4070 GPU, 32 GB RAM, i7-12700 CPU**) to achieve maximum training throughput. 

*   **The RAM Bottleneck:** Even with 32 GB of RAM, memory remains the primary constraint. A 4-node cluster typically utilizes ~75% of available memory (approx. 24 GB). This is due to the Godot engine's overhead per instance and the large shared rollout buffers in Python.
*   **CPU Utilization:** CPU usage fluctuates between 30% and 90% depending on the current game screens. Combat animations and room transitions are the primary drivers of CPU spikes.
*   **GPU Load:** The RTX 4070 maintains a steady 40-50% utilization, handling both the game renders and the Synergy-CNN neural updates with significant headroom.
*   **Target Throughput:** This hardware configuration is capable of maintaining a massive aggregate throughput of **500+ FPS/SPS**, ensuring rapid policy convergence across the 6-phase curriculum.

You can scale the cluster based on your own hardware; however, decreasing memory below 32 GB may require reducing the number of active training nodes to prevent system instability.

### Scaling the Cluster (Adding/Removing Nodes)
If you wish to run more (or fewer) nodes, you must perform the following:
1.  **Clone more game folders** and assign them unique ports via the STS2MCP mod config.
2.  Update the `TRAIN_PORTS` list in `train.py` to include the new ports.
3.  **Crucial:** Adjust `n_steps` and `batch_size` in the `BUFFER_STAGES` matrix in `train.py`. Because the total rollout buffer is `n_steps * number_of_nodes`, adding nodes means you should lower `n_steps` to maintain an optimal update frequency.
4.  *Note: You **DO NOT** need to change the Box Dimensions (1536) or Action Space (317). These define a single agent's interface and remain identical regardless of the cluster size.*

### ⚠️ Crucial Mod Credits (STS2MCP)
**This project would be absolutely impossible without the incredible STS2MCP API Mod.** The mod acts as the core bridge, exposing the internal Slay the Spire 2 game state as a JSON payload and accepting external HTTP POST commands. This project relies entirely on this mod to provide the Python RL environment with visibility and control inside the game. 
**Huge thanks to the creator:** [https://github.com/Gennadiyev/STS2MCP](https://github.com/Gennadiyev/STS2MCP)

### How to Replicate the Cluster for Your Machine
Since this project runs on local hardware with physical node clones, every new user must perform the following manual setup:

1.  **Clone the Game Node Folders:** 
    *   Navigate to your Steam installation: `...\steamapps\common\SlayTheSpire2`.
    *   Create 4 identical copies of the folder, naming them `STS2_Node_1`, `STS2_Node_2`, `STS2_Node_3`, and `STS2_Node_4`.
2.  **Bypass Steam Instance Lock:** 
    *   Inside each cloned folder, create a text file named `steam_appid.txt` and paste the Slay the Spire 2 Steam App ID (`1932700`) inside it. This allows the clones to run independently.
3.  **Assign Unique Ports:** 
    *   Install the **STS2MCP Mod** into every node folder.
    *   Open the mod's configuration file (`STS2MCP.conf`) in each node and assign a unique port. The defaults in `train.py` are:
        *   Node 1: **15526**
        *   Node 2: **15527**
        *   Node 3: **15528**
        *   Node 4: **15529** (Evaluation Node)
4.  **Configure Local Paths in Code:**
    *   **Environment Paths:** Open `sts2_env.py` and search for the `# MANUAL CONFIG` comment. Update the `base_node_path` variable to point to your new node folders.
    *   **Training Paths:** Open `train.py` and search for the `# MANUAL CONFIG` comment inside the `make_env` function. Update the `base_path` there as well.
5.  **Initialize Profiles:** 
    *   Manually open the game for each node at least once to ensure **3 profiles** are created. The orchestrator uses these to refresh memory and clear ghost saves.

### 🛡️ Automated Profile Isolation (API Stability)
To avoid the "Corrupt Save File" API startup error and ensure state integrity during parallel rollouts, the environment **automatically enforces** a unique profile mapping for each node. Slay the Spire 2 supports exactly 3 profiles:
*   **Node 1 (Port 15526):** Automatically switches to **Profile 1**.
*   **Node 2 (Port 15527):** Automatically switches to **Profile 2**.
*   **Node 3 (Port 15528):** Automatically switches to **Profile 3**.
*   **Node 4 (Evaluation - Port 15529):** Defaults to **Profile 1**.

This automated enforcement prevents the game engine from attempting to write/load from the same save metadata simultaneously. The orchestrator detects the current profile at the main menu and programmatically switches it if a mismatch is found.

### 🚨 CRITICAL CAUTION: Storage Bloat Bug (Godot Engine Logs)
Slay the Spire 2 is powered by the Godot engine, which generates detailed debug logs in the system's `AppData` directory (`%APPDATA%\SlayTheSpire2\logs\`). Under normal conditions, log utilization is remarkably low—typically remaining below a few megabytes or a gigabyte at most throughout extensive training flights.

⚠️ **KNOWN ENGINE BUG LOOP:** A massive storage consumption spike can happen if the evaluation or training nodes encounter a deterministic preloading/rendering exception. Specifically, when the model triggers the `PunchOff` event (most commonly observed on **Floor 14**), a native C# NullReferenceException inside the game's asynchronous animation thread causes an infinite error dump. Because this failure loops endlessly on every single frame tick (`_Process`), the log file can rapidly balloon by tens of gigabytes within minutes.

**High-Stability Hierarchy (Zero-Intervention Recovery):**
The environment features a five-layer **Self-Healing** system to handle engine crashes, lag spikes, and mod bugs autonomously:

1.  **Patient API Handshake (Level 1):** Godot engine loading screens can cause the API to return empty states for several seconds. The environment performs **20 micro-retries** (10ms intervals) before declaring a failure, providing a **10-second safety window** for room transitions.
2.  **60s Startup Safety (Level 2):** The orchestrator provides a generous **60-second window** for the game to reach the Main Menu during reboots, physically preventing the "Murder Loop" where the script kills the game while it is still booting.
3.  **Surgical Reboot Protocol (Level 3):** For standard failures (deadlocks or API stutters), the system identifies and kills **ONLY the specific frozen node** by its absolute process name (e.g., 'Node 1'). All other nodes continue training uninterrupted.
4.  **Ghost Save Recovery (Level 4):** Automatically detects and clears cached "Continue" states by toggling profiles (e.g., 1 -> 2 -> 1) upon reboot. This forces the Godot engine to refresh its metadata and realize the run is gone.
5.  **Full Cluster Purge (Level 5):** Triggered ONLY if `godot.log` exceeds 1GB. The system performs an exhaustive force-kill of Node 1, Node 2, Node 3, Node 4, and SlayTheSpire2 simultaneously to release shared file locks and autonomously delete the bloated log.
6.  **Void Reward Failsafe:** To maintain training integrity, any run interrupted by a technical reboot (Deadlock, Bloat, or Engine Bug) is automatically **Voided**. The environment returns a **0.0 reward** (Neutral Abort), ensuring that technical failures do not influence the neural policy.

**Additional Precision Layers:**
* **Poisoned Save Recovery:** Every reboot surgically deletes the `current_run.save` and `current_run.save.backup` for the specific profile. This clears bugged game states and ensures the node always starts at a fresh Main Menu.
* **Safe Throttle:** Implements a global **50ms input delay** (0.05s) to provide a safety buffer for the Godot engine's C# animation handlers, preventing "Thread Collisions" and hard crashes during heavy enemy turns.
* **Storage Bloat Failsafe:** The environment proactively monitors the Godot engine's log size. If `godot.log` exceeds **1 GB**, the system automatically triggers a Level 5 Purge reset.
* **Deadlock Recovery:** Lowers the technical reboot timer to **100 stagnant steps**, ensuring high cluster uptime and fast self-healing from rare engine freezes.

## ⚙️ Technical Deep Dive

### 1. The State Observation Vector (1536 Floats)
The environment translates the raw JSON game state into a massive, flattened observation vector to feed into the neural network. This allows the model to "see" every aspect of the game state simultaneously:
* **Observation Space:** `spaces.Box(low=-1.0, high=1.0, shape=(1536,), dtype=np.float32)`
    * **[0-10]**: Core Meta State (Floor, HP Ratio, Gold, current Act, Ascension level).
    * **[10-20]**: Pile Metadata (Draw, Discard, and Exhaust pile counts).
    * **[20-50]**: Player Status Effects (Detailed parsing for Strength, Dexterity, Vulnerable, etc.).
    * **[55-70]**: Orb States (Support for the Defect's unique mechanics).
    * **[70-120]**: Potion Bar (IDs, usability flags, and slot mapping).
    * **[120-320]**: Enemy Vision (Entity IDs, HP Ratios, Block, and detailed Intent Damage labels for up to 5 enemies).
    * **[320-620]**: Hand Vision (Full encoding for 10 cards: ID, Cost, Upgrades, and Keywords like 'Exhaust' or 'Block').
    * **[620-870]**: Dynamic Screen Logic (Handling Shop item costs, Reward item types, and Relic choice indices).
    * **[870-1370]**: Full Map Vision (The complete layout of the current act's navigation nodes).
    * **[1370-1536]**: Specialized States (Support for the Crystal Sphere minigame and Summoned Pets).

### 2. The Action Space (317 Discrete Actions)
A fully flattened discrete space representing exactly 317 possible interactions, categorized as:
*   **[0-49]**: `play_card` (Targets for 10 card slots * 5 possible enemies).
*   **[50-74]**: `use_potion` (5 slots * 5 targets).
*   **[75-79]**: `discard_potion`.
*   **[80]**: `end_turn`.
*   **[124-133]**: `choose_event_option` (Dialogue selections).
*   **[140-154]**: `shop_purchase` (Merchant inventory).
*   **[155-159]**: `choose_map_node` (Navigating the Spire).
*   **[215-316]**: Crystal Sphere minigame grid interaction.

### Synchronous Training Architecture (Stable Baselines3)
This project utilizes **Stable Baselines3 (SB3)** to orchestrate the training loop using vectorized environments:
* **SubprocVecEnv (Training):** Used for the training nodes. SB3's neural network update mechanism is entirely **synchronous**—it waits for the `n_steps` buffer to fill across *all* subprocesses before freezing to compute gradients.
* **DummyVecEnv (Evaluation):** Used for the single Evaluation node. It executes sequentially on the main thread.
* **The Bottleneck:** Because game APIs can sometimes stall, the synchronous nature means the fastest node is always waiting for the slowest node.
* **Experimentation Encouraged!** Experimentation with **asynchronous architectures** (such as Ray RLlib, SampleFactory, or async forks of SB3) is highly encouraged to solve the synchronous bottleneck!

### Synergy-CNN & MaskablePPO (Neural Framework)
The agent relies on a specialized neural network architecture tailored for Slay the Spire:
* **Maskable PPO (Proximal Policy Optimization):** Standard PPO wastes millions of steps trying to click disabled buttons. By utilizing `sb3-contrib`'s MaskablePPO, the environment passes a boolean mask array of length 317 alongside every observation. This guarantees that 100% of the gradient updates are spent evaluating valid tactical choices.
* **Safe Commitment (Multi-Card Selection):** To prevent infinite "toggle loops" where the agent unselects its first choice, the masking logic employs a **One-Way Gate**. Once a card is selected, its action slot is immediately blocked and the **Cancel** button is disabled until the menu is closed.
* **Repetition Protection:** A built-in blocker tracks the number of times an identical action is taken within the same game state. If any action is attempted **10 consecutive times** without producing a physical change in the game state, that action is strictly disabled.
* **Synergy-CNN Vision:** Rather than treating the player's hand as a flat array of numbers, the environment extracts the hand into a 1D tensor and passes it through a 1D-Convolutional branch (`nn.Conv1d`). This allows the network to natively detect spatial synergies—such as a zero-cost card sitting next to a high-damage card.

### PPO Brain Hyperparameters
* **Unrestricted Model Loader:** The training script performs an exhaustive recursive search across all sub-directories and automatically resumes from the absolute newest `.zip` file based on disk timestamp.
* **Intelligent Stage Selection (Zip-Peeking):** Upon resuming, the orchestrator "peeks" inside the model zip to identify the current internal step count and automatically selects the correct Stage parameters.
* **Rollout Buffer (Main Phase):** The main training stage uses `n_steps=3072` per node. With 3 training nodes, this results in exactly **9,216 steps** in the buffer before every neural weight update.
* **Checkpoints:** The model is saved to the `checkpoints/` directory immediately after every brain update.
* **Hall of Fame:** If a high score is achieved during a rollout, the **updated brain** (containing the new learning) is saved to `hall_of_fame/` at the start of the next update iteration.
* **Ultimate Best:** Reserved for **Phase Promotions**. When an agent successfully completes a Phase, its graduating brain is secured in the `ultimate_best/` directory for historical reference.

### 📊 Persistent Data & Session Logging
To guarantee 100% data integrity across cluster reboots, the project implements a **Fragment-and-Merge** logging architecture:
*   **Session Isolation:** Every time `train.py` starts, it creates a unique session folder (`logs/sb3_tech/session_{ID}/`). All SB3 metrics and TensorBoard events are isolated here to prevent overwriting historical data.
*   **Automatic Consolidation:** Upon startup, the orchestrator scans all session folders and automatically merges fragmented `progress.csv` files into a single master log.
*   **Ghost Step Truncation:** During synchronization, the system identifies and purges "ghost steps"—data recorded after the last brain checkpoint—to ensure your telemetry exactly matches your model's neural state.

### The Reward System (Master Class Telemetry)
A highly specialized, multi-layered reward system was designed to balance short-term survival with long-term scaling:

* **Step Tax (`-0.01`):** A minimal flat penalty applied to every single action. This allows the agent to safely scale in long Ascension 8+ fights.
* **Death Penalty (`-50.0`):** A massive flat penalty applied upon reaching the game over screen.
* **Combat Stall Penalty:** If a fight exceeds 20 turns, a compounding penalty (`-0.1` per extra turn) is applied. This prevents "infinite defense" loops and forces the agent to find efficient killing patterns.
* **Stagnation Tax Scaling:** To prevent behavioral "ruts" or infinite menu loops, the environment implements a compounding penalty based on consecutive stagnant steps.
    * **15+ steps:** `-2.0` penalty per action.
    * **30+ steps:** `-5.0` penalty per action.
    * **100+ steps:** `-20.0` penalty per action.
    * **200+ steps:** **Autonomous Surgical Reboot.**
* **Indecision Tax (Masked Actions):** To prevent the agent from "stalling" by spamming invalid inputs, every masked action attempt results in a light **`-1.0` penalty**.
* **Reward Telemetry:** Every action reward is logged in real-time within the `node_trace_{port}.jsonl` files, featuring a **Granular Breakdown** (e.g., `-> ACCEPTED. Net: +10.59 (Floor: +10.0, Dmg: +0.6, Tax: -0.01)`).
* **Dynamic Survival Instinct (Campfires):** Healing rewards and smithing bounties scale dynamically based on the agent's current **HP Ratio**:
    * **Ratio > 90%:** Healing is penalized (**-0.5x**), Smithing is boosted (**1.5x**).
    * **Ratio 80-90%:** Healing is neutral (**0.0x**), Smithing is encouraged (**1.2x**).
    * **Ratio 50-80%:** Standard operation (**0.5x** Heal, **1.0x** Smith).
    * **Ratio 30-50%:** Healing is prioritized (**1.0x**), Smithing is discouraged (**0.2x**).
    * **Ratio < 30%:** Healing is vital (**2.0x**), Smithing is heavily penalized (**-2.0x**).
* **HP Delta (Dynamic Strictness):** Dynamic rewards (**`+0.1`** per enemy HP damage dealt). Damage rewards are calculated using a **Unique Enemy Key** (ID + Position) to ensure 100% mathematical accuracy.
    * *Note: Class-specific passive bonuses (like Ironclad's Burning Blood) are naturally integrated into the HP Delta system, providing additional positive reinforcement for successful combat victories.*
* **The Universal Bounty Matrix:** To prevent exponential point inflation across phases, bounties are flattened into a universal constant:
    * *Universal Bounties:* **Floor (+10.0)**, Boss (+100.0), Elite (+30.0), Smith (+15.0 Base)

**Design Rationale:**
In early iterations, the agent suffered from "superstitious learning" or simply surviving without getting stronger. Implementing a strict Step Tax forces the agent to act efficiently. Scaling up the Bounty Matrix in later phases forces the agent to actively hunt Elites and prioritize Campfire Smithing.

**Reward Experimentation!**
RL is highly sensitive to reward shaping. Experimentation is highly encouraged—alter these numbers in `sts2_env.py` to test new hypotheses, experiment with new topologies, or incentivize entirely different playstyles.

### Phased Mastery Training (6-Phase Curriculum)
The agent progresses through a dynamic curriculum (Phase 1 to Phase 6). Progression requires passing a "Mastery Exam" on the dedicated Evaluation Node.

**Promotion Requirements:**
To advance to the next phase, the agent must meet both a Mean Floor and Mean Reward target during its 10-episode exam.
* **Phase 1 (Act 1 - Floor 16):** Requires Mean Floor >= 16.0 AND Mean Reward >= 200.0.
* **Phase 2 (Act 2 - Floor 33):** Requires Mean Floor >= 33.0 AND Mean Reward >= 450.0.
* **Phase 3 (Act 3 - Floor 50):** Requires Mean Floor >= 50.0 AND Mean Reward >= 700.0.
* **Phase 4 (Ascension 1-4):** Requires Mean Floor >= 51.0 AND Mean Reward >= 650.0.
* **Phase 5 (Ascension 5-8):** Requires Mean Floor >= 51.0 AND Mean Reward >= 600.0.
* **Phase 6 (Ascension 9-10):** Requires Mean Floor >= 51.0 AND Mean Reward >= 550.0.

**Demotion (Step-Down Recovery):**
If performance stalls in Phase 2+, the system implements a recovery step-down. If the agent fails to reach the doorstep thresholds (Floor 16, 33, or 50) for **3 consecutive exams**, it drops down exactly 1 Phase.

## 🛠️ Prerequisites

Before running this project, ensure the following are installed:
* **Windows PC:** (Required to run the game client and STS2MCP API mod natively)
* **Python 3.11.15:** (The environment was built and tested on 3.11.15 using **Miniconda**)
* **Slay the Spire 2:** Version `0.103.2 (2026.04.16)` was used during development.
* **STS2MCP API Mod:** Version `0.4.0` was used during development.

## 📦 Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/BonillaJP/STS2RL.git
    cd STS2RL
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 🎮 Usage

Initialize the training environment and connect to the local game nodes by running the main Python script:

```bash
python train.py
```

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

To modify the core behavior of the agent or optimize for different hardware, refer to these critical variables:

### 1. Training Dynamics (`train.py`)
*   **`BUFFER_STAGES`**: A curriculum list where each entry is `(Batch Size, Steps per Update, Max Step)`. 
    *   *Tip:* If your GPU memory is low, decrease the **Batch Size** (e.g., 3072 -> 1024).
*   **`ent_coef`**: Controls the entropy coefficient (exploration). 
    *   *Tip:* Set this to **0.05** for early training (Phase 1-2). The **Smart Entropy Defibrillator** will automatically boost this if the agent becomes stuck in a tactical rut in Act 1.
*   **`learning_rate`**: Fixed at **1e-4** for early acts. If the agent's policy collapses (High `approx_kl`), lower this to **5e-5**.

### 2. Environment Stability (`sts2_env.py`)
*   **`time.sleep(0.05)`**: The **Stability Throttle**. 
    *   *Tip:* If the Godot engine crashes during heavy enemy animations, increase this to **0.1s**. If your PC is high-end, you can try lowering it to **0.01s** for extreme SPS.
* **Deadlock Threshold:** If the agent is training on high Ascension levels where combat lasts longer, the `stagnant_steps` threshold (default 100) may be increased to prevent premature reboots.
*   **`rejection_penalty`**: Currently set to **-1.0**. 
    *   *Tip:* If the agent is "lazily" spamming end-turn to avoid fighting, increase this penalty to **-5.0** to force better mask adherence.

:** If the agent is training on high Ascension levels where combat lasts longer, the `stagnant_steps` threshold (default 100) may be increased to prevent premature reboots.
*   **`rejection_penalty`**: Currently set to **-1.0**. 
    *   *Tip:* If the agent is "lazily" spamming end-turn to avoid fighting, increase this penalty to **-5.0** to force better mask adherence.

