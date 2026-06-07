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
        * **Patient API Fetching:** Implements a robust state retriever that performs 10 micro-retries (10ms intervals) before declaring an API failure. This allows the Godot engine to finish room transitions and loading spikes without triggering technical reboots.
        * **Lightning Reboot Sequence:** Replaces hardcoded 30s warmup delays with a high-speed Port Polling loop. The orchestrator checks the game's API port every 1s, resuming training the absolute millisecond the game becomes alive.
        * **Deterministic Loop Protection:** Differentiates between **Hard Progress** (changes in Floor, HP, Gold, or Deck Size) and **Screen Progress**. Action-use counters (the 10-use limit) are strictly preserved across screen transitions and only reset when Hard Progress is detected, physically preventing infinite menu loops.
        * **Fast Mode Dynamic Polling:** Replaces hardcoded thread sleeps with a high-speed polling loop. During screen transitions, the environment polls the game state every 50ms, instantly resuming training the absolute millisecond the transition finishes.
        * **Animation Cooldown Enforcement:** Applies a strict 5-step RL mask cooldown to rapid actions (like playing cards or using potions) to prevent the agent from spamming actions faster than the engine can process them.
        * **Smart Progress Tracking:** The environment implements a multi-dimensional stagnation tracker. Counters are automatically reset if the system detects changes in **HP, Gold, Floor, Screen Type, Energy, Block, Deck Size, or Card Selection history.** This ensures the 100-step reboot failsafe only triggers for true engine freezes.
    * **Absolute path-based Force-Kill:** The reboot logic now performs an exhaustive path-based cleanup of the node's specific folder before relaunching. This physically prevents the spawning of duplicate game instances and ensures file locks on `godot.log` are released.
    * **Overlay-First Priority:** Selection overlays (Exhaust, Scry, Choose) are prioritized over combat actions, physically forcing the agent to resolve choices before fighting to prevent deadlocks.
    * **Safe Throttle:** Implements a global **10ms input delay** (0.01s) to provide a safety buffer for the Godot engine's C# animation handlers.
    * **Boss-Only Map Proceed:** The map `proceed` button is strictly disabled unless the agent is on a Boss Floor (16, 33, 51), preventing the "Fake Proceed" trap.
    * **Real-Map Verification:** The environment only identifies the screen as "map" if actual navigation nodes are present in the API response.
* **Stagnation Detection & Failsafes:** Built-in orchestrator that detects engine bugs, infinite loops, or **API/State Invalid** errors. It automatically reboots the specific game node and triggers a **Neutral Abort (0.0 reward)** to protect the model's policy.
* **Deadlock Recovery:** Lowers the technical reboot timer to **100 stagnant steps**, ensuring high cluster uptime and fast self-healing from rare engine freezes.
* **Environment Stability:** Features module-level global constant management and robust dictionary-safe state parsing to ensure reliable distributed training across multiple nodes.
* **Closed-Loop Telemetry:** Implements a "verify-then-delete" log consolidation system. Upon reboot, the orchestrator automatically merges fragmented CSVs into master files, truncates "ghost steps" to match the brain checkpoint, and purges technical telemetry created after the checkpoint timestamp.

## 🖥️ Cluster Setup & Game Cloning

To maximize training throughput, this project utilizes a **4-Node Cluster** running simultaneously. 

* **Training Nodes:** 3 Instances (Ports 15526, 15527, 15528)
* **Evaluation Node:** 1 Instance (Port 15529)

**Hardware Scaling Note:** This specific 4-node limit was chosen because the original host PC is limited to **16 GB of RAM**, which acts as the primary bottleneck for running Slay the Spire 2 instances concurrently. You can scale the cluster to run as few or as many nodes as your PC's memory can handle!

### Scaling the Cluster (Adding/Removing Nodes)
If you wish to run more (or fewer) nodes, you must perform the following:
1.  **Clone more game folders** and assign them unique ports via the STS2MCP mod config.
2.  Update the `TRAIN_PORTS` list in `train.py` to include the new ports.
3.  **Crucial:** Adjust `n_steps` and `batch_size` in the `BUFFER_STAGES` matrix in `train.py`. Because the total rollout buffer is `n_steps * number_of_nodes`, adding nodes means you should lower `n_steps` to maintain an optimal update frequency.
4.  *Note: You **DO NOT** need to change the Box Dimensions (1536) or Action Space (317). These define a single agent's interface and remain identical regardless of the cluster size.*

### ⚠️ Crucial Mod Credits (STS2MCP)
**This project would be absolutely impossible without the incredible STS2MCP API Mod.** The mod acts as the core bridge, exposing the internal Slay the Spire 2 game state as a JSON payload and accepting external HTTP POST commands. This project relies entirely on this mod to provide the Python RL environment with visibility and control inside the game. 
**Huge thanks to the creator:** [https://github.com/Gennadiyev/STS2MCP](https://github.com/Gennadiyev/STS2MCP)

### How to Replicate the Cluster
Because Steam normally restricts running multiple instances of a game, the following manual setup is required:

1.  **Clone the Game:** Copy the base `Slay the Spire 2` Steam installation directory to create separate game folders for each node.
2.  **Bypass Steam Lock:** Inside each cloned folder, create a text file named `steam_appid.txt` and paste the Slay the Spire 2 Steam App ID inside it. This allows the clones to run independently.
3.  **Differentiate Processes:** To allow the Python orchestrator to kill and restart specific frozen nodes independently, rename the game executable and its associated data file in each folder (e.g., rename `Node 1.exe` and `Node 1.pck`, `Node 2.exe`, etc.). Rename the folders as well (e.g., `STS2_Node_1`, `STS2_Node_2`, etc.).
4.  **Configure Paths:** Update the hardcoded `base_path` variables in `sts2_env.py` and `train.py` to match the exact directory where your game nodes are located.
5.  **Configure the API Mod:** Install the STS2MCP mod into each folder. Open the mod's `.conf` file in each respective folder and assign their unique ports.

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

**Automated Mitigation Strategy:**
The environment now includes a multi-layered **Self-Healing** system to handle this automatically:
1.  **Fast Timeout:** API timeouts are set to **5.0s**. If a node freezes, it is detected and rebooted within seconds.
2.  **Real-Time Size Monitor:** The environment checks the size of `godot.log` every 25 actions. If it exceeds **1 GB**, an immediate "Storage Bloat Failsafe" reboot is triggered.
3.  **Aggressive Vacuuming:** Upon reboot, the system force-kills the game node and purges `godot.log` if it is over the 1 GB safety threshold. **Note:** All game nodes must be terminated to release the shared file lock on `godot.log` for successful deletion.
4.  **Absolute Mask Hardening:** To prevent technical deadlocks, the environment employs a **Finality-First** masking architecture. 
    *   **Deterministic Map Navigation:** The `proceed` button is strictly disabled if Map Nodes are available, forcing a destination choice.
    *   **Repetition Finality:** The 10-click repetition blocker is the **final step** in the masking process, ensuring it can never be overridden by safety fallbacks.
5. **Global Stagnation Shield:** To prevent technical deadlocks and minimize cluster-wide reboots, the environment employs a **Surgical Recovery** system. If an agent fails to produce physical progress after **20 rejected attempts** or **100 stagnant steps**, the environment programmatically executes an intelligent "Smart Kick". For example, if stuck on the map screen, it intelligently identifies and selects an available map node. Crucially, the 10-use action limit is only reset on **Hard Progress**, ensuring that if a Smart Kick triggers a screen change without real progress, the looping action remains masked. This safely breaks the loop and resets the stagnation timers before the 100-step technical reboot failsafe can trigger.

## ⚙️ Technical Deep Dive

### The State Observation Vector
The environment translates the raw JSON game state into a massive, flattened observation vector to feed into the neural network:
* **Observation Space:** `spaces.Box(low=-1.0, high=1.0, shape=(1536,), dtype=np.float32)`
    * *Includes HP, Block, Energy, Gold, Buffs/Debuffs (20 slots), Orbs, Potions, detailed intent and HP parsing for up to 5 enemies, dense card encoding, Map Nodes, Relics, and even the new Crystal Sphere mechanic.*
* **Action Space:** `spaces.Discrete(317)` 
    * *A fully flattened discrete space representing exactly 317 possible interactions.*

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
    * **50+ steps:** `-20.0` penalty per action.
* **Indecision Tax (Masked Actions):** To prevent the agent from "stalling" by spamming invalid inputs, every masked action attempt results in a light **`-1.0` penalty**.
* **Reward Telemetry:** Every action reward is logged in real-time within the `node_trace_{port}.jsonl` files, now featuring a **Granular Breakdown** (e.g., `-> ACCEPTED. Net: +10.59 (Floor: +10.0, Dmg: +0.6, Tax: -0.01)`).
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
