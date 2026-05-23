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

* **Synergy-CNN Vision:** The agent processes its hand using a 1D-Convolutional branch, allowing it to "scan" for card combos (e.g., Strength + Multi-hit) rather than just reading raw scalar numbers.
* **State-First Architecture:** Ensures perfect synchronization between the game's internal state and the agent's action masking. The environment performs a **Fresh API Fetch** at the start of every masking cycle, preventing desyncs in high-speed menus.
* **Dynamic Action Masking:** Implements rigorous action masking (exactly 317 discrete actions) with a **Persistent Repetition Blocker**.
    * **Selection Enforcement:** Distinguishes between **Grid Screens** (requiring confirmation) and **Choose Screens** (terminal selection). Automatically parses selection requirements from game prompts and masks terminal actions until requirements are met.
    * **Abyssal Baths Safety Threshold:** Includes a surgical safety mask for the repeatable 'Linger' option in Act 1. If the agent's HP falls below **30%**, the option is automatically disabled, forcing an exit and preventing intentional suicide for Max HP gains.
* **Stagnation Detection & Failsafes:** Built-in orchestrator that detects engine bugs, infinite loops, or **API/State Invalid** errors. It automatically reboots the specific game node and triggers a **Neutral Abort (0.0 reward)** to protect the model's policy from superstitious learning without halting the entire training cluster.
* **Environment Stability:** Features module-level global constant management and robust dictionary-safe state parsing to ensure reliable distributed training across multiple nodes.
* **Closed-Loop Telemetry:** Implements a "verify-then-delete" log consolidation system. Upon reboot, the orchestrator automatically merges fragmented CSVs into master files, truncates "ghost steps" to match the brain checkpoint, and purges technical telemetry (TensorBoard/CSV) created after the checkpoint timestamp to ensure 100% data integrity.

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
3.  **Differentiate Processes:** To allow the Python orchestrator to kill and restart specific frozen nodes independently, rename the game executable and its associated data file in each folder (e.g., rename `SlayTheSpire2.exe` and `SlayTheSpire2.pck` to `Node 1.exe` and `Node 1.pck`, `Node 2.exe`, etc.).
4.  **Configure the API Mod:** Install the STS2MCP mod into each folder. Open the mod's `.conf` file in each respective folder and assign their unique ports.

### 🚨 CRITICAL CAUTION: Storage Bloat Bug (Godot Engine Logs)
Slay the Spire 2 is powered by the Godot engine, which generates detailed debug logs in the system's `AppData` directory (`%APPDATA%\SlayTheSpire2\logs\`). Under normal conditions, log utilization is remarkably low—typically remaining below a few megabytes or a gigabyte at most throughout extensive training flights.

⚠️ **KNOWN ENGINE BUG LOOP:** A massive storage consumption spike can happen if the evaluation or training nodes encounter a deterministic preloading/rendering exception. Specifically, when the model triggers the `PunchOff` event (most commonly observed on **Floor 14**), a native C# NullReferenceException inside the game's asynchronous animation thread causes an infinite error dump. Because this failure loops endlessly on every single frame tick (`_Process`), the log file can rapidly balloon by tens of gigabytes within minutes.

**Current Mitigation Strategy:**
We do not currently have an automated software-level patch or command-line override configuration capable of breaking this internal engine loop. If you observe astronomical storage consumption or notice that your hard drive is losing space rapidly during exams, **you must manually delete the log files** from your system log directory. While the `_vacuum_disk` routine inside `sts2_env.py` performs aggressive disk cleanup during full system reboots, it is unable to release the file lock while an active rendering loop is running.

## ⚙️ Technical Deep Dive

### The State Observation Vector
The environment translates the raw JSON game state into a massive, flattened observation vector to feed into the neural network:
* **Observation Space:** `spaces.Box(low=-1.0, high=1.0, shape=(1536,), dtype=np.float32)`
    * *Includes HP, Block, Energy, Gold, Buffs/Debuffs (20 slots), Orbs, Potions, detailed intent and HP parsing for up to 5 enemies, dense card encoding (up to 10 hand cards with cost, type, enchants, and exhaust flags), Map Nodes, Relics, and even the new Crystal Sphere mechanic.*
* **Action Space:** `spaces.Discrete(317)` 
    * *A fully flattened discrete space representing exactly 317 possible interactions (e.g., 50 slots for playing 10 hand cards on 5 enemy targets, 99 slots for the Crystal Sphere 11x9 grid, plus shops, rewards, events, and rests).*

### Synchronous Training Architecture (Stable Baselines3)
This project utilizes **Stable Baselines3 (SB3)** to orchestrate the training loop using vectorized environments:
* **SubprocVecEnv (Training):** Used for the training nodes. It spawns completely separate Python multiprocessing threads for each game instance. While the python processes run in parallel, SB3's neural network update mechanism is still entirely **synchronous**—it waits for the `n_steps` buffer to fill across *all* subprocesses before freezing to compute gradients.
* **DummyVecEnv (Evaluation):** Used for the single Evaluation node. It executes sequentially on the main thread, which is perfectly efficient for a single instance.
* **The Bottleneck:** Because game APIs can sometimes stall or wait for animations, the synchronous nature means the fastest node is always waiting for the slowest node.
* **Experimentation Encouraged!** Experimentation with **asynchronous architectures** (such as Ray RLlib, SampleFactory, or async forks of SB3) is highly encouraged. Decoupling the game node stepping from the neural network updates would drastically increase training throughput (FPS) and solve the synchronous bottleneck!

### Synergy-CNN & MaskablePPO (Neural Framework)
The agent relies on a specialized neural network architecture tailored for Slay the Spire:
* **Maskable PPO (Proximal Policy Optimization):** Standard PPO wastes millions of steps trying to click disabled buttons. By utilizing `sb3-contrib`'s MaskablePPO, the environment passes a boolean mask array of length 317 alongside every observation. The network forces the logits of valid actions to `-infinity` before the softmax layer. This guarantees that 100% of the gradient updates are spent evaluating valid tactical choices.
* **Robust Multi-Card Selection:** To prevent infinite "toggle loops" where the agent unselects its first choice, the masking logic employs aggressive string-casting (`str()`) for all indices and scans for multiple API flag variations (`is_selected`, `isSelected`, `is_chosen`, `isChosen`). This ensures that once a card is selected, its action slot is immediately blocked, forcing the agent to fulfill multi-card requirements (e.g., "Choose 2") efficiently.
* **Synergy-CNN Vision:** Rather than treating the player's hand as a flat array of numbers, the environment extracts the hand into a 1D tensor and passes it through a 1D-Convolutional branch (`nn.Conv1d`).
    * *Technical Implementation:* It sweeps a `kernel_size=3` over the hand's feature channels (cost, exhaust flags, damage values, enchants). This allows the network to natively detect spatial synergies—such as a zero-cost card sitting next to a high-damage card, or overlapping enchantments—before feeding the flattened spatial data into the fully connected `[512, 256]` policy layers.

### PPO Brain Hyperparameters
* **Unrestricted Model Loader:** The training script performs an exhaustive recursive search across all sub-directories (`checkpoints/`, `models/`, `hall_of_fame/`, etc.) and automatically resumes from the absolute newest `.zip` file based on disk timestamp.
* **Intelligent Stage Selection (Zip-Peeking):** Upon resuming, the orchestrator "peeks" inside the model zip to identify the current internal step count. It then automatically selects the correct Stage parameters (`n_steps`, `batch_size`) *before* initializing the brain, eliminating redundant stage hand-off transitions during reboots.
* **Rollout Buffer (Warmup Phase):** For the first 18,432 steps, the buffer uses `n_steps=1024` per node, resulting in a **3,072 step** update frequency to quickly establish early baselines.
* **Rollout Buffer (Main Phase):** Following the warmup, the main training stage uses `n_steps=3072` per node. With 3 training nodes, this results in exactly **9,216 steps** in the buffer before every neural weight update.
* **Batch Size:** Dynamically scaled stage-by-stage (`256` in Warmup, `512` in Main).
* **Learning Rate:** Dynamically scaled per phase (`1e-4` in Phase 1, dropping to `5e-5` in Phase 4).
* **Checkpoints:** The model is saved to the `checkpoints/` directory immediately after every brain update. The "Hall of Fame" and "Ultimate Best" models are only overwritten during Mastery Exams if the agent breaks a historical high score.

### The Reward System (Master Class Telemetry)
A highly specialized, multi-layered reward system was designed to balance short-term survival with long-term scaling:

* **Step Tax (`-0.01`):** A minimal flat penalty applied to every single action. This allows the agent to safely scale in long Ascension 8+ fights without mathematical bankruptcy.
* **Death Penalty (`-50.0`):** A massive flat penalty applied upon reaching the game over screen. Because normal play is now net-positive (`+2.0`), this penalty is sufficient to discourage suicide without causing Value Scale Divergence.
* **Stagnation Tax Scaling:** To prevent behavioral "ruts" or infinite menu loops, the environment implements a compounding penalty based on consecutive stagnant steps. **Note:** Progress is now tracked via changes in Floor, HP, Gold, Block, Energy, and **Deck Size**, ensuring card removals and optimizations are recognized as progress.
    * **15+ steps:** `-2.0` penalty per action.
    * **30+ steps:** `-5.0` penalty per action.
    * **50+ steps:** `-20.0` penalty per action.
* **Indecision Tax (Masked Actions):** To prevent the agent from "stalling" by spamming invalid inputs, every masked action attempt results in a sharp **`-5.0` penalty**. This forces the policy to quickly explore valid paths forward when stuck.
* **Reward Telemetry:** Every action reward is logged in real-time within the `node_trace_{port}.jsonl` files (e.g., `-> ACCEPTED. Net Reward: +2.49`). A **1GB Rotation Failsafe** ensures these logs never bloat system storage.
* **Dynamic Survival Instinct (Campfires):** To mirror human risk assessment, healing rewards and smithing bounties scale dynamically based on the agent's current **HP Ratio**:
    * **Ratio > 90%:** Healing is penalized (**-0.5x**), Smithing is boosted (**1.5x**).
    * **Ratio 80-90%:** Healing is neutral (**0.0x**), Smithing is encouraged (**1.2x**).
    * **Ratio 50-80%:** Standard operation (**0.5x** Heal, **1.0x** Smith).
    * **Ratio 30-50%:** Healing is prioritized (**1.5x**), Smithing is discouraged (**0.2x**).
    * **Ratio < 30%:** Healing is vital (**4.0x**), Smithing is heavily penalized (**-2.0x**).
* **HP Delta (Dynamic Strictness):** Dynamic rewards (`+0.05` per enemy HP damage dealt). To enforce "Tiered Strictness" across the curriculum, the penalty multiplier for Player HP lost scales aggressively per Phase: **Phase 1 (0.1x), Phase 2 (0.15x), Phase 3 (0.2x), Phase 4 (0.4x), Phase 5 (0.6x), Phase 6 (0.8x)**. This explicitly allows sloppy plays in Phases 1-3, but demands absolute perfection in late Ascensions.
* **The Universal Bounty Matrix:** To prevent exponential point inflation across phases, bounties are flattened into a universal constant. The difficulty scales entirely through the HP Delta penalty. Note that **Smithing** values below are base values modified by the Survival Instinct multiplier:
    * *Universal Bounties:* Floor (+5.0), Boss (+100.0), Elite (+30.0), Smith (+15.0 Base)

**Design Rationale:**
In early iterations, the agent suffered from "superstitious learning" (e.g., repeating useless actions) or simply surviving without getting stronger. Implementing a strict Step Tax forces the agent to act efficiently. Scaling up the Bounty Matrix in later phases forces the agent to actively hunt Elites and prioritize Campfire Smithing, which are mathematically necessary to survive high Ascensions.

**Reward Experimentation!**
The current numeric values are estimated based on the "Master Class" architecture and have proven sufficient for advanced progression. However, RL is highly sensitive to reward shaping. Experimentation is highly encouraged—alter these numbers in `sts2_env.py` to test new hypotheses, experiment with new topologies, or incentivize entirely different playstyles.

### Phased Mastery Training (6-Phase Curriculum)
The agent progresses through a dynamic curriculum (Phase 1 to Phase 6). Progression requires passing a "Mastery Exam" on the dedicated Evaluation Node.

**Promotion Requirements:**
To advance to the next phase, the agent must meet both a Mean Floor and Mean Reward target during its 10-episode exam.
* **Phase 1 (Act 1 - Floor 16):** Focuses on basic survival. **Sloppy play is allowed.** Requires Mean Floor >= 16.0 AND Mean Reward >= 100.0.
* **Phase 2 (Act 2 - Floor 33):** Focuses on Act 2 scaling. **Sloppy play is allowed.** Requires Mean Floor >= 33.0 AND Mean Reward >= 325.0.
* **Phase 3 (Act 3 - Floor 50):** Focuses on beating the base game. **Sloppy play is allowed.** Requires Mean Floor >= 50.0 AND Mean Reward >= 550.0.
    * **Phase 3 Graduation Backup:** Upon successfully beating Phase 3, the environment secures a permanent backup of the model in `models/phase_3_graduates/`. It rotates the top 5 highest-scoring graduates. This ensures the baseline "game-beating" model is preserved before it faces the extreme mathematical strictness of the Ascension phases.
* **Phase 4 (Ascension 1-4):** Focuses on early Ascension mechanics (e.g., deadlier enemies, less max HP). **Strict efficiency is demanded.** Requires Mean Floor >= 51.0 AND Mean Reward >= 500.0.
* **Phase 5 (Ascension 5-8):** Focuses on mid Ascension scaling (tougher enemies, less healing). **High optimization required.** Requires Mean Floor >= 51.0 AND Mean Reward >= 450.0.
* **Phase 6 (Ascension 9-10):** Focuses on late Ascension mastery (tougher bosses, Ascender's Bane curse). **Demands absolute perfection.** Requires Mean Floor >= 51.0 AND Mean Reward >= 400.0.

**Demotion (Step-Down Recovery):**
If performance stalls in Phase 2+, the system implements a recovery step-down to repair foundational logic.
* **Doorstep Thresholds:** To prevent accidental demotion loops caused by boss-fight deaths, "success" is defined by reaching the Act's final door (Floor 16, 33, or 50).
* **Condition:** If the agent fails to reach these doorstep thresholds (0% success rate) for **3 consecutive exams**, it drops down exactly 1 Phase.

*Note: All targets (floors, rewards, and demotion triggers) are fully customizable in `train.py` depending on training goals.*

## 🛠️ Prerequisites

Before running this project, ensure the following are installed:
* **Windows PC:** (Required to run the game client and STS2MCP API mod natively)
* **Python 3.11.15:** (The environment was built and tested on 3.11.15 using **Miniconda** for environment management)
* **Slay the Spire 2:** Version `0.103.2 (2026.04.16)` was used during development.
* **STS2MCP API Mod:** Version `0.4.0` was used during development.
* The 4-Node game cluster set up as described above.

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

### Running the Training Cluster

Initialize the training environment and connect to the local game nodes by running the main Python script:

```bash
python train.py