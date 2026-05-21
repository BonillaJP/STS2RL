# STS2RL: Slay the Spire 2 Reinforcement Learning Agent

## 📜 Statement of Intent & Disclaimer
This project is developed strictly for **educational and research purposes**. It is an independent implementation designed to explore the application of specialized Reinforcement Learning (RL) in complex strategy games.

**Primary Objectives:**
*   **Specialized Logic:** While the base STS2MCP API allows for LLM-based interactions, LLMs often struggle with the intricate mathematical scaling of Slay the Spire 2. This project aims to build a dedicated, low-latency neural model capable of consistent, high-level tactical play.
*   **Future Interoperability:** A long-term goal is to leverage the API's cooperative multiplayer capabilities to facilitate multi-agent synergy benchmarking and collaborative AI-human gameplay.
*   **Integrity:** This project is not intended for exploitation, cheating, or gaining an unfair advantage in a commercial or competitive environment. It is a sandbox for advancement in game-centric AI.

An advanced Reinforcement Learning (RL) agent designed to play **Slay the Spire 2** autonomously. Powered by Stable Baselines3 (Maskable PPO) and a custom Gymnasium environment, this project implements a "Master Class" architecture capable of navigating complex combat, map routing, and menu interactions.

## 🚀 Key Features

*   **Synergy-CNN Vision:** The agent processes its hand using a 1D-Convolutional branch, allowing it to "scan" for card combos (e.g., Strength + Multi-hit) rather than just reading raw scalar numbers.
*   **State-First Architecture:** Ensures perfect synchronization between the game's internal state and the agent's action masking, preventing invalid moves and desyncs.
*   **Dynamic Action Masking:** Implements rigorous action masking (exactly 317 discrete actions) with robust selection detection. It natively prevents infinite toggle loops during multi-card selection events (e.g., "Choose 2 Cards") by utilizing aggressive string-casting for indices and expanded API flag detection.
*   **Stagnation Detection & Failsafes:** Built-in orchestrator that detects engine bugs or infinite loops and automatically reboots the specific game node without halting the entire training cluster.
*   **Environment Stability:** Features module-level global constant management and robust dictionary-safe state parsing to ensure reliable distributed training across multiple nodes.
*   **Closed-Loop Telemetry:** Implements a "verify-then-delete" log consolidation system. Upon reboot, the orchestrator automatically merges fragmented CSVs into master files, truncates "ghost steps" to match the brain checkpoint, and purges technical telemetry (TensorBoard/CSV) created after the checkpoint timestamp to ensure 100% data integrity.

## 🖥️ Cluster Setup & Game Cloning

To maximize training throughput, this project utilizes a **4-Node Cluster** running simultaneously. 

*   **Training Nodes:** 3 Instances (Ports 15526, 15527, 15528)
*   **Evaluation Node:** 1 Instance (Port 15529)

**Hardware Scaling Note:** This specific 4-node limit was chosen because the original host PC is limited to **16 GB of RAM**, which acts as the primary bottleneck for running Slay the Spire 2 instances concurrently. You can scale the cluster to run as few or as many nodes as your PC's memory can handle!

### Scaling the Cluster (Adding/Removing Nodes)
If you wish to run more (or fewer) nodes, you must perform the following:
1.  **Clone more game folders** and assign them unique ports via the STS2MCP mod config.
2.  Update the `TRAIN_PORTS` list in `train.py` to include the new ports.
3.  **Crucial:** Adjust `n_steps` and `batch_size` in the `BUFFER_STAGES` matrix in `train.py`. Because the total rollout buffer is `n_steps * number_of_nodes`, adding nodes means you should lower `n_steps` to maintain an optimal update frequency.
4.  *Note: You **DO NOT** need to change the Box Dimensions (1536) or Action Space (317). These define a single agent's interface and remain identical regardless of the cluster size.*

### ⚠️ Crucial Mod Credits (STS2MCP)
**This project would be absolutely impossible without the incredible STS2MCP API Mod.** 
The mod acts as the core bridge, exposing the internal Slay the Spire 2 game state as a JSON payload and accepting external HTTP POST commands. This project relies entirely on this mod to provide the Python RL environment with visibility and control inside the game. 
**Huge thanks to the creator:** [https://github.com/Gennadiyev/STS2MCP](https://github.com/Gennadiyev/STS2MCP)

### How to Replicate the Cluster
Because Steam normally restricts running multiple instances of a game, the following manual setup is required:

1.  **Clone the Game:** Copy the base `Slay the Spire 2` Steam installation directory to create separate game folders for each node.
2.  **Bypass Steam Lock:** Inside each cloned folder, create a text file named `steam_appid.txt` and paste the Slay the Spire 2 Steam App ID inside it. This allows the clones to run independently.
3.  **Differentiate Processes:** To allow the Python orchestrator to kill and restart specific frozen nodes independently, rename the game executable and its associated data file in each folder (e.g., rename `SlayTheSpire2.exe` and `SlayTheSpire2.pck` to `Node 1.exe` and `Node 1.pck`, `Node 2.exe`, etc.).
4.  **Configure the API Mod:** Install the STS2MCP mod into each folder. Open the mod's `.conf` file in each respective folder and assign their unique ports.

## ⚙️ Technical Deep Dive

### The State Observation Vector
The environment translates the raw JSON game state into a massive, flattened observation vector to feed into the neural network:
*   **Observation Space:** `spaces.Box(low=-1.0, high=1.0, shape=(1536,), dtype=np.float32)`
    *   *Includes HP, Block, Energy, Gold, Buffs/Debuffs (20 slots), Orbs, Potions, detailed intent and HP parsing for up to 5 enemies, dense card encoding (up to 10 hand cards with cost, type, enchants, and exhaust flags), Map Nodes, Relics, and even the new Crystal Sphere mechanic.*
*   **Action Space:** `spaces.Discrete(317)` 
    *   *A fully flattened discrete space representing exactly 317 possible interactions (e.g., 50 slots for playing 10 hand cards on 5 enemy targets, 99 slots for the Crystal Sphere 11x9 grid, plus shops, rewards, events, and rests).*

### Synchronous Training Architecture (Stable Baselines3)
This project utilizes **Stable Baselines3 (SB3)** to orchestrate the training loop using vectorized environments:
*   **SubprocVecEnv (Training):** Used for the training nodes. It spawns completely separate Python multiprocessing threads for each game instance. While the python processes run in parallel, SB3's neural network update mechanism is still entirely **synchronous**—it waits for the `n_steps` buffer to fill across *all* subprocesses before freezing to compute gradients.
*   **DummyVecEnv (Evaluation):** Used for the single Evaluation node. It executes sequentially on the main thread, which is perfectly efficient for a single instance.
*   **The Bottleneck:** Because game APIs can sometimes stall or wait for animations, the synchronous nature means the fastest node is always waiting for the slowest node.
*   **Experimentation Encouraged!** Experimentation with **asynchronous architectures** (such as Ray RLlib, SampleFactory, or async forks of SB3) is highly encouraged. Decoupling the game node stepping from the neural network updates would drastically increase training throughput (FPS) and solve the synchronous bottleneck!

### Synergy-CNN & MaskablePPO (Neural Framework)
The agent relies on a specialized neural network architecture tailored for Slay the Spire:
*   **Maskable PPO (Proximal Policy Optimization):** Standard PPO wastes millions of steps trying to click disabled buttons. By utilizing `sb3-contrib`'s MaskablePPO, the environment passes a boolean mask array of length 317 alongside every observation. The network forces the logits of invalid actions to `-infinity` before the softmax layer. This guarantees that 100% of the gradient updates are spent evaluating valid tactical choices.
*   **Robust Multi-Card Selection:** To prevent infinite "toggle loops" where the agent unselects its first choice, the masking logic employs aggressive string-casting (`str()`) for all indices and scans for multiple API flag variations (`is_selected`, `isSelected`, `is_chosen`, `isChosen`). This ensures that once a card is selected, its action slot is immediately blocked, forcing the agent to fulfill multi-card requirements (e.g., "Choose 2") efficiently.
*   **Synergy-CNN Vision:** Rather than treating the player's hand as a flat array of numbers, the environment extracts the hand into a 1D tensor and passes it through a 1D-Convolutional branch (`nn.Conv1d`).
    *   *Technical Implementation:* It sweeps a `kernel_size=3` over the hand's feature channels (cost, exhaust flags, damage values, enchants). This allows the network to natively detect spatial synergies—such as a zero-cost card sitting next to a high-damage card, or overlapping enchantments—before feeding the flattened spatial data into the fully connected `[256, 256]` policy layers.

### PPO Brain Hyperparameters
*   **Unrestricted Model Loader:** The training script performs an exhaustive recursive search across all sub-directories (`checkpoints/`, `models/`, `hall_of_fame/`, etc.) and automatically resumes from the absolute newest `.zip` file based on disk timestamp.
*   **Intelligent Stage Selection (Zip-Peeking):** Upon resuming, the orchestrator "peeks" inside the model zip to identify the current internal step count. It then automatically selects the correct Stage parameters (`n_steps`, `batch_size`) *before* initializing the brain, eliminating redundant stage hand-off transitions during reboots.
*   **Rollout Buffer (Warmup Phase):** For the first 18,432 steps, the buffer uses `n_steps=1024` per node, resulting in a **3,072 step** update frequency to quickly establish early baselines.
*   **Rollout Buffer (Main Phase):** Following the warmup, the main training stage uses `n_steps=3072` per node. With 3 training nodes, this results in exactly **9,216 steps** in the buffer before every neural weight update.
*   **Batch Size:** 512.
*   **Learning Rate:** Dynamically scaled per phase (`3e-4` in Phase 1, dropping to `5e-5` in Phase 4).
*   **Checkpoints:** The model is saved to the `checkpoints/` directory immediately after every brain update. The "Hall of Fame" and "Ultimate Best" models are only overwritten during Mastery Exams if the agent breaks a historical high score.

### The Reward System (Master Class Telemetry)
A highly specialized, multi-layered reward system was designed to balance short-term survival with long-term scaling:

*   **Step Tax (`-0.10`):** A flat penalty applied to every single action. 
*   **HP Delta:** Dynamic rewards (`+0.05` per enemy HP damage dealt) and penalties based on the agent's health lost. 
*   **The Phase Bounty Matrix:** Massive point spikes awarded for crucial game milestones. 
    *   *Phase 1:* Floor (+2.5), Boss (+100.0), Elite (+25.0), Smith (+10.0), HP Multiplier (x0.1)
    *   *Phase 2:* Floor (+5.0), Boss (+150.0), Elite (+45.0), Smith (+20.0), HP Multiplier (x0.2)
    *   *Phase 3:* Floor (+7.5), Boss (+200.0), Elite (+70.0), Smith (+30.0), HP Multiplier (x0.3)
    *   *Phase 4:* Floor (+10.0), Boss (+300.0), Elite (+100.0), Smith (+50.0), HP Multiplier (x0.5)

**Design Rationale:**
In early iterations, the agent suffered from "superstitious learning" (e.g., repeating useless actions) or simply surviving without getting stronger. Implementing a strict Step Tax forces the agent to act efficiently. Scaling up the Bounty Matrix in later phases forces the agent to actively hunt Elites and prioritize Campfire Smithing, which are mathematically necessary to survive high Ascensions.

**Reward Experimentation!**
The current numeric values are estimated based on the "Master Class" architecture and have proven sufficient for advanced progression. However, RL is highly sensitive to reward shaping. Experimentation is highly encouraged—alter these numbers in `sts2_env.py` to test new hypotheses, experiment with new topologies, or incentivize entirely different playstyles.

### Phased Mastery Training
The agent progresses through a dynamic curriculum (Phase 1 to Phase 4). Progression requires passing a "Mastery Exam" on the dedicated Evaluation Node.

**Promotion Requirements:**
To advance to the next phase, the agent must meet *both* a Mean Floor and Mean Reward target during its 10-episode exam:
*   **Phase 1 (Act 1 Mastery):** To promote, requires **Mean Floor >= 12.0** AND **Mean Reward >= 15.0**. (Target Win Floor: 17)
*   **Phase 2 (Act 2 Mastery):** To promote, requires **Mean Floor >= 28.0** AND **Mean Reward >= 40.0**. (Target Win Floor: 34)
*   **Phase 3 (Act 3 Mastery):** To promote, requires **Mean Floor >= 45.0** AND **Mean Reward >= 80.0**. (Target Win Floor: 51)
*   **Phase 4 (Maximum Mastery):** The highest difficulty. The `ent_coef` (entropy) is aggressively decayed from `0.05` down to `0.01`, shifting the brain from wild exploration to pure, deterministic exploitation.

**Demotion (Step-Down Recovery):**
If the agent is struggling in a higher phase (Phase 2+), it is dynamically demoted to repair its foundational logic. 
*   **Condition:** If the agent achieves a **0% success rate** (0 wins) at hitting the target floor for **3 consecutive exams**, it drops down exactly 1 Phase (e.g., Phase 3 -> Phase 2).

*Note: All targets (floors, rewards, and demotion triggers) are fully customizable in `train.py` depending on training goals.*

## 🛠️ Prerequisites

Before running this project, ensure the following are installed:
*   **Windows PC:** (Required to run the game client and STS2MCP API mod natively)
*   **Python 3.11.15:** (The environment was built and tested on 3.11.15 using **Miniconda** for environment management)
*   **Slay the Spire 2:** Version `0.103.2 (2026.04.16)` was used during development.
*   **STS2MCP API Mod:** Version `0.4.0` was used during development.
*   The 4-Node game cluster set up as described above.

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

### Running the Training Cluster

Initialize the training environment and connect to the local game nodes by running the main Python script:

```bash
python train.py
```

### Monitoring Progress

To monitor the agent's learning progress using TensorBoard:

```bash
tensorboard --logdir ./tensorboard
```

To run the custom log analysis script for a formatted progress report:
```bash
python analyze_logs.py
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
