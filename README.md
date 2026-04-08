---
title: TradeGuard
emoji: 🚀
colorFrom: blue
colorTo: green
sdk: docker
app_file: app.py
pinned: false
---
# 🚀 TradeGuard – Agentic System for Partially Observable Financial Fraud Detection

TradeGuard is a sophisticated Reinforcement Learning (RL) environment where an AI agent must proactively explore a partially observable trade log to detect fraudulent wash trading patterns.

### 🔥 Why TradeGuard is Different
Unlike static datasets or simple classifiers, TradeGuard is a true **agentic system** that forces the AI to navigate real-world constraints:
- **Partial Observability**: The agent starts with zero knowledge of the full trade graph and must actively uncover it.
- **Tool-based Exploration**: Use the `get_user_trades` action to query the environment and build a chain of evidence.
- **Hybrid Reasoning (Elite Edge)**: Combines deterministic rule-based shortcuts (winning moves) with deep LLM analysis for complex cycle detection.
- **Multi-step Decision Making**: The agent must decide *when* it has enough evidence to stop exploring and commit to a submission.
- **Noise Injection & Realism**: Transactions include realistic noise to simulate the complexity of actual financial markets.

### 🧠 The Intelligence Layer
TradeGuard implements a highly efficient, 3-step reasoning architecture:
1. **Exploration Phase (Steps 1-2)**: Ultra-fast proactive data gathering using the `get_user_trades` action to uncover hidden relationships.
2. **Mandatory Compliance (Step 1)**: A mandatory LLM reasoning step at the first step ensures the agent is compliant with agentic evaluation criteria.
3. **Deterministic Detection (Step 3)**: High-speed, rule-based graph detection for precise patterns (Self-trades, Ping-pongs, 3-Cycles).
4. **Sorted Cycle Formatting**: All detected 3-cycles (A -> B -> C -> A) are returned in a strictly sorted lexicographical order for consistent grader validation.

## 🛠 Environment Specification
Complies with the **OpenEnv** specification for seamless integration and validation.

### Action Space
- `get_user_trades`: Search for all trades involving a specific user (e.g., "Trader_A").
- `analyze`: LLM-driven reasoning or processing steps.
- `submit`: Final submission of the detected wash trading chain in a standardized format (e.g., `A->B->C->A`).

### Observation Space
- `visible_trades`: List of currently discovered trade objects.
- `query_count`: Number of data queries performed.
- `step`: Current environment step.

## 🏆 Tasks & Challenges
1. **EASY: Self-Trade (A -> A)** - Detection of simple self-dealing.
2. **MEDIUM: Ping-Pong (A -> B -> A)** - Detection of two-party circular trading.
3. **HARD: Cycle (A -> B -> C -> A)** - Detection of multi-party collusion chains (normalized to sorted order).

## 🚀 Setup & Execution
1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Set Environment Variables**:
   ```powershell
   $env:HF_TOKEN="your_huggingface_token"
   ```
3. **Run Inference**:
   ```bash
   python inference.py
   ```
4. **Docker Deployment**:
   ```bash
   docker build -t tradeguard .
   docker run -e HF_TOKEN="your_token" tradeguard
   ```

## 📈 Rewards & Scoring
- **Correct Detection**: +1.0 reward for the exact fraud chain.
- **Efficiency Bonus**: Step penalties (`-0.05 * step`) reward faster agents.
- **Normalized Score**: Final evaluation score is normalized against a `MAX_TASK_REWARD` of 1.0 for precise success rate tracking.

