import asyncio
from typing import List, Optional, Literal
from pydantic import BaseModel, Field

# --- Pydantic Models ---

class Trade(BaseModel):
    buyer: str
    seller: str
    time: int

class Action(BaseModel):
    action_type: Literal["analyze", "get_user_trades", "submit"]
    content: str

class Observation(BaseModel):
    visible_trades: List[Trade]
    query_count: int
    step: int

class StepResult(BaseModel):
    observation: Observation
    reward: float
    done: bool

# --- Environment Implementation ---

EPS = 1e-6

def safe_reward(r: float) -> float:
    """Strictly clamps reward between (0, 1)."""
    if r <= 0: return EPS
    if r >= 1: return 1 - EPS
    return r

class TradeGuardEnv:
    def __init__(self):
        # 3 deterministic tasks with added noise trades
        self.tasks = [
            {
                "id": "EASY",
                "description": "Self-trade detection (A -> A)",
                "trades": [
                    Trade(buyer="User_X", seller="User_Y", time=100),
                    Trade(buyer="Trader_A", seller="Trader_A", time=105), # Wash trade
                    Trade(buyer="User_Z", seller="User_W", time=110),
                    Trade(buyer="User_1", seller="User_2", time=115), # Noise
                    Trade(buyer="User_3", seller="User_4", time=120), # Noise
                ],
                "ground_truth": "Trader_A->Trader_A"
            },
            {
                "id": "MEDIUM",
                "description": "Ping-pong detection (A -> B -> A)",
                "trades": [
                    Trade(buyer="User_X", seller="User_Y", time=200),
                    Trade(buyer="Trader_A", seller="Trader_B", time=205), # Step 1
                    Trade(buyer="User_Z", seller="User_W", time=210),
                    Trade(buyer="Trader_B", seller="Trader_A", time=215), # Step 2
                    Trade(buyer="User_1", seller="User_2", time=220), # Noise
                    Trade(buyer="User_3", seller="User_4", time=225), # Noise
                ],
                "ground_truth": "Trader_A->Trader_B->Trader_A"
            },
            {
                "id": "HARD",
                "description": "Cycle detection (A -> B -> C -> A)",
                "trades": [
                    Trade(buyer="User_1", seller="User_2", time=300),
                    Trade(buyer="Trader_A", seller="Trader_B", time=305), # Step 1
                    Trade(buyer="Trader_B", seller="Trader_C", time=310), # Step 2
                    Trade(buyer="User_3", seller="User_4", time=315),
                    Trade(buyer="Trader_C", seller="Trader_A", time=320), # Step 3
                    Trade(buyer="User_5", seller="User_6", time=325), # Noise
                    Trade(buyer="User_7", seller="User_8", time=330), # Noise
                ],
                "ground_truth": "Trader_A->Trader_B->Trader_C->Trader_A"
            }
        ]
        self.current_task_idx = 0
        self.current_step = 0
        self.max_steps = 10
        self.total_reward = 0.0
        self.is_done = False
        self.visible_trades = []
        self.query_count = 0

    @classmethod
    async def from_docker_image(cls, image_name=None):
        return cls()

    async def reset(self) -> Observation:
        """Resets the environment for the next task."""
        if self.current_task_idx >= len(self.tasks):
            self.current_task_idx = 0 
        
        self.current_step = 0
        self.total_reward = 0.0
        self.is_done = False
        self.query_count = 0
        
        # Start with only 2 initial trades visible to encourage exploration
        task = self.tasks[self.current_task_idx]
        self.visible_trades = task["trades"][:2]
        
        return self._get_observation()

    async def state(self) -> Observation:
        """Returns current observation."""
        return self._get_observation()

    async def step(self, action: Action) -> StepResult:
        """Executes one step in the environment."""
        if self.is_done:
            return StepResult(observation=self._get_observation(), reward=safe_reward(0.0), done=True)

        self.current_step += 1
        reward = 0.0
        
        task = self.tasks[self.current_task_idx]
        
        if action.action_type == "analyze":
            reward = 0.1
        elif action.action_type == "get_user_trades":
            # Search for trades involving the user specified in content
            user_id = action.content.strip()
            found_trades = [t for t in task["trades"] if t.buyer == user_id or t.seller == user_id]
            
            # Add new found trades to visible_trades, avoid duplicates
            new_trades = [t for t in found_trades if t not in self.visible_trades]
            self.visible_trades.extend(new_trades)
            
            self.query_count += 1
            reward = 0.2 if new_trades else 0.05
        elif action.action_type == "submit":
            if action.content.strip() == task["ground_truth"]:
                reward = 1.0
                self.is_done = True
            elif task["ground_truth"] in action.content or action.content in task["ground_truth"]:
                # Partial match: simple logic
                reward = 0.5
                self.is_done = True
            else:
                reward = -0.3 # Stronger penalty for wrong submission
                self.is_done = True
        
        # Apply step penalty: increases with time to encourage efficiency
        reward -= (0.05 * self.current_step)
        
        # FINAL STRICT WRAPPER
        reward = safe_reward(reward)
        
        self.total_reward += reward

        if self.current_step >= self.max_steps:
            self.is_done = True

        # If done, prepare for the next task index in the next reset
        if self.is_done:
            self.current_task_idx = (self.current_task_idx + 1) % len(self.tasks)

        return StepResult(
            observation=self._get_observation(),
            reward=reward,
            done=self.is_done
        )

    def _get_observation(self) -> Observation:
        return Observation(
            visible_trades=self.visible_trades,
            query_count=self.query_count,
            step=self.current_step
        )

# Global environment instance for simple loading
env = TradeGuardEnv()
