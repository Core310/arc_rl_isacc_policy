import numpy as np
from typing import Dict
from agent.config import IDX_TURN_TOKEN, IDX_GO_SIGNAL

class MainNode:
    """
    Vehicle driver node inside an Agent.

    The Main node doesn't own the policy, the policy lives in SB3's
    RecurrentPPO and is called by the training loop. What Main does is:

    1. Inject Worker's commands into the observation before the policy
       sees it (turn_token -> vec[0], go_signal -> vec[1])

    2. Post-process the policy's raw action output with the go/brake
       inner loop: if go_signal == 0 (Scheduler says wait), override
       throttle to 0 and apply brake regardless of what the policy
       wants. This is a safety layer, not learned behavior.

    The nested loop structure is:
        OUTER: Worker decides direction -> turn_token conditions anchors
        INNER: Scheduler + visual scene -> go or brake
            - Scheduler go_signal == 0 -> hard brake (safety override)
            - Scheduler go_signal == 1 -> policy controls throttle/brake
              (policy still sees go_signal in vec[1] so it can learn
               to anticipate stops, but the override catches failures)

    This means:
        - The policy LEARNS to stop when go_signal is 0 (from experience)
        - The safety override GUARANTEES it stops (even if policy ignores)
        - Over training, the policy's behavior converges with the override
          and the override triggers less often
    """

    def __init__(self, agent_id: str, brake_decel: float = 0.8):
        """
        Args:
            agent_id: Parent agent identifier
            brake_decel: How hard to brake when go_signal is 0
                         Range [0, 1] where 1.0 = full brake
        """
        self.agent_id = agent_id
        self.brake_decel = brake_decel

    def prepare_observation(
        self,
        obs: Dict[str, np.ndarray],
        turn_token: int,
        go_signal: float,
    ) -> Dict[str, np.ndarray]:
        """
        Inject Worker's commands into the observation vector

        This is called BEFORE the policy forward pass so the policy
        can read the turn_token and go_signal as part of its input

        Args:
            obs: Raw observation from environment
            turn_token: Worker's discrete turn command {-1, 0, 1}
            go_signal: Scheduler's go/wait {0.0, 1.0}

        Returns:
            Modified observation dict (vec[0:2] overwritten)
        """
        obs = dict(obs)  # Shallow copy to avoid mutating env's obs
        vec = obs["vec"].copy()
        vec[IDX_TURN_TOKEN] = float(turn_token)
        vec[IDX_GO_SIGNAL] = float(go_signal)
        obs["vec"] = vec
        return obs

    def apply_go_brake_gate(
        self,
        action: np.ndarray,
        go_signal: float,
    ) -> np.ndarray:
        """
        Inner-loop go/brake safety override.

        If the Scheduler says WAIT (go_signal == 0), we override the
        policy's throttle/brake output to force a stop. This is the
        hard safety layer, the policy also sees go_signal and should
        learn to stop on its own, but this catches policy failures.

        If go_signal == 1 (GO), the policy's action passes through
        unchanged, the policy owns throttle/brake decisions during
        normal driving.

        Args:
            action: [steer, throttle, brake] from policy.
            go_signal: 0.0 = WAIT, 1.0 = GO.

        Returns:
            Potentially modified action array.
        """
        if go_signal < 0.5:
            # WAIT: override to stop
            action = action.copy()
            action[1] = 0.0                  # Zero throttle
            action[2] = self.brake_decel     # Apply brake
        return action


