"""Simple classroom reward: small net-damage feedback + clear match outcome.

Compatible with SF2_Flexible_Classroom_PPO (ordinary PPO, six-frame steps).
The arena supplies damage_dealt/damage_taken with negative KO HP treated as 0.
A natural timeout uses the arena's win/loss/draw result, just like a KO.
A safety cutoff is not a loss. No distance, survival, action or time bonus.

With <=176 HP per fighter and no in-round healing, the sum of damage rewards
is bounded by [-0.5, 0.5]. A normal completed win therefore has positive raw
episode return, and a loss negative raw episode return. This is NOT a claim
about discounted returns, win rate, or empirically verified learning success.
"""
import math

REWARD_VERSION = "net_damage_outcome_v1"
FULL_HP = 176.0
DAMAGE_WEIGHT = 0.5
OUTCOME_BONUS = {
    "ongoing": 0.0,
    "win": 1.0,
    "loss": -1.0,
    "draw": 0.0,
    "cutoff": 0.0,
}


def compute_reward(t: dict) -> float:
    dealt = float(t["damage_dealt"])
    taken = float(t["damage_taken"])
    result = t["result"]

    # Do not silently replace invalid readings with a plausible reward.
    if not (math.isfinite(dealt) and math.isfinite(taken)):
        raise ValueError("Damage must be finite.")
    if not (0.0 <= dealt <= FULL_HP and 0.0 <= taken <= FULL_HP):
        raise ValueError("Damage is outside the expected 0..176 range.")
    if result not in OUTCOME_BONUS:
        raise ValueError(f"Unknown match result: {result!r}")

    # Count this step's damage even on the final step, then add outcome once.
    damage_reward = DAMAGE_WEIGHT * (dealt - taken) / FULL_HP
    return float(damage_reward + OUTCOME_BONUS[result])
