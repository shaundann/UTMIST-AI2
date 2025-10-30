# # SUBMISSION: Agent
# This will be the Agent class we run in the 1v1. We've started you off with a functioning RL agent (`SB3Agent(Agent)`) and if-statement agent (`BasedAgent(Agent)`). Feel free to copy either to `SubmittedAgent(Agent)` then begin modifying.
# 
# Requirements:
# - Your submission **MUST** be of type `SubmittedAgent(Agent)`
# - Any instantiated classes **MUST** be defined within and below this code block.
# 
# Remember, your agent can be either machine learning, OR if-statement based. I've seen many successful agents arising purely from if-statements - give them a shot as well, if ML is too complicated at first!!
# 
# Also PLEASE ask us questions in the Discord server if any of the API is confusing. We'd be more than happy to clarify and get the team on the right track.
# Requirements:
# - **DO NOT** import any modules beyond the following code block. They will not be parsed and may cause your submission to fail validation.
# - Only write imports that have not been used above this code block
# - Only write imports that are from libraries listed here
# We're using PPO by default, but feel free to experiment with other Stable-Baselines 3 algorithms!

import os
import gdown
from typing import Optional
from environment.agent import Agent
from stable_baselines3 import PPO, A2C # Sample RL Algo imports
from sb3_contrib import RecurrentPPO # Importing an LSTM

# To run the sample TTNN model, you can uncomment the 2 lines below: 
# import ttnn
# from user.my_agent_tt import TTMLPPolicy


class SubmittedAgent(Agent):
    """
    Tournament-tuned, weapon-aware, finite-state agent.
    Action: [move (-1/0/1), jump (0/1), attack (0/1)]

    Pillars:
      • Aggro-first with 1-step opponent prediction
      • Weapon-aware spacing (Spear > Hammer > Punch)
      • Spawner denial when unarmed (Spear priority)
      • Edge-guard when opponent is recovering below you
      • Evasive window on damage spikes; reset when very high total damage
      • Small combo-chains after a connect/pre-swing
    """

    # ---- Tunables (safe to tweak) ----
    PREFERRED_WEAPONS = ["Spear", "Hammer", "Punch", "Random"]
    DANGER_TOTAL_DAMAGE = 85.0
    DANGER_FRAME_DAMAGE = 11.0
    EVADE_FRAMES = 20           # frames to stay evasive after a spike
    RESET_FRAMES = 90           # frames to play neutral after very high total damage

    DEFAULT_ATTACK_DIST = 1.2
    DEFAULT_CLOSE_DIST  = 2.0

    WEAPON_DIST = {
        "Spear":  {"attack": 2.0, "close": 3.0},
        "Hammer": {"attack": 1.5, "close": 2.4},
        "Punch":  {"attack": 1.1, "close": 1.8},
        "Random": {"attack": 1.4, "close": 2.2},
        None:     {"attack": 1.2, "close": 2.0},
    }

    COMBO_CHAIN_FRAMES = 8
    HIGH_DIFF = 1.0
    LOW_DIFF  = -0.8

    def __init__(self, file_path: Optional[str] = None):
        super().__init__(file_path)
        self._loaded_model = False

        # Memory
        self.frame = 0
        self.last_player_pos = (0.0, 0.0)
        self.last_opponent_pos = (1.0, 0.0)
        self.last_player_weapon = None
        self.last_player_frame_dmg = 0.0

        # State machine
        self.state = "ENGAGE"  # ENGAGE | EVADE | CHASE_SPAWNER | EDGE_GUARD | RESET
        self.state_timer = 0
        self.combo_timer = 0
        self.evasive_timer = 0
        self.reset_timer = 0

    # ---------- Lifecycle ----------
    def _initialize(self) -> None:
        if self.file_path is None:
            self.model = PPO("MlpPolicy", self.env, verbose=0)
            del self.env
            self._loaded_model = True
        else:
            try:
                self.model = PPO.load(self.file_path)
                self._loaded_model = True
            except Exception as e:
                print(f"[SubmittedAgent] PPO load failed: {e}")
                self.model = None
                self._loaded_model = False

    def _gdown(self) -> str:
        data_path = "rl-model.zip"
        if not os.path.isfile(data_path):
            print(f"Downloading {data_path}...")
            url = "https://drive.google.com/file/d/1JIokiBOrOClh8piclbMlpEEs6mj3H1HJ/view?usp=sharing"
            gdown.download(url, output=data_path, fuzzy=True)
        return data_path

    # ---------- Core ----------
    def predict(self, obs):
        self.frame += 1

        # If PPO present and compatible, use it
        if getattr(self, "model", None) is not None and self._loaded_model:
            try:
                act, _ = self.model.predict(obs, deterministic=True)
                return act
            except Exception:
                pass  # fall back to rules

        # Parse observation
        px, py, ox, oy, p_total_dmg, p_frame_dmg = self._parse_positions_and_damage(obs)
        weapon = self._parse_weapon(obs)
        spawners = self._parse_spawners(obs)

        # Derived
        dx, dy = (ox - px), (oy - py)
        dist2 = dx*dx + dy*dy
        vx_opp, vy_opp = self._velocity(self.last_opponent_pos, (ox, oy))
        pred_ox = ox + vx_opp * 0.4
        pred_oy = oy + vy_opp * 0.4

        # Update memory
        self.last_player_pos = (px, py)
        self.last_opponent_pos = (ox, oy)
        self.last_player_weapon = weapon

        # --- Global interrupts ---
        if p_frame_dmg is not None and p_frame_dmg > self.DANGER_FRAME_DAMAGE:
            self.evasive_timer = self.EVADE_FRAMES
        if self.evasive_timer > 0:
            self.state = "EVADE"

        if p_total_dmg is not None and p_total_dmg > self.DANGER_TOTAL_DAMAGE and self.state != "EVADE":
            self.state = "RESET"
            self.reset_timer = self.RESET_FRAMES

        if dy < self.LOW_DIFF and abs(dx) < 2.2 and self.state != "EVADE":
            self.state = "EDGE_GUARD"
            self.state_timer = 10

        if weapon is None and spawners and self.state not in ("EVADE", "EDGE_GUARD"):
            self.state = "CHASE_SPAWNER"
            self.state_timer = 18

        # timers
        if self.state_timer > 0:
            self.state_timer -= 1
        else:
            if self.state in ("CHASE_SPAWNER", "EDGE_GUARD"):
                self.state = "ENGAGE"

        if self.reset_timer > 0:
            self.reset_timer -= 1
            if self.reset_timer == 0 and self.state == "RESET":
                self.state = "ENGAGE"

        if self.evasive_timer > 0:
            self.evasive_timer -= 1
            if self.evasive_timer == 0 and self.state == "EVADE":
                self.state = "ENGAGE"

        # Spacing from weapon
        dcfg = self.WEAPON_DIST.get(weapon, self.WEAPON_DIST[None])
        atk2 = dcfg["attack"] ** 2
        close2 = dcfg["close"] ** 2

        # --- State policies ---
        if self.state == "EVADE":
            move = -1 if dx > 0 else 1
            jump = 1 if (abs(dy) > 0.5 or (p_frame_dmg and p_frame_dmg > self.DANGER_FRAME_DAMAGE/2)) else 0
            attack = 1 if dist2 < (self.DEFAULT_CLOSE_DIST**2) and dy <= 0.4 else 0
            return [move, jump, attack]

        if self.state == "RESET":
            move = -1 if (pred_ox - px) > 0 else 1
            jump = 1 if (oy > py + 0.8) else 0
            attack = 0
            return [move, jump, attack]

        if self.state == "EDGE_GUARD":
            move = 0 if abs(dx) < 0.6 else (1 if dx > 0 else -1)
            jump = 0 if dy < -0.5 else 1
            attack = 1 if dist2 <= atk2 * 1.3 else 0
            return [move, jump, attack]

        if self.state == "CHASE_SPAWNER" and weapon is None:
            tx, ty = self._best_spawner_target((px, py), (ox, oy), spawners)
            if tx is not None:
                tdx, tdy = (tx - px), (ty - py)
                move = 1 if tdx > 0 else -1
                jump = 1 if tdy > 0.8 else 0
                attack = 1 if dist2 <= (self.DEFAULT_ATTACK_DIST ** 2) else 0
                return [move, jump, attack]
            # else fallthrough to ENGAGE

        # ENGAGE (default): prediction, aggression, combos
        pdx, pdy = (pred_ox - px), (pred_oy - py)
        pdist2 = pdx*pdx + pdy*pdy

        if self.combo_timer > 0:
            self.combo_timer -= 1
            move = 0 if dist2 <= atk2 else (1 if dx > 0 else -1)
            jump = 1 if dy > 0.6 else 0
            attack = 1
            return [move, jump, attack]

        if pdist2 > atk2:
            move = 1 if pdx > 0 else -1
            if weapon in ("Hammer", "Punch"):
                jump = 1 if (pdy > 0.8 or (0.5 < pdy <= 0.8 and self.frame % 6 == 0)) else 0
            else:
                jump = 1 if pdy > 1.0 else 0
            attack = 1 if pdist2 <= min(close2, atk2 * 1.5) else 0
            if attack == 1:
                self.combo_timer = self.COMBO_CHAIN_FRAMES
            return [move, jump, attack]

        # In range
        if weapon == "Spear":
            move = 0 if dist2 >= (atk2 * 0.85) else (-1 if dx < 0 else 1)
            jump = 1 if dy > 0.7 else 0
            attack = 1
            self.combo_timer = self.COMBO_CHAIN_FRAMES // 2
        elif weapon == "Hammer":
            move = 0
            jump = 1 if dy > 0.5 else 0
            attack = 1
            self.combo_timer = self.COMBO_CHAIN_FRAMES
        elif weapon == "Punch":
            move = 0
            jump = 1 if dy > 0.5 else 0
            attack = 1
            self.combo_timer = self.COMBO_CHAIN_FRAMES
        else:
            move = 0
            jump = 1 if dy > 0.6 else 0
            attack = 1
            self.combo_timer = self.COMBO_CHAIN_FRAMES // 2

        return [move, jump, attack]

    # ---------- Train/Save ----------
    def save(self, file_path: str) -> None:
        if getattr(self, "model", None) is not None:
            self.model.save(file_path)

    def learn(self, env, total_timesteps, log_interval: int = 4):
        if getattr(self, "model", None) is not None:
            self.model.set_env(env)
            self.model.learn(total_timesteps=total_timesteps, log_interval=log_interval)
        else:
            raise RuntimeError("No model present to learn with")

    # ---------- Helpers (no extra imports) ----------
    def _parse_positions_and_damage(self, obs):
        px, py = self.last_player_pos
        ox, oy = self.last_opponent_pos
        p_total_dmg, p_frame_dmg = None, None
        try:
            if isinstance(obs, dict):
                if "player" in obs:
                    p = obs["player"]
                    px = self._safe_get(p, ["body", "position", "x"], px)
                    py = self._safe_get(p, ["body", "position", "y"], py)
                    p_total_dmg = self._safe_get(p, ["damage_taken_total"], p_total_dmg)
                    p_frame_dmg = self._safe_get(p, ["damage_taken_this_frame"], p_frame_dmg)
                if "opponent" in obs:
                    o = obs["opponent"]
                    ox = self._safe_get(o, ["body", "position", "x"], ox)
                    oy = self._safe_get(o, ["body", "position", "y"], oy)
                if "agent" in obs and ("enemy" in obs or "opponent" in obs):
                    a = obs.get("agent", {})
                    e = obs.get("enemy", obs.get("opponent", {}))
                    px = self._safe_get(a, ["body", "position", "x"], px)
                    py = self._safe_get(a, ["body", "position", "y"], py)
                    ox = self._safe_get(e, ["body", "position", "x"], ox)
                    oy = self._safe_get(e, ["body", "position", "y"], oy)

                if "objects" in obs and isinstance(obs["objects"], dict):
                    for k, v in obs["objects"].items():
                        lk = str(k).lower()
                        if "player" in lk or "agent" in lk:
                            px = self._safe_get(v, ["body", "position", "x"], px)
                            py = self._safe_get(v, ["body", "position", "y"], py)
                            p_total_dmg = self._safe_get(v, ["damage_taken_total"], p_total_dmg)
                            p_frame_dmg = self._safe_get(v, ["damage_taken_this_frame"], p_frame_dmg)
                        if "opp" in lk or "enemy" in lk:
                            ox = self._safe_get(v, ["body", "position", "x"], ox)
                            oy = self._safe_get(v, ["body", "position", "y"], oy)
            elif isinstance(obs, (list, tuple)) and len(obs) >= 4:
                try:
                    px, py, ox, oy = float(obs[0]), float(obs[1]), float(obs[2]), float(obs[3])
                except Exception:
                    pass
        except Exception:
            pass
        self.last_player_frame_dmg = p_frame_dmg or 0.0
        return px, py, ox, oy, p_total_dmg, p_frame_dmg

    def _parse_weapon(self, obs):
        weapon = self.last_player_weapon
        try:
            if isinstance(obs, dict):
                if "player" in obs:
                    weapon = self._safe_get(obs["player"], ["weapon"], weapon)
                elif "agent" in obs:
                    weapon = self._safe_get(obs["agent"], ["weapon"], weapon)
                if "objects" in obs and isinstance(obs["objects"], dict):
                    for k, v in obs["objects"].items():
                        if "player" in str(k).lower() or "agent" in str(k).lower():
                            w = self._safe_get(v, ["weapon"], None)
                            if w is not None:
                                weapon = w
                                break
        except Exception:
            pass
        return weapon

    def _parse_spawners(self, obs):
        spawners = []
        try:
            if isinstance(obs, dict):
                cand = obs.get("spawners", None) or obs.get("spawner_info", None) or obs.get("get_spawner_info", None)
                if isinstance(cand, list):
                    for it in cand:
                        if isinstance(it, (list, tuple)) and len(it) >= 2:
                            wtype, wpos = it[0], it[1]
                            if isinstance(wpos, (list, tuple)) and len(wpos) >= 2:
                                spawners.append((str(wtype), (float(wpos[0]), float(wpos[1]))))
                envd = obs.get("env", None)
                if isinstance(envd, dict) and "spawners" in envd and isinstance(envd["spawners"], list):
                    for it in envd["spawners"]:
                        if isinstance(it, (list, tuple)) and len(it) >= 2:
                            wtype, wpos = it[0], it[1]
                            if isinstance(wpos, (list, tuple)) and len(wpos) >= 2:
                                spawners.append((str(wtype), (float(wpos[0]), float(wpos[1]))))
        except Exception:
            pass
        return spawners

    def _best_spawner_target(self, pxy, oxy, spawners):
        if not spawners:
            return (None, None)
        px, py = pxy
        ox, oy = oxy
        pref_rank = {w: i for i, w in enumerate(self.PREFERRED_WEAPONS)}

        best = (None, None)
        best_key = None
        for wtype, (sx, sy) in spawners:
            rank = pref_rank.get(wtype, len(self.PREFERRED_WEAPONS))
            d2_us = (sx - px)**2 + (sy - py)**2
            d2_opp = (sx - ox)**2 + (sy - oy)**2
            key = (rank, d2_us, -d2_opp)  # prefer better weapon, closer to us, farther from them
            if best_key is None or key < best_key:
                best_key = key
                best = (sx, sy)
        return best

    def _velocity(self, last, cur):
        try:
            return (cur[0] - last[0], cur[1] - last[1])
        except Exception:
            return (0.0, 0.0)

    def _safe_get(self, d, path, default=None):
        cur = d
        for p in path:
            if cur is None:
                return default
            try:
                if isinstance(cur, dict) and p in cur:
                    cur = cur[p]
                    continue
            except Exception:
                pass
            try:
                cur = getattr(cur, p)
                continue
            except Exception:
                try:
                    idx = int(p)
                    if isinstance(cur, (list, tuple)) and 0 <= idx < len(cur):
                        cur = cur[idx]
                        continue
                except Exception:
                    return default
        return cur if cur is not None else default
