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
    '''
    Ultimate Weapon Master - Dominates through weapon control and combat mastery
    '''
    def __init__(
        self,
        file_path: Optional[str] = None,
    ):
        super().__init__(file_path)
        # Combat tracking
        self.time = 0
        self.last_attack_time = 0
        self.last_grab_attempt = 0
        self.combo_count = 0
        
        # Weapon priorities and tracking
        self.weapon_priority = {'Hammer': 3, 'Spear': 2, 'Punch': 0}
        self.current_weapon = 'Punch'
        self.seeking_weapon = False
        
        # Position tracking
        self.last_opponent_pos = None
        self.opponent_velocity = [0, 0]

    def _initialize(self) -> None:
        if self.file_path is None:
            self.model = PPO("MlpPolicy", self.env, verbose=0)
            del self.env
        else:
            self.model = PPO.load(self.file_path)

    def _gdown(self) -> str:
        data_path = "rl-model.zip"
        if not os.path.isfile(data_path):
            print(f"Downloading {data_path}...")
            url = "https://drive.google.com/file/d/1JIokiBOrOClh8piclbMlpEEs6mj3H1HJ/view?usp=sharing"
            gdown.download(url, output=data_path, fuzzy=True)
        return data_path

    def calculate_distance(self, pos1, pos2):
        """Calculate distance between two positions"""
        dx = pos1[0] - pos2[0]
        dy = pos1[1] - pos2[1]
        return (dx**2 + dy**2)**0.5

    def predict_opponent_position(self, opp_pos):
        """Predict where opponent will be"""
        if self.last_opponent_pos is not None:
            self.opponent_velocity[0] = opp_pos[0] - self.last_opponent_pos[0]
            self.opponent_velocity[1] = opp_pos[1] - self.last_opponent_pos[1]
            predicted_x = opp_pos[0] + self.opponent_velocity[0] * 5
            predicted_y = opp_pos[1] + self.opponent_velocity[1] * 5
            self.last_opponent_pos = opp_pos
            return [predicted_x, predicted_y]
        self.last_opponent_pos = opp_pos
        return opp_pos

    def is_vulnerable(self, state):
        """Check if opponent is vulnerable - KO, stunned, attacking, or knocked back"""
        return state in [3, 5, 8, 11]
    
    def is_opponent_falling(self, opp_pos, opp_state):
        """Check if opponent is falling off map - DON'T FOLLOW THEM"""
        # Opponent is KO'd (state 5) or very high up (falling)
        if opp_state == 5 or opp_pos[1] > 4.0:
            return True
        # Opponent is near edge and high up
        if abs(opp_pos[0]) > 4.5 and opp_pos[1] > 2.5:
            return True
        return False
    
    def is_safe_position(self, pos):
        """Check if our position is safe (not near edges or high up)"""
        edge_safe = 10.67 / 2 - 1.5  # Safe zone
        if abs(pos[0]) > edge_safe:
            return False
        if pos[1] > 3.5:
            return False
        return True

    def should_grab_weapon(self, distance, opp_state):
        """Smart weapon grabbing logic"""
        # Don't grab if on cooldown
        if self.time - self.last_grab_attempt < 60:
            return False
        
        # Grab if far away and safe
        if distance > 4.5:
            return True
        
        # Grab if opponent is vulnerable and we're close to weapon spawn
        if self.is_vulnerable(opp_state) and distance > 2.5:
            return True
        
        # Grab if we have no weapon and opponent not attacking
        if self.current_weapon == 'Punch' and opp_state != 3 and distance > 3.0:
            return True
        
        return False

    def get_weapon_attack(self, weapon, distance, opp_state):
        """Weapon-specific attack patterns"""
        if weapon == 'Hammer':
            # Hammer: Powerful, slow - use at mid range
            if distance < 2.8:
                if self.is_vulnerable(opp_state):
                    return ['l']  # Heavy smash on vulnerable
                else:
                    return ['k']  # Medium swing
            return ['j']  # Light attack
        
        elif weapon == 'Spear':
            # Spear: Long range, fast - poke from distance
            if distance < 3.5:
                if self.is_vulnerable(opp_state):
                    return ['k', 'j']  # Quick combo
                else:
                    return ['j']  # Quick poke
            return ['k']  # Medium range thrust
        
        else:  # Punch
            # No weapon: Fast combos at close range
            if distance < 1.8:
                if self.combo_count % 8 < 4:
                    return ['j']  # Jab
                else:
                    return ['k']  # Hook
            return ['j']

    def predict(self, obs):
        self.time += 1
        
        # Extract game state
        pos = self.obs_helper.get_section(obs, 'player_pos')
        opp_pos = self.obs_helper.get_section(obs, 'opponent_pos')
        player_state = self.obs_helper.get_section(obs, 'player_state')
        opp_state = self.obs_helper.get_section(obs, 'opponent_state')
        
        # Get ML prediction
        ml_action, _ = self.model.predict(obs, deterministic=True)
        action = self.act_helper.zeros()
        
        # Calculate metrics
        distance = self.calculate_distance(pos, opp_pos)
        predicted_opp_pos = self.predict_opponent_position(opp_pos)
        
        # === PRIORITY 0: EMERGENCY EDGE PROTECTION (STRONGEST) ===
        # Multiple safety zones for bulletproof protection
        edge_critical = 10.67 / 2 - 0.6   # CRITICAL - immediate danger
        edge_danger = 10.67 / 2 - 1.2     # HIGH danger
        edge_caution = 10.67 / 2 - 2.0    # Medium caution
        
        # CRITICAL EDGE - Emergency recovery (overrides everything)
        if pos[0] > edge_critical:
            action = self.act_helper.press_keys(['a', 'space'])
            return action
        elif pos[0] < -edge_critical:
            action = self.act_helper.press_keys(['d', 'space'])
            return action
        
        # HIGH DANGER EDGE - Strong correction
        if pos[0] > edge_danger:
            action = self.act_helper.press_keys(['a'])
            if pos[1] > 2.0:
                action = self.act_helper.press_keys(['a', 'space'], action)
            return action
        elif pos[0] < -edge_danger:
            action = self.act_helper.press_keys(['d'])
            if pos[1] > 2.0:
                action = self.act_helper.press_keys(['d', 'space'], action)
            return action
        
        # Height danger - don't get too high
        if pos[1] > 4.2:
            action = self.act_helper.press_keys(['s'])
            return action
        elif pos[1] > 3.5:
            # Start moving down and toward center
            action = self.act_helper.press_keys(['s'])
            if abs(pos[0]) > 2.0:
                if pos[0] > 0:
                    action = self.act_helper.press_keys(['a'], action)
                else:
                    action = self.act_helper.press_keys(['d'], action)
            return action
        
        # === PRIORITY 1: DON'T FOLLOW FALLING OPPONENTS ===
        if self.is_opponent_falling(opp_pos, opp_state):
            # Opponent is falling - STAY SAFE, move to center
            action = self.act_helper.zeros()
            
            # Move toward center of map
            if pos[0] > 1.5:
                action = self.act_helper.press_keys(['a'])
            elif pos[0] < -1.5:
                action = self.act_helper.press_keys(['d'])
            
            # Get to ground level if high up
            if pos[1] > 2.0:
                action = self.act_helper.press_keys(['s'], action)
            
            return action
        
        # === PRIORITY 2: WEAPON CONTROL (only when safe) ===
        if self.should_grab_weapon(distance, opp_state) and self.is_safe_position(pos):
            self.last_grab_attempt = self.time
            action = self.act_helper.press_keys(['g'])
            
            # Move toward center/weapon spawns while grabbing
            if abs(pos[0]) > 2.0:
                if pos[0] > 0:
                    action = self.act_helper.press_keys(['a'], action)
                else:
                    action = self.act_helper.press_keys(['d'], action)
            
            return action
        
        # === PRIORITY 3: DODGE INCOMING ATTACKS (edge-aware) ===
        if opp_state == 3 and distance < 2.2:
            # Dodge away from attack but CHECK FOR EDGES
            if pos[0] < opp_pos[0]:
                # Opponent on right, we're on left
                if pos[0] > -edge_caution:
                    # Safe to dodge left
                    action = self.act_helper.press_keys(['a', 'space'])
                else:
                    # Near left edge, dodge right or up
                    action = self.act_helper.press_keys(['space'])
            else:
                # Opponent on left, we're on right
                if pos[0] < edge_caution:
                    # Safe to dodge right
                    action = self.act_helper.press_keys(['d', 'space'])
                else:
                    # Near right edge, dodge left or up
                    action = self.act_helper.press_keys(['space'])
            return action
        
        # === PRIORITY 4: AGGRESSIVE PUNISH ON VULNERABLE (but stay safe) ===
        if self.is_vulnerable(opp_state) and distance < 3.5:
            # Only chase if we're in a safe position
            if not self.is_safe_position(pos):
                # We're not safe, retreat to center first
                if pos[0] > 1.0:
                    action = self.act_helper.press_keys(['a'])
                elif pos[0] < -1.0:
                    action = self.act_helper.press_keys(['d'])
                return action
            
            # Safe to attack - close gap if needed
            if distance > 2.0:
                # Move toward opponent BUT check edges
                if predicted_opp_pos[0] > pos[0] and pos[0] < edge_caution:
                    action = self.act_helper.press_keys(['d'])
                elif predicted_opp_pos[0] < pos[0] and pos[0] > -edge_caution:
                    action = self.act_helper.press_keys(['a'])
                
                if predicted_opp_pos[1] < pos[1] - 0.5 and pos[1] < 3.0:
                    action = self.act_helper.press_keys(['space'], action)
            
            # Execute weapon-specific attack with DIRECTION
            attack_keys = self.get_weapon_attack(self.current_weapon, distance, opp_state)
            if attack_keys and (self.time - self.last_attack_time) > 6:
                # Face opponent before attacking
                if opp_pos[0] > pos[0]:
                    action = self.act_helper.press_keys(['d'] + attack_keys, action)
                else:
                    action = self.act_helper.press_keys(['a'] + attack_keys, action)
                self.last_attack_time = self.time
                self.combo_count += 1
            
            return action
        
        # === PRIORITY 5: POSITIONING & SPACING (edge-aware) ===
        # Weapon-based optimal distance
        if self.current_weapon == 'Hammer':
            optimal_distance = 2.5
        elif self.current_weapon == 'Spear':
            optimal_distance = 3.0
        else:
            optimal_distance = 2.0
        
        if distance > optimal_distance + 1.5:
            # Approach BUT respect edges
            if predicted_opp_pos[0] > pos[0] and pos[0] < edge_caution:
                action = self.act_helper.press_keys(['d'])
            elif predicted_opp_pos[0] < pos[0] and pos[0] > -edge_caution:
                action = self.act_helper.press_keys(['a'])
            
            # Jump approach if opponent higher (but not too high)
            if predicted_opp_pos[1] < pos[1] - 0.8 and self.time % 5 == 0 and pos[1] < 3.0:
                action = self.act_helper.press_keys(['space'], action)
        
        elif distance < optimal_distance - 0.5 and opp_state == 3:
            # Create space if opponent attacking (but check edges)
            if opp_pos[0] > pos[0] and pos[0] > -edge_caution:
                action = self.act_helper.press_keys(['a'])
            elif opp_pos[0] < pos[0] and pos[0] < edge_caution:
                action = self.act_helper.press_keys(['d'])
        
        else:
            # Optimal range - attack with DIRECTION!
            if (self.time - self.last_attack_time) > 10:
                attack_keys = self.get_weapon_attack(self.current_weapon, distance, opp_state)
                if attack_keys:
                    # ALWAYS face opponent when attacking
                    if opp_pos[0] > pos[0]:
                        action = self.act_helper.press_keys(['d'] + attack_keys)
                    else:
                        action = self.act_helper.press_keys(['a'] + attack_keys)
                    
                    self.last_attack_time = self.time
                    self.combo_count += 1
                    return action
        
        # === PRIORITY 6: AERIAL ATTACKS (with direction) ===
        if pos[1] < opp_pos[1] - 1.0 and distance < 2.5:
            attack_keys = self.get_weapon_attack(self.current_weapon, distance, opp_state)
            if attack_keys:
                # Face opponent in air
                if opp_pos[0] > pos[0]:
                    action = self.act_helper.press_keys(['d'] + attack_keys, action)
                else:
                    action = self.act_helper.press_keys(['a'] + attack_keys, action)
        
        # Anti-air
        if opp_pos[1] < pos[1] - 1.2 and distance < 2.8:
            action = self.act_helper.press_keys(['k'], action)
        
        # === PRIORITY 7: WEAPON DENIAL (safe only) ===
        if self.time % 100 == 0 and distance > 2.5 and distance < 4.5 and self.is_safe_position(pos):
            action = self.act_helper.press_keys(['g'], action)
            self.last_grab_attempt = self.time
        
        # === FALLBACK: Use ML if no strategic action ===
        if sum(action) == 0:
            action = ml_action
        
        return action

    def save(self, file_path: str) -> None:
        self.model.save(file_path)

    def learn(self, env, total_timesteps, log_interval: int = 4):
        self.model.set_env(env)
        self.model.learn(total_timesteps=total_timesteps, log_interval=log_interval)