# Step 1: Import required modules
import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Step 2: Set up SUMO path
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

# Step 3: Import TraCI
import traci

# Step 4: SUMO Configuration
Sumo_config = [
    'sumo-gui',
    '-c', 'RL.sumocfg',
    '--step-length', '0.10',
    '--delay', '1000',
    '--lateral-resolution', '0'
]

# -------------------------
# Step 5: Define Core Parameters
# -------------------------
STATE_SIZE = 7
ACTION_SIZE = 2
MIN_GREEN_STEPS = 100
LAST_SWITCH_STEP = -MIN_GREEN_STEPS
TOTAL_TRAINING_STEPS = 10000
EPISODE_MAX_STEPS = 200

# PPO Hyperparameters
BATCH_SIZE = 1024
GAMMA = 0.99
GAE_LAMBDA = 0.95
PPO_EPSILON = 0.2
LEARNING_RATE = 3e-4
EPOCHS = 4
ENTROPY_COEF = 0.01
CRITIC_COEF = 0.5

# Model saving/loading
SAVE_MODEL = True
LOAD_MODEL = False
MODEL_PATH = "traffic_ppo.weights.h5"

# -------------------------
# Step 6: Define PPO Model
# -------------------------
class PPOModel(tf.keras.Model):
    def __init__(self, state_size, action_size):
        super(PPOModel, self).__init__()
        self.shared = keras.Sequential([
            layers.Dense(64, activation='relu', dtype=tf.float32),
            layers.Dense(64, activation='relu', dtype=tf.float32)
        ])
        self.actor = layers.Dense(action_size, activation='softmax', dtype=tf.float32)
        self.critic = layers.Dense(1, dtype=tf.float32)

    def call(self, inputs):
        x = self.shared(inputs)
        policy = self.actor(x)
        value = self.critic(x)
        return policy, value

model = PPOModel(STATE_SIZE, ACTION_SIZE)
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

# Load pre-trained weights if available
if LOAD_MODEL and os.path.exists(MODEL_PATH):
    model.load_weights(MODEL_PATH)
    print(f"\nLoaded pre-trained model from {MODEL_PATH}")

# -------------------------
# Step 7: Experience Buffer
# -------------------------
class PPOBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.log_probs = []
    
    def store(self, state, action, reward, value, done, log_prob):
        self.states.append(state.astype(np.float32))
        self.actions.append(action)
        self.rewards.append(float(reward))
        self.values.append(float(value))
        self.dones.append(done)
        self.log_probs.append(float(log_prob))
    
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()
        self.log_probs.clear()

buffer = PPOBuffer()

# -------------------------
# Step 8: Environment Functions
# -------------------------
def get_state():
    detectors_EB = ["Node1_2_EB_0", "Node1_2_EB_1", "Node1_2_EB_2"]
    detectors_SB = ["Node2_7_SB_0", "Node2_7_SB_1", "Node2_7_SB_2"]
    
    queues_EB = [traci.lanearea.getLastStepVehicleNumber(d) for d in detectors_EB]
    queues_SB = [traci.lanearea.getLastStepVehicleNumber(d) for d in detectors_SB]
    
    current_phase = traci.trafficlight.getPhase("Node2")
    
    return np.array([*queues_EB, *queues_SB, current_phase], dtype=np.float32)

def get_reward(state):
    return -np.sum(state[:-1]).astype(np.float32)

def get_average_speed():
    vehicle_ids = traci.vehicle.getIDList()
    if not vehicle_ids:
        return 0.0
    speeds = [traci.vehicle.getSpeed(v_id) for v_id in vehicle_ids]
    return np.mean(speeds)

def apply_action(action, current_step):
    global LAST_SWITCH_STEP
    
    if action == 1:  # Switch phase action
        if current_step - LAST_SWITCH_STEP >= MIN_GREEN_STEPS:
            program = traci.trafficlight.getAllProgramLogics("Node2")[0]
            num_phases = len(program.phases)
            next_phase = (traci.trafficlight.getPhase("Node2") + 1) % num_phases
            traci.trafficlight.setPhase("Node2", next_phase)
            LAST_SWITCH_STEP = current_step

# -------------------------
# Step 9: PPO Core Functions
# -------------------------
def select_action(state):
    state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
    policy, value = model(state_tensor)
    action_probs = tf.math.log(policy)
    action = tf.random.categorical(action_probs, 1)[0,0].numpy()
    log_prob = action_probs[0, action]
    return action, log_prob, value[0,0].numpy()

def calculate_advantages(rewards, values, dones):
    advantages = []
    last_advantage = 0.0
    next_value = 0.0
    
    for t in reversed(range(len(rewards))):
        if dones[t]:
            delta = rewards[t] - values[t]
            next_value = 0.0
        else:
            delta = rewards[t] + GAMMA * next_value - values[t]
            next_value = values[t]
        advantages.insert(0, delta + GAMMA * GAE_LAMBDA * last_advantage)
        last_advantage = advantages[0]
    
    advantages = np.array(advantages, dtype=np.float32)
    return (advantages - advantages.mean()) / (advantages.std() + 1e-8)

def update_model():
    states = np.array(buffer.states, dtype=np.float32)
    actions = np.array(buffer.actions, dtype=np.int32)
    old_log_probs = np.array(buffer.log_probs, dtype=np.float32)
    returns = calculate_advantages(buffer.rewards, buffer.values, buffer.dones)
    
    dataset = tf.data.Dataset.from_tensor_slices(
        (states, actions, old_log_probs, returns.astype(np.float32)))

    dataset = dataset.shuffle(len(states), reshuffle_each_iteration=True)
    dataset = dataset.batch(64, drop_remainder=True)
    dataset = dataset.repeat(EPOCHS)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    for batch in dataset:
        s_batch, a_batch, old_logp_batch, ret_batch = batch
        
        with tf.GradientTape() as tape:
            # Forward pass
            policy, value = model(s_batch)
            
            # Policy loss
            new_logp = tf.math.log(tf.reduce_sum(policy * tf.one_hot(a_batch, ACTION_SIZE), axis=1))
            ratio = tf.exp(new_logp - old_logp_batch)
            clipped_ratio = tf.clip_by_value(ratio, 1-PPO_EPSILON, 1+PPO_EPSILON)
            policy_loss = -tf.reduce_mean(tf.minimum(ratio * ret_batch, clipped_ratio * ret_batch))
            
            # Value loss
            value_loss = tf.reduce_mean((value[:,0] - ret_batch)**2)
            
            # Entropy bonus
            entropy = -tf.reduce_mean(policy * tf.math.log(policy + 1e-10))
            
            # Total loss
            total_loss = policy_loss + CRITIC_COEF * value_loss - ENTROPY_COEF * entropy
            
        # Apply gradients
        grads = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

# -------------------------
# Step 10: Training Loop
# -------------------------
step_history = []
reward_history = []
queue_history = []
epoch_speeds = []

# Initialize SUMO connection
traci.start(Sumo_config)

try:
    current_step = 0
    cumulative_reward = 0
    current_batch_speeds = []

    print("\n=== Starting PPO Training ===")

    while current_step < TOTAL_TRAINING_STEPS:
        state = get_state()
        done = False
        episode_reward = 0
        episode_steps = 0
        
        while not done and current_step < TOTAL_TRAINING_STEPS:
            action, log_prob, value = select_action(state)
            apply_action(action, current_step)
            traci.simulationStep()
            
            # Collect metrics
            new_state = get_state()
            reward = get_reward(new_state)
            avg_speed = get_average_speed()
            done = (episode_steps >= EPISODE_MAX_STEPS)
            
            # Store experience and metrics
            buffer.store(state, action, reward, value, done, log_prob)
            current_batch_speeds.append(avg_speed)
            
            # Update tracking variables
            state = new_state
            cumulative_reward += reward
            episode_reward += reward
            current_step += 1
            episode_steps += 1
            
            # Perform PPO update when buffer is full
            if len(buffer.states) >= BATCH_SIZE:
                # Calculate and store epoch metrics
                epoch_avg_speed = np.mean(current_batch_speeds)
                epoch_speeds.append(epoch_avg_speed)
                current_batch_speeds = []
                
                update_model()
                buffer.clear()
        
        # Record episode metrics
        step_history.append(current_step)
        reward_history.append(cumulative_reward)
        queue_history.append(np.sum(state[:-1]))
        
        print(f"Step {current_step}/{TOTAL_TRAINING_STEPS} | "
            f"Episode Reward: {episode_reward:.1f} | "
            f"Total Queues: {np.sum(state[:-1]):.1f} | "
            f"Current Speed: {avg_speed:.2f} m/s")

finally:
    if SAVE_MODEL:
        try:
            model.save_weights(MODEL_PATH)
            print(f"\nSaved trained model to {MODEL_PATH}")
        except Exception as e:
            print(f"\nError saving model: {str(e)}")
    
    # Process remaining steps
    if len(current_batch_speeds) > 0:
        epoch_avg_speed = np.mean(current_batch_speeds)
        epoch_speeds.append(epoch_avg_speed)
    
    traci.close()

# -------------------------
# Step 11: Visualization
# -------------------------
plt.figure(figsize=(12, 12))

# Cumulative Reward Plot
plt.subplot(3, 1, 1)
plt.plot(step_history, reward_history, label='Cumulative Reward')
plt.xlabel('Training Steps')
plt.ylabel('Reward')
plt.title('PPO Training Performance')
plt.legend()

# Queue Length Plot
plt.subplot(3, 1, 2)
plt.plot(step_history, queue_history, color='orange', label='Total Queue Length')
plt.xlabel('Training Steps')
plt.ylabel('Vehicles in Queue')
plt.legend()

# Average Speed Plot
plt.subplot(3, 1, 3)
plt.plot(range(len(epoch_speeds)), epoch_speeds, color='green', label='Average Speed per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Speed (m/s)')
plt.ylim(0, 15)
plt.legend()

plt.tight_layout()
plt.show()

