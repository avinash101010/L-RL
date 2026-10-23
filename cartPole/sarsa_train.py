import os
import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

def create_model(state_size, action_size):
    """
    Creates a Neural Network to approximate Q-values for SARSA.
    """
    model = tf.keras.Sequential([
        layers.Input(shape=(state_size,)),
        layers.Dense(24, activation="relu"),
        layers.Dense(24, activation="relu"),
        layers.Dense(action_size, activation="linear")
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse")
    return model

def train_sarsa():
    env = gym.make("CartPole-v1")
    state_size = int(env.observation_space.shape[0])
    action_size = int(env.action_space.n)
    
    model = create_model(state_size, action_size)
    
    # Hyperparameters
    GAMMA = 0.99
    EPSILON = 1.0
    EPSILON_MIN = 0.01
    EPSILON_DECAY = 0.995
    EPISODES = 300
    
    print("\n--- Starting SARSA Training ---")
    
    for episode in range(EPISODES):
        state, _ = env.reset()
        state = np.reshape(state, [1, state_size])
        
        def get_action(s):
            if np.random.rand() <= EPSILON:
                return env.action_space.sample()
            return np.argmax(model.predict(s, verbose=0)[0])
            
        current_action = get_action(state)
        total_reward = 0
        done = False
        
        while not done:
            next_state, reward, terminated, truncated, _ = env.step(current_action)
            next_state = np.reshape(next_state, [1, state_size])
            done = terminated or truncated
            
            if done and total_reward < 499:
                reward = -100
                
            target = model.predict(state, verbose=0)
            
            # SARSA Target: Uses the exact next action chosen by policy (On-Policy)
            next_action = get_action(next_state)
            next_q = model.predict(next_state, verbose=0)[0][next_action]
            target[0][current_action] = reward + GAMMA * next_q * (not done)
            
            model.fit(state, target, epochs=1, verbose=0)
            
            state = next_state
            current_action = next_action
            total_reward += 1
            
        if EPSILON > EPSILON_MIN:
            EPSILON *= EPSILON_DECAY
            
        print(f"Episode {episode + 1}/{EPISODES} | Total Reward: {total_reward} | Epsilon: {EPSILON:.2f}")
        
    os.makedirs("models", exist_ok=True)
    model.save("models/sarsa_cartpole.keras")
    print("SARSA Model saved to models/sarsa_cartpole.keras")
    env.close()

if __name__ == "__main__":
    train_sarsa()