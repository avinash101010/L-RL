import os
import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

def create_model(state_size, action_size):
    """
    Creates a Neural Network to approximate Q-values.
    - Why: Maps the 4 CartPole state variables to action values without a lookup table.
    """
    model = tf.keras.Sequential([
        layers.Input(shape=(state_size,)),
        layers.Dense(24, activation="relu"),
        layers.Dense(24, activation="relu"),
        layers.Dense(action_size, activation="linear")
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse")
    return model

def train_q_learning():
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
    
    print("\n--- Starting Q-Learning Training ---")
    
    for episode in range(EPISODES):
        state, _ = env.reset()
        state = np.reshape(state, [1, state_size])
        total_reward = 0
        done = False
        
        while not done:
            # Epsilon-greedy action selection
            if np.random.rand() <= EPSILON:
                action = env.action_space.sample()
            else:
                q_values = model.predict(state, verbose=0)
                action = np.argmax(q_values[0])
                
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            done = terminated or truncated
            
            if done and total_reward < 499:
                reward = -100 # Penalty for failing early
                
            # Q-Learning Target: Uses max Q-value of the next state (Off-Policy)
            target = model.predict(state, verbose=0)
            max_next_q = np.max(model.predict(next_state, verbose=0)[0])
            target[0][action] = reward + GAMMA * max_next_q * (not done)
            
            # Train model on single step
            model.fit(state, target, epochs=1, verbose=0)
            
            state = next_state
            total_reward += 1
            
        if EPSILON > EPSILON_MIN:
            EPSILON *= EPSILON_DECAY
            
        print(f"Episode {episode + 1}/{EPISODES} | Total Reward: {total_reward} | Epsilon: {EPSILON:.2f}")
        
    os.makedirs("models", exist_ok=True)
    model.save("models/q_learning_cartpole.keras")
    print("Q-Learning Model saved to models/q_learning_cartpole.keras")
    env.close()

if __name__ == "__main__":
    train_q_learning()