import os
import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

def create_model(state_size, action_size):
    model = tf.keras.Sequential([
        layers.Input(shape=(state_size,)),
        layers.Dense(24, activation="relu"),
        layers.Dense(24, activation="relu"),
        layers.Dense(action_size, activation="linear")
    ])
    return model

# Force execution on CPU to eliminate GPU-to-CPU data transfer latency
tf.config.set_visible_devices([], 'GPU')

def train_sarsa():
    env = gym.make("CartPole-v1")
    state_size = int(env.observation_space.shape[0])
    action_size = int(env.action_space.n)
    
    model = create_model(state_size, action_size)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss_fn = tf.keras.losses.MeanSquaredError()
    
    # ⚡ Compile the training step into a highly optimized graph
    @tf.function
    def train_step(state, target):
        with tf.GradientTape() as tape:
            q_values = model(state, training=True)
            loss = loss_fn(target, q_values)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # Hyperparameters
    GAMMA = 0.99
    EPSILON = 1.0
    EPSILON_MIN = 0.01
    EPSILON_DECAY = 0.995
    EPISODES = 300
    
    print("\n--- Starting SARSA Training (Optimized) ---")
    
    for episode in range(EPISODES):
        state, _ = env.reset()
        state = np.reshape(state, [1, state_size]).astype(np.float32)
        
        def get_action(s):
            if np.random.rand() <= EPSILON:
                return env.action_space.sample()
            # Calling the model directly like a function is 100x faster than model.predict()
            return np.argmax(model(s, training=False).numpy()[0])
            
        current_action = get_action(state)
        total_reward = 0
        done = False
        
        while not done:
            next_state, reward, terminated, truncated, _ = env.step(current_action)
            next_state = np.reshape(next_state, [1, state_size]).astype(np.float32)
            done = terminated or truncated
            
            if done and total_reward < 499:
                reward = -100
                
            # Predict target without Keras overhead
            target = model(state, training=False).numpy()
            
            next_action = get_action(next_state)
            next_q = model(next_state, training=False).numpy()[0][next_action]
            target[0][current_action] = reward + GAMMA * next_q * (not done)
            
            # Execute the fast graph training step
            train_step(tf.convert_to_tensor(state), tf.convert_to_tensor(target))
            
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
