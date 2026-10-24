import os
import cv2
import gymnasium as gym
import numpy as np
import tensorflow as tf

def evaluate_agent(algorithm="q_learning", num_episodes=5):
    """
    Evaluator snippet: Loads saved models and runs pure exploitation.
    - Set algorithm to either 'q_learning' or 'sarsa'.
    """
    # model_path = f"models/{algorithm}_cartpole.keras"
    model_path = f"dqn_lunarLander0.keras"
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Train it first!")
        return

    print(f"\n--- Evaluating Model: {algorithm.upper()} ---")
    
    model = tf.keras.models.load_model(model_path)
    # model = tf.keras.models.load_model("/dqn_qnet2.keras")
    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        state = np.reshape(state, [1, env.observation_space.shape[0]])
        done = False
        episode_reward = 0
        
        while not done:
            frame = env.render()
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow(f"Evaluating {algorithm.upper()}", frame_bgr)
            cv2.waitKey(20) # Controls simulation speed
            
            # Pure exploitation: Pick the highest scoring action
            q_values = model.predict(state, verbose=0)
            action = np.argmax(q_values[0])
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            state = np.reshape(next_state, [1, env.observation_space.shape[0]])
            done = terminated or truncated
            episode_reward += 1
            
        print(f"Evaluation Episode {episode + 1} | Score: {episode_reward}")
        
    env.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Choose which model to test: "q_learning" or "sarsa"
    evaluate_agent(algorithm="q_learning", num_episodes=5)
    # evaluate_agent(algorithm="sarsa", num_episodes=3)