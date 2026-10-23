import cv2
import gymnasium as gym
import tensorflow as tf

# Use Gymnasium instead of Gym
env = gym.make("CartPole-v1", render_mode="rgb_array")

for episode in range(5):
  done = False
  # Gymnasium's reset returns (state, info)
  state, info = env.reset()

  while not done:
    frame = env.render()  # render_mode is defined in gym.make() now
    
    # Gymnasium returns RGB frames; OpenCV expects BGR for correct colors
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    cv2.imshow("CartPole", frame_bgr)
    cv2.waitKey(100)

    # Valid actions for CartPole-v1 are 0 (push left) and 1 (push right)
    action = tf.random.uniform(
        shape=(), minval=0, maxval=2, dtype=tf.int32
    ).numpy()
    # action=0
    
    # Gymnasium step returns: next_state, reward, terminated, truncated, info
    state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated

env.close()
cv2.destroyAllWindows()