import collections
import gymnasium as gym
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
from keras import Model, Input
from keras.models import clone_model
from keras.layers import Dense
from keras.losses import Huber

env = gym.make("MountainCar-v0")

net_input = Input(shape=(2,))
x = Dense(64, activation='relu')(net_input)
x = Dense(32, activation="relu")(x)
output = Dense(3, activation="linear")(x)
q_net = Model(inputs=net_input, outputs=output)
q_net.compile(optimizer="adam")
loss_fn = Huber()

target_net = clone_model(q_net)

# Parameters
EPSILON = 1.0
EPSILON_DECAY = 1.005
GAMMA = 0.99
NUM_EPISODES = 1500
MAX_TRANSITIONS = 1_00_000
BATCH_SIZE = 64
TARGET_UPDATE_AFTER = 1000
LEARN_AFTER_STEPS = 4

REPLAY_BUFFER = collections.deque(maxlen=MAX_TRANSITIONS)

def insert_transition(transition):
    REPLAY_BUFFER.append(transition)

def sample_transitions(batch_size=BATCH_SIZE):
    # Generate random indices tensor and convert to numpy ONCE to avoid per-item overhead
    random_indices = tf.random.uniform(shape=(batch_size,), minval=0, maxval=len(REPLAY_BUFFER), dtype=tf.int32).numpy()
    
    batch = [REPLAY_BUFFER[i] for i in random_indices]
    
    sampled_current_states = tf.convert_to_tensor([t[0] for t in batch], dtype=tf.float32)
    sampled_actions = tf.convert_to_tensor([t[1] for t in batch], dtype=tf.int32)
    sampled_rewards = tf.convert_to_tensor([t[2] for t in batch], dtype=tf.float32)
    sampled_next_states = tf.convert_to_tensor([t[3] for t in batch], dtype=tf.float32)
    sampled_terminals = tf.convert_to_tensor([t[4] for t in batch], dtype=tf.bool)
    
    return sampled_current_states, sampled_actions, sampled_rewards, sampled_next_states, sampled_terminals

def policy(state, explore=0.0):
    if tf.random.uniform(shape=(), maxval=1.0) <= explore:
        return tf.random.uniform(shape=(), maxval=3, dtype=tf.int32)
    action = tf.argmax(q_net(tf.expand_dims(state, axis=0), training=False)[0], output_type=tf.int32)
    return action

# Collect initial random states safely (fixed the i + 1 bug)
random_states = []
done = False
i = 0
state, _ = env.reset()

while i < 20 and not done:
    random_states.append(state)
    action = tf.random.uniform(shape=(), maxval=3, dtype=tf.int32)
    state, reward, terminated, truncated, _ = env.step(action.numpy())
    done = terminated or truncated
    i += 1  # Fixed increment bug

random_states = tf.convert_to_tensor(random_states, dtype=tf.float32)

def get_q_values(states):
    return tf.reduce_max(q_net(states, training=False), axis=1)

step_counter = 0

@tf.function
def train_step(current_states, actions, rewards, next_states, terminals):
    next_action_values = tf.reduce_max(target_net(next_states, training=False), axis=1)
    targets = tf.where(terminals, rewards, rewards + GAMMA * next_action_values)

    with tf.GradientTape() as tape:
        preds = q_net(current_states, training=True)
        batch_nums = tf.range(0, limit=BATCH_SIZE)
        indices = tf.stack((batch_nums, actions), axis=1)
        current_values = tf.gather_nd(preds, indices)
        loss = loss_fn(targets, current_values)
        
    grads = tape.gradient(loss, q_net.trainable_weights)
    q_net.optimizer.apply_gradients(zip(grads, q_net.trainable_weights))

for episode in range(NUM_EPISODES):
    done = False
    total_reward = 0
    episode_length = 0
    state, _ = env.reset()
    
    while not done:
        action = policy(state, EPSILON)
        next_state, reward, terminated, truncated, _ = env.step(action.numpy())
        done = terminated or truncated
        
        insert_transition((state, action, reward, next_state, done))
        state = next_state
        step_counter += 1
        total_reward += reward
        episode_length += 1
        
        if step_counter % LEARN_AFTER_STEPS == 0:
            current_states, actions, rewards, next_states, terminals = sample_transitions(BATCH_SIZE)

            train_step(current_states, actions, rewards, next_states, terminals)

        if step_counter % TARGET_UPDATE_AFTER == 0:
            target_net.set_weights(q_net.get_weights())

    avg_q = tf.reduce_mean(get_q_values(random_states)).numpy()
    print(f"Episode: {episode} | Length: {episode_length} | Reward: {total_reward} | Epsilon: {EPSILON:.4f} | Avg_Q: {avg_q:.2f}")
    
    EPSILON /= EPSILON_DECAY

env.close()
q_net.save("dqn_mountainCar0.keras")