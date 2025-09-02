import gym
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers
import cv2
import collections
import imageio
#Init GPU
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Training Prameters
GAMMA = 0.99
LEARNING_RATE = 0.0003
BATCH_SIZE = 50
MEMORY_SIZE = 300000
EPSILON_START = 0.5
EPSILON_END = 0.01
EPSILON_DECAY = 0.99925
EPISODES = 1000
STEPS = 1000
#Save pramters
TARGET_UPDATE = STEPS-1
SAVE_ITERVAL = 10

# Resize image for faster run
def preprocess_state(state):
    state = cv2.resize(state, (50, 50))  
    state = state / 255.0  
    return np.expand_dims(state, axis=(0, -1)) 

# Nuaral Netrwork
def build_q_network(action_size):
    model = models.Sequential([
        layers.InputLayer(input_shape=(50, 50, 3)),
        layers.Conv2D(32, (8, 8), strides=4, activation="relu"),
        layers.Conv2D(64, (4, 4), strides=2, activation="relu"),
        layers.Conv2D(128, (3, 3), strides=1, activation="relu"),
        layers.Flatten(),
        layers.Dense(2052, activation="relu"),
        layers.Dense(512, activation="relu"),
        layers.Dense(action_size, )
    ])
    model.compile(optimizer=optimizers.Adam(learning_rate=LEARNING_RATE), loss="mse")
    return model

#Expirance replay buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

# Action Policy
def select_action(state, q_network, epsilon, action_size):
    if np.random.rand() < epsilon:
        return random.randint(0, action_size - 1)  
    else:
        q_values = q_network.predict(state, verbose=0)
        return np.argmax(q_values)

# Training function
def train_dqn():

    action_size = 5  # (accelerate, brake, left, right, nothing)
    #Init enviorment
    env = gym.make("CarRacing-v2", domain_randomize=False,continuous=False,render_mode="rgb_array")
    #Create Networks
    q_network = build_q_network(action_size)
    target_network = models.clone_model(q_network)
    #Init Buffer
    replay_buffer = ReplayBuffer(MEMORY_SIZE)

    epsilon = EPSILON_START
    total_rewards = []
    #Training
    for episode in range(170,EPISODES):
        state = preprocess_state(env.reset()[0])
        episode_reward = 0
        images =[]
        for step in range(STEPS):
            if episode%SAVE_ITERVAL == 0:
               images.append(env.render())
    
            action = select_action(state, q_network, epsilon, action_size)
            step_result = env.step(action,)  
            next_state = step_result[0]  
            reward = step_result[1]  
            done = step_result[2]  
        
            next_state = preprocess_state(next_state)
            replay_buffer.add((state, action, reward, next_state, done))

            state = next_state
            episode_reward += reward
        
            if len(replay_buffer) > BATCH_SIZE:
                #Get Random Sample from buffer
                batch = replay_buffer.sample(BATCH_SIZE) 
                states, actions, rewards, next_states = zip(*batch) 

                states = np.vstack(states)
                next_states = np.vstack(next_states)
                actions = np.array(actions)
                rewards = np.array(rewards)
                #Create Training Data
                next_q_values = target_network.predict(next_states, verbose=0)
                target_q_values = rewards + GAMMA * np.max(next_q_values, axis=1) 
                q_values = q_network.predict(states, verbose=0)
                for i, action in enumerate(actions):
                    q_values[i][action] = target_q_values[i]
                #Training Network
                q_network.fit(states, q_values, epochs=1, verbose=0, batch_size=BATCH_SIZE)
            #Update Target network
            if step % TARGET_UPDATE == 0:
                target_network.set_weights(q_network.get_weights())
            #exit failed run
            if episode_reward<-1000:
                break

            total_rewards.append(episode_reward)
            epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
            print(f"Episode {episode + 1}_({step}/{STEPS}): Reward = {episode_reward:.2f}, Epsilon = {epsilon:.4f}")
        if len(images)>0:
            q_network.save('model.keras')
            for img in images:
                cv2.imshow('DQL',img)
                cv2.waitKey(100)
            imageio.mimsave(f'DQL_episode_{episode}_reward_{episode_reward:.2f}.gif', images)
    env.close()
    return q_network
trained_model = train_dqn()
