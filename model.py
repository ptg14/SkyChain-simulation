import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque
import tensorflow as tf
from tqdm import tqdm

# Triển khai thuật toán Deep Deterministic Policy Gradient (DDPG) cho chính sách sharding

class ReplayBuffer:
    def __init__(self, buffer_size=100000):
        self.buffer = deque(maxlen=buffer_size)  # Khởi tạo bộ đệm với kích thước tối đa

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))  # Thêm trải nghiệm vào bộ đệm

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)  # Lấy mẫu ngẫu nhiên từ bộ đệm
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

    def size(self):
        return len(self.buffer)  # Trả về kích thước hiện tại của bộ đệm

# Mạng Actor
class Actor(tf.keras.Model):
    def __init__(self, action_dim, action_bound):
        super(Actor, self).__init__()
        self.action_bound = action_bound  # Giới hạn hành động
        self.fc1 = tf.keras.layers.Dense(256, activation='relu')  # Lớp fully connected 1
        self.fc2 = tf.keras.layers.Dense(256, activation='relu')  # Lớp fully connected 2
        self.fc3 = tf.keras.layers.Dense(action_dim, activation='tanh')  # Lớp fully connected 3

    def call(self, state):
        x = self.fc1(state)  # Truyền qua lớp fully connected 1
        x = self.fc2(x)  # Truyền qua lớp fully connected 2
        action = self.fc3(x)  # Truyền qua lớp fully connected 3
        return action * self.action_bound  # Trả về hành động với giới hạn

# Mạng Critic
class Critic(tf.keras.Model):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc1 = tf.keras.layers.Dense(256, activation='relu')  # Lớp fully connected 1
        self.fc2 = tf.keras.layers.Dense(256, activation='relu')  # Lớp fully connected 2
        self.fc3 = tf.keras.layers.Dense(1, activation=None)  # Lớp fully connected 3

    def call(self, state, action):
        state = tf.cast(state, tf.float32)  # Đảm bảo state là kiểu float32
        x = tf.concat([state, action], axis=-1)  # Kết hợp state và action
        x = self.fc1(x)  # Truyền qua lớp fully connected 1
        x = self.fc2(x)  # Truyền qua lớp fully connected 2
        q_value = self.fc3(x)  # Truyền qua lớp fully connected 3
        return q_value  # Trả về giá trị Q

# Agent DDPG
class DDPGAgent:
    def __init__(self, state_dim, action_dim, action_bound, actor_lr=1e-4, critic_lr=1e-3, gamma=0.99, tau=0.005, buffer_size=100000):
        self.state_dim = state_dim  # Kích thước state
        self.action_dim = action_dim  # Kích thước action
        self.action_bound = action_bound  # Giới hạn action
        self.gamma = gamma  # Hệ số chiết khấu
        self.tau = tau  # Hệ số cập nhật target network

        # Mạng Actor và Critic
        self.actor = Actor(action_dim, action_bound)
        self.critic = Critic()
        self.target_actor = Actor(action_dim, action_bound)
        self.target_critic = Critic()

        # Bộ tối ưu hóa
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr)

        # Bộ đệm Replay
        self.replay_buffer = ReplayBuffer(buffer_size)

        # Sao chép trọng số sang target networks
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())

    def get_action(self, state, noise_std=0.1):
        state = np.expand_dims(state, axis=0)  # Thêm chiều cho state
        action = self.actor(state).numpy()[0]  # Lấy hành động từ mạng Actor
        noise = np.random.normal(0, noise_std, size=self.action_dim)  # Thêm noise
        return np.clip(action + noise, -self.action_bound, self.action_bound)  # Trả về hành động với noise

    def train(self, batch_size=64):
        if self.replay_buffer.size() < batch_size:
            return

        # Lấy mẫu từ bộ đệm
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        # Chuẩn hóa rewards
        rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-6)

        # Cập nhật Critic
        with tf.GradientTape() as tape:
            target_actions = self.target_actor(next_states)
            target_q_values = self.target_critic(next_states, target_actions)
            y = rewards + self.gamma * (1 - dones) * tf.squeeze(target_q_values)
            critic_value = tf.squeeze(self.critic(states, actions))
            critic_loss = tf.reduce_mean(tf.square(y - critic_value))
        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

        # Cập nhật Actor
        with tf.GradientTape() as tape:
            actions_pred = self.actor(states)
            actor_loss = -tf.reduce_mean(self.critic(states, actions_pred))
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

        # Cập nhật Target Networks
        self.update_target(self.target_actor, self.actor)
        self.update_target(self.target_critic, self.critic)

    def update_target(self, target_net, net):
        for target_param, param in zip(target_net.trainable_variables, net.trainable_variables):
            target_param.assign(self.tau * param + (1 - self.tau) * target_param)

    def add_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)

# Thêm hàm Moving Average
def moving_average(data, window_size=10):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# Thay đổi hàm Reward
def compute_reward(action):
    # Phần thưởng được cải thiện: Khuyến khích giá trị hành động nhỏ hơn một ngưỡng
    penalty = np.sum(np.abs(action))  # Phạt hành động lớn
    reward = penalty  # Tổng phần thưởng là âm của penalty
    return reward

# Điều chỉnh vòng lặp Training
def train_drl_sharding_agent():
    state_dim = 2  # [Transaction Pool Size, Number of Nodes]
    action_dim = 3  # [Epoch Length, Number of Shards, Block Size]
    action_bound = np.array([1000, 64, 8])  # Giới hạn trên của actions

    agent = DDPGAgent(state_dim, action_dim, action_bound)

    num_episodes = 1000  # Tăng số lượng episodes
    batch_size = 64
    rewards = []  # Lưu trữ phần thưởng để trực quan hóa
    noise_std = 0.2  # Độ lệch chuẩn của noise ban đầu

    for episode in tqdm(range(num_episodes)):
        state = np.array([100, 1000])  # Trạng thái ban đầu [Transaction Pool, Node Count]
        episode_reward = 0

        for t in range(50):  # Mô phỏng 50 bước thời gian mỗi episode
            action = agent.get_action(state, noise_std)
            next_state = state + np.random.randint(-10, 10, size=state.shape)  # Mô phỏng thay đổi môi trường
            reward = compute_reward(action)  # Sử dụng hàm reward mới
            done = t == 49

            agent.add_experience(state, action, reward, next_state, done)
            agent.train(batch_size)

            state = next_state
            episode_reward += reward

        rewards.append(episode_reward)

        # Giảm noise dần để tập trung vào khai thác
        noise_std = max(0.05, noise_std * 0.99)  # Noise giảm dần theo thời gian

    # Làm mịn phần thưởng bằng Moving Average
    smoothed_rewards = moving_average(rewards, window_size=10)

    # Vẽ đồ thị phần thưởng
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, label='Raw Rewards', alpha=0.5)
    plt.plot(range(len(smoothed_rewards)), smoothed_rewards, label='Smoothed Rewards', color='red')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Rewards Over Episodes')
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    train_drl_sharding_agent()
