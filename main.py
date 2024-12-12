import gym
from gym import spaces
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import seaborn as sns
import time

class TumourTreatmentEnv(gym.Env):
    def __init__(self):
        super(TumourTreatmentEnv, self).__init__()

        self.action_space = spaces.Discrete(11)
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32
        )

        self.initial_tumour_volume = 0.1
        self.tumour_growth_rate = 0.02
        self.drug_effectiveness = 0.05
        self.drug_decay = 0.03
        self.health_recovery_rate = 0.01
        self.toxicity_threshold = 0.2
        self.max_dose_penalty = 0.02
        self.state = None
        self.max_steps = 100
        self.current_step = 0

    def reset(self):
        self.current_step = 0
        tumour_volume = self.initial_tumour_volume
        drug_conc = 0.0
        health_score = 1.0
        self.state = np.array([tumour_volume, drug_conc, health_score], dtype=np.float32)
        return self.state

    def step(self, action):
        tumour_volume, drug_conc, health_score = self.state
        dose = action * 0.1
        drug_conc = min(1.0, drug_conc + dose)
        tumour_volume += self.tumour_growth_rate * tumour_volume - self.drug_effectiveness * drug_conc
        tumour_volume = max(0, min(tumour_volume, 1.0))
        drug_conc *= (1 - self.drug_decay)
        health_score += self.health_recovery_rate - dose * 0.01
        health_score = max(0, min(1.0, health_score))

        reward = -tumour_volume * 5
        reward += health_score * 2
        reward -= dose * self.max_dose_penalty

        if tumour_volume <= 0.01:
            reward += 10
            done = True
        elif health_score <= 0.0 or self.current_step >= self.max_steps:
            reward -= 5
            done = True
        else:
            done = False

        self.current_step += 1
        self.state = np.array([tumour_volume, drug_conc, health_score], dtype=np.float32)

        return self.state, reward, done, {}

    def render(self, mode='human'):
        tumour_volume, drug_conc, health_score = self.state
        print(f"Tumour Volume: {tumour_volume:.3f}, Drug Concentration: {drug_conc:.3f}, Health Score: {health_score:.3f}")

    def close(self):
        pass

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate = 0.001
        self.batch_size = 64
        self.memory_limit = 5000
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = self._build_model().to(self.device)
        self.target_model = self._build_model().to(self.device)
        self.update_target_model()

    def _build_model(self):
        return nn.Sequential(
            nn.Linear(self.state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_size)
        )

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        if len(self.memory) >= self.memory_limit:
            self.memory.pop(0)
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        state_tensor = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return torch.argmax(q_values).item()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch_indices = np.random.choice(len(self.memory), self.batch_size, replace=False)
        batch = [self.memory[i] for i in batch_indices]

        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            target_q_values = rewards + self.gamma * (1 - dones) * torch.max(self.target_model(next_states), dim=1)[0]

        loss = nn.MSELoss()(q_values, target_q_values)
        self.optimiser = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

if __name__ == "__main__":
    env = TumourTreatmentEnv()
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    runs = 5
    episodes = 3500
    steps_per_episode = 100
    all_rewards = []

    total_run_time = 0

    policy_heatmap = np.zeros((11, episodes))

    for run in range(runs):
        print(f"Starting run {run + 1}/{runs}")
        start_time = time.time()

        agent = DQNAgent(state_size, action_size)
        rewards = []

        for episode in range(episodes):
            state = env.reset()
            total_reward = 0

            for step in range(steps_per_episode):
                action = agent.act(state)
                next_state, reward, done, _ = env.step(action)
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                if done:
                    break

            agent.replay()
            agent.update_target_model()
            rewards.append(total_reward)

            policy_heatmap[:, episode] += np.histogram(
                [agent.act(state) for _ in range(100)], bins=np.arange(12))[0]

        all_rewards.append(rewards)
        run_time = time.time() - start_time
        total_run_time += run_time
        print(f"Completed run {run + 1}/{runs} in {run_time:.2f} seconds")

    average_episode_time = total_run_time / (runs * episodes)
    print(f"Average time per episode: {average_episode_time:.4f} seconds")

    all_rewards = np.array(all_rewards)
    mean_rewards = np.mean(all_rewards, axis=0)
    std_rewards = np.std(all_rewards, axis=0)

    plt.figure(figsize=(12, 6))
    episodes_range = range(1, episodes + 1)
    plt.plot(episodes_range, mean_rewards, label='Mean Total Reward', color='blue')
    plt.fill_between(episodes_range, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.3, color='blue', label='Std Dev')
    plt.title("Learning Curve with Mean and Standard Deviation")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.heatmap(policy_heatmap, cmap="YlGnBu", cbar=True, xticklabels=500)
    plt.title("Policy Heatmap: Action Frequency over Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Actions (0-10)")

    num_episodes = policy_heatmap.shape[1]
    ticks = np.arange(0, num_episodes + 1, 500)
    plt.xticks(ticks=ticks, labels=ticks, rotation=45)

    plt.show()

    env.close()
