import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import inf
from torch.utils.tensorboard import SummaryWriter

from replay_buffer import ReplayBuffer
from velodyne_env import GazeboEnv

# ==========================================
# 工具函数：数据归一化 (新增重点)
# ==========================================
def process_state(state):
    """
    将原始环境状态进行归一化处理，防止数值过大导致网络发散。
    假设 state 结构: [20个雷达数据, 距离, 角度, 线速度, 角速度]
    """
    # 1. 处理雷达数据 (前20维)
    # 将 inf 替换为 10 (假设最大探测距离)
    lidar = state[:-4]
    lidar = np.clip(lidar, 0, 10) # 限制在 0-10 之间
    lidar /= 10.0                 # 归一化到 0-1

    # 2. 处理机器人状态 (后4维)
    robot_info = state[-4:]
    robot_info[0] /= 10.0         # 距离归一化 (假设最大距离10米)
    # 角度(index 1) 已经是 -pi 到 pi，可以保持或除以 pi
    # 速度(index 2, 3) 通常较小，可以保持

    return np.concatenate((lidar, robot_info))

# ==========================================
# 评估函数
# ==========================================
def evaluate(network, epoch, eval_episodes=10):
    avg_reward = 0.0
    col = 0
    for _ in range(eval_episodes):
        count = 0
        raw_state = env.reset()
        state = process_state(raw_state) # <--- 加入归一化
        done = False
        episode_collision = False 

        while not done and count < 501:
            action = network.get_action(np.array(state))
            a_in = [(action[0] + 1) / 2, action[1]]
            raw_state, reward, done, _ = env.step(a_in)
            state = process_state(raw_state) # <--- 加入归一化
            avg_reward += reward
            count += 1
            # --- 只要发生过一次小于 -90，就标记为碰撞 ---
            if reward < -90:
                episode_collision = True
        # --- 每一局结束后，只计一次数 ---
        if episode_collision:
            col += 1
    avg_reward /= eval_episodes
    avg_col = col / eval_episodes
    print("..............................................")
    print(
        "Average Reward over %i Evaluation Episodes, Epoch %i: %f, Collision Rate: %f"
        % (eval_episodes, epoch, avg_reward, avg_col)
    )
    print("..............................................")
    return avg_reward

# ==========================================
# 网络定义 (参考 TD3.py 优化)
# ==========================================
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()

        # 优化：减小网络规模 800->400
        self.layer_1 = nn.Linear(state_dim, 400)
        # 优化：Kaiming 初始化
        torch.nn.init.kaiming_uniform_(self.layer_1.weight, nonlinearity="leaky_relu")
        
        self.layer_2 = nn.Linear(400, 300)
        torch.nn.init.kaiming_uniform_(self.layer_2.weight, nonlinearity="leaky_relu")
        
        self.layer_3 = nn.Linear(300, action_dim)
        self.tanh = nn.Tanh()

    def forward(self, s):
        # 优化：使用 LeakyReLU
        s = F.leaky_relu(self.layer_1(s))
        s = F.leaky_relu(self.layer_2(s))
        a = self.tanh(self.layer_3(s))
        return a

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 Architecture
        self.layer_1 = nn.Linear(state_dim + action_dim, 400) # 输入合并
        torch.nn.init.kaiming_uniform_(self.layer_1.weight, nonlinearity="leaky_relu")
        
        self.layer_2 = nn.Linear(400, 300)
        torch.nn.init.kaiming_uniform_(self.layer_2.weight, nonlinearity="leaky_relu")
        
        self.layer_3 = nn.Linear(300, 1)

        # Q2 Architecture
        self.layer_4 = nn.Linear(state_dim + action_dim, 400)
        torch.nn.init.kaiming_uniform_(self.layer_4.weight, nonlinearity="leaky_relu")
        
        self.layer_5 = nn.Linear(400, 300)
        torch.nn.init.kaiming_uniform_(self.layer_5.weight, nonlinearity="leaky_relu")
        
        self.layer_6 = nn.Linear(300, 1)

    def forward(self, s, a):
        sa = torch.cat([s, a], 1) # 将状态和动作拼接

        q1 = F.leaky_relu(self.layer_1(sa))
        q1 = F.leaky_relu(self.layer_2(q1))
        q1 = self.layer_3(q1)

        q2 = F.leaky_relu(self.layer_4(sa))
        q2 = F.leaky_relu(self.layer_5(q2))
        q2 = self.layer_6(q2)
        return q1, q2

# ==========================================
# TD3 算法类
# ==========================================
class TD3(object):
    def __init__(self, state_dim, action_dim, max_action):
        # 优化：显式设置学习率为 1e-4
        lr = 1e-4 
        
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.max_action = max_action
        self.writer = SummaryWriter()
        self.iter_count = 0

    def get_action(self, state):
        state = torch.tensor(state.reshape(1, -1), dtype=torch.float32).to(device)
        with torch.no_grad():
            return self.actor(state).cpu().detach().numpy().flatten()

    def train(
        self,
        replay_buffer,
        iterations,
        batch_size=100,
        discount=0.99, # 建议改为 0.99，原来 0.99999 太大了
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
    ):
        av_Q = 0
        max_Q = -inf
        av_loss = 0
        
        for it in range(iterations):
            # Sample batch
            (
                batch_states,
                batch_actions,
                batch_rewards,
                batch_dones,
                batch_next_states,
            ) = replay_buffer.sample_batch(batch_size)
            
            state = torch.tensor(batch_states, dtype=torch.float32).to(device)
            next_state = torch.tensor(batch_next_states, dtype=torch.float32).to(device)
            action = torch.tensor(batch_actions, dtype=torch.float32).to(device)
            reward = torch.tensor(batch_rewards, dtype=torch.float32).to(device)
            done = torch.tensor(batch_dones, dtype=torch.float32).to(device)

            # Target Q calculation
            with torch.no_grad():
                next_action = self.actor_target(next_state)
                noise = torch.randn_like(action).normal_(0, policy_noise).to(device)
                noise = noise.clamp(-noise_clip, noise_clip)
                next_action = (next_action + noise).clamp(-self.max_action, self.max_action)

                target_Q1, target_Q2 = self.critic_target(next_state, next_action)
                target_Q = torch.min(target_Q1, target_Q2)
                # 记录 Max Q 用于分析
                av_Q += torch.mean(target_Q)
                max_Q = max(max_Q, torch.max(target_Q).item())
                
                target_Q = reward + ((1 - done) * discount * target_Q)

            # Current Q calculation
            current_Q1, current_Q2 = self.critic(state, action)

            # 优化：使用 SmoothL1Loss (Huber Loss) 防止梯度爆炸
            loss = F.smooth_l1_loss(current_Q1, target_Q) + F.smooth_l1_loss(current_Q2, target_Q)

            # Gradient Descent
            self.critic_optimizer.zero_grad()
            loss.backward()
            # 优化：加入梯度裁剪，进一步防止爆炸
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0) 
            self.critic_optimizer.step()

            # Delayed Policy Update
            if it % policy_freq == 0:
                actor_grad, _ = self.critic(state, self.actor(state))
                actor_grad = -actor_grad.mean()
                self.actor_optimizer.zero_grad()
                actor_grad.backward()
                self.actor_optimizer.step()

                # Soft Update
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            av_loss += loss.item()
            
        self.iter_count += 1
        self.writer.add_scalar("loss", av_loss / iterations, self.iter_count)
        self.writer.add_scalar("Av. Q", av_Q / iterations, self.iter_count)
        self.writer.add_scalar("Max. Q", max_Q, self.iter_count)

    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), "%s/%s_actor.pth" % (directory, filename))
        torch.save(self.critic.state_dict(), "%s/%s_critic.pth" % (directory, filename))

    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load("%s/%s_actor.pth" % (directory, filename), map_location=device))
        self.critic.load_state_dict(torch.load("%s/%s_critic.pth" % (directory, filename), map_location=device))

# ==========================================
# 主程序设置
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 0
eval_freq = 5e3
max_ep = 500
eval_ep = 10
max_timesteps = 5e6
expl_noise = 1
expl_decay_steps = 500000
expl_min = 0.1
batch_size = 40
discount = 0.99 # 修改回标准的 0.99
tau = 0.005
policy_noise = 0.2
noise_clip = 0.5
policy_freq = 2
buffer_size = int(1e6)
# file_name = "TD3_velodyne"
file_name = "TD3_velodyne_best"
save_model = True
load_model = True
random_near_obstacle = True

if not os.path.exists("./results"):
    os.makedirs("./results")
if save_model and not os.path.exists("./pytorch_models"):
    os.makedirs("./pytorch_models")

environment_dim = 20
robot_dim = 4
env = GazeboEnv("multi_robot_scenario.launch", environment_dim)
print("Waiting for Gazebo and ROS nodes to fully initialize...")
time.sleep(10)

torch.manual_seed(seed)
np.random.seed(seed)
state_dim = environment_dim + robot_dim
action_dim = 2
max_action = 1

network = TD3(state_dim, action_dim, max_action)
replay_buffer = ReplayBuffer(buffer_size, seed)

if load_model:
    try:
        network.load(file_name, "./pytorch_models")
    except Exception as e:
        print(f"Could not load model: {e}")

evaluations = []
timestep = 0
timesteps_since_eval = 0
episode_num = 0
done = True
epoch = 1

count_rand_actions = 0
random_action = []

# 初始化 best_avg_reward 为负无穷
best_avg_reward = -np.inf 

# ==========================================
# 训练循环
# ==========================================
while timestep < max_timesteps:

    if done:
        if timestep != 0:
            network.train(
                replay_buffer,
                episode_timesteps,
                batch_size,
                discount,
                tau,
                policy_noise,
                noise_clip,
                policy_freq,
            )

        if timesteps_since_eval >= eval_freq:
            print("Validating")
            timesteps_since_eval %= eval_freq

            # 1. 获取当前评估的平均分
            # 注意：evaluate 函数必须有返回值 return avg_reward
            avg_reward = evaluate(network=network, epoch=epoch, eval_episodes=eval_ep)
            evaluations.append(avg_reward)
            # 2. 保存“最新”模型 (覆盖旧的，用于断点续传)
            # 文件名例如: TD3_velodyne_actor.pth
            network.save(file_name, directory="./pytorch_models")
            # 3. 保存“最优”模型 (!!! 核心修改 !!!)
            # 如果当前分数比历史最高分还高，就额外存一份带 _best 后缀的文件
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                print(f"🌟 New Best Model Found! Reward: {best_avg_reward:.2f} Saving...")
                
                # 保存为 TD3_velodyne_best_actor.pth
                network.save(file_name + "_best", directory="./pytorch_models")
            
            # 4. (可选) 每 50 或 100 个 Epoch 强制备份一次
            if epoch % 10 == 0:
                network.save(f"{file_name}_epoch_{epoch}", directory="./pytorch_models")

            np.save("./results/%s" % (file_name), evaluations)
            epoch += 1

        raw_state = env.reset()
        state = process_state(raw_state) # <--- 归一化
        done = False

        episode_reward = 0
        episode_timesteps = 0
        episode_num += 1

    if expl_noise > expl_min:
        expl_noise = expl_noise - ((1 - expl_min) / expl_decay_steps)

    action = network.get_action(np.array(state))
    action = (action + np.random.normal(0, expl_noise, size=action_dim)).clip(
        -max_action, max_action
    )

    if random_near_obstacle:
        if (
            np.random.uniform(0, 1) > 0.85
            and min(state[0:environment_dim]) < 0.06 # 注意：这里用归一化后的数据判断，0.06 = 0.6米
            and count_rand_actions < 1
        ):
            count_rand_actions = np.random.randint(8, 15)
            random_action = np.random.uniform(-1, 1, 2)

        if count_rand_actions > 0:
            count_rand_actions -= 1
            action = random_action
            action[0] = -1

    a_in = [(action[0] + 1) / 2, action[1]]
    
    # 执行动作，获取原始数据
    raw_next_state, reward, done, target = env.step(a_in)
    
    # 对 Next State 进行归一化处理
    next_state = process_state(raw_next_state) # <--- 归一化

    done_bool = 0 if episode_timesteps + 1 == max_ep else int(done)
    done = 1 if episode_timesteps + 1 == max_ep else int(done)
    episode_reward += reward

    # 存入 Buffer 的必须是处理后的 state
    replay_buffer.add(state, action, reward, done_bool, next_state)

    state = next_state
    episode_timesteps += 1
    timestep += 1
    timesteps_since_eval += 1

evaluations.append(evaluate(network=network, epoch=epoch, eval_episodes=eval_ep))
if save_model:
    network.save("%s" % file_name, directory="./pytorch_models")
np.save("./results/%s" % file_name, evaluations)