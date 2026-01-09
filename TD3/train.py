import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import inf
from torch.utils.tensorboard import SummaryWriter

# 导入自定义模块
from replay_buffer import ReplayBuffer
from velodyne_env import GazeboEnv

# ==========================================
# 工具函数：数据归一化 (预处理)
# ==========================================
def process_state(state):
    """
    状态预处理函数。
    作用：将原始环境状态进行归一化处理，防止数值过大导致神经网络梯度爆炸或难以收敛。
    
    输入 state 结构假设: [20个雷达射线距离, 目标距离, 目标角度, 线速度, 角速度]
    """
    # 1. 处理雷达数据 (前20维)
    # 假设雷达最大探测距离为 10米
    lidar = state[:-4]
    lidar = np.clip(lidar, 0, 10) # 截断数据，限制在 0-10 之间
    lidar /= 10.0                 # 归一化到 0-1 之间

    # 2. 处理机器人状态信息 (后4维)
    robot_info = state[-4:]
    robot_info[0] /= 10.0         # 目标距离归一化 (假设最大关注距离为10米)
    # robot_info[1] 是角度 (-pi 到 pi)，通常数值范围尚可，可不处理或除以 pi
    # robot_info[2], [3] 是线速度和角速度，数值较小，通常保持原样

    # 重新拼接归一化后的数据
    return np.concatenate((lidar, robot_info))

# ==========================================
# 评估函数 (Testing/Validation)
# ==========================================
def evaluate(network, epoch, eval_episodes=10):
    """
    在不添加探索噪声的情况下评估当前策略的表现。
    """
    avg_reward = 0.0
    col = 0 # 碰撞计数
    
    for _ in range(eval_episodes):
        count = 0
        raw_state = env.reset()
        state = process_state(raw_state) # 记得评估时也要归一化
        done = False
        episode_collision = False 

        while not done and count < 501:
            # 获取动作 (测试模式下不加噪声)
            action = network.get_action(np.array(state))
            # 将输出动作映射回环境所需范围
            # action[0] (线速度): 网络输出 [-1, 1] -> 映射到 [0, 1]
            # action[1] (角速度): 网络输出 [-1, 1] -> 保持 [-1, 1]
            a_in = [(action[0] + 1) / 2, action[1]]
            
            raw_state, reward, done, _ = env.step(a_in)
            state = process_state(raw_state)
            avg_reward += reward
            count += 1
            
            # --- 碰撞检测逻辑 ---
            # 假设环境设定 reward < -90 表示发生了严重碰撞
            if reward < -90:
                episode_collision = True
        
        # 统计发生碰撞的回合数
        if episode_collision:
            col += 1

    avg_reward /= eval_episodes
    avg_col = col / eval_episodes # 计算碰撞率
    
    print("..............................................")
    print(
        "Average Reward over %i Evaluation Episodes, Epoch %i: %f, Collision Rate: %f"
        % (eval_episodes, epoch, avg_reward, avg_col)
    )
    print("..............................................")
    return avg_reward

# ==========================================
# 网络定义 (Actor 和 Critic)
# ==========================================
class Actor(nn.Module):
    """
    策略网络 (Policy Network)
    输入: 状态 (state)
    输出: 动作 (action) - 连续值
    """
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()

        # 网络层定义 (已优化网络规模)
        self.layer_1 = nn.Linear(state_dim, 400)
        # Kaiming 初始化有助于深层网络收敛
        torch.nn.init.kaiming_uniform_(self.layer_1.weight, nonlinearity="leaky_relu")
        
        self.layer_2 = nn.Linear(400, 300)
        torch.nn.init.kaiming_uniform_(self.layer_2.weight, nonlinearity="leaky_relu")
        
        self.layer_3 = nn.Linear(300, action_dim)
        self.tanh = nn.Tanh() # 输出层使用 Tanh 将动作限制在 [-1, 1]

    def forward(self, s):
        s = F.leaky_relu(self.layer_1(s))
        s = F.leaky_relu(self.layer_2(s))
        a = self.tanh(self.layer_3(s))
        return a

class Critic(nn.Module):
    """
    价值网络 (Value Network) - Twin Critic 结构
    输入: 状态 (state) + 动作 (action)
    输出: Q值 (Q-value)
    TD3 使用两个 Critic (Q1, Q2) 来缓解过高估计 (Overestimation) 问题。
    """
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # --- Q1 网络架构 ---
        self.layer_1 = nn.Linear(state_dim + action_dim, 400) # 输入是 state 和 action 的拼接
        torch.nn.init.kaiming_uniform_(self.layer_1.weight, nonlinearity="leaky_relu")
        
        self.layer_2 = nn.Linear(400, 300)
        torch.nn.init.kaiming_uniform_(self.layer_2.weight, nonlinearity="leaky_relu")
        
        self.layer_3 = nn.Linear(300, 1) # 输出单个 Q 值

        # --- Q2 网络架构 (结构相同，参数独立) ---
        self.layer_4 = nn.Linear(state_dim + action_dim, 400)
        torch.nn.init.kaiming_uniform_(self.layer_4.weight, nonlinearity="leaky_relu")
        
        self.layer_5 = nn.Linear(400, 300)
        torch.nn.init.kaiming_uniform_(self.layer_5.weight, nonlinearity="leaky_relu")
        
        self.layer_6 = nn.Linear(300, 1)

    def forward(self, s, a):
        sa = torch.cat([s, a], 1) # 拼接状态和动作

        # 计算 Q1
        q1 = F.leaky_relu(self.layer_1(sa))
        q1 = F.leaky_relu(self.layer_2(q1))
        q1 = self.layer_3(q1)

        # 计算 Q2
        q2 = F.leaky_relu(self.layer_4(sa))
        q2 = F.leaky_relu(self.layer_5(q2))
        q2 = self.layer_6(q2)
        return q1, q2

# ==========================================
# TD3 算法核心逻辑
# ==========================================
class TD3(object):
    def __init__(self, state_dim, action_dim, max_action):
        lr = 1e-4 # 学习率
        
        # 初始化 Actor 及其目标网络
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        # 初始化 Critic 及其目标网络
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.max_action = max_action
        self.writer = SummaryWriter() # 用于 Tensorboard 记录
        self.iter_count = 0

    def get_action(self, state):
        """前向传播获取动作，用于交互"""
        state = torch.tensor(state.reshape(1, -1), dtype=torch.float32).to(device)
        with torch.no_grad():
            return self.actor(state).cpu().detach().numpy().flatten()

    def train(
        self,
        replay_buffer,
        iterations,
        batch_size=40,
        discount=0.99,   # 折扣因子 gamma
        tau=0.005,       # 软更新系数
        policy_noise=0.2,# 目标动作平滑噪声标准差
        noise_clip=0.5,  # 噪声裁剪范围
        policy_freq=2,   # 策略更新频率 (延迟更新)
    ):
        av_Q = 0
        max_Q = -inf
        av_loss = 0
        
        for it in range(iterations):
            # 1. 从经验回放池采样
            (
                batch_states,
                batch_actions,
                batch_rewards,
                batch_dones,
                batch_next_states,
            ) = replay_buffer.sample_batch(batch_size)
            
            # 转换为 Tensor 并放入 GPU/CPU
            state = torch.tensor(batch_states, dtype=torch.float32).to(device)
            next_state = torch.tensor(batch_next_states, dtype=torch.float32).to(device)
            action = torch.tensor(batch_actions, dtype=torch.float32).to(device)
            reward = torch.tensor(batch_rewards, dtype=torch.float32).to(device)
            done = torch.tensor(batch_dones, dtype=torch.float32).to(device)

            # 2. 计算目标 Q 值 (Target Q)
            with torch.no_grad():
                # 目标策略平滑 (Target Policy Smoothing):
                # 在目标动作上添加噪声，使 Value 估计更平滑，防止过拟合到尖峰
                next_action = self.actor_target(next_state)
                noise = torch.randn_like(action).normal_(0, policy_noise).to(device)
                noise = noise.clamp(-noise_clip, noise_clip)
                next_action = (next_action + noise).clamp(-self.max_action, self.max_action)

                # 获取两个目标 Critic 的 Q 值
                target_Q1, target_Q2 = self.critic_target(next_state, next_action)
                # 取最小值 (Clipped Double Q-learning)，缓解过估计
                target_Q = torch.min(target_Q1, target_Q2)
                
                # 记录数据用于分析
                av_Q += torch.mean(target_Q)
                max_Q = max(max_Q, torch.max(target_Q).item())
                
                # Bellman 方程计算目标值
                target_Q = reward + ((1 - done) * discount * target_Q)

            # 3. 计算当前 Q 值并更新 Critic
            current_Q1, current_Q2 = self.critic(state, action)

            # 使用 SmoothL1Loss (Huber Loss) 相比 MSE 对异常值更不敏感
            loss = F.smooth_l1_loss(current_Q1, target_Q) + F.smooth_l1_loss(current_Q2, target_Q)

            self.critic_optimizer.zero_grad()
            loss.backward()
            # 梯度裁剪 (防止梯度爆炸)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0) 
            self.critic_optimizer.step()

            # 4. 延迟更新 Actor (Delayed Policy Update)
            # 只有 Critic 更新 policy_freq 次后，才更新一次 Actor
            if it % policy_freq == 0:
                # 计算 Actor 损失: 最大化 Q1 值 -> 最小化 -Q1
                actor_grad, _ = self.critic(state, self.actor(state))
                actor_grad = -actor_grad.mean()
                
                self.actor_optimizer.zero_grad()
                actor_grad.backward()
                self.actor_optimizer.step()

                # 5. 软更新目标网络 (Soft Update)
                # target_param = tau * param + (1 - tau) * target_param
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            av_loss += loss.item()
            
        self.iter_count += 1
        # 写入 Tensorboard
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
eval_freq = 5e3       # 每 5000 步评估一次
max_ep = 500          # 每个 Episode 最大步数
eval_ep = 10          # 每次评估跑几个 Episode
max_timesteps = 5e6   # 总训练步数
expl_noise = 1        # 初始探索噪声
expl_decay_steps = 500000 # 噪声衰减步数
expl_min = 0.1        # 最小探索噪声
batch_size = 40       # 训练批次大小
discount = 0.99       # 折扣因子
tau = 0.005           # 软更新参数
policy_noise = 0.2    # 策略平滑噪声
noise_clip = 0.5      # 噪声裁剪
policy_freq = 2       # Actor 更新延迟频率
buffer_size = int(1e6)# 经验回放池大小

# 模型保存名称配置
file_name = "TD3_velodyne_best"
save_model = True
load_model = False    # 是否加载已有模型
random_near_obstacle = True # 是否启用遇障随机策略
SAFE_DIST = 0.6       # 安全距离 (米)
LIDAR_MAX = 10.0      # 雷达最大距离

# 创建目录
if not os.path.exists("./results"):
    os.makedirs("./results")
if save_model and not os.path.exists("./pytorch_models"):
    os.makedirs("./pytorch_models")

environment_dim = 20 # 雷达维度
robot_dim = 4        # 机器人状态维度
# 初始化环境
env = GazeboEnv("multi_robot_scenario.launch", environment_dim)
print("Waiting for Gazebo and ROS nodes to fully initialize...")
time.sleep(10)

# 设置随机种子
torch.manual_seed(seed)
np.random.seed(seed)
state_dim = environment_dim + robot_dim
action_dim = 2
max_action = 1

# 初始化 TD3 网络
network = TD3(state_dim, action_dim, max_action)
replay_buffer = ReplayBuffer(buffer_size, seed)

# 加载模型 (如果启用)
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

# 初始化最佳奖励记录 (用于保存最佳模型)
best_avg_reward = -np.inf 

# ==========================================
# 训练主循环
# ==========================================
while timestep < max_timesteps:

    # 如果一个 Episode 结束
    if done:
        # 只要不是刚开始，就进行网络训练
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

        # 检查是否需要评估
        if timesteps_since_eval >= eval_freq:
            print("Validating")
            timesteps_since_eval %= eval_freq

            # 1. 评估当前模型
            avg_reward = evaluate(network=network, epoch=epoch, eval_episodes=eval_ep)
            evaluations.append(avg_reward)
            
            # 2. 保存“最新”模型 (方便断点续传)
            network.save(file_name, directory="./pytorch_models")
            
            # 3. 保存“最优”模型 (如果当前分数突破历史最高)
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                print(f"🌟 New Best Model Found! Reward: {best_avg_reward:.2f} Saving...")
                network.save(file_name + "_best", directory="./pytorch_models")
            
            # 4. 定期备份 (可选)
            if epoch % 10 == 0:
                network.save(f"{file_name}_epoch_{epoch}", directory="./pytorch_models")

            np.save("./results/%s" % (file_name), evaluations)
            epoch += 1

        # 重置环境，开始新 Episode
        raw_state = env.reset()
        state = process_state(raw_state) # <--- 归一化输入
        done = False

        episode_reward = 0
        episode_timesteps = 0
        episode_num += 1

    # --- 探索噪声衰减 ---
    if expl_noise > expl_min:
        expl_noise = expl_noise - ((1 - expl_min) / expl_decay_steps)

    # 获取动作
    action = network.get_action(np.array(state))
    # 添加高斯噪声进行探索 (Exploration)
    action = (action + np.random.normal(0, expl_noise, size=action_dim)).clip(
        -max_action, max_action
    )

    # --- 特殊恢复策略: 如果离障碍物太近，强制执行随机动作 ---
    # 目的: 防止机器人陷入局部死胡同
    if random_near_obstacle:
        if (
            np.random.uniform(0, 1) > 0.85 # 有一定概率触发
            # 检查雷达最小值，判断是否过近 (注意这里使用归一化后的数据比较)
            # SAFE_DIST(0.6) / LIDAR_MAX(10.0) = 0.06
            and min(state[0:environment_dim]) < SAFE_DIST / LIDAR_MAX 
            and count_rand_actions < 1
        ):
            count_rand_actions = np.random.randint(8, 15) # 持续随机动作 8-15 步
            random_action = np.random.uniform(-1, 1, 2)

        if count_rand_actions > 0:
            count_rand_actions -= 1
            action = random_action
            action[0] = -1 # 强制倒车或其它行为

    # 将动作转换为环境可接受的格式 (线速度 [0,1], 角速度 [-1,1])
    a_in = [(action[0] + 1) / 2, action[1]]
    
    # 执行动作
    raw_next_state, reward, done, target = env.step(a_in)
    
    # 处理 Next State (归一化)
    next_state = process_state(raw_next_state)

    # 标记是否因为达到最大步数而结束 (TimeLimit)
    # 如果是因为步数耗尽，done_bool 应为 0 (以便 Critic 依然计算后续价值)
    done_bool = 0 if episode_timesteps + 1 == max_ep else int(done)
    done = 1 if episode_timesteps + 1 == max_ep else int(done)
    episode_reward += reward

    # 存入 Replay Buffer (存入的是归一化后的 state)
    replay_buffer.add(state, action, reward, done_bool, next_state)

    state = next_state
    episode_timesteps += 1
    timestep += 1
    timesteps_since_eval += 1

# 训练结束后的最终保存
evaluations.append(evaluate(network=network, epoch=epoch, eval_episodes=eval_ep))
if save_model:
    network.save("%s" % file_name, directory="./pytorch_models")
np.save("./results/%s" % file_name, evaluations)