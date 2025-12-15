import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from velodyne_env import GazeboEnv

# ==========================================
# 1. 必须加上这个归一化函数，保持和训练时一致
# ==========================================
def process_state(state):
    """
    将原始环境状态进行归一化处理。
    """
    # 1. 处理雷达数据 (前20维)
    lidar = state[:-4]
    lidar = np.clip(lidar, 0, 10) # 限制在 0-10 之间
    lidar /= 10.0                 # 归一化到 0-1

    # 2. 处理机器人状态 (后4维)
    robot_info = state[-4:]
    robot_info[0] /= 10.0         # 距离归一化
    
    return np.concatenate((lidar, robot_info))

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()

        # ==========================================
        # 2. 修改网络结构以匹配训练代码 (800->400, 600->300)
        # ==========================================
        self.layer_1 = nn.Linear(state_dim, 400)
        self.layer_2 = nn.Linear(400, 300)
        self.layer_3 = nn.Linear(300, action_dim)
        self.tanh = nn.Tanh()

    def forward(self, s):
        # 使用 LeakyReLU 匹配训练时的设置
        s = F.leaky_relu(self.layer_1(s))
        s = F.leaky_relu(self.layer_2(s))
        a = self.tanh(self.layer_3(s))
        return a

# TD3 network
class TD3(object):
    def __init__(self, state_dim, action_dim):
        # Initialize the Actor network
        self.actor = Actor(state_dim, action_dim).to(device)

    def get_action(self, state):
        # Function to get the action from the actor
        state = torch.Tensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def load(self, filename, directory):
        # Function to load network parameters
        print(f"Loading model: {filename} from {directory}...")
        self.actor.load_state_dict(
            torch.load("%s/%s_actor.pth" % (directory, filename), map_location=device)
        )

# Set the parameters for the implementation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
seed = 0  
max_ep = 500

# ==========================================
# 3. 这里修改为你保存的“最强模型”的名字
# ==========================================
file_name = "TD3_velodyne_best" 
# 如果你没有由 _best 结尾的文件，就改回 "TD3_velodyne"

# Create the testing environment
environment_dim = 20
robot_dim = 4
env = GazeboEnv("multi_robot_scenario.launch", environment_dim)
time.sleep(5)
torch.manual_seed(seed)
np.random.seed(seed)
state_dim = environment_dim + robot_dim
action_dim = 2

# Create the network
network = TD3(state_dim, action_dim)
try:
    network.load(file_name, "./pytorch_models")
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Trying default model name 'TD3_velodyne'...")
    try:
        network.load("TD3_velodyne", "./pytorch_models")
    except:
        raise ValueError("Could not load any stored model parameters")

done = False
episode_timesteps = 0

# 重置环境
raw_state = env.reset()
# 归一化初始状态
state = process_state(raw_state)

print("Starting testing loop...")

# Begin the testing loop
while True:
    # 获取动作 (不加噪声，纯贪婪策略)
    action = network.get_action(np.array(state))

    # Update action to fall in range [0,1] for linear velocity and [-1,1] for angular velocity
    a_in = [(action[0] + 1) / 2, action[1]]
    
    # 执行动作
    raw_next_state, reward, done, target = env.step(a_in)
    
    # ==========================================
    # 4. 对 Next State 进行归一化
    # ==========================================
    next_state = process_state(raw_next_state)

    done = 1 if episode_timesteps + 1 == max_ep else int(done)

    # On termination of episode
    if done:
        if target:
            print(f"SUCCESS! Reached target in {episode_timesteps} steps.")
        else:
            print(f"Failed (Collision/Timeout) at step {episode_timesteps}.")
            
        raw_state = env.reset()
        state = process_state(raw_state)
        done = False
        episode_timesteps = 0
    else:
        state = next_state
        episode_timesteps += 1