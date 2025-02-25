#!/usr/bin/env python3
import rospy
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import gym
from gym import spaces
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Float32
import sensor_msgs.point_cloud2 as pc2
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState, GetModelState
from geometry_msgs.msg import Pose, Twist, Point, Quaternion
import threading
import time
import os
import pickle
import math
import scipy.signal

class PPOBuffer:
    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        
        # GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = self.discount_cumsum(deltas, self.gamma * self.lam)
        
        # rewards-to-go, which we will use as targets for the value function
        self.ret_buf[path_slice] = self.discount_cumsum(rews, self.gamma)[:-1]
        
        self.path_start_idx = self.ptr

    def get(self):
        assert self.ptr == self.max_size
        self.ptr, self.path_start_idx = 0, 0
        
        # Advantage normalization
        adv_mean, adv_std = np.mean(self.adv_buf), np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / (adv_std + 1e-10)
        
        return self.obs_buf, self.act_buf, self.adv_buf, self.ret_buf, self.logp_buf

    def discount_cumsum(self, x, discount):
        """
        计算折扣累加和
        [x0, x1, x2] -> [x0 + discount * x1 + discount^2 * x2, x1 + discount * x2, x2]
        """
        return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class Actor(tf.keras.Model):
    def __init__(self, action_dim):
        super(Actor, self).__init__()
        self.dense1 = layers.Dense(256, activation='relu')
        self.dense2 = layers.Dense(256, activation='relu')
        # 添加一个额外的层以获得更平滑的输出
        self.dense3 = layers.Dense(128, activation='relu')
        self.mean = layers.Dense(action_dim, activation='tanh')  # 输出范围为[-1, 1]
        # 初始化为更小的标准差，使动作更确定
        self.log_std = tf.Variable(np.zeros(action_dim, dtype=np.float32) - 1.0)
        
    def call(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        x = self.dense3(x)  # 额外的层使输出更平滑
        mean = self.mean(x)
        std = tf.exp(self.log_std)
        return mean, std
    
    def get_action(self, state, deterministic=False):
        mean, std = self.call(state)
        if deterministic:
            return mean
        else:
            normal = tf.random.normal(shape=mean.shape)
            action = mean + normal * std
            return tf.clip_by_value(action, -1.0, 1.0)
    
    def log_prob(self, state, action):
        mean, std = self.call(state)
        logp = -0.5 * tf.reduce_sum(
            ((action - mean) / (std + 1e-8))**2 + 2 * tf.math.log(std) + tf.math.log(2 * np.pi),
            axis=-1
        )
        return logp


class Critic(tf.keras.Model):
    def __init__(self):
        super(Critic, self).__init__()
        self.dense1 = layers.Dense(256, activation='relu')
        self.dense2 = layers.Dense(256, activation='relu')
        self.value = layers.Dense(1)
        
    def call(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        return self.value(x)


class WAMVEnv:
    def __init__(self, lidar_points=360, max_episode_steps=1000, target_distance=20.0):
        # 状态和动作空间定义
        self.lidar_points = lidar_points  # 使用激光雷达点的数量
        self.observation_space = spaces.Box(low=0, high=100, shape=(lidar_points + 1,), dtype=np.float32)  # +1 for distance traveled
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        
        # 目标距离（前进20米）
        self.target_distance = target_distance
        
        # ROS 节点初始化
        rospy.init_node('wamv_rl_controller', anonymous=True)
        
        # 订阅激光雷达点云主题
        self.lidar_sub = rospy.Subscriber(
            '/wamv/sensors/lidars/lidar_wamv/points', 
            PointCloud2, 
            self.lidar_callback,
            queue_size=1,
            buff_size=2**24  # 增加缓冲区大小以防止丢弃消息
        )
        
        # 等待gazebo服务可用
        rospy.loginfo("等待gazebo服务...")
        try:
            rospy.wait_for_service('/gazebo/set_model_state', timeout=10)
            self.set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            rospy.wait_for_service('/gazebo/get_model_state', timeout=10)
            self.get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
            rospy.loginfo("成功连接到gazebo服务")
        except rospy.ROSException:
            rospy.logwarn("无法连接到gazebo服务，无法重置船位置")
            self.set_model_state = None
            self.get_model_state = None
        
        # 发布推力命令主题 - 增加发布频率
        self.left_thrust_pub = rospy.Publisher(
            '/wamv/thrusters/left_thrust_cmd', 
            Float32, 
            queue_size=10  # 增加队列大小以实现更连续的控制
        )
        self.right_thrust_pub = rospy.Publisher(
            '/wamv/thrusters/right_thrust_cmd', 
            Float32, 
            queue_size=10  # 增加队列大小以实现更连续的控制
        )
        
        # 添加动作平滑机制
        self.last_action = [0.0, 0.0]
        self.action_smoother_alpha = 0.7  # 平滑因子，越大代表新动作的权重越大
        
        # 状态变量
        self.current_lidar_scan = np.zeros(lidar_points, dtype=np.float32)
        self.previous_lidar_scan = np.zeros(lidar_points, dtype=np.float32)
        self.lidar_lock = threading.Lock()
        self.collision_detected = False
        self.step_count = 0
        self.max_episode_steps = max_episode_steps
        
        # 位置和任务进度跟踪
        self.start_position = None
        self.current_position = None
        self.distance_traveled = 0.0
        self.target_reached = False
        
        # 等待ROS连接
        time.sleep(1)
        rospy.loginfo("WAMV RL环境已初始化，目标前进距离: %.2f 米", self.target_distance)
    
    def lidar_callback(self, data):
        """处理激光雷达点云数据"""
        with self.lidar_lock:
            # 将3D点云处理为2D距离数据
            points = list(pc2.read_points(data, field_names=("x", "y", "z"), skip_nans=True))
            
            if not points:
                return
            
            # 计算点到船的距离并进行角度采样
            angles = np.linspace(0, 2*np.pi, self.lidar_points, endpoint=False)
            ranges = np.ones(self.lidar_points) * 100.0  # 默认最大距离
            
            for p in points:
                x, y, _ = p
                distance = np.sqrt(x**2 + y**2)
                angle = np.arctan2(y, x) % (2*np.pi)
                
                # 找到最接近的角度索引
                idx = int((angle / (2*np.pi)) * self.lidar_points)
                if idx < self.lidar_points and distance < ranges[idx]:
                    ranges[idx] = distance
            
            # 修改碰撞检测阈值，防止误报
            # 只有当多个点都非常接近时才认为是碰撞
            close_points = np.sum(ranges < 0.3)  # 降低阈值并检查多个点
            if close_points > 5:  # 至少需要5个点来确认碰撞
                self.collision_detected = True
                rospy.logwarn("碰撞检测：距离过近，点数=%d", close_points)
            
            # 更新激光雷达数据
            self.previous_lidar_scan = self.current_lidar_scan.copy()
            self.current_lidar_scan = ranges
    
    def get_front_indices(self):
        """获取船舶前方区域的激光雷达索引，正确处理环绕情况"""
        start = int(self.lidar_points * 11/12)
        end = int(self.lidar_points * 1/12)
        
        # 处理环绕情况
        if start > end:
            # 环绕情况：创建两段范围 (start到lidar_points-1 和 0到end)
            indices = [i % self.lidar_points for i in range(start, self.lidar_points)]
            indices.extend([i for i in range(0, end)])
        else:
            # 正常情况
            indices = [i for i in range(start, end)]
            
        return indices
    
    def get_wamv_position(self):
        """获取WAM-V当前位置"""
        if self.get_model_state is None:
            rospy.logwarn("无法获取船位置：gazebo服务不可用")
            return None
            
        try:
            result = self.get_model_state("wamv", "world")
            if result.success:
                return result.pose.position
            else:
                rospy.logwarn(f"获取船位置失败: {result.status_message}")
                return None
        except rospy.ServiceException as e:
            rospy.logerr(f"服务调用失败: {e}")
            return None
            
    def calculate_distance_traveled(self):
        """计算船沿向前方向移动的距离"""
        if self.start_position is None or self.current_position is None:
            return 0.0
            
        # 添加额外检查
        if hasattr(self.start_position, 'x') and hasattr(self.current_position, 'x'):
            # 计算在起始点朝向方向上的位移
            dx = self.current_position.x - self.start_position.x
            dy = self.current_position.y - self.start_position.y
            
            # 简单使用欧几里得距离
            distance = math.sqrt(dx**2 + dy**2)
            
            # 防止NaN
            if np.isnan(distance) or np.isinf(distance):
                rospy.logwarn("距离计算出现NaN，使用0.0")
                return 0.0
                
            return max(0.0, distance)  # 确保距离为非负
        else:
            rospy.logwarn("位置对象缺少x/y属性")
            return 0.0
        
    def reset(self):
        """重置环境，将船移动到指定的初始位置和朝向"""
        rospy.loginfo("重置环境开始...")
        
        # 发送零推力命令以停止船只
        self.send_thrust_cmd(0.0, 0.0)
        
        # 打印重置前的位置
        prev_pos = self.get_wamv_position()
        if prev_pos:
            rospy.loginfo(f"重置前位置: x={prev_pos.x:.2f}, y={prev_pos.y:.2f}")
        
        # 重置状态变量
        self.collision_detected = False
        self.step_count = 0
        self.last_action = [0.0, 0.0]  # 重置动作平滑器
        self.target_reached = False
        self.distance_traveled = 0.0
        
        # 重置船的位置到指定坐标和朝向
        success = self.reset_wamv_position(x=100.0, y=28.0, z=0.0, yaw=-2.7)
        rospy.loginfo(f"位置重置{'成功' if success else '失败'}")
        
        # 等待新的激光雷达数据
        time.sleep(1.5)  # 增加等待时间，确保船位置重置后能获取新数据
        
        # 记录起始位置
        self.start_position = self.get_wamv_position()
        if self.start_position:
            rospy.loginfo(f"新起始位置: x={self.start_position.x:.2f}, y={self.start_position.y:.2f}")
        else:
            rospy.logerr("无法获取起始位置!")
        self.current_position = self.start_position
        
        # 返回初始观测 (激光雷达数据 + 已行驶距离/目标距离)
        with self.lidar_lock:
            lidar_data = self.current_lidar_scan.copy()
        
        # 构建观察空间：激光雷达数据 + 已行驶距离比例
        observation = np.append(lidar_data, [0.0])  # 添加已行驶距离，初始为0
        
        return observation
        
    def reset_wamv_position(self, x, y, z=0.0, yaw=-2.7):
        """将WAM-V重置到指定位置和朝向"""
        if self.set_model_state is None:
            rospy.logwarn("无法重置船位置：gazebo服务不可用")
            return False
            
        try:
            # 创建新的位姿
            pose = Pose()
            pose.position = Point(x, y, z)
            
            # 将偏航角转换为四元数
            q = Quaternion()
            q.x = 0
            q.y = 0
            q.z = math.sin(yaw / 2)
            q.w = math.cos(yaw / 2)
            pose.orientation = q
            
            # 创建零速度
            twist = Twist()
            
            # 创建模型状态消息
            model_state = ModelState()
            model_state.model_name = "wamv"  # 这里需要确认您的船模型在gazebo中的名称
            model_state.pose = pose
            model_state.twist = twist
            model_state.reference_frame = "world"
            
            # 调用服务
            result = self.set_model_state(model_state)
            
            if result.success:
                rospy.loginfo(f"已将船重置到位置 ({x}, {y}, {z}) 朝向 yaw={yaw}")
                return True
            else:
                rospy.logwarn(f"重置船位置失败: {result.status_message}")
                return False
                
        except rospy.ServiceException as e:
            rospy.logerr(f"服务调用失败: {e}")
            return False
    
    def step(self, action):
        """执行一步动作"""
        # 限制动作范围并发送推力命令
        clipped_action = np.clip(action, -1.0, 1.0)
        
        # 为了实现向前20米的任务，强制左右推力都为正，且有最小值
        # 这样船只会主要向前移动
        left_thrust = max(0.3, 0.5 * (clipped_action[0] + 1.0))  # 映射到[0.3, 1.0]范围
        right_thrust = max(0.3, 0.5 * (clipped_action[1] + 1.0))  # 映射到[0.3, 1.0]范围
        
        # 发送推力命令
        self.send_thrust_cmd(left_thrust, right_thrust)
        
        # 更改：减少等待时间以实现更连续的动作
        time.sleep(0.05)
        
        # 增加步数计数
        self.step_count += 1
        
        # 更新当前位置和行驶距离
        self.current_position = self.get_wamv_position()
        if self.current_position is not None and self.start_position is not None:
            self.distance_traveled = self.calculate_distance_traveled()
            
            # 检查是否达到目标距离
            if self.distance_traveled >= self.target_distance:
                self.target_reached = True
                rospy.loginfo("目标达成！已前进 %.2f 米", self.distance_traveled)
        
        # 获取当前状态
        with self.lidar_lock:
            lidar_data = self.current_lidar_scan.copy()
            collision = self.collision_detected
            # 重置碰撞标志，避免误报
            self.collision_detected = False
        
        # 构建观察空间：激光雷达数据 + 已行驶距离比例
        progress = min(1.0, self.distance_traveled / self.target_distance)
        observation = np.append(lidar_data, [progress])
        
        # 计算奖励
        reward = self.compute_reward(lidar_data, collision, self.distance_traveled, self.target_reached)
        
        # 判断回合是否结束
        done = collision or self.target_reached or (self.step_count >= self.max_episode_steps)
        
        # 提供附加信息
        info = {
            'collision': collision,
            'step': self.step_count,
            'distance_traveled': self.distance_traveled,
            'target_reached': self.target_reached
        }
        
        return observation, reward, done, info
    
    def send_thrust_cmd(self, left_thrust, right_thrust):
        """发送推力命令到船只，带平滑处理"""
        # 应用动作平滑以减少突变
        smoothed_left = self.action_smoother_alpha * left_thrust + (1 - self.action_smoother_alpha) * self.last_action[0]
        smoothed_right = self.action_smoother_alpha * right_thrust + (1 - self.action_smoother_alpha) * self.last_action[1]
        
        # 保存当前平滑后的动作以供下次使用
        self.last_action = [smoothed_left, smoothed_right]
        
        # 创建消息并发布命令
        left_msg = Float32()
        right_msg = Float32()
        left_msg.data = float(smoothed_left)
        right_msg.data = float(smoothed_right)
        
        # 发布命令
        self.left_thrust_pub.publish(left_msg)
        self.right_thrust_pub.publish(right_msg)
        
        # 为确保连续运动，短时间内多次发送相同命令
        # 这有助于保持控制信号的持续性
        rospy.Timer(rospy.Duration(0.01), lambda _: self.left_thrust_pub.publish(left_msg), oneshot=True)
        rospy.Timer(rospy.Duration(0.01), lambda _: self.right_thrust_pub.publish(right_msg), oneshot=True)
    
    def compute_reward(self, lidar_scan, collision, distance_traveled, target_reached):
        """计算奖励函数 - 优先前进20米并避障"""
        if collision:
            return -100.0  # 碰撞惩罚
            
        if target_reached:
            return 100.0  # 达到目标奖励
        
        # 前进奖励：根据距离进度给予奖励
        progress_reward = 0.5 * (distance_traveled / self.target_distance)
        
        # 安全距离奖励：激励保持与障碍物的安全距离
        min_distance = np.min(lidar_scan)
        safety_threshold = 2.0
        if min_distance < safety_threshold:
            safety_reward = -2.0 * (safety_threshold - min_distance) / safety_threshold
        else:
            safety_reward = 0.5
        
        # 前进方向奖励：鼓励船只朝前方开阔的方向行驶
        # 获取前方区域的索引（环绕处理）
        front_indices = self.get_front_indices()
        
        # 安全处理前方区域数据
        if front_indices and len(front_indices) > 0:
            front_distances = lidar_scan[front_indices]
            # 过滤无效值
            valid_distances = front_distances[~np.isnan(front_distances) & ~np.isinf(front_distances)]
            if len(valid_distances) > 0:
                heading_reward = 0.3 * min(np.mean(valid_distances), 10.0) / 10.0
            else:
                heading_reward = 0.1  # 默认值
        else:
            heading_reward = 0.1  # 默认值
        
        # 平衡左右推力奖励：鼓励船只直线前进
        try:
            thrust_balance_reward = 0.1 * (1.0 - abs(self.last_action[0] - self.last_action[1]))
            if np.isnan(thrust_balance_reward) or np.isinf(thrust_balance_reward):
                thrust_balance_reward = 0.0
        except:
            thrust_balance_reward = 0.0
        
        # 总奖励：前进 + 安全 + 方向 + 平衡
        total_reward = progress_reward + safety_reward + heading_reward + thrust_balance_reward
        
        # 检查是否有NaN
        if np.isnan(total_reward) or np.isinf(total_reward):
            rospy.logwarn("奖励计算出现NaN/Inf，使用默认奖励值")
            total_reward = 0.1  # 小的正奖励
        
        # 记录调试信息
        if self.step_count % 20 == 0:  # 每20步记录一次，避免日志过多
            rospy.loginfo(f"距离={distance_traveled:.2f}m, 奖励={total_reward:.2f} (进度={progress_reward:.2f}, 安全={safety_reward:.2f}, 方向={heading_reward:.2f}, 平衡={thrust_balance_reward:.2f})")
            
        return total_reward


class PPOAgent:
    def __init__(self, state_dim, action_dim, save_path="./wamv_forward_model"):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_path = save_path
        
        # 创建actor和critic网络
        self.actor = Actor(action_dim)
        self.critic = Critic()
        
        # 优化器
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        
        # PPO参数
        self.clip_ratio = 0.2
        self.target_kl = 0.01
        self.train_actor_iterations = 80
        self.train_critic_iterations = 80
        
        # 创建保存目录
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    
    def save_model(self, episode):
        self.actor.save_weights(f"{self.save_path}/actor_{episode}.h5")
        self.critic.save_weights(f"{self.save_path}/critic_{episode}.h5")
        print(f"模型已保存到 {self.save_path}")
    
    def load_model(self, episode):
        try:
            self.actor.load_weights(f"{self.save_path}/actor_{episode}.h5")
            self.critic.load_weights(f"{self.save_path}/critic_{episode}.h5")
            print(f"成功加载模型 episode {episode}")
            return True
        except:
            print(f"无法加载模型 episode {episode}")
            return False
    
    def get_action(self, state, deterministic=False):
        state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
        action = self.actor.get_action(state_tensor, deterministic)[0]
        value = self.critic(state_tensor)[0, 0]
        logp = self.actor.log_prob(state_tensor, action)[0]
        
        return action.numpy(), value.numpy(), logp.numpy()
    
    def train(self, buffer):
        obs_buf, act_buf, adv_buf, ret_buf, logp_buf = buffer.get()
        
        # 将numpy数组转换为tensorflow张量
        obs_tensor = tf.convert_to_tensor(obs_buf, dtype=tf.float32)
        act_tensor = tf.convert_to_tensor(act_buf, dtype=tf.float32)
        adv_tensor = tf.convert_to_tensor(adv_buf, dtype=tf.float32)
        ret_tensor = tf.convert_to_tensor(ret_buf, dtype=tf.float32)
        logp_old_tensor = tf.convert_to_tensor(logp_buf, dtype=tf.float32)
        
        # 训练Actor网络
        for i in range(self.train_actor_iterations):
            with tf.GradientTape() as tape:
                # 计算新的log概率
                logp_tensor = self.actor.log_prob(obs_tensor, act_tensor)
                
                # 计算比率 r(θ) = π_θ(a|s) / π_θ_old(a|s)
                ratio = tf.exp(logp_tensor - logp_old_tensor)
                
                # 裁剪目标
                clip_adv = tf.clip_by_value(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * adv_tensor
                
                # PPO 损失函数
                actor_loss = -tf.reduce_mean(tf.minimum(ratio * adv_tensor, clip_adv))
                
            # 计算梯度并应用
            actor_gradients = tape.gradient(actor_loss, self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))
            
            # 早停：如果KL散度过大则提前停止
            if i % 10 == 0:
                kl = tf.reduce_mean(logp_old_tensor - self.actor.log_prob(obs_tensor, act_tensor))
                if kl > 1.5 * self.target_kl:
                    break
        
        # 训练Critic网络
        for i in range(self.train_critic_iterations):
            with tf.GradientTape() as tape:
                values = self.critic(obs_tensor)
                critic_loss = tf.reduce_mean((ret_tensor - values)**2)
            
            # 计算梯度并应用
            critic_gradients = tape.gradient(critic_loss, self.critic.trainable_variables)
            self.critic_optimizer.apply_gradients(zip(critic_gradients, self.critic.trainable_variables))
        
        return actor_loss.numpy(), critic_loss.numpy()


def train_forward_navigation():
    """训练强化学习代理实现向前导航并避障的任务"""
    # 环境配置
    env = WAMVEnv(lidar_points=360, max_episode_steps=1000, target_distance=20.0)
    
    # 状态维度需要+1，因为添加了距离信息
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # 创建PPO代理
    agent = PPOAgent(state_dim, action_dim)
    
    # 训练参数
    episodes = 200  # 训练回合数
    buffer_size = 2000  # 缓冲区大小
    batch_size = 64
    buffer = PPOBuffer(state_dim, action_dim, buffer_size)
    
    # 记录训练信息
    rewards_history = []
    episode_length_history = []
    collision_history = []
    distance_history = []
    target_reached_history = []
    
    # 训练循环
    for episode in range(episodes):
        # 每个episode开始时重置环境（会自动重置到指定坐标）
        state = env.reset()
        episode_reward = 0
        done = False
        timestep = 0
        
        rospy.loginfo(f"=== 开始 Episode {episode}/{episodes} ===")
        rospy.loginfo("无人船已重置到起始位置，目标前进距离: %.2f 米", env.target_distance)
        
        # 短暂暂停，确保船已完全重置
        time.sleep(0.5)
        
        # 以下循环处理一个完整的episode
        while not done and timestep < env.max_episode_steps:
            # 获取动作
            action, value, logp = agent.get_action(state)
            
            # 执行动作
            next_state, reward, done, info = env.step(action)
            
            # 存储轨迹
            buffer.store(state, action, reward, value, logp)
            
            # 更新状态
            state = next_state
            episode_reward += reward
            timestep += 1
            
            # Episode结束的条件：
            # 1. 碰撞发生 (done为True)
            # 2. 达到目标距离 (target_reached为True)
            # 3. 达到最大步数 (timestep >= env.max_episode_steps)
            if done:
                if info.get('collision', False):
                    reason = "碰撞"
                elif info.get('target_reached', False):
                    reason = "达到目标"
                else:
                    reason = "达到最大步数"
                print(f"Episode {episode} 结束于步骤 {timestep}，原因：{reason}，总距离：{info.get('distance_traveled', 0):.2f}米")
                
                # 如果由于碰撞结束，给予较低的最终价值估计
                if info.get('collision', False):
                    buffer.finish_path(0)
                # 如果成功到达目标，给予较高的最终价值估计
                elif info.get('target_reached', False):
                    buffer.finish_path(10)
                else:
                    # 对于其他情况，使用当前状态的估计值
                    _, last_val, _ = agent.get_action(state)
                    buffer.finish_path(last_val)
            
            # 如果缓冲区已满，执行训练
            if buffer.ptr >= buffer_size:
                # 如果还没有结束，获取最后状态的值，用于计算优势函数
                if not done:
                    _, last_val, _ = agent.get_action(state)
                    buffer.finish_path(last_val)
                
                # 训练网络
                actor_loss, critic_loss = agent.train(buffer)
                print(f"Episode {episode}, 训练损失: Actor={actor_loss:.4f}, Critic={critic_loss:.4f}")
        
        # 记录训练信息
        rewards_history.append(episode_reward)
        episode_length_history.append(timestep)
        collision_history.append(1 if info.get('collision', False) else 0)
        distance_history.append(info.get('distance_traveled', 0))
        target_reached_history.append(1 if info.get('target_reached', False) else 0)
        
        # 打印回合信息
        print(f"Episode {episode}: 奖励={episode_reward:.2f}, 步数={timestep}, 距离={info.get('distance_traveled', 0):.2f}米, 目标达成={info.get('target_reached', False)}")
        
        # 确保船停止运动
        env.send_thrust_cmd(0.0, 0.0)
        
        # 如果不是最后一个episode，将船重置到指定位置为下一轮做准备
        if episode < episodes - 1:
            rospy.loginfo(f"Episode {episode} 结束，重置船准备下一回合")
            # 这里不需要手动调用reset_wamv_position，因为下一个循环开始时会调用env.reset()
        
        # 每10回合保存一次模型
        if episode % 10 == 0:
            agent.save_model(episode)
            
            # 保存训练历史
            history = {
                'rewards': rewards_history,
                'episode_lengths': episode_length_history,
                'collisions': collision_history,
                'distances': distance_history,
                'target_reached': target_reached_history
            }
            with open(f"{agent.save_path}/history_{episode}.pkl", 'wb') as f:
                pickle.dump(history, f)
    
    # 训练结束，保存最终模型
    agent.save_model("final")
    print("训练完成!")


def run_forward_test(model_episode="final", test_episodes=1):
    """
    使用训练好的模型执行向前导航测试
    只是前进20米并避障，不使用强化学习训练
    """
    # 环境配置
    env = WAMVEnv(lidar_points=360, max_episode_steps=1000, target_distance=20.0)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # 创建代理并加载模型
    agent = PPOAgent(state_dim, action_dim)
    if not agent.load_model(model_episode):
        print(f"无法加载模型 episode {model_episode}, 将使用随机初始化的模型")
    
    # 测试循环
    for episode in range(test_episodes):
        # 重置环境
        state = env.reset()
        done = False
        timestep = 0
        total_reward = 0
        
        rospy.loginfo(f"=== 开始测试 Episode {episode+1}/{test_episodes} ===")
        rospy.loginfo("无人船已重置到起始位置, 目标前进距离: %.2f 米", env.target_distance)
        
        # 短暂暂停，确保船已完全重置
        time.sleep(0.5)
        
        while not done and timestep < env.max_episode_steps:
            # 使用确定性动作
            action, _, _ = agent.get_action(state, deterministic=True)
            
            # 执行动作
            next_state, reward, done, info = env.step(action)
            
            # 更新状态
            state = next_state
            total_reward += reward
            timestep += 1
            
            # 打印进度
            if timestep % 20 == 0:
                rospy.loginfo(f"步骤 {timestep}: 距离 = {info.get('distance_traveled', 0):.2f}米 / {env.target_distance}米")
        
        # 总结测试结果
        if info.get('collision', False):
            result = "碰撞"
        elif info.get('target_reached', False):
            result = "成功达到目标"
        else:
            result = "未能在时间内到达目标"
            
        rospy.loginfo(f"测试 Episode {episode+1} 结束: {result}")
        rospy.loginfo(f"总步数: {timestep}, 总奖励: {total_reward:.2f}, 总距离: {info.get('distance_traveled', 0):.2f}米")
        
        # 确保船停止运动
        env.send_thrust_cmd(0.0, 0.0)
        
        # 等待下一次测试
        if episode < test_episodes - 1:
            time.sleep(2.0)
            

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='WAMV前进避障导航')
    parser.add_argument('--mode', type=str, default='test', choices=['train', 'test'], 
                        help='运行模式: train(训练)或test(测试)')
    parser.add_argument('--episodes', type=int, default=1, 
                        help='测试模式下的测试次数')
    parser.add_argument('--model', type=str, default='final', 
                        help='测试模式下加载的模型')
    args = parser.parse_args()
    
    try:
        if args.mode == 'train':
            train_forward_navigation()
        elif args.mode == 'test':
            run_forward_test(model_episode=args.model, test_episodes=args.episodes)
    except rospy.ROSInterruptException:
        pass