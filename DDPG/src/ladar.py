#!/usr/bin/env python3
import rospy
import numpy as np
import threading
import time
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import matplotlib.pyplot as plt

class LidarTest:
    def __init__(self, lidar_points=360):
        # 初始化 ROS 节点
        rospy.init_node('lidar_test_node', anonymous=True)
        
        # 激光雷达数据参数
        self.lidar_points = lidar_points
        self.current_lidar_scan = np.zeros(lidar_points, dtype=np.float32)
        self.lidar_lock = threading.Lock()
        self.data_received = False
        self.raw_points = []  # 存储原始点云数据
        
        # 订阅激光雷达点云主题
        self.lidar_sub = rospy.Subscriber(
            '/wamv/sensors/lidars/lidar_wamv/points', 
            PointCloud2, 
            self.lidar_callback,
            queue_size=1,
            buff_size=2**24
        )
        
        rospy.loginfo("激光雷达测试节点已初始化，等待数据...")
    
    def lidar_callback(self, data):
        """处理激光雷达点云数据"""
        with self.lidar_lock:
            # 记录原始点云
            self.raw_points = list(pc2.read_points(data, field_names=("x", "y", "z"), skip_nans=True))
            
            if not self.raw_points:
                rospy.logwarn("激光雷达数据为空!")
                return
            
            rospy.loginfo(f"收到 {len(self.raw_points)} 个点云数据点")
            
            # 将3D点云处理为2D距离数据
            angles = np.linspace(0, 2*np.pi, self.lidar_points, endpoint=False)
            ranges = np.ones(self.lidar_points) * 100.0  # 默认最大距离
            
            valid_points = 0
            min_dist = float('inf')
            max_dist = 0
            
            for p in self.raw_points:
                x, y, _ = p
                distance = np.sqrt(x**2 + y**2)
                
                # 记录最小和最大距离
                if distance < min_dist:
                    min_dist = distance
                if distance > max_dist and distance < 100.0:  # 忽略默认最大值
                    max_dist = distance
                
                angle = np.arctan2(y, x) % (2*np.pi)
                
                # 找到最接近的角度索引
                idx = int((angle / (2*np.pi)) * self.lidar_points)
                if idx < self.lidar_points and distance < ranges[idx]:
                    ranges[idx] = distance
                    valid_points += 1
            
            # 更新激光雷达数据
            self.current_lidar_scan = ranges
            self.data_received = True
            
            # 打印调试信息
            nan_count = np.sum(np.isnan(ranges))
            inf_count = np.sum(np.isinf(ranges))
            zero_count = np.sum(ranges == 0)
            default_count = np.sum(ranges == 100.0)
            
            rospy.loginfo(f"激光雷达数据处理完成: 有效点={valid_points}, NaN={nan_count}, Inf={inf_count}, 零值={zero_count}, 默认值={default_count}")
            rospy.loginfo(f"距离范围: 最小={min_dist:.2f}m, 最大={max_dist:.2f}m")
            
            # 检查前方区域的数据
            front_indices = [i % self.lidar_points for i in range(
                int(self.lidar_points * 11/12),
                int(self.lidar_points * 1/12)
            )]
            
            if front_indices:
                front_data = ranges[front_indices]
                front_nan = np.sum(np.isnan(front_data))
                front_inf = np.sum(np.isinf(front_data))
                front_min = np.min(front_data) if len(front_data) > 0 else float('nan')
                try:
                    front_mean = np.mean(front_data)
                except:
                    front_mean = float('nan')
                
                rospy.loginfo(f"前方区域: 索引数量={len(front_indices)}, NaN={front_nan}, Inf={front_inf}, 最小距离={front_min:.2f}m, 平均距离={front_mean:.2f}m")
            else:
                rospy.logwarn("前方区域索引为空!")
    
    def plot_lidar_data(self):
        """绘制激光雷达数据的极坐标图"""
        if not self.data_received:
            rospy.logwarn("尚未收到激光雷达数据，无法绘图")
            return
        
        with self.lidar_lock:
            # 创建极坐标图
            plt.figure(figsize=(10, 10))
            ax = plt.subplot(111, projection='polar')
            
            # 获取角度和距离
            angles = np.linspace(0, 2*np.pi, self.lidar_points, endpoint=False)
            ranges = self.current_lidar_scan
            
            # 过滤无效值
            valid_mask = ~np.isnan(ranges) & ~np.isinf(ranges) & (ranges < 50.0)
            valid_angles = angles[valid_mask]
            valid_ranges = ranges[valid_mask]
            
            # 绘制数据点
            ax.scatter(valid_angles, valid_ranges, s=2, c='blue')
            
            # 高亮前方区域
            front_indices = [i % self.lidar_points for i in range(
                int(self.lidar_points * 11/12),
                int(self.lidar_points * 1/12)
            )]
            
            if front_indices and len(front_indices) > 0:
                front_angles = angles[front_indices]
                front_ranges = ranges[front_indices]
                
                # 过滤前方区域的无效值
                front_valid = ~np.isnan(front_ranges) & ~np.isinf(front_ranges) & (front_ranges < 50.0)
                front_valid_angles = front_angles[front_valid]
                front_valid_ranges = front_ranges[front_valid]
                
                ax.scatter(front_valid_angles, front_valid_ranges, s=5, c='red')
            
            # 设置图表属性
            ax.set_theta_zero_location('N')  # 0度在北方（上方）
            ax.set_theta_direction(-1)  # 顺时针方向
            ax.set_rlim(0, 20)  # 限制半径范围为0-20米
            ax.set_title('激光雷达数据可视化')
            ax.grid(True)
            
            # 添加描述
            plt.figtext(0.5, 0.01, '蓝色: 所有激光雷达数据点\n红色: 前方区域数据点', 
                       ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
            
            # 保存图表
            plt.savefig('lidar_visualization.png')
            rospy.loginfo("激光雷达数据可视化已保存到 lidar_visualization.png")
    
    def test_front_calculation(self):
        """测试前方区域计算逻辑"""
        if not self.data_received:
            rospy.logwarn("尚未收到激光雷达数据，无法测试")
            return
        
        with self.lidar_lock:
            # 获取前方区域索引
            front_indices = [i % self.lidar_points for i in range(
                int(self.lidar_points * 11/12),
                int(self.lidar_points * 1/12)
            )]
            
            rospy.loginfo(f"前方区域索引计算: 11/12 * {self.lidar_points} = {int(self.lidar_points * 11/12)}")
            rospy.loginfo(f"前方区域索引计算: 1/12 * {self.lidar_points} = {int(self.lidar_points * 1/12)}")
            rospy.loginfo(f"前方区域索引数量: {len(front_indices)}")
            
            if len(front_indices) > 0:
                # 测试索引的有效性
                min_idx = min(front_indices)
                max_idx = max(front_indices)
                rospy.loginfo(f"前方区域索引范围: {min_idx} 到 {max_idx}")
                
                # 获取前方区域数据
                front_data = self.current_lidar_scan[front_indices]
                
                # 测试各种统计方法
                try:
                    front_min = np.min(front_data)
                    front_max = np.max(front_data)
                    front_mean = np.mean(front_data)
                    front_median = np.median(front_data)
                    
                    # 过滤后的统计
                    valid_data = front_data[~np.isnan(front_data) & ~np.isinf(front_data)]
                    valid_min = np.min(valid_data) if len(valid_data) > 0 else float('nan')
                    valid_mean = np.mean(valid_data) if len(valid_data) > 0 else float('nan')
                    
                    rospy.loginfo(f"前方数据统计: 最小={front_min:.2f}, 最大={front_max:.2f}, 均值={front_mean:.2f}, 中位数={front_median:.2f}")
                    rospy.loginfo(f"过滤后统计: 有效点数={len(valid_data)}, 最小={valid_min:.2f}, 均值={valid_mean:.2f}")
                    
                    # 测试奖励计算
                    heading_reward = 0.3 * min(front_mean, 10.0) / 10.0
                    safe_heading_reward = 0.3 * min(valid_mean if not np.isnan(valid_mean) else 5.0, 10.0) / 10.0
                    
                    rospy.loginfo(f"奖励计算测试: 常规={heading_reward:.2f}, 安全版本={safe_heading_reward:.2f}")
                    
                except Exception as e:
                    rospy.logerr(f"统计计算出错: {str(e)}")
            else:
                rospy.logwarn("前方区域索引为空，无法进行测试")
    
    def visualize_raw_points(self):
        """可视化原始点云数据"""
        if not self.raw_points:
            rospy.logwarn("尚未收到原始点云数据，无法可视化")
            return
        
        # 提取x,y,z坐标
        x = [p[0] for p in self.raw_points]
        y = [p[1] for p in self.raw_points]
        
        # 创建散点图
        plt.figure(figsize=(10, 10))
        plt.scatter(x, y, s=1, c='green', alpha=0.5)
        
        # 添加船只位置标记
        plt.scatter(0, 0, s=100, c='red', marker='*')
        
        # 添加前方区域指示
        angles = np.linspace(11/12 * 2*np.pi, 2*np.pi, 50)
        angles = np.append(angles, np.linspace(0, 1/12 * 2*np.pi, 50))
        r = 15  # 半径
        front_x = r * np.cos(angles)
        front_y = r * np.sin(angles)
        plt.plot(front_x, front_y, 'r--')
        
        # 设置图表属性
        plt.grid(True)
        plt.axis('equal')
        plt.xlim(-20, 20)
        plt.ylim(-20, 20)
        plt.title('原始点云数据可视化')
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        
        # 添加描述
        plt.figtext(0.5, 0.01, '绿色: 原始点云数据\n红色星标: 船只位置\n红色虚线: 前方区域', 
                   ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
        
        # 保存图表
        plt.savefig('raw_points_visualization.png')
        rospy.loginfo("原始点云数据可视化已保存到 raw_points_visualization.png")
    
    def run_tests(self):
        """运行所有测试"""
        # 等待接收数据
        timeout = 10.0  # 10秒超时
        start_time = time.time()
        
        rospy.loginfo("等待接收激光雷达数据...")
        while not self.data_received and time.time() - start_time < timeout:
            time.sleep(0.1)
        
        if not self.data_received:
            rospy.logerr("超时: 未收到激光雷达数据!")
            return False
        
        rospy.loginfo("收到激光雷达数据，开始测试...")
        
        # 运行各项测试
        self.test_front_calculation()
        self.plot_lidar_data()
        self.visualize_raw_points()
        
        return True

if __name__ == "__main__":
    try:
        # 创建测试对象
        tester = LidarTest(lidar_points=360)
        
        # 运行测试
        success = tester.run_tests()
        
        # 保持节点运行以便查看结果
        if success:
            rospy.loginfo("测试完成，等待5秒后退出...")
            time.sleep(5)
        else:
            rospy.loginfo("测试失败，等待5秒后退出...")
            time.sleep(5)
            
    except rospy.ROSInterruptException:
        pass