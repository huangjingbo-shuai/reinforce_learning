import rospy
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
import numpy as np
import open3d as o3d

# 雷达点云回调函数
def lidar_callback(msg):
    # 转换点云数据
    pc_data = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
    pc_data = np.array(list(pc_data))

    # 输出点云数据的基本信息
    rospy.loginfo(f"Received {pc_data.shape[0]} points")

    # 创建 open3d 点云对象
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(pc_data)

    # 使用VoxelGrid进行下采样
    voxel_size = 0.1
    downsampled_cloud = cloud.voxel_down_sample(voxel_size)

    # 输出滤波后的点云信息
    rospy.loginfo(f"Filtered {len(downsampled_cloud.points)} points")

    # 可视化点云
    o3d.visualization.draw_geometries([downsampled_cloud])

# 初始化ROS节点
rospy.init_node('lidar_data_processor', anonymous=True)

# 订阅雷达点云话题
rospy.Subscriber("/wamv/sensors/lidars/lidar_wamv/points", PointCloud2, lidar_callback)

# 保持程序运行
rospy.spin()
