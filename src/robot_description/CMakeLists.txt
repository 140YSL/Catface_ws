cmake_minimum_required(VERSION 3.8)
project(robot_description)

# ROS2 不再有 catkin，改用 ament_cmake
find_package(ament_cmake REQUIRED)

# 安装 share 目录下的资源文件（URDF、meshes、launch 等）
install(
  DIRECTORY urdf meshes launch rviz blender
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
