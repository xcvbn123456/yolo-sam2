cmake_minimum_required(VERSION 3.0.2)
project(sam2_ros)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  cv_bridge
)


catkin_package()

catkin_install_python(PROGRAMS
  scripts/sam2_ros.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)
