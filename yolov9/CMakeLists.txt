cmake_minimum_required(VERSION 3.0.2)
project(yolov9_ros)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  cv_bridge
  message_generation
)

add_message_files(
  FILES
  BoundingBox.msg
  BoundingBoxes.msg
  )
  
generate_messages(
  DEPENDENCIES
  std_msgs
  )

catkin_package()

catkin_install_python(PROGRAMS
  scripts/detect_ros.py
  scripts/detect_ros2.py
  #scripts/yolov9_node.sh
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)
