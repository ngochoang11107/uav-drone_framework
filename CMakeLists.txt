cmake_minimum_required(VERSION 3.8)
project(aruco_detection)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)

install(PROGRAMS
  aruco_detection/aruco_pose_estimation_node.py
  aruco_detection/aruco_pose_node.py
  #aruco_detection/aruco_pose_node_mavros.py
  aruco_detection/calibration.py
  aruco_detection/detector.py
  aruco_detection/transformation.py
  aruco_detection/visualization.py 
  aruco_detection/load_camera.py
  aruco_detection/marker_localizer_node.py
  #aruco_detection/marker_localizer_node_mavros.py
  aruco_detection/gimbal_control.py
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY scripts/
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
