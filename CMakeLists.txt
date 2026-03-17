cmake_minimum_required(VERSION 3.14)
project(point_seg)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -O2)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(image_transport REQUIRED)

find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED COMPONENTS common filters)
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(yaml-cpp REQUIRED)
find_package(OpenVINO REQUIRED)

add_executable(fusion_seg_paint_node_cpp
  src/fusion_seg_paint_node_cpp.cpp
  src/inference.cpp
)

target_include_directories(fusion_seg_paint_node_cpp
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${PCL_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

ament_target_dependencies(fusion_seg_paint_node_cpp
  rclcpp
  sensor_msgs
  std_msgs
  cv_bridge
  pcl_conversions
  image_transport
)

target_link_libraries(fusion_seg_paint_node_cpp
  ${PCL_LIBRARIES}
  ${OpenCV_LIBS}
  Eigen3::Eigen
  yaml-cpp
  openvino::runtime
)

target_compile_definitions(fusion_seg_paint_node_cpp
  PRIVATE ${PCL_DEFINITIONS}
)

install(TARGETS fusion_seg_paint_node_cpp
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
)

ament_python_install_package(${PROJECT_NAME})
install(PROGRAMS
  point_seg/fusion_seg_paint_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME fusion_seg_paint_node
)

ament_package()
