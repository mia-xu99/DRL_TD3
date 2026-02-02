# CMake generated Testfile for 
# Source directory: /home/mia/DRL-robot-navigation/catkin_ws/src/ouster_example/ouster_description
# Build directory: /home/mia/DRL-robot-navigation/catkin_ws/build_isolated/ouster_description
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(_ctest_ouster_description_rostest_tests_os1_gazebo.test "/home/mia/DRL-robot-navigation/catkin_ws/build_isolated/ouster_description/catkin_generated/env_cached.sh" "/usr/bin/python3" "/opt/ros/noetic/share/catkin/cmake/test/run_tests.py" "/home/mia/DRL-robot-navigation/catkin_ws/build_isolated/ouster_description/test_results/ouster_description/rostest-tests_os1_gazebo.xml" "--return-code" "/usr/bin/python3 /opt/ros/noetic/share/rostest/cmake/../../../bin/rostest --pkgdir=/home/mia/DRL-robot-navigation/catkin_ws/src/ouster_example/ouster_description --package=ouster_description --results-filename tests_os1_gazebo.xml --results-base-dir \"/home/mia/DRL-robot-navigation/catkin_ws/build_isolated/ouster_description/test_results\" /home/mia/DRL-robot-navigation/catkin_ws/src/ouster_example/ouster_description/tests/os1_gazebo.test ")
set_tests_properties(_ctest_ouster_description_rostest_tests_os1_gazebo.test PROPERTIES  _BACKTRACE_TRIPLES "/opt/ros/noetic/share/catkin/cmake/test/tests.cmake;160;add_test;/opt/ros/noetic/share/rostest/cmake/rostest-extras.cmake;52;catkin_run_tests_target;/home/mia/DRL-robot-navigation/catkin_ws/src/ouster_example/ouster_description/CMakeLists.txt;14;add_rostest;/home/mia/DRL-robot-navigation/catkin_ws/src/ouster_example/ouster_description/CMakeLists.txt;0;")
subdirs("gtest")
