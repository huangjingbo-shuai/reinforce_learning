# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/ubuntu/vrx_ws/src

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/ubuntu/vrx_ws/build

# Utility rule file for vrx_2019__xacro_auto_generate.

# Include the progress variables for this target.
include vrx/vrx_2019/CMakeFiles/vrx_2019__xacro_auto_generate.dir/progress.make

vrx/vrx_2019/CMakeFiles/vrx_2019__xacro_auto_generate: vrx/vrx_2019/worlds/dock.world
vrx/vrx_2019/CMakeFiles/vrx_2019__xacro_auto_generate: vrx/vrx_2019/worlds/example_course.world
vrx/vrx_2019/CMakeFiles/vrx_2019__xacro_auto_generate: vrx/vrx_2019/worlds/navigation_task.world
vrx/vrx_2019/CMakeFiles/vrx_2019__xacro_auto_generate: vrx/vrx_2019/worlds/ocean.world
vrx/vrx_2019/CMakeFiles/vrx_2019__xacro_auto_generate: vrx/vrx_2019/worlds/perception_task.world
vrx/vrx_2019/CMakeFiles/vrx_2019__xacro_auto_generate: vrx/vrx_2019/worlds/sandisland.world
vrx/vrx_2019/CMakeFiles/vrx_2019__xacro_auto_generate: vrx/vrx_2019/worlds/scan_and_dock.world
vrx/vrx_2019/CMakeFiles/vrx_2019__xacro_auto_generate: vrx/vrx_2019/worlds/stationkeeping_task.world
vrx/vrx_2019/CMakeFiles/vrx_2019__xacro_auto_generate: vrx/vrx_2019/worlds/wayfinding_task.world
vrx/vrx_2019/CMakeFiles/vrx_2019__xacro_auto_generate: vrx/vrx_2019/worlds/wind_test.world


vrx/vrx_2019/worlds/dock.world: /home/ubuntu/vrx_ws/src/vrx/vrx_2019/worlds/dock.world.xacro
vrx/vrx_2019/worlds/dock.world: /home/ubuntu/vrx_ws/src/vrx/vrx_2019/worlds/sandisland.xacro
vrx/vrx_2019/worlds/dock.world: /home/ubuntu/vrx_ws/src/vrx/wave_gazebo/world_models/ocean_waves/model.xacro
vrx/vrx_2019/worlds/dock.world: /home/ubuntu/vrx_ws/src/vrx/vrx_gazebo/worlds/xacros/usv_wind_plugin.xacro
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/ubuntu/vrx_ws/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "xacro: generating worlds/dock.world from worlds/dock.world.xacro"
	cd /home/ubuntu/vrx_ws/src/vrx/vrx_2019 && /home/ubuntu/vrx_ws/build/catkin_generated/env_cached.sh xacro -o /home/ubuntu/vrx_ws/build/vrx/vrx_2019/worlds/dock.world worlds/dock.world.xacro

vrx/vrx_2019/worlds/example_course.world: /home/ubuntu/vrx_ws/src/vrx/vrx_2019/worlds/example_course.world.xacro
vrx/vrx_2019/worlds/example_course.world: /home/ubuntu/vrx_ws/src/vrx/vrx_2019/worlds/sandisland.xacro
vrx/vrx_2019/worlds/example_course.world: /home/ubuntu/vrx_ws/src/vrx/wave_gazebo/world_models/ocean_waves/model.xacro
vrx/vrx_2019/worlds/example_course.world: /home/ubuntu/vrx_ws/src/vrx/vrx_gazebo/worlds/xacros/usv_wind_plugin.xacro
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/ubuntu/vrx_ws/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "xacro: generating worlds/example_course.world from worlds/example_course.world.xacro"
	cd /home/ubuntu/vrx_ws/src/vrx/vrx_2019 && /home/ubuntu/vrx_ws/build/catkin_generated/env_cached.sh xacro -o /home/ubuntu/vrx_ws/build/vrx/vrx_2019/worlds/example_course.world worlds/example_course.world.xacro

vrx/vrx_2019/worlds/navigation_task.world: /home/ubuntu/vrx_ws/src/vrx/vrx_2019/worlds/navigation_task.world.xacro
vrx/vrx_2019/worlds/navigation_task.world: /home/ubuntu/vrx_ws/src/vrx/wave_gazebo/world_models/ocean_waves/model.xacro
vrx/vrx_2019/worlds/navigation_task.world: /home/ubuntu/vrx_ws/src/vrx/vrx_2019/worlds/sandisland.xacro
vrx/vrx_2019/worlds/navigation_task.world: /home/ubuntu/vrx_ws/src/vrx/vrx_gazebo/worlds/xacros/usv_wind_plugin.xacro
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/ubuntu/vrx_ws/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "xacro: generating worlds/navigation_task.world from worlds/navigation_task.world.xacro"
	cd /home/ubuntu/vrx_ws/src/vrx/vrx_2019 && /home/ubuntu/vrx_ws/build/catkin_generated/env_cached.sh xacro -o /home/ubuntu/vrx_ws/build/vrx/vrx_2019/worlds/navigation_task.world worlds/navigation_task.world.xacro

vrx/vrx_2019/worlds/ocean.world: /home/ubuntu/vrx_ws/src/vrx/vrx_2019/worlds/ocean.world.xacro
vrx/vrx_2019/worlds/ocean.world: /home/ubuntu/vrx_ws/src/vrx/wave_gazebo/world_models/ocean_waves/model.xacro
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/ubuntu/vrx_ws/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "xacro: generating worlds/ocean.world from worlds/ocean.world.xacro"
	cd /home/ubuntu/vrx_ws/src/vrx/vrx_2019 && /home/ubuntu/vrx_ws/build/catkin_generated/env_cached.sh xacro -o /home/ubuntu/vrx_ws/build/vrx/vrx_2019/worlds/ocean.world worlds/ocean.world.xacro

vrx/vrx_2019/worlds/perception_task.world: /home/ubuntu/vrx_ws/src/vrx/vrx_2019/worlds/perception_task.world.xacro
vrx/vrx_2019/worlds/perception_task.world: /home/ubuntu/vrx_ws/src/vrx/wave_gazebo/world_models/ocean_waves/model.xacro
vrx/vrx_2019/worlds/perception_task.world: /home/ubuntu/vrx_ws/src/vrx/vrx_2019/worlds/sandisland.xacro
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/ubuntu/vrx_ws/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "xacro: generating worlds/perception_task.world from worlds/perception_task.world.xacro"
	cd /home/ubuntu/vrx_ws/src/vrx/vrx_2019 && /home/ubuntu/vrx_ws/build/catkin_generated/env_cached.sh xacro -o /home/ubuntu/vrx_ws/build/vrx/vrx_2019/worlds/perception_task.world worlds/perception_task.world.xacro

vrx/vrx_2019/worlds/sandisland.world: /home/ubuntu/vrx_ws/src/vrx/vrx_2019/worlds/sandisland.world.xacro
vrx/vrx_2019/worlds/sandisland.world: /home/ubuntu/vrx_ws/src/vrx/vrx_gazebo/worlds/xacros/usv_wind_plugin.xacro
vrx/vrx_2019/worlds/sandisland.world: /home/ubuntu/vrx_ws/src/vrx/wave_gazebo/world_models/ocean_waves/model.xacro
vrx/vrx_2019/worlds/sandisland.world: /home/ubuntu/vrx_ws/src/vrx/vrx_2019/worlds/sandisland.xacro
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/ubuntu/vrx_ws/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "xacro: generating worlds/sandisland.world from worlds/sandisland.world.xacro"
	cd /home/ubuntu/vrx_ws/src/vrx/vrx_2019 && /home/ubuntu/vrx_ws/build/catkin_generated/env_cached.sh xacro -o /home/ubuntu/vrx_ws/build/vrx/vrx_2019/worlds/sandisland.world worlds/sandisland.world.xacro

vrx/vrx_2019/worlds/scan_and_dock.world: /home/ubuntu/vrx_ws/src/vrx/vrx_2019/worlds/scan_and_dock.world.xacro
vrx/vrx_2019/worlds/scan_and_dock.world: /home/ubuntu/vrx_ws/src/vrx/vrx_2019/worlds/sandisland.xacro
vrx/vrx_2019/worlds/scan_and_dock.world: /home/ubuntu/vrx_ws/src/vrx/vrx_gazebo/worlds/xacros/usv_wind_plugin.xacro
vrx/vrx_2019/worlds/scan_and_dock.world: /home/ubuntu/vrx_ws/src/vrx/wave_gazebo/world_models/ocean_waves/model.xacro
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/ubuntu/vrx_ws/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "xacro: generating worlds/scan_and_dock.world from worlds/scan_and_dock.world.xacro"
	cd /home/ubuntu/vrx_ws/src/vrx/vrx_2019 && /home/ubuntu/vrx_ws/build/catkin_generated/env_cached.sh xacro -o /home/ubuntu/vrx_ws/build/vrx/vrx_2019/worlds/scan_and_dock.world worlds/scan_and_dock.world.xacro

vrx/vrx_2019/worlds/stationkeeping_task.world: /home/ubuntu/vrx_ws/src/vrx/vrx_2019/worlds/stationkeeping_task.world.xacro
vrx/vrx_2019/worlds/stationkeeping_task.world: /home/ubuntu/vrx_ws/src/vrx/vrx_gazebo/worlds/xacros/usv_wind_plugin.xacro
vrx/vrx_2019/worlds/stationkeeping_task.world: /home/ubuntu/vrx_ws/src/vrx/wave_gazebo/world_models/ocean_waves/model.xacro
vrx/vrx_2019/worlds/stationkeeping_task.world: /home/ubuntu/vrx_ws/src/vrx/vrx_2019/worlds/sandisland.xacro
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/ubuntu/vrx_ws/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "xacro: generating worlds/stationkeeping_task.world from worlds/stationkeeping_task.world.xacro"
	cd /home/ubuntu/vrx_ws/src/vrx/vrx_2019 && /home/ubuntu/vrx_ws/build/catkin_generated/env_cached.sh xacro -o /home/ubuntu/vrx_ws/build/vrx/vrx_2019/worlds/stationkeeping_task.world worlds/stationkeeping_task.world.xacro

vrx/vrx_2019/worlds/wayfinding_task.world: /home/ubuntu/vrx_ws/src/vrx/vrx_2019/worlds/wayfinding_task.world.xacro
vrx/vrx_2019/worlds/wayfinding_task.world: /home/ubuntu/vrx_ws/src/vrx/wave_gazebo/world_models/ocean_waves/model.xacro
vrx/vrx_2019/worlds/wayfinding_task.world: /home/ubuntu/vrx_ws/src/vrx/vrx_gazebo/worlds/xacros/usv_wind_plugin.xacro
vrx/vrx_2019/worlds/wayfinding_task.world: /home/ubuntu/vrx_ws/src/vrx/vrx_2019/worlds/sandisland.xacro
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/ubuntu/vrx_ws/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "xacro: generating worlds/wayfinding_task.world from worlds/wayfinding_task.world.xacro"
	cd /home/ubuntu/vrx_ws/src/vrx/vrx_2019 && /home/ubuntu/vrx_ws/build/catkin_generated/env_cached.sh xacro -o /home/ubuntu/vrx_ws/build/vrx/vrx_2019/worlds/wayfinding_task.world worlds/wayfinding_task.world.xacro

vrx/vrx_2019/worlds/wind_test.world: /home/ubuntu/vrx_ws/src/vrx/vrx_2019/worlds/wind_test.world.xacro
vrx/vrx_2019/worlds/wind_test.world: /home/ubuntu/vrx_ws/src/vrx/vrx_2019/worlds/sandisland.xacro
vrx/vrx_2019/worlds/wind_test.world: /home/ubuntu/vrx_ws/src/vrx/vrx_gazebo/worlds/xacros/usv_wind_plugin.xacro
vrx/vrx_2019/worlds/wind_test.world: /home/ubuntu/vrx_ws/src/vrx/wave_gazebo/world_models/ocean_waves/model.xacro
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/ubuntu/vrx_ws/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_10) "xacro: generating worlds/wind_test.world from worlds/wind_test.world.xacro"
	cd /home/ubuntu/vrx_ws/src/vrx/vrx_2019 && /home/ubuntu/vrx_ws/build/catkin_generated/env_cached.sh xacro -o /home/ubuntu/vrx_ws/build/vrx/vrx_2019/worlds/wind_test.world worlds/wind_test.world.xacro

vrx_2019__xacro_auto_generate: vrx/vrx_2019/CMakeFiles/vrx_2019__xacro_auto_generate
vrx_2019__xacro_auto_generate: vrx/vrx_2019/worlds/dock.world
vrx_2019__xacro_auto_generate: vrx/vrx_2019/worlds/example_course.world
vrx_2019__xacro_auto_generate: vrx/vrx_2019/worlds/navigation_task.world
vrx_2019__xacro_auto_generate: vrx/vrx_2019/worlds/ocean.world
vrx_2019__xacro_auto_generate: vrx/vrx_2019/worlds/perception_task.world
vrx_2019__xacro_auto_generate: vrx/vrx_2019/worlds/sandisland.world
vrx_2019__xacro_auto_generate: vrx/vrx_2019/worlds/scan_and_dock.world
vrx_2019__xacro_auto_generate: vrx/vrx_2019/worlds/stationkeeping_task.world
vrx_2019__xacro_auto_generate: vrx/vrx_2019/worlds/wayfinding_task.world
vrx_2019__xacro_auto_generate: vrx/vrx_2019/worlds/wind_test.world
vrx_2019__xacro_auto_generate: vrx/vrx_2019/CMakeFiles/vrx_2019__xacro_auto_generate.dir/build.make

.PHONY : vrx_2019__xacro_auto_generate

# Rule to build all files generated by this target.
vrx/vrx_2019/CMakeFiles/vrx_2019__xacro_auto_generate.dir/build: vrx_2019__xacro_auto_generate

.PHONY : vrx/vrx_2019/CMakeFiles/vrx_2019__xacro_auto_generate.dir/build

vrx/vrx_2019/CMakeFiles/vrx_2019__xacro_auto_generate.dir/clean:
	cd /home/ubuntu/vrx_ws/build/vrx/vrx_2019 && $(CMAKE_COMMAND) -P CMakeFiles/vrx_2019__xacro_auto_generate.dir/cmake_clean.cmake
.PHONY : vrx/vrx_2019/CMakeFiles/vrx_2019__xacro_auto_generate.dir/clean

vrx/vrx_2019/CMakeFiles/vrx_2019__xacro_auto_generate.dir/depend:
	cd /home/ubuntu/vrx_ws/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ubuntu/vrx_ws/src /home/ubuntu/vrx_ws/src/vrx/vrx_2019 /home/ubuntu/vrx_ws/build /home/ubuntu/vrx_ws/build/vrx/vrx_2019 /home/ubuntu/vrx_ws/build/vrx/vrx_2019/CMakeFiles/vrx_2019__xacro_auto_generate.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : vrx/vrx_2019/CMakeFiles/vrx_2019__xacro_auto_generate.dir/depend

