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

# Utility rule file for wave_gazebo__xacro_auto_generate.

# Include the progress variables for this target.
include vrx/wave_gazebo/CMakeFiles/wave_gazebo__xacro_auto_generate.dir/progress.make

vrx/wave_gazebo/CMakeFiles/wave_gazebo__xacro_auto_generate: vrx/wave_gazebo/worlds/ocean.world
vrx/wave_gazebo/CMakeFiles/wave_gazebo__xacro_auto_generate: vrx/wave_gazebo/worlds/ocean_buoys.world
vrx/wave_gazebo/CMakeFiles/wave_gazebo__xacro_auto_generate: vrx/wave_gazebo/worlds/ocean_wamv.world


vrx/wave_gazebo/worlds/ocean.world: /home/ubuntu/vrx_ws/src/vrx/wave_gazebo/worlds/ocean.world.xacro
vrx/wave_gazebo/worlds/ocean.world: /home/ubuntu/vrx_ws/src/vrx/wave_gazebo/world_models/ocean_waves/model.xacro
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/ubuntu/vrx_ws/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "xacro: generating worlds/ocean.world from worlds/ocean.world.xacro"
	cd /home/ubuntu/vrx_ws/src/vrx/wave_gazebo && /home/ubuntu/vrx_ws/build/catkin_generated/env_cached.sh xacro -o /home/ubuntu/vrx_ws/build/vrx/wave_gazebo/worlds/ocean.world worlds/ocean.world.xacro

vrx/wave_gazebo/worlds/ocean_buoys.world: /home/ubuntu/vrx_ws/src/vrx/wave_gazebo/worlds/ocean_buoys.world.xacro
vrx/wave_gazebo/worlds/ocean_buoys.world: /home/ubuntu/vrx_ws/src/vrx/wave_gazebo/world_models/ocean_waves/model.xacro
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/ubuntu/vrx_ws/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "xacro: generating worlds/ocean_buoys.world from worlds/ocean_buoys.world.xacro"
	cd /home/ubuntu/vrx_ws/src/vrx/wave_gazebo && /home/ubuntu/vrx_ws/build/catkin_generated/env_cached.sh xacro -o /home/ubuntu/vrx_ws/build/vrx/wave_gazebo/worlds/ocean_buoys.world worlds/ocean_buoys.world.xacro

vrx/wave_gazebo/worlds/ocean_wamv.world: /home/ubuntu/vrx_ws/src/vrx/wave_gazebo/worlds/ocean_wamv.world.xacro
vrx/wave_gazebo/worlds/ocean_wamv.world: /home/ubuntu/vrx_ws/src/vrx/wave_gazebo/world_models/ocean_waves/model.xacro
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/ubuntu/vrx_ws/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "xacro: generating worlds/ocean_wamv.world from worlds/ocean_wamv.world.xacro"
	cd /home/ubuntu/vrx_ws/src/vrx/wave_gazebo && /home/ubuntu/vrx_ws/build/catkin_generated/env_cached.sh xacro -o /home/ubuntu/vrx_ws/build/vrx/wave_gazebo/worlds/ocean_wamv.world worlds/ocean_wamv.world.xacro

/home/ubuntu/vrx_ws/src/vrx/wave_gazebo/world_models/ocean_waves/model.xacro: /home/ubuntu/vrx_ws/src/vrx/wave_gazebo/world_models/ocean_waves/model.xacro.erb
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/ubuntu/vrx_ws/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Generating /home/ubuntu/vrx_ws/src/vrx/wave_gazebo/world_models/ocean_waves/model.xacro"
	cd /home/ubuntu/vrx_ws/src/vrx/wave_gazebo && /usr/bin/erb world_models/ocean_waves/model.xacro.erb > /home/ubuntu/vrx_ws/src/vrx/wave_gazebo/world_models/ocean_waves/model.xacro

wave_gazebo__xacro_auto_generate: vrx/wave_gazebo/CMakeFiles/wave_gazebo__xacro_auto_generate
wave_gazebo__xacro_auto_generate: vrx/wave_gazebo/worlds/ocean.world
wave_gazebo__xacro_auto_generate: vrx/wave_gazebo/worlds/ocean_buoys.world
wave_gazebo__xacro_auto_generate: vrx/wave_gazebo/worlds/ocean_wamv.world
wave_gazebo__xacro_auto_generate: /home/ubuntu/vrx_ws/src/vrx/wave_gazebo/world_models/ocean_waves/model.xacro
wave_gazebo__xacro_auto_generate: vrx/wave_gazebo/CMakeFiles/wave_gazebo__xacro_auto_generate.dir/build.make

.PHONY : wave_gazebo__xacro_auto_generate

# Rule to build all files generated by this target.
vrx/wave_gazebo/CMakeFiles/wave_gazebo__xacro_auto_generate.dir/build: wave_gazebo__xacro_auto_generate

.PHONY : vrx/wave_gazebo/CMakeFiles/wave_gazebo__xacro_auto_generate.dir/build

vrx/wave_gazebo/CMakeFiles/wave_gazebo__xacro_auto_generate.dir/clean:
	cd /home/ubuntu/vrx_ws/build/vrx/wave_gazebo && $(CMAKE_COMMAND) -P CMakeFiles/wave_gazebo__xacro_auto_generate.dir/cmake_clean.cmake
.PHONY : vrx/wave_gazebo/CMakeFiles/wave_gazebo__xacro_auto_generate.dir/clean

vrx/wave_gazebo/CMakeFiles/wave_gazebo__xacro_auto_generate.dir/depend:
	cd /home/ubuntu/vrx_ws/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ubuntu/vrx_ws/src /home/ubuntu/vrx_ws/src/vrx/wave_gazebo /home/ubuntu/vrx_ws/build /home/ubuntu/vrx_ws/build/vrx/wave_gazebo /home/ubuntu/vrx_ws/build/vrx/wave_gazebo/CMakeFiles/wave_gazebo__xacro_auto_generate.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : vrx/wave_gazebo/CMakeFiles/wave_gazebo__xacro_auto_generate.dir/depend

