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

# Utility rule file for cora_description__xacro_auto_generate_to_devel_space_.

# Include the progress variables for this target.
include vrx/cora_description/CMakeFiles/cora_description__xacro_auto_generate_to_devel_space_.dir/progress.make

vrx/cora_description/CMakeFiles/cora_description__xacro_auto_generate_to_devel_space_: /home/ubuntu/vrx_ws/devel/share/cora_description/urdf/cora.urdf


/home/ubuntu/vrx_ws/devel/share/cora_description/urdf/cora.urdf: /home/ubuntu/vrx_ws/devel/share/cora_description/urdf
/home/ubuntu/vrx_ws/devel/share/cora_description/urdf/cora.urdf: vrx/cora_description/urdf/cora.urdf
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/ubuntu/vrx_ws/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Copying to devel space: /home/ubuntu/vrx_ws/devel/share/cora_description/urdf/cora.urdf"
	cd /home/ubuntu/vrx_ws/build/vrx/cora_description && /usr/bin/cmake -E copy_if_different /home/ubuntu/vrx_ws/build/vrx/cora_description/urdf/cora.urdf /home/ubuntu/vrx_ws/devel/share/cora_description/urdf/cora.urdf

/home/ubuntu/vrx_ws/devel/share/cora_description/urdf:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/ubuntu/vrx_ws/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "creating dir /home/ubuntu/vrx_ws/devel/share/cora_description/urdf"
	cd /home/ubuntu/vrx_ws/build/vrx/cora_description && /usr/bin/cmake -E make_directory /home/ubuntu/vrx_ws/devel/share/cora_description/urdf

vrx/cora_description/urdf/cora.urdf: /home/ubuntu/vrx_ws/src/vrx/cora_description/urdf/cora.urdf.xacro
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/ubuntu/vrx_ws/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "xacro: generating urdf/cora.urdf from urdf/cora.urdf.xacro"
	cd /home/ubuntu/vrx_ws/src/vrx/cora_description && /home/ubuntu/vrx_ws/build/catkin_generated/env_cached.sh xacro -o /home/ubuntu/vrx_ws/build/vrx/cora_description/urdf/cora.urdf urdf/cora.urdf.xacro

cora_description__xacro_auto_generate_to_devel_space_: vrx/cora_description/CMakeFiles/cora_description__xacro_auto_generate_to_devel_space_
cora_description__xacro_auto_generate_to_devel_space_: /home/ubuntu/vrx_ws/devel/share/cora_description/urdf/cora.urdf
cora_description__xacro_auto_generate_to_devel_space_: /home/ubuntu/vrx_ws/devel/share/cora_description/urdf
cora_description__xacro_auto_generate_to_devel_space_: vrx/cora_description/urdf/cora.urdf
cora_description__xacro_auto_generate_to_devel_space_: vrx/cora_description/CMakeFiles/cora_description__xacro_auto_generate_to_devel_space_.dir/build.make

.PHONY : cora_description__xacro_auto_generate_to_devel_space_

# Rule to build all files generated by this target.
vrx/cora_description/CMakeFiles/cora_description__xacro_auto_generate_to_devel_space_.dir/build: cora_description__xacro_auto_generate_to_devel_space_

.PHONY : vrx/cora_description/CMakeFiles/cora_description__xacro_auto_generate_to_devel_space_.dir/build

vrx/cora_description/CMakeFiles/cora_description__xacro_auto_generate_to_devel_space_.dir/clean:
	cd /home/ubuntu/vrx_ws/build/vrx/cora_description && $(CMAKE_COMMAND) -P CMakeFiles/cora_description__xacro_auto_generate_to_devel_space_.dir/cmake_clean.cmake
.PHONY : vrx/cora_description/CMakeFiles/cora_description__xacro_auto_generate_to_devel_space_.dir/clean

vrx/cora_description/CMakeFiles/cora_description__xacro_auto_generate_to_devel_space_.dir/depend:
	cd /home/ubuntu/vrx_ws/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ubuntu/vrx_ws/src /home/ubuntu/vrx_ws/src/vrx/cora_description /home/ubuntu/vrx_ws/build /home/ubuntu/vrx_ws/build/vrx/cora_description /home/ubuntu/vrx_ws/build/vrx/cora_description/CMakeFiles/cora_description__xacro_auto_generate_to_devel_space_.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : vrx/cora_description/CMakeFiles/cora_description__xacro_auto_generate_to_devel_space_.dir/depend

