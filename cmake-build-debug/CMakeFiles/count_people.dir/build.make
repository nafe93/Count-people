# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.14

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
CMAKE_COMMAND = /snap/clion/69/bin/cmake/linux/bin/cmake

# The command to remove a file.
RM = /snap/clion/69/bin/cmake/linux/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/C++/computerVision/opencv/count_people

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/C++/computerVision/opencv/count_people/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/count_people.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/count_people.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/count_people.dir/flags.make

CMakeFiles/count_people.dir/main.cpp.o: CMakeFiles/count_people.dir/flags.make
CMakeFiles/count_people.dir/main.cpp.o: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/C++/computerVision/opencv/count_people/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/count_people.dir/main.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/count_people.dir/main.cpp.o -c /home/C++/computerVision/opencv/count_people/main.cpp

CMakeFiles/count_people.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/count_people.dir/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/C++/computerVision/opencv/count_people/main.cpp > CMakeFiles/count_people.dir/main.cpp.i

CMakeFiles/count_people.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/count_people.dir/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/C++/computerVision/opencv/count_people/main.cpp -o CMakeFiles/count_people.dir/main.cpp.s

# Object files for target count_people
count_people_OBJECTS = \
"CMakeFiles/count_people.dir/main.cpp.o"

# External object files for target count_people
count_people_EXTERNAL_OBJECTS =

count_people: CMakeFiles/count_people.dir/main.cpp.o
count_people: CMakeFiles/count_people.dir/build.make
count_people: /usr/local/lib/libopencv_gapi.so.4.1.0
count_people: /usr/local/lib/libopencv_stitching.so.4.1.0
count_people: /usr/local/lib/libopencv_aruco.so.4.1.0
count_people: /usr/local/lib/libopencv_bgsegm.so.4.1.0
count_people: /usr/local/lib/libopencv_bioinspired.so.4.1.0
count_people: /usr/local/lib/libopencv_ccalib.so.4.1.0
count_people: /usr/local/lib/libopencv_dnn_objdetect.so.4.1.0
count_people: /usr/local/lib/libopencv_dpm.so.4.1.0
count_people: /usr/local/lib/libopencv_face.so.4.1.0
count_people: /usr/local/lib/libopencv_freetype.so.4.1.0
count_people: /usr/local/lib/libopencv_fuzzy.so.4.1.0
count_people: /usr/local/lib/libopencv_hfs.so.4.1.0
count_people: /usr/local/lib/libopencv_img_hash.so.4.1.0
count_people: /usr/local/lib/libopencv_line_descriptor.so.4.1.0
count_people: /usr/local/lib/libopencv_quality.so.4.1.0
count_people: /usr/local/lib/libopencv_reg.so.4.1.0
count_people: /usr/local/lib/libopencv_rgbd.so.4.1.0
count_people: /usr/local/lib/libopencv_saliency.so.4.1.0
count_people: /usr/local/lib/libopencv_stereo.so.4.1.0
count_people: /usr/local/lib/libopencv_structured_light.so.4.1.0
count_people: /usr/local/lib/libopencv_superres.so.4.1.0
count_people: /usr/local/lib/libopencv_surface_matching.so.4.1.0
count_people: /usr/local/lib/libopencv_tracking.so.4.1.0
count_people: /usr/local/lib/libopencv_videostab.so.4.1.0
count_people: /usr/local/lib/libopencv_xfeatures2d.so.4.1.0
count_people: /usr/local/lib/libopencv_xobjdetect.so.4.1.0
count_people: /usr/local/lib/libopencv_xphoto.so.4.1.0
count_people: /usr/local/lib/libopencv_shape.so.4.1.0
count_people: /usr/local/lib/libopencv_datasets.so.4.1.0
count_people: /usr/local/lib/libopencv_plot.so.4.1.0
count_people: /usr/local/lib/libopencv_text.so.4.1.0
count_people: /usr/local/lib/libopencv_dnn.so.4.1.0
count_people: /usr/local/lib/libopencv_highgui.so.4.1.0
count_people: /usr/local/lib/libopencv_ml.so.4.1.0
count_people: /usr/local/lib/libopencv_phase_unwrapping.so.4.1.0
count_people: /usr/local/lib/libopencv_optflow.so.4.1.0
count_people: /usr/local/lib/libopencv_ximgproc.so.4.1.0
count_people: /usr/local/lib/libopencv_video.so.4.1.0
count_people: /usr/local/lib/libopencv_videoio.so.4.1.0
count_people: /usr/local/lib/libopencv_imgcodecs.so.4.1.0
count_people: /usr/local/lib/libopencv_objdetect.so.4.1.0
count_people: /usr/local/lib/libopencv_calib3d.so.4.1.0
count_people: /usr/local/lib/libopencv_features2d.so.4.1.0
count_people: /usr/local/lib/libopencv_flann.so.4.1.0
count_people: /usr/local/lib/libopencv_photo.so.4.1.0
count_people: /usr/local/lib/libopencv_imgproc.so.4.1.0
count_people: /usr/local/lib/libopencv_core.so.4.1.0
count_people: CMakeFiles/count_people.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/C++/computerVision/opencv/count_people/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable count_people"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/count_people.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/count_people.dir/build: count_people

.PHONY : CMakeFiles/count_people.dir/build

CMakeFiles/count_people.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/count_people.dir/cmake_clean.cmake
.PHONY : CMakeFiles/count_people.dir/clean

CMakeFiles/count_people.dir/depend:
	cd /home/C++/computerVision/opencv/count_people/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/C++/computerVision/opencv/count_people /home/C++/computerVision/opencv/count_people /home/C++/computerVision/opencv/count_people/cmake-build-debug /home/C++/computerVision/opencv/count_people/cmake-build-debug /home/C++/computerVision/opencv/count_people/cmake-build-debug/CMakeFiles/count_people.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/count_people.dir/depend

