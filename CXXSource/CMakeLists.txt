cmake_minimum_required(VERSION 3.15)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED true)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(libtorch_cpp VERSION 1.0.0)

include_directories(
	./include
	)
include_directories(
	/usr/local/include
	/opt/local/include
	/opt/homebrew/include
	)

link_directories(
        /usr/local/lib
        /opt/local/lib
        /opt/homebrew/lib
    )

file(
	GLOB srcs
	./*.cpp
	./include/*.h
	./include/*.hpp
	./src/*.cpp
	)

add_executable(c ${srcs})

find_package(PkgConfig REQUIRED)
pkg_check_modules(fmt_dep REQUIRED IMPORTED_TARGET fmt)
pkg_check_modules(opencv REQUIRED IMPORTED_TARGET opencv4)
find_package(Torch REQUIRED)

target_include_directories(c PUBLIC
	${PROJECT_BINARY_DIR}
	${PROJECT_SOURCE_DIR}/include
	)

target_link_libraries(c PUBLIC
	PkgConfig::fmt_dep
	PkgConfig::opencv	
	Torch
	${TORCH_LIBRARIES}
	${OpenCV_LIBS}
	)
