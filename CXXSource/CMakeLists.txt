cmake_minimum_required(VERSION 3.15)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED true)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(libtorch_cpp VERSION 1.0.0)

include_directories(
	./include
	)
include_directories(
	/opt/homebrew/include
	)

link_directories(
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
find_package(Torch REQUIRED)
find_package(TorchVision REQUIRED)
find_package(OpenCV REQUIRED)

target_include_directories(c PUBLIC
	${PROJECT_BINARY_DIR}
	${PROJECT_SOURCE_DIR}/include
	)

target_link_libraries(c PUBLIC
	PkgConfig::fmt_dep
	Torch
	TorchVision
	${TORCH_LIBRARIES}
	${OpenCV_LIBS}
	)

install(TARGETS c DESTINATION bin)
install(DIRECTORY ${TORCH_INSTALL_PREFIX}/lib/ DESTINATION bin)
install(DIRECTORY ${OpenCV_INSTALL_PATH}/lib/ DESTINATION bin)
