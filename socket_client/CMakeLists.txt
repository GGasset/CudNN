# CMakeLists.txt
cmake_minimum_required (VERSION 3.0)

# Add linking to stdlib
SET(GCC_COVERAGE_COMPILE_FLAGS "-lstdc++")
SET(GCC_COVERAGE_LINK_FLAGS    "-lstdc++")

# Setup projectname
project (NN_client)

file(GLOB sourcefiles
	*.h
	*.cpp
	test_main/*.cpp
)

message("Executable for testing only")

add_executable(${PROJECT_NAME}
	${sourcefiles}
)
