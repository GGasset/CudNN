# CMakeLists.txt
cmake_minimum_required (VERSION 3.0)

# Add linking to stdlib
SET(GCC_COVERAGE_COMPILE_FLAGS "-lstdc++")
SET(GCC_COVERAGE_LINK_FLAGS    "-lstdc++")

# Setup projectname
project (NN_client)

file(GLOB sourcefiles
	*.cpp
	test_main/*.cpp
)

message("Executable for testing only")

add_executable(${PROJECT_NAME}
	${sourcefiles}
)

if(WIN32)
	target_link_libraries(${PROJECT_NAME} PUBLIC wsock32 ws2_32)
endif()
