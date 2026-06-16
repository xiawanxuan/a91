cmake_minimum_required(VERSION 3.20)
project(HairFurSim LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

find_package(CUDA REQUIRED)
find_package(OpenGL REQUIRED)

set(GLFW_DIR ${CMAKE_SOURCE_DIR}/thirdparty/glfw)
set(GLAD_DIR ${CMAKE_SOURCE_DIR}/thirdparty/glad)
set(IMGUI_DIR ${CMAKE_SOURCE_DIR}/thirdparty/imgui)
set(GLM_DIR ${CMAKE_SOURCE_DIR}/thirdparty/glm)
set(ALEMBIC_DIR ${CMAKE_SOURCE_DIR}/thirdparty/alembic)

include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CUDA_INCLUDE_DIRS}
    ${GLFW_DIR}/include
    ${GLAD_DIR}/include
    ${IMGUI_DIR}
    ${IMGUI_DIR}/backends
    ${GLM_DIR}
    ${ALEMBIC_DIR}/include
)

file(GLOB_RECURSE HEADER_FILES
    ${CMAKE_SOURCE_DIR}/include/*.h
    ${CMAKE_SOURCE_DIR}/include/*.hpp
    ${CMAKE_SOURCE_DIR}/include/*.cuh
)

file(GLOB_RECURSE SOURCE_FILES
    ${CMAKE_SOURCE_DIR}/src/*.cpp
    ${CMAKE_SOURCE_DIR}/src/*.cu
)

set(IMGUI_SOURCES
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/imgui_demo.cpp
    ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
    ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp
)

set(GLAD_SOURCES
    ${GLAD_DIR}/src/glad.c
)

add_executable(HairFurSim
    ${SOURCE_FILES}
    ${HEADER_FILES}
    ${IMGUI_SOURCES}
    ${GLAD_SOURCES}
)

set_target_properties(HairFurSim PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "86;89;90"
)

target_link_libraries(HairFurSim
    ${CUDA_LIBRARIES}
    ${OPENGL_LIBRARIES}
    ${GLFW_DIR}/lib/glfw3.lib
    ${ALEMBIC_DIR}/lib/Alembic.lib
)

if(MSVC)
    target_compile_options(HairFurSim PRIVATE /W4)
    set_target_properties(HairFurSim PROPERTIES
        LINK_FLAGS "/SUBSYSTEM:WINDOWS"
    )
endif()

add_custom_command(TARGET HairFurSim POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_SOURCE_DIR}/shaders
    $<TARGET_FILE_DIR:HairFurSim>/shaders
)
