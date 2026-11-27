# ========================================================================
# Проект и исходные файлы
# ========================================================================
cmake_minimum_required(VERSION 3.20)
include(FetchContent)
project(vulkan-starter-app LANGUAGES C CXX)
add_executable(${PROJECT_NAME} source/main.cpp source/graphics_internal.cpp source/application.cpp)

# ========================================================================
# Параметры компилятора и стандарт C++20
# ========================================================================
if(LINUX)
	target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -Wno-missing-field-initializers)
elseif(APPLE)
	target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -Wno-missing-field-initializers)
elseif(WIN32)
	if(MSVC)
		target_compile_options(${PROJECT_NAME} PRIVATE /W4 /wd4201)
	else()
		target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -Wno-missing-field-initializers)
	endif()
	target_compile_definitions(${PROJECT_NAME} PRIVATE -D_USE_MATH_DEFINES)
else()
	message(FATAL_ERROR "${CMAKE_SYSTEM_NAME} is not supported")
endif()
set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD_REQUIRED TRUE CXX_STANDARD 20)

# ========================================================================
# Сторонние библиотеки: GLFW, vk-bootstrap, VMA, ImGui
# ========================================================================
FetchContent_Declare(
	glfw
	GIT_REPOSITORY https://github.com/glfw/glfw.git
	GIT_TAG 3.5.1
)
FetchContent_Declare(
	vk-bootstrap
	GIT_REPOSITORY https://github.com/charles-lunarg/vk-bootstrap.git
	GIT_TAG v1.4.321
)
FetchContent_Declare(
	VulkanMemoryAllocator
	GIT_REPOSITORY https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator.git
	GIT_TAG v3.4.0
)
FetchContent_Declare(
	imgui
	GIT_REPOSITORY https://github.com/ocornut/imgui.git
	GIT_TAG v1.92.9
)
find_package(Vulkan REQUIRED)
set(GLFW_LIBRARY_TYPE STATIC)
set(GLFW_BUILD_EXAMPLES OFF)
set(GLFW_BUILD_TESTS OFF)
set(GLFW_BUILD_DOCS OFF)

# Используем уже скачанные исходники, если рядом находится Lab1; иначе FetchContent скачает их.
foreach(dependency glfw vk-bootstrap VulkanMemoryAllocator imgui)
    string(TOLOWER "${dependency}" folder)
    string(TOUPPER "${dependency}" key)
    set(cached "${CMAKE_CURRENT_SOURCE_DIR}/../Lab1/build-debug/_deps/${folder}-src")
    if(EXISTS "${cached}")
        set(FETCHCONTENT_SOURCE_DIR_${key} "${cached}")
    endif()
endforeach()

FetchContent_MakeAvailable(
	glfw
	vk-bootstrap
	VulkanMemoryAllocator
	imgui
)

# ========================================================================
# Подключение библиотек и исходников ImGui
# ========================================================================
target_link_libraries(${PROJECT_NAME} PRIVATE
	glfw
	Vulkan::Vulkan
	vk-bootstrap::vk-bootstrap
	GPUOpen::VulkanMemoryAllocator
)
target_include_directories(${PROJECT_NAME} PRIVATE ${imgui_SOURCE_DIR} ${imgui_SOURCE_DIR}/backends)
target_sources(${PROJECT_NAME} PRIVATE
	${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
	${imgui_SOURCE_DIR}/backends/imgui_impl_vulkan.cpp
	${imgui_SOURCE_DIR}/imgui.cpp
	${imgui_SOURCE_DIR}/imgui_draw.cpp
	${imgui_SOURCE_DIR}/imgui_tables.cpp
	${imgui_SOURCE_DIR}/imgui_widgets.cpp
	${imgui_SOURCE_DIR}/imgui_demo.cpp
)

# ========================================================================
# Компиляция шейдеров GLSL в SPIR-V
# ========================================================================
find_program(GLSLC_FOUND NAMES glslc HINTS "$ENV{VULKAN_SDK}/Bin")
if(NOT GLSLC_FOUND)
	message(FATAL_ERROR "glslc not found. Install Vulkan SDK and set VULKAN_SDK.")
endif()
if(GLSLC_FOUND)
	set(_SHADER_BINARIES)
	file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/shaders")
	target_compile_definitions(${PROJECT_NAME} PRIVATE CG_SHADER_DIR="${CMAKE_CURRENT_BINARY_DIR}/shaders")
	macro(compile_shader SHADER_FILE)
		set(SHADER_SOURCE ${CMAKE_SOURCE_DIR}/shaders/${SHADER_FILE})
		set(SHADER_BINARY ${SHADER_FILE}.spv)
		set(SHADER_BINARY_PATH ${CMAKE_CURRENT_BINARY_DIR}/shaders/${SHADER_BINARY})
		add_custom_command(
			OUTPUT ${SHADER_BINARY_PATH}
			COMMAND "${GLSLC_FOUND}" "${SHADER_SOURCE}" -o "${SHADER_BINARY_PATH}"
			DEPENDS ${SHADER_SOURCE}
			VERBATIM
			COMMENT "Compiling ${SHADER_FILE} shader"
		)
		list(APPEND _SHADER_BINARIES ${SHADER_BINARY_PATH})
	endmacro()
	compile_shader(lighting.vert)
	compile_shader(lighting.frag)
	add_custom_target(shaders DEPENDS ${_SHADER_BINARIES})
	add_dependencies(${PROJECT_NAME} shaders)
endif()
