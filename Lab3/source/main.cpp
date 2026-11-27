#include <cstdint>
#include <climits>
#include <iostream>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include "graphics_internal.hpp"
#include "application.hpp"

namespace {
constexpr int32_t default_window_width = 1280;
constexpr int32_t default_window_height = 720;
constexpr char default_window_title[] = "CG Lab 3 - Texture mapping";
GLFWwindow* glfw_window;
}

int main() {
	int status = EXIT_SUCCESS;

	// ========================================================================
	// Создание окна GLFW без OpenGL
	// ========================================================================
	if (!glfwInit()) {
		std::cerr << "Failed to initialize GLFW\n";
		return EXIT_FAILURE;
	}
	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
	glfw_window = glfwCreateWindow(default_window_width, default_window_height,
	                               default_window_title, nullptr, nullptr);
	if (glfw_window == nullptr) {
		status = EXIT_FAILURE;
		goto err_null_window;
	}

	// ========================================================================
	// Обработка изменения размера окна
	// ========================================================================
	glfwSetFramebufferSizeCallback(glfw_window, [](GLFWwindow*, int width, int height){
		if (width == 0 || height == 0) {
			return;
		}
		graphics::internal::resize(width, height);
	});

	// ========================================================================
	// Инициализация ImGui, Vulkan и ресурсов лабораторной
	// ========================================================================
	if (ImGui::CreateContext() == nullptr) {
		std::cerr << "Failed to create ImGUI context\n";
		status = EXIT_FAILURE;
		goto err_imgui_init;
	}
	ImGui::StyleColorsDark();
	// --- Более крупный шрифт и элементы интерфейса ---
	{
		ImFontConfig font;
		font.SizePixels = 18.0f;
		ImGui::GetIO().Fonts->AddFontDefault(&font);
		ImGui::GetStyle().ScaleAllSizes(1.4f);
	}
	if (!ImGui_ImplGlfw_InitForVulkan(glfw_window, true)) {
		std::cerr << "Failed to initialize ImGUI GLFW backend for Vulkan renderer\n";
		status = EXIT_FAILURE;
		goto err_imgui_glfw_init;
	}
	if (!graphics::internal::initialize(glfw_window)) {
		std::cerr << "Failed to initialize graphics\n";
		status = EXIT_FAILURE;
		goto err_graphics_init;
	}
	if (!application::initialize()) {
		std::cerr << "Failed to initialize application\n";
		status = EXIT_FAILURE;
		goto err_application_init;
	}

	// ========================================================================
	// Кадровый цикл
	// ========================================================================
	while (!glfwWindowShouldClose(glfw_window)) {
		const double time = glfwGetTime();

		// --- События, выход по Esc и ожидание при сворачивании ---
		glfwPollEvents();
		if (glfwGetKey(glfw_window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
			glfwSetWindowShouldClose(glfw_window, GLFW_TRUE);
			break;
		}
		int width = 0, height = 0;
		glfwGetFramebufferSize(glfw_window, &width, &height);
		if (width == 0 || height == 0) {
			glfwWaitEventsTimeout(0.1);
			continue;
		}

		// --- Обновление сцены и подготовка интерфейса ---
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();
		application::update(time);
		ImGui::Render();

		// --- Получение изображения, запись команд, отправка и показ ---
		try {
			graphics::internal::FrameData fd = graphics::internal::prepare();
			if (!fd.command_buffer) {
				status = glfwWindowShouldClose(glfw_window) ? EXIT_SUCCESS : EXIT_FAILURE;
				break;
			}
			application::render(fd);
			graphics::internal::submitAndPresent();
		} catch (const std::exception& error) {
			std::cerr << "Rendering failed: " << error.what() << '\n';
			status = EXIT_FAILURE;
			break;
		}
	}

	// ========================================================================
	// Освобождение ресурсов в обратном порядке
	// ========================================================================
	application::shutdown();
err_application_init:
	graphics::internal::shutdown();
err_graphics_init:
	ImGui_ImplGlfw_Shutdown();
err_imgui_glfw_init:
	ImGui::DestroyContext();
err_imgui_init:
	glfwDestroyWindow(glfw_window);
err_null_window:
	glfwTerminate();
	return status;
}
