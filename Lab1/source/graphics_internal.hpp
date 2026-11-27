#pragma once
#include <cstdint>
#include <vulkan/vulkan_core.h>
#include <vk_mem_alloc.h>

struct GLFWwindow;
namespace graphics::internal {

// ========================================================================
// Общие объекты Vulkan, доступные коду лабораторной
// ========================================================================
struct Context {
	VkPhysicalDevice physical_device;
	VkDevice device;
	VmaAllocator allocator;
	VkQueue graphics_queue;
	uint32_t graphics_queue_index;
	VkFormat swapchain_format;
	VkExtent2D swapchain_extent;
	VkRenderPass render_pass;
};

// ========================================================================
// Изображения и буфер команд текущего кадра
// ========================================================================
struct FrameData {
	VkFramebuffer framebuffer;
	VkCommandBuffer command_buffer;
};

extern Context context;

// ========================================================================
// Управление инфраструктурой Vulkan и кадрами
// ========================================================================
bool initialize(GLFWwindow* const window);
void shutdown();
void resize(uint32_t width, uint32_t height);
FrameData prepare();
void submitAndPresent();
}
