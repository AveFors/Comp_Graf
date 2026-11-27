#include "application.hpp"
#include <imgui.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace application {
namespace {

// ========================================================================
// Данные вершин, матриц и состояния сцены
// ========================================================================
struct Vertex { float position[3]; float normal[3]; };
struct Matrix { float data[16]{}; };
struct SceneSettings {
    int projection = 0;
    float position[3] = {0.0f, 0.0f, 0.0f};
    float rotation[3] = {22.0f, 35.0f, 0.0f};
    float scale[3] = {1.0f, 1.0f, 1.0f};
};

SceneSettings scene;
struct Transform {
    Matrix mvp;
    Matrix normal;
};

static_assert(sizeof(Transform) == 128);

// ========================================================================
// Операции с матрицами 4x4 (хранение по столбцам)
// ========================================================================
Matrix identity() {
    Matrix m;
    m.data[0] = m.data[5] = m.data[10] = m.data[15] = 1.0f;
    return m;
}

Matrix multiply(const Matrix& a, const Matrix& b) {
    Matrix result;
    for (int column = 0; column < 4; ++column)
        for (int row = 0; row < 4; ++row)
            for (int k = 0; k < 4; ++k)
                result.data[column * 4 + row] += a.data[k * 4 + row] * b.data[column * 4 + k];
    return result;
}

// ========================================================================
// Матрицы объекта, камеры и проекции
// ========================================================================
Transform sceneTransform(float aspect) {
    constexpr float pi = 3.14159265358979323846f;
    const float x = scene.rotation[0] * pi / 180.0f;
    const float y = scene.rotation[1] * pi / 180.0f;
    const float z = scene.rotation[2] * pi / 180.0f;
    Matrix rx = identity(), ry = identity(), rz = identity();
    rx.data[5] = rx.data[10] = std::cos(x);
    rx.data[6] = std::sin(x);
    rx.data[9] = -std::sin(x);
    ry.data[0] = ry.data[10] = std::cos(y);
    ry.data[2] = -std::sin(y);
    ry.data[8] = std::sin(y);
    rz.data[0] = rz.data[5] = std::cos(z);
    rz.data[1] = std::sin(z);
    rz.data[4] = -std::sin(z);
    Matrix translation = identity(), scaling = identity(), inverseScaling = identity();
    for (int i = 0; i < 3; ++i) {
        translation.data[12 + i] = scene.position[i];
        scaling.data[i * 5] = scene.scale[i];
        inverseScaling.data[i * 5] = 1.0f / scene.scale[i];
    }
    const Matrix rotation = multiply(rz, multiply(ry, rx));

    // --- Модель: M = перенос * поворот * масштаб ---
    const Matrix model = multiply(translation, multiply(rotation, scaling));

    // --- Камера в точке (0, 0, 5.5) ---
    Matrix view = identity();
    view.data[14] = -5.5f;
    constexpr float nearPlane = 0.1f, farPlane = 100.0f;
    Matrix projection;

    // --- Перспективная или ортографическая проекция; глубина Vulkan [0, 1] ---
    if (scene.projection == 0) {
        const float focal = 1.0f / std::tan(45.0f * pi / 360.0f);
        projection.data[0] = focal / aspect;
        projection.data[5] = -focal;
        projection.data[10] = farPlane / (nearPlane - farPlane);
        projection.data[11] = -1.0f;
        projection.data[14] = farPlane * nearPlane / (nearPlane - farPlane);
    } else {
        constexpr float halfHeight = 2.3f;
        projection.data[0] = 1.0f / (halfHeight * aspect);
        projection.data[5] = -1.0f / halfHeight;
        projection.data[10] = 1.0f / (nearPlane - farPlane);
        projection.data[14] = nearPlane / (nearPlane - farPlane);
        projection.data[15] = 1.0f;
    }

    // --- MVP = P * V * M; нормали = R * inverse(S) ---
    return {multiply(projection, multiply(view, model)), multiply(rotation, inverseScaling)};
}

// ========================================================================
// Ресурсы Vulkan для фигуры
// ========================================================================
VkBuffer vertexBuffer{}, indexBuffer{};
VmaAllocation vertexAllocation{}, indexAllocation{};
VkPipelineLayout pipelineLayout{};
VkPipeline pipeline{};
constexpr uint32_t indexCount = 36;

// ========================================================================
// Проверка ошибок Vulkan
// ========================================================================
void check(VkResult result, const char* operation) {
    if (result != VK_SUCCESS)
        throw std::runtime_error(std::string(operation) + ": VkResult=" + std::to_string(result));
}

// ========================================================================
// Создание буфера и загрузка данных CPU
// ========================================================================
void createBuffer(const void* data, VkDeviceSize size, VkBufferUsageFlags usage,
                  VkBuffer& buffer, VmaAllocation& allocation) {
    auto& context = graphics::internal::context;
    VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    info.size = size;
    info.usage = usage;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VmaAllocationCreateInfo memory{};
    memory.usage = VMA_MEMORY_USAGE_AUTO;
    memory.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
    VmaAllocationInfo allocationInfo{};
    check(vmaCreateBuffer(context.allocator, &info, &memory, &buffer, &allocation, &allocationInfo), "vmaCreateBuffer");
    std::memcpy(allocationInfo.pMappedData, data, static_cast<size_t>(size));
    check(vmaFlushAllocation(context.allocator, allocation, 0, size), "vmaFlushAllocation");
}

// ========================================================================
// Загрузка шейдера SPIR-V
// ========================================================================
VkShaderModule loadShader(const char* filename) {
    const std::string path = std::string(CG_SHADER_DIR) + "/" + filename;
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("Cannot open shader: " + path);
    const auto size = file.tellg();
    if (size <= 0 || static_cast<size_t>(size) % sizeof(uint32_t) != 0)
        throw std::runtime_error("Invalid SPIR-V shader: " + path);
    std::vector<uint32_t> code(static_cast<size_t>(size) / sizeof(uint32_t));
    file.seekg(0);
    if (!file.read(reinterpret_cast<char*>(code.data()), size))
        throw std::runtime_error("Cannot read shader: " + path);
    VkShaderModuleCreateInfo info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    info.codeSize = static_cast<size_t>(size);
    info.pCode = code.data();
    VkShaderModule module{};
    check(vkCreateShaderModule(graphics::internal::context.device, &info, nullptr, &module), "vkCreateShaderModule");
    return module;
}

// ========================================================================
// Создание графического конвейера
// ========================================================================
void createPipeline() {
    auto& context = graphics::internal::context;
    VkShaderModule vertexShader{}, fragmentShader{};
    try {
        vertexShader = loadShader("parallelepiped.vert.spv");
        fragmentShader = loadShader("parallelepiped.frag.spv");

        // --- Матрицы для вершинного шейдера: 128 байт push constants ---
        const VkPushConstantRange constants{VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(Transform)};
        VkPipelineLayoutCreateInfo layout{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        layout.pushConstantRangeCount = 1;
        layout.pPushConstantRanges = &constants;
        check(vkCreatePipelineLayout(context.device, &layout, nullptr, &pipelineLayout), "vkCreatePipelineLayout");

        // --- Вершинный и фрагментный шейдеры ---
        std::array<VkPipelineShaderStageCreateInfo, 2> stages{};
        for (auto& stage : stages) {
            stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stage.pName = "main";
        }
        stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
        stages[0].module = vertexShader;
        stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        stages[1].module = fragmentShader;

        // --- Формат вершины: координаты и нормаль ---
        const VkVertexInputBindingDescription binding{0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX};
        const VkVertexInputAttributeDescription attributes[] = {
            {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, position)},
            {1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal)},
        };
        VkPipelineVertexInputStateCreateInfo input{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        input.vertexBindingDescriptionCount = 1;
        input.pVertexBindingDescriptions = &binding;
        input.vertexAttributeDescriptionCount = 2;
        input.pVertexAttributeDescriptions = attributes;

        // --- Треугольники и правила растеризации ---
        VkPipelineInputAssemblyStateCreateInfo assembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
        assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        VkPipelineViewportStateCreateInfo viewport{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
        viewport.viewportCount = viewport.scissorCount = 1;
        VkPipelineRasterizationStateCreateInfo raster{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        raster.polygonMode = VK_POLYGON_MODE_FILL;
        raster.cullMode = VK_CULL_MODE_NONE;
        raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        raster.lineWidth = 1.0f;
        VkPipelineMultisampleStateCreateInfo multisample{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        // --- Проверка глубины: ближняя поверхность закрывает дальнюю ---
        VkPipelineDepthStencilStateCreateInfo depth{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        depth.depthTestEnable = depth.depthWriteEnable = VK_TRUE;
        depth.depthCompareOp = VK_COMPARE_OP_LESS;

        // --- Запись цвета RGBA без смешивания ---
        VkPipelineColorBlendAttachmentState attachment{};
        attachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                    VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        VkPipelineColorBlendStateCreateInfo blend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        blend.attachmentCount = 1;
        blend.pAttachments = &attachment;

        // --- Размеры viewport и scissor задаются в каждом кадре ---
        const VkDynamicState dynamicStates[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
        dynamic.dynamicStateCount = 2;
        dynamic.pDynamicStates = dynamicStates;

        // --- Сборка настроек в один pipeline ---
        VkGraphicsPipelineCreateInfo info{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        info.stageCount = static_cast<uint32_t>(stages.size());
        info.pStages = stages.data();
        info.pVertexInputState = &input;
        info.pInputAssemblyState = &assembly;
        info.pViewportState = &viewport;
        info.pRasterizationState = &raster;
        info.pMultisampleState = &multisample;
        info.pDepthStencilState = &depth;
        info.pColorBlendState = &blend;
        info.pDynamicState = &dynamic;
        info.layout = pipelineLayout;
        info.renderPass = context.render_pass;
        check(vkCreateGraphicsPipelines(context.device, VK_NULL_HANDLE, 1, &info, nullptr, &pipeline), "vkCreateGraphicsPipelines");
    } catch (...) {
        if (vertexShader) vkDestroyShaderModule(context.device, vertexShader, nullptr);
        if (fragmentShader) vkDestroyShaderModule(context.device, fragmentShader, nullptr);
        throw;
    }
    vkDestroyShaderModule(context.device, vertexShader, nullptr);
    vkDestroyShaderModule(context.device, fragmentShader, nullptr);
}
}

// ========================================================================
// Инициализация геометрии и ресурсов
// ========================================================================
bool initialize() {
    try {

        // --- Параллелепипед 2.4 x 1.5 x 1.0: 24 записи вершин для шести нормалей ---
        constexpr float x = 1.2f, y = 0.75f, z = 0.5f;
        const Vertex vertices[] = {
            {{-x,-y, z},{0,0,1}}, {{ x,-y, z},{0,0,1}}, {{ x, y, z},{0,0,1}}, {{-x, y, z},{0,0,1}},
            {{ x,-y,-z},{0,0,-1}},{{-x,-y,-z},{0,0,-1}},{{-x, y,-z},{0,0,-1}},{{ x, y,-z},{0,0,-1}},
            {{ x,-y, z},{1,0,0}}, {{ x,-y,-z},{1,0,0}}, {{ x, y,-z},{1,0,0}}, {{ x, y, z},{1,0,0}},
            {{-x,-y,-z},{-1,0,0}},{{-x,-y, z},{-1,0,0}},{{-x, y, z},{-1,0,0}},{{-x, y,-z},{-1,0,0}},
            {{-x, y, z},{0,1,0}}, {{ x, y, z},{0,1,0}}, {{ x, y,-z},{0,1,0}}, {{-x, y,-z},{0,1,0}},
            {{-x,-y,-z},{0,-1,0}},{{ x,-y,-z},{0,-1,0}},{{ x,-y, z},{0,-1,0}},{{-x,-y, z},{0,-1,0}},
        };

        // --- Шесть граней по два треугольника: 36 индексов ---
        std::array<uint16_t, indexCount> indices{};
        for (uint16_t face = 0; face < 6; ++face) {
            const uint16_t base = face * 4;
            const uint16_t triangles[] = {base, uint16_t(base + 1), uint16_t(base + 2),
                                         base, uint16_t(base + 2), uint16_t(base + 3)};
            std::memcpy(indices.data() + face * 6, triangles, sizeof(triangles));
        }

        // --- Загрузка геометрии и создание конвейера ---
        createBuffer(vertices, sizeof(vertices), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, vertexBuffer, vertexAllocation);
        createBuffer(indices.data(), sizeof(indices), VK_BUFFER_USAGE_INDEX_BUFFER_BIT, indexBuffer, indexAllocation);
        createPipeline();
        return true;
    } catch (const std::exception& error) {
        std::cerr << "Application initialization: " << error.what() << '\n';
        shutdown();
        return false;
    }
}

// ========================================================================
// Ожидание GPU и освобождение ресурсов
// ========================================================================
void shutdown() {
    auto& context = graphics::internal::context;
    vkQueueWaitIdle(context.graphics_queue);
    if (pipeline) vkDestroyPipeline(context.device, pipeline, nullptr);
    if (pipelineLayout) vkDestroyPipelineLayout(context.device, pipelineLayout, nullptr);
    if (indexBuffer) vmaDestroyBuffer(context.allocator, indexBuffer, indexAllocation);
    if (vertexBuffer) vmaDestroyBuffer(context.allocator, vertexBuffer, vertexAllocation);
    pipeline = VK_NULL_HANDLE;
    pipelineLayout = VK_NULL_HANDLE;
    indexBuffer = vertexBuffer = VK_NULL_HANDLE;
    indexAllocation = vertexAllocation = VK_NULL_HANDLE;
}

// ========================================================================
// Управление объектом через ImGui
// ========================================================================
void update([[maybe_unused]] double time) {
    ImGui::SetNextWindowPos({16,16},ImGuiCond_Always);
    ImGui::SetNextWindowSize({510,0},ImGuiCond_Always);
    ImGui::Begin("Lab 1 - Object controls",nullptr,ImGuiWindowFlags_AlwaysAutoResize |
        ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse);
    ImGui::TextUnformatted("Variant 3: parallelepiped");
    ImGui::TextUnformatted("Dimensions: 2.4 x 1.5 x 1.0");
    ImGui::PushItemWidth(320);
    ImGui::Combo("Projection",&scene.projection,"Perspective\0Orthographic\0");
    const auto flags=ImGuiSliderFlags_AlwaysClamp;
    ImGui::DragFloat3("Position",scene.position,0.02f,-3,3,"%.2f",flags);
    ImGui::DragFloat3("Rotation",scene.rotation,0.5f,-180,180,"%.1f",flags);
    ImGui::DragFloat3("Scale",scene.scale,0.01f,0.1f,3,"%.2f",flags);
    ImGui::PopItemWidth();
    if(ImGui::Button("Reset object")) scene=SceneSettings{};
    ImGui::TextUnformatted("Drag values; Ctrl+click to type.");
    ImGui::Text("%.0f FPS | Esc: exit",ImGui::GetIO().Framerate);
    ImGui::End();
}

// ========================================================================
// Запись команд рисования одного кадра
// ========================================================================
void render(const graphics::internal::FrameData& fd) {
    const auto& context = graphics::internal::context;
    const auto extent = context.swapchain_extent;
    const auto command = fd.command_buffer;

    // --- Начало записи command buffer ---
    check(vkResetCommandBuffer(command, 0), "vkResetCommandBuffer");
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    check(vkBeginCommandBuffer(command, &begin), "vkBeginCommandBuffer");

    // --- Очистка цвета и глубины; начало render pass ---
    VkClearValue clear[2]{};
    clear[0].color = {{0.045f, 0.065f, 0.10f, 1.0f}};
    clear[1].depthStencil = {1.0f, 0};
    VkRenderPassBeginInfo pass{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    pass.renderPass = context.render_pass;
    pass.framebuffer = fd.framebuffer;
    pass.renderArea.extent = extent;
    pass.clearValueCount = 2;
    pass.pClearValues = clear;
    vkCmdBeginRenderPass(command, &pass, VK_SUBPASS_CONTENTS_INLINE);

    // --- Область рисования по текущему размеру окна ---
    const VkViewport viewport{0, 0, float(extent.width), float(extent.height), 0, 1};
    const VkRect2D scissor{{0, 0}, extent};
    vkCmdSetViewport(command, 0, 1, &viewport);
    vkCmdSetScissor(command, 0, 1, &scissor);

    // --- Подключение конвейера, вершин и индексов ---
    vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    const VkDeviceSize offset = 0;
    vkCmdBindVertexBuffers(command, 0, 1, &vertexBuffer, &offset);
    vkCmdBindIndexBuffer(command, indexBuffer, 0, VK_INDEX_TYPE_UINT16);

    // --- Передача актуальных матриц на GPU ---
    const Transform transform = sceneTransform(float(extent.width) / float(extent.height));
    vkCmdPushConstants(command, pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(transform), &transform);

    // --- Рисование 12 треугольников и завершение записи ---
    vkCmdDrawIndexed(command, indexCount, 1, 0, 0, 0);
    vkCmdEndRenderPass(command);
    check(vkEndCommandBuffer(command), "vkEndCommandBuffer");
}
}
