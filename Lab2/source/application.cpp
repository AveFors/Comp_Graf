#include "application.hpp"
#include "scene_math.hpp"
#include <imgui.h>
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace application {
namespace {
using math::Vec3;
using math::Matrix;
struct Vertex { Vec3 position; Vec3 normal; };

// ========================================================================
// Параметры камеры, направленного света, материала и объекта
// ========================================================================
struct Parameters {
    float cameraPosition[3]={3.6f,2.6f,6.0f};
    float cameraRotation[3]={-20.4f,31.0f,0};
    float fov=45, nearPlane=0.1f, farPlane=100, orthoHeight=5;
    int projection=0;
    float position[3]={0,0,0}, rotation[3]={0,0,0}, scale[3]={1,1,1};
    float lightDirection[3]={-0.4f,-0.8f,-0.6f};
    float lightColor[3]={1,0.94f,0.85f}, lightIntensity=1.2f;
    float ambientColor[3]={1,1,1}, ambientIntensity=0.12f;
    float albedo[3]={0.75f,0.42f,0.18f}, specular[3]={1,1,1}, shininess=32;
};
Parameters settings;

// std140: три mat4 и шесть vec4; структура совпадает с блоком Scene в GLSL.
struct alignas(16) SceneUniform {
    Matrix viewProjection,model,normal;
    std::array<float,4> camera,lightDirection,light,ambient,albedo,specular;
};
static_assert(sizeof(SceneUniform)==288);
VkBuffer vertexBuffer{},indexBuffer{},uniformBuffer{};
VmaAllocation vertexAllocation{},indexAllocation{},uniformAllocation{};
VkPipelineLayout pipelineLayout{};
VkPipeline pipeline{};
VkDescriptorSetLayout descriptorLayout{};
VkDescriptorPool descriptorPool{};
VkDescriptorSet descriptorSet{};
constexpr uint32_t indexCount=36;

SceneUniform sceneUniform(float aspect) {
    const Matrix r=math::rotation(settings.rotation);
    const Matrix model=math::multiply(math::translation(math::vector(settings.position)),
        math::multiply(r,math::scaling(math::vector(settings.scale))));
    const Vec3 eye=math::vector(settings.cameraPosition);
    const Matrix view=math::multiply(math::transpose(math::rotation(settings.cameraRotation)),
        math::translation({-eye.x,-eye.y,-eye.z}));
    const Matrix projection=math::projection(settings.projection,aspect,settings.fov,
        settings.nearPlane,settings.farPlane,settings.orthoHeight);
    const Matrix normal=math::multiply(r,math::scaling({1/settings.scale[0],1/settings.scale[1],1/settings.scale[2]}));
    const Vec3 direction=math::normalized(math::vector(settings.lightDirection));
    return {math::multiply(projection,view),model,normal,
        {eye.x,eye.y,eye.z,1},{direction.x,direction.y,direction.z,0},
        {settings.lightColor[0],settings.lightColor[1],settings.lightColor[2],settings.lightIntensity},
        {settings.ambientColor[0],settings.ambientColor[1],settings.ambientColor[2],settings.ambientIntensity},
        {settings.albedo[0],settings.albedo[1],settings.albedo[2],1},
        {settings.specular[0],settings.specular[1],settings.specular[2],settings.shininess}};
}

// ========================================================================
// Один uniform-буфер и один набор дескрипторов для базового задания
// ========================================================================
void check(VkResult result,const char* operation);
void createDescriptors() {
    auto device=graphics::internal::context.device;
    const VkDescriptorSetLayoutBinding binding{0,VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,1,
        VK_SHADER_STAGE_VERTEX_BIT|VK_SHADER_STAGE_FRAGMENT_BIT,nullptr};
    VkDescriptorSetLayoutCreateInfo layout{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layout.bindingCount=1; layout.pBindings=&binding;
    check(vkCreateDescriptorSetLayout(device,&layout,nullptr,&descriptorLayout),"descriptor layout");
    const VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,1};
    VkDescriptorPoolCreateInfo pool{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pool.maxSets=1; pool.poolSizeCount=1; pool.pPoolSizes=&size;
    check(vkCreateDescriptorPool(device,&pool,nullptr,&descriptorPool),"descriptor pool");
    VkDescriptorSetAllocateInfo allocate{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocate.descriptorPool=descriptorPool; allocate.descriptorSetCount=1; allocate.pSetLayouts=&descriptorLayout;
    check(vkAllocateDescriptorSets(device,&allocate,&descriptorSet),"descriptor set");
    const VkDescriptorBufferInfo buffer{uniformBuffer,0,sizeof(SceneUniform)};
    VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    write.dstSet=descriptorSet; write.dstBinding=0; write.descriptorCount=1;
    write.descriptorType=VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; write.pBufferInfo=&buffer;
    vkUpdateDescriptorSets(device,1,&write,0,nullptr);
}

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
    if (data) std::memcpy(allocationInfo.pMappedData, data, static_cast<size_t>(size));
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
        vertexShader = loadShader("lighting.vert.spv");
        fragmentShader = loadShader("lighting.frag.spv");

        // --- Один набор дескрипторов с uniform-буфером ---
        VkPipelineLayoutCreateInfo layout{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        layout.setLayoutCount = 1;
        layout.pSetLayouts = &descriptorLayout;
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
// Геометрия варианта 3: нормали вычисляются векторным произведением
// ========================================================================
bool initialize() {
    try {
        constexpr float x=1.2f,y=0.75f,z=0.5f;
        const Vec3 corners[]={{-x,-y,z},{x,-y,z},{x,y,z},{-x,y,z},
                              {-x,-y,-z},{x,-y,-z},{x,y,-z},{-x,y,-z}};
        const uint16_t faces[6][4]={{0,1,2,3},{5,4,7,6},{1,5,6,2},{4,0,3,7},{3,2,6,7},{4,5,1,0}};
        std::array<Vertex,24> vertices{};
        std::array<uint16_t,indexCount> indices{};
        for (uint16_t face=0;face<6;++face) {
            const auto& f=faces[face];
            const Vec3 normal=math::normalized(math::cross(corners[f[1]]-corners[f[0]],corners[f[2]]-corners[f[0]]));
            for (int i=0;i<4;++i) vertices[face*4+i]={corners[f[i]],normal};
            const uint16_t base=face*4;
            const uint16_t triangle[]={base,uint16_t(base+1),uint16_t(base+2),base,uint16_t(base+2),uint16_t(base+3)};
            std::memcpy(indices.data()+face*6,triangle,sizeof(triangle));
        }
        createBuffer(vertices.data(),sizeof(vertices),VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,vertexBuffer,vertexAllocation);
        createBuffer(indices.data(),sizeof(indices),VK_BUFFER_USAGE_INDEX_BUFFER_BIT,indexBuffer,indexAllocation);
        createBuffer(nullptr,sizeof(SceneUniform),VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,uniformBuffer,uniformAllocation);
        createDescriptors();
        createPipeline();
        return true;
    } catch (const std::exception& e) {
        std::cerr<<"Application initialization: "<<e.what()<<'\n'; shutdown(); return false;
    }
}

void shutdown() {
    auto& c=graphics::internal::context;
    vkQueueWaitIdle(c.graphics_queue);
    if(pipeline) vkDestroyPipeline(c.device,pipeline,nullptr);
    if(pipelineLayout) vkDestroyPipelineLayout(c.device,pipelineLayout,nullptr);
    if(descriptorPool) vkDestroyDescriptorPool(c.device,descriptorPool,nullptr);
    if(descriptorLayout) vkDestroyDescriptorSetLayout(c.device,descriptorLayout,nullptr);
    if(uniformBuffer) vmaDestroyBuffer(c.allocator,uniformBuffer,uniformAllocation);
    if(indexBuffer) vmaDestroyBuffer(c.allocator,indexBuffer,indexAllocation);
    if(vertexBuffer) vmaDestroyBuffer(c.allocator,vertexBuffer,vertexAllocation);
    pipeline={};pipelineLayout={};descriptorPool={};descriptorLayout={};
    uniformBuffer={};indexBuffer={};vertexBuffer={};
}

// ========================================================================
// Все параметры изменяются через ImGui, без клавиатурных комбинаций
// ========================================================================
void update([[maybe_unused]] double time) {
    ImGui::SetNextWindowPos({12,12},ImGuiCond_Always);
    ImGui::SetNextWindowSize({560,std::max(120.0f,ImGui::GetIO().DisplaySize.y-24)},ImGuiCond_Always);
    ImGui::Begin("Lab 2 - Camera and lighting",nullptr,ImGuiWindowFlags_NoMove|ImGuiWindowFlags_NoResize|ImGuiWindowFlags_NoCollapse);
    ImGui::TextUnformatted("Variant 3: parallelepiped");
    if(ImGui::Button("Reset scene")) settings=Parameters{};
    ImGui::SameLine(); ImGui::Text("%.0f FPS",ImGui::GetIO().Framerate);
    ImGui::PushItemWidth(310);
    const auto flags=ImGuiSliderFlags_AlwaysClamp;
    if(ImGui::CollapsingHeader("Camera",ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::DragFloat3("Eye position",settings.cameraPosition,0.03f,-15,15,"%.2f",flags);
        ImGui::DragFloat3("Eye rotation",settings.cameraRotation,0.3f,-180,180,"%.1f",flags);
        ImGui::Combo("Projection",&settings.projection,"Perspective\0Orthographic\0");
        ImGui::SliderFloat("FOV",&settings.fov,15,110,"%.1f",flags);
        ImGui::SliderFloat("Ortho height",&settings.orthoHeight,1,15,"%.2f",flags);
        ImGui::DragFloat("Near plane",&settings.nearPlane,0.01f,0.01f,2,"%.2f",flags);
        ImGui::DragFloat("Far plane",&settings.farPlane,0.5f,10,200,"%.1f",flags);
    }
    if(ImGui::CollapsingHeader("Directional light",ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::DragFloat3("Ray direction",settings.lightDirection,0.02f,-1,1,"%.2f",flags);
        if(math::dot(math::vector(settings.lightDirection),math::vector(settings.lightDirection))<0.000001f)
            ImGui::TextWrapped("Zero direction: fallback (0, 0, 1) is used.");
        ImGui::ColorEdit3("Light color",settings.lightColor);
        ImGui::SliderFloat("Light intensity",&settings.lightIntensity,0,4,"%.2f",flags);
    }
    if(ImGui::CollapsingHeader("Ambient light",ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::ColorEdit3("Ambient color",settings.ambientColor);
        ImGui::SliderFloat("Ambient intensity",&settings.ambientIntensity,0,1,"%.2f",flags);
    }
    if(ImGui::CollapsingHeader("Material",ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::ColorEdit3("Albedo",settings.albedo);
        ImGui::ColorEdit3("Specular color",settings.specular);
        ImGui::SliderFloat("Shininess",&settings.shininess,1,256,"%.1f",flags);
    }
    if(ImGui::CollapsingHeader("Object",ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::DragFloat3("Position",settings.position,0.02f,-3,3,"%.2f",flags);
        ImGui::DragFloat3("Rotation",settings.rotation,0.5f,-180,180,"%.1f",flags);
        ImGui::DragFloat3("Scale",settings.scale,0.01f,0.1f,3,"%.2f",flags);
    }
    ImGui::PopItemWidth();
    ImGui::TextWrapped("Drag values; Ctrl+click to type. Esc: exit.");
    ImGui::End();
}

// ========================================================================
// Uniform обновляется после ожидания fence в prepare(): нет гонки с GPU
// ========================================================================
void render(const graphics::internal::FrameData& fd) {
    const auto& c=graphics::internal::context;
    const auto extent=c.swapchain_extent;
    const auto uniform=sceneUniform(float(extent.width)/float(extent.height));
    VmaAllocationInfo memory{};
    vmaGetAllocationInfo(c.allocator,uniformAllocation,&memory);
    std::memcpy(memory.pMappedData,&uniform,sizeof(uniform));
    check(vmaFlushAllocation(c.allocator,uniformAllocation,0,sizeof(uniform)),"uniform flush");
    const auto cmd=fd.command_buffer;
    check(vkResetCommandBuffer(cmd,0),"reset command buffer");
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    check(vkBeginCommandBuffer(cmd,&begin),"begin command buffer");
    VkClearValue clear[2]{};
    clear[0].color={{0.025f,0.035f,0.055f,1}}; clear[1].depthStencil={1,0};
    VkRenderPassBeginInfo pass{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    pass.renderPass=c.render_pass;pass.framebuffer=fd.framebuffer;pass.renderArea.extent=extent;
    pass.clearValueCount=2;pass.pClearValues=clear;
    vkCmdBeginRenderPass(cmd,&pass,VK_SUBPASS_CONTENTS_INLINE);
    const VkViewport viewport{0,0,float(extent.width),float(extent.height),0,1};
    const VkRect2D scissor{{0,0},extent};
    vkCmdSetViewport(cmd,0,1,&viewport);vkCmdSetScissor(cmd,0,1,&scissor);
    vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_GRAPHICS,pipeline);
    const VkDeviceSize offset=0;
    vkCmdBindVertexBuffers(cmd,0,1,&vertexBuffer,&offset);
    vkCmdBindIndexBuffer(cmd,indexBuffer,0,VK_INDEX_TYPE_UINT16);
    vkCmdBindDescriptorSets(cmd,VK_PIPELINE_BIND_POINT_GRAPHICS,pipelineLayout,0,1,&descriptorSet,0,nullptr);
    vkCmdDrawIndexed(cmd,indexCount,1,0,0,0);
    vkCmdEndRenderPass(cmd);
    check(vkEndCommandBuffer(cmd),"end command buffer");
}
}
