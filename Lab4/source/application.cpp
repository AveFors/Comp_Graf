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
#include <memory>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

namespace application {
namespace {
using math::Vec3;
using math::Matrix;
struct Vertex { Vec3 position; Vec3 normal; float uv[2]; };

// ========================================================================
// Параметры камеры, направленного света, материала и объекта
// ========================================================================
struct Parameters {
    float cameraPosition[3]={3.6f,2.6f,6.0f};
    float cameraRotation[3]={-20.4f,31.0f,0};
    float fov=45, nearPlane=0.1f, farPlane=100, orthoHeight=5;
    int projection=0;
    float shadowSize = 12, depthBias = 8, slopeBias = 1.75f;
    float lightDirection[3]={0.65f,-0.8f,-0.35f};
    float lightColor[3]={1,0.94f,0.85f}, lightIntensity=1.2f;
    float ambientColor[3]={1,1,1}, ambientIntensity=0.12f;
    float specular[3]={1,1,1}, shininess=32;
};
Parameters settings;

// std140: четыре mat4 и шесть vec4; структура совпадает с блоком Scene в GLSL.
struct alignas(16) SceneUniform {
    Matrix viewProjection,model,normal,lightViewProjection;
    std::array<float,4> camera,lightDirection,light,ambient,surface,specular;
};
static_assert(sizeof(SceneUniform)==352);
VkBuffer vertexBuffer{},indexBuffer{},uniformBuffer{};
VmaAllocation vertexAllocation{},indexAllocation{},uniformAllocation{};
VkPipelineLayout pipelineLayout{};
VkPipeline pipeline{}, shadowPipeline{};
constexpr VkFormat shadowFormat = VK_FORMAT_D32_SFLOAT;
constexpr uint32_t shadowSizePixels = 2048;
VkImage shadowImage{};
VmaAllocation shadowAllocation{};
VkImageView shadowView{};
VkSampler shadowSampler{};
bool shadowInitialized = false;
VkDescriptorSetLayout descriptorLayout{};
VkDescriptorPool descriptorPool{};
VkDescriptorSet descriptorSet{};
constexpr uint32_t indexCount=36;
constexpr uint32_t modelCount=2;
constexpr uint32_t sceneObjectCount = modelCount + 1;
struct ObjectParameters { float position[3],rotation[3],scale[3]; };
const std::array<ObjectParameters,modelCount> initialObjects={
    ObjectParameters{{-1.4f,-0.1125f,0},{0,0,0},{0.85f,0.85f,0.85f}},
    ObjectParameters{{1.5f,-0.225f,0},{0,-20,0},{0.7f,0.7f,0.7f}}};
auto objects=initialObjects;
const ObjectParameters floorObject{{0,-0.795f,0},{0,0,0},{4,0.06f,7}};
int selectedObject=0;
VkDeviceSize uniformStride{};
VkCommandPool uploadPool{};
VkImage textureImage{};
VmaAllocation textureAllocation{};
VkImageView textureView{};
VkSampler textureSampler{};


Matrix lightViewProjection(Vec3 direction) {
    const Vec3 eye{-12 * direction.x, -12 * direction.y, -12 * direction.z};
    const Vec3 up = std::abs(direction.y) > 0.98f ? Vec3{0,0,1} : Vec3{0,1,0};
    return math::multiply(math::projection(1, 1, 0, 0.1f, 26, settings.shadowSize),
                          math::lookAt(eye, {0,0,0}, up));
}

SceneUniform sceneUniform(float aspect,uint32_t modelIndex) {
    const bool isFloor = modelIndex == modelCount;
    const auto& object = isFloor ? floorObject : objects[modelIndex];
    const Matrix r=math::rotation(object.rotation);
    const Matrix model=math::multiply(math::translation(math::vector(object.position)),
        math::multiply(r,math::scaling(math::vector(object.scale))));
    const Vec3 eye=math::vector(settings.cameraPosition);
    const Matrix view=math::multiply(math::transpose(math::rotation(settings.cameraRotation)),
        math::translation({-eye.x,-eye.y,-eye.z}));
    const Matrix projection=math::projection(settings.projection,aspect,settings.fov,
        settings.nearPlane,settings.farPlane,settings.orthoHeight);
    const Matrix normal=math::multiply(r,math::scaling({1/object.scale[0],1/object.scale[1],1/object.scale[2]}));
    const Vec3 direction=math::normalized(math::vector(settings.lightDirection));
    return {math::multiply(projection,view),model,normal,lightViewProjection(direction),
        {eye.x,eye.y,eye.z,1},{direction.x,direction.y,direction.z,0},
        {settings.lightColor[0],settings.lightColor[1],settings.lightColor[2],settings.lightIntensity},
        {settings.ambientColor[0],settings.ambientColor[1],settings.ambientColor[2],settings.ambientIntensity},
        {0.34f,0.38f,0.44f,isFloor ? 0.0f : 1.0f},
        {settings.specular[0],settings.specular[1],settings.specular[2],settings.shininess}};
}

// ========================================================================
// Динамический uniform-буфер и текстура в одном наборе дескрипторов
// ========================================================================
void check(VkResult result,const char* operation);
void createDescriptors() {
    auto device=graphics::internal::context.device;
    const VkDescriptorSetLayoutBinding bindings[]={
        {0,VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,1,VK_SHADER_STAGE_VERTEX_BIT|VK_SHADER_STAGE_FRAGMENT_BIT,nullptr},
        {1,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1,VK_SHADER_STAGE_FRAGMENT_BIT,nullptr},
        {2,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1,VK_SHADER_STAGE_FRAGMENT_BIT,nullptr}};
    VkDescriptorSetLayoutCreateInfo layout{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layout.bindingCount=3; layout.pBindings=bindings;
    check(vkCreateDescriptorSetLayout(device,&layout,nullptr,&descriptorLayout),"descriptor layout");
    const VkDescriptorPoolSize sizes[]={{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,1},{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,2}};
    VkDescriptorPoolCreateInfo pool{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pool.maxSets=1;pool.poolSizeCount=2;pool.pPoolSizes=sizes;
    check(vkCreateDescriptorPool(device,&pool,nullptr,&descriptorPool),"descriptor pool");
    VkDescriptorSetAllocateInfo allocate{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocate.descriptorPool=descriptorPool;allocate.descriptorSetCount=1;allocate.pSetLayouts=&descriptorLayout;
    check(vkAllocateDescriptorSets(device,&allocate,&descriptorSet),"descriptor set");
    const VkDescriptorBufferInfo buffer{uniformBuffer,0,sizeof(SceneUniform)};
    const VkDescriptorImageInfo image{textureSampler,textureView,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    const VkDescriptorImageInfo shadow{shadowSampler, shadowView, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL};
    VkWriteDescriptorSet writes[3]{};
    for(auto& w:writes){w.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;w.dstSet=descriptorSet;w.descriptorCount=1;}
    writes[0].dstBinding=0;writes[0].descriptorType=VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;writes[0].pBufferInfo=&buffer;
    writes[1].dstBinding=1;writes[1].descriptorType=VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;writes[1].pImageInfo=&image;
    writes[2].dstBinding = 2;
    writes[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[2].pImageInfo = &shadow;
    vkUpdateDescriptorSets(device,3,writes,0,nullptr);
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
void createPipeline(bool depthOnly) {
    auto& context = graphics::internal::context;
    VkShaderModule vertexShader{}, fragmentShader{};
    try {
        vertexShader = loadShader(depthOnly ? "shadow.vert.spv" : "textured.vert.spv");
        if (!depthOnly)
            fragmentShader = loadShader("textured.frag.spv");

        // --- Один набор дескрипторов с uniform-буфером ---
        VkPipelineLayoutCreateInfo layout{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        layout.setLayoutCount = 1;
        layout.pSetLayouts = &descriptorLayout;
        if (!pipelineLayout) check(vkCreatePipelineLayout(context.device, &layout, nullptr, &pipelineLayout), "vkCreatePipelineLayout");

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
            {2, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(Vertex, uv)},
        };
        VkPipelineVertexInputStateCreateInfo input{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        input.vertexBindingDescriptionCount = 1;
        input.pVertexBindingDescriptions = &binding;
        input.vertexAttributeDescriptionCount = depthOnly ? 1 : 3;
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
        raster.depthBiasEnable = depthOnly ? VK_TRUE : VK_FALSE;
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
        blend.attachmentCount = depthOnly ? 0 : 1;
        blend.pAttachments = &attachment;

        // --- Размеры viewport и scissor задаются в каждом кадре ---
        const VkDynamicState dynamicStates[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_DEPTH_BIAS};
        VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
        dynamic.dynamicStateCount = depthOnly ? 3 : 2;
        dynamic.pDynamicStates = dynamicStates;

        // --- Сборка настроек в один pipeline ---
        VkGraphicsPipelineCreateInfo info{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        info.stageCount = depthOnly ? 1 : 2;
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
        VkPipelineRenderingCreateInfo rendering{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
        rendering.depthAttachmentFormat = shadowFormat;
        info.pNext = depthOnly ? &rendering : nullptr;
        info.renderPass = depthOnly ? VK_NULL_HANDLE : context.render_pass;
        auto& handle = depthOnly ? shadowPipeline : pipeline;
        check(vkCreateGraphicsPipelines(context.device, VK_NULL_HANDLE, 1, &info, nullptr, &handle), "vkCreateGraphicsPipelines");
    } catch (...) {
        if (vertexShader) vkDestroyShaderModule(context.device, vertexShader, nullptr);
        if (fragmentShader) vkDestroyShaderModule(context.device, fragmentShader, nullptr);
        throw;
    }
    vkDestroyShaderModule(context.device, vertexShader, nullptr);
    if (fragmentShader) vkDestroyShaderModule(context.device, fragmentShader, nullptr);
}
// ========================================================================
// Передача геометрии и изображения через staging в DEVICE_LOCAL память
// ========================================================================
VkCommandBuffer beginUpload() {
    const auto device=graphics::internal::context.device;
    VkCommandBufferAllocateInfo allocation{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    allocation.commandPool=uploadPool;allocation.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY;allocation.commandBufferCount=1;
    VkCommandBuffer cmd{};
    check(vkAllocateCommandBuffers(device,&allocation,&cmd),"allocate upload command");
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    check(vkBeginCommandBuffer(cmd,&begin),"begin upload");
    return cmd;
}
void finishUpload(VkCommandBuffer cmd) {
    const auto& c=graphics::internal::context;
    check(vkEndCommandBuffer(cmd),"end upload");
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};submit.commandBufferCount=1;submit.pCommandBuffers=&cmd;
    check(vkQueueSubmit(c.graphics_queue,1,&submit,VK_NULL_HANDLE),"submit upload");
    check(vkQueueWaitIdle(c.graphics_queue),"wait upload");
    vkFreeCommandBuffers(c.device,uploadPool,1,&cmd);
}
void createDeviceBuffer(const void* data,VkDeviceSize size,VkBufferUsageFlags usage,VkBuffer& target,VmaAllocation& memory) {
    auto& c=graphics::internal::context;
    VkBuffer staging{};VmaAllocation stagingMemory{};
    try {
        createBuffer(data,size,VK_BUFFER_USAGE_TRANSFER_SRC_BIT,staging,stagingMemory);
        VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};info.size=size;
        info.usage=usage|VK_BUFFER_USAGE_TRANSFER_DST_BIT;info.sharingMode=VK_SHARING_MODE_EXCLUSIVE;
        VmaAllocationCreateInfo allocation{};allocation.usage=VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        allocation.requiredFlags=VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        check(vmaCreateBuffer(c.allocator,&info,&allocation,&target,&memory,nullptr),"device buffer");
        auto cmd=beginUpload();
        const VkBufferCopy copy{0,0,size};vkCmdCopyBuffer(cmd,staging,target,1,&copy);
        VkBufferMemoryBarrier barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        barrier.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask=usage==VK_BUFFER_USAGE_VERTEX_BUFFER_BIT?VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT:VK_ACCESS_INDEX_READ_BIT;
        barrier.srcQueueFamilyIndex=barrier.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;
        barrier.buffer=target;barrier.size=VK_WHOLE_SIZE;
        vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,0,0,nullptr,1,&barrier,0,nullptr);
        finishUpload(cmd);
    } catch(...) {if(staging)vmaDestroyBuffer(c.allocator,staging,stagingMemory);throw;}
    vmaDestroyBuffer(c.allocator,staging,stagingMemory);
}

// ========================================================================
// Одна JPEG-текстура: файл -> stb_image -> staging -> VkImage -> sampler
// ========================================================================
void createTexture() {
    auto& c=graphics::internal::context;
    const std::string path=std::string(CG_ASSET_DIR)+"/leopard.jpg";
    int width{},height{},channels{};
    std::unique_ptr<stbi_uc,decltype(&stbi_image_free)> pixels(stbi_load(path.c_str(),&width,&height,&channels,STBI_rgb_alpha),stbi_image_free);
    if(!pixels) throw std::runtime_error("Texture load failed: "+path+": "+stbi_failure_reason());
    VkPhysicalDeviceProperties properties{};vkGetPhysicalDeviceProperties(c.physical_device,&properties);
    if(width<=0||height<=0||uint32_t(width)>properties.limits.maxImageDimension2D||uint32_t(height)>properties.limits.maxImageDimension2D)
        throw std::runtime_error("Texture dimensions exceed device limits");
    VkBuffer staging{};VmaAllocation stagingMemory{};
    try {
        const VkDeviceSize size=VkDeviceSize(width)*VkDeviceSize(height)*4;
        createBuffer(pixels.get(),size,VK_BUFFER_USAGE_TRANSFER_SRC_BIT,staging,stagingMemory);
        VkImageCreateInfo image{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        image.imageType=VK_IMAGE_TYPE_2D;image.format=VK_FORMAT_R8G8B8A8_SRGB;
        image.extent={uint32_t(width),uint32_t(height),1};image.mipLevels=1;image.arrayLayers=1;
        image.samples=VK_SAMPLE_COUNT_1_BIT;image.tiling=VK_IMAGE_TILING_OPTIMAL;
        image.usage=VK_IMAGE_USAGE_TRANSFER_DST_BIT|VK_IMAGE_USAGE_SAMPLED_BIT;
        image.sharingMode=VK_SHARING_MODE_EXCLUSIVE;image.initialLayout=VK_IMAGE_LAYOUT_UNDEFINED;
        VmaAllocationCreateInfo allocation{};allocation.usage=VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        allocation.requiredFlags=VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        check(vmaCreateImage(c.allocator,&image,&allocation,&textureImage,&textureAllocation,nullptr),"texture image");
        auto cmd=beginUpload();
        VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        barrier.srcQueueFamilyIndex=barrier.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;
        barrier.image=textureImage;barrier.subresourceRange={VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1};
        barrier.oldLayout=VK_IMAGE_LAYOUT_UNDEFINED;barrier.newLayout=VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.dstAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,0,0,nullptr,0,nullptr,1,&barrier);
        VkBufferImageCopy copy{};copy.imageSubresource={VK_IMAGE_ASPECT_COLOR_BIT,0,0,1};
        copy.imageExtent={uint32_t(width),uint32_t(height),1};
        vkCmdCopyBufferToImage(cmd,staging,textureImage,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,1,&copy);
        barrier.oldLayout=VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;barrier.newLayout=VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrier.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT;barrier.dstAccessMask=VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,0,0,nullptr,0,nullptr,1,&barrier);
        finishUpload(cmd);
    } catch(...) {if(staging)vmaDestroyBuffer(c.allocator,staging,stagingMemory);throw;}
    vmaDestroyBuffer(c.allocator,staging,stagingMemory);
    VkImageViewCreateInfo view{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    view.image=textureImage;view.viewType=VK_IMAGE_VIEW_TYPE_2D;view.format=VK_FORMAT_R8G8B8A8_SRGB;
    view.subresourceRange={VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1};
    check(vkCreateImageView(c.device,&view,nullptr,&textureView),"texture view");
    VkSamplerCreateInfo sampler{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    sampler.magFilter=sampler.minFilter=VK_FILTER_LINEAR;sampler.mipmapMode=VK_SAMPLER_MIPMAP_MODE_NEAREST;
    sampler.addressModeU=sampler.addressModeV=sampler.addressModeW=VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sampler.maxLod=0;
    check(vkCreateSampler(c.device,&sampler,nullptr,&textureSampler),"texture sampler");
}

// ========================================================================
// Карта глубины и сэмплер, сравнивающий глубину фрагмента с картой
// ========================================================================
void createShadowMap() {
    const auto& c = graphics::internal::context;
    VkFormatProperties properties{};
    vkGetPhysicalDeviceFormatProperties(c.physical_device, shadowFormat, &properties);
    constexpr auto features = VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT |
                              VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;
    if ((properties.optimalTilingFeatures & features) != features)
        throw std::runtime_error("D32 shadow map format is not supported");

    VkImageCreateInfo image{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    image.imageType = VK_IMAGE_TYPE_2D;
    image.format = shadowFormat;
    image.extent = {shadowSizePixels, shadowSizePixels, 1};
    image.mipLevels = image.arrayLayers = 1;
    image.samples = VK_SAMPLE_COUNT_1_BIT;
    image.tiling = VK_IMAGE_TILING_OPTIMAL;
    image.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    VmaAllocationCreateInfo memory{};
    memory.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    check(vmaCreateImage(c.allocator, &image, &memory, &shadowImage, &shadowAllocation, nullptr), "shadow image");

    VkImageViewCreateInfo view{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    view.image = shadowImage;
    view.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view.format = shadowFormat;
    view.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
    check(vkCreateImageView(c.device, &view, nullptr, &shadowView), "shadow image view");

    VkSamplerCreateInfo sampler{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    sampler.minFilter = sampler.magFilter = VK_FILTER_NEAREST;
    sampler.addressModeU = sampler.addressModeV = sampler.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler.compareEnable = VK_TRUE;
    sampler.compareOp = VK_COMPARE_OP_LESS;
    check(vkCreateSampler(c.device, &sampler, nullptr, &shadowSampler), "shadow sampler");
}

void drawScene(VkCommandBuffer cmd) {
    for (uint32_t model = 0; model < sceneObjectCount; ++model) {
        const auto offset = static_cast<uint32_t>(model * uniformStride);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0,
                                1, &descriptorSet, 1, &offset);
        vkCmdDrawIndexed(cmd, indexCount, 1, 0, 0, 0);
    }
}

// ========================================================================
// Первый проход кадра: заполняем карту глубины с точки зрения солнца
// ========================================================================
void renderShadowMap(VkCommandBuffer cmd) {
    constexpr auto depthStages = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                                 VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = shadowImage;
    barrier.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
    barrier.oldLayout = shadowInitialized ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL :
                                           VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    barrier.srcAccessMask = shadowInitialized ? VK_ACCESS_SHADER_READ_BIT : 0;
    barrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, shadowInitialized ? VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT :
                                                  VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         depthStages, 0, 0, nullptr, 0, nullptr, 1, &barrier);

    VkRenderingAttachmentInfo depth{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    depth.imageView = shadowView;
    depth.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    depth.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depth.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    depth.clearValue.depthStencil = {1.0f, 0};

    VkRenderingInfo rendering{VK_STRUCTURE_TYPE_RENDERING_INFO};
    rendering.renderArea = {{0, 0}, {shadowSizePixels, shadowSizePixels}};
    rendering.layerCount = 1;
    rendering.pDepthAttachment = &depth;
    vkCmdBeginRendering(cmd, &rendering);
    const VkViewport viewport{0, 0, float(shadowSizePixels), float(shadowSizePixels), 0, 1};
    vkCmdSetViewport(cmd, 0, 1, &viewport);
    vkCmdSetScissor(cmd, 0, 1, &rendering.renderArea);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, shadowPipeline);
    vkCmdSetDepthBias(cmd, settings.depthBias, 0, settings.slopeBias);
    drawScene(cmd);
    vkCmdEndRendering(cmd);

    // Перед чтением из фрагментного шейдера завершаем запись глубины.
    barrier.oldLayout = barrier.newLayout;
    barrier.newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
    barrier.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd, depthStages, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                         0, 0, nullptr, 0, nullptr, 1, &barrier);
    shadowInitialized = true;
}

}

// ========================================================================
// Геометрия варианта 3: нормали вычисляются векторным произведением
// ========================================================================
bool initialize() {
    try {
        auto& context=graphics::internal::context;
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(context.physical_device,&properties);
        const VkDeviceSize alignment=properties.limits.minUniformBufferOffsetAlignment;
        uniformStride=(sizeof(SceneUniform)+alignment-1)/alignment*alignment;
        VkCommandPoolCreateInfo pool{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pool.flags=VK_COMMAND_POOL_CREATE_TRANSIENT_BIT; pool.queueFamilyIndex=context.graphics_queue_index;
        check(vkCreateCommandPool(context.device,&pool,nullptr,&uploadPool),"upload pool");
        constexpr float x=1.2f,y=0.75f,z=0.5f;
        const Vec3 corners[]={{-x,-y,z},{x,-y,z},{x,y,z},{-x,y,z},
                              {-x,-y,-z},{x,-y,-z},{x,y,-z},{-x,y,-z}};
        const uint16_t faces[6][4]={{0,1,2,3},{5,4,7,6},{1,5,6,2},{4,0,3,7},{3,2,6,7},{4,5,1,0}};
        std::array<Vertex,24> vertices{};
        std::array<uint16_t,indexCount> indices{};
        for (uint16_t face=0;face<6;++face) {
            const auto& f=faces[face];
            const Vec3 normal=math::normalized(math::cross(corners[f[1]]-corners[f[0]],corners[f[2]]-corners[f[0]]));
            const float uv[4][2]={{0,1},{1,1},{1,0},{0,0}};
            for (int i=0;i<4;++i) vertices[face*4+i]={corners[f[i]],normal,{uv[i][0],uv[i][1]}};
            const uint16_t base=face*4;
            const uint16_t triangle[]={base,uint16_t(base+1),uint16_t(base+2),base,uint16_t(base+2),uint16_t(base+3)};
            std::memcpy(indices.data()+face*6,triangle,sizeof(triangle));
        }
        createDeviceBuffer(vertices.data(),sizeof(vertices),VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,vertexBuffer,vertexAllocation);
        createDeviceBuffer(indices.data(),sizeof(indices),VK_BUFFER_USAGE_INDEX_BUFFER_BIT,indexBuffer,indexAllocation);
        createBuffer(nullptr,uniformStride*sceneObjectCount,VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,uniformBuffer,uniformAllocation);
        createTexture();
        createShadowMap();
        createDescriptors();
        createPipeline(false);
        createPipeline(true);
        return true;
    } catch (const std::exception& e) {
        std::cerr<<"Application initialization: "<<e.what()<<'\n'; shutdown(); return false;
    }
}

void shutdown() {
    auto& c=graphics::internal::context;
    vkQueueWaitIdle(c.graphics_queue);
    if(shadowPipeline) vkDestroyPipeline(c.device,shadowPipeline,nullptr);
    if(pipeline) vkDestroyPipeline(c.device,pipeline,nullptr);
    if(pipelineLayout) vkDestroyPipelineLayout(c.device,pipelineLayout,nullptr);
    if(descriptorPool) vkDestroyDescriptorPool(c.device,descriptorPool,nullptr);
    if(descriptorLayout) vkDestroyDescriptorSetLayout(c.device,descriptorLayout,nullptr);
    if(shadowSampler) vkDestroySampler(c.device,shadowSampler,nullptr);
    if(shadowView) vkDestroyImageView(c.device,shadowView,nullptr);
    if(shadowImage) vmaDestroyImage(c.allocator,shadowImage,shadowAllocation);
    if(textureSampler) vkDestroySampler(c.device,textureSampler,nullptr);
    if(textureView) vkDestroyImageView(c.device,textureView,nullptr);
    if(textureImage) vmaDestroyImage(c.allocator,textureImage,textureAllocation);
    if(uploadPool) vkDestroyCommandPool(c.device,uploadPool,nullptr);
    if(uniformBuffer) vmaDestroyBuffer(c.allocator,uniformBuffer,uniformAllocation);
    if(indexBuffer) vmaDestroyBuffer(c.allocator,indexBuffer,indexAllocation);
    if(vertexBuffer) vmaDestroyBuffer(c.allocator,vertexBuffer,vertexAllocation);
    shadowPipeline={};shadowSampler={};shadowView={};shadowImage={};shadowInitialized=false;
    pipeline={};pipelineLayout={};descriptorPool={};descriptorLayout={};
    uniformBuffer={};indexBuffer={};vertexBuffer={};
    textureSampler={};textureView={};textureImage={};uploadPool={};
}

// ========================================================================
// Все параметры изменяются через ImGui, без клавиатурных комбинаций
// ========================================================================
void update([[maybe_unused]] double time) {
    ImGui::SetNextWindowPos({12,12},ImGuiCond_Always);
    ImGui::SetNextWindowSize({440,std::max(120.0f,ImGui::GetIO().DisplaySize.y-24)},ImGuiCond_Always);
    ImGui::Begin("Lab 4 - Shadow mapping",nullptr,ImGuiWindowFlags_NoMove|ImGuiWindowFlags_NoResize|ImGuiWindowFlags_NoCollapse);
    ImGui::TextUnformatted("Variant 3: parallelepiped");
    if(ImGui::Button("Reset scene")) {settings=Parameters{};objects=initialObjects;selectedObject=0;}
    ImGui::SameLine(); ImGui::Text("%.0f FPS",ImGui::GetIO().Framerate);
    ImGui::PushItemWidth(215);
    const auto flags=ImGuiSliderFlags_AlwaysClamp;
    if(ImGui::CollapsingHeader("Camera")) {
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
    if (ImGui::CollapsingHeader("Shadow map", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::SliderFloat("Ortho area width", &settings.shadowSize, 4, 24, "%.1f", flags);
        ImGui::SliderFloat("Depth bias", &settings.depthBias, 0, 64, "%.1f", flags);
        ImGui::SliderFloat("Slope bias", &settings.slopeBias, 0, 4, "%.2f", flags);
    }
    if(ImGui::CollapsingHeader("Ambient light",ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::ColorEdit3("Ambient color",settings.ambientColor);
        ImGui::SliderFloat("Ambient intensity",&settings.ambientIntensity,0,1,"%.2f",flags);
    }
    if(ImGui::CollapsingHeader("Material",ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::TextUnformatted("Albedo: textures/leopard.jpg");
        ImGui::ColorEdit3("Specular color",settings.specular);
        ImGui::SliderFloat("Shininess",&settings.shininess,1,256,"%.1f",flags);
    }
    if(ImGui::CollapsingHeader("Object",ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Combo("Model",&selectedObject,"Parallelepiped 1\0Parallelepiped 2\0");
        auto& object=objects[selectedObject];
        ImGui::DragFloat3("Position",object.position,0.02f,-3,3,"%.2f",flags);
        ImGui::DragFloat3("Rotation",object.rotation,0.5f,-180,180,"%.1f",flags);
        ImGui::DragFloat3("Scale",object.scale,0.01f,0.1f,3,"%.2f",flags);
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
    VmaAllocationInfo memory{};
    vmaGetAllocationInfo(c.allocator,uniformAllocation,&memory);
    for(uint32_t model=0;model<sceneObjectCount;++model) {
        const auto uniform=sceneUniform(float(extent.width)/float(extent.height),model);
        std::memcpy(static_cast<char*>(memory.pMappedData)+model*uniformStride,&uniform,sizeof(uniform));
    }
    check(vmaFlushAllocation(c.allocator,uniformAllocation,0,VK_WHOLE_SIZE),"uniform flush");
    const auto cmd=fd.command_buffer;
    check(vkResetCommandBuffer(cmd,0),"reset command buffer");
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    check(vkBeginCommandBuffer(cmd,&begin),"begin command buffer");
    const VkDeviceSize offset = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &vertexBuffer, &offset);
    vkCmdBindIndexBuffer(cmd, indexBuffer, 0, VK_INDEX_TYPE_UINT16);
    renderShadowMap(cmd);
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
    drawScene(cmd);
    vkCmdEndRenderPass(cmd);
    check(vkEndCommandBuffer(cmd),"end command buffer");
}
}
