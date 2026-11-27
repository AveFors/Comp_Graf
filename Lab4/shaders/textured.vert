#version 450
layout(location=0) in vec3 position;
layout(location=1) in vec3 normal;
layout(location=2) in vec2 texCoord;
layout(location=0) out vec3 worldPosition;
layout(location=1) out vec3 worldNormal;
layout(location=2) out vec2 uv;
layout(set=0,binding=0,std140) uniform Scene {
    mat4 viewProjection; mat4 model; mat4 normalMatrix; mat4 lightViewProjection;
    vec4 camera; vec4 lightDirection; vec4 light; vec4 ambient; vec4 surface; vec4 specular;
} scene;
void main() {
    vec4 world=scene.model*vec4(position,1);
    uv=texCoord;
    worldPosition=world.xyz;
    worldNormal=mat3(scene.normalMatrix)*normal;
    gl_Position=scene.viewProjection*world;
}
