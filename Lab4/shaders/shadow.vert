#version 450
layout(location=0) in vec3 position;
layout(set=0,binding=0,std140) uniform Scene {
    mat4 viewProjection; mat4 model; mat4 normalMatrix; mat4 lightViewProjection;
} scene;

void main() {
    gl_Position = scene.lightViewProjection * scene.model * vec4(position, 1);
}
