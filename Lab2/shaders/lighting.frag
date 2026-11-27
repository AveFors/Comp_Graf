#version 450
layout(location=0) in vec3 worldPosition;
layout(location=1) in vec3 worldNormal;
layout(location=0) out vec4 outColor;
layout(set=0,binding=0,std140) uniform Scene {
    mat4 viewProjection; mat4 model; mat4 normalMatrix;
    vec4 camera; vec4 lightDirection; vec4 light; vec4 ambient; vec4 albedo; vec4 specular;
} scene;

// --- Блинн–Фонг: ambient + diffuse + specular для одного направленного света ---
vec3 safeNormalize(vec3 v) { return v*inversesqrt(max(dot(v,v),1e-12)); }
void main() {
    vec3 N=safeNormalize(worldNormal);
    vec3 L=safeNormalize(-scene.lightDirection.xyz);
    vec3 V=safeNormalize(scene.camera.xyz-worldPosition);
    vec3 H=safeNormalize(L+V);
    float diffuse=max(dot(N,L),0);
    float highlight=diffuse>0 ? pow(max(dot(N,H),0),scene.specular.a) : 0;
    vec3 ambient=scene.ambient.rgb*scene.ambient.a*scene.albedo.rgb;
    vec3 direct=scene.light.rgb*scene.light.a*(scene.albedo.rgb*diffuse+scene.specular.rgb*highlight);
    outColor=vec4(ambient+direct,1);
}
