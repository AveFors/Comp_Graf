#version 450
layout(location=0) in vec3 worldPosition;
layout(location=1) in vec3 worldNormal;
layout(location=2) in vec2 uv;
layout(set=0,binding=1) uniform sampler2D albedoTexture;
layout(set=0,binding=2) uniform sampler2DShadow shadowMap;
layout(location=0) out vec4 outColor;
layout(set=0,binding=0,std140) uniform Scene {
    mat4 viewProjection; mat4 model; mat4 normalMatrix; mat4 lightViewProjection;
    vec4 camera; vec4 lightDirection; vec4 light; vec4 ambient; vec4 surface; vec4 specular;
} scene;

// --- Блинн–Фонг: ambient + diffuse + specular для одного направленного света ---
vec3 safeNormalize(vec3 v) { return v*inversesqrt(max(dot(v,v),1e-12)); }
float lightVisibility() {
    vec4 clip = scene.lightViewProjection * vec4(worldPosition, 1);
    vec3 ndc = clip.xyz / clip.w;
    vec2 shadowUV = ndc.xy * 0.5 + 0.5;
    if (any(lessThan(shadowUV, vec2(0))) || any(greaterThan(shadowUV, vec2(1))) ||
        ndc.z < 0.0 || ndc.z > 1.0)
        return 1.0;
    return texture(shadowMap, vec3(shadowUV, ndc.z));
}

void main() {
    vec3 albedo = scene.surface.w > 0.5 ? texture(albedoTexture, uv).rgb : scene.surface.rgb;
    vec3 N=safeNormalize(worldNormal);
    vec3 L=safeNormalize(-scene.lightDirection.xyz);
    vec3 V=safeNormalize(scene.camera.xyz-worldPosition);
    vec3 H=safeNormalize(L+V);
    float diffuse=max(dot(N,L),0);
    float highlight=diffuse>0 ? pow(max(dot(N,H),0),scene.specular.a) : 0;
    vec3 ambient=scene.ambient.rgb*scene.ambient.a*albedo;
    vec3 direct=scene.light.rgb*scene.light.a*(albedo*diffuse+scene.specular.rgb*highlight);
    outColor=vec4(ambient+lightVisibility()*direct,1);
}
