// ========================================================================
// Фрагментный шейдер: цвет и освещение
// ========================================================================
#version 450
layout(location = 0) in vec3 faceNormal;
layout(location = 0) out vec4 outColor;

// --- Фоновая яркость + направленный свет; непрозрачный оранжевый цвет ---
void main() {
    const vec3 lightDirection = normalize(vec3(0.4, 0.8, 0.6));
    float brightness = 0.30 + 0.70 * max(dot(normalize(faceNormal), lightDirection), 0.0);
    outColor = vec4(vec3(0.95, 0.58, 0.22) * brightness, 1.0);
}
