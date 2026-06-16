#version 450 core

layout(location = 0) in vec3 aPosition;
layout(location = 1) in float aStrandIndex;
layout(location = 2) in float aSegmentIndex;

uniform mat4 uViewMatrix;
uniform mat4 uProjMatrix;
uniform vec3 uCameraPos;
uniform float uHairThickness;
uniform float uTipScale;
uniform uint uSegmentsPerStrand;

out vec3 vWorldPos;
out vec3 vTangent;
out float vSegmentT;
out float vStrandRand;

float hash(float n) {
    return fract(sin(n) * 43758.5453123);
}

void main() {
    vec3 worldPos = aPosition;

    float segT = aSegmentIndex / float(uSegmentsPerStrand - 1);
    vSegmentT = segT;

    float strandRand = hash(aStrandIndex * 12.9898 + 78.233);
    vStrandRand = strandRand;

    vTangent = vec3(0.0, 1.0, 0.0);

    vWorldPos = worldPos;
    gl_Position = uProjMatrix * uViewMatrix * vec4(worldPos, 1.0);
}
