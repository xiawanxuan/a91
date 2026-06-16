#version 450 core

layout(lines) in;
layout(triangle_strip, max_vertices = 4) out;

uniform mat4 uViewMatrix;
uniform mat4 uProjMatrix;
uniform vec3 uCameraPos;
uniform float uHairThickness;
uniform float uTipScale;

in vec3 vWorldPos[];
in vec3 vTangent[];
in float vSegmentT[];
in float vStrandRand[];

out vec3 gWorldPos;
out vec3 gNormal;
out float gSegmentT;
out float gStrandRand;

void main() {
    vec3 p0 = vWorldPos[0];
    vec3 p1 = vWorldPos[1];

    vec3 tangent = normalize(p1 - p0);
    vec3 viewDir0 = normalize(uCameraPos - p0);
    vec3 viewDir1 = normalize(uCameraPos - p1);

    vec3 normal0 = normalize(cross(tangent, viewDir0));
    vec3 normal1 = normalize(cross(tangent, viewDir1));

    float t0 = vSegmentT[0];
    float t1 = vSegmentT[1];

    float thickness0 = uHairThickness * (1.0 - t0 * (1.0 - uTipScale));
    float thickness1 = uHairThickness * (1.0 - t1 * (1.0 - uTipScale));

    vec3 offset0 = normal0 * thickness0 * 0.5;
    vec3 offset1 = normal1 * thickness1 * 0.5;

    gNormal = -normal0;
    gSegmentT = t0;
    gStrandRand = vStrandRand[0];
    gWorldPos = p0 - offset0;
    gl_Position = uProjMatrix * uViewMatrix * vec4(gWorldPos, 1.0);
    EmitVertex();

    gNormal = -normal1;
    gSegmentT = t1;
    gStrandRand = vStrandRand[1];
    gWorldPos = p1 - offset1;
    gl_Position = uProjMatrix * uViewMatrix * vec4(gWorldPos, 1.0);
    EmitVertex();

    gNormal = normal0;
    gSegmentT = t0;
    gStrandRand = vStrandRand[0];
    gWorldPos = p0 + offset0;
    gl_Position = uProjMatrix * uViewMatrix * vec4(gWorldPos, 1.0);
    EmitVertex();

    gNormal = normal1;
    gSegmentT = t1;
    gStrandRand = vStrandRand[1];
    gWorldPos = p1 + offset1;
    gl_Position = uProjMatrix * uViewMatrix * vec4(gWorldPos, 1.0);
    EmitVertex();

    EndPrimitive();
}
