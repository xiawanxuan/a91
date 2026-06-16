#version 450 core

in vec3 gWorldPos;
in vec3 gNormal;
in float gSegmentT;
in float gStrandRand;

uniform vec3 uHairColor;
uniform vec3 uLightDir;
uniform vec3 uLightColor;
uniform vec3 uAmbientColor;
uniform float uSpecularStrength;
uniform float uShininess;
uniform vec3 uCameraPos;

out vec4 fragColor;

void main() {
    vec3 normal = normalize(gNormal);
    vec3 lightDir = normalize(uLightDir);
    vec3 viewDir = normalize(uCameraPos - gWorldPos);

    float diffuse = max(dot(normal, lightDir), 0.0);
    diffuse = diffuse * 0.6 + 0.4;

    vec3 halfway = normalize(lightDir + viewDir);
    float spec = pow(max(dot(normal, halfway), 0.0), uShininess);
    vec3 specular = uSpecularStrength * spec * uLightColor;

    float tipFactor = gSegmentT;
    vec3 baseColor = uHairColor * (1.0 + (gStrandRand - 0.5) * 0.2);
    vec3 tipColor = baseColor * 1.3;
    vec3 finalColor = mix(baseColor, tipColor, tipFactor * 0.3);

    vec3 ambient = uAmbientColor * finalColor;
    vec3 diffuseColor = diffuse * finalColor * uLightColor;

    vec3 result = ambient + diffuseColor + specular;

    float alpha = 1.0 - gSegmentT * 0.3;

    fragColor = vec4(result, alpha);
}
