#include "hair_renderer.h"
#include <glad/glad.h>
#include <glm/gtc/type_ptr.hpp>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cstring>

namespace hair {

HairRenderer::HairRenderer()
    : m_vao(0)
    , m_positionVBO(0)
    , m_strandIndexVBO(0)
    , m_segmentIndexVBO(0)
    , m_shaderProgram(0)
    , m_guideShaderProgram(0)
    , m_guideVAO(0)
    , m_guideVBO(0)
    , m_maxStrands(0)
    , m_segmentsPerStrand(0)
    , m_totalVertices(0)
    , m_hairColor(0.5f, 0.35f, 0.2f)
    , m_hairThickness(0.002f)
    , m_tipScale(0.3f)
    , m_wireframe(false)
    , m_showGuideCurves(false)
    , m_wetness(0.0f)
    , m_activeStrands(0)
    , m_initialized(false)
    , m_lastRenderTimeMs(0.0f)
    , m_renderTimerQuery(0)
{
}

HairRenderer::~HairRenderer() {
    shutdown();
}

bool HairRenderer::initialize(uint32_t maxStrands, uint32_t segmentsPerStrand) {
    m_maxStrands = maxStrands;
    m_segmentsPerStrand = segmentsPerStrand;
    m_totalVertices = maxStrands * segmentsPerStrand;
    m_activeStrands = maxStrands;

    if (!createShaderProgram()) {
        std::cerr << "Failed to create hair shader program" << std::endl;
        return false;
    }

    if (!createGuideShaderProgram()) {
        std::cerr << "Failed to create guide shader program" << std::endl;
    }

    if (!createBuffers()) {
        std::cerr << "Failed to create hair buffers" << std::endl;
        return false;
    }

    computeMultiDrawParams();

    glGenQueries(1, &m_renderTimerQuery);

    m_initialized = true;
    return true;
}

void HairRenderer::shutdown() {
    if (!m_initialized) return;

    destroyBuffers();

    if (m_shaderProgram) {
        glDeleteProgram(m_shaderProgram);
        m_shaderProgram = 0;
    }

    if (m_guideShaderProgram) {
        glDeleteProgram(m_guideShaderProgram);
        m_guideShaderProgram = 0;
    }

    if (m_guideVAO) {
        glDeleteVertexArrays(1, &m_guideVAO);
        m_guideVAO = 0;
    }

    if (m_guideVBO) {
        glDeleteBuffers(1, &m_guideVBO);
        m_guideVBO = 0;
    }

    if (m_renderTimerQuery) {
        glDeleteQueries(1, &m_renderTimerQuery);
        m_renderTimerQuery = 0;
    }

    m_initialized = false;
}

bool HairRenderer::createShaderProgram() {
    const char* vertexShaderSource = R"(
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
    )";

    const char* geometryShaderSource = R"(
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
    )";

    const char* fragmentShaderSource = R"(
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
    )";

    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, nullptr);
    glCompileShader(vertexShader);

    GLint success;
    GLchar infoLog[1024];
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(vertexShader, 1024, nullptr, infoLog);
        std::cerr << "Vertex shader compilation failed: " << infoLog << std::endl;
        return false;
    }

    GLuint geometryShader = glCreateShader(GL_GEOMETRY_SHADER);
    glShaderSource(geometryShader, 1, &geometryShaderSource, nullptr);
    glCompileShader(geometryShader);

    glGetShaderiv(geometryShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(geometryShader, 1024, nullptr, infoLog);
        std::cerr << "Geometry shader compilation failed: " << infoLog << std::endl;
        glDeleteShader(vertexShader);
        glDeleteShader(geometryShader);
        return false;
    }

    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, nullptr);
    glCompileShader(fragmentShader);

    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(fragmentShader, 1024, nullptr, infoLog);
        std::cerr << "Fragment shader compilation failed: " << infoLog << std::endl;
        glDeleteShader(vertexShader);
        glDeleteShader(geometryShader);
        glDeleteShader(fragmentShader);
        return false;
    }

    m_shaderProgram = glCreateProgram();
    glAttachShader(m_shaderProgram, vertexShader);
    glAttachShader(m_shaderProgram, geometryShader);
    glAttachShader(m_shaderProgram, fragmentShader);
    glLinkProgram(m_shaderProgram);

    glGetProgramiv(m_shaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(m_shaderProgram, 1024, nullptr, infoLog);
        std::cerr << "Shader program linking failed: " << infoLog << std::endl;
        glDeleteShader(vertexShader);
        glDeleteShader(geometryShader);
        glDeleteShader(fragmentShader);
        return false;
    }

    glDeleteShader(vertexShader);
    glDeleteShader(geometryShader);
    glDeleteShader(fragmentShader);

    return true;
}

bool HairRenderer::createGuideShaderProgram() {
    const char* vsSource = R"(
        #version 450 core
        layout(location = 0) in vec3 aPosition;
        uniform mat4 uViewMatrix;
        uniform mat4 uProjMatrix;
        void main() {
            gl_Position = uProjMatrix * uViewMatrix * vec4(aPosition, 1.0);
        }
    )";

    const char* fsSource = R"(
        #version 450 core
        out vec4 fragColor;
        void main() {
            fragColor = vec4(1.0, 0.8, 0.2, 1.0);
        }
    )";

    GLuint vs = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &vsSource, nullptr);
    glCompileShader(vs);

    GLint success;
    glGetShaderiv(vs, GL_COMPILE_STATUS, &success);
    if (!success) return false;

    GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &fsSource, nullptr);
    glCompileShader(fs);

    glGetShaderiv(fs, GL_COMPILE_STATUS, &success);
    if (!success) {
        glDeleteShader(vs);
        glDeleteShader(fs);
        return false;
    }

    m_guideShaderProgram = glCreateProgram();
    glAttachShader(m_guideShaderProgram, vs);
    glAttachShader(m_guideShaderProgram, fs);
    glLinkProgram(m_guideShaderProgram);

    glGetProgramiv(m_guideShaderProgram, GL_LINK_STATUS, &success);
    glDeleteShader(vs);
    glDeleteShader(fs);

    if (!success) return false;

    glGenVertexArrays(1, &m_guideVAO);
    glGenBuffers(1, &m_guideVBO);

    return true;
}

bool HairRenderer::createBuffers() {
    glGenVertexArrays(1, &m_vao);
    glBindVertexArray(m_vao);

    glGenBuffers(1, &m_positionVBO);
    glBindBuffer(GL_ARRAY_BUFFER, m_positionVBO);
    glBufferData(GL_ARRAY_BUFFER, m_totalVertices * sizeof(glm::vec3),
                 nullptr, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);
    glEnableVertexAttribArray(0);

    std::vector<float> strandIndices(m_totalVertices);
    std::vector<float> segmentIndices(m_totalVertices);
    for (uint32_t i = 0; i < m_maxStrands; i++) {
        for (uint32_t j = 0; j < m_segmentsPerStrand; j++) {
            uint32_t idx = i * m_segmentsPerStrand + j;
            strandIndices[idx] = (float)i;
            segmentIndices[idx] = (float)j;
        }
    }

    glGenBuffers(1, &m_strandIndexVBO);
    glBindBuffer(GL_ARRAY_BUFFER, m_strandIndexVBO);
    glBufferData(GL_ARRAY_BUFFER, m_totalVertices * sizeof(float),
                 strandIndices.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);

    glGenBuffers(1, &m_segmentIndexVBO);
    glBindBuffer(GL_ARRAY_BUFFER, m_segmentIndexVBO);
    glBufferData(GL_ARRAY_BUFFER, m_totalVertices * sizeof(float),
                 segmentIndices.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, sizeof(float), (void*)0);
    glEnableVertexAttribArray(2);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    return true;
}

void HairRenderer::destroyBuffers() {
    if (m_vao) {
        glDeleteVertexArrays(1, &m_vao);
        m_vao = 0;
    }
    if (m_positionVBO) {
        glDeleteBuffers(1, &m_positionVBO);
        m_positionVBO = 0;
    }
    if (m_strandIndexVBO) {
        glDeleteBuffers(1, &m_strandIndexVBO);
        m_strandIndexVBO = 0;
    }
    if (m_segmentIndexVBO) {
        glDeleteBuffers(1, &m_segmentIndexVBO);
        m_segmentIndexVBO = 0;
    }
}

void HairRenderer::computeMultiDrawParams() {
    m_multiDrawFirsts.resize(m_maxStrands);
    m_multiDrawCounts.resize(m_maxStrands);

    for (uint32_t i = 0; i < m_maxStrands; i++) {
        m_multiDrawFirsts[i] = (int32_t)(i * m_segmentsPerStrand);
        m_multiDrawCounts[i] = (int32_t)m_segmentsPerStrand;
    }
    m_activeStrands = m_maxStrands;
}

void HairRenderer::updateBuffers(const std::vector<glm::vec3>& positions) {
    if (!m_initialized) return;

    glBindBuffer(GL_ARRAY_BUFFER, m_positionVBO);
    glBufferSubData(GL_ARRAY_BUFFER, 0,
                    positions.size() * sizeof(glm::vec3),
                    positions.data());
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void HairRenderer::updateGuideCurveBuffers(const std::vector<glm::vec3>& guidePositions,
                                             uint32_t numGuideCurves,
                                             uint32_t segmentsPerStrand) {
    if (!m_guideVAO || !m_guideVBO) return;

    glBindVertexArray(m_guideVAO);
    glBindBuffer(GL_ARRAY_BUFFER, m_guideVBO);
    glBufferData(GL_ARRAY_BUFFER, guidePositions.size() * sizeof(glm::vec3),
                 guidePositions.data(), GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

void HairRenderer::render(const glm::mat4& viewMatrix,
                           const glm::mat4& projMatrix,
                           const glm::vec3& cameraPos) {
    if (!m_initialized) return;

    glBeginQuery(GL_TIME_ELAPSED, m_renderTimerQuery);

    glUseProgram(m_shaderProgram);

    glUniformMatrix4fv(glGetUniformLocation(m_shaderProgram, "uViewMatrix"),
                       1, GL_FALSE, glm::value_ptr(viewMatrix));
    glUniformMatrix4fv(glGetUniformLocation(m_shaderProgram, "uProjMatrix"),
                       1, GL_FALSE, glm::value_ptr(projMatrix));
    glUniform3fv(glGetUniformLocation(m_shaderProgram, "uCameraPos"),
                 1, glm::value_ptr(cameraPos));

    glUniform1f(glGetUniformLocation(m_shaderProgram, "uHairThickness"),
                m_hairThickness);
    glUniform1f(glGetUniformLocation(m_shaderProgram, "uTipScale"),
                m_tipScale);
    glUniform1ui(glGetUniformLocation(m_shaderProgram, "uSegmentsPerStrand"),
                 m_segmentsPerStrand);

    glUniform3fv(glGetUniformLocation(m_shaderProgram, "uHairColor"),
                 1, glm::value_ptr(m_hairColor));

    glm::vec3 lightDir = glm::normalize(glm::vec3(0.5f, 1.0f, 0.5f));
    glUniform3fv(glGetUniformLocation(m_shaderProgram, "uLightDir"),
                 1, glm::value_ptr(lightDir));

    glm::vec3 lightColor(1.0f, 0.95f, 0.9f);
    glUniform3fv(glGetUniformLocation(m_shaderProgram, "uLightColor"),
                 1, glm::value_ptr(lightColor));

    glm::vec3 ambientColor(0.15f, 0.15f, 0.2f);
    glUniform3fv(glGetUniformLocation(m_shaderProgram, "uAmbientColor"),
                 1, glm::value_ptr(ambientColor));

    glUniform1f(glGetUniformLocation(m_shaderProgram, "uSpecularStrength"),
                0.3f);
    glUniform1f(glGetUniformLocation(m_shaderProgram, "uShininess"),
                32.0f);
    glUniform1f(glGetUniformLocation(m_shaderProgram, "uWetness"),
                m_wetness);

    glBindVertexArray(m_vao);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDepthMask(GL_FALSE);

    if (m_wireframe) {
        for (uint32_t i = 0; i < m_activeStrands; i++) {
            glDrawArrays(GL_LINE_STRIP,
                         i * m_segmentsPerStrand,
                         m_segmentsPerStrand);
        }
    } else {
        uint32_t batchCount = 5000;
        uint32_t numBatches = (m_activeStrands + batchCount - 1) / batchCount;

        for (uint32_t batch = 0; batch < numBatches; batch++) {
            uint32_t startStrand = batch * batchCount;
            uint32_t endStrand = (batch + 1) * batchCount;
            if (endStrand > m_activeStrands) endStrand = m_activeStrands;
            uint32_t strandsInBatch = endStrand - startStrand;

            glMultiDrawArrays(GL_LINE_STRIP,
                              &m_multiDrawFirsts[startStrand],
                              &m_multiDrawCounts[startStrand],
                              strandsInBatch);
        }
    }

    glDisable(GL_BLEND);
    glDepthMask(GL_TRUE);

    glBindVertexArray(0);
    glUseProgram(0);

    if (m_showGuideCurves && m_guideShaderProgram && m_guideVAO) {
        glUseProgram(m_guideShaderProgram);
        glUniformMatrix4fv(glGetUniformLocation(m_guideShaderProgram, "uViewMatrix"),
                           1, GL_FALSE, glm::value_ptr(viewMatrix));
        glUniformMatrix4fv(glGetUniformLocation(m_guideShaderProgram, "uProjMatrix"),
                           1, GL_FALSE, glm::value_ptr(projMatrix));

        glBindVertexArray(m_guideVAO);
        glLineWidth(3.0f);
        glDrawArrays(GL_LINES, 0, m_maxStrands * m_segmentsPerStrand);
        glLineWidth(1.0f);
        glBindVertexArray(0);
        glUseProgram(0);
    }

    glEndQuery(GL_TIME_ELAPSED);

    GLuint64 timeNs = 0;
    glGetQueryObjectui64v(m_renderTimerQuery, GL_QUERY_RESULT, &timeNs);
    m_lastRenderTimeMs = (float)(timeNs / 1000000.0);
}

}
