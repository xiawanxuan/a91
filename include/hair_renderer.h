#pragma once

#include "hair_types.h"
#include <glm/glm.hpp>
#include <memory>
#include <vector>
#include <string>

namespace hair {

class HairRenderer {
public:
    HairRenderer();
    ~HairRenderer();

    bool initialize(uint32_t maxStrands, uint32_t segmentsPerStrand);
    void shutdown();

    void updateBuffers(const std::vector<glm::vec3>& positions);
    void render(const glm::mat4& viewMatrix, const glm::mat4& projMatrix,
                const glm::vec3& cameraPos);

    void setHairColor(const glm::vec3& color) { m_hairColor = color; }
    void setHairThickness(float thickness) { m_hairThickness = thickness; }
    void setTipScale(float scale) { m_tipScale = scale; }
    void setWireframe(bool enabled) { m_wireframe = enabled; }
    void setShowGuideCurves(bool show) { m_showGuideCurves = show; }
    void setWetness(float wetness) { m_wetness = wetness; }

    uint32_t getMaxStrands() const { return m_maxStrands; }
    uint32_t getSegmentsPerStrand() const { return m_segmentsPerStrand; }

    float getLastRenderTimeMs() const { return m_lastRenderTimeMs; }

    void updateGuideCurveBuffers(const std::vector<glm::vec3>& guidePositions,
                                  uint32_t numGuideCurves,
                                  uint32_t segmentsPerStrand);

private:
    bool createShaderProgram();
    bool createGuideShaderProgram();
    bool createBuffers();
    void destroyBuffers();
    void computeMultiDrawParams();

    uint32_t m_vao;
    uint32_t m_positionVBO;
    uint32_t m_strandIndexVBO;
    uint32_t m_segmentIndexVBO;
    uint32_t m_shaderProgram;
    uint32_t m_guideShaderProgram;

    uint32_t m_guideVAO;
    uint32_t m_guideVBO;

    uint32_t m_maxStrands;
    uint32_t m_segmentsPerStrand;
    uint32_t m_totalVertices;

    glm::vec3 m_hairColor;
    float m_hairThickness;
    float m_tipScale;
    bool m_wireframe;
    bool m_showGuideCurves;
    float m_wetness;

    std::vector<int32_t> m_multiDrawFirsts;
    std::vector<int32_t> m_multiDrawCounts;
    uint32_t m_activeStrands;

    bool m_initialized;

    float m_lastRenderTimeMs;

    uint32_t m_renderTimerQuery;
};

}
