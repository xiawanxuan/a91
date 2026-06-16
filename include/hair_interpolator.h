#pragma once

#include "hair_types.h"
#include <memory>
#include <vector>
#include <glm/glm.hpp>

namespace hair {

class HairSimulationGPU;

class HairInterpolator {
public:
    HairInterpolator();
    ~HairInterpolator();

    bool initialize(uint32_t numGuideCurves,
                    uint32_t numHairStrands,
                    uint32_t segmentsPerStrand);

    void shutdown();

    void setGuideCurve(uint32_t index, const std::vector<Vec3>& points);

    void interpolate(const std::vector<Vec3>& guidePositions,
                      std::vector<Vec3>& hairPositions);

    void interpolateOnGPU(HairSimulationGPU* simulation);

    void setInterpolationMode(const std::string& mode) { m_interpMode = mode; }
    const std::string& getInterpolationMode() const { return m_interpMode; }

    void setSmoothingIterations(uint32_t iterations) { m_smoothingIterations = iterations; }
    uint32_t getSmoothingIterations() const { return m_smoothingIterations; }

    uint32_t getNumGuideCurves() const { return m_numGuideCurves; }
    uint32_t getNumHairStrands() const { return m_numHairStrands; }
    uint32_t getSegmentsPerStrand() const { return m_segmentsPerStrand; }

    const std::vector<uint32_t>& getGuideIndices() const { return m_guideIndices; }
    const std::vector<float>& getGuideWeights() const { return m_guideWeights; }
    const std::vector<glm::vec2>& getBarycentricCoords() const { return m_barycentricCoords; }
    const std::vector<GPUInterpolationData>& getGPUInterpData() const { return m_gpuInterpData; }
    const std::vector<Vec3>& getGuidePoints() const { return m_guidePoints; }

private:
    void buildInterpolationMap();
    void computeBarycentricWeights();
    void generateRootPositions();
    void buildGPUInterpolationData();
    void interpolateStrand(uint32_t hairIndex,
                           const std::vector<Vec3>& guidePositions,
                           std::vector<Vec3>& hairPositions);

    uint32_t m_numGuideCurves;
    uint32_t m_numHairStrands;
    uint32_t m_segmentsPerStrand;

    std::vector<Vec3> m_guideRootPositions;
    std::vector<float> m_guideLengths;
    std::vector<Vec3> m_guidePoints;

    std::vector<uint32_t> m_guideIndices;
    std::vector<float> m_guideWeights;
    std::vector<glm::vec2> m_barycentricCoords;

    std::vector<GPUInterpolationData> m_gpuInterpData;

    std::string m_interpMode;
    uint32_t m_smoothingIterations;

    bool m_initialized;
    bool m_gpuDataDirty;
};

}
