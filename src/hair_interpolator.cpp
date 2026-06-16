#include "hair_interpolator.h"
#include "hair_simulation.h"
#include <glm/glm.hpp>
#include <glm/gtc/random.hpp>
#include <cmath>
#include <random>
#include <algorithm>

namespace hair {

HairInterpolator::HairInterpolator()
    : m_numGuideCurves(0)
    , m_numHairStrands(0)
    , m_segmentsPerStrand(0)
    , m_interpMode("barycentric")
    , m_smoothingIterations(2)
    , m_initialized(false)
    , m_gpuDataDirty(true)
{
}

HairInterpolator::~HairInterpolator() {
    shutdown();
}

bool HairInterpolator::initialize(uint32_t numGuideCurves,
                                   uint32_t numHairStrands,
                                   uint32_t segmentsPerStrand) {
    m_numGuideCurves = numGuideCurves;
    m_numHairStrands = numHairStrands;
    m_segmentsPerStrand = segmentsPerStrand;

    m_guideRootPositions.resize(numGuideCurves);
    m_guideLengths.resize(numGuideCurves);
    m_guidePoints.resize(numGuideCurves * segmentsPerStrand);

    m_guideIndices.resize(numHairStrands * 3);
    m_guideWeights.resize(numHairStrands * 3);
    m_barycentricCoords.resize(numHairStrands);
    m_gpuInterpData.resize(numHairStrands);

    buildInterpolationMap();
    generateRootPositions();
    buildGPUInterpolationData();

    m_initialized = true;
    return true;
}

void HairInterpolator::shutdown() {
    m_initialized = false;
}

void HairInterpolator::setGuideCurve(uint32_t index, const std::vector<Vec3>& points) {
    if (index >= m_numGuideCurves) return;
    if (points.empty()) return;

    m_guideRootPositions[index] = points[0];

    float totalLength = 0.0f;
    for (size_t i = 1; i < points.size(); i++) {
        Vec3 diff = points[i] - points[i - 1];
        totalLength += length(diff);
    }
    m_guideLengths[index] = totalLength;

    uint32_t segsPerStrand = m_segmentsPerStrand;
    uint32_t baseIdx = index * segsPerStrand;

    if (points.size() >= segsPerStrand) {
        for (uint32_t s = 0; s < segsPerStrand; s++) {
            m_guidePoints[baseIdx + s] = points[s];
        }
    } else {
        for (uint32_t s = 0; s < segsPerStrand; s++) {
            float t = (float)s / (segsPerStrand - 1);
            float srcT = t * (points.size() - 1);
            uint32_t seg0 = (uint32_t)srcT;
            if (seg0 >= points.size() - 1) seg0 = (uint32_t)points.size() - 2;
            float frac = srcT - seg0;
            m_guidePoints[baseIdx + s] = points[seg0] * (1.0f - frac) + points[seg0 + 1] * frac;
        }
    }

    m_gpuDataDirty = true;
}

void HairInterpolator::buildInterpolationMap() {
    std::vector<glm::vec2> guideUVs(m_numGuideCurves);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (uint32_t i = 0; i < m_numGuideCurves; i++) {
        float angle = (float)i / m_numGuideCurves * 6.283185307f;
        float radius = 0.3f + dist(rng) * 0.7f;
        guideUVs[i] = glm::vec2(0.5f + cosf(angle) * radius * 0.5f,
                                0.5f + sinf(angle) * radius * 0.5f);
    }

    for (uint32_t i = 0; i < m_numHairStrands; i++) {
        float u = dist(rng);
        float v = dist(rng);
        m_barycentricCoords[i] = glm::vec2(u, v);

        std::vector<std::pair<float, uint32_t>> distances;
        distances.reserve(m_numGuideCurves);

        for (uint32_t g = 0; g < m_numGuideCurves; g++) {
            float dx = u - guideUVs[g].x;
            float dy = v - guideUVs[g].y;
            float d = sqrtf(dx * dx + dy * dy);
            distances.push_back({ d, g });
        }

        std::sort(distances.begin(), distances.end());

        for (int k = 0; k < 3; k++) {
            m_guideIndices[i * 3 + k] = distances[k].second;
        }

        float d0 = distances[0].first;
        float d1 = distances[1].first;
        float d2 = distances[2].first;

        float w0 = 1.0f / (d0 + 0.01f);
        float w1 = 1.0f / (d1 + 0.01f);
        float w2 = 1.0f / (d2 + 0.01f);

        float sum = w0 + w1 + w2;
        m_guideWeights[i * 3 + 0] = w0 / sum;
        m_guideWeights[i * 3 + 1] = w1 / sum;
        m_guideWeights[i * 3 + 2] = w2 / sum;
    }
}

void HairInterpolator::generateRootPositions() {
    for (uint32_t i = 0; i < m_numGuideCurves; i++) {
        float angle = (float)i / m_numGuideCurves * 6.283185307f;
        float radius = 0.05f + (i % 16) * 0.01f;
        m_guideRootPositions[i] = Vec3(
            cosf(angle) * radius,
            0.0f,
            sinf(angle) * radius
        );
        m_guideLengths[i] = 1.0f;

        uint32_t baseIdx = i * m_segmentsPerStrand;
        for (uint32_t s = 0; s < m_segmentsPerStrand; s++) {
            float t = (float)s / m_segmentsPerStrand;
            m_guidePoints[baseIdx + s] = Vec3(
                cosf(angle) * radius,
                t * 1.0f,
                sinf(angle) * radius
            );
        }
    }
}

void HairInterpolator::buildGPUInterpolationData() {
    for (uint32_t i = 0; i < m_numHairStrands; i++) {
        GPUInterpolationData& data = m_gpuInterpData[i];
        data.guideIndex0 = m_guideIndices[i * 3 + 0];
        data.guideIndex1 = m_guideIndices[i * 3 + 1];
        data.guideIndex2 = m_guideIndices[i * 3 + 2];
        data.weight0 = m_guideWeights[i * 3 + 0];
        data.weight1 = m_guideWeights[i * 3 + 1];
        data.weight2 = m_guideWeights[i * 3 + 2];

        std::mt19937 rng(i + 12345);
        std::uniform_real_distribution<float> dist(-0.01f, 0.01f);
        data.rootOffsetX = dist(rng);
        data.rootOffsetZ = dist(rng);
    }
    m_gpuDataDirty = true;
}

void HairInterpolator::interpolate(const std::vector<Vec3>& guidePositions,
                                    std::vector<Vec3>& hairPositions) {
    uint32_t totalHairParticles = m_numHairStrands * m_segmentsPerStrand;
    hairPositions.resize(totalHairParticles);

    uint32_t guideSegments = m_segmentsPerStrand;

    for (uint32_t hairIdx = 0; hairIdx < m_numHairStrands; hairIdx++) {
        uint32_t g0 = m_guideIndices[hairIdx * 3 + 0];
        uint32_t g1 = m_guideIndices[hairIdx * 3 + 1];
        uint32_t g2 = m_guideIndices[hairIdx * 3 + 2];

        float w0 = m_guideWeights[hairIdx * 3 + 0];
        float w1 = m_guideWeights[hairIdx * 3 + 1];
        float w2 = m_guideWeights[hairIdx * 3 + 2];

        for (uint32_t seg = 0; seg < m_segmentsPerStrand; seg++) {
            uint32_t hpIdx = hairIdx * m_segmentsPerStrand + seg;

            uint32_t gp0 = g0 * guideSegments + seg;
            uint32_t gp1 = g1 * guideSegments + seg;
            uint32_t gp2 = g2 * guideSegments + seg;

            if (gp0 < guidePositions.size() &&
                gp1 < guidePositions.size() &&
                gp2 < guidePositions.size()) {

                Vec3 pos = guidePositions[gp0] * w0 +
                           guidePositions[gp1] * w1 +
                           guidePositions[gp2] * w2;

                hairPositions[hpIdx] = pos;
            }
        }
    }

    for (uint32_t iter = 0; iter < m_smoothingIterations; iter++) {
        std::vector<Vec3> smoothed = hairPositions;

        for (uint32_t hairIdx = 0; hairIdx < m_numHairStrands; hairIdx++) {
            for (uint32_t seg = 1; seg < m_segmentsPerStrand - 1; seg++) {
                uint32_t idx = hairIdx * m_segmentsPerStrand + seg;
                uint32_t prev = idx - 1;
                uint32_t next = idx + 1;

                smoothed[idx] = (hairPositions[prev] + hairPositions[next]) * 0.5f * 0.3f +
                                hairPositions[idx] * 0.7f;
            }
        }

        hairPositions.swap(smoothed);
    }
}

void HairInterpolator::interpolateOnGPU(HairSimulationGPU* simulation) {
    if (!simulation || !m_initialized) return;

    if (m_gpuDataDirty) {
        simulation->setInterpolationData(m_gpuInterpData);
        simulation->setGuideCurvePositions(m_guidePoints);
        m_gpuDataDirty = false;
    }
}

void HairInterpolator::computeBarycentricWeights() {
}

void HairInterpolator::interpolateStrand(uint32_t hairIndex,
                                          const std::vector<Vec3>& guidePositions,
                                          std::vector<Vec3>& hairPositions) {
}

}
