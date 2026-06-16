#pragma once

#include "hair_types.h"
#include <memory>
#include <string>

namespace hair {

class HairSimulationGPU {
public:
    HairSimulationGPU();
    ~HairSimulationGPU();

    bool initialize(const SimulationParams& params);
    void shutdown();

    void update(float deltaTime);

    void setGuideCurve(uint32_t index, const std::vector<Vec3>& points);
    void setVortexField(uint32_t index, const VortexField& field);
    void setCapsuleCollider(uint32_t index, const CapsuleCollider& collider);

    const SimulationParams& getParams() const { return m_params; }
    void setParams(const SimulationParams& params);

    StrandParticle* getDeviceParticles() const { return m_d_particles; }
    Vec3* getDevicePositions() const { return m_d_positionsOnly; }
    uint32_t getNumParticles() const { return m_numParticles; }
    uint32_t getNumHairStrands() const { return m_params.numHairStrands; }
    uint32_t getSegmentsPerStrand() const { return m_params.segmentsPerStrand; }

    void downloadPositions(std::vector<glm::vec3>& positions);

    void setInterpolationData(const std::vector<GPUInterpolationData>& interpData);
    void setGuideCurvePositions(const std::vector<Vec3>& guidePositions);

    void reset();

    float getLastSimTimeMs() const { return m_lastSimTimeMs; }

    uint32_t getNumVortexFields() const { return m_numVortexFields; }
    uint32_t getNumColliders() const { return m_numColliders; }

private:
    void allocateMemory();
    void freeMemory();
    void initializeStrands();
    void computePhysics(float dt);
    void applyWindForces();
    void handleCollisions();
    void buildSpatialHash();

    SimulationParams m_params;
    uint32_t m_numParticles;
    bool m_initialized;

    StrandParticle* m_d_particles;
    StrandParticle* m_d_oldParticles;
    Vec3* m_d_constraintPositions;
    uint32_t* m_d_strandOffsets;
    GuideCurve* m_d_guideCurves;
    VortexField* m_d_vortexFields;
    CapsuleCollider* m_d_colliders;

    float* m_d_restLengths;
    Vec3* m_d_restTangents;
    MaterialFrame* m_d_restFrames;

    Vec3* m_d_positionsOnly;

    GPUInterpolationData* m_d_interpData;
    Vec3* m_d_guidePositions;
    uint32_t m_numGuideCurvesStored;

    uint32_t* m_d_spatialHashBuckets;
    uint32_t* m_d_spatialHashCounts;
    uint32_t* m_d_spatialHashSorted;
    uint32_t* m_d_spatialHashParticleCells;

    uint32_t m_numGuideCurves;
    uint32_t m_numVortexFields;
    uint32_t m_numColliders;

    float m_lastSimTimeMs;

    cudaEvent_t m_startEvent;
    cudaEvent_t m_stopEvent;
};

}
