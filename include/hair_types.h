#pragma once

#include <glm/glm.hpp>
#include <vector>
#include <cstdint>

namespace hair {

constexpr uint32_t MAX_GUIDE_CURVES = 512;
constexpr uint32_t MAX_HAIR_STRANDS = 500000;
constexpr uint32_t DEFAULT_SEGMENTS_PER_STRAND = 20;
constexpr uint32_t MAX_VORTEX_FIELDS = 5;
constexpr uint32_t MAX_COLLISION_CAPSULES = 32;
constexpr uint32_t SPATIAL_HASH_GRID_SIZE = 512;
constexpr uint32_t SPATIAL_HASH_BUCKET_SIZE = 16;
constexpr uint32_t SPATIAL_HASH_NUM_BUCKETS = 2097169;  // Prime ~2M for hash table

struct Vec3 {
    float x, y, z;

    __host__ __device__ Vec3() : x(0), y(0), z(0) {}
    __host__ __device__ Vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
    __host__ __device__ Vec3(const glm::vec3& v) : x(v.x), y(v.y), z(v.z) {}

    __host__ __device__ glm::vec3 to_glm() const { return glm::vec3(x, y, z); }

    __host__ __device__ Vec3 operator+(const Vec3& other) const {
        return Vec3(x + other.x, y + other.y, z + other.z);
    }
    __host__ __device__ Vec3 operator-(const Vec3& other) const {
        return Vec3(x - other.x, y - other.y, z - other.z);
    }
    __host__ __device__ Vec3 operator*(float s) const {
        return Vec3(x * s, y * s, z * s);
    }
    __host__ __device__ Vec3 operator/(float s) const {
        return Vec3(x / s, y / s, z / s);
    }
    __host__ __device__ Vec3& operator+=(const Vec3& other) {
        x += other.x; y += other.y; z += other.z;
        return *this;
    }
    __host__ __device__ Vec3& operator-=(const Vec3& other) {
        x -= other.x; y -= other.y; z -= other.z;
        return *this;
    }
};

__host__ __device__ inline Vec3 operator*(float s, const Vec3& v) {
    return v * s;
}

__host__ __device__ inline float dot(const Vec3& a, const Vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ inline Vec3 cross(const Vec3& a, const Vec3& b) {
    return Vec3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

__host__ __device__ inline float length(const Vec3& v) {
    return sqrtf(dot(v, v));
}

__host__ __device__ inline float lengthSq(const Vec3& v) {
    return dot(v, v);
}

__host__ __device__ inline Vec3 normalize(const Vec3& v) {
    float len = length(v);
    if (len < 1e-6f) return Vec3(0, 0, 0);
    return v / len;
}

struct StrandParticle {
    Vec3 position;
    Vec3 velocity;
    Vec3 oldPosition;
    float mass;
    float radius;
};

struct GuideCurve {
    uint32_t strandId;
    uint32_t numSegments;
    float length;
    Vec3 rootPosition;
};

struct VortexField {
    Vec3 position;
    Vec3 axis;
    float radius;
    float strength;
    float falloff;
    bool enabled;
};

struct CapsuleCollider {
    Vec3 start;
    Vec3 end;
    float radius;
    bool enabled;
};

struct SimulationParams {
    float gravity;
    float bendStiffness;
    float twistStiffness;
    float damping;
    float hairRadius;
    float massPerParticle;
    float timeStep;
    uint32_t numIterations;
    Vec3 globalWind;
    float windStrength;
    uint32_t numGuideCurves;
    uint32_t numHairStrands;
    uint32_t segmentsPerStrand;
    bool enableSelfCollision;
    float selfCollisionRadius;
    float friction;
    float wetness;
    float wetClumpingStrength;
    float wetDampingBoost;
    float wetStiffnessBoost;
};

struct GPUInterpolationData {
    uint32_t guideIndex0;
    uint32_t guideIndex1;
    uint32_t guideIndex2;
    float weight0;
    float weight1;
    float weight2;
    float rootOffsetX;
    float rootOffsetZ;
};

struct MaterialFrame {
    Vec3 tangent;
    Vec3 normal;
    Vec3 binormal;
};

struct SpatialHashEntry {
    uint32_t particleIndex;
    uint32_t nextEntry;
};

struct SpatialHashGrid {
    int32_t cellSize;
    uint32_t gridSize;
    uint32_t numBuckets;
    uint32_t* bucketStart;
    uint32_t* bucketCount;
    SpatialHashEntry* entries;
    uint32_t* sortedIndices;
};

struct RenderStrandData {
    std::vector<glm::vec3> positions;
    std::vector<float> thickness;
    std::vector<glm::vec3> colors;
};

struct SharedMemoryHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t numHairStrands;
    uint32_t segmentsPerStrand;
    uint32_t frameNumber;
    float deltaTime;
    uint32_t dataOffset;
    uint32_t dataSize;
};

struct PerformanceTimings {
    float simTimeMs;
    float renderTimeMs;
    float totalFrameMs;
    float uploadTimeMs;
    float interpolationTimeMs;
    float fps;
    uint32_t frameCount;
};

struct AlembicHeader {
    char magic[8];
    uint32_t version;
    uint32_t numStrands;
    uint32_t segmentsPerStrand;
    uint32_t fps;
    uint32_t totalFrames;
    float startTime;
    float endTime;
    uint32_t compressionType;
    uint32_t reserved[8];
};

}
