#include "hair_simulation.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <cmath>
#include <cstring>
#include <iostream>

namespace hair {

__global__ void initParticlesKernel(
    StrandParticle* particles,
    uint32_t numStrands,
    uint32_t segmentsPerStrand,
    Vec3 basePos,
    float strandLength,
    float massPerParticle)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t totalParticles = numStrands * segmentsPerStrand;

    if (idx >= totalParticles) return;

    uint32_t strandIdx = idx / segmentsPerStrand;
    uint32_t segIdx = idx % segmentsPerStrand;

    float t = (float)segIdx / segmentsPerStrand;
    float segLength = strandLength / segmentsPerStrand;

    float angle = strandIdx * 0.137f * 3.14159f * 2.0f;
    float radius = 0.02f + (strandIdx % 16) * 0.01f;

    Vec3 rootOffset(
        cosf(angle) * radius,
        0.0f,
        sinf(angle) * radius
    );

    Vec3 pos = basePos + rootOffset;
    pos.y += t * strandLength;

    float jitter = sinf(strandIdx * 7.13f + segIdx * 3.7f) * 0.002f;
    pos.x += jitter;
    pos.z += jitter * 0.5f;

    particles[idx].position = pos;
    particles[idx].oldPosition = pos;
    particles[idx].velocity = Vec3(0, 0, 0);
    particles[idx].mass = massPerParticle;
    particles[idx].radius = 0.005f;
}

__global__ void extractPositionsKernel(
    const StrandParticle* particles,
    Vec3* positionsOnly,
    uint32_t totalParticles)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalParticles) return;

    positionsOnly[idx] = particles[idx].position;
}

__global__ void applyGravityKernel(
    StrandParticle* particles,
    uint32_t totalParticles,
    Vec3 gravity,
    float dt)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalParticles) return;

    StrandParticle& p = particles[idx];
    p.velocity = p.velocity + gravity * dt;
}

__global__ void applyDampingKernel(
    StrandParticle* particles,
    uint32_t totalParticles,
    float damping,
    float dt)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalParticles) return;

    StrandParticle& p = particles[idx];
    float damp = expf(-damping * dt);
    p.velocity = p.velocity * damp;
}

__global__ void integratePositionsKernel(
    StrandParticle* particles,
    uint32_t totalParticles,
    float dt)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalParticles) return;

    StrandParticle& p = particles[idx];
    p.position = p.position + p.velocity * dt;
}

__global__ void applyGlobalWindKernel(
    StrandParticle* particles,
    uint32_t totalParticles,
    Vec3 windDir,
    float windStrength,
    float dt,
    float hairRadius)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalParticles) return;

    StrandParticle& p = particles[idx];

    uint32_t segIdx = idx % 20;
    float segT = (float)segIdx / 20.0f;
    float windMultiplier = 0.3f + 0.7f * segT;

    float timePhase = idx * 0.0001f;
    Vec3 turbulence(
        sinf(timePhase * 3.7f) * 0.3f,
        cosf(timePhase * 2.1f) * 0.1f,
        sinf(timePhase * 5.3f) * 0.3f
    );

    Vec3 effectiveWind = windDir * windStrength + turbulence * windStrength * 0.2f;
    Vec3 relativeVel = effectiveWind - p.velocity;
    float speed = length(relativeVel);

    if (speed < 1e-5f) return;

    float dragCoeff = 0.5f;
    float area = 3.14159f * hairRadius * hairRadius;
    float airDensity = 1.225f;

    float dragForceMag = 0.5f * airDensity * speed * speed * dragCoeff * area * windMultiplier;
    Vec3 dragForce = normalize(relativeVel) * dragForceMag;

    Vec3 acceleration = dragForce / p.mass;
    p.velocity = p.velocity + acceleration * dt;
}

__global__ void applyVortexWindKernel(
    StrandParticle* particles,
    uint32_t totalParticles,
    VortexField* vortexFields,
    uint32_t numVortexFields,
    float dt,
    float hairRadius)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalParticles) return;

    StrandParticle& p = particles[idx];

    for (uint32_t v = 0; v < numVortexFields; v++) {
        VortexField& vortex = vortexFields[v];
        if (!vortex.enabled) continue;

        Vec3 toParticle = p.position - vortex.position;
        float dist = length(toParticle);

        if (dist > vortex.radius * 2.0f) continue;

        Vec3 radialDir = normalize(toParticle);
        Vec3 tangentialDir = cross(vortex.axis, radialDir);

        float tanLen = length(tangentialDir);
        if (tanLen < 1e-6f) continue;
        tangentialDir = tangentialDir / tanLen;

        float falloff = 1.0f - dist / (vortex.radius * 2.0f);
        if (falloff < 0.0f) falloff = 0.0f;
        falloff = powf(falloff, vortex.falloff);

        float tangentialSpeed = vortex.strength * falloff;
        Vec3 windVel = tangentialDir * tangentialSpeed;

        Vec3 axisVel = vortex.axis * (vortex.strength * 0.3f * falloff);
        windVel = windVel + axisVel;

        Vec3 relativeVel = windVel - p.velocity;
        float speed = length(relativeVel);

        if (speed < 1e-5f) continue;

        float dragCoeff = 0.5f;
        float area = 3.14159f * hairRadius * hairRadius;
        float airDensity = 1.225f;

        float dragForceMag = 0.5f * airDensity * speed * speed * dragCoeff * area;
        Vec3 dragForce = normalize(relativeVel) * dragForceMag;

        Vec3 acceleration = dragForce / p.mass;
        p.velocity = p.velocity + acceleration * dt;
    }
}

__global__ void distanceConstraintsKernel(
    StrandParticle* particles,
    uint32_t numStrands,
    uint32_t segmentsPerStrand,
    float* restLengths,
    float stiffness)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= numStrands) return;

    for (uint32_t iter = 0; iter < 4; iter++) {
        for (uint32_t seg = 0; seg < segmentsPerStrand - 1; seg++) {
            uint32_t i0 = idx * segmentsPerStrand + seg;
            uint32_t i1 = i0 + 1;

            StrandParticle& p0 = particles[i0];
            StrandParticle& p1 = particles[i1];

            Vec3 delta = p1.position - p0.position;
            float dist = length(delta);

            if (dist < 1e-6f) continue;

            float restLen = restLengths[i0];
            float diff = (dist - restLen) / dist;

            float w0 = 1.0f / p0.mass;
            float w1 = 1.0f / p1.mass;
            float wSum = w0 + w1;

            if (wSum < 1e-6f) continue;

            Vec3 correction = delta * diff * stiffness;

            if (seg == 0) {
                p1.position = p1.position - correction * (w1 / wSum);
            } else {
                p0.position = p0.position + correction * (w0 / wSum);
                p1.position = p1.position - correction * (w1 / wSum);
            }
        }
    }
}

__global__ void bendConstraintsKernel(
    StrandParticle* particles,
    uint32_t numStrands,
    uint32_t segmentsPerStrand,
    Vec3* restTangents,
    float bendStiffness)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= numStrands) return;

    for (uint32_t seg = 1; seg < segmentsPerStrand - 1; seg++) {
        uint32_t i0 = idx * segmentsPerStrand + seg - 1;
        uint32_t i1 = idx * segmentsPerStrand + seg;
        uint32_t i2 = idx * segmentsPerStrand + seg + 1;

        StrandParticle& p0 = particles[i0];
        StrandParticle& p1 = particles[i1];
        StrandParticle& p2 = particles[i2];

        Vec3 e0 = p1.position - p0.position;
        Vec3 e1 = p2.position - p1.position;

        float len0 = length(e0);
        float len1 = length(e1);

        if (len0 < 1e-6f || len1 < 1e-6f) continue;

        Vec3 t0 = e0 / len0;
        Vec3 t1 = e1 / len1;

        float cosTheta = dot(t0, t1);
        cosTheta = fminf(1.0f, fmaxf(-1.0f, cosTheta));

        Vec3 restTan = restTangents[seg];
        float restCos = dot(t0, restTan);
        restCos = fminf(1.0f, fmaxf(-1.0f, restCos));

        float targetAngle = acosf(restCos);
        float currentAngle = acosf(cosTheta);

        float angleDiff = currentAngle - targetAngle;

        if (fabsf(angleDiff) < 1e-4f) continue;

        Vec3 bendAxis = cross(t0, t1);
        float axisLen = length(bendAxis);

        if (axisLen < 1e-6f) continue;

        bendAxis = bendAxis / axisLen;

        float correction = bendStiffness * angleDiff * 0.5f;

        Vec3 rot0 = cross(bendAxis, e0) * correction;
        Vec3 rot1 = cross(bendAxis, e1) * correction;

        p0.position = p0.position + rot0 * 0.25f;
        p1.position = p1.position - rot0 * 0.25f;
        p1.position = p1.position + rot1 * 0.25f;
        p2.position = p2.position - rot1 * 0.25f;
    }
}

__global__ void twistConstraintsKernel(
    StrandParticle* particles,
    uint32_t numStrands,
    uint32_t segmentsPerStrand,
    MaterialFrame* restFrames,
    float twistStiffness)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= numStrands) return;

    for (uint32_t seg = 1; seg < segmentsPerStrand - 2; seg++) {
        uint32_t i0 = idx * segmentsPerStrand + seg;
        uint32_t i1 = i0 + 1;
        uint32_t i2 = i0 + 2;

        Vec3 p0 = particles[i0].position;
        Vec3 p1 = particles[i1].position;
        Vec3 p2 = particles[i2].position;

        Vec3 e0 = p1 - p0;
        Vec3 e1 = p2 - p1;

        float len0 = length(e0);
        float len1 = length(e1);

        if (len0 < 1e-6f || len1 < 1e-6f) continue;

        Vec3 t0 = e0 / len0;
        Vec3 t1 = e1 / len1;

        Vec3 b0 = cross(t0, t1);
        float bLen = length(b0);
        if (bLen < 1e-6f) continue;
        b0 = b0 / bLen;

        Vec3 n0 = cross(b0, t0);
        n0 = normalize(n0);

        MaterialFrame restFrame = restFrames[seg];
        Vec3 restN = restFrame.normal;

        float twistAngle = atan2f(dot(cross(restN, n0), t0), dot(restN, n0));

        if (fabsf(twistAngle) < 1e-4f) continue;

        float correction = twistStiffness * twistAngle * 0.3f;

        Vec3 twistCorrection = t0 * correction;

        particles[i1].position = particles[i1].position - twistCorrection * 0.25f;
        particles[i2].position = particles[i2].position + twistCorrection * 0.25f;
    }
}

__global__ void fixRootParticlesKernel(
    StrandParticle* particles,
    StrandParticle* oldParticles,
    uint32_t numStrands,
    uint32_t segmentsPerStrand)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= numStrands) return;

    uint32_t rootIdx = idx * segmentsPerStrand;
    particles[rootIdx].position = oldParticles[rootIdx].position;
    particles[rootIdx].velocity = Vec3(0, 0, 0);
}

__global__ void capsuleCollisionKernel(
    StrandParticle* particles,
    uint32_t totalParticles,
    CapsuleCollider* colliders,
    uint32_t numColliders,
    float friction)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalParticles) return;

    StrandParticle& p = particles[idx];

    for (uint32_t c = 0; c < numColliders; c++) {
        CapsuleCollider& col = colliders[c];
        if (!col.enabled) continue;

        Vec3 ab = col.end - col.start;
        Vec3 ap = p.position - col.start;

        float t = dot(ap, ab) / dot(ab, ab);
        t = fminf(1.0f, fmaxf(0.0f, t));

        Vec3 closestPoint = col.start + ab * t;
        Vec3 diff = p.position - closestPoint;
        float dist = length(diff);

        float totalRadius = col.radius + p.radius;

        if (dist < totalRadius && dist > 1e-6f) {
            Vec3 normal = diff / dist;
            float penetration = totalRadius - dist;

            p.position = p.position + normal * penetration;

            Vec3 velNormal = normal * dot(p.velocity, normal);
            Vec3 velTangent = p.velocity - velNormal;

            p.velocity = velTangent * (1.0f - friction);
            p.velocity = p.velocity + velNormal * -0.3f;
        }
    }
}

__device__ __forceinline__ uint32_t computeCellHash(
    int32_t cx, int32_t cy, int32_t cz,
    uint32_t gridSize,
    uint32_t numBuckets);

__global__ void selfCollisionKernel(
    StrandParticle* particles,
    const uint32_t* sortedIndices,
    const uint32_t* bucketStart,
    const uint32_t* bucketCounts,
    const uint32_t* particleStrandIds,
    uint32_t totalParticles,
    float collisionRadius,
    float stiffness,
    uint32_t segmentsPerStrand)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalParticles) return;

    uint32_t seg0 = idx % segmentsPerStrand;
    if (seg0 == 0) return;

    uint32_t strand0 = particleStrandIds[idx];
    StrandParticle& p0 = particles[idx];
    Vec3 pos0 = p0.position;

    float cellSize = collisionRadius * 3.0f;
    int32_t cx = (int32_t)floorf(pos0.x / cellSize);
    int32_t cy = (int32_t)floorf(pos0.y / cellSize);
    int32_t cz = (int32_t)floorf(pos0.z / cellSize);

    uint32_t neighborCells[27];
    #pragma unroll
    for (int i = 0; i < 27; i++) {
        int32_t dz = (i / 9) - 1;
        int32_t dy = ((i % 9) / 3) - 1;
        int32_t dx = (i % 3) - 1;
        neighborCells[i] = computeCellHash(
            cx + dx, cy + dy, cz + dz,
            SPATIAL_HASH_GRID_SIZE,
            SPATIAL_HASH_NUM_BUCKETS);
    }

    float minDist = collisionRadius * 2.0f;
    float minDistSq = minDist * minDist;

    Vec3 totalCorrection(0, 0, 0);
    uint32_t collisionCount = 0;
    const uint32_t bucketCap = SPATIAL_HASH_BUCKET_SIZE;

    #pragma unroll
    for (int nc = 0; nc < 27; nc++) {
        uint32_t hashVal = neighborCells[nc];
        uint32_t start = bucketStart[hashVal];
        uint32_t count = bucketCounts[hashVal];

        if (count > bucketCap) count = bucketCap;

        for (uint32_t j = 0; j < count; j++) {
            uint32_t otherIdx = sortedIndices[start + j];

            if (otherIdx >= totalParticles || otherIdx == idx) continue;

            uint32_t strand1 = particleStrandIds[otherIdx];
            if (strand0 == strand1) continue;

            Vec3 p1pos = particles[otherIdx].position;
            Vec3 diff = pos0 - p1pos;
            float distSq = diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;

            if (distSq < minDistSq && distSq > 1e-10f) {
                float dist = sqrtf(distSq);
                float invDist = 1.0f / dist;
                float penetration = (minDist - dist) * 0.5f * stiffness;
                totalCorrection.x += diff.x * invDist * penetration;
                totalCorrection.y += diff.y * invDist * penetration;
                totalCorrection.z += diff.z * invDist * penetration;
                collisionCount++;
            }
        }
    }

    if (collisionCount > 0) {
        float invCount = 1.0f / (float)collisionCount;
        p0.position.x += totalCorrection.x * invCount;
        p0.position.y += totalCorrection.y * invCount;
        p0.position.z += totalCorrection.z * invCount;
    }
}

__global__ void computeRestLengthsKernel(
    StrandParticle* particles,
    float* restLengths,
    uint32_t numStrands,
    uint32_t segmentsPerStrand)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numStrands * (segmentsPerStrand - 1)) return;

    uint32_t strandIdx = idx / (segmentsPerStrand - 1);
    uint32_t segIdx = idx % (segmentsPerStrand - 1);

    uint32_t i0 = strandIdx * segmentsPerStrand + segIdx;
    uint32_t i1 = i0 + 1;

    Vec3 delta = particles[i1].position - particles[i0].position;
    restLengths[idx] = length(delta);
}

__global__ void computeRestTangentsKernel(
    StrandParticle* particles,
    Vec3* restTangents,
    MaterialFrame* restFrames,
    uint32_t numStrands,
    uint32_t segmentsPerStrand)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numStrands * segmentsPerStrand) return;

    uint32_t strandIdx = idx / segmentsPerStrand;
    uint32_t segIdx = idx % segmentsPerStrand;

    if (segIdx >= segmentsPerStrand - 1) return;

    uint32_t i0 = strandIdx * segmentsPerStrand + segIdx;
    uint32_t i1 = i0 + 1;

    Vec3 delta = particles[i1].position - particles[i0].position;
    float len = length(delta);

    if (len > 1e-6f) {
        Vec3 tangent = delta / len;
        restTangents[segIdx] = tangent;

        Vec3 up(0, 1, 0);
        if (fabsf(dot(tangent, up)) > 0.99f) {
            up = Vec3(1, 0, 0);
        }
        Vec3 normal = normalize(cross(tangent, up));
        Vec3 binormal = cross(tangent, normal);

        restFrames[segIdx].tangent = tangent;
        restFrames[segIdx].normal = normal;
        restFrames[segIdx].binormal = binormal;
    }
}

__global__ void gpuInterpolationKernel(
    const Vec3* guidePositions,
    const GPUInterpolationData* interpData,
    Vec3* outputPositions,
    uint32_t numHairStrands,
    uint32_t segmentsPerStrand,
    uint32_t guideSegments)
{
    uint32_t hairIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (hairIdx >= numHairStrands) return;

    GPUInterpolationData interp = interpData[hairIdx];

    for (uint32_t seg = 0; seg < segmentsPerStrand; seg++) {
        uint32_t outIdx = hairIdx * segmentsPerStrand + seg;

        float t = (float)seg / (segmentsPerStrand - 1);

        uint32_t gSeg = (uint32_t)(t * (guideSegments - 1));
        if (gSeg >= guideSegments - 1) gSeg = guideSegments - 2;
        float frac = t * (guideSegments - 1) - gSeg;

        uint32_t gp0 = interp.guideIndex0 * guideSegments + gSeg;
        uint32_t gp1 = interp.guideIndex1 * guideSegments + gSeg;
        uint32_t gp2 = interp.guideIndex2 * guideSegments + gSeg;

        uint32_t gp0n = interp.guideIndex0 * guideSegments + gSeg + 1;
        uint32_t gp1n = interp.guideIndex1 * guideSegments + gSeg + 1;
        uint32_t gp2n = interp.guideIndex2 * guideSegments + gSeg + 1;

        Vec3 p0 = guidePositions[gp0] * (1.0f - frac) + guidePositions[gp0n] * frac;
        Vec3 p1 = guidePositions[gp1] * (1.0f - frac) + guidePositions[gp1n] * frac;
        Vec3 p2 = guidePositions[gp2] * (1.0f - frac) + guidePositions[gp2n] * frac;

        Vec3 pos = p0 * interp.weight0 + p1 * interp.weight1 + p2 * interp.weight2;

        float tipFactor = t * t;
        pos.x += interp.rootOffsetX * (1.0f - tipFactor);
        pos.z += interp.rootOffsetZ * (1.0f - tipFactor);

        outputPositions[outIdx] = pos;
    }
}

__global__ void initStrandIdsKernel(
    uint32_t* strandIds,
    uint32_t numStrands,
    uint32_t segmentsPerStrand)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t totalParticles = numStrands * segmentsPerStrand;
    if (idx >= totalParticles) return;
    strandIds[idx] = idx / segmentsPerStrand;
}

__device__ __forceinline__ uint32_t computeCellHash(
    int32_t cx, int32_t cy, int32_t cz,
    uint32_t gridSize,
    uint32_t numBuckets)
{
    cx = (cx % (int32_t)gridSize + (int32_t)gridSize) % (int32_t)gridSize;
    cy = (cy % (int32_t)gridSize + (int32_t)gridSize) % (int32_t)gridSize;
    cz = (cz % (int32_t)gridSize + (int32_t)gridSize) % (int32_t)gridSize;

    uint32_t hx = (uint32_t)cx * 73856093u;
    uint32_t hy = (uint32_t)cy * 19349663u;
    uint32_t hz = (uint32_t)cz * 83492791u;
    return (hx ^ hy ^ hz) % numBuckets;
}

__global__ void spatialHashAssignKernel(
    const StrandParticle* particles,
    uint32_t totalParticles,
    float cellSize,
    uint32_t gridSize,
    uint32_t numBuckets,
    uint32_t* particleCells)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalParticles) return;

    Vec3 pos = particles[idx].position;
    int32_t cx = (int32_t)floorf(pos.x / cellSize);
    int32_t cy = (int32_t)floorf(pos.y / cellSize);
    int32_t cz = (int32_t)floorf(pos.z / cellSize);

    uint32_t hashVal = computeCellHash(cx, cy, cz, gridSize, numBuckets);
    particleCells[idx] = hashVal;
}

__global__ void bucketBoundaryMarkersKernel(
    const uint32_t* sortedCells,
    uint32_t totalParticles,
    uint32_t numBuckets,
    uint32_t* bucketStart,
    uint32_t* bucketCounts)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t totalWork = totalParticles + 1;
    if (idx >= totalWork) return;

    if (idx == 0) {
        if (totalParticles > 0) {
            uint32_t firstCell = sortedCells[0];
            if (firstCell < numBuckets) {
                bucketStart[firstCell] = 0;
            }
        }
        return;
    }

    if (idx >= totalParticles) return;

    uint32_t curCell = sortedCells[idx];
    uint32_t prevCell = sortedCells[idx - 1];

    if (curCell != prevCell) {
        if (curCell < numBuckets) {
            bucketStart[curCell] = idx;
        }
    }
}

__global__ void countBucketSizesKernel(
    const uint32_t* sortedCells,
    uint32_t totalParticles,
    uint32_t numBuckets,
    const uint32_t* bucketStart,
    uint32_t* bucketCounts)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalParticles) return;

    uint32_t curCell = sortedCells[idx];
    if (curCell >= numBuckets) return;

    if (idx == totalParticles - 1) {
        uint32_t start = bucketStart[curCell];
        bucketCounts[curCell] = totalParticles - start;
        return;
    }

    uint32_t nextCell = sortedCells[idx + 1];
    if (curCell != nextCell) {
        uint32_t start = bucketStart[curCell];
        bucketCounts[curCell] = idx + 1 - start;
    }
}

HairSimulationGPU::HairSimulationGPU()
    : m_numParticles(0)
    , m_initialized(false)
    , m_d_particles(nullptr)
    , m_d_oldParticles(nullptr)
    , m_d_constraintPositions(nullptr)
    , m_d_strandOffsets(nullptr)
    , m_d_guideCurves(nullptr)
    , m_d_vortexFields(nullptr)
    , m_d_colliders(nullptr)
    , m_d_restLengths(nullptr)
    , m_d_restTangents(nullptr)
    , m_d_restFrames(nullptr)
    , m_d_positionsOnly(nullptr)
    , m_d_interpData(nullptr)
    , m_d_guidePositions(nullptr)
    , m_numGuideCurvesStored(0)
    , m_d_spatialHashBuckets(nullptr)
    , m_d_spatialHashCounts(nullptr)
    , m_d_spatialHashSorted(nullptr)
    , m_d_spatialHashParticleCells(nullptr)
    , m_d_particleStrandIds(nullptr)
    , m_d_cellMarkers(nullptr)
    , m_d_cellScatterIndices(nullptr)
    , m_numGuideCurves(0)
    , m_numVortexFields(0)
    , m_numColliders(0)
    , m_lastSimTimeMs(0.0f)
{
    m_params = {};
    cudaEventCreate(&m_startEvent);
    cudaEventCreate(&m_stopEvent);
}

HairSimulationGPU::~HairSimulationGPU() {
    shutdown();
    cudaEventDestroy(m_startEvent);
    cudaEventDestroy(m_stopEvent);
}

bool HairSimulationGPU::initialize(const SimulationParams& params) {
    m_params = params;
    m_numParticles = params.numHairStrands * params.segmentsPerStrand;

    allocateMemory();

    uint32_t blockSize = 256;
    uint32_t gridSize = (m_numParticles + blockSize - 1) / blockSize;

    initParticlesKernel<<<gridSize, blockSize>>>(
        m_d_particles,
        m_params.numHairStrands,
        m_params.segmentsPerStrand,
        Vec3(0, 0, 0),
        1.0f,
        m_params.massPerParticle
    );

    cudaMemcpy(m_d_oldParticles, m_d_particles,
               m_numParticles * sizeof(StrandParticle),
               cudaMemcpyDeviceToDevice);

    computeRestLengthsKernel<<<gridSize, blockSize>>>(
        m_d_particles,
        m_d_restLengths,
        m_params.numHairStrands,
        m_params.segmentsPerStrand
    );

    computeRestTangentsKernel<<<gridSize, blockSize>>>(
        m_d_particles,
        m_d_restTangents,
        m_d_restFrames,
        m_params.numHairStrands,
        m_params.segmentsPerStrand
    );

    extractPositionsKernel<<<gridSize, blockSize>>>(
        m_d_particles,
        m_d_positionsOnly,
        m_numParticles
    );

    initStrandIdsKernel<<<gridSize, blockSize>>>(
        m_d_particleStrandIds,
        m_params.numHairStrands,
        m_params.segmentsPerStrand
    );

    cudaDeviceSynchronize();

    m_initialized = true;
    return true;
}

void HairSimulationGPU::shutdown() {
    if (!m_initialized) return;

    freeMemory();
    m_initialized = false;
}

void HairSimulationGPU::allocateMemory() {
    cudaMalloc(&m_d_particles, m_numParticles * sizeof(StrandParticle));
    cudaMalloc(&m_d_oldParticles, m_numParticles * sizeof(StrandParticle));
    cudaMalloc(&m_d_constraintPositions, m_numParticles * sizeof(Vec3));
    cudaMalloc(&m_d_positionsOnly, m_numParticles * sizeof(Vec3));
    cudaMalloc(&m_d_restLengths, m_params.numHairStrands * (m_params.segmentsPerStrand - 1) * sizeof(float));
    cudaMalloc(&m_d_restTangents, m_params.segmentsPerStrand * sizeof(Vec3));
    cudaMalloc(&m_d_restFrames, m_params.segmentsPerStrand * sizeof(MaterialFrame));
    cudaMalloc(&m_d_strandOffsets, m_params.numHairStrands * sizeof(uint32_t));
    cudaMalloc(&m_d_guideCurves, MAX_GUIDE_CURVES * sizeof(GuideCurve));
    cudaMalloc(&m_d_vortexFields, MAX_VORTEX_FIELDS * sizeof(VortexField));
    cudaMalloc(&m_d_colliders, MAX_COLLISION_CAPSULES * sizeof(CapsuleCollider));
    cudaMalloc(&m_d_interpData, MAX_HAIR_STRANDS * sizeof(GPUInterpolationData));
    cudaMalloc(&m_d_guidePositions, MAX_GUIDE_CURVES * DEFAULT_SEGMENTS_PER_STRAND * sizeof(Vec3));

    cudaMalloc(&m_d_particleStrandIds, m_numParticles * sizeof(uint32_t));

    cudaMalloc(&m_d_spatialHashBuckets, SPATIAL_HASH_NUM_BUCKETS * sizeof(uint32_t));
    cudaMalloc(&m_d_spatialHashCounts, SPATIAL_HASH_NUM_BUCKETS * sizeof(uint32_t));
    cudaMalloc(&m_d_spatialHashSorted, m_numParticles * sizeof(uint32_t));
    cudaMalloc(&m_d_spatialHashParticleCells, m_numParticles * sizeof(uint32_t));
    cudaMalloc(&m_d_cellMarkers, m_numParticles * sizeof(uint32_t));
    cudaMalloc(&m_d_cellScatterIndices, m_numParticles * sizeof(uint32_t));
}

void HairSimulationGPU::freeMemory() {
    if (m_d_particles) cudaFree(m_d_particles);
    if (m_d_oldParticles) cudaFree(m_d_oldParticles);
    if (m_d_constraintPositions) cudaFree(m_d_constraintPositions);
    if (m_d_positionsOnly) cudaFree(m_d_positionsOnly);
    if (m_d_restLengths) cudaFree(m_d_restLengths);
    if (m_d_restTangents) cudaFree(m_d_restTangents);
    if (m_d_restFrames) cudaFree(m_d_restFrames);
    if (m_d_strandOffsets) cudaFree(m_d_strandOffsets);
    if (m_d_guideCurves) cudaFree(m_d_guideCurves);
    if (m_d_vortexFields) cudaFree(m_d_vortexFields);
    if (m_d_colliders) cudaFree(m_d_colliders);
    if (m_d_interpData) cudaFree(m_d_interpData);
    if (m_d_guidePositions) cudaFree(m_d_guidePositions);
    if (m_d_particleStrandIds) cudaFree(m_d_particleStrandIds);
    if (m_d_spatialHashBuckets) cudaFree(m_d_spatialHashBuckets);
    if (m_d_spatialHashCounts) cudaFree(m_d_spatialHashCounts);
    if (m_d_spatialHashSorted) cudaFree(m_d_spatialHashSorted);
    if (m_d_spatialHashParticleCells) cudaFree(m_d_spatialHashParticleCells);
    if (m_d_cellMarkers) cudaFree(m_d_cellMarkers);
    if (m_d_cellScatterIndices) cudaFree(m_d_cellScatterIndices);

    m_d_particles = nullptr;
    m_d_oldParticles = nullptr;
    m_d_constraintPositions = nullptr;
    m_d_positionsOnly = nullptr;
    m_d_restLengths = nullptr;
    m_d_restTangents = nullptr;
    m_d_restFrames = nullptr;
    m_d_strandOffsets = nullptr;
    m_d_guideCurves = nullptr;
    m_d_vortexFields = nullptr;
    m_d_colliders = nullptr;
    m_d_interpData = nullptr;
    m_d_guidePositions = nullptr;
    m_d_particleStrandIds = nullptr;
    m_d_spatialHashBuckets = nullptr;
    m_d_spatialHashCounts = nullptr;
    m_d_spatialHashSorted = nullptr;
    m_d_spatialHashParticleCells = nullptr;
    m_d_cellMarkers = nullptr;
    m_d_cellScatterIndices = nullptr;
}

void HairSimulationGPU::update(float deltaTime) {
    if (!m_initialized) return;

    cudaEventRecord(m_startEvent);

    float dt = m_params.timeStep;
    float remainingTime = deltaTime;

    while (remainingTime > 0.0f) {
        float stepDt = fminf(dt, remainingTime);
        computePhysics(stepDt);
        remainingTime -= stepDt;
    }

    uint32_t blockSize = 256;
    uint32_t gridSize = (m_numParticles + blockSize - 1) / blockSize;
    extractPositionsKernel<<<gridSize, blockSize>>>(
        m_d_particles, m_d_positionsOnly, m_numParticles);

    cudaEventRecord(m_stopEvent);
    cudaEventSynchronize(m_stopEvent);

    float ms = 0;
    cudaEventElapsedTime(&ms, m_startEvent, m_stopEvent);
    m_lastSimTimeMs = ms;
}

void HairSimulationGPU::computePhysics(float dt) {
    uint32_t blockSize = 256;
    uint32_t gridSize = (m_numParticles + blockSize - 1) / blockSize;
    uint32_t strandGridSize = (m_params.numHairStrands + blockSize - 1) / blockSize;

    applyGravityKernel<<<gridSize, blockSize>>>(
        m_d_particles,
        m_numParticles,
        Vec3(0, -m_params.gravity, 0),
        dt
    );

    applyGlobalWindKernel<<<gridSize, blockSize>>>(
        m_d_particles,
        m_numParticles,
        m_params.globalWind,
        m_params.windStrength,
        dt,
        m_params.hairRadius
    );

    applyVortexWindKernel<<<gridSize, blockSize>>>(
        m_d_particles,
        m_numParticles,
        m_d_vortexFields,
        m_numVortexFields,
        dt,
        m_params.hairRadius
    );

    applyDampingKernel<<<gridSize, blockSize>>>(
        m_d_particles,
        m_numParticles,
        m_params.damping,
        dt
    );

    integratePositionsKernel<<<gridSize, blockSize>>>(
        m_d_particles,
        m_numParticles,
        dt
    );

    fixRootParticlesKernel<<<strandGridSize, blockSize>>>(
        m_d_particles,
        m_d_oldParticles,
        m_params.numHairStrands,
        m_params.segmentsPerStrand
    );

    for (uint32_t iter = 0; iter < m_params.numIterations; iter++) {
        distanceConstraintsKernel<<<strandGridSize, blockSize>>>(
            m_d_particles,
            m_params.numHairStrands,
            m_params.segmentsPerStrand,
            m_d_restLengths,
            0.5f
        );

        bendConstraintsKernel<<<strandGridSize, blockSize>>>(
            m_d_particles,
            m_params.numHairStrands,
            m_params.segmentsPerStrand,
            m_d_restTangents,
            m_params.bendStiffness
        );

        if (m_params.twistStiffness > 0.001f) {
            twistConstraintsKernel<<<strandGridSize, blockSize>>>(
                m_d_particles,
                m_params.numHairStrands,
                m_params.segmentsPerStrand,
                m_d_restFrames,
                m_params.twistStiffness
            );
        }
    }

    capsuleCollisionKernel<<<gridSize, blockSize>>>(
        m_d_particles,
        m_numParticles,
        m_d_colliders,
        m_numColliders,
        m_params.friction
    );

    if (m_params.enableSelfCollision) {
        buildSpatialHash();

        selfCollisionKernel<<<gridSize, blockSize>>>(
            m_d_particles,
            m_d_spatialHashSorted,
            m_d_spatialHashBuckets,
            m_d_spatialHashCounts,
            m_d_particleStrandIds,
            m_numParticles,
            m_params.selfCollisionRadius,
            0.8f,
            m_params.segmentsPerStrand
        );
    }

    cudaDeviceSynchronize();
}

void HairSimulationGPU::buildSpatialHash() {
    uint32_t blockSize = 256;
    uint32_t gridSize = (m_numParticles + blockSize - 1) / blockSize;
    uint32_t numBuckets = SPATIAL_HASH_NUM_BUCKETS;

    cudaMemset(m_d_spatialHashBuckets, 0xFFFFFFFF, numBuckets * sizeof(uint32_t));
    cudaMemset(m_d_spatialHashCounts, 0, numBuckets * sizeof(uint32_t));

    float cellSize = m_params.selfCollisionRadius * 3.0f;

    spatialHashAssignKernel<<<gridSize, blockSize>>>(
        m_d_particles,
        m_numParticles,
        cellSize,
        SPATIAL_HASH_GRID_SIZE,
        numBuckets,
        m_d_spatialHashParticleCells
    );

    thrust::device_ptr<uint32_t> d_cells(m_d_spatialHashParticleCells);
    thrust::device_ptr<uint32_t> d_indices(m_d_spatialHashSorted);
    thrust::sequence(d_indices, d_indices + m_numParticles);
    thrust::sort_by_key(d_cells, d_cells + m_numParticles, d_indices);

    uint32_t boundaryGrid = (m_numParticles + blockSize) / blockSize;
    bucketBoundaryMarkersKernel<<<boundaryGrid, blockSize>>>(
        m_d_spatialHashParticleCells,
        m_numParticles,
        numBuckets,
        m_d_spatialHashBuckets,
        m_d_spatialHashCounts
    );

    countBucketSizesKernel<<<gridSize, blockSize>>>(
        m_d_spatialHashParticleCells,
        m_numParticles,
        numBuckets,
        m_d_spatialHashBuckets,
        m_d_spatialHashCounts
    );
}

void HairSimulationGPU::setGuideCurve(uint32_t index, const std::vector<Vec3>& points) {
    if (index >= MAX_GUIDE_CURVES) return;

    GuideCurve guide;
    guide.strandId = index;
    guide.numSegments = (uint32_t)points.size() - 1;
    guide.rootPosition = points.empty() ? Vec3(0, 0, 0) : points[0];

    float totalLength = 0.0f;
    for (size_t i = 1; i < points.size(); i++) {
        totalLength += length(points[i] - points[i - 1]);
    }
    guide.length = totalLength;

    cudaMemcpy(m_d_guideCurves + index, &guide, sizeof(GuideCurve), cudaMemcpyHostToDevice);
}

void HairSimulationGPU::setVortexField(uint32_t index, const VortexField& field) {
    if (index >= MAX_VORTEX_FIELDS) return;

    if (index >= m_numVortexFields) {
        m_numVortexFields = index + 1;
    }

    cudaMemcpy(m_d_vortexFields + index, &field, sizeof(VortexField), cudaMemcpyHostToDevice);
}

void HairSimulationGPU::setCapsuleCollider(uint32_t index, const CapsuleCollider& collider) {
    if (index >= MAX_COLLISION_CAPSULES) return;

    if (index >= m_numColliders) {
        m_numColliders = index + 1;
    }

    cudaMemcpy(m_d_colliders + index, &collider, sizeof(CapsuleCollider), cudaMemcpyHostToDevice);
}

void HairSimulationGPU::setParams(const SimulationParams& params) {
    m_params = params;
}

void HairSimulationGPU::downloadPositions(std::vector<glm::vec3>& positions) {
    positions.resize(m_numParticles);

    std::vector<Vec3> tempPos(m_numParticles);
    cudaMemcpy(tempPos.data(), m_d_positionsOnly,
               m_numParticles * sizeof(Vec3),
               cudaMemcpyDeviceToHost);

    for (uint32_t i = 0; i < m_numParticles; i++) {
        positions[i] = tempPos[i].to_glm();
    }
}

void HairSimulationGPU::setInterpolationData(const std::vector<GPUInterpolationData>& interpData) {
    if (interpData.empty()) return;
    uint32_t count = (uint32_t)interpData.size();
    cudaMemcpy(m_d_interpData, interpData.data(),
               count * sizeof(GPUInterpolationData),
               cudaMemcpyHostToDevice);
}

void HairSimulationGPU::setGuideCurvePositions(const std::vector<Vec3>& guidePositions) {
    if (guidePositions.empty()) return;
    uint32_t count = (uint32_t)guidePositions.size();
    cudaMemcpy(m_d_guidePositions, guidePositions.data(),
               count * sizeof(Vec3),
               cudaMemcpyHostToDevice);
    m_numGuideCurvesStored = count / m_params.segmentsPerStrand;
}

void HairSimulationGPU::reset() {
    if (!m_initialized) return;

    uint32_t blockSize = 256;
    uint32_t gridSize = (m_numParticles + blockSize - 1) / blockSize;

    initParticlesKernel<<<gridSize, blockSize>>>(
        m_d_particles,
        m_params.numHairStrands,
        m_params.segmentsPerStrand,
        Vec3(0, 0, 0),
        1.0f,
        m_params.massPerParticle
    );

    cudaMemcpy(m_d_oldParticles, m_d_particles,
               m_numParticles * sizeof(StrandParticle),
               cudaMemcpyDeviceToDevice);

    extractPositionsKernel<<<gridSize, blockSize>>>(
        m_d_particles, m_d_positionsOnly, m_numParticles);

    cudaDeviceSynchronize();
}

}
