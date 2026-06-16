#include "alembic_exporter.h"
#include <iostream>
#include <fstream>
#include <cstring>
#include <cmath>
#include <chrono>
#include <algorithm>

namespace hair {

AlembicExporter::AlembicExporter()
    : m_isOpen(false)
    , m_numFrames(0)
    , m_numStrands(0)
    , m_segmentsPerStrand(0)
    , m_fps(30)
    , m_abcArchive(nullptr)
    , m_abcPoints(nullptr)
    , m_abcCurves(nullptr)
    , m_dataOffset(0)
    , m_hierarchyOffset(0)
    , m_compressionLevel(0)
    , m_lastFrameWriteMs(0.0f)
{
}

AlembicExporter::~AlembicExporter() {
    close();
}

bool AlembicExporter::open(const std::string& filename,
                            uint32_t numStrands,
                            uint32_t segmentsPerStrand) {
    if (m_isOpen) {
        close();
    }

    m_filename = filename;
    m_numStrands = numStrands;
    m_segmentsPerStrand = segmentsPerStrand;
    m_numFrames = 0;
    m_dataOffset = 0;
    m_hierarchyOffset = 0;

    std::ofstream file(filename, std::ios::binary | std::ios::trunc);
    if (!file.is_open()) {
        std::cerr << "Failed to create Alembic file: " << filename << std::endl;
        return false;
    }

    writeOgawaHeader();
    writeHierarchy();

    file.close();

    m_isOpen = true;

    std::cout << "Alembic file opened: " << filename << std::endl;
    std::cout << "  Strands: " << numStrands << std::endl;
    std::cout << "  Segments/Strand: " << segmentsPerStrand << std::endl;
    std::cout << "  FPS: " << m_fps << std::endl;

    return true;
}

void AlembicExporter::writeOgawaHeader() {
    std::ofstream file(m_filename, std::ios::binary | std::ios::app);

    AlembicHeader header;
    memset(&header, 0, sizeof(header));
    memcpy(header.magic, "HAIRABC", 7);
    header.version = 2;
    header.numStrands = m_numStrands;
    header.segmentsPerStrand = m_segmentsPerStrand;
    header.fps = m_fps;
    header.totalFrames = 0;
    header.startTime = 0.0f;
    header.endTime = 0.0f;
    header.compressionType = 0;

    file.write((const char*)&header, sizeof(AlembicHeader));

    uint64_t topologySize = m_numStrands * sizeof(uint32_t);
    file.write((const char*)&topologySize, sizeof(uint64_t));

    std::vector<uint32_t> curveCounts(m_numStrands, m_segmentsPerStrand);
    file.write((const char*)curveCounts.data(), topologySize);

    m_dataOffset = sizeof(AlembicHeader) + sizeof(uint64_t) + topologySize;

    file.close();
}

void AlembicExporter::writeHierarchy() {
    std::ofstream file(m_filename, std::ios::binary | std::ios::app);

    uint32_t numGroups = 1;
    uint32_t numCurvesInGroup = m_numStrands;
    file.write((const char*)&numGroups, sizeof(uint32_t));
    file.write((const char*)&numCurvesInGroup, sizeof(uint32_t));

    m_hierarchyOffset = m_dataOffset + sizeof(uint32_t) * 2;
    m_dataOffset = m_hierarchyOffset;

    file.close();
}

void AlembicExporter::addFrame(const std::vector<glm::vec3>& positions, float time) {
    if (!m_isOpen) return;

    auto startTime = std::chrono::high_resolution_clock::now();

    uint32_t numParticles = m_numStrands * m_segmentsPerStrand;
    if (positions.size() < numParticles) return;

    writeFrameData(positions, time);

    m_numFrames++;

    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
    m_lastFrameWriteMs = (float)duration.count() / 1000.0f;
}

void AlembicExporter::writeFrameData(const std::vector<glm::vec3>& positions, float time) {
    std::ofstream file(m_filename, std::ios::binary | std::ios::app);
    if (!file.is_open()) return;

    uint8_t frameMarker = 0xFE;
    file.write((const char*)&frameMarker, sizeof(uint8_t));

    file.write((const char*)&time, sizeof(float));

    uint32_t frameIndex = m_numFrames;
    file.write((const char*)&frameIndex, sizeof(uint32_t));

    uint32_t numPoints = (uint32_t)positions.size();
    file.write((const char*)&numPoints, sizeof(uint32_t));

    std::vector<float> posX(numPoints);
    std::vector<float> posY(numPoints);
    std::vector<float> posZ(numPoints);

    for (uint32_t i = 0; i < numPoints; i++) {
        posX[i] = positions[i].x;
        posY[i] = positions[i].y;
        posZ[i] = positions[i].z;
    }

    file.write((const char*)posX.data(), numPoints * sizeof(float));
    file.write((const char*)posY.data(), numPoints * sizeof(float));
    file.write((const char*)posZ.data(), numPoints * sizeof(float));

    std::vector<float> widths(m_numStrands, 1.0f);
    file.write((const char*)widths.data(), m_numStrands * sizeof(float));

    file.close();
}

void AlembicExporter::close() {
    if (!m_isOpen) return;

    std::fstream file(m_filename, std::ios::binary | std::ios::in | std::ios::out);
    if (file.is_open()) {
        file.seekp(offsetof(AlembicHeader, totalFrames));
        file.write((const char*)&m_numFrames, sizeof(uint32_t));

        float endTime = m_numFrames > 0 ? (float)m_numFrames / (float)m_fps : 0.0f;
        file.write((const char*)&endTime, sizeof(float));
        file.close();
    }

    m_isOpen = false;

    std::cout << "Alembic file closed. Total frames: " << m_numFrames << std::endl;
    std::cout << "  Duration: " << (m_numFrames / (float)m_fps) << "s" << std::endl;
    std::cout << "  File: " << m_filename << std::endl;
}

}
