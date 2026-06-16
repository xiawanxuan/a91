#pragma once

#include "hair_types.h"
#include <string>
#include <vector>
#include <glm/glm.hpp>

namespace hair {

class AlembicExporter {
public:
    AlembicExporter();
    ~AlembicExporter();

    bool open(const std::string& filename,
              uint32_t numStrands,
              uint32_t segmentsPerStrand);
    void close();

    void addFrame(const std::vector<glm::vec3>& positions, float time);

    bool isOpen() const { return m_isOpen; }

    uint32_t getNumFrames() const { return m_numFrames; }
    uint32_t getNumStrands() const { return m_numStrands; }
    uint32_t getSegmentsPerStrand() const { return m_segmentsPerStrand; }

    void setFps(uint32_t fps) { m_fps = fps; }
    uint32_t getFps() const { return m_fps; }

    void setDescription(const std::string& desc) { m_description = desc; }

    float getLastFrameWriteTimeMs() const { return m_lastFrameWriteMs; }

private:
    void writeOgawaHeader();
    void writeHierarchy();
    void writeFrameData(const std::vector<glm::vec3>& positions, float time);

    bool m_isOpen;
    uint32_t m_numFrames;
    uint32_t m_numStrands;
    uint32_t m_segmentsPerStrand;
    uint32_t m_fps;
    std::string m_filename;
    std::string m_description;

    void* m_abcArchive;
    void* m_abcPoints;
    void* m_abcCurves;

    uint64_t m_dataOffset;
    uint64_t m_hierarchyOffset;
    uint32_t m_compressionLevel;

    float m_lastFrameWriteMs;
};

}
