#pragma once

#include "hair_types.h"
#include <functional>
#include <string>
#include <vector>
#include <glm/glm.hpp>

struct GLFWwindow;

namespace hair {

class Camera;
class HairSimulationGPU;
class HairRenderer;
class HairInterpolator;
class AlembicExporter;
class SharedMemoryBridge;

class UISystem {
public:
    UISystem();
    ~UISystem();

    bool initialize(GLFWwindow* window,
                    HairSimulationGPU* simulation,
                    HairRenderer* renderer,
                    HairInterpolator* interpolator,
                    Camera* camera,
                    AlembicExporter* exporter,
                    SharedMemoryBridge* sharedMem);

    void shutdown();

    void beginFrame();
    void render();
    void endFrame();

    void setViewportSize(int width, int height) {
        m_viewportWidth = width;
        m_viewportHeight = height;
    }

    void setPerformanceTimings(const PerformanceTimings& timings) {
        m_timings = timings;
    }

    bool isMouseOverUI() const { return m_mouseOverUI; }

private:
    void renderParameterPanel();
    void renderTimeline();
    void renderToolbar();
    void renderViewportOverlay();
    void renderWindPanel();
    void renderCollisionPanel();
    void renderExportPanel();
    void renderPerformancePanel();
    void renderSharedMemoryPanel();

    GLFWwindow* m_window;
    HairSimulationGPU* m_simulation;
    HairRenderer* m_renderer;
    HairInterpolator* m_interpolator;
    Camera* m_camera;
    AlembicExporter* m_exporter;
    SharedMemoryBridge* m_sharedMem;

    int m_viewportWidth;
    int m_viewportHeight;
    bool m_mouseOverUI;

    PerformanceTimings m_timings;

    bool m_isPlaying;
    bool m_isRecording;
    float m_recordDuration;
    uint32_t m_recordedFrames;

    std::string m_exportPath;
    uint32_t m_exportFps;

    bool m_showGuideCurves;
    bool m_showCollisionShapes;
    bool m_showWireframe;

    bool m_sharedMemConnected;
    std::string m_sharedMemName;
    uint32_t m_sharedMemSize;
};

}
