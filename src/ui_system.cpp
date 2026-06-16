#include "ui_system.h"
#include "hair_simulation.h"
#include "hair_renderer.h"
#include "hair_interpolator.h"
#include "camera.h"
#include "alembic_exporter.h"
#include "shared_memory.h"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <GLFW/glfw3.h>
#include <cstdio>
#include <sstream>
#include <iomanip>

namespace hair {

UISystem::UISystem()
    : m_window(nullptr)
    , m_simulation(nullptr)
    , m_renderer(nullptr)
    , m_interpolator(nullptr)
    , m_camera(nullptr)
    , m_exporter(nullptr)
    , m_sharedMem(nullptr)
    , m_viewportWidth(1280)
    , m_viewportHeight(720)
    , m_mouseOverUI(false)
    , m_isPlaying(true)
    , m_isRecording(false)
    , m_recordDuration(0.0f)
    , m_recordedFrames(0)
    , m_exportPath("hair_cache.abc")
    , m_exportFps(30)
    , m_showGuideCurves(false)
    , m_showCollisionShapes(false)
    , m_showWireframe(false)
    , m_sharedMemConnected(false)
    , m_sharedMemName("HairFurSimSharedMem")
    , m_sharedMemSize(64 * 1024 * 1024)
{
    m_timings = {};
}

UISystem::~UISystem() {
    shutdown();
}

bool UISystem::initialize(GLFWwindow* window,
                           HairSimulationGPU* simulation,
                           HairRenderer* renderer,
                           HairInterpolator* interpolator,
                           Camera* camera,
                           AlembicExporter* exporter,
                           SharedMemoryBridge* sharedMem) {
    m_window = window;
    m_simulation = simulation;
    m_renderer = renderer;
    m_interpolator = interpolator;
    m_camera = camera;
    m_exporter = exporter;
    m_sharedMem = sharedMem;

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    (void)io;

    ImGui::StyleColorsDark();

    ImGuiStyle& style = ImGui::GetStyle();
    style.WindowRounding = 4.0f;
    style.FrameRounding = 2.0f;
    style.GrabRounding = 2.0f;
    style.WindowPadding = ImVec2(8, 8);
    style.FramePadding = ImVec2(4, 2);

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 450 core");

    return true;
}

void UISystem::shutdown() {
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}

void UISystem::beginFrame() {
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
}

void UISystem::render() {
    renderParameterPanel();
    renderTimeline();
    renderViewportOverlay();
    renderPerformancePanel();
}

void UISystem::endFrame() {
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    m_mouseOverUI = ImGui::GetIO().WantCaptureMouse;
}

void UISystem::renderParameterPanel() {
    const float panelWidth = 320.0f;
    ImGui::SetNextWindowPos(ImVec2(m_viewportWidth - panelWidth - 10, 10), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(panelWidth, m_viewportHeight - 160), ImGuiCond_FirstUseEver);

    if (!ImGui::Begin("Parameters", nullptr,
                      ImGuiWindowFlags_NoCollapse)) {
        ImGui::End();
        return;
    }

    if (!m_simulation) {
        ImGui::End();
        return;
    }

    SimulationParams params = m_simulation->getParams();

    if (ImGui::CollapsingHeader("Physics", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::PushItemWidth(-1);
        ImGui::SliderFloat("Gravity", &params.gravity, 0.0f, 20.0f, "%.2f");
        ImGui::SliderFloat("Bend Stiffness", &params.bendStiffness, 0.0f, 2.0f, "%.3f");
        ImGui::SliderFloat("Twist Stiffness", &params.twistStiffness, 0.0f, 1.0f, "%.3f");
        ImGui::SliderFloat("Damping", &params.damping, 0.0f, 5.0f, "%.2f");
        ImGui::SliderFloat("Hair Radius", &params.hairRadius, 0.001f, 0.02f, "%.4f");
        ImGui::SliderFloat("Mass", &params.massPerParticle, 0.001f, 0.1f, "%.4f");
        ImGui::SliderFloat("Time Step", &params.timeStep, 0.001f, 0.033f, "%.4f");
        ImGui::SliderInt("Iterations", (int*)&params.numIterations, 1, 10);
        ImGui::SliderFloat("Friction", &params.friction, 0.0f, 1.0f, "%.2f");
        ImGui::PopItemWidth();
    }

    if (ImGui::CollapsingHeader("Wind", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::PushItemWidth(-1);
        ImGui::SliderFloat("Wind Strength", &params.windStrength, 0.0f, 20.0f, "%.1f");

        static float windDir[3] = {1.0f, 0.0f, 0.0f};
        ImGui::SliderFloat3("Wind Direction", windDir, -1.0f, 1.0f);
        params.globalWind = Vec3(windDir[0], windDir[1], windDir[2]);
        ImGui::PopItemWidth();

        renderWindPanel();
    }

    if (ImGui::CollapsingHeader("Collision")) {
        ImGui::PushItemWidth(-1);
        ImGui::Checkbox("Self Collision", &params.enableSelfCollision);
        if (params.enableSelfCollision) {
            ImGui::SliderFloat("Collision Radius", &params.selfCollisionRadius, 0.001f, 0.05f, "%.4f");
        }
        ImGui::Checkbox("Show Collision Shapes", &m_showCollisionShapes);
        ImGui::PopItemWidth();
        renderCollisionPanel();
    }

    if (ImGui::CollapsingHeader("Hair")) {
        ImGui::PushItemWidth(-1);
        ImGui::SliderInt("Guide Curves", (int*)&params.numGuideCurves, 10, 512);
        ImGui::SliderInt("Hair Strands", (int*)&params.numHairStrands, 1000, 500000);
        ImGui::SliderInt("Segments", (int*)&params.segmentsPerStrand, 5, 50);

        static float hairColor[3] = {0.5f, 0.35f, 0.2f};
        ImGui::ColorEdit3("Hair Color", hairColor);
        if (m_renderer) {
            m_renderer->setHairColor(glm::vec3(hairColor[0], hairColor[1], hairColor[2]));
        }

        static float thickness = 0.002f;
        ImGui::SliderFloat("Thickness", &thickness, 0.0005f, 0.01f, "%.4f");
        if (m_renderer) {
            m_renderer->setHairThickness(thickness);
        }

        static float tipScale = 0.3f;
        ImGui::SliderFloat("Tip Scale", &tipScale, 0.1f, 1.0f, "%.2f");
        if (m_renderer) {
            m_renderer->setTipScale(tipScale);
        }

        ImGui::Checkbox("Show Guide Curves", &m_showGuideCurves);
        if (m_renderer) {
            m_renderer->setShowGuideCurves(m_showGuideCurves);
        }

        ImGui::Checkbox("Wireframe", &m_showWireframe);
        if (m_renderer) {
            m_renderer->setWireframe(m_showWireframe);
        }
        ImGui::PopItemWidth();
    }

    if (ImGui::CollapsingHeader("Export")) {
        renderExportPanel();
    }

    if (ImGui::CollapsingHeader("Shared Memory")) {
        renderSharedMemoryPanel();
    }

    m_simulation->setParams(params);

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    if (ImGui::Button("Reset Simulation", ImVec2(-1, 0))) {
        m_simulation->reset();
        m_timings.frameCount = 0;
    }

    ImGui::End();
}

void UISystem::renderWindPanel() {
    ImGui::Text("Vortex Fields (max 5)");

    static int selectedVortex = 0;
    static bool vortexEnabled[5] = {true, false, false, false, false};
    static float vortexPos[5][3] = {
        {0.0f, 0.5f, 0.0f},
        {0.5f, 0.5f, 0.0f},
        {-0.5f, 0.5f, 0.0f},
        {0.0f, 1.0f, 0.5f},
        {0.0f, 0.5f, -0.5f}
    };
    static float vortexAxis[5][3] = {
        {0.0f, 1.0f, 0.0f},
        {0.0f, 1.0f, 0.0f},
        {0.0f, 1.0f, 0.0f},
        {0.0f, 1.0f, 0.0f},
        {0.0f, 1.0f, 0.0f}
    };
    static float vortexRadius[5] = {0.5f, 0.3f, 0.3f, 0.4f, 0.4f};
    static float vortexStrength[5] = {5.0f, 3.0f, 3.0f, 4.0f, 4.0f};
    static float vortexFalloff[5] = {1.5f, 1.5f, 1.5f, 1.5f, 1.5f};

    for (int i = 0; i < 5; i++) {
        ImGui::PushID(i);
        char label[32];
        snprintf(label, sizeof(label), "Vortex %d", i + 1);
        bool isSelected = (selectedVortex == i);
        if (ImGui::Selectable(vortexEnabled[i] ? "ON" : "OFF", isSelected, ImGuiSelectableFlags_None, ImVec2(40, 0))) {
            selectedVortex = i;
        }
        ImGui::SameLine();
        ImGui::Checkbox(label, &vortexEnabled[i]);
        ImGui::PopID();
    }

    int idx = selectedVortex;
    if (idx >= 0 && idx < 5) {
        ImGui::Spacing();
        ImGui::PushItemWidth(-1);
        ImGui::SliderFloat3("Position", vortexPos[idx], -2.0f, 2.0f);
        ImGui::SliderFloat3("Axis", vortexAxis[idx], -1.0f, 1.0f);
        ImGui::SliderFloat("Radius", &vortexRadius[idx], 0.1f, 3.0f, "%.2f");
        ImGui::SliderFloat("Strength", &vortexStrength[idx], 0.1f, 20.0f, "%.1f");
        ImGui::SliderFloat("Falloff", &vortexFalloff[idx], 0.5f, 4.0f, "%.1f");
        ImGui::PopItemWidth();

        VortexField field;
        field.position = Vec3(vortexPos[idx][0], vortexPos[idx][1], vortexPos[idx][2]);
        field.axis = Vec3(vortexAxis[idx][0], vortexAxis[idx][1], vortexAxis[idx][2]);
        field.radius = vortexRadius[idx];
        field.strength = vortexStrength[idx];
        field.falloff = vortexFalloff[idx];
        field.enabled = vortexEnabled[idx];

        if (m_simulation) {
            m_simulation->setVortexField(idx, field);
        }
    }
}

void UISystem::renderCollisionPanel() {
    ImGui::Text("Capsule Colliders");

    static int selectedCollider = 0;
    static bool colliderEnabled[8] = {true, false, false, false, false, false, false, false};
    static float colliderStart[8][3] = {
        {0.0f, 0.0f, 0.0f}, {0.0f, 0.3f, 0.0f}, {0.0f, 0.6f, 0.0f}, {0.0f, 0.9f, 0.0f},
        {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}
    };
    static float colliderEnd[8][3] = {
        {0.0f, 0.2f, 0.0f}, {0.0f, 0.5f, 0.0f}, {0.0f, 0.8f, 0.0f}, {0.0f, 1.1f, 0.0f},
        {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}
    };
    static float colliderRadius[8] = {
        0.1f, 0.08f, 0.06f, 0.04f, 0.05f, 0.05f, 0.05f, 0.05f
    };

    for (int i = 0; i < 8; i++) {
        ImGui::PushID(i);
        char label[32];
        snprintf(label, sizeof(label), "Capsule %d", i + 1);
        bool isSelected = (selectedCollider == i);
        if (ImGui::Selectable(colliderEnabled[i] ? "ON" : "OFF", isSelected, ImGuiSelectableFlags_None, ImVec2(40, 0))) {
            selectedCollider = i;
        }
        ImGui::SameLine();
        ImGui::Checkbox(label, &colliderEnabled[i]);
        ImGui::PopID();
    }

    int idx = selectedCollider;
    if (idx >= 0 && idx < 8) {
        ImGui::Spacing();
        ImGui::PushItemWidth(-1);
        ImGui::SliderFloat3("Start", colliderStart[idx], -1.0f, 1.0f);
        ImGui::SliderFloat3("End", colliderEnd[idx], -1.0f, 1.0f);
        ImGui::SliderFloat("Radius", &colliderRadius[idx], 0.01f, 0.5f, "%.3f");
        ImGui::PopItemWidth();

        CapsuleCollider collider;
        collider.start = Vec3(colliderStart[idx][0], colliderStart[idx][1], colliderStart[idx][2]);
        collider.end = Vec3(colliderEnd[idx][0], colliderEnd[idx][1], colliderEnd[idx][2]);
        collider.radius = colliderRadius[idx];
        collider.enabled = colliderEnabled[idx];

        if (m_simulation) {
            m_simulation->setCapsuleCollider(idx, collider);
        }
    }
}

void UISystem::renderExportPanel() {
    static char exportPathBuffer[256] = "hair_cache.abc";
    strncpy_s(exportPathBuffer, m_exportPath.c_str(), sizeof(exportPathBuffer));

    ImGui::PushItemWidth(-1);
    ImGui::InputText("Path", exportPathBuffer, sizeof(exportPathBuffer));
    m_exportPath = exportPathBuffer;

    ImGui::SliderInt("FPS", (int*)&m_exportFps, 24, 60);
    ImGui::PopItemWidth();

    ImGui::Spacing();

    if (!m_isRecording) {
        if (ImGui::Button("Start Recording", ImVec2(-1, 0))) {
            if (m_simulation && m_exporter) {
                m_exporter->open(m_exportPath,
                                m_simulation->getNumHairStrands(),
                                m_simulation->getSegmentsPerStrand());
                m_exporter->setFps(m_exportFps);
                m_isRecording = true;
                m_recordedFrames = 0;
                m_recordDuration = 0.0f;
            }
        }
    } else {
        if (ImGui::Button("Stop Recording", ImVec2(-1, 0))) {
            if (m_exporter) {
                m_exporter->close();
                m_isRecording = false;
            }
        }
    }

    if (m_isRecording) {
        ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "RECORDING");
        ImGui::SameLine();
        ImGui::Text("Frames: %d, Duration: %.2fs", m_recordedFrames, m_recordDuration);

        if (m_exporter) {
            ImGui::Text("Write: %.2f ms/frame", m_exporter->getLastFrameWriteTimeMs());
        }
    }
}

void UISystem::renderSharedMemoryPanel() {
    if (!m_sharedMem) {
        ImGui::Text("Shared memory not available");
        return;
    }

    static char memName[256] = "HairFurSimSharedMem";
    strncpy_s(memName, m_sharedMemName.c_str(), sizeof(memName));

    ImGui::PushItemWidth(-1);
    ImGui::InputText("Name", memName, sizeof(memName));
    m_sharedMemName = memName;

    ImGui::SliderInt("Size (MB)", (int*)&m_sharedMemSize, 1, 256);
    m_sharedMemSize *= 1024 * 1024;
    ImGui::PopItemWidth();

    ImGui::Spacing();

    if (!m_sharedMemConnected) {
        if (ImGui::Button("Create Server", ImVec2(-1, 0))) {
            if (m_sharedMem->create(m_sharedMemName, m_sharedMemSize)) {
                m_sharedMemConnected = true;
                std::cout << "Shared memory created: " << m_sharedMemName << std::endl;
            }
        }
        ImGui::Button("Connect Client", ImVec2(-1, 0));
    } else {
        if (ImGui::Button("Disconnect", ImVec2(-1, 0))) {
            m_sharedMem->close();
            m_sharedMemConnected = false;
        }

        ImGui::Text("Status: Connected");
        ImGui::Text("Name: %s", m_sharedMemName.c_str());
        ImGui::Text("Size: %u bytes", m_sharedMem->getSize());
        ImGui::Text("Role: %s", m_sharedMem->isServer() ? "Server" : "Client");
    }
}

void UISystem::renderTimeline() {
    const float timelineHeight = 80.0f;
    ImGui::SetNextWindowPos(ImVec2(10, m_viewportHeight - timelineHeight - 10), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(m_viewportWidth - 340, timelineHeight), ImGuiCond_FirstUseEver);

    if (!ImGui::Begin("Timeline", nullptr,
                      ImGuiWindowFlags_NoCollapse |
                      ImGuiWindowFlags_NoTitleBar)) {
        ImGui::End();
        return;
    }

    if (ImGui::Button(m_isPlaying ? "Pause" : "Play", ImVec2(60, 0))) {
        m_isPlaying = !m_isPlaying;
    }
    ImGui::SameLine();

    if (ImGui::Button("Step", ImVec2(50, 0))) {
    }
    ImGui::SameLine();

    if (ImGui::Button("Reset", ImVec2(50, 0))) {
        if (m_simulation) {
            m_simulation->reset();
        }
    }
    ImGui::SameLine();

    if (ImGui::Button(m_isRecording ? "Stop Rec" : "Record", ImVec2(70, 0))) {
        if (m_isRecording) {
            if (m_exporter) m_exporter->close();
            m_isRecording = false;
        } else {
            if (m_simulation && m_exporter) {
                m_exporter->open(m_exportPath,
                                m_simulation->getNumHairStrands(),
                                m_simulation->getSegmentsPerStrand());
                m_exporter->setFps(m_exportFps);
                m_isRecording = true;
                m_recordedFrames = 0;
                m_recordDuration = 0.0f;
            }
        }
    }

    ImGui::SameLine();
    ImGui::Text("FPS: %.1f", m_timings.fps);

    if (m_isRecording) {
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "REC %d", m_recordedFrames);
    }

    ImGui::Spacing();

    float timelineWidth = ImGui::GetContentRegionAvail().x;
    float cursorX = (float)(m_timings.frameCount % 300) / 300.0f * timelineWidth;

    ImVec2 cursorPos = ImGui::GetCursorScreenPos();
    ImDrawList* drawList = ImGui::GetWindowDrawList();

    drawList->AddRectFilled(
        ImVec2(cursorPos.x, cursorPos.y),
        ImVec2(cursorPos.x + timelineWidth, cursorPos.y + 20),
        IM_COL32(40, 40, 50, 255)
    );

    drawList->AddLine(
        ImVec2(cursorPos.x + cursorX, cursorPos.y),
        ImVec2(cursorPos.x + cursorX, cursorPos.y + 20),
        IM_COL32(255, 100, 100, 255),
        2.0f
    );

    ImGui::Dummy(ImVec2(0, 22));

    ImGui::End();
}

void UISystem::renderViewportOverlay() {
    ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_Always);
    ImGui::SetNextWindowBgAlpha(0.7f);

    if (!ImGui::Begin("Viewport Info", nullptr,
                      ImGuiWindowFlags_NoDecoration |
                      ImGuiWindowFlags_AlwaysAutoResize |
                      ImGuiWindowFlags_NoMove)) {
        ImGui::End();
        return;
    }

    if (m_simulation) {
        ImGui::Text("Strands: %d  Segs: %d  Particles: %d",
                    m_simulation->getNumHairStrands(),
                    m_simulation->getSegmentsPerStrand(),
                    m_simulation->getNumParticles());
    }

    if (m_camera) {
        glm::vec3 camPos = m_camera->getPosition();
        ImGui::Text("Camera: %.2f, %.2f, %.2f", camPos.x, camPos.y, camPos.z);
    }

    ImGui::End();
}

void UISystem::renderPerformancePanel() {
    ImGui::SetNextWindowPos(ImVec2(m_viewportWidth - 330, m_viewportHeight - 250), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(320, 140), ImGuiCond_FirstUseEver);

    if (!ImGui::Begin("Performance", nullptr,
                      ImGuiWindowFlags_NoCollapse |
                      ImGuiWindowFlags_AlwaysAutoResize)) {

        ImGui::End();
        return;
    }

    ImGui::Text("FPS: %.1f (%.2f ms)", m_timings.fps, m_timings.totalFrameMs);
    ImGui::Text("Sim: %.2f ms", m_timings.simTimeMs);
    ImGui::Text("Render: %.2f ms", m_timings.renderTimeMs);

    if (m_timings.uploadTimeMs > 0.001f) {
        ImGui::Text("Upload: %.2f ms", m_timings.uploadTimeMs);
    }

    ImGui::Spacing();

    static float fpsHistory[120] = {0};
    static int fpsIndex = 0;
    fpsHistory[fpsIndex] = m_timings.fps;
    fpsIndex = (fpsIndex + 1) % 120;

    ImGui::PlotLines("FPS", fpsHistory, 120, fpsIndex,
                     nullptr, 0.0f, 120.0f, ImVec2(280, 40));

    ImGui::End();
}

}
