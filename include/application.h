#pragma once

#include "hair_types.h"
#include <memory>
#include <string>

struct GLFWwindow;

namespace hair {

class Camera;
class HairSimulationGPU;
class HairRenderer;
class HairInterpolator;
class UISystem;
class AlembicExporter;
class SharedMemoryBridge;

class Application {
public:
    Application();
    ~Application();

    bool initialize(int width, int height, const std::string& title);
    void run();
    void shutdown();

    void setInitialParams(const SimulationParams& params) { m_initialParams = params; }

    static Application* getInstance() { return s_instance; }
    GLFWwindow* getWindow() const { return m_window; }

    float getDeltaTime() const { return m_deltaTime; }
    float getElapsedTime() const { return m_elapsedTime; }

private:
    bool initGLFW(int width, int height, const std::string& title);
    bool initOpenGL();
    bool initSystems();
    void shutdownSystems();

    void mainLoop();
    void processInput();
    void update();
    void render();

    void handleMouseInput();
    void handleKeyboardInput();
    void handleFramebufferSize(int width, int height);
    void handleScroll(double xoffset, double yoffset);

    void updatePerformanceTimings();

    static void framebufferSizeCallback(GLFWwindow* window, int width, int height);
    static void scrollCallback(GLFWwindow* window, double xoffset, double yoffset);
    static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
    static void cursorPosCallback(GLFWwindow* window, double xpos, double ypos);
    static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);

    GLFWwindow* m_window;
    int m_windowWidth;
    int m_windowHeight;

    std::unique_ptr<Camera> m_camera;
    std::unique_ptr<HairSimulationGPU> m_simulation;
    std::unique_ptr<HairRenderer> m_renderer;
    std::unique_ptr<HairInterpolator> m_interpolator;
    std::unique_ptr<UISystem> m_ui;
    std::unique_ptr<AlembicExporter> m_exporter;
    std::unique_ptr<SharedMemoryBridge> m_sharedMem;

    SimulationParams m_initialParams;
    PerformanceTimings m_timings;

    float m_deltaTime;
    float m_elapsedTime;
    float m_lastFrameTime;

    bool m_isRunning;
    bool m_showDemoWindow;

    double m_lastMouseX;
    double m_lastMouseY;
    bool m_mouseLeftDown;
    bool m_mouseRightDown;
    bool m_mouseMiddleDown;

    static Application* s_instance;
};

}
