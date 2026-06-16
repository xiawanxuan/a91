#include "application.h"
#include "hair_simulation.h"
#include "hair_renderer.h"
#include "hair_interpolator.h"
#include "camera.h"
#include "ui_system.h"
#include "alembic_exporter.h"
#include "shared_memory.h"
#include <GLFW/glfw3.h>
#include <glad/glad.h>
#include <iostream>
#include <cstdio>
#include <chrono>

namespace hair {

Application* Application::s_instance = nullptr;

Application::Application()
    : m_window(nullptr)
    , m_windowWidth(1600)
    , m_windowHeight(900)
    , m_deltaTime(0.0f)
    , m_elapsedTime(0.0f)
    , m_lastFrameTime(0.0f)
    , m_isRunning(false)
    , m_showDemoWindow(false)
    , m_lastMouseX(0.0)
    , m_lastMouseY(0.0)
    , m_mouseLeftDown(false)
    , m_mouseRightDown(false)
    , m_mouseMiddleDown(false)
{
    m_timings = {};
    s_instance = this;
}

Application::~Application() {
    shutdown();
}

bool Application::initialize(int width, int height, const std::string& title) {
    m_windowWidth = width;
    m_windowHeight = height;

    if (!initGLFW(width, height, title)) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return false;
    }

    if (!initOpenGL()) {
        std::cerr << "Failed to initialize OpenGL" << std::endl;
        return false;
    }

    if (!initSystems()) {
        std::cerr << "Failed to initialize systems" << std::endl;
        return false;
    }

    m_isRunning = true;
    return true;
}

bool Application::initGLFW(int width, int height, const std::string& title) {
    if (!glfwInit()) {
        return false;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);
    glfwWindowHint(GLFW_SAMPLES, 4);

    m_window = glfwCreateWindow(width, height, title.c_str(), nullptr, nullptr);
    if (!m_window) {
        glfwTerminate();
        return false;
    }

    glfwMakeContextCurrent(m_window);
    glfwSetFramebufferSizeCallback(m_window, framebufferSizeCallback);
    glfwSetScrollCallback(m_window, scrollCallback);
    glfwSetMouseButtonCallback(m_window, mouseButtonCallback);
    glfwSetCursorPosCallback(m_window, cursorPosCallback);
    glfwSetKeyCallback(m_window, keyCallback);

    glfwSwapInterval(0);

    return true;
}

bool Application::initOpenGL() {
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        return false;
    }

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glClearColor(0.08f, 0.1f, 0.12f, 1.0f);

    int width, height;
    glfwGetFramebufferSize(m_window, &width, &height);
    glViewport(0, 0, width, height);

    return true;
}

bool Application::initSystems() {
    m_camera = std::make_unique<Camera>();
    m_camera->setAspectRatio((float)m_windowWidth / m_windowHeight);
    m_camera->setPosition(glm::vec3(0.0f, 0.5f, 2.0f));
    m_camera->setTarget(glm::vec3(0.0f, 0.5f, 0.0f));

    m_initialParams = {};
    m_initialParams.gravity = 9.81f;
    m_initialParams.bendStiffness = 0.5f;
    m_initialParams.twistStiffness = 0.2f;
    m_initialParams.damping = 0.5f;
    m_initialParams.hairRadius = 0.005f;
    m_initialParams.massPerParticle = 0.001f;
    m_initialParams.timeStep = 0.016f;
    m_initialParams.numIterations = 3;
    m_initialParams.globalWind = Vec3(1.0f, 0.0f, 0.0f);
    m_initialParams.windStrength = 2.0f;
    m_initialParams.numGuideCurves = 200;
    m_initialParams.numHairStrands = 10000;
    m_initialParams.segmentsPerStrand = 20;
    m_initialParams.enableSelfCollision = false;
    m_initialParams.selfCollisionRadius = 0.01f;
    m_initialParams.friction = 0.3f;

    m_simulation = std::make_unique<HairSimulationGPU>();
    if (!m_simulation->initialize(m_initialParams)) {
        std::cerr << "Failed to initialize hair simulation" << std::endl;
        return false;
    }

    m_renderer = std::make_unique<HairRenderer>();
    if (!m_renderer->initialize(m_initialParams.numHairStrands,
                                 m_initialParams.segmentsPerStrand)) {
        std::cerr << "Failed to initialize hair renderer" << std::endl;
        return false;
    }

    m_interpolator = std::make_unique<HairInterpolator>();
    m_interpolator->initialize(m_initialParams.numGuideCurves,
                               m_initialParams.numHairStrands,
                               m_initialParams.segmentsPerStrand);

    m_interpolator->interpolateOnGPU(m_simulation.get());

    m_exporter = std::make_unique<AlembicExporter>();

    m_sharedMem = std::make_unique<SharedMemoryBridge>();

    m_ui = std::make_unique<UISystem>();
    if (!m_ui->initialize(m_window,
                          m_simulation.get(),
                          m_renderer.get(),
                          m_interpolator.get(),
                          m_camera.get(),
                          m_exporter.get(),
                          m_sharedMem.get())) {
        std::cerr << "Failed to initialize UI system" << std::endl;
        return false;
    }

    m_ui->setViewportSize(m_windowWidth, m_windowHeight);

    return true;
}

void Application::shutdownSystems() {
    if (m_ui) m_ui->shutdown();
    if (m_renderer) m_renderer->shutdown();
    if (m_simulation) m_simulation->shutdown();
    if (m_exporter) m_exporter->close();
    if (m_sharedMem) m_sharedMem->close();
}

void Application::run() {
    m_lastFrameTime = (float)glfwGetTime();

    while (m_isRunning && !glfwWindowShouldClose(m_window)) {
        mainLoop();
    }
}

void Application::mainLoop() {
    auto frameStart = std::chrono::high_resolution_clock::now();

    float currentTime = (float)glfwGetTime();
    m_deltaTime = currentTime - m_lastFrameTime;
    m_lastFrameTime = currentTime;
    m_elapsedTime += m_deltaTime;

    if (m_deltaTime > 0.1f) m_deltaTime = 0.016f;

    processInput();
    update();
    render();

    auto frameEnd = std::chrono::high_resolution_clock::now();
    auto frameDuration = std::chrono::duration_cast<std::chrono::microseconds>(frameEnd - frameStart);
    m_timings.totalFrameMs = (float)frameDuration.count() / 1000.0f;

    updatePerformanceTimings();

    glfwSwapBuffers(m_window);
    glfwPollEvents();
}

void Application::processInput() {
    handleMouseInput();
    handleKeyboardInput();
}

void Application::handleMouseInput() {
    double xpos, ypos;
    glfwGetCursorPos(m_window, &xpos, &ypos);

    float deltaX = (float)(xpos - m_lastMouseX);
    float deltaY = (float)(ypos - m_lastMouseY);

    m_lastMouseX = xpos;
    m_lastMouseY = ypos;

    ImGuiIO& io = ImGui::GetIO();
    if (io.WantCaptureMouse) return;

    if (m_mouseLeftDown) {
        m_camera->orbit(deltaX * 0.01f, deltaY * 0.01f, 0.0f);
    }

    if (m_mouseRightDown) {
        m_camera->pan(deltaX, deltaY);
    }

    if (m_mouseMiddleDown) {
        m_camera->zoom(-deltaY * 0.01f);
    }
}

void Application::handleKeyboardInput() {
    if (glfwGetKey(m_window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
        m_isRunning = false;
    }

    if (glfwGetKey(m_window, GLFW_KEY_R) == GLFW_PRESS) {
        static double lastResetTime = 0.0;
        double currentTime = glfwGetTime();
        if (currentTime - lastResetTime > 0.3) {
            if (m_simulation) {
                m_simulation->reset();
            }
            lastResetTime = currentTime;
        }
    }
}

void Application::update() {
    if (m_simulation) {
        m_simulation->update(m_deltaTime);

        auto uploadStart = std::chrono::high_resolution_clock::now();

        std::vector<glm::vec3> positions;
        m_simulation->downloadPositions(positions);

        auto uploadEnd = std::chrono::high_resolution_clock::now();
        auto uploadDuration = std::chrono::duration_cast<std::chrono::microseconds>(uploadEnd - uploadStart);
        m_timings.uploadTimeMs = (float)uploadDuration.count() / 1000.0f;

        if (m_renderer) {
            m_renderer->updateBuffers(positions);
        }

        if (m_exporter && m_exporter->isOpen()) {
            m_exporter->addFrame(positions, m_elapsedTime);
        }

        if (m_sharedMem && m_sharedMem->isOpen() && m_sharedMem->isServer()) {
            uint32_t dataSize = (uint32_t)(positions.size() * sizeof(glm::vec3));
            m_sharedMem->writeData(positions.data(), dataSize);

            SharedMemoryHeader header;
            m_sharedMem->readHeader(header);
            header.numHairStrands = m_simulation->getNumHairStrands();
            header.segmentsPerStrand = m_simulation->getSegmentsPerStrand();
            header.deltaTime = m_deltaTime;
            m_sharedMem->writeHeader(header);

            m_sharedMem->signalFrameReady();
        }
    }
}

void Application::render() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    if (m_ui) {
        m_ui->beginFrame();
    }

    if (m_renderer && m_camera) {
        m_renderer->render(
            m_camera->getViewMatrix(),
            m_camera->getProjMatrix(),
            m_camera->getPosition()
        );
    }

    if (m_ui) {
        m_ui->render();
        m_ui->endFrame();
    }
}

void Application::updatePerformanceTimings() {
    if (m_simulation) {
        m_timings.simTimeMs = m_simulation->getLastSimTimeMs();
    }

    if (m_renderer) {
        m_timings.renderTimeMs = m_renderer->getLastRenderTimeMs();
    }

    if (m_deltaTime > 0.0001f) {
        m_timings.fps = 1.0f / m_deltaTime;
    }

    m_timings.frameCount++;

    if (m_ui) {
        m_ui->setPerformanceTimings(m_timings);
    }
}

void Application::handleFramebufferSize(int width, int height) {
    m_windowWidth = width;
    m_windowHeight = height;

    glViewport(0, 0, width, height);

    if (m_camera) {
        m_camera->setAspectRatio((float)width / height);
    }

    if (m_ui) {
        m_ui->setViewportSize(width, height);
    }
}

void Application::handleScroll(double xoffset, double yoffset) {
    ImGuiIO& io = ImGui::GetIO();
    if (io.WantCaptureMouse) return;

    if (m_camera) {
        m_camera->zoom(-(float)yoffset * 0.1f);
    }
}

void Application::shutdown() {
    if (!m_isRunning) return;

    shutdownSystems();

    if (m_window) {
        glfwDestroyWindow(m_window);
        m_window = nullptr;
    }

    glfwTerminate();

    m_isRunning = false;
}

void Application::framebufferSizeCallback(GLFWwindow* window, int width, int height) {
    if (s_instance) {
        s_instance->handleFramebufferSize(width, height);
    }
}

void Application::scrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
    if (s_instance) {
        s_instance->handleScroll(xoffset, yoffset);
    }
}

void Application::mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
    if (!s_instance) return;

    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        s_instance->m_mouseLeftDown = (action == GLFW_PRESS);
    } else if (button == GLFW_MOUSE_BUTTON_RIGHT) {
        s_instance->m_mouseRightDown = (action == GLFW_PRESS);
    } else if (button == GLFW_MOUSE_BUTTON_MIDDLE) {
        s_instance->m_mouseMiddleDown = (action == GLFW_PRESS);
    }
}

void Application::cursorPosCallback(GLFWwindow* window, double xpos, double ypos) {
}

void Application::keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
}

}
