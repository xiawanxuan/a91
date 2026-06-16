#include "camera.h"
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <cmath>

namespace hair {

Camera::Camera()
    : m_position(0.0f, 1.0f, 3.0f)
    , m_target(0.0f, 0.5f, 0.0f)
    , m_up(0.0f, 1.0f, 0.0f)
    , m_fov(60.0f)
    , m_aspectRatio(16.0f / 9.0f)
    , m_near(0.01f)
    , m_far(100.0f)
    , m_orbitalMode(true)
    , m_orbitRadius(3.0f)
    , m_orbitAzimuth(0.0f)
    , m_orbitElevation(0.5f)
{
    updateViewMatrix();
    updateProjMatrix();
    updateOrbitalPosition();
}

void Camera::updateViewMatrix() {
    m_viewMatrix = glm::lookAt(m_position, m_target, m_up);
}

void Camera::updateProjMatrix() {
    m_projMatrix = glm::perspective(
        glm::radians(m_fov),
        m_aspectRatio,
        m_near,
        m_far
    );
}

void Camera::updateOrbitalPosition() {
    float x = m_target.x + m_orbitRadius * cosf(m_orbitElevation) * sinf(m_orbitAzimuth);
    float y = m_target.y + m_orbitRadius * sinf(m_orbitElevation);
    float z = m_target.z + m_orbitRadius * cosf(m_orbitElevation) * cosf(m_orbitAzimuth);

    m_position = glm::vec3(x, y, z);
    updateViewMatrix();
}

void Camera::orbit(float deltaAzimuth, float deltaElevation, float deltaRadius) {
    m_orbitAzimuth += deltaAzimuth;
    m_orbitElevation += deltaElevation;

    float maxElevation = glm::pi<float>() * 0.49f;
    m_orbitElevation = fmaxf(-maxElevation, fminf(maxElevation, m_orbitElevation));

    m_orbitRadius += deltaRadius;
    m_orbitRadius = fmaxf(0.1f, fminf(100.0f, m_orbitRadius));

    updateOrbitalPosition();
}

void Camera::pan(float deltaX, float deltaY) {
    glm::vec3 right = getRight();
    glm::vec3 up = getUp();

    float panSpeed = m_orbitRadius * 0.002f;

    m_target += right * (-deltaX * panSpeed);
    m_target += up * (deltaY * panSpeed);

    updateOrbitalPosition();
}

void Camera::zoom(float delta) {
    m_orbitRadius += delta * m_orbitRadius * 0.1f;
    m_orbitRadius = fmaxf(0.1f, fminf(100.0f, m_orbitRadius));
    updateOrbitalPosition();
}

glm::vec3 Camera::getForward() const {
    return glm::normalize(m_target - m_position);
}

glm::vec3 Camera::getRight() const {
    return glm::normalize(glm::cross(getForward(), m_up));
}

glm::vec3 Camera::getUp() const {
    return m_up;
}

}
