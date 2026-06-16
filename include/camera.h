#pragma once

#include "hair_types.h"
#include <glm/glm.hpp>
#include <string>

namespace hair {

class Camera {
public:
    Camera();

    void setPosition(const glm::vec3& pos) { m_position = pos; updateViewMatrix(); }
    const glm::vec3& getPosition() const { return m_position; }

    void setTarget(const glm::vec3& target) { m_target = target; updateViewMatrix(); }
    const glm::vec3& getTarget() const { return m_target; }

    void setUp(const glm::vec3& up) { m_up = up; updateViewMatrix(); }

    void setFOV(float fov) { m_fov = fov; updateProjMatrix(); }
    float getFOV() const { return m_fov; }

    void setAspectRatio(float aspect) { m_aspectRatio = aspect; updateProjMatrix(); }
    float getAspectRatio() const { return m_aspectRatio; }

    void setNearPlane(float near_) { m_near = near_; updateProjMatrix(); }
    void setFarPlane(float far_) { m_far = far_; updateProjMatrix(); }

    const glm::mat4& getViewMatrix() const { return m_viewMatrix; }
    const glm::mat4& getProjMatrix() const { return m_projMatrix; }
    glm::mat4 getViewProjMatrix() const { return m_projMatrix * m_viewMatrix; }

    void orbit(float deltaAzimuth, float deltaElevation, float deltaRadius);
    void pan(float deltaX, float deltaY);
    void zoom(float delta);

    void setOrbitalMode(bool enable) { m_orbitalMode = enable; }
    bool isOrbitalMode() const { return m_orbitalMode; }

    glm::vec3 getForward() const;
    glm::vec3 getRight() const;
    glm::vec3 getUp() const;

private:
    void updateViewMatrix();
    void updateProjMatrix();
    void updateOrbitalPosition();

    glm::vec3 m_position;
    glm::vec3 m_target;
    glm::vec3 m_up;

    float m_fov;
    float m_aspectRatio;
    float m_near;
    float m_far;

    glm::mat4 m_viewMatrix;
    glm::mat4 m_projMatrix;

    bool m_orbitalMode;
    float m_orbitRadius;
    float m_orbitAzimuth;
    float m_orbitElevation;
};

}
