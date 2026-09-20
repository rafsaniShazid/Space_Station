#include <GL/freeglut.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>

namespace {

constexpr int InitialWidth = 1000;
constexpr int InitialHeight = 700;
constexpr float MinimumCameraDistance = 10.0f;
constexpr float MaximumCameraDistance = 45.0f;

int windowWidth = InitialWidth;
int windowHeight = InitialHeight;
float cameraDistance = 22.0f;
float cameraYaw = 45.0f;
float cameraPitch = 25.0f;
bool cameraDragging = false;
int lastMouseX = 0;
int lastMouseY = 0;
bool showTransformationExamples = false;
float armBaseAngle = 0.0f;
float armShoulderAngle = -25.0f;
float armElbowAngle = 35.0f;
bool animationPaused = false;
bool dockingActive = false;
float solarPanelAngle = 0.0f;
float satelliteAngle = 0.0f;
float debrisOffset = 0.0f;
float dockingProgress = 0.0f;

void drawCube(float width, float height, float depth) {
    glPushMatrix();
    glScalef(width, height, depth);
    glutSolidCube(1.0f);
    glPopMatrix();
}

void drawSphere(float radius) {
    glutSolidSphere(radius, 32, 20);
}

void drawCylinder(float radius, float height) {
    GLUquadric* quadric = gluNewQuadric();
    if (quadric == nullptr) {
        return;
    }

    glPushMatrix();
    glRotatef(-90.0f, 1.0f, 0.0f, 0.0f);
    gluCylinder(quadric, radius, radius, height, 32, 4);
    glPopMatrix();

    gluDeleteQuadric(quadric);
}

void setMaterial(float red, float green, float blue, float shine = 24.0f) {
    const GLfloat color[] = {red, green, blue, 1.0f};
    const GLfloat specular[] = {0.7f, 0.7f, 0.7f, 1.0f};
    glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, color);
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, specular);
    glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, shine);
}

void drawStationModule(float x, float y, float z,
                       float rotation, float axisX, float axisY, float axisZ) {
    glPushMatrix();
    glTranslatef(x, y, z);             // Translation places a module around the hub.
    glRotatef(rotation, axisX, axisY, axisZ);
    setMaterial(0.38f, 0.48f, 0.58f);
    drawCylinder(1.15f, 3.2f);

    glPushMatrix();
    glTranslatef(0.0f, 0.55f, 0.0f);
    glRotatef(90.0f, 1.0f, 0.0f, 0.0f);
    setMaterial(0.55f, 0.58f, 0.64f);
    glutSolidTorus(0.10f, 1.16f, 12, 32);
    glPopMatrix();

    glPushMatrix();
    glTranslatef(0.0f, -0.55f, 0.0f);
    glRotatef(90.0f, 1.0f, 0.0f, 0.0f);
    setMaterial(0.55f, 0.58f, 0.64f);
    glutSolidTorus(0.10f, 1.16f, 12, 32);
    glPopMatrix();

    setMaterial(0.08f, 0.22f, 0.38f, 48.0f);
    for (int window = 0; window < 6; ++window) {
        const float angle = window * 60.0f * 3.14159265f / 180.0f;
        glPushMatrix();
        glTranslatef(std::cos(angle) * 1.13f,
                     static_cast<float>(window % 2) * 0.55f - 0.28f,
                     std::sin(angle) * 1.13f);
        glScalef(0.12f, 0.22f, 0.32f);
        glutSolidCube(1.0f);
        glPopMatrix();
    }

    glTranslatef(0.0f, 1.8f, 0.0f);
    setMaterial(0.12f, 0.28f, 0.42f);
    drawSphere(0.45f);
    glPopMatrix();
}

void drawCentralCore() {
    glPushMatrix();
    glRotatef(90.0f, 1.0f, 0.0f, 0.0f);
    setMaterial(0.48f, 0.54f, 0.62f);
    drawCylinder(1.55f, 4.2f);
    glPopMatrix();

    setMaterial(0.16f, 0.24f, 0.32f);
    for (int section = -1; section <= 1; ++section) {
        glPushMatrix();
        glTranslatef(0.0f, 0.0f, static_cast<float>(section) * 1.25f);
        glRotatef(90.0f, 1.0f, 0.0f, 0.0f);
        glutSolidTorus(0.11f, 1.58f, 12, 32);
        glPopMatrix();
    }
}

void drawStationRing(float y, float radius) {
    glPushMatrix();
    glTranslatef(0.0f, y, 0.0f);
    glRotatef(90.0f, 1.0f, 0.0f, 0.0f);
    setMaterial(0.48f, 0.52f, 0.58f);
    glutSolidTorus(0.16f, radius, 16, 48);
    glPopMatrix();
}

void drawRadialTrusses() {
    setMaterial(0.24f, 0.31f, 0.38f);
    for (int spoke = 0; spoke < 8; ++spoke) {
        glPushMatrix();
        glRotatef(spoke * 45.0f, 0.0f, 1.0f, 0.0f);
        glTranslatef(3.1f, 0.0f, 0.0f);
        glRotatef(90.0f, 0.0f, 0.0f, 1.0f);
        drawCube(0.18f, 5.8f, 0.18f);
        glPopMatrix();
    }
}

void drawConnectingBeam(float x, float y, float z, float rotation) {
    glPushMatrix();
    glTranslatef(x, y, z);
    glRotatef(rotation, 0.0f, 0.0f, 1.0f);
    setMaterial(0.22f, 0.30f, 0.38f);
    drawCube(8.0f, 0.28f, 0.28f);
    glPopMatrix();
}

void drawDockingPort(float x, float y, float z, float rotation) {
    glPushMatrix();
    glTranslatef(x, y, z);
    glRotatef(rotation, 0.0f, 1.0f, 0.0f);

    setMaterial(0.50f, 0.54f, 0.60f);
    glutSolidTorus(0.32f, 0.82f, 20, 32);
    glTranslatef(0.0f, 0.0f, 0.22f);
    setMaterial(0.08f, 0.12f, 0.18f);
    drawCylinder(0.62f, 0.16f);

    glPopMatrix();
}

void drawSolarPanel(float x, float y, float z, float rotation) {
    glPushMatrix();
    glTranslatef(x, y, z);
    glRotatef(rotation + solarPanelAngle, 0.0f, 1.0f, 0.0f);

    setMaterial(0.35f, 0.38f, 0.42f);
    drawCylinder(0.14f, 2.0f);

    glTranslatef(0.0f, 2.1f, 0.0f);
    glRotatef(90.0f, 1.0f, 0.0f, 0.0f);
    setMaterial(0.06f, 0.16f, 0.32f, 48.0f);
    drawCube(0.12f, 5.4f, 2.7f);

    setMaterial(0.65f, 0.68f, 0.72f);
    for (int row = -2; row <= 2; ++row) {
        glPushMatrix();
        glTranslatef(0.0f, static_cast<float>(row) * 0.48f, 0.0f);
        drawCube(0.15f, 0.035f, 2.72f);
        glPopMatrix();
    }
    for (int column = -1; column <= 1; ++column) {
        glPushMatrix();
        glTranslatef(0.0f, 0.0f, static_cast<float>(column) * 0.88f);
        drawCube(0.15f, 5.42f, 0.035f);
        glPopMatrix();
    }

    glPopMatrix();
}

void drawSatellite() {
    const float radians = satelliteAngle * 3.14159265f / 180.0f;
    const float x = std::cos(radians) * 11.0f;
    const float z = std::sin(radians) * 11.0f;

    glPushMatrix();
    glTranslatef(x, 4.0f, z);
    glRotatef(-satelliteAngle, 0.0f, 1.0f, 0.0f);
    setMaterial(0.72f, 0.74f, 0.78f);
    drawCube(1.0f, 0.8f, 1.2f);

    setMaterial(0.08f, 0.20f, 0.42f, 48.0f);
    glPushMatrix();
    glTranslatef(-1.2f, 0.0f, 0.0f);
    drawCube(1.3f, 0.06f, 0.75f);
    glPopMatrix();
    glPushMatrix();
    glTranslatef(1.2f, 0.0f, 0.0f);
    drawCube(1.3f, 0.06f, 0.75f);
    glPopMatrix();
    glPopMatrix();
}

void drawDebris() {
    const float debrisPositions[][3] = {
        {-12.0f, 5.0f, -7.0f},
        {10.0f, -4.0f, -9.0f},
        {-7.0f, -5.0f, 10.0f},
        {13.0f, 2.0f, 5.0f}
    };

    setMaterial(0.42f, 0.30f, 0.22f);
    for (int index = 0; index < 4; ++index) {
        glPushMatrix();
        glTranslatef(debrisPositions[index][0] + debrisOffset * (index + 1) * 0.6f,
                     debrisPositions[index][1], debrisPositions[index][2]);
        glRotatef(debrisOffset * 35.0f + index * 25.0f, 1.0f, 1.0f, 0.0f);
        drawCube(0.45f + index * 0.12f, 0.35f, 0.55f);
        glPopMatrix();
    }
}

void drawSpacecraft() {
    const float startZ = 18.0f;
    const float dockedZ = 7.0f;
    const float z = startZ + (dockedZ - startZ) * dockingProgress;
    const float doorOpening = dockingProgress * 0.75f;

    glPushMatrix();
    glTranslatef(0.0f, 0.0f, z);
    setMaterial(0.70f, 0.72f, 0.76f);
    drawCube(1.25f, 1.25f, 2.4f);

    glPushMatrix();
    glTranslatef(0.0f, 0.0f, 1.35f);
    setMaterial(0.18f, 0.28f, 0.42f);
    drawSphere(0.75f);
    glPopMatrix();

    setMaterial(0.78f, 0.48f, 0.16f);
    glPushMatrix();
    glTranslatef(-0.42f - doorOpening, 0.0f, 1.38f);
    drawCube(0.35f, 0.75f, 0.12f);
    glPopMatrix();
    glPushMatrix();
    glTranslatef(0.42f + doorOpening, 0.0f, 1.38f);
    drawCube(0.35f, 0.75f, 0.12f);
    glPopMatrix();
    glPopMatrix();
}

void updateAnimation(int) {
    if (!animationPaused) {
        solarPanelAngle = std::fmod(solarPanelAngle + 1.2f, 360.0f);
        satelliteAngle = std::fmod(satelliteAngle + 0.8f, 360.0f);
        debrisOffset = std::fmod(debrisOffset + 0.02f, 10.0f);

        if (dockingActive) {
            dockingProgress = std::min(1.0f, dockingProgress + 0.008f);
        }
    }

    glutPostRedisplay();
    glutTimerFunc(16, updateAnimation, 0);
}

void drawShearedCube() {
    const GLfloat shearMatrix[] = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.55f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f
    };
    glMultMatrixf(shearMatrix); // Shearing changes one axis relative to another.
    drawCube(2.0f, 2.0f, 2.0f);
}

void drawTransformationExamples() {
    glPushMatrix();
    glTranslatef(-8.0f, 3.0f, -3.5f);
    setMaterial(0.90f, 0.55f, 0.16f);
    glScalef(1.0f, 2.0f, 1.0f); // Scaling makes this block taller.
    drawCube(1.4f, 1.4f, 1.4f);
    glPopMatrix();

    glPushMatrix();
    glTranslatef(8.0f, 3.0f, -3.5f);
    glScalef(-1.0f, 1.0f, 1.0f); // Reflection mirrors the object across the YZ plane.
    setMaterial(0.25f, 0.78f, 0.68f);
    drawCube(2.0f, 1.3f, 1.3f);
    glPopMatrix();

    glPushMatrix();
    glTranslatef(-7.0f, -3.0f, 1.5f);
    setMaterial(0.78f, 0.28f, 0.30f);
    drawShearedCube();
    glPopMatrix();

    glPushMatrix();
    glTranslatef(7.0f, -3.0f, 1.5f);  // Translation
    glRotatef(35.0f, 0.0f, 1.0f, 0.0f); // Rotation
    glScalef(1.0f, 0.7f, 1.8f);        // Scaling: a composite transformation
    setMaterial(0.65f, 0.36f, 0.88f);
    drawCube(1.8f, 1.8f, 1.8f);
    glPopMatrix();
}

void drawArmSegment(float length) {
    glPushMatrix();
    glTranslatef(0.0f, length * 0.5f, 0.0f);
    setMaterial(0.70f, 0.72f, 0.76f);
    drawCube(0.42f, length, 0.42f);
    glPopMatrix();
}

void drawEndEffector() {
    glPushMatrix();
    glTranslatef(0.0f, 0.55f, 0.0f);
    setMaterial(0.85f, 0.48f, 0.18f);
    drawCube(0.55f, 0.35f, 0.55f);

    glPushMatrix();
    glTranslatef(0.0f, 0.35f, 0.28f);
    glRotatef(-18.0f, 1.0f, 0.0f, 0.0f);
    drawCube(0.14f, 0.65f, 0.14f);
    glPopMatrix();

    glPushMatrix();
    glTranslatef(0.0f, 0.35f, -0.28f);
    glRotatef(18.0f, 1.0f, 0.0f, 0.0f);
    drawCube(0.14f, 0.65f, 0.14f);
    glPopMatrix();
    glPopMatrix();
}

void drawRoboticArm() {
    glPushMatrix();
    glTranslatef(-1.7f, -1.8f, 2.0f);

    setMaterial(0.42f, 0.46f, 0.52f);
    drawCylinder(0.75f, 0.65f);

    glTranslatef(0.0f, 0.65f, 0.0f);
    glRotatef(armBaseAngle, 0.0f, 1.0f, 0.0f);
    setMaterial(0.86f, 0.50f, 0.18f);
    drawSphere(0.45f);

    glRotatef(armShoulderAngle, 0.0f, 0.0f, 1.0f);
    drawArmSegment(2.8f);

    glTranslatef(0.0f, 2.8f, 0.0f);
    setMaterial(0.86f, 0.50f, 0.18f);
    drawSphere(0.38f);

    glRotatef(armElbowAngle, 0.0f, 0.0f, 1.0f);
    drawArmSegment(2.3f);

    glTranslatef(0.0f, 2.3f, 0.0f);
    drawEndEffector();
    glPopMatrix();
}

void drawSpaceStation() {
    glPushMatrix();
    glRotatef(-12.0f, 0.0f, 1.0f, 0.0f);

    drawCentralCore();

    setMaterial(0.65f, 0.70f, 0.76f);
    drawSphere(1.45f);

    drawStationRing(0.0f, 2.45f);
    drawStationRing(0.0f, 3.05f);
    drawRadialTrusses();

    setMaterial(0.30f, 0.38f, 0.48f);
    drawCube(4.8f, 0.65f, 0.65f);
    drawCube(0.65f, 4.8f, 0.65f);

    drawConnectingBeam(0.0f, 0.0f, 0.0f, 35.0f);
    drawConnectingBeam(0.0f, 0.0f, 0.0f, -35.0f);

    drawStationModule(0.0f, 0.0f, 4.4f, 90.0f, 1.0f, 0.0f, 0.0f);
    drawStationModule(0.0f, 0.0f, -4.4f, -90.0f, 1.0f, 0.0f, 0.0f);
    drawStationModule(4.4f, 0.0f, 0.0f, 90.0f, 0.0f, 0.0f, 1.0f);
    drawStationModule(-4.4f, 0.0f, 0.0f, -90.0f, 0.0f, 0.0f, 1.0f);

    drawDockingPort(0.0f, 0.0f, 6.2f, 0.0f);
    drawDockingPort(6.2f, 0.0f, 0.0f, 90.0f);

    drawSolarPanel(2.8f, 0.0f, 0.0f, 90.0f);
    drawSolarPanel(-2.8f, 0.0f, 0.0f, -90.0f);

    glPopMatrix();

    drawRoboticArm();
    drawSatellite();
    drawDebris();
    if (dockingProgress > 0.0f || dockingActive) {
        drawSpacecraft();
    }

    if (showTransformationExamples) {
        drawTransformationExamples();
    }
}

void resetCamera() {
    cameraDistance = 22.0f;
    cameraYaw = 45.0f;
    cameraPitch = 25.0f;
}

void resetArm() {
    armBaseAngle = 0.0f;
    armShoulderAngle = -25.0f;
    armElbowAngle = 35.0f;
}

void applyCamera() {
    const float yawRadians = cameraYaw * 3.14159265f / 180.0f;
    const float pitchRadians = cameraPitch * 3.14159265f / 180.0f;
    const float horizontalDistance = cameraDistance * std::cos(pitchRadians);
    const float eyeX = horizontalDistance * std::cos(yawRadians);
    const float eyeY = cameraDistance * std::sin(pitchRadians);
    const float eyeZ = horizontalDistance * std::sin(yawRadians);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    gluLookAt(eyeX, eyeY, eyeZ,
              0.0f, 0.0f, 0.0f,
              0.0f, 1.0f, 0.0f);
}

void drawOverlayText(float x, float y, const char* text) {
    glRasterPos2f(x, y);
    for (const char* character = text; *character != '\0'; ++character) {
        glutBitmapCharacter(GLUT_BITMAP_8_BY_13, *character);
    }
}

void drawOverlay() {
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    gluOrtho2D(0.0, windowWidth, 0.0, windowHeight);
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    glDisable(GL_DEPTH_TEST);
    glDisable(GL_LIGHTING);
    glColor3f(0.82f, 0.90f, 0.98f);
    drawOverlayText(18.0f, windowHeight - 24.0f, "SPACE STATION SIMULATOR");
    glColor3f(0.66f, 0.76f, 0.86f);
    drawOverlayText(18.0f, windowHeight - 44.0f, "Drag mouse: orbit camera | Wheel: zoom | R: reset camera");
    drawOverlayText(18.0f, windowHeight - 62.0f, "Space: pause | N: dock spacecraft | B: reset docking");
    drawOverlayText(18.0f, windowHeight - 80.0f, "A/D W/S Q/E: robotic arm | T: transformations | Esc: exit");

    glColor3f(0.96f, 0.68f, 0.26f);
    drawOverlayText(windowWidth - 150.0f, windowHeight - 24.0f,
                    animationPaused ? "ANIMATION PAUSED" : "ANIMATION ACTIVE");

    glEnable(GL_LIGHTING);
    glEnable(GL_DEPTH_TEST);
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
}

void display() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    applyCamera();
    drawSpaceStation();
    drawOverlay();
    glutSwapBuffers();
}

void reshape(int width, int height) {
    windowWidth = std::max(width, 1);
    windowHeight = std::max(height, 1);

    glViewport(0, 0, windowWidth, windowHeight);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(45.0, static_cast<double>(windowWidth) / windowHeight, 0.1, 100.0);
    glMatrixMode(GL_MODELVIEW);
}

void keyboard(unsigned char key, int, int) {
    switch (key) {
    case 27:
        std::exit(EXIT_SUCCESS);
        break;
    case 'r':
    case 'R':
        resetCamera();
        break;
    case '+':
    case '=':
        cameraDistance = std::max(MinimumCameraDistance, cameraDistance - 1.0f);
        break;
    case '-':
    case '_':
        cameraDistance = std::min(MaximumCameraDistance, cameraDistance + 1.0f);
        break;
    case 't':
    case 'T':
        showTransformationExamples = !showTransformationExamples;
        break;
    case 'a':
        armBaseAngle = std::max(-90.0f, armBaseAngle - 5.0f);
        break;
    case 'd':
        armBaseAngle = std::min(90.0f, armBaseAngle + 5.0f);
        break;
    case 'w':
        armShoulderAngle = std::min(80.0f, armShoulderAngle + 5.0f);
        break;
    case 's':
        armShoulderAngle = std::max(-80.0f, armShoulderAngle - 5.0f);
        break;
    case 'q':
        armElbowAngle = std::min(120.0f, armElbowAngle + 5.0f);
        break;
    case 'e':
        armElbowAngle = std::max(-20.0f, armElbowAngle - 5.0f);
        break;
    case 'z':
    case 'Z':
        resetArm();
        break;
    case ' ':
        animationPaused = !animationPaused;
        break;
    case 'n':
    case 'N':
        dockingActive = true;
        break;
    case 'b':
    case 'B':
        dockingActive = false;
        dockingProgress = 0.0f;
        break;
    default:
        return;
    }

    glutPostRedisplay();
}

void mouseButton(int button, int state, int x, int y) {
    if (button == GLUT_LEFT_BUTTON) {
        cameraDragging = state == GLUT_DOWN;
        lastMouseX = x;
        lastMouseY = y;
    } else if (button == 3 && state == GLUT_DOWN) {
        cameraDistance = std::max(MinimumCameraDistance, cameraDistance - 1.0f);
    } else if (button == 4 && state == GLUT_DOWN) {
        cameraDistance = std::min(MaximumCameraDistance, cameraDistance + 1.0f);
    }

    glutPostRedisplay();
}

void mouseMotion(int x, int y) {
    if (!cameraDragging) {
        return;
    }

    cameraYaw += static_cast<float>(x - lastMouseX) * 0.5f;
    cameraPitch += static_cast<float>(y - lastMouseY) * 0.5f;
    cameraPitch = std::clamp(cameraPitch, -75.0f, 75.0f);
    lastMouseX = x;
    lastMouseY = y;
    glutPostRedisplay();
}

void initializeOpenGL() {
    const GLfloat lightPosition[] = {8.0f, 12.0f, 10.0f, 1.0f};
    const GLfloat ambientLight[] = {0.16f, 0.16f, 0.20f, 1.0f};

    glClearColor(0.005f, 0.008f, 0.02f, 1.0f);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
    glEnable(GL_COLOR_MATERIAL);
    glEnable(GL_NORMALIZE);
    glLightfv(GL_LIGHT0, GL_POSITION, lightPosition);
    glLightModelfv(GL_LIGHT_MODEL_AMBIENT, ambientLight);
    glShadeModel(GL_SMOOTH);
}

} // namespace

int main(int argc, char** argv) {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
    glutInitWindowSize(InitialWidth, InitialHeight);
    glutCreateWindow("Interactive Space Station Simulator - Stage 6");

    initializeOpenGL();
    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutKeyboardFunc(keyboard);
    glutMouseFunc(mouseButton);
    glutMotionFunc(mouseMotion);
    glutTimerFunc(16, updateAnimation, 0);

    glutMainLoop();
    return 0;
}
