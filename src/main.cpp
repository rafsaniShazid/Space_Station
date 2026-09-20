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
bool showTransformationExamples = false;
float armBaseAngle = 0.0f;
float armShoulderAngle = -25.0f;
float armElbowAngle = 35.0f;

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

void drawStationModule(float x, float y, float z, float rotation) {
    glPushMatrix();
    glTranslatef(x, y, z);             // Translation places a module around the hub.
    glRotatef(rotation, 0.0f, 1.0f, 0.0f);
    setMaterial(0.38f, 0.48f, 0.58f);
    drawCylinder(1.15f, 3.2f);

    glTranslatef(0.0f, 1.8f, 0.0f);
    setMaterial(0.12f, 0.28f, 0.42f);
    drawSphere(0.45f);
    glPopMatrix();
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
    glRotatef(rotation, 0.0f, 1.0f, 0.0f);

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

    setMaterial(0.65f, 0.70f, 0.76f);
    drawSphere(2.0f);

    setMaterial(0.30f, 0.38f, 0.48f);
    drawCube(4.8f, 0.65f, 0.65f);
    drawCube(0.65f, 4.8f, 0.65f);

    drawConnectingBeam(0.0f, 0.0f, 0.0f, 35.0f);
    drawConnectingBeam(0.0f, 0.0f, 0.0f, -35.0f);

    drawStationModule(0.0f, 0.0f, 4.4f, 0.0f);
    drawStationModule(0.0f, 0.0f, -4.4f, 180.0f);
    drawStationModule(4.4f, 0.0f, 0.0f, 90.0f);
    drawStationModule(-4.4f, 0.0f, 0.0f, -90.0f);

    drawDockingPort(0.0f, 0.0f, 6.2f, 0.0f);
    drawDockingPort(6.2f, 0.0f, 0.0f, 90.0f);

    drawSolarPanel(2.8f, 0.0f, 0.0f, 90.0f);
    drawSolarPanel(-2.8f, 0.0f, 0.0f, -90.0f);

    glPopMatrix();

    drawRoboticArm();

    if (showTransformationExamples) {
        drawTransformationExamples();
    }
}

void resetCamera() {
    cameraDistance = 22.0f;
}

void resetArm() {
    armBaseAngle = 0.0f;
    armShoulderAngle = -25.0f;
    armElbowAngle = 35.0f;
}

void applyCamera() {
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    gluLookAt(cameraDistance, cameraDistance * 0.62f, cameraDistance,
              0.0f, 0.0f, 0.0f,
              0.0f, 1.0f, 0.0f);
}

void display() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    applyCamera();
    drawSpaceStation();
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
    default:
        return;
    }

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
    glutCreateWindow("Interactive Space Station Simulator - Stage 4");

    initializeOpenGL();
    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutKeyboardFunc(keyboard);

    glutMainLoop();
    return 0;
}
