/*
 * ============================================================================
 *  INTERACTIVE SPACE STATION SIMULATOR USING OPENGL
 *  Computer Graphics and Image Processing Lab
 *  Name: Md. Rafsani Shazid                                  Roll: 2107022
 * ============================================================================
 *
 *  BUILD
 *    Linux   : g++ -std=c++11 space_station_simulator.cpp -o station -lglut -lGLU -lGL -lm
 *    MinGW   : g++ -std=c++11 space_station_simulator.cpp -o station.exe -lfreeglut -lglu32 -lopengl32
 *    Code::Blocks / Dev-C++ : link freeglut, opengl32, glu32 (in that order).
 *
 *  WHERE EACH PROPOSAL OBJECTIVE LIVES IN THE CODE
 *    Primitives + custom meshes ... section 2/3 (cube, sphere, cylinder, torus, and
 *                                    hand-built meshes: lathe dish/nozzle, asteroid, probe)
 *    Translate/rotate/scale/
 *    reflect/shear/composite ...... Mat4 class + "TRANSFORMATION LAB" (press T)
 *    Hierarchical transforms ...... 6-DOF robotic arm, rotating solar wings, habitat ring,
 *                                    tracking antenna dish, docking adapter doors
 *    Real-time animation .......... docking spacecraft (state machine), solar wings,
 *                                    orbiting satellite, randomly drifting debris
 *    User interaction ............. keyboard + mouse (see the on-screen help, F1)
 *    Camera/view transformations .. orbit / pan / zoom camera, view presets, chase camera
 *
 *  CONTROLS (case-insensitive)
 *    Mouse   : left-drag orbit | right-drag zoom | middle-drag pan | wheel zoom
 *    Camera  : 1 iso  2 docking axis  3 top  4 aft  5 chase spacecraft  6 robotic arm
 *              arrows orbit | + - zoom | R reset
 *    Sim     : Space pause | N dock / undock | B reset docking | P solar wings: free spin / track sun
 *    Arm     : A/D base   W/S shoulder   Q/E elbow   F/G wrist   C/V gripper   Z reset
 *    View    : T transformation lab | L wireframe | O guides | F1 help | F11 fullscreen | F12 screenshot | Esc quit
 *    Lab (T) : 1-6 or [ ] choose demo | arrows move probe in X/Z | PgUp/PgDn move Y
 *              Shift+arrows/PgUp/PgDn rotate | , . scale | M mirror (reflection) | H shear | Z reset demo
 *
 *  OPTIONAL COMMAND-LINE SWITCHES (for demos / automated screenshots)
 *    --preset N   start in camera preset N (1-6)      --dock       start the docking sequence
 *    --lab N      open transformation-lab demo N      --skip SEC   fast-forward the simulation
 *    --nohelp     hide the help panel                 --shot FRAMES FILE.bmp   save a screenshot and quit
 * ============================================================================
 */

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif

#include <GL/freeglut.h>

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <utility>
#include <vector>

// Tokens newer than OpenGL 1.1 (the Windows gl.h only ships 1.1) - the drivers still accept them.
#ifndef GL_MULTISAMPLE
#define GL_MULTISAMPLE 0x809D
#endif
#ifndef GL_CLAMP_TO_EDGE
#define GL_CLAMP_TO_EDGE 0x812F
#endif
#ifndef GL_LIGHT_MODEL_COLOR_CONTROL
#define GL_LIGHT_MODEL_COLOR_CONTROL 0x81F8
#endif
#ifndef GL_SEPARATE_SPECULAR_COLOR
#define GL_SEPARATE_SPECULAR_COLOR 0x81FA
#endif

namespace {

// ============================================================================
// 1. MATH: scalars, vectors, and an explicit 4x4 matrix class
// ============================================================================

const float PI = 3.14159265358979323846f;

inline float radians(float d) { return d * PI / 180.0f; }
inline float degrees(float r) { return r * 180.0f / PI; }
inline float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }
inline float lerpf(float a, float b, float t) { return a + (b - a) * t; }
inline float smooth01(float t) { t = clampf(t, 0.0f, 1.0f); return t * t * (3.0f - 2.0f * t); }
inline float moveToward(float cur, float target, float maxDelta) {
    if (cur < target) return (cur + maxDelta > target) ? target : cur + maxDelta;
    return (cur - maxDelta < target) ? target : cur - maxDelta;
}
inline float wrap180(float a) {
    while (a > 180.0f) a -= 360.0f;
    while (a < -180.0f) a += 360.0f;
    return a;
}

struct Vec3 {
    float x, y, z;
    Vec3() : x(0), y(0), z(0) {}
    Vec3(float X, float Y, float Z) : x(X), y(Y), z(Z) {}
};
inline Vec3 operator+(Vec3 a, Vec3 b) { return Vec3(a.x + b.x, a.y + b.y, a.z + b.z); }
inline Vec3 operator-(Vec3 a, Vec3 b) { return Vec3(a.x - b.x, a.y - b.y, a.z - b.z); }
inline Vec3 operator*(Vec3 a, float s) { return Vec3(a.x * s, a.y * s, a.z * s); }
inline float dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline Vec3 cross(Vec3 a, Vec3 b) {
    return Vec3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
inline float length(Vec3 a) { return std::sqrt(dot(a, a)); }
inline Vec3 normalize(Vec3 a) {
    const float l = length(a);
    return l > 1e-6f ? a * (1.0f / l) : Vec3(0.0f, 1.0f, 0.0f);
}

// Column-major 4x4 matrix (the same memory layout OpenGL uses).
// Element (row r, column c) is stored at m[c * 4 + r].
// Points are transformed as column vectors:  p' = M * p,  so  M = A * B  means "apply B first, then A".
struct Mat4 {
    float m[16];

    Mat4() {
        for (int i = 0; i < 16; ++i) m[i] = 0.0f;
        m[0] = m[5] = m[10] = m[15] = 1.0f;
    }
    float& operator()(int r, int c) { return m[c * 4 + r]; }
    float operator()(int r, int c) const { return m[c * 4 + r]; }

    Mat4 operator*(const Mat4& o) const {
        Mat4 out;
        for (int r = 0; r < 4; ++r)
            for (int c = 0; c < 4; ++c) {
                float sum = 0.0f;
                for (int k = 0; k < 4; ++k) sum += (*this)(r, k) * o(k, c);
                out(r, c) = sum;
            }
        return out;
    }
    Vec3 point(Vec3 p) const {
        return Vec3(m[0] * p.x + m[4] * p.y + m[8] * p.z + m[12],
                    m[1] * p.x + m[5] * p.y + m[9] * p.z + m[13],
                    m[2] * p.x + m[6] * p.y + m[10] * p.z + m[14]);
    }
    // Determinant of the upper-left 3x3 block. Negative => the transform contains a reflection.
    float det3() const {
        return (*this)(0, 0) * ((*this)(1, 1) * (*this)(2, 2) - (*this)(1, 2) * (*this)(2, 1))
             - (*this)(0, 1) * ((*this)(1, 0) * (*this)(2, 2) - (*this)(1, 2) * (*this)(2, 0))
             + (*this)(0, 2) * ((*this)(1, 0) * (*this)(2, 1) - (*this)(1, 1) * (*this)(2, 0));
    }

    static Mat4 translation(float x, float y, float z) {
        Mat4 r; r(0, 3) = x; r(1, 3) = y; r(2, 3) = z; return r;
    }
    static Mat4 scaling(float x, float y, float z) {
        Mat4 r; r(0, 0) = x; r(1, 1) = y; r(2, 2) = z; return r;
    }
    static Mat4 rotationX(float deg) {
        const float c = std::cos(radians(deg)), s = std::sin(radians(deg));
        Mat4 r; r(1, 1) = c; r(1, 2) = -s; r(2, 1) = s; r(2, 2) = c; return r;
    }
    static Mat4 rotationY(float deg) {
        const float c = std::cos(radians(deg)), s = std::sin(radians(deg));
        Mat4 r; r(0, 0) = c; r(0, 2) = s; r(2, 0) = -s; r(2, 2) = c; return r;
    }
    static Mat4 rotationZ(float deg) {
        const float c = std::cos(radians(deg)), s = std::sin(radians(deg));
        Mat4 r; r(0, 0) = c; r(0, 1) = -s; r(1, 0) = s; r(1, 1) = c; return r;
    }
    // Shear: x' = x + k*y  (slides layers of constant y sideways).
    static Mat4 shearXY(float k) { Mat4 r; r(0, 1) = k; return r; }
    // Reflection through the YZ plane: x' = -x.
    static Mat4 reflectionYZ() { return scaling(-1.0f, 1.0f, 1.0f); }
    // Builds a frame from three orthogonal axes and an origin.
    static Mat4 basis(Vec3 x, Vec3 y, Vec3 z, Vec3 o) {
        Mat4 r;
        r(0, 0) = x.x; r(1, 0) = x.y; r(2, 0) = x.z;
        r(0, 1) = y.x; r(1, 1) = y.y; r(2, 1) = y.z;
        r(0, 2) = z.x; r(1, 2) = z.y; r(2, 2) = z.z;
        r(0, 3) = o.x; r(1, 3) = o.y; r(2, 3) = o.z;
        return r;
    }
};

// Pushes the matrix stack and multiplies in an explicit Mat4. A reflection flips triangle winding,
// so the front-face convention is flipped too, which keeps two-sided lighting correct.
struct MatrixScope {
    explicit MatrixScope(const Mat4& M) {
        glPushMatrix();
        glMultMatrixf(M.m);
        glFrontFace(M.det3() < 0.0f ? GL_CW : GL_CCW);
    }
    ~MatrixScope() { glFrontFace(GL_CCW); glPopMatrix(); }
};

// Small deterministic random generator (xorshift32) so every run looks the same.
struct Rng {
    unsigned state;
    explicit Rng(unsigned seed = 1u) : state(seed ? seed : 1u) {}
    unsigned nextU() { state ^= state << 13; state ^= state >> 17; state ^= state << 5; return state; }
    float uniform() { return (nextU() & 0xFFFFFFu) / 16777216.0f; }
    float range(float a, float b) { return a + (b - a) * uniform(); }
};

// ============================================================================
// 2. PRIMITIVES, MATERIALS, TEXTURES
// ============================================================================

GLUquadric* gQuadric = nullptr;
GLuint gTexHull = 0, gTexSolar = 0, gTexEarth = 0;

void drawCube(float w, float h, float d) {
    glPushMatrix();
    glScalef(w, h, d);           // scaling turns the unit cube into a box
    glutSolidCube(1.0f);
    glPopMatrix();
}

void drawSphere(float r, int slices = 32, int stacks = 24) { gluSphere(gQuadric, r, slices, stacks); }
void drawDisk(float inner, float outer, int slices = 36) { gluDisk(gQuadric, inner, outer, slices, 1); }
void drawTorus(float tube, float ring, int sides = 14, int rings = 40) { glutSolidTorus(tube, ring, sides, rings); }

// Cone / cylinder along +Z from z = 0 to z = length, optional end caps.
void drawCone(float r0, float r1, float length, bool caps = true, int slices = 36) {
    gluCylinder(gQuadric, r0, r1, length, slices, 1);
    if (caps) {
        if (r0 > 0.0f) {
            glPushMatrix(); glRotatef(180.0f, 1, 0, 0); gluDisk(gQuadric, 0.0, r0, slices, 1); glPopMatrix();
        }
        if (r1 > 0.0f) {
            glPushMatrix(); glTranslatef(0, 0, length); gluDisk(gQuadric, 0.0, r1, slices, 1); glPopMatrix();
        }
    }
}
void drawTube(float radius, float length, bool caps = true, int slices = 36) {
    drawCone(radius, radius, length, caps, slices);
}

// Half disc in the z=0 plane (side +1 => x>0, side -1 => x<0). Used for the sliding hatch doors.
void drawHalfDisc(float radius, int side) {
    const int steps = 20;
    glNormal3f(0, 0, 1);
    glBegin(GL_TRIANGLE_FAN);
    glVertex3f(0, 0, 0);
    for (int i = 0; i <= steps; ++i) {
        const float a = radians(-90.0f + 180.0f * i / steps) + (side < 0 ? PI : 0.0f);
        glVertex3f(radius * std::cos(a), radius * std::sin(a), 0.0f);
    }
    glEnd();
}

// One place sets every material property so nothing "leaks" from one object to the next.
void setMaterial(float r, float g, float b, float shine = 32.0f, float spec = 0.5f,
                 float emit = 0.0f, float alpha = 1.0f) {
    const GLfloat diffuse[]  = {r, g, b, alpha};
    const GLfloat ambient[]  = {r * 0.85f, g * 0.85f, b * 0.85f, alpha};
    const GLfloat specular[] = {spec, spec, spec, 1.0f};
    const GLfloat emission[] = {r * emit, g * emit, b * emit, 1.0f};
    glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, ambient);
    glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, diffuse);
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, specular);
    glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, emission);
    glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, shine);
}

void beginTexture(GLuint tex, float repeatS = 1.0f, float repeatT = 1.0f) {
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, tex);
    glMatrixMode(GL_TEXTURE);
    glLoadIdentity();
    glScalef(repeatS, repeatT, 1.0f);   // texture-matrix scaling tiles the image
    glMatrixMode(GL_MODELVIEW);
}
void endTexture() {
    glMatrixMode(GL_TEXTURE);
    glLoadIdentity();
    glMatrixMode(GL_MODELVIEW);
    glDisable(GL_TEXTURE_2D);
}

// ---- Procedural textures (no image files needed, so the program is self-contained) ----

GLuint uploadTexture(int w, int h, const std::vector<unsigned char>& px, bool clampT) {
    GLuint id = 0;
    glGenTextures(1, &id);
    glBindTexture(GL_TEXTURE_2D, id);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    gluBuild2DMipmaps(GL_TEXTURE_2D, 3, w, h, GL_RGB, GL_UNSIGNED_BYTE, &px[0]);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, clampT ? GL_CLAMP_TO_EDGE : GL_REPEAT);
    return id;
}

inline void putPixel(std::vector<unsigned char>& px, int idx, float r, float g, float b) {
    px[idx * 3 + 0] = (unsigned char)(clampf(r, 0, 1) * 255.0f);
    px[idx * 3 + 1] = (unsigned char)(clampf(g, 0, 1) * 255.0f);
    px[idx * 3 + 2] = (unsigned char)(clampf(b, 0, 1) * 255.0f);
}

GLuint makeHullTexture() {           // hull plating: seams, rivets, grain
    const int N = 64;
    std::vector<unsigned char> px(N * N * 3);
    Rng rng(7);
    for (int y = 0; y < N; ++y)
        for (int x = 0; x < N; ++x) {
            float v = 0.90f + 0.08f * rng.uniform();
            if (x < 2 || y < 2) v *= 0.60f;
            if ((x == 8 || x == N - 8) && (y == 8 || y == N - 8)) v *= 0.65f;
            putPixel(px, y * N + x, v, v, v);
        }
    return uploadTexture(N, N, px, false);
}

GLuint makeSolarTexture() {          // one photovoltaic cell: blue silicon, silver border, bus bars
    const int N = 64;
    std::vector<unsigned char> px(N * N * 3);
    for (int y = 0; y < N; ++y)
        for (int x = 0; x < N; ++x) {
            float r, g, b;
            if (x < 2 || y < 2) { r = 0.72f; g = 0.75f; b = 0.82f; }
            else {
                const float t = y / (float)N;
                r = 0.03f + 0.05f * t; g = 0.09f + 0.09f * t; b = 0.30f + 0.22f * t;
                if (x % 16 == 8) { r += 0.12f; g += 0.12f; b += 0.14f; }
            }
            putPixel(px, y * N + x, r, g, b);
        }
    return uploadTexture(N, N, px, false);
}

float hash01(int x, int y, int seed) {
    unsigned h = (unsigned)x * 374761393u + (unsigned)y * 668265263u + (unsigned)seed * 2246822519u;
    h = (h ^ (h >> 13)) * 1274126177u;
    h ^= h >> 16;
    return (h & 0xFFFFFFu) / 16777216.0f;
}
// Value noise that wraps horizontally so the planet texture has no seam.
float valueNoise(float x, float y, int periodX, int seed) {
    const int xi = (int)std::floor(x), yi = (int)std::floor(y);
    const float tx = smooth01(x - xi), ty = smooth01(y - yi);
    const int x0 = ((xi % periodX) + periodX) % periodX;
    const int x1 = (((xi + 1) % periodX) + periodX) % periodX;
    const float a = hash01(x0, yi, seed),     b = hash01(x1, yi, seed);
    const float c = hash01(x0, yi + 1, seed), d = hash01(x1, yi + 1, seed);
    return lerpf(lerpf(a, b, tx), lerpf(c, d, tx), ty);
}
float fbmPeriodic(float u, float v, int basePeriod, int octaves, int seed) {
    float sum = 0.0f, amp = 0.5f, norm = 0.0f;
    int period = basePeriod;
    for (int o = 0; o < octaves; ++o) {
        sum += amp * valueNoise(u * period, v * period * 0.5f, period, seed + o * 17);
        norm += amp; amp *= 0.5f; period *= 2;
    }
    return sum / norm;
}

GLuint makeEarthTexture() {          // oceans, continents, deserts, ice caps, clouds
    const int W = 1024, H = 512;
    std::vector<unsigned char> px(W * H * 3);
    for (int y = 0; y < H; ++y) {
        const float v = (y + 0.5f) / H;
        const float lat = std::fabs(v - 0.5f) * 2.0f;                  // 0 equator .. 1 pole
        for (int x = 0; x < W; ++x) {
            const float u = (x + 0.5f) / W;
            const float h = fbmPeriodic(u, v, 4, 6, 11);
            const float detail = fbmPeriodic(u, v, 16, 4, 29);
            const float sea = 0.53f;
            float r, g, b;
            if (h < sea) {
                const float d = clampf((sea - h) / 0.22f, 0.0f, 1.0f);
                r = lerpf(0.06f, 0.01f, d); g = lerpf(0.34f, 0.07f, d); b = lerpf(0.58f, 0.24f, d);
            } else {
                const float e = clampf((h - sea) / 0.20f, 0.0f, 1.0f);
                if (e < 0.55f) {
                    const float k = e / 0.55f;
                    r = lerpf(0.14f, 0.36f, k); g = lerpf(0.38f, 0.34f, k); b = lerpf(0.10f, 0.17f, k);
                } else {
                    const float k = (e - 0.55f) / 0.45f;
                    r = lerpf(0.36f, 0.93f, k); g = lerpf(0.34f, 0.94f, k); b = lerpf(0.17f, 0.96f, k);
                }
                const float belt = smooth01((lat - 0.12f) / 0.08f) * (1.0f - smooth01((lat - 0.38f) / 0.10f));
                if (belt > 0.0f && detail > 0.52f) {                     // desert belts (soft latitude edges)
                    const float k = smooth01((detail - 0.52f) / 0.15f) * 0.8f * belt;
                    r = lerpf(r, 0.74f, k); g = lerpf(g, 0.62f, k); b = lerpf(b, 0.36f, k);
                }
            }
            const float ice = smooth01((lat - 0.82f) / 0.08f);            // polar caps
            r = lerpf(r, 0.93f, ice); g = lerpf(g, 0.96f, ice); b = lerpf(b, 0.98f, ice);
            const float c = smooth01((fbmPeriodic(u, v, 6, 5, 101) - 0.56f) / 0.16f) * 0.85f;
            r = lerpf(r, 0.97f, c); g = lerpf(g, 0.97f, c); b = lerpf(b, 0.98f, c);
            putPixel(px, y * W + x, r, g, b);
        }
    }
    return uploadTexture(W, H, px, true);
}

// ============================================================================
// 3. CUSTOM MESHES (built from raw vertices, drawn from display lists)
// ============================================================================

struct Mesh {
    std::vector<Vec3> pos, nrm;      // triangle soup: every 3 consecutive vertices form one triangle
    GLuint list;
    Mesh() : list(0) {}

    // Adds a triangle whose winding is corrected to agree with the supplied vertex normals.
    void tri(Vec3 a, Vec3 b, Vec3 c, Vec3 na, Vec3 nb, Vec3 nc) {
        const Vec3 g = cross(b - a, c - a);
        if (length(g) < 1e-7f) return;                       // skip degenerate triangles
        if (dot(g, na + nb + nc) < 0.0f) { std::swap(b, c); std::swap(nb, nc); }
        pos.push_back(a); pos.push_back(b); pos.push_back(c);
        nrm.push_back(na); nrm.push_back(nb); nrm.push_back(nc);
    }
    // Flat-shaded triangle whose face normal points away from the interior point `inside`.
    void flatTri(Vec3 a, Vec3 b, Vec3 c, Vec3 inside) {
        Vec3 n = normalize(cross(b - a, c - a));
        if (dot(n, (a + b + c) * (1.0f / 3.0f) - inside) < 0.0f) n = n * -1.0f;
        tri(a, b, c, n, n, n);
    }
    void compile() {
        list = glGenLists(1);
        glNewList(list, GL_COMPILE);
        glBegin(GL_TRIANGLES);
        for (size_t i = 0; i < pos.size(); ++i) {
            glNormal3f(nrm[i].x, nrm[i].y, nrm[i].z);
            glVertex3f(pos[i].x, pos[i].y, pos[i].z);
        }
        glEnd();
        glEndList();
    }
    void draw() const { glCallList(list); }
};

struct ProfilePt { float r, z; };

// Surface of revolution about the Z axis from a 2D (radius, z) profile. Normals come from the profile
// tangent; repeat a point twice to get a sharp crease.
Mesh makeLathe(const std::vector<ProfilePt>& p, int slices) {
    Mesh m;
    const int n = (int)p.size();
    std::vector<Vec3> pn(n);
    for (int i = 0; i < n; ++i) {
        const int i0 = i > 0 ? i - 1 : i, i1 = i < n - 1 ? i + 1 : i;
        const float dr = p[i1].r - p[i0].r, dz = p[i1].z - p[i0].z;
        const float l = std::sqrt(dr * dr + dz * dz);
        pn[i] = l < 1e-6f ? Vec3(1, 0, 0) : Vec3(dz / l, 0, -dr / l);
    }
    for (int i = 0; i + 1 < n; ++i)
        for (int j = 0; j < slices; ++j) {
            const float a0 = 2.0f * PI * j / slices, a1 = 2.0f * PI * (j + 1) / slices;
            const float c0 = std::cos(a0), s0 = std::sin(a0), c1 = std::cos(a1), s1 = std::sin(a1);
            const Vec3 P00(p[i].r * c0, p[i].r * s0, p[i].z),         P01(p[i].r * c1, p[i].r * s1, p[i].z);
            const Vec3 P10(p[i + 1].r * c0, p[i + 1].r * s0, p[i + 1].z), P11(p[i + 1].r * c1, p[i + 1].r * s1, p[i + 1].z);
            const Vec3 N00(pn[i].x * c0, pn[i].x * s0, pn[i].z),       N01(pn[i].x * c1, pn[i].x * s1, pn[i].z);
            const Vec3 N10(pn[i + 1].x * c0, pn[i + 1].x * s0, pn[i + 1].z), N11(pn[i + 1].x * c1, pn[i + 1].x * s1, pn[i + 1].z);
            m.tri(P00, P01, P11, N00, N01, N11);
            m.tri(P00, P11, P10, N00, N11, N10);
        }
    m.compile();
    return m;
}

// Adds a convex polyhedron given by vertices and polygon faces; every face is oriented outward.
void addConvexSolid(Mesh& m, const std::vector<Vec3>& v, const std::vector<std::vector<int> >& faces) {
    Vec3 c;
    for (size_t i = 0; i < v.size(); ++i) c = c + v[i];
    c = c * (1.0f / (float)v.size());
    for (size_t f = 0; f < faces.size(); ++f)
        for (size_t k = 1; k + 1 < faces[f].size(); ++k)
            m.flatTri(v[faces[f][0]], v[faces[f][k]], v[faces[f][k + 1]], c);
}

// Lumpy rock: subdivided icosahedron with radial noise, flat shaded. Used for space debris.
Mesh makeAsteroid(unsigned seed, int subdiv, float roughness, Vec3 stretch) {
    Rng rng(seed);
    const float t = (1.0f + std::sqrt(5.0f)) / 2.0f;
    std::vector<Vec3> v;
    const float base[12][3] = {{-1, t, 0}, {1, t, 0}, {-1, -t, 0}, {1, -t, 0}, {0, -1, t}, {0, 1, t},
                               {0, -1, -t}, {0, 1, -t}, {t, 0, -1}, {t, 0, 1}, {-t, 0, -1}, {-t, 0, 1}};
    for (int i = 0; i < 12; ++i) v.push_back(normalize(Vec3(base[i][0], base[i][1], base[i][2])));
    struct Tri { int a, b, c; };
    const int faceIdx[20][3] = {{0, 11, 5}, {0, 5, 1}, {0, 1, 7}, {0, 7, 10}, {0, 10, 11}, {1, 5, 9}, {5, 11, 4},
                                {11, 10, 2}, {10, 7, 6}, {7, 1, 8}, {3, 9, 4}, {3, 4, 2}, {3, 2, 6}, {3, 6, 8},
                                {3, 8, 9}, {4, 9, 5}, {2, 4, 11}, {6, 2, 10}, {8, 6, 7}, {9, 8, 1}};
    std::vector<Tri> faces;
    for (int i = 0; i < 20; ++i) { Tri f = {faceIdx[i][0], faceIdx[i][1], faceIdx[i][2]}; faces.push_back(f); }

    for (int s = 0; s < subdiv; ++s) {
        std::map<std::pair<int, int>, int> cache;
        std::vector<Tri> next;
        for (size_t i = 0; i < faces.size(); ++i) {
            const int id[3] = {faces[i].a, faces[i].b, faces[i].c};
            int mid[3];
            for (int e = 0; e < 3; ++e) {
                const int a = id[e], b = id[(e + 1) % 3];
                const std::pair<int, int> key(a < b ? a : b, a < b ? b : a);
                std::map<std::pair<int, int>, int>::iterator it = cache.find(key);
                if (it == cache.end()) {
                    v.push_back(normalize(v[a] + v[b]));
                    it = cache.insert(std::make_pair(key, (int)v.size() - 1)).first;
                }
                mid[e] = it->second;
            }
            Tri t0 = {id[0], mid[0], mid[2]}, t1 = {id[1], mid[1], mid[0]};
            Tri t2 = {id[2], mid[2], mid[1]}, t3 = {mid[0], mid[1], mid[2]};
            next.push_back(t0); next.push_back(t1); next.push_back(t2); next.push_back(t3);
        }
        faces = next;
    }
    const float p1 = rng.range(0, 6), p2 = rng.range(0, 6), p3 = rng.range(0, 6);
    for (size_t i = 0; i < v.size(); ++i) {
        const float lobes = 0.5f * std::sin(2.3f * v[i].x + p1) + 0.35f * std::sin(3.1f * v[i].y + p2)
                          + 0.30f * std::sin(2.7f * v[i].z + p3);
        const float rad = 1.0f + roughness * (0.6f * lobes + 0.4f * rng.range(-1.0f, 1.0f));
        v[i] = Vec3(v[i].x * rad * stretch.x, v[i].y * rad * stretch.y, v[i].z * rad * stretch.z);
    }
    Mesh m;
    for (size_t i = 0; i < faces.size(); ++i)
        m.flatTri(v[faces[i].a], v[faces[i].b], v[faces[i].c], Vec3(0, 0, 0));
    m.compile();
    return m;
}

// Asymmetric "probe" (fuselage + two swept wings + tail fin). Being asymmetric makes rotation,
// reflection and shear easy to see in the transformation lab.
void addWing(Mesh& m, float sx) {
    std::vector<Vec3> v;
    v.push_back(Vec3(sx * 0.5f, 0.03f, 0.2f));  v.push_back(Vec3(sx * 0.5f, 0.03f, -1.0f));
    v.push_back(Vec3(sx * 0.5f, -0.03f, -1.0f)); v.push_back(Vec3(sx * 0.5f, -0.03f, 0.2f));
    v.push_back(Vec3(sx * 2.4f, 0.03f, -0.7f)); v.push_back(Vec3(sx * 2.4f, 0.03f, -1.2f));
    v.push_back(Vec3(sx * 2.4f, -0.03f, -1.2f)); v.push_back(Vec3(sx * 2.4f, -0.03f, -0.7f));
    const int f[6][4] = {{0, 1, 5, 4}, {3, 7, 6, 2}, {0, 3, 2, 1}, {4, 5, 6, 7}, {0, 4, 7, 3}, {1, 2, 6, 5}};
    std::vector<std::vector<int> > faces;
    for (int i = 0; i < 6; ++i) faces.push_back(std::vector<int>(f[i], f[i] + 4));
    addConvexSolid(m, v, faces);
}

Mesh makeProbeMesh() {
    Mesh m;
    // fuselage: three rings (rear box, mid box, tapered nose) + a tip
    const float ringZ[3] = {-1.2f, 0.3f, 1.0f}, ringW[3] = {0.5f, 0.5f, 0.28f}, ringH[3] = {0.4f, 0.4f, 0.22f};
    std::vector<Vec3> v;
    for (int r = 0; r < 3; ++r) {
        v.push_back(Vec3(-ringW[r], -ringH[r], ringZ[r])); v.push_back(Vec3(ringW[r], -ringH[r], ringZ[r]));
        v.push_back(Vec3(ringW[r], ringH[r], ringZ[r]));   v.push_back(Vec3(-ringW[r], ringH[r], ringZ[r]));
    }
    v.push_back(Vec3(0, 0, 1.7f));
    std::vector<std::vector<int> > faces;
    const int rear[4] = {0, 1, 2, 3};
    faces.push_back(std::vector<int>(rear, rear + 4));
    for (int a = 0; a < 2; ++a)
        for (int k = 0; k < 4; ++k) {
            const int q[4] = {4 * a + k, 4 * a + (k + 1) % 4, 4 * (a + 1) + (k + 1) % 4, 4 * (a + 1) + k};
            faces.push_back(std::vector<int>(q, q + 4));
        }
    for (int k = 0; k < 4; ++k) {
        const int q[3] = {8 + k, 8 + (k + 1) % 4, 12};
        faces.push_back(std::vector<int>(q, q + 3));
    }
    addConvexSolid(m, v, faces);
    addWing(m, +1.0f);
    addWing(m, -1.0f);
    // tail fin
    std::vector<Vec3> fin;
    fin.push_back(Vec3(-0.04f, 0.35f, 0.0f));  fin.push_back(Vec3(0.04f, 0.35f, 0.0f));
    fin.push_back(Vec3(0.04f, 0.35f, -1.1f));  fin.push_back(Vec3(-0.04f, 0.35f, -1.1f));
    fin.push_back(Vec3(-0.04f, 1.3f, -0.8f));  fin.push_back(Vec3(0.04f, 1.3f, -0.8f));
    fin.push_back(Vec3(0.04f, 1.3f, -1.2f));   fin.push_back(Vec3(-0.04f, 1.3f, -1.2f));
    std::vector<std::vector<int> > ff;
    const int fq[6][4] = {{0, 1, 2, 3}, {4, 5, 6, 7}, {0, 1, 5, 4}, {1, 2, 6, 5}, {2, 3, 7, 6}, {3, 0, 4, 7}};
    for (int i = 0; i < 6; ++i) ff.push_back(std::vector<int>(fq[i], fq[i] + 4));
    addConvexSolid(m, fin, ff);
    m.compile();
    return m;
}

struct MeshLibrary {
    Mesh dish, nozzle, asteroid[3], probe;
} gMesh;

void buildMeshes() {
    // parabolic dish: z = r^2 / (4 f), opening toward +Z
    std::vector<ProfilePt> dish;
    for (int i = 0; i <= 14; ++i) {
        const float r = 1.15f * i / 14.0f;
        ProfilePt p = {r, r * r / (4.0f * 0.75f)};
        dish.push_back(p);
    }
    gMesh.dish = makeLathe(dish, 40);

    // engine bell, throat at z=0, opening toward -Z
    const float bell[][2] = {{0.30f, 0.0f}, {0.36f, -0.20f}, {0.50f, -0.50f}, {0.72f, -0.90f},
                             {0.92f, -1.25f}, {1.02f, -1.45f}};
    std::vector<ProfilePt> nz;
    for (int i = 0; i < 6; ++i) { ProfilePt p = {bell[i][0], bell[i][1]}; nz.push_back(p); }
    gMesh.nozzle = makeLathe(nz, 36);

    gMesh.asteroid[0] = makeAsteroid(3u, 2, 0.32f, Vec3(1.25f, 0.85f, 1.0f));
    gMesh.asteroid[1] = makeAsteroid(11u, 2, 0.28f, Vec3(1.0f, 1.0f, 1.35f));
    gMesh.asteroid[2] = makeAsteroid(29u, 1, 0.38f, Vec3(0.9f, 1.2f, 1.0f));
    gMesh.probe = makeProbeMesh();
}

// ============================================================================
// 4. APPLICATION STATE
// ============================================================================

const int InitialWidth = 1280;
const int InitialHeight = 760;
const float MinCamDist = 6.0f;
const float MaxCamDist = 90.0f;

// Docking geometry. The station frame equals the world frame and the docking axis is +Z.
const float PortBaseZ = 4.5f;                          // adapter starts on the core here
const float AdapterLength = 1.23f;                     // adapter + collar length
const float PortFaceZ = PortBaseZ + AdapterLength;     // station mating face
const float ShipNoseLength = 2.73f;                    // ship origin -> ship mating face
const float HoldZ = 16.0f;                             // final-approach hold point
const float ParkZ = 36.0f;                             // parking position

const Vec3 SunDir = normalize(Vec3(0.85f, 0.45f, 0.30f));   // direction *towards* the sun
const Vec3 EarthPos(-117.0f, -119.0f, -54.0f);
const float EarthRadius = 62.0f;
const Vec3 DishPivot(0.0f, 4.9f, -1.8f);               // antenna gimbal position on the station

// ---- window & camera ----
int winW = InitialWidth, winH = InitialHeight;

struct CameraState { float yaw, pitch, dist; Vec3 target; };
CameraState cam = {40.0f, 22.0f, 34.0f, Vec3(0.0f, 0.0f, 1.5f)};
CameraState camGoal = cam;
bool chaseCamera = false;
Vec3 camEye;
int camPresetId = 1;

// ---- input ----
int dragButton = -1, lastMouseX = 0, lastMouseY = 0;
bool keyHeld[256];
bool specialHeld[256];

// ---- UI flags ----
bool paused = false, wireframe = false, showGuides = true, showHelp = true, labMode = false;
char toastText[160] = "";
float toastTimer = 0.0f;
float fpsValue = 0.0f;

// ---- animation state ----
float simTime = 0.0f;
float ringAngle = 0.0f;               // habitat ring spin
float solarAngle = 0.0f;              // continuous solar wing rotation
float solarTrackAngle = 0.0f;         // smoothed sun-tracking angle
bool solarTracking = false;
float earthSpin = 0.0f;
float satAngle = 20.0f;               // orbital phase, degrees
Vec3 satPos;
Mat4 satFrame;
float dishYaw = 0.0f, dishPitch = 0.0f;


// robotic arm (all angles in degrees; grip is the open fraction 0..1)
struct ArmState { float base, shoulder, elbow, wrist, grip; };
const ArmState ArmHome = {90.0f, 28.0f, 55.0f, -15.0f, 0.6f};
ArmState arm = ArmHome;
const Mat4 ArmMount = Mat4::translation(0.0f, 1.3f, 2.8f);

// space debris
struct Debris { Vec3 pos, vel, axis; float angle, spin, size, retarget; int kind; };
std::vector<Debris> debris;

// transformation lab
struct ProbeState {
    Vec3 pos; float yaw, pitch, roll, scale;
    float mirror, mirrorTarget, shear, shearTarget;
};
ProbeState probe;
float labTime = 0.0f;
int labSelected = 0;
const int LabDemoCount = 6;

GLuint gListStars = 0, gListTruss = 0, gListRing = 0;

// ============================================================================
// 5. STATION GEOMETRY (modelled with primitives, meshes and hierarchical transforms)
// ============================================================================

// Main truss (static geometry, compiled into a display list).
void drawTrussStatic() {
    const float halfLen = 9.0f, h = 0.45f, bay = 1.0f;
    glPushMatrix();
    glTranslatef(0, 0, -3.4f);
    setMaterial(0.72f, 0.74f, 0.78f, 30.0f, 0.4f);
    for (int sy = -1; sy <= 1; sy += 2)
        for (int sz = -1; sz <= 1; sz += 2) {
            glPushMatrix(); glTranslatef(0, sy * h, sz * h); drawCube(2 * halfLen, 0.11f, 0.11f); glPopMatrix();
        }
    const int bays = (int)(2 * halfLen / bay);
    for (int i = 0; i <= bays; ++i) {
        const float x = -halfLen + i * bay;
        for (int s = -1; s <= 1; s += 2) {
            glPushMatrix(); glTranslatef(x, s * h, 0); drawCube(0.07f, 0.07f, 2 * h); glPopMatrix();
            glPushMatrix(); glTranslatef(x, 0, s * h); drawCube(0.07f, 2 * h, 0.07f); glPopMatrix();
        }
    }
    const float len = std::sqrt(bay * bay + 4 * h * h), ang = degrees(std::atan2(2 * h, bay));
    for (int i = 0; i < bays; ++i) {
        const float x = -halfLen + (i + 0.5f) * bay, dir = (i % 2 == 0) ? 1.0f : -1.0f;
        for (int sz = -1; sz <= 1; sz += 2) {
            glPushMatrix(); glTranslatef(x, 0, sz * h); glRotatef(dir * ang, 0, 0, 1);
            drawCube(len, 0.05f, 0.05f); glPopMatrix();
        }
    }
    // radiators on stand-offs (scale + translate of a thin box)
    for (int side = -1; side <= 1; side += 2) {
        setMaterial(0.92f, 0.93f, 0.95f, 20.0f, 0.3f);
        glPushMatrix(); glTranslatef(side * 3.4f, 1.7f, 0); drawCube(2.6f, 0.05f, 1.6f); glPopMatrix();
        setMaterial(0.40f, 0.43f, 0.48f, 20.0f, 0.3f);
        glPushMatrix(); glTranslatef(side * 3.4f, 1.1f, 0); drawCube(0.07f, 1.2f, 0.07f); glPopMatrix();
    }
    glPopMatrix();
}

// Habitat ring: torus + spokes + habitat blocks + window lights. Spins as one rigid body.
void drawHabitatRingStatic() {
    setMaterial(0.74f, 0.77f, 0.82f, 40.0f, 0.5f);
    drawTorus(0.5f, 5.0f, 24, 96);
    setMaterial(0.40f, 0.44f, 0.50f, 30.0f, 0.4f);
    for (int i = 0; i < 6; ++i) {                     // spokes
        glPushMatrix();
        glRotatef(i * 60.0f, 0, 0, 1);
        glTranslatef(1.5f, 0, 0);
        glRotatef(90.0f, 0, 1, 0);
        drawTube(0.2f, 3.1f, true, 16);
        glPopMatrix();
    }
    setMaterial(0.62f, 0.66f, 0.72f, 30.0f, 0.4f);
    for (int i = 0; i < 6; ++i) {                     // habitat blocks
        glPushMatrix(); glRotatef(i * 60.0f, 0, 0, 1); glTranslatef(4.8f, 0, 0);
        drawCube(1.0f, 1.2f, 1.3f); glPopMatrix();
    }
    for (int i = 0; i < 48; ++i) {                    // window lights (front face and outer rim)
        const bool lit = ((i * 7 + i / 5) % 3) != 0;
        if (lit) {
            if (i % 2) setMaterial(1.0f, 0.78f, 0.40f, 8.0f, 0.0f, 0.95f);
            else       setMaterial(0.60f, 0.85f, 1.0f, 8.0f, 0.0f, 0.90f);
        } else setMaterial(0.05f, 0.08f, 0.12f, 60.0f, 0.8f);
        glPushMatrix();
        glRotatef(i * 7.5f, 0, 0, 1);
        glPushMatrix(); glTranslatef(5.0f, 0, 0.5f); drawCube(0.30f, 0.36f, 0.04f); glPopMatrix();
        glPushMatrix(); glTranslatef(5.5f, 0, 0.0f); drawCube(0.04f, 0.36f, 0.30f); glPopMatrix();
        glPopMatrix();
    }
}

void drawSolarBlade(float w, float l, float tilesU, float tilesV) {
    setMaterial(0.50f, 0.53f, 0.58f, 30.0f, 0.5f);
    drawCube(w + 0.12f, l + 0.12f, 0.07f);            // frame + back side
    glEnable(GL_POLYGON_OFFSET_FILL);
    glPolygonOffset(-1.0f, -1.0f);
    setMaterial(0.95f, 0.97f, 1.0f, 90.0f, 0.9f);
    beginTexture(gTexSolar);
    glBegin(GL_QUADS);
    glNormal3f(0, 0, 1);
    glTexCoord2f(0, 0);          glVertex3f(-w / 2, -l / 2, 0.036f);
    glTexCoord2f(tilesU, 0);     glVertex3f(w / 2, -l / 2, 0.036f);
    glTexCoord2f(tilesU, tilesV); glVertex3f(w / 2, l / 2, 0.036f);
    glTexCoord2f(0, tilesV);     glVertex3f(-w / 2, l / 2, 0.036f);
    glEnd();
    endTexture();
    glDisable(GL_POLYGON_OFFSET_FILL);
}

// Solar wing: fixed rotary-joint housing -> rotating hub -> two blades (hierarchical transform).
void drawSolarWing(float side, float angle) {
    glPushMatrix();
    glTranslatef(side * 8.0f, 0, -3.4f);
    setMaterial(0.35f, 0.38f, 0.44f, 40.0f, 0.5f);
    glPushMatrix(); glTranslatef(0, 0, 0); glRotatef(90.0f, 0, 1, 0); glTranslatef(0, 0, -1.4f);
    drawTube(0.42f, 2.8f, true, 24); glPopMatrix();                // housing on the truss
    glRotatef(angle, 1, 0, 0);                                     // rotation about the truss axis
    setMaterial(0.80f, 0.60f, 0.20f, 60.0f, 0.7f);
    glPushMatrix(); glRotatef(90.0f, 0, 1, 0); glTranslatef(0, 0, -1.2f);
    drawTube(0.52f, 2.4f, true, 24); glPopMatrix();                // rotating hub
    for (int s = -1; s <= 1; s += 2) {
        glPushMatrix(); glTranslatef(0, s * 3.05f, 0); drawSolarBlade(2.4f, 5.0f, 5.0f, 10.0f); glPopMatrix();
    }
    glPopMatrix();
}

void drawCoreAndHub() {
    glPushMatrix();
    glTranslatef(0, 0, -5.0f);
    setMaterial(0.82f, 0.84f, 0.88f, 40.0f, 0.5f);
    beginTexture(gTexHull, 16.0f, 8.0f);
    drawTube(1.3f, 9.5f, true);
    endTexture();
    glPopMatrix();

    setMaterial(0.36f, 0.40f, 0.46f, 30.0f, 0.4f);
    for (int i = 0; i < 7; ++i) {
        glPushMatrix(); glTranslatef(0, 0, -4.3f + i * 1.4f); drawTorus(0.07f, 1.31f, 10, 40); glPopMatrix();
    }
    setMaterial(0.74f, 0.78f, 0.84f, 60.0f, 0.7f);
    drawSphere(1.75f, 36, 24);                                     // central hub node

    // aft engine module + nozzle
    glPushMatrix();
    glTranslatef(0, 0, -7.5f);
    setMaterial(0.80f, 0.62f, 0.24f, 50.0f, 0.7f);
    beginTexture(gTexHull, 10.0f, 2.0f);
    drawTube(1.5f, 2.5f, true);
    endTexture();
    glTranslatef(0, 0, 0.0f);
    setMaterial(0.26f, 0.28f, 0.32f, 60.0f, 0.8f);
    gMesh.nozzle.draw();
    glPopMatrix();
    setMaterial(0.55f, 0.58f, 0.64f, 30.0f, 0.5f);
    for (int i = 0; i < 4; ++i) {                                  // RCS thruster blocks
        glPushMatrix(); glRotatef(45.0f + i * 90.0f, 0, 0, 1); glTranslatef(1.62f, 0, -5.7f);
        drawCube(0.25f, 0.25f, 0.5f); glPopMatrix();
    }
}

// Zenith lab + tracking antenna (two-level hierarchy: azimuth then elevation) + nadir airlock.
void drawLabModules() {
    glPushMatrix();
    glTranslatef(0, 1.0f, -1.8f);
    glRotatef(-90.0f, 1, 0, 0);                                    // +Z of the tube becomes +Y
    setMaterial(0.80f, 0.82f, 0.86f, 40.0f, 0.5f);
    beginTexture(gTexHull, 10.0f, 3.0f);
    drawTube(1.0f, 3.3f, true);
    endTexture();
    glTranslatef(0, 0, 3.3f);
    setMaterial(0.92f, 0.52f, 0.14f, 50.0f, 0.7f);
    drawTorus(0.1f, 1.0f, 10, 32);
    glPopMatrix();

    glPushMatrix();                                                // antenna mast + gimbal
    glTranslatef(0, 4.3f, -1.8f);
    setMaterial(0.35f, 0.38f, 0.44f, 30.0f, 0.4f);
    glPushMatrix(); glRotatef(-90.0f, 1, 0, 0); drawTube(0.10f, 0.6f, true, 16); glPopMatrix();
    glTranslatef(0, 0.6f, 0);
    glRotatef(dishYaw, 0, 1, 0);                                   // level 1: azimuth
    drawCube(0.5f, 0.22f, 0.5f);
    glRotatef(-dishPitch, 1, 0, 0);                                // level 2: elevation
    setMaterial(0.92f, 0.93f, 0.96f, 60.0f, 0.7f);
    gMesh.dish.draw();
    setMaterial(0.30f, 0.32f, 0.36f, 30.0f, 0.4f);
    drawTube(0.03f, 0.75f, false, 8);
    glTranslatef(0, 0, 0.75f);
    drawSphere(0.09f, 12, 8);
    glPopMatrix();

    glPushMatrix();                                                // nadir airlock
    glTranslatef(0, -1.0f, -1.8f);
    glRotatef(90.0f, 1, 0, 0);
    setMaterial(0.78f, 0.80f, 0.85f, 40.0f, 0.5f);
    beginTexture(gTexHull, 10.0f, 3.0f);
    drawTube(1.0f, 3.0f, true);
    endTexture();
    glTranslatef(0, 0, 3.0f);
    setMaterial(0.92f, 0.52f, 0.14f, 50.0f, 0.7f);
    drawTorus(0.1f, 1.0f, 10, 32);
    glPopMatrix();
    setMaterial(0.60f, 0.64f, 0.70f, 30.0f, 0.4f);
    glPushMatrix(); glTranslatef(1.4f, -2.3f, -1.8f); drawCube(0.9f, 1.1f, 1.2f); glPopMatrix();
    glPushMatrix(); glTranslatef(-1.4f, -2.7f, -1.8f); drawCube(0.9f, 1.4f, 1.2f); glPopMatrix();
}

void drawBeacons() {
    const float blink = 0.5f + 0.5f * std::sin(simTime * 5.0f);
    const float pts[5][3] = {{-9.2f, 0, -3.4f}, {9.2f, 0, -3.4f}, {0, 0, -8.5f}, {0, 5.6f, 0}, {0, -5.6f, 0}};
    for (int i = 0; i < 5; ++i) {
        const float on = (i == 0) ? 1.0f : blink;
        if (i == 0) setMaterial(1.0f, 0.15f, 0.15f, 8, 0, 1.0f);
        else if (i == 1) setMaterial(0.15f, 1.0f, 0.25f, 8, 0, 1.0f);
        else setMaterial(1.0f, 1.0f, 1.0f, 8, 0, on);
        glPushMatrix(); glTranslatef(pts[i][0], pts[i][1], pts[i][2]); drawSphere(0.11f, 10, 8); glPopMatrix();
    }
}

// ---- Robotic arm: every joint is an explicit Mat4 so drawing and forward kinematics share one chain ----
struct ArmFrames { Mat4 turret, upper, fore, hand, tool; };

ArmFrames computeArmFrames() {
    ArmFrames f;
    f.turret = ArmMount * Mat4::translation(0, 0.35f, 0) * Mat4::rotationY(arm.base);            // base yaw
    f.upper  = f.turret * Mat4::translation(0, 0.35f, 0) * Mat4::rotationZ(arm.shoulder);          // shoulder
    f.fore   = f.upper * Mat4::translation(0, 2.8f, 0) * Mat4::rotationZ(arm.elbow);               // elbow
    f.hand   = f.fore * Mat4::translation(0, 2.4f, 0) * Mat4::rotationZ(arm.wrist);                // wrist
    f.tool   = f.hand * Mat4::translation(0, 0.7f, 0);                                              // end effector
    return f;
}
Vec3 armToolPosition() { return computeArmFrames().tool.point(Vec3(0, 0.05f, 0)); }


void drawSpaceStation() {
    glCallList(gListTruss);
    drawSolarWing(+1.0f, solarTracking ? solarTrackAngle : solarAngle);
    drawSolarWing(-1.0f, solarTracking ? solarTrackAngle : solarAngle);
    drawCoreAndHub();
    glPushMatrix();                                   // the whole ring spins about the station axis
    glRotatef(ringAngle, 0, 0, 1);
    glCallList(gListRing);
    glPopMatrix();
    drawLabModules();
    drawBeacons();
}

// ============================================================================
// 6. SPACECRAFT, SATELLITE, DEBRIS, PLANET AND SKY
// ============================================================================

// Additive-blend "light" pass: lighting off, depth writes off. Restores everything on exit.
struct GlowScope {
    GlowScope() {
        glDisable(GL_LIGHTING);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE);
        glDepthMask(GL_FALSE);
    }
    ~GlowScope() {
        glDepthMask(GL_TRUE);
        glDisable(GL_BLEND);
        glEnable(GL_LIGHTING);
    }
};

// Camera-facing axes taken from the current modelview matrix (call with only the view matrix loaded).
void viewAxes(Vec3& right, Vec3& up) {
    GLfloat mv[16];
    glGetFloatv(GL_MODELVIEW_MATRIX, mv);
    right = Vec3(mv[0], mv[4], mv[8]);
    up = Vec3(mv[1], mv[5], mv[9]);
}

Rng gRng(20240921u);

Vec3 randomUnit() {
    const float z = gRng.range(-1.0f, 1.0f), a = gRng.range(0.0f, 2.0f * PI);
    const float r = std::sqrt(1.0f - z * z);
    return Vec3(r * std::cos(a), r * std::sin(a), z);
}


const Vec3 PathP0(13.0f, 8.0f, ParkZ);           // parking position
const Vec3 PathP1(9.0f, 4.0f, 28.0f);
const Vec3 PathP2(0.0f, 0.0f, HoldZ + 9.0f);
const Vec3 PathP3(0.0f, 0.0f, HoldZ);            // hold point, exactly on the docking axis
const float ApproachSeconds = 16.0f;

float shipSpeed = 0.0f;
Mat4 shipFrame;

// Curved approach (cubic Bezier eased with smoothstep) followed by a straight final approach.
void shipPath(float s, Vec3& pos, Vec3& fwd) {
    s = clampf(s, 0.0f, 2.0f);
    if (s <= 1.0f) {
        const float u = smooth01(s), v = 1.0f - u;
        pos = PathP0 * (v * v * v) + PathP1 * (3.0f * v * v * u) + PathP2 * (3.0f * v * u * u) + PathP3 * (u * u * u);
        fwd = normalize((PathP1 - PathP0) * (3.0f * v * v) + (PathP2 - PathP1) * (6.0f * v * u) +
                        (PathP3 - PathP2) * (3.0f * u * u));
    } else {
        fwd = Vec3(0.0f, 0.0f, -1.0f);
    }
}

void toast(const char* text, float seconds = 2.5f) {
    std::strncpy(toastText, text, sizeof(toastText) - 1);
    toastText[sizeof(toastText) - 1] = '\0';
    toastTimer = seconds;
}










// ---------------------------------------------------------------------------
// Planet, atmosphere, stars, sun
// ---------------------------------------------------------------------------
void buildStars() {
    gListStars = glGenLists(1);
    glNewList(gListStars, GL_COMPILE);
    Rng rng(99u);
    const float R = 420.0f;
    glPointSize(1.0f);
    glBegin(GL_POINTS);
    for (int i = 0; i < 1800; ++i) {                               // background stars
        const Vec3 p = normalize(Vec3(rng.range(-1, 1), rng.range(-1, 1), rng.range(-1, 1)));
        const float b = 0.12f + 0.75f * rng.uniform() * rng.uniform();
        const float tint = rng.uniform();
        glColor3f(b * (0.85f + 0.15f * tint), b * 0.92f, b * (1.0f - 0.2f * tint));
        glVertex3f(p.x * R, p.y * R, p.z * R);
    }
    const Vec3 n = normalize(Vec3(0.3f, 0.8f, -0.5f));             // milky-way band
    const Vec3 u = normalize(cross(n, Vec3(0, 0, 1))), v = cross(n, u);
    for (int i = 0; i < 2600; ++i) {
        const float a = rng.range(0.0f, 2.0f * PI);
        const float spread = (rng.uniform() + rng.uniform() + rng.uniform() - 1.5f) * 0.22f;
        const Vec3 p = normalize(u * std::cos(a) + v * std::sin(a) + n * spread);
        const float b = 0.10f + 0.28f * rng.uniform();
        glColor3f(b * 0.85f, b * 0.9f, b);
        glVertex3f(p.x * R, p.y * R, p.z * R);
    }
    glEnd();
    glPointSize(2.2f);
    glBegin(GL_POINTS);
    for (int i = 0; i < 70; ++i) {                                 // a few bright stars
        const Vec3 p = normalize(Vec3(rng.range(-1, 1), rng.range(-1, 1), rng.range(-1, 1)));
        glColor3f(1.0f, 0.95f - 0.15f * rng.uniform(), 0.85f + 0.15f * rng.uniform());
        glVertex3f(p.x * R, p.y * R, p.z * R);
    }
    glEnd();
    glEndList();
}

void drawSky() {
    glDisable(GL_LIGHTING);
    glDepthMask(GL_FALSE);
    glEnable(GL_POINT_SMOOTH);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glPushMatrix();
    glTranslatef(camEye.x, camEye.y, camEye.z);                    // stars follow the camera: infinitely far away
    glCallList(gListStars);
    glTranslatef(SunDir.x * 400.0f, SunDir.y * 400.0f, SunDir.z * 400.0f);
    Vec3 right, up;
    viewAxes(right, up);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);
    const float radii[3] = {70.0f, 26.0f, 9.0f}, alpha[3] = {0.22f, 0.5f, 1.0f};
    for (int k = 0; k < 3; ++k) {                                  // sun glow: three radial gradients
        glBegin(GL_TRIANGLE_FAN);
        glColor4f(1.0f, 0.96f, 0.82f, alpha[k]);
        glVertex3f(0, 0, 0);
        glColor4f(1.0f, 0.75f, 0.35f, 0.0f);
        for (int i = 0; i <= 40; ++i) {
            const float a = 2.0f * PI * i / 40;
            const Vec3 p = right * (radii[k] * std::cos(a)) + up * (radii[k] * std::sin(a));
            glVertex3f(p.x, p.y, p.z);
        }
        glEnd();
    }
    glPopMatrix();
    glDisable(GL_POINT_SMOOTH);
    glDisable(GL_BLEND);
    glDepthMask(GL_TRUE);
    glEnable(GL_LIGHTING);
}

// ---------------------------------------------------------------------------
// Guides (key O): axis triad, docking corridor, flight path, satellite orbit
// ---------------------------------------------------------------------------
void drawText3D(const Vec3& p, const char* s, void* font = GLUT_BITMAP_HELVETICA_12) {
    glRasterPos3f(p.x, p.y, p.z);
    for (; *s; ++s) glutBitmapCharacter(font, *s);
}

void drawGuides() {
    glDisable(GL_LIGHTING);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_LINE_SMOOTH);
    glLineWidth(1.6f);
    // world axes at the station origin
    glBegin(GL_LINES);
    glColor3f(1.0f, 0.3f, 0.3f); glVertex3f(0, 0, 0); glVertex3f(7.5f, 0, 0);
    glColor3f(0.3f, 1.0f, 0.3f); glVertex3f(0, 0, 0); glVertex3f(0, 7.5f, 0);
    glColor3f(0.4f, 0.55f, 1.0f); glVertex3f(0, 0, 0); glVertex3f(0, 0, 7.5f);
    glEnd();
    glColor3f(1.0f, 0.5f, 0.5f); drawText3D(Vec3(7.9f, 0, 0), "X");
    glColor3f(0.5f, 1.0f, 0.5f); drawText3D(Vec3(0, 7.9f, 0), "Y");
    glColor3f(0.6f, 0.7f, 1.0f); drawText3D(Vec3(0, 0, 7.9f), "Z");
    // docking corridor and flight path (dashed)
    glEnable(GL_LINE_STIPPLE);
    glLineStipple(2, 0x00FF);
    glColor4f(1.0f, 0.75f, 0.25f, 0.8f);
    glBegin(GL_LINES); glVertex3f(0, 0, PortFaceZ); glVertex3f(0, 0, HoldZ); glEnd();
    glColor4f(0.45f, 0.9f, 1.0f, 0.65f);
    glBegin(GL_LINE_STRIP);
    for (int i = 0; i <= 60; ++i) { Vec3 p, f; shipPath(i / 60.0f, p, f); glVertex3f(p.x, p.y, p.z); }
    glEnd();
    // satellite orbit
    glColor4f(0.55f, 1.0f, 0.65f, 0.55f);
    glPushMatrix();
    glBegin(GL_LINE_LOOP);
    for (int i = 0; i < 120; ++i) {
        const float a = 2.0f * PI * i / 120;
    }
    glEnd();
    glPopMatrix();
    glDisable(GL_LINE_STIPPLE);
    // range ticks along the corridor
    glBegin(GL_LINES);
    glColor4f(1.0f, 0.75f, 0.25f, 0.9f);
    for (float z = PortFaceZ + 2.0f; z <= HoldZ; z += 2.0f) { glVertex3f(-0.3f, 0, z); glVertex3f(0.3f, 0, z); }
    glEnd();
    glColor3f(1.0f, 0.8f, 0.4f);
    drawText3D(Vec3(0.5f, 0.3f, HoldZ), "hold point");
    drawText3D(Vec3(0.5f, 0.3f, PortFaceZ), "port");
    glColor3f(0.6f, 1.0f, 0.7f);
    drawText3D(satPos + Vec3(0.0f, 1.3f, 0.0f), "satellite");
    glDisable(GL_LINE_SMOOTH);
    glLineWidth(1.0f);
    glDisable(GL_BLEND);
    glEnable(GL_LIGHTING);
}

// ---------------------------------------------------------------------------
// Whole-simulation update (fixed logic, called once per frame with the real dt)
// ---------------------------------------------------------------------------
void updateSimulation(float dt) {
    simTime += dt;
    ringAngle = std::fmod(ringAngle + 0.0f * dt, 360.0f);
    solarAngle = std::fmod(solarAngle + 0.0f * dt, 360.0f);
    earthSpin = std::fmod(earthSpin + 1.2f * dt, 360.0f);
    satAngle = std::fmod(satAngle + 9.0f * dt, 360.0f);

    // solar wings turn smoothly toward the sun when tracking is on
    const float sunTarget = degrees(std::atan2(-SunDir.y, SunDir.z));
    solarTrackAngle += wrap180(sunTarget - solarTrackAngle) * (1.0f - std::exp(-dt * 1.5f));
 


    // antenna gimbal follows the satellite (azimuth then elevation)
    const Vec3 d = normalize(satPos - DishPivot);
    const float yawT = degrees(std::atan2(d.x, d.z));
    const float pitchT = clampf(degrees(std::asin(clampf(d.y, -1.0f, 1.0f))), -12.0f, 90.0f);
    const float k = 1.0f - std::exp(-dt * 3.0f);
    dishYaw = wrap180(dishYaw + wrap180(yawT - dishYaw) * k);
    dishPitch += (pitchT - dishPitch) * k;
}

// ============================================================================
// 7. TRANSFORMATION LAB (key T): one probe mesh, every transformation type, live matrix read-out
// ============================================================================

const char* const LabNames[LabDemoCount] = {"TRANSLATION", "ROTATION", "SCALING", "REFLECTION", "SHEARING", "COMPOSITE"};
const char* const LabText[LabDemoCount][3] = {
    {"T(tx,ty,tz):  x' = x+tx,  y' = y+ty,  z' = z+tz", "Every vertex moves by the same vector; the shape",
     "is unchanged. Arrows / PgUp / PgDn move the probe."},
    {"Ry(a) * Rx(b) * Rz(c): vertices turn about the origin", "through sin/cos entries in the upper-left 3x3 block.",
     "Shift + arrows / PgUp / PgDn add your own angles."},
    {"S(sx,sy,sz): each coordinate is multiplied per axis", "(the diagonal). Non-uniform scale changes the",
     "proportions. Keys , and . change the base scale."},
    {"Mirror through the YZ plane:  x' = -x  = scale(-1,1,1).", "det(3x3) turns negative, so triangle winding flips and",
     "glFrontFace(GL_CW) keeps the lighting right. M toggles."},
    {"Shear XY:  x' = x + k*y.  Layers of constant height", "slide sideways: the tall tail fin leans over while",
     "the base stays put. H toggles the shear."},
    {"Matrices multiply right-to-left, so order matters.", "Orange:  M = T * R * S  spins about its own centre.",
     "Cyan:    M = R * T * S  orbits the world origin."}};

struct LabParams { Vec3 pos; float yaw, pitch, roll; Vec3 scale; float mirror, shear; };

void labEnterDemo(int d) {
    labSelected = ((d % LabDemoCount) + LabDemoCount) % LabDemoCount;
    labTime = 0.0f;
    probe.pos = Vec3(0, 0, 0);
    probe.yaw = probe.pitch = probe.roll = 0.0f;
    probe.scale = 0.9f;
    probe.mirror = probe.mirrorTarget = 1.0f;
    probe.shear = probe.shearTarget = 0.0f;
    switch (labSelected) {
    case 3: probe.pos = Vec3(3.0f, 0.6f, 0.0f); probe.yaw = 20.0f; probe.mirrorTarget = -1.0f; break;
    case 4: probe.shearTarget = 0.8f; break;
    case 5: probe.pos = Vec3(3.5f, 0.0f, 0.0f); break;
    default: break;
    }
}

// User-controlled pose plus the automatic animation each demo adds on top.
LabParams labEffective() {
    LabParams p;
    const float s = probe.scale, t = labTime;
    p.pos = probe.pos; p.yaw = probe.yaw; p.pitch = probe.pitch; p.roll = probe.roll;
    p.scale = Vec3(s, s, s); p.mirror = probe.mirror; p.shear = probe.shear;
    switch (labSelected) {
    case 0: p.pos = p.pos + Vec3(3.2f * std::sin(0.7f * t), 0.8f * std::sin(1.1f * t), 2.4f * std::cos(0.5f * t)); break;
    case 1: p.yaw += 50.0f * t; p.pitch += 14.0f * std::sin(0.6f * t); break;
    case 2: p.scale = Vec3(s * (1.0f + 0.55f * std::sin(t)), s * (1.0f + 0.55f * std::sin(1.3f * t + 1.0f)),
                           s * (1.0f + 0.55f * std::sin(0.8f * t + 2.0f))); break;
    case 5: p.yaw += 40.0f * t; break;
    default: break;
    }
    return p;
}

// variant 0:  Mirror * T * R * S * Shear      variant 1 (composite demo):  Mirror * R * T * S * Shear
Mat4 labModel(const LabParams& p, int variant, bool applyMirror) {
    const Mat4 T = Mat4::translation(p.pos.x, p.pos.y, p.pos.z);
    const Mat4 R = Mat4::rotationY(p.yaw) * Mat4::rotationX(p.pitch) * Mat4::rotationZ(p.roll);
    const Mat4 S = Mat4::scaling(p.scale.x, p.scale.y, p.scale.z);
    const Mat4 H = Mat4::shearXY(p.shear);
    const Mat4 Mi = applyMirror ? Mat4::scaling(p.mirror, 1.0f, 1.0f) : Mat4();
    return variant == 0 ? Mi * T * R * S * H : Mi * R * T * S * H;
}

void drawLabStage() {
    glDisable(GL_LIGHTING);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_LINE_SMOOTH);
    glLineWidth(1.0f);
    glBegin(GL_LINES);                                              // floor grid
    for (int i = -9; i <= 9; ++i) {
        const float a = (i % 5 == 0) ? 0.55f : 0.28f;
        glColor4f(0.35f, 0.5f, 0.72f, a);
        glVertex3f((float)i, -2.0f, -9.0f); glVertex3f((float)i, -2.0f, 9.0f);
        glVertex3f(-9.0f, -2.0f, (float)i); glVertex3f(9.0f, -2.0f, (float)i);
    }
    glEnd();
    glLineWidth(2.5f);                                              // world axes with arrow heads
    const float col[3][3] = {{1.0f, 0.3f, 0.3f}, {0.3f, 1.0f, 0.35f}, {0.4f, 0.55f, 1.0f}};
    glBegin(GL_LINES);
    for (int a = 0; a < 3; ++a) {
        glColor3f(col[a][0], col[a][1], col[a][2]);
        glVertex3f(0, 0, 0);
        glVertex3f(a == 0 ? 6.0f : 0.0f, a == 1 ? 6.0f : 0.0f, a == 2 ? 6.0f : 0.0f);
    }
    glEnd();
    for (int a = 0; a < 3; ++a) {
        glColor3f(col[a][0], col[a][1], col[a][2]);
        glPushMatrix();
        if (a == 0) glTranslatef(6.0f, 0, 0); else if (a == 1) glTranslatef(0, 6.0f, 0); else glTranslatef(0, 0, 6.0f);
        if (a == 0) glRotatef(90.0f, 0, 1, 0); else if (a == 1) glRotatef(-90.0f, 1, 0, 0);
        drawCone(0.16f, 0.0f, 0.5f, false, 14);
        glPopMatrix();
    }
    drawText3D(Vec3(6.7f, 0, 0), "X"); drawText3D(Vec3(0, 6.7f, 0), "Y"); drawText3D(Vec3(0, 0, 6.7f), "Z");
    glColor3f(0.8f, 0.85f, 0.95f);
    drawText3D(Vec3(0.15f, -0.35f, 0.15f), "origin");
    glLineWidth(1.0f);
    glDisable(GL_LINE_SMOOTH);
    glDisable(GL_BLEND);
    glEnable(GL_LIGHTING);
}

void drawMirrorPlane() {                                            // translucent, so drawn last and without depth writes
    glDisable(GL_LIGHTING);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDepthMask(GL_FALSE);
    glColor4f(0.4f, 0.85f, 1.0f, 0.13f);
    glBegin(GL_QUADS);
    glVertex3f(0, -2, -5); glVertex3f(0, 3.5f, -5); glVertex3f(0, 3.5f, 5); glVertex3f(0, -2, 5);
    glEnd();
    glDepthMask(GL_TRUE);
    glEnable(GL_LINE_SMOOTH);
    glLineWidth(1.5f);
    glColor4f(0.5f, 0.9f, 1.0f, 0.8f);
    glBegin(GL_LINE_LOOP);
    glVertex3f(0, -2, -5); glVertex3f(0, 3.5f, -5); glVertex3f(0, 3.5f, 5); glVertex3f(0, -2, 5);
    glEnd();
    drawText3D(Vec3(0.0f, 3.8f, -4.5f), "mirror plane (x = 0)");
    glLineWidth(1.0f);
    glDisable(GL_LINE_SMOOTH);
    glDisable(GL_BLEND);
    glEnable(GL_LIGHTING);
}

void drawWireGhost(const Mat4& M, float r, float g, float b) {
    MatrixScope s(M);
    glDisable(GL_LIGHTING);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    glColor4f(r, g, b, 0.55f);
    gMesh.probe.draw();
    glPolygonMode(GL_FRONT_AND_BACK, wireframe ? GL_LINE : GL_FILL);
    glDisable(GL_BLEND);
    glEnable(GL_LIGHTING);
}

void drawProbe(const Mat4& M, float r, float g, float b) {
    MatrixScope s(M);                                               // reflection => front-face flip handled here
    setMaterial(r, g, b, 50.0f, 0.5f);
    gMesh.probe.draw();
    glDisable(GL_LIGHTING);                                         // the probe's own axes go through the same matrix
    glLineWidth(2.5f);
    glBegin(GL_LINES);
    glColor3f(1.0f, 0.3f, 0.3f); glVertex3f(0, 0, 0); glVertex3f(2.9f, 0, 0);
    glColor3f(0.3f, 1.0f, 0.35f); glVertex3f(0, 0, 0); glVertex3f(0, 2.9f, 0);
    glColor3f(0.4f, 0.55f, 1.0f); glVertex3f(0, 0, 0); glVertex3f(0, 0, 2.9f);
    glEnd();
    glLineWidth(1.0f);
    glEnable(GL_LIGHTING);
}

void updateLab(float dt) {
    probe.mirror = moveToward(probe.mirror, probe.mirrorTarget, dt * 2.0f);   // reflection eases through the mirror plane
    probe.shear = moveToward(probe.shear, probe.shearTarget, dt * 1.6f);
}

Mat4 labMainMatrix() { return labModel(labEffective(), 0, true); }

void drawLab() {
    drawLabStage();
    const LabParams p = labEffective();
    if (labSelected == 3) drawWireGhost(labModel(p, 0, false), 0.6f, 0.85f, 1.0f);      // un-mirrored original
    else if (labSelected != 5) drawWireGhost(Mat4::scaling(probe.scale, probe.scale, probe.scale), 0.6f, 0.85f, 1.0f);
    if (labSelected == 0) {                                          // displacement vector
        glDisable(GL_LIGHTING);
        glColor3f(1.0f, 0.85f, 0.3f);
        glBegin(GL_LINES); glVertex3f(0, 0, 0); glVertex3f(p.pos.x, p.pos.y, p.pos.z); glEnd();
        glEnable(GL_LIGHTING);
    }
    drawProbe(labModel(p, 0, true), 0.95f, 0.55f, 0.18f);
    if (labSelected == 5) drawProbe(labModel(p, 1, true), 0.20f, 0.72f, 0.85f);
    if (labSelected == 3) drawMirrorPlane();
}

// ============================================================================
// 8. CAMERA (orbit / pan / zoom with smoothing, presets, chase camera)
// ============================================================================

Vec3 camTarget, camUp(0.0f, 1.0f, 0.0f);
Vec3 chaseEye, chaseTarget, chaseUp(0.0f, 1.0f, 0.0f);
CameraState labSavedCam = {40.0f, 22.0f, 34.0f, Vec3(0.0f, 0.0f, 1.5f)};
bool shiftDown = false;
bool screenshotRequested = false;
char screenshotPath[256] = "";
int screenshotAfterFrames = -1;
int frameCounter = 0;

const char* const PresetNames[7] = {"", "ISOMETRIC", "DOCKING AXIS", "TOP", "AFT", "CHASE (SPACECRAFT)", "ROBOTIC ARM"};

Vec3 orbitOffset(const CameraState& c) {
    const float cp = std::cos(radians(c.pitch)), sp = std::sin(radians(c.pitch));
    return Vec3(cp * std::sin(radians(c.yaw)), sp, cp * std::cos(radians(c.yaw))) * c.dist;
}

void clampCamera(CameraState& c) {
    c.pitch = clampf(c.pitch, -88.0f, 88.0f);
    c.dist = clampf(c.dist, labMode ? 4.0f : MinCamDist, MaxCamDist);
    const float l = length(c.target);
    if (l > 45.0f) c.target = c.target * (45.0f / l);
}

// Leaving the chase camera: continue smoothly from wherever the chase camera currently is.
void leaveChase() {
    if (!chaseCamera) return;
    chaseCamera = false;
    const Vec3 off = camEye - camTarget;
    const float d = clampf(length(off), MinCamDist, MaxCamDist);
    cam.target = camTarget;
    cam.dist = d;
    cam.yaw = degrees(std::atan2(off.x, off.z));
    cam.pitch = degrees(std::asin(clampf(off.y / (d + 1e-6f), -1.0f, 1.0f)));
    camGoal = cam;
}

void setPreset(int id) {
    if (id != 5) leaveChase();
    camPresetId = id;
    switch (id) {
    case 1: camGoal.yaw = 40.0f;  camGoal.pitch = 22.0f; camGoal.dist = 34.0f; camGoal.target = Vec3(0, 0, 1.5f); break;
    case 2: camGoal.yaw = 72.0f;  camGoal.pitch = 10.0f; camGoal.dist = 34.0f; camGoal.target = Vec3(0, 0, 12.0f); break;
    case 3: camGoal.yaw = 0.0f;   camGoal.pitch = 86.0f; camGoal.dist = 36.0f; camGoal.target = Vec3(0, 0, 2.0f); break;
    case 4: camGoal.yaw = 180.0f; camGoal.pitch = 10.0f; camGoal.dist = 34.0f; camGoal.target = Vec3(0, 0, 0); break;
    case 5:
        chaseCamera = true;
        chaseEye = camEye; chaseTarget = camTarget; chaseUp = Vec3(0, 1, 0);
        break;
    case 6: camGoal.yaw = 30.0f;  camGoal.pitch = 25.0f; camGoal.dist = 11.0f; camGoal.target = Vec3(0, 3.6f, 3.0f); break;
    default: break;
    }
    toast(PresetNames[id], 1.5f);
}

void updateCamera(float dt) {
    if (chaseCamera && !labMode) {
        const Vec3 goalEye = shipFrame.point(Vec3(0.0f, 4.2f, 14.0f));
        const Vec3 goalTarget = shipFrame.point(Vec3(0.0f, 0.0f, -14.0f));
        const Vec3 goalUp = Vec3(shipFrame.m[4], shipFrame.m[5], shipFrame.m[6]);
        const float k = 1.0f - std::exp(-dt * 3.5f);
        chaseEye = chaseEye + (goalEye - chaseEye) * k;
        chaseTarget = chaseTarget + (goalTarget - chaseTarget) * k;
        chaseUp = normalize(chaseUp + (goalUp - chaseUp) * k);
        camEye = chaseEye; camTarget = chaseTarget; camUp = chaseUp;
        return;
    }
    const float k = 1.0f - std::exp(-dt * 9.0f);
    clampCamera(camGoal);
    cam.yaw += wrap180(camGoal.yaw - cam.yaw) * k;
    cam.pitch += (camGoal.pitch - cam.pitch) * k;
    cam.dist += (camGoal.dist - cam.dist) * k;
    cam.target = cam.target + (camGoal.target - cam.target) * k;
    camTarget = cam.target;
    camEye = cam.target + orbitOffset(cam);
    camUp = Vec3(0.0f, 1.0f, 0.0f);
}

void enterLab(bool on) {
    if (on == labMode) return;
    if (on) {
        leaveChase();
        labSavedCam = camGoal;
        labMode = true;
        if (probe.scale <= 0.0f) labEnterDemo(0);
        camGoal.yaw = 35.0f; camGoal.pitch = 24.0f; camGoal.dist = 13.0f; camGoal.target = Vec3(0, 0.3f, 0);
        toast("Transformation lab (T returns to the station)");
    } else {
        labMode = false;
        camGoal = labSavedCam;
        toast("Back to the station");
    }
}

// ============================================================================
// 9. HEADS-UP DISPLAY
// ============================================================================

void hudBegin() {
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_LIGHTING);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glMatrixMode(GL_PROJECTION); glPushMatrix(); glLoadIdentity();
    gluOrtho2D(0.0, winW, winH, 0.0);                                // origin top-left, y grows downward
    glMatrixMode(GL_MODELVIEW); glPushMatrix(); glLoadIdentity();
}
void hudEnd() {
    glMatrixMode(GL_PROJECTION); glPopMatrix();
    glMatrixMode(GL_MODELVIEW); glPopMatrix();
    glDisable(GL_BLEND);
    glEnable(GL_LIGHTING);
    glEnable(GL_DEPTH_TEST);
}
void hudText(float x, float y, void* font, const char* fmt, ...) {
    char buf[256];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    glRasterPos2f(x, y);
    for (const char* c = buf; *c; ++c) glutBitmapCharacter(font, *c);
}
void hudPanel(float x, float y, float w, float h, float alpha = 0.62f) {
    glColor4f(0.02f, 0.05f, 0.10f, alpha);
    glBegin(GL_QUADS); glVertex2f(x, y); glVertex2f(x + w, y); glVertex2f(x + w, y + h); glVertex2f(x, y + h); glEnd();
    glColor4f(0.35f, 0.55f, 0.80f, 0.55f);
    glBegin(GL_LINE_LOOP); glVertex2f(x, y); glVertex2f(x + w, y); glVertex2f(x + w, y + h); glVertex2f(x, y + h); glEnd();
}

void drawHelpPanel(float y, const char* const* lines, int n) {
    if (!showHelp) return;
    hudPanel(12.0f, y, 440.0f, 26.0f + 15.0f * n);
    glColor3f(0.55f, 0.85f, 1.0f);
    hudText(24.0f, y + 18.0f, GLUT_BITMAP_HELVETICA_12, "CONTROLS   (F1 hides this panel)");
    glColor3f(0.88f, 0.92f, 1.0f);
    for (int i = 0; i < n; ++i) hudText(24.0f, y + 36.0f + 15.0f * i, GLUT_BITMAP_8_BY_13, "%s", lines[i]);
}

void drawStationHUD() {
    void* big = GLUT_BITMAP_HELVETICA_18;
    void* med = GLUT_BITMAP_HELVETICA_12;
    void* mono = GLUT_BITMAP_8_BY_13;

    hudPanel(12.0f, 12.0f, 440.0f, 100.0f);
    glColor3f(0.55f, 0.85f, 1.0f);  hudText(24.0f, 34.0f, big, "SPACE STATION SIMULATOR");
    glColor3f(0.9f, 0.95f, 1.0f);
    hudText(24.0f, 90.0f, med, "View: %s    FPS %.0f%s", chaseCamera ? PresetNames[5] : (camPresetId ? PresetNames[camPresetId] : "FREE"),
            fpsValue, paused ? "    [PAUSED]" : "");
    

    static const char* const help[] = {
        "Mouse  L-drag orbit  R-drag/wheel zoom  M-drag pan",
        "Camera 1 iso  2 dock  3 top  4 aft  5 chase  6 arm",
        "       arrows orbit   +/- zoom   R reset",
        "Sim    Space pause  N dock/undock  B reset",
        "       P solar wings: free spin / track sun",
        "Arm    A/D base   W/S shoulder   Q/E elbow",
        "       F/G wrist  C/V gripper    Z reset arm",
        "View   T transformation lab  L wireframe",
        "       O guides  F11 fullscreen  F12 screenshot"};
    drawHelpPanel(126.0f, help, 9);

    // telemetry (right)
    const float px = winW - 282.0f;
    hudPanel(px, 12.0f, 270.0f, 214.0f);
    glColor3f(0.55f, 0.85f, 1.0f);  hudText(px + 12.0f, 32.0f, med, "TELEMETRY");
    glColor3f(0.88f, 0.92f, 1.0f);
    float nearest = 1e9f;
    for (size_t i = 0; i < debris.size(); ++i) nearest = std::min(nearest, length(debris[i].pos));
    const Vec3 tip = armToolPosition();
    float y = 52.0f;
    hudText(px + 12.0f, y, mono, "Sim time   %7.1f s", simTime);                       y += 15.0f;
    hudText(px + 12.0f, y, mono, "Solar      %s", solarTracking ? "TRACKING SUN" : "FREE SPIN");  y += 15.0f;
    hudText(px + 12.0f, y, mono, "Ring spin  6.0 deg/s");                                y += 15.0f;
    hudText(px + 12.0f, y, mono, "Antenna    az %4.0f  el %3.0f", dishYaw, dishPitch);   y += 15.0f;
    hudText(px + 12.0f, y, mono, "Satellite  phase %3.0f deg", satAngle);               y += 15.0f;
    hudText(px + 12.0f, y, mono, "Debris     %d objs, min %.1f", (int)debris.size(), nearest);  y += 22.0f;
    glColor3f(1.0f, 0.75f, 0.35f);
    hudText(px + 12.0f, y, mono, "ROBOTIC ARM (deg)");                                  y += 15.0f;
    glColor3f(0.88f, 0.92f, 1.0f);
    hudText(px + 12.0f, y, mono, " base %4.0f  shoulder %4.0f", arm.base, arm.shoulder); y += 15.0f;
    hudText(px + 12.0f, y, mono, " elbow %4.0f  wrist %4.0f", arm.elbow, arm.wrist);     y += 15.0f;
    hudText(px + 12.0f, y, mono, " grip %3.0f%%", arm.grip * 100.0f);                    y += 15.0f;
    hudText(px + 12.0f, y, mono, " tip (%.1f, %.1f, %.1f)", tip.x, tip.y, tip.z);
}

void drawLabHUD() {
    void* big = GLUT_BITMAP_HELVETICA_18;
    void* med = GLUT_BITMAP_HELVETICA_12;
    void* mono = GLUT_BITMAP_9_BY_15;

    hudPanel(12.0f, 12.0f, 440.0f, 62.0f);
    glColor3f(0.55f, 0.85f, 1.0f);  hudText(24.0f, 34.0f, big, "TRANSFORMATION LAB");
    glColor3f(1.0f, 0.82f, 0.35f);  hudText(24.0f, 54.0f, med, "Demo %d / %d :  %s", labSelected + 1, LabDemoCount, LabNames[labSelected]);
    glColor3f(0.9f, 0.95f, 1.0f);   hudText(24.0f, 68.0f, med, "FPS %.0f%s", fpsValue, paused ? "   [PAUSED]" : "");

    static const char* const help[] = {
        "Demo   1-6 or [ ] choose     T back to station",
        "Move   arrows = X/Z     PgUp/PgDn = Y",
        "Rotate Shift + arrows / PgUp / PgDn",
        "Scale  ,  shrink      .  grow",
        "Effect M mirror   H shear   Z reset demo",
        "Mouse  L-drag orbit  wheel zoom  M-drag pan",
        "View   Space pause  L wire  F12 screenshot"};
    drawHelpPanel(88.0f, help, 7);

    const float pw = 400.0f, px = winW - pw - 12.0f;
    hudPanel(px, 12.0f, pw, 232.0f);
    glColor3f(1.0f, 0.82f, 0.35f);  hudText(px + 12.0f, 32.0f, med, "%s", LabNames[labSelected]);
    glColor3f(0.88f, 0.92f, 1.0f);
    for (int i = 0; i < 3; ++i) hudText(px + 12.0f, 52.0f + 16.0f * i, med, "%s", LabText[labSelected][i]);
    const Mat4 M = labMainMatrix();
    glColor3f(0.55f, 0.85f, 1.0f);  hudText(px + 12.0f, 118.0f, med, "M (orange probe), points as column vectors: p' = M p");
    glColor3f(0.95f, 0.97f, 1.0f);
    for (int r = 0; r < 4; ++r)
        hudText(px + 20.0f, 140.0f + 19.0f * r, mono, "[%7.3f %7.3f %7.3f %7.3f]", M(r, 0), M(r, 1), M(r, 2), M(r, 3));
    const float det = M.det3();
    if (det < -1e-4f) glColor3f(1.0f, 0.6f, 0.4f); else glColor3f(0.7f, 1.0f, 0.75f);
    hudText(px + 12.0f, 232.0f, med, "det(3x3) = %.3f   %s", det, det < -1e-4f ? "(negative: reflection, winding flipped)" : "(positive: orientation kept)");
}

void drawHUD() {
    hudBegin();
    if (labMode) drawLabHUD(); else drawStationHUD();
    if (!showHelp) {
        glColor3f(0.7f, 0.8f, 0.95f);
        hudText(14.0f, winH - 12.0f, GLUT_BITMAP_HELVETICA_12, "F1: help");
    }
    if (toastTimer > 0.0f) {
        const float a = clampf(toastTimer / 0.6f, 0.0f, 1.0f);
        const int w = glutBitmapLength(GLUT_BITMAP_HELVETICA_18, (const unsigned char*)toastText);
        const float x = (winW - w) * 0.5f, y = winH - 46.0f;
        hudPanel(x - 16.0f, y - 22.0f, w + 32.0f, 34.0f, 0.7f * a);
        glColor4f(1.0f, 0.95f, 0.8f, a);
        hudText(x, y, GLUT_BITMAP_HELVETICA_18, "%s", toastText);
    }
    hudEnd();
}

// ============================================================================
// 10. RENDERING, SCREENSHOT, UPDATE LOOP
// ============================================================================

void writeScreenshot(const char* path) {
    std::vector<unsigned char> px((size_t)winW * winH * 3);
    glReadBuffer(GL_BACK);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(0, 0, winW, winH, GL_RGB, GL_UNSIGNED_BYTE, &px[0]);
    FILE* f = std::fopen(path, "wb");
    if (!f) { toast("Could not write the screenshot file"); return; }
    const int rowBytes = (winW * 3 + 3) & ~3;
    const unsigned dataSize = (unsigned)rowBytes * winH, fileSize = 54u + dataSize;
    unsigned char hdr[54];
    std::memset(hdr, 0, sizeof(hdr));
    hdr[0] = 'B'; hdr[1] = 'M';
    std::memcpy(hdr + 2, &fileSize, 4);
    hdr[10] = 54; hdr[14] = 40;
    std::memcpy(hdr + 18, &winW, 4); std::memcpy(hdr + 22, &winH, 4);
    hdr[26] = 1; hdr[28] = 24;
    std::memcpy(hdr + 34, &dataSize, 4);
    std::fwrite(hdr, 1, 54, f);
    std::vector<unsigned char> row(rowBytes, 0);
    for (int y = 0; y < winH; ++y) {                                 // OpenGL rows are already bottom-up, like BMP
        for (int x = 0; x < winW; ++x) {
            row[x * 3 + 0] = px[((size_t)y * winW + x) * 3 + 2];
            row[x * 3 + 1] = px[((size_t)y * winW + x) * 3 + 1];
            row[x * 3 + 2] = px[((size_t)y * winW + x) * 3 + 0];
        }
        std::fwrite(&row[0], 1, rowBytes, f);
    }
    std::fclose(f);
}

void display() {
    glClearColor(0.0f, 0.0f, 0.015f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    gluLookAt(camEye.x, camEye.y, camEye.z, camTarget.x, camTarget.y, camTarget.z, camUp.x, camUp.y, camUp.z);

    const GLfloat sunPos[4] = {SunDir.x, SunDir.y, SunDir.z, 0.0f};
    const Vec3 ed = normalize(EarthPos);
    const GLfloat earthshine[4] = {ed.x, ed.y, ed.z, 0.0f};
    glLightfv(GL_LIGHT0, GL_POSITION, sunPos);                       // set after the view matrix: lights live in world space
    glLightfv(GL_LIGHT1, GL_POSITION, earthshine);

    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    // drawSky();
    glPolygonMode(GL_FRONT_AND_BACK, wireframe ? GL_LINE : GL_FILL);
    if (labMode) {
        drawLab();
    } else {
        drawSpaceStation();
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        if (showGuides) drawGuides();
    }
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    drawHUD();

    ++frameCounter;
    if (screenshotRequested || (screenshotAfterFrames > 0 && frameCounter >= screenshotAfterFrames)) {
        char name[300];
        static int shotNo = 0;
        if (screenshotPath[0]) std::snprintf(name, sizeof(name), "%s", screenshotPath);
        else std::snprintf(name, sizeof(name), "space_station_shot_%03d.bmp", ++shotNo);
        writeScreenshot(name);
        if (!screenshotRequested) std::exit(0);                      // scripted single-shot run
        screenshotRequested = false;
        char msg[320];
        std::snprintf(msg, sizeof(msg), "Saved %s", name);
        toast(msg);
    }
    glutSwapBuffers();
}

void reshape(int w, int h) {
    winW = w > 1 ? w : 1;
    winH = h > 1 ? h : 1;
    glViewport(0, 0, winW, winH);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(45.0, (double)winW / winH, 0.2, 900.0);
    glMatrixMode(GL_MODELVIEW);
}

void processHeldInput(float dt) {
    if (labMode) {
        const float mv = 4.0f * dt, rot = 90.0f * dt;
        const bool L = specialHeld[GLUT_KEY_LEFT], R = specialHeld[GLUT_KEY_RIGHT];
        const bool U = specialHeld[GLUT_KEY_UP], D = specialHeld[GLUT_KEY_DOWN];
        const bool PU = specialHeld[GLUT_KEY_PAGE_UP], PD = specialHeld[GLUT_KEY_PAGE_DOWN];
        if (!shiftDown) {
            if (L) probe.pos.x -= mv;
            if (R) probe.pos.x += mv;
            if (U) probe.pos.z -= mv;
            if (D) probe.pos.z += mv;
            if (PU) probe.pos.y += mv;
            if (PD) probe.pos.y -= mv;
            probe.pos.x = clampf(probe.pos.x, -7.0f, 7.0f);
            probe.pos.y = clampf(probe.pos.y, -1.5f, 5.0f);
            probe.pos.z = clampf(probe.pos.z, -7.0f, 7.0f);
        } else {
            if (L) probe.yaw += rot;
            if (R) probe.yaw -= rot;
            if (U) probe.pitch += rot;
            if (D) probe.pitch -= rot;
            if (PU) probe.roll += rot;
            if (PD) probe.roll -= rot;
        }
        if (keyHeld[(unsigned char)'.']) probe.scale *= std::exp(dt * 0.9f);
        if (keyHeld[(unsigned char)',']) probe.scale *= std::exp(-dt * 0.9f);
        probe.scale = clampf(probe.scale, 0.3f, 2.5f);
    } else {
        if (!chaseCamera) {
            if (specialHeld[GLUT_KEY_LEFT])  camGoal.yaw += 70.0f * dt;
            if (specialHeld[GLUT_KEY_RIGHT]) camGoal.yaw -= 70.0f * dt;
            if (specialHeld[GLUT_KEY_UP])    camGoal.pitch += 50.0f * dt;
            if (specialHeld[GLUT_KEY_DOWN])  camGoal.pitch -= 50.0f * dt;
        }
        // arm: each key drives one joint of the chain
        const float v = 60.0f * dt;
        if (keyHeld[(unsigned char)'a']) arm.base += v;
        if (keyHeld[(unsigned char)'d']) arm.base -= v;
        if (keyHeld[(unsigned char)'w']) arm.shoulder += v * 0.7f;
        if (keyHeld[(unsigned char)'s']) arm.shoulder -= v * 0.7f;
        if (keyHeld[(unsigned char)'q']) arm.elbow += v * 0.8f;
        if (keyHeld[(unsigned char)'e']) arm.elbow -= v * 0.8f;
        if (keyHeld[(unsigned char)'f']) arm.wrist += v * 0.8f;
        if (keyHeld[(unsigned char)'g']) arm.wrist -= v * 0.8f;
        if (keyHeld[(unsigned char)'c']) arm.grip += 0.8f * dt;
        if (keyHeld[(unsigned char)'v']) arm.grip -= 0.8f * dt;
        arm.base = wrap180(arm.base);
        arm.shoulder = clampf(arm.shoulder, -100.0f, 100.0f);
        arm.elbow = clampf(arm.elbow, -20.0f, 150.0f);
        arm.wrist = clampf(arm.wrist, -100.0f, 100.0f);
        arm.grip = clampf(arm.grip, 0.0f, 1.0f);
    }
    // zoom keys work in both modes
    if (keyHeld[(unsigned char)'+'] || keyHeld[(unsigned char)'=']) camGoal.dist *= std::exp(-dt * 1.3f);
    if (keyHeld[(unsigned char)'-'] || keyHeld[(unsigned char)'_']) camGoal.dist *= std::exp(dt * 1.3f);
}

int lastTick = 0;

void tick(int) {
    const int now = glutGet(GLUT_ELAPSED_TIME);
    const float rawDt = (now - lastTick) / 1000.0f;
    lastTick = now;
    const float dt = clampf(rawDt, 0.0f, 0.05f);
    if (rawDt > 1e-4f) fpsValue = fpsValue <= 0.0f ? 1.0f / rawDt : lerpf(fpsValue, 1.0f / rawDt, 0.05f);

    processHeldInput(dt);
    if (!paused) {
        updateSimulation(dt);
        labTime += dt;
    }
    updateLab(dt);
    updateCamera(dt);
    if (toastTimer > 0.0f) toastTimer -= dt;
    glutPostRedisplay();
    glutTimerFunc(8, tick, 0);
}

// ============================================================================
// 11. INPUT
// ============================================================================

void toggleFullscreen() {
    static bool full = false;
    full = !full;
    if (full) glutFullScreen();
    else { glutReshapeWindow(InitialWidth, InitialHeight); glutPositionWindow(60, 40); }
}

void keyboard(unsigned char key, int, int) {
    shiftDown = (glutGetModifiers() & GLUT_ACTIVE_SHIFT) != 0;
    const unsigned char k = (key >= 'A' && key <= 'Z') ? (unsigned char)(key - 'A' + 'a') : key;
    keyHeld[k] = true;
    switch (k) {
    case 27: std::exit(0);
    case ' ': paused = !paused; toast(paused ? "Simulation paused" : "Simulation running", 1.2f); return;
    case 't': enterLab(!labMode); return;
    case 'l': wireframe = !wireframe; toast(wireframe ? "Wireframe on" : "Wireframe off", 1.2f); return;
    case 'o': showGuides = !showGuides; toast(showGuides ? "Guides on" : "Guides off", 1.2f); return;
    default: break;
    }
    if (labMode) {
        if (k >= '1' && k <= '6') labEnterDemo(k - '1');
        else if (k == '[') labEnterDemo(labSelected - 1);
        else if (k == ']') labEnterDemo(labSelected + 1);
        else if (k == 'm') { probe.mirrorTarget = -probe.mirrorTarget; }
        else if (k == 'h') { probe.shearTarget = probe.shearTarget > 0.1f ? 0.0f : 0.8f; }
        else if (k == 'z') labEnterDemo(labSelected);
        else if (k == 'r') { camGoal.yaw = 35.0f; camGoal.pitch = 24.0f; camGoal.dist = 13.0f; camGoal.target = Vec3(0, 0.3f, 0); }
        return;
    }
    if (k >= '1' && k <= '6') setPreset(k - '0');
    else if (k == 'r') setPreset(1);
    else if (k == 'p') { solarTracking = !solarTracking; toast(solarTracking ? "Solar wings track the sun" : "Solar wings spinning freely", 1.8f); }
    else if (k == 'z') { arm = ArmHome; toast("Robotic arm reset", 1.2f); }
}

void keyboardUp(unsigned char key, int, int) {
    shiftDown = (glutGetModifiers() & GLUT_ACTIVE_SHIFT) != 0;
    const unsigned char k = (key >= 'A' && key <= 'Z') ? (unsigned char)(key - 'A' + 'a') : key;
    keyHeld[k] = false;
    // Shift changes the character that arrives for the same physical key: release both spellings.
    if (key == '<') keyHeld[(unsigned char)','] = false;
    if (key == '>') keyHeld[(unsigned char)'.'] = false;
    if (key == '+') keyHeld[(unsigned char)'='] = false;
    if (key == '=') keyHeld[(unsigned char)'+'] = false;
}

void special(int key, int, int) {
    shiftDown = (glutGetModifiers() & GLUT_ACTIVE_SHIFT) != 0;
    if (key >= 0 && key < 256) specialHeld[key] = true;
    if (key == GLUT_KEY_F1) showHelp = !showHelp;
    else if (key == GLUT_KEY_F11) toggleFullscreen();
    else if (key == GLUT_KEY_F12) screenshotRequested = true;
}

void specialUp(int key, int, int) {
    shiftDown = (glutGetModifiers() & GLUT_ACTIVE_SHIFT) != 0;
    if (key >= 0 && key < 256) specialHeld[key] = false;
}

void mouse(int button, int state, int x, int y) {
    shiftDown = (glutGetModifiers() & GLUT_ACTIVE_SHIFT) != 0;
    if (state == GLUT_DOWN) { dragButton = button; lastMouseX = x; lastMouseY = y; }
    else if (button == dragButton) dragButton = -1;
}

void motion(int x, int y) {
    const float dx = (float)(x - lastMouseX), dy = (float)(y - lastMouseY);
    lastMouseX = x; lastMouseY = y;
    if (dragButton < 0) return;
    leaveChase();
    camPresetId = 0;
    if (dragButton == GLUT_LEFT_BUTTON) {
        camGoal.yaw -= dx * 0.4f;
        camGoal.pitch += dy * 0.4f;
    } else if (dragButton == GLUT_RIGHT_BUTTON) {
        camGoal.dist *= std::exp(dy * 0.01f);
    } else if (dragButton == GLUT_MIDDLE_BUTTON) {
        const Vec3 fwd = normalize(cam.target - camEye);
        const Vec3 right = normalize(cross(fwd, Vec3(0, 1, 0)));
        const Vec3 up = cross(right, fwd);
        const float s = cam.dist * 0.0016f;
        camGoal.target = camGoal.target - right * (dx * s) + up * (dy * s);
    }
    clampCamera(camGoal);
}

void mouseWheel(int, int dir, int, int) {
    leaveChase();
    camPresetId = 0;
    camGoal.dist *= std::pow(0.9f, (float)dir);
    clampCamera(camGoal);
}

// ============================================================================
// 12. INITIALISATION AND MAIN
// ============================================================================

void initGL(bool multisample) {
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
    glEnable(GL_LIGHT1);
    glEnable(GL_NORMALIZE);                                          // needed because of glScalef / Mat4 scaling
    glShadeModel(GL_SMOOTH);
    if (multisample) glEnable(GL_MULTISAMPLE);
    glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);

    const GLfloat sunDiffuse[]  = {1.00f, 0.96f, 0.88f, 1.0f};
    const GLfloat sunSpecular[] = {1.00f, 1.00f, 1.00f, 1.0f};
    const GLfloat none[]        = {0.0f, 0.0f, 0.0f, 1.0f};
    glLightfv(GL_LIGHT0, GL_DIFFUSE, sunDiffuse);
    glLightfv(GL_LIGHT0, GL_SPECULAR, sunSpecular);
    glLightfv(GL_LIGHT0, GL_AMBIENT, none);
    const GLfloat shine[] = {0.10f, 0.18f, 0.34f, 1.0f};              // faint blue light reflected from the planet
    glLightfv(GL_LIGHT1, GL_DIFFUSE, shine);
    glLightfv(GL_LIGHT1, GL_SPECULAR, none);
    glLightfv(GL_LIGHT1, GL_AMBIENT, none);
    const GLfloat globalAmbient[] = {0.12f, 0.13f, 0.17f, 1.0f};
    glLightModelfv(GL_LIGHT_MODEL_AMBIENT, globalAmbient);
    glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE);
    glLightModeli(GL_LIGHT_MODEL_LOCAL_VIEWER, GL_TRUE);
    glLightModeli(GL_LIGHT_MODEL_COLOR_CONTROL, GL_SEPARATE_SPECULAR_COLOR);
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);

    gQuadric = gluNewQuadric();
    gluQuadricNormals(gQuadric, GLU_SMOOTH);
    gluQuadricTexture(gQuadric, GL_TRUE);

    gTexHull = makeHullTexture();
    gTexSolar = makeSolarTexture();
    gTexEarth = makeEarthTexture();
    buildMeshes();
    buildStars();

    gListTruss = glGenLists(1);
    glNewList(gListTruss, GL_COMPILE); drawTrussStatic(); glEndList();
    gListRing = glGenLists(1);
    glNewList(gListRing, GL_COMPILE); drawHabitatRingStatic(); glEndList();

    probe.scale = 0.9f;
    labEnterDemo(0);
    solarTrackAngle = degrees(std::atan2(-SunDir.y, SunDir.z));
    cam = camGoal;
    camTarget = cam.target;
    camEye = cam.target + orbitOffset(cam);
}

}  // namespace

int main(int argc, char** argv) {
    glutInit(&argc, argv);
    unsigned mode = GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH | GLUT_MULTISAMPLE;
    glutInitDisplayMode(mode);
    bool multisample = glutGet(GLUT_DISPLAY_MODE_POSSIBLE) != 0;
    if (!multisample) { mode &= ~(unsigned)GLUT_MULTISAMPLE; glutInitDisplayMode(mode); }
    glutInitWindowSize(InitialWidth, InitialHeight);
    glutInitWindowPosition(60, 40);
    glutCreateWindow("Interactive Space Station Simulator - OpenGL");
    glutIgnoreKeyRepeat(1);

    initGL(multisample);

    // Optional command-line switches (handy for demos and for automated screenshots):
    //   --preset N   --lab N   --dock   --skip SECONDS   --shot FRAMES FILE.bmp
    float skipSeconds = 0.0f;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--preset") && i + 1 < argc) { camPresetId = std::atoi(argv[++i]); if (camPresetId >= 1 && camPresetId <= 6) setPreset(camPresetId); }
        else if (!std::strcmp(argv[i], "--lab") && i + 1 < argc) { enterLab(true); labEnterDemo(std::atoi(argv[++i]) - 1); }
        else if (!std::strcmp(argv[i], "--skip") && i + 1 < argc) skipSeconds = (float)std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "--shot") && i + 2 < argc) { screenshotAfterFrames = std::atoi(argv[++i]); std::snprintf(screenshotPath, sizeof(screenshotPath), "%s", argv[++i]); }
        else if (!std::strcmp(argv[i], "--nohelp")) showHelp = false;
    }
    for (float t = 0.0f; t < skipSeconds; t += 1.0f / 60.0f) { updateSimulation(1.0f / 60.0f); updateLab(1.0f / 60.0f); labTime += 1.0f / 60.0f; }
    if (skipSeconds > 0.0f) {
        for (int i = 0; i < 400; ++i) updateCamera(0.05f);
    }
    toastTimer = 0.0f;

    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutKeyboardFunc(keyboard);
    glutKeyboardUpFunc(keyboardUp);
    glutSpecialFunc(special);
    glutSpecialUpFunc(specialUp);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);
    glutMouseWheelFunc(mouseWheel);
    lastTick = glutGet(GLUT_ELAPSED_TIME);
    glutTimerFunc(8, tick, 0);
    glutMainLoop();
    return 0;
}
