# Interactive Space Station Simulator

Stage 4 is a small C++ OpenGL and FreeGLUT scene. It creates a perspective 3D window, basic lighting, a dark space background, and a station assembled from cubes, cylinders, spheres, docking ports, solar panels, and a hierarchical robotic arm.

## Prerequisites

Use the **MSYS2 UCRT64** terminal. Required packages:

```bash
pacman -S --needed mingw-w64-ucrt-x86_64-toolchain mingw-w64-ucrt-x86_64-freeglut mingw-w64-ucrt-x86_64-cmake
```

The project expects these tools to resolve from `/ucrt64/bin`:

```bash
which g++
which cmake
```

## Build

From the project root:

```bash
cmake -S . -B build -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Debug
cmake --build build
```

## Run

```bash
./build/space_station.exe
```

If Windows reports a missing DLL, run the executable from the MSYS2 UCRT64 terminal or ensure `C:\msys64\ucrt64\bin` is available on the process PATH.

## Stage 1 controls

- `Esc`: close the application
- `R`: reset the camera
- `+` or `=`: move closer
- `-` or `_`: move farther
- `T`: show or hide transformation examples
- `A` / `D`: rotate the robotic-arm base
- `W` / `S`: raise or lower the shoulder joint
- `Q` / `E`: bend or extend the elbow joint
- `Z`: reset robotic-arm joints

When enabled, the transformation examples demonstrate scaling, reflection, shearing through `glMultMatrixf`, and composite translation/rotation/scaling transformations.

The robotic arm uses nested `glPushMatrix()` / `glPopMatrix()` blocks, so shoulder and elbow transformations affect all child segments and the end-effector.

Stage 5 features such as continuous animation, spacecraft docking, satellite orbiting, and moving debris are intentionally not implemented yet.
