#define _USE_MATH_DEFINES
#include <GL/glut.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <string>
#include <cmath>
#include <algorithm>
#include "line_surface_intersection.cuh"

using namespace std;

// 全局变量
vector<Line> lines;
vector<Point3D> intersections;
vector<int> validFlags;
vector<Line> visibleLines; // 与 lines 对齐（相同大小），只有 validFlags[i]==1 时该条线段有效

// 相机控制变量
float cameraAngleX = 30.0f;
float cameraAngleY = 45.0f;
float cameraDistance = 25.0f;
int mouseX = 0, mouseY = 0;
bool mouseLeftDown = false;
int windowWidth = 800;
int windowHeight = 600;

// 圆柱体参数
const float CYLINDER_RADIUS = 3.0f;
const float CYLINDER_HEIGHT = 10.0f;

// 平面参数（改为 10×10 平面，每条直线间距 1）
const float PLANE_SIZE = 10.0f;
const float GRID_SPACING = 1.0f;

// 检查点是否在圆柱体内部
bool isPointInsideCylinder(const Point3D& p) {
    bool insideRadius = (p.x * p.x + p.y * p.y) <= (CYLINDER_RADIUS * CYLINDER_RADIUS);
    bool insideHeight = (p.z >= -CYLINDER_HEIGHT / 2) && (p.z <= CYLINDER_HEIGHT / 2);
    return insideRadius && insideHeight;
}

// 解析方法：根据轴向（Z / Y / X）计算交段并填充 intersections、validFlags、visibleLines（与 lines 同长度）
void computeAnalyticIntersectionsAndSegments() {
    int gridPoints = static_cast<int>(PLANE_SIZE / GRID_SPACING) + 1;
    int linesPerPlane = gridPoints * gridPoints;
    int total = lines.size();

    intersections.assign(total, Point3D(0,0,0));
    validFlags.assign(total, 0);
    visibleLines.assign(total, Line{Point3D(0,0,0), Point3D(0,0,0)});

    for (int i = 0; i < total; ++i) {
        const Line& L = lines[i];

        if (i < linesPerPlane) {
            // Z 方向线：x,y 常量，判断是否落在圆柱投影内
            float x = L.origin.x;
            float y = L.origin.y;
            if (x*x + y*y <= CYLINDER_RADIUS*CYLINDER_RADIUS) {
                // 可见段为 z ∈ [-h/2, h/2]
                Line seg;
                seg.origin = Point3D(x, y, -CYLINDER_HEIGHT/2.0f);
                seg.direction = Point3D(0.0f, 0.0f, CYLINDER_HEIGHT);
                visibleLines[i] = seg;
                intersections[i] = Point3D(x, y, 0.0f); // 代表交点（选择中点）
                validFlags[i] = 1;
            }
        }
        else if (i < 2 * linesPerPlane) {
            // Y 方向线：x,z 常量，判断 z 在高度范围且 |x| <= radius
            float x = L.origin.x;
            float z = L.origin.z;
            if (fabs(z) <= CYLINDER_HEIGHT/2.0f && fabs(x) <= CYLINDER_RADIUS) {
                float y_range = sqrtf(max(0.0f, CYLINDER_RADIUS*CYLINDER_RADIUS - x*x));
                Line seg;
                seg.origin = Point3D(x, -y_range, z);
                seg.direction = Point3D(0.0f, 2.0f*y_range, 0.0f);
                visibleLines[i] = seg;
                intersections[i] = Point3D(x, 0.0f, z);
                validFlags[i] = 1;
            }
        }
        else {
            // X 方向线：y,z 常量，判断 z 在高度范围且 |y| <= radius
            float y = L.origin.y;
            float z = L.origin.z;
            if (fabs(z) <= CYLINDER_HEIGHT/2.0f && fabs(y) <= CYLINDER_RADIUS) {
                float x_range = sqrtf(max(0.0f, CYLINDER_RADIUS*CYLINDER_RADIUS - y*y));
                Line seg;
                seg.origin = Point3D(-x_range, y, z);
                seg.direction = Point3D(2.0f*x_range, 0.0f, 0.0f);
                visibleLines[i] = seg;
                intersections[i] = Point3D(0.0f, y, z);
                validFlags[i] = 1;
            }
        }
    }
}

// 生成等间距网格直线（每组 gridPoints × gridPoints，顺序：先 Z 方向，再 Y，再 X）
vector<Line> generateGridLines() {
    vector<Line> lines;

    float halfSize = PLANE_SIZE / 2.0f;
    int gridPoints = static_cast<int>(PLANE_SIZE / GRID_SPACING) + 1;

    cout << "生成 " << gridPoints << "×" << gridPoints << " 网格，间隔 " << GRID_SPACING << " 单位" << endl;

    // Z 方向（垂直于 XOY）
    float zStart = -halfSize * 1.5f;
    for (int i = 0; i < gridPoints; ++i) {
        for (int j = 0; j < gridPoints; ++j) {
            float x = -halfSize + i * GRID_SPACING;
            float y = -halfSize + j * GRID_SPACING;
            Line line;
            line.origin = Point3D(x, y, zStart);
            line.direction = Point3D(0.0f, 0.0f, 1.0f);
            lines.push_back(line);
        }
    }

    // Y 方向（垂直于 XOZ）
    float yStart = -halfSize * 1.5f;
    for (int i = 0; i < gridPoints; ++i) {
        for (int j = 0; j < gridPoints; ++j) {
            float x = -halfSize + i * GRID_SPACING;
            float z = -halfSize + j * GRID_SPACING;
            Line line;
            line.origin = Point3D(x, yStart, z);
            line.direction = Point3D(0.0f, 1.0f, 0.0f);
            lines.push_back(line);
        }
    }

    // X 方向（垂直于 YOZ）
    float xStart = -halfSize * 1.5f;
    for (int i = 0; i < gridPoints; ++i) {
        for (int j = 0; j < gridPoints; ++j) {
            float y = -halfSize + i * GRID_SPACING;
            float z = -halfSize + j * GRID_SPACING;
            Line line;
            line.origin = Point3D(xStart, y, z);
            line.direction = Point3D(1.0f, 0.0f, 0.0f);
            lines.push_back(line);
        }
    }

    cout << "总共生成 " << lines.size() << " 条直线" << endl;
    return lines;
}

// 绘制圆柱体内部的线段（根据 validFlags 与 visibleLines 按原始索引绘制）
void drawVisibleLineSegments() {
    glLineWidth(3.0f);

    int gridPoints = static_cast<int>(PLANE_SIZE / GRID_SPACING) + 1;
    int linesPerPlane = gridPoints * gridPoints;
    int total = lines.size();

    // 逐条按索引绘制，颜色按平面分组
    glBegin(GL_LINES);
    for (int i = 0; i < total; ++i) {
        if (!validFlags[i]) continue;
        const Line& seg = visibleLines[i];
        if (i < linesPerPlane) {
            glColor3f(1.0f, 0.5f, 0.0f); // Z 方向
        } else if (i < 2 * linesPerPlane) {
            glColor3f(0.0f, 1.0f, 0.5f); // Y 方向
        } else {
            glColor3f(0.5f, 0.0f, 1.0f); // X 方向
        }

        Point3D end;
        end.x = seg.origin.x + seg.direction.x;
        end.y = seg.origin.y + seg.direction.y;
        end.z = seg.origin.z + seg.direction.z;
        glVertex3f(seg.origin.x, seg.origin.y, seg.origin.z);
        glVertex3f(end.x, end.y, end.z);
    }
    glEnd();

    glLineWidth(1.0f);
}

void drawCoordinateSystem() {
    glLineWidth(2.0f);
    glColor3f(1.0f, 0.0f, 0.0f);
    glBegin(GL_LINES);
    glVertex3f(0,0,0); glVertex3f(15,0,0);
    glEnd();
    glColor3f(0.0f,1.0f,0.0f);
    glBegin(GL_LINES);
    glVertex3f(0,0,0); glVertex3f(0,15,0);
    glEnd();
    glColor3f(0.0f,0.0f,1.0f);
    glBegin(GL_LINES);
    glVertex3f(0,0,0); glVertex3f(0,0,15);
    glEnd();
    glLineWidth(1.0f);
}

void drawCylinder() {
    const int segments = 36;
    const float height = CYLINDER_HEIGHT;
    const float radius = CYLINDER_RADIUS;

    glColor4f(0.3f, 0.5f, 0.8f, 0.3f);
    glBegin(GL_QUADS);
    for (int i = 0; i < segments; ++i) {
        float angle1 = 2.0f * M_PI * i / segments;
        float angle2 = 2.0f * M_PI * (i + 1) / segments;
        float x1 = radius * cos(angle1), y1 = radius * sin(angle1);
        float x2 = radius * cos(angle2), y2 = radius * sin(angle2);
        glVertex3f(x1, y1, -height/2); glVertex3f(x1, y1, height/2);
        glVertex3f(x2, y2, height/2); glVertex3f(x2, y2, -height/2);
    }
    glEnd();

    glColor4f(0.2f, 0.4f, 0.7f, 0.8f);
    glBegin(GL_LINES);
    for (int i = 0; i < segments; ++i) {
        float angle1 = 2.0f * M_PI * i / segments;
        float angle2 = 2.0f * M_PI * (i + 1) / segments;
        float x1 = radius * cos(angle1), y1 = radius * sin(angle1);
        float x2 = radius * cos(angle2), y2 = radius * sin(angle2);
        glVertex3f(x1, y1, -height/2); glVertex3f(x1, y1, height/2);
        glVertex3f(x1, y1, -height/2); glVertex3f(x2, y2, -height/2);
        glVertex3f(x1, y1, height/2);  glVertex3f(x2, y2, height/2);
    }
    glEnd();
}

void drawIntersections() {
    glPointSize(6.0f);
    glBegin(GL_POINTS);
    for (int i = 0; i < intersections.size(); ++i) {
        if (!validFlags[i]) continue;
        const Point3D& p = intersections[i];
        glColor3f(1.0f,0.0f,0.0f);
        glVertex3f(p.x,p.y,p.z);
    }
    glEnd();
    glPointSize(1.0f);
}

void printIntersectionCoordinates() {
    cout << "\n=== 交点坐标 ===" << endl;
    int gridPoints = static_cast<int>(PLANE_SIZE / GRID_SPACING) + 1;
    int linesPerPlane = gridPoints * gridPoints;
    int cnt = 0;
    for (int i = 0; i < intersections.size(); ++i) {
        if (!validFlags[i]) continue;
        string type = (i < linesPerPlane) ? "Z" : (i < 2*linesPerPlane) ? "Y" : "X";
        const Point3D& p = intersections[i];
        cout << "交点 " << ++cnt << " (" << type << "): (" << p.x << ", " << p.y << ", " << p.z << ")" << endl;
    }
    cout << "总共找到 " << cnt << " 个交点，显示 " << count(validFlags.begin(), validFlags.end(), 1) << " 条可见线段" << endl;
}

void drawText() {
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    gluOrtho2D(0, windowWidth, 0, windowHeight);
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    glColor3f(1,1,1);
    int gridPoints = static_cast<int>(PLANE_SIZE / GRID_SPACING) + 1;
    int linesPerPlane = gridPoints * gridPoints;
    int zCnt=0,yCnt=0,xCnt=0;
    for (int i=0;i<lines.size();++i) if (validFlags[i]) {
        if (i<linesPerPlane) ++zCnt;
        else if (i<2*linesPerPlane) ++yCnt;
        else ++xCnt;
    }

    string info = "圆柱体: 半径=" + to_string(CYLINDER_RADIUS) + ", 高度=" + to_string(CYLINDER_HEIGHT)
        + " | 交点: Z:" + to_string(zCnt) + " Y:" + to_string(yCnt) + " X:" + to_string(xCnt)
        + " | 显示线段:" + to_string(count(validFlags.begin(), validFlags.end(), 1));

    glRasterPos2f(10, 30);
    for (char c : info) glutBitmapCharacter(GLUT_BITMAP_9_BY_15, c);

    string controls = "控制: 鼠标旋转, W/S缩放, R重置, P打印坐标, ESC退出";
    glRasterPos2f(10, 10);
    for (char c : controls) glutBitmapCharacter(GLUT_BITMAP_9_BY_15, c);

    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
}

void drawLines() {
    glLineWidth(1.0f);
    glBegin(GL_LINES);
    for (const Line& line : lines) {
        glColor3f(0.8f,0.8f,0.8f);
        Point3D end;
        end.x = line.origin.x + line.direction.x *  (PLANE_SIZE * 2.0f);
        end.y = line.origin.y + line.direction.y *  (PLANE_SIZE * 2.0f);
        end.z = line.origin.z + line.direction.z *  (PLANE_SIZE * 2.0f);
        glVertex3f(line.origin.x, line.origin.y, line.origin.z);
        glVertex3f(end.x, end.y, end.z);
    }
    glEnd();
    glLineWidth(1.0f);
}

void display() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glLoadIdentity();

    gluLookAt(cameraDistance, cameraDistance, cameraDistance,
        0.0, 0.0, 0.0,
        0.0, 1.0, 0.0);

    glRotatef(cameraAngleX, 1.0f, 0.0f, 0.0f);
    glRotatef(cameraAngleY, 0.0f, 1.0f, 0.0f);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    drawCoordinateSystem();
    drawCylinder();

    // 绘制可见线段（按索引分组着色）
    drawVisibleLineSegments();

    // 绘制所有原始直线（可注释掉）
    // drawLines();

    drawIntersections();

    glDisable(GL_BLEND);

    drawText();

    glutSwapBuffers();
}

void reshape(int width, int height) {
    windowWidth = width; windowHeight = height;
    glViewport(0,0,width,height);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(45.0f, (float)width/height, 0.1f, 100.0f);
    glMatrixMode(GL_MODELVIEW);
}

void keyboard(unsigned char key, int x, int y) {
    switch (key) {
    case 27: exit(0); break;
    case 'r': case 'R':
        cameraAngleX = 30.0f; cameraAngleY = 45.0f; cameraDistance = 25.0f; break;
    case 'w': case 'W': cameraDistance -= 1.0f; break;
    case 's': case 'S': cameraDistance += 1.0f; break;
    case 'p': case 'P': printIntersectionCoordinates(); break;
    }
    glutPostRedisplay();
}

void mouse(int button, int state, int x, int y) {
    if (button == GLUT_LEFT_BUTTON) {
        if (state == GLUT_DOWN) { mouseLeftDown = true; mouseX = x; mouseY = y; }
        else mouseLeftDown = false;
    }
}

void motion(int x, int y) {
    if (mouseLeftDown) {
        cameraAngleY += (x - mouseX) * 0.5f;
        cameraAngleX += (y - mouseY) * 0.5f;
        mouseX = x; mouseY = y;
    }
    glutPostRedisplay();
}

void initializeOpenGL(int argc, char** argv) {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
    glutInitWindowSize(800,600);
    glutCreateWindow("圆柱体与轴向网格直线相交 - 解析解");

    glEnable(GL_DEPTH_TEST);
    glClearColor(0.1f,0.1f,0.1f,1.0f);

    lines = generateGridLines();

    // 使用解析方法计算交点与可见线段
    auto start = chrono::high_resolution_clock::now();
    computeAnalyticIntersectionsAndSegments();
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    cout << "解析计算完成，耗时: " << duration.count() << " ms" << endl;
    cout << "有效交点数: " << count(validFlags.begin(), validFlags.end(), 1) << endl;

    printIntersectionCoordinates();

    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutKeyboardFunc(keyboard);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);

    glutMainLoop();
}

int main(int argc, char** argv) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    cout << "使用GPU: " << prop.name << endl;
    cout << "显存: " << prop.totalGlobalMem / (1024*1024) << " MB" << endl;

    initializeOpenGL(argc, argv);
    return 0;
}