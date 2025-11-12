#ifndef LINE_SURFACE_INTERSECTION_CUH
#define LINE_SURFACE_INTERSECTION_CUH

#include <cuda_runtime.h>
#include <vector>
#include <cmath>

using namespace std;


// 三维点结构体
struct Point3D {
    float x, y, z;
    __host__ __device__ Point3D() : x(0), y(0), z(0) {}
    __host__ __device__ Point3D(float x, float y, float z) : x(x), y(y), z(z) {}
};


// 直线结构体
struct Line {
    Point3D origin;
    Point3D direction;
};

// 曲面函数声明
__device__ float surface_function(float x, float y, float z);

// 在GPU上计算单条直线与曲面的交点
__global__ void findIntersectionPoints(
    const Line* lines,
    Point3D* intersections,
    int* validFlags,
    int numLines,
    float stepSize,
    int maxSteps
);

// 主机端包装函数
void findIntersectionsGPU(
    const vector<Line>& lines,
    vector<Point3D>& intersections,
    vector<int>& validFlags,
    float stepSize = 0.01f,
    int maxSteps = 1000
);

#endif