#include "line_surface_intersection.cuh"
#include <iostream>
#include <chrono>

using namespace std;

// 曲面函数实现

__device__ float surface_function(float x, float y, float z) {
    // 圆柱体方程: x² + y² = radius², 且 -height/2 <= z <= height/2
    const float radius = 5.0f;      // 与主程序保持一致
    const float height = 12.0f;     // 与主程序保持一致

    // 计算点到圆柱体侧面的距离
    float side_distance = sqrtf(x * x + y * y) - radius;

    // 计算点到圆柱体顶面/底面的距离
    float top_distance = fabsf(z) - height / 2.0f;

    // 使用有向距离场，返回0表示在表面上
    // 正值表示在外部，负值表示在内部
    float cylinder_distance = fmaxf(side_distance, top_distance);

    return cylinder_distance;
}


__global__ void findIntersectionPoints(
    const Line* lines,
    Point3D* intersections,
    int* validFlags,
    int numLines,
    float stepSize,
    int maxSteps) {

    // 使用CUDA内置变量 - 这些只能在设备代码中使用
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numLines) return;

    Line line = lines[idx];
    Point3D current = line.origin;

    // 标准化方向向量
    float length = sqrtf(
        line.direction.x * line.direction.x +
        line.direction.y * line.direction.y +
        line.direction.z * line.direction.z
    );

    // 避免除零错误
    if (length < 1e-6f) {
        intersections[idx] = Point3D(0, 0, 0);
        validFlags[idx] = 0;
        return;
    }

    Point3D dir = {
        line.direction.x / length,
        line.direction.y / length,
        line.direction.z / length
    };

    // 沿着直线搜索交点
    float prev_value = surface_function(current.x, current.y, current.z);
    bool found = false;

    for (int step = 0; step < maxSteps; ++step) {
        current.x += dir.x * stepSize;
        current.y += dir.y * stepSize;
        current.z += dir.z * stepSize;

        float current_value = surface_function(current.x, current.y, current.z);

        // 检查符号变化（表示穿过曲面）
        if (prev_value * current_value <= 0.0f && fabsf(prev_value - current_value) > 1e-6f) {
            // 使用二分法精确化交点
            Point3D low = {
                current.x - dir.x * stepSize,
                current.y - dir.y * stepSize,
                current.z - dir.z * stepSize
            };
            Point3D high = current;
            float low_value = prev_value;
            float high_value = current_value;

            for (int refine = 0; refine < 10; ++refine) {
                Point3D mid = {
                    (low.x + high.x) * 0.5f,
                    (low.y + high.y) * 0.5f,
                    (low.z + high.z) * 0.5f
                };

                float mid_value = surface_function(mid.x, mid.y, mid.z);

                if (low_value * mid_value <= 0.0f) {
                    high = mid;
                    high_value = mid_value;
                }
                else {
                    low = mid;
                    low_value = mid_value;
                }
            }

            intersections[idx] = Point3D(
                (low.x + high.x) * 0.5f,
                (low.y + high.y) * 0.5f,
                (low.z + high.z) * 0.5f
            );
            validFlags[idx] = 1;
            found = true;
            break;
        }

        prev_value = current_value;
    }

    if (!found) {
        intersections[idx] = Point3D(0, 0, 0);
        validFlags[idx] = 0;
    }
}

void findIntersectionsGPU(
    const vector<Line>& lines,
    vector<Point3D>& intersections,
    vector<int>& validFlags,
    float stepSize,
    int maxSteps) {

    int numLines = lines.size();
    intersections.resize(numLines);
    validFlags.resize(numLines);

    // 分配设备内存
    Line* d_lines = nullptr;
    Point3D* d_intersections = nullptr;
    int* d_validFlags = nullptr;

    cudaError_t err;

    // 分配GPU内存
    err = cudaMalloc(&d_lines, numLines * sizeof(Line));
    if (err != cudaSuccess) {
        cerr << "cudaMalloc failed for d_lines: " << cudaGetErrorString(err) << endl;
        return;
    }

    err = cudaMalloc(&d_intersections, numLines * sizeof(Point3D));
    if (err != cudaSuccess) {
        cerr << "cudaMalloc failed for d_intersections: " << cudaGetErrorString(err) << endl;
        cudaFree(d_lines);
        return;
    }

    err = cudaMalloc(&d_validFlags, numLines * sizeof(int));
    if (err != cudaSuccess) {
        cerr << "cudaMalloc failed for d_validFlags: " << cudaGetErrorString(err) << endl;
        cudaFree(d_lines);
        cudaFree(d_intersections);
        return;
    }

    // 复制数据到设备
    err = cudaMemcpy(d_lines, lines.data(), numLines * sizeof(Line), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cerr << "cudaMemcpy failed for d_lines: " << cudaGetErrorString(err) << endl;
        cudaFree(d_lines);
        cudaFree(d_intersections);
        cudaFree(d_validFlags);
        return;
    }

    // 计算网格和块大小
    int blockSize = 256;
    int gridSize = (numLines + blockSize - 1) / blockSize;

    cout << "启动CUDA内核: gridSize=" << gridSize << ", blockSize=" << blockSize << endl;

    // 启动内核
    auto start = chrono::high_resolution_clock::now();

    findIntersectionPoints << <gridSize, blockSize >> > (
        d_lines, d_intersections, d_validFlags, numLines, stepSize, maxSteps
        );

    cudaDeviceSynchronize();

    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end - start);

    // 检查CUDA错误
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cerr << "CUDA kernel error: " << cudaGetErrorString(err) << endl;
    }

    // 复制结果回主机
    err = cudaMemcpy(intersections.data(), d_intersections, numLines * sizeof(Point3D), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        cerr << "cudaMemcpy failed for intersections: " << cudaGetErrorString(err) << endl;
    }

    err = cudaMemcpy(validFlags.data(), d_validFlags, numLines * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        cerr << "cudaMemcpy failed for validFlags: " << cudaGetErrorString(err) << endl;
    }

    // 释放设备内存
    cudaFree(d_lines);
    cudaFree(d_intersections);
    cudaFree(d_validFlags);

    cout << "GPU计算完成，耗时: " << duration.count() << " 微秒" << endl;
    cout << "约 " << duration.count() / 1000.0 << " 毫秒" << endl;
}