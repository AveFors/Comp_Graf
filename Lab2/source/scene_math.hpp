#pragma once
#include <cmath>

namespace math {
struct Vec3 { float x, y, z; };
struct Matrix { float data[16]{}; };
inline Vec3 operator-(Vec3 a, Vec3 b) { return {a.x-b.x, a.y-b.y, a.z-b.z}; }
inline float dot(Vec3 a, Vec3 b) { return a.x*b.x+a.y*b.y+a.z*b.z; }
inline Vec3 cross(Vec3 a, Vec3 b) {
    return {a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x};
}
inline Vec3 normalized(Vec3 v) {
    const float length = std::sqrt(dot(v,v));
    if (length < 0.000001f) return {0,0,1};
    return {v.x/length, v.y/length, v.z/length};
}
inline Vec3 vector(const float v[3]) { return {v[0],v[1],v[2]}; }

// ========================================================================
// Матрицы 4x4: хранение по столбцам, вектор умножается справа
// ========================================================================
inline Matrix identity() {
    Matrix m; m.data[0]=m.data[5]=m.data[10]=m.data[15]=1; return m;
}
inline Matrix multiply(const Matrix& a, const Matrix& b) {
    Matrix m;
    for (int c=0;c<4;++c) for (int r=0;r<4;++r) for (int k=0;k<4;++k)
        m.data[c*4+r] += a.data[k*4+r]*b.data[c*4+k];
    return m;
}
inline Matrix transpose(const Matrix& a) {
    Matrix m;
    for (int c=0;c<4;++c) for (int r=0;r<4;++r) m.data[c*4+r]=a.data[r*4+c];
    return m;
}
inline Matrix translation(Vec3 p) {
    Matrix m=identity(); m.data[12]=p.x; m.data[13]=p.y; m.data[14]=p.z; return m;
}
inline Matrix scaling(Vec3 p) {
    Matrix m=identity(); m.data[0]=p.x; m.data[5]=p.y; m.data[10]=p.z; return m;
}
inline Matrix rotation(const float degrees[3]) {
    constexpr float radians=3.14159265358979323846f/180;
    const float x=degrees[0]*radians, y=degrees[1]*radians, z=degrees[2]*radians;
    Matrix rx=identity(),ry=identity(),rz=identity();
    rx.data[5]=rx.data[10]=std::cos(x); rx.data[6]=std::sin(x); rx.data[9]=-std::sin(x);
    ry.data[0]=ry.data[10]=std::cos(y); ry.data[2]=-std::sin(y); ry.data[8]=std::sin(y);
    rz.data[0]=rz.data[5]=std::cos(z); rz.data[1]=std::sin(z); rz.data[4]=-std::sin(z);
    return multiply(rz,multiply(ry,rx));
}

// ========================================================================
// Проекция Vulkan: глубина [0,1], инверсия экранной оси Y
// ========================================================================
inline Matrix projection(int mode,float aspect,float fov,float nearPlane,float farPlane,float height) {
    Matrix p;
    if (mode==0) {
        const float focal=1/std::tan(fov*3.14159265358979323846f/360);
        p.data[0]=focal/aspect; p.data[5]=-focal;
        p.data[10]=farPlane/(nearPlane-farPlane); p.data[11]=-1;
        p.data[14]=farPlane*nearPlane/(nearPlane-farPlane);
    } else {
        p.data[0]=2/(height*aspect); p.data[5]=-2/height;
        p.data[10]=1/(nearPlane-farPlane); p.data[14]=nearPlane/(nearPlane-farPlane); p.data[15]=1;
    }
    return p;
}
}
