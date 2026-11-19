/* 정사영 깊이·삼각형 번호 버퍼 (raster.py가 ctypes로 부른다).
   빌드(윈도우용): x86_64-w64-mingw32-gcc -O3 -shared -o elevation_ai/raster_win64.dll native/raster.c
   계산식은 예전 numba 판과 같은 순서라 결과가 같다. */
#include <stdint.h>
#include <math.h>

#ifdef _WIN32
#define API __declspec(dllexport)
#else
#define API __attribute__((visibility("default")))
#endif

API void rasterize(const double *sx, const double *sy, const double *sz, const int32_t *tris, int64_t T,
                   int32_t W, int32_t H, float *zb, int32_t *ib)
{
    for (int64_t t = 0; t < T; t++) {
        int32_t a = tris[3 * t], b = tris[3 * t + 1], c = tris[3 * t + 2];
        double x0 = sx[a], y0 = sy[a], z0 = sz[a];
        double x1 = sx[b], y1 = sy[b], z1 = sz[b];
        double x2 = sx[c], y2 = sy[c], z2 = sz[c];
        double den = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2);
        if (fabs(den) < 1e-12) continue;
        double mnx = fmin(x0, fmin(x1, x2)), mxx = fmax(x0, fmax(x1, x2));
        double mny = fmin(y0, fmin(y1, y2)), mxy = fmax(y0, fmax(y1, y2));
        if (!(mxx >= 0.0) || !(mxy >= 0.0) || !(mnx <= (double)W) || !(mny <= (double)H)) continue; /* 화면 밖·NaN */
        int32_t minx = (int32_t)fmax(floor(mnx), 0.0), maxx = (int32_t)fmin(ceil(mxx), (double)(W - 1));
        int32_t miny = (int32_t)fmax(floor(mny), 0.0), maxy = (int32_t)fmin(ceil(mxy), (double)(H - 1));
        double a0 = y1 - y2, b0 = x2 - x1, a1 = y2 - y0, b1 = x0 - x2;
        for (int32_t py = miny; py <= maxy; py++) {
            double cy = py + 0.5;
            float *zrow = zb + (int64_t)py * W;
            int32_t *irow = ib + (int64_t)py * W;
            double e0 = b0 * (cy - y2), e1 = b1 * (cy - y2);
            for (int32_t px = minx; px <= maxx; px++) {
                double cx = px + 0.5;
                double w0 = (a0 * (cx - x2) + e0) / den;
                double w1 = (a1 * (cx - x2) + e1) / den;
                double w2 = 1.0 - w0 - w1;
                if (w0 < -1e-6 || w1 < -1e-6 || w2 < -1e-6) continue;
                double z = w0 * z0 + w1 * z1 + w2 * z2;
                if (z < (double)zrow[px]) { zrow[px] = (float)z; irow[px] = (int32_t)t; }
            }
        }
    }
}

/* 가우스 흐림(가로→세로, 가장자리는 거울 반사). k: 길이 2r+1 커널, tmp: H*W float 작업 공간 */
static inline int64_t refl(int64_t i, int64_t n)
{
    if (n == 1) return 0;
    int64_t p = 2 * n;
    i %= p; if (i < 0) i += p;
    return i < n ? i : p - 1 - i;
}

API void gauss_blur(const float *src, float *dst, float *tmp, int32_t H, int32_t W, const double *k, int32_t r)
{
    /* tmp는 (W + 2r) + H*W 칸: 앞쪽은 거울 반사로 늘린 한 줄 */
    float *pad = tmp, *mid = tmp + (W + 2 * r);
    for (int32_t y = 0; y < H; y++) {
        const float *s = src + (int64_t)y * W;
        float *o = mid + (int64_t)y * W;
        for (int32_t x = -r; x < W + r; x++) pad[x + r] = s[refl(x, W)];
        for (int32_t x = 0; x < W; x++) o[x] = 0.0f;
        for (int32_t i = 0; i <= 2 * r; i++) {
            float w = (float)k[i];
            const float *q = pad + i;
            for (int32_t x = 0; x < W; x++) o[x] += w * q[x];
        }
    }
    for (int32_t y = 0; y < H; y++) {
        float *o = dst + (int64_t)y * W;
        for (int32_t x = 0; x < W; x++) o[x] = 0.0f;
        for (int32_t i = -r; i <= r; i++) {
            const float *s = mid + refl((int64_t)y + i, H) * W;
            float w = (float)k[i + r];
            for (int32_t x = 0; x < W; x++) o[x] += w * s[x];
        }
    }
}
