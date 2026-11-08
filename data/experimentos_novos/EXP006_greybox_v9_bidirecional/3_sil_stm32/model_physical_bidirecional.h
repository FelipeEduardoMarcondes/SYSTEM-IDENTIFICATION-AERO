/*
 * model_physical_bidirecional.h
 * =============================
 * Simulador de Planta Fisica do Aeropendulo baseado no Modelo Grey-Box v9 Bidirecional
 * (Arrasto Aerodinamico -ca*w*|w| + Rigidez Torsional -k_th*th + Empuxo Bidirecional +-80%)
 */

#ifndef MODEL_PHYSICAL_BIDIRECIONAL_H
#define MODEL_PHYSICAL_BIDIRECIONAL_H

#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

#define GB_C1          (-6.858314f)
#define GB_K_TH        (1.565398f)
#define GB_C2          (-0.331455f)
#define GB_CA          (0.019310f)
#define GB_FC          (0.255820f)
#define GB_C3          (0.01241200f)
#define GB_P_THRUST    (1.759243f)
#define GB_U0          (11.8522f)
#define GB_TAU_M       (0.113127f)

static float sil_th_rad = 0.0f;
static float sil_v_rad  = 0.0f;
static float sil_a_pct  = 0.0f;

static inline float u_equilibrium(float th_deg)
{
    float th_rad = th_deg * ((float)M_PI / 180.0f);
    float restoring = -GB_C1 * sinf(th_rad) + GB_K_TH * th_rad;
    if (fabsf(restoring) < 1e-5f) return 0.0f;
    float dir = (restoring > 0.0f) ? 1.0f : -1.0f;
    float u_val = GB_U0 + powf(fabsf(restoring) / GB_C3, 1.0f / GB_P_THRUST);
    if (u_val > 80.0f) u_val = 80.0f;
    return dir * u_val;
}

static inline void phys_derivs(float th, float v, float a, float u_cmd,
                               float *dth, float *dv, float *da)
{
    float u_motor = (u_cmd > 80.0f) ? 80.0f : ((u_cmd < -80.0f) ? -80.0f : u_cmd);
    float a_abs = fabsf(a);
    float z = a_abs - GB_U0;
    float thrust = 0.0f;
    if (z > 0.0f) {
        float dir = (a > 0.0f) ? 1.0f : ((a < 0.0f) ? -1.0f : 0.0f);
        thrust = dir * powf(z, GB_P_THRUST);
    }
    float aero = -GB_CA * v * fabsf(v);
    float fric = -GB_FC * tanhf(50.0f * v);

    *dth = v;
    *dv  = GB_C1 * sinf(th) - GB_K_TH * th + GB_C2 * v + aero + fric + GB_C3 * thrust;
    *da  = (u_motor - a) / GB_TAU_M;
}

static inline float simulate_physical_plant(float u_cmd)
{
    const float dt = 0.01f;
    const float h  = dt / 2.0f;
    float th = sil_th_rad;
    float v  = sil_v_rad;
    float a  = sil_a_pct;

    for (int sub = 0; sub < 2; sub++) {
        float k1_th, k1_v, k1_a;
        phys_derivs(th, v, a, u_cmd, &k1_th, &k1_v, &k1_a);

        float k2_th, k2_v, k2_a;
        phys_derivs(th + 0.5f * h * k1_th, v + 0.5f * h * k1_v, a + 0.5f * h * k1_a, u_cmd, &k2_th, &k2_v, &k2_a);

        float k3_th, k3_v, k3_a;
        phys_derivs(th + 0.5f * h * k2_th, v + 0.5f * h * k2_v, a + 0.5f * h * k2_a, u_cmd, &k3_th, &k3_v, &k3_a);

        float k4_th, k4_v, k4_a;
        phys_derivs(th + h * k3_th, v + h * k3_v, a + h * k3_a, u_cmd, &k4_th, &k4_v, &k4_a);

        th += (h / 6.0f) * (k1_th + 2.0f * k2_th + 2.0f * k3_th + k4_th);
        v  += (h / 6.0f) * (k1_v  + 2.0f * k2_v  + 2.0f * k3_v  + k4_v);
        a  += (h / 6.0f) * (k1_a  + 2.0f * k2_a  + 2.0f * k3_a  + k4_a);
    }

    sil_th_rad = th;
    sil_v_rad  = v;
    sil_a_pct  = a;

    return th * (180.0f / (float)M_PI);
}

#endif /* MODEL_PHYSICAL_BIDIRECIONAL_H */
