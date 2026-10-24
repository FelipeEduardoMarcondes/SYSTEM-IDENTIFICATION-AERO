/*
 * model_physical.h
 * ================
 * Simulador de Planta Fisica do Aeropendulo baseado no Modelo Grey-Box v5
 * Identificado em: ident_greybox_prof_multi_v5.py (Modelo: mecanica regularizada)
 *
 * Equacoes de estado nao-lineares em tempo continuo (3a ordem):
 *   d(theta)/dt = v
 *   d(v)/dt     = c1*sin(theta) + c2*v - c_aero*v*|v| - Fc*tanh(50*v) + c3*(max(a - u0, 0))^2
 *   d(a)/dt     = (u_motor - a) / tau_m
 *
 * Integrador numerico: Runge-Kutta 4a Ordem (RK4) com 2 subpassos de 5 ms a cada ciclo de 10 ms (100 Hz).
 */

#ifndef MODEL_PHYSICAL_H
#define MODEL_PHYSICAL_H

#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

/* ── Parametros fisicos do modelo Grey-Box v5 ───────────────────────────── */
#ifndef GB_C1
#define GB_C1      (-7.228132f)   /* rad/s^2 - torque gravitacional normalizado */
#endif

#ifndef GB_C2
#define GB_C2      (-0.160000f)   /* s^-1    - amortecimento viscoso dinamico */
#endif

#ifndef GB_C3
#define GB_C3      (0.00383230f)  /* rad/(s^2 * %^2) - ganho de empuxo quadratico da helice */
#endif

#ifndef GB_FC
#define GB_FC      (0.121047f)    /* rad/s^2 - atrito seco de Coulomb no mancal */
#endif

#ifndef GB_C_AERO
#define GB_C_AERO  (0.246141f)    /* rad^-1  - arrasto aerodinamico quadratico */
#endif

#ifndef GB_U0
#define GB_U0      (11.2637f)     /* %       - zona morta do conjunto motor/esc */
#endif

#ifndef GB_TAU_M
#define GB_TAU_M   (0.0800f)      /* s       - constante de tempo do motor (80 ms) */
#endif

/* ── Estados internos da planta fisica no modo SIL ──────────────────────── */
static float sil_th_rad = 0.0f; /* Posicao angular (rad, 0 = repouso vertical para baixo) */
static float sil_v_rad  = 0.0f; /* Velocidade angular (rad/s) */
static float sil_a_pct  = 0.0f; /* Estado dinamico do atuador/motor (%) */

/*
 * Derivadas continuas do modelo fisico Grey-Box v5
 */
static inline void phys_derivs(float th, float v, float a, float u_cmd,
                               float *dth, float *dv, float *da)
{
    float u_motor = (u_cmd > 0.0f) ? ((u_cmd > 100.0f) ? 100.0f : u_cmd) : 0.0f;
    float z = a - GB_U0;
    float thrust = (z > 0.0f) ? (z * z) : 0.0f;
    float fric   = -GB_FC * tanhf(50.0f * v);
    float aero   = -GB_C_AERO * v * fabsf(v);

    *dth = v;
    *dv  = GB_C1 * sinf(th) + GB_C2 * v + aero + fric + GB_C3 * thrust;
    *da  = (u_motor - a) / GB_TAU_M;
}

/*
 * Simulador de Planta Fisica Grey-Box v5 (RK4 a 100 Hz com 2 subpassos de 5 ms)
 * Executado a cada chamada no modo SIL para propagar a dinamica real da bancada.
 * Atualiza os historicos sil_y_hist e sil_u_hist para qualquer controlador (PID, NARX ou Greybox).
 */
static inline float simulate_physical_plant(float u_cmd)
{
    const float dt = 0.01f;
    const float h  = dt / 2.0f; /* 2 subpassos de 5 ms para maxima estabilidade */
    float th = sil_th_rad;
    float v  = sil_v_rad;
    float a  = sil_a_pct;

    for (int sub = 0; sub < 2; sub++) {
        float k1_th, k1_v, k1_a;
        phys_derivs(th, v, a, u_cmd, &k1_th, &k1_v, &k1_a);

        float k2_th, k2_v, k2_a;
        phys_derivs(th + 0.5f * h * k1_th,
                    v  + 0.5f * h * k1_v,
                    a  + 0.5f * h * k1_a,
                    u_cmd, &k2_th, &k2_v, &k2_a);

        float k3_th, k3_v, k3_a;
        phys_derivs(th + 0.5f * h * k2_th,
                    v  + 0.5f * h * k2_v,
                    a  + 0.5f * h * k2_a,
                    u_cmd, &k3_th, &k3_v, &k3_a);

        float k4_th, k4_v, k4_a;
        phys_derivs(th + h * k3_th,
                    v  + h * k3_v,
                    a  + h * k3_a,
                    u_cmd, &k4_th, &k4_v, &k4_a);

        th += (h / 6.0f) * (k1_th + 2.0f * k2_th + 2.0f * k3_th + k4_th);
        v  += (h / 6.0f) * (k1_v  + 2.0f * k2_v  + 2.0f * k3_v  + k4_v);
        a  += (h / 6.0f) * (k1_a  + 2.0f * k2_a  + 2.0f * k3_a  + k4_a);
    }

    sil_th_rad = th;
    sil_v_rad  = v;
    sil_a_pct  = a;

    float ang_deg = th * (180.0f / (float)M_PI);

#if defined(NY_MODEL) && (NY_MODEL > 1)
    for (int i = NY_MODEL - 1; i > 0; i--) sil_y_hist[i] = sil_y_hist[i-1];
    sil_y_hist[0] = ang_deg;
#else
    sil_y_hist[0] = ang_deg;
#endif

#if defined(NU_MODEL) && (NU_MODEL > 1)
    for (int i = NU_MODEL - 1; i > 0; i--) sil_u_hist[i] = sil_u_hist[i-1];
    sil_u_hist[0] = u_cmd;
#else
    sil_u_hist[0] = u_cmd;
#endif

    return ang_deg;
}

/*
 * Alias de compatibilidade retroativa para chamadas legadas de simulate_narx
 */
static inline float simulate_narx(float u_cmd)
{
    return simulate_physical_plant(u_cmd);
}

#endif /* MODEL_PHYSICAL_H */
