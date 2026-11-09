/*
 * model_physical.h
 * ================
 * Simulador de Planta Fisica do Aeropendulo baseado no Modelo Grey-Box v9 Completo
 * (Arrasto Aerodinamico Quadratico -ca*w*|w| + Rigidez Torsional do Chicote -k_th*th + Dinamica de 1a Ordem tau_m)
 * Identificado em: ident_greybox_prof_multi_v9.py (Modelo Selecionado: v9_otimizado)
 *
 * Equacoes de estado nao-lineares em tempo continuo (3a ordem):
 *   d(theta)/dt = omega
 *   d(omega)/dt = c1*sin(theta) - k_th*theta + c2*omega - ca*omega*|omega| - Fc*tanh(50*omega) + c3*(max(a - u0, 0))^p
 *   d(a)/dt     = (u_motor - a) / tau_m
 *
 * Parametros identificados (v9_otimizado):
 *   c1    = -6.858314f rad/s^2   (torque gravitacional normalizado)
 *   k_th  = 1.565398f s^-2       (rigidez torsional elastica do chicote no pivo)
 *   c2    = -0.331455f s^-1      (amortecimento viscoso mecanico)
 *   ca    = 0.019310f rad^-1     (coeficiente de arrasto quadratico aerodinamico)
 *   Fc    = 0.255820f rad/s^2    (atrito seco de Coulomb no mancal)
 *   c3    = 0.01241200f            (ganho de empuxo nao-linear)
 *   p     = 1.759243f            (expoente de empuxo aerodinamico da helice)
 *   u0    = 11.8522f %         (zona morta da turbina/ESC)
 *   tau_m = 0.113127f s          (constante de tempo de resposta do motor)
 *
 * Integrador numerico: Runge-Kutta 4a Ordem (RK4) com 2 subpassos de 5 ms a cada ciclo de 10 ms (100 Hz).
 */

#ifndef MODEL_PHYSICAL_H
#define MODEL_PHYSICAL_H

#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

/* ── Parametros fisicos do modelo Grey-Box v9 (v9_otimizado) ─────────────── */
#ifndef GB_C1
#define GB_C1          (-6.858314f)   /* rad/s^2 - torque gravitacional normalizado */
#endif

#ifndef GB_K_TH
#define GB_K_TH        (1.565398f)   /* s^-2    - rigidez torsional do chicote no pivo */
#endif

#ifndef GB_C2
#define GB_C2          (-0.331455f)   /* s^-1    - amortecimento viscoso mecanico */
#endif

#ifndef GB_CA
#define GB_CA          (0.019310f)   /* rad^-1  - coeficiente de arrasto quadratico */
#endif

#ifndef GB_FC
#define GB_FC          (0.255820f)   /* rad/s^2 - atrito seco de Coulomb no mancal */
#endif

#ifndef GB_C3
#define GB_C3          (0.01241200f)  /* ganho de empuxo nao-linear */
#endif

#ifndef GB_P_THRUST
#define GB_P_THRUST    (1.759243f)   /* expoente de empuxo nao-linear */
#endif

#ifndef GB_U0
#define GB_U0          (11.8522f)    /* %       - zona morta da turbina/ESC */
#endif

#ifndef GB_TAU_M
#define GB_TAU_M       (0.113127f)   /* s       - constante de tempo do atuador */
#endif

/* ── Estados internos da planta fisica no modo SIL ──────────────────────── */
static float sil_th_rad = 0.0f; /* Posicao angular (rad, 0 = repouso vertical para baixo) */
static float sil_v_rad  = 0.0f; /* Velocidade angular (rad/s) */
static float sil_a_pct  = 0.0f; /* Estado dinamico do atuador/turbina (%) */

/*
 * Calculo do controle PWM de equilibrio estatico u_eq(theta_deg)
 * Balanceamento de momento: c1*sin(th) - k_th*th + c3*(u - u0)^p = 0
 */
static inline float u_equilibrium(float th_deg)
{
    float sign = (th_deg >= 0.0f) ? 1.0f : -1.0f;
    float abs_th = fabsf(th_deg);
    if (abs_th <= 0.01f) return 0.0f;
    float th_rad = abs_th * ((float)M_PI / 180.0f);
    float restoring = -GB_C1 * sinf(th_rad) + GB_K_TH * th_rad;
    if (restoring <= 0.0f) return 0.0f;
    float u_val = GB_U0 + powf(restoring / GB_C3, 1.0f / GB_P_THRUST);
    if (u_val > 80.0f) u_val = 80.0f;
    return sign * u_val;
}

/*
 * Derivadas continuas do modelo fisico Grey-Box v9
 */
static inline void phys_derivs(float th, float v, float a, float u_cmd,
                               float *dth, float *dv, float *da)
{
    float u_motor = u_cmd;
    if (u_motor > 80.0f)  u_motor = 80.0f;
    if (u_motor < -80.0f) u_motor = -80.0f;

    float sign_a = (a >= 0.0f) ? 1.0f : -1.0f;
    float abs_a  = fabsf(a);
    float z = abs_a - GB_U0;
    float thrust = 0.0f;
    if (z > 0.0f) {
        thrust = sign_a * powf(z, GB_P_THRUST);
    }
    float aero = -GB_CA * v * fabsf(v);
    float fric = -GB_FC * tanhf(50.0f * v);

    *dth = v;
    *dv  = GB_C1 * sinf(th) - GB_K_TH * th + GB_C2 * v + aero + fric + GB_C3 * thrust;
    *da  = (u_motor - a) / GB_TAU_M;
}

/*
 * Simulador de Planta Fisica Grey-Box v9 (RK4 a 100 Hz com 2 subpassos de 5 ms)
 */
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

    float y_deg = th * (180.0f / (float)M_PI);
    sil_y_hist[0] = y_deg;
    return y_deg;
}

/* Alias para compatibilidade com rotinas legadas de simulacao */
static inline float simulate_narx(float u_cmd)
{
    return simulate_physical_plant(u_cmd);
}

#endif /* MODEL_PHYSICAL_H */
