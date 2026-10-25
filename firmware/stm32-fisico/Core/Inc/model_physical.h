/*
 * model_physical.h
 * ================
 * Simulador de Planta Fisica do Aeropendulo baseado no Modelo Grey-Box v6 (Turbina EDF Assimetrica)
 * Identificado em: ident_greybox_prof_multi_v6.py (Modelo Selecionado: turbina fixada)
 *
 * Equacoes de estado nao-lineares em tempo continuo (3a ordem com assimetria de turbina):
 *   d(theta)/dt = v
 *   d(v)/dt     = c1*sin(theta) + c2*v - c_aero(v)*v*|v| - Fc*tanh(50*v) + c3*(max(a - u0, 0))^2
 *   d(a)/dt     = (u_motor - a) / tau_eff(u_motor, a)
 *
 * Onde:
 *   c_aero(v) = c_aero_fwd * sigma_v + c_aero_bwd * (1 - sigma_v)    [Arrasto Direcional de Duto]
 *   tau_eff   = tau_dec + (tau_acc - tau_dec) * sigma_u              [Dinamica de Rotacao Assimetrica]
 *
 * Integrador numerico: Runge-Kutta 4a Ordem (RK4) com 2 subpassos de 5 ms a cada ciclo de 10 ms (100 Hz).
 */

#ifndef MODEL_PHYSICAL_H
#define MODEL_PHYSICAL_H

#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

/* ── Parametros fisicos do modelo Grey-Box v6 (Turbina EDF) ──────────────── */
#ifndef GB_C1
#define GB_C1          (-8.209674f)   /* rad/s^2 - torque gravitacional normalizado */
#endif

#ifndef GB_C2
#define GB_C2          (-0.684460f)   /* s^-1    - amortecimento viscoso mecanico */
#endif

#ifndef GB_C3
#define GB_C3          (0.00428000f)  /* rad/(s^2 * %^2) - ganho de empuxo quadratico */
#endif

#ifndef GB_FC
#define GB_FC          (0.300000f)    /* rad/s^2 - atrito seco de Coulomb no mancal */
#endif

#ifndef GB_CA_FWD
#define GB_CA_FWD      (0.700000f)    /* rad^-1  - arrasto com duto avancando */
#endif

#ifndef GB_CA_BWD
#define GB_CA_BWD      (0.850000f)    /* rad^-1  - arrasto com duto recuando */
#endif

#ifndef GB_TAU_ACC
#define GB_TAU_ACC     (0.0500f)      /* s       - constante de aceleracao motriz */
#endif

#ifndef GB_TAU_DEC
#define GB_TAU_DEC     (0.1200f)      /* s       - constante de desaceleracao roda-livre */
#endif

#ifndef GB_U0
#define GB_U0          (11.2637f)     /* %       - zona morta da turbina/esc */
#endif

/* ── Estados internos da planta fisica no modo SIL ──────────────────────── */
static float sil_th_rad = 0.0f; /* Posicao angular (rad, 0 = repouso vertical para baixo) */
static float sil_v_rad  = 0.0f; /* Velocidade angular (rad/s) */
static float sil_a_pct  = 0.0f; /* Estado dinamico do atuador/turbina (%) */

/*
 * Derivadas continuas do modelo fisico Grey-Box v6
 */
static inline void phys_derivs(float th, float v, float a, float u_cmd,
                               float *dth, float *dv, float *da)
{
    float u_motor = (u_cmd > 0.0f) ? ((u_cmd > 100.0f) ? 100.0f : u_cmd) : 0.0f;
    float z = a - GB_U0;
    float thrust = (z > 0.0f) ? (z * z) : 0.0f;
    float fric   = -GB_FC * tanhf(50.0f * v);

    /* Arrasto direcional do duto da turbina */
    float sigma_v = 0.5f * (1.0f + tanhf(20.0f * v));
    float ca_eff  = GB_CA_FWD * sigma_v + GB_CA_BWD * (1.0f - sigma_v);
    float aero    = -ca_eff * v * fabsf(v);

    /* Dinamica assimetrica de rotacao do rotor da turbina */
    float sigma_u = 0.5f * (1.0f + tanhf(10.0f * (u_motor - a)));
    float tau_eff = GB_TAU_DEC + (GB_TAU_ACC - GB_TAU_DEC) * sigma_u;

    *dth = v;
    *dv  = GB_C1 * sinf(th) + GB_C2 * v + aero + fric + GB_C3 * thrust;
    *da  = (u_motor - a) / (tau_eff > 0.01f ? tau_eff : 0.01f);
}

/*
 * Simulador de Planta Fisica Grey-Box v6 (RK4 a 100 Hz com 2 subpassos de 5 ms)
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

    return th * (180.0f / (float)M_PI);
}

static inline float simulate_narx(float u_cmd)
{
    return simulate_physical_plant(u_cmd);
}

#endif /* MODEL_PHYSICAL_H */
