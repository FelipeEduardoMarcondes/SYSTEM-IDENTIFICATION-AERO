import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.signal import savgol_filter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def carregar_semi_estatica(url):
    df = pd.read_csv(url)
    if 'referencia' in df.columns:
        df = df[df['referencia'] > 0]

    if 'u_pct' in df.columns:
        u_full = df['u_pct'].values.astype(np.float64)
    else:
        u_full = df['motor_percent'].values.astype(np.float64)
    y_full = df['angulo_deg'].values.astype(np.float64)

    t_raw = np.arange(len(y_full)) * 0.010
    y_rad = y_full * (np.pi / 180.0)
    u_norm = np.clip(u_full / 100.0, -1.0, 1.0)
    
    # Calcular velocidade e aceleração usando janela larga para atenuar ruído
    v_rad_s = savgol_filter(y_rad, 101, 3, deriv=1, delta=0.010) 
    a_rad_s2 = savgol_filter(v_rad_s, 101, 3, deriv=1, delta=0.010)
    
    # Filtrar para remover instantes de alta aceleração ou transientes sujos
    # Limiar empírico: aceleração < 0.2 rad/s^2 é bem próxima de estático
    mask = np.abs(a_rad_s2) < 0.2
    
    t_f = t_raw[mask]
    u_f = u_norm[mask]
    y_f = y_rad[mask]
    v_f = v_rad_s[mask]
    a_f = a_rad_s2[mask]
    
    print(f"Total original: {len(t_raw)} amostras.")
    print(f"Após filtro de aceleração (|a| < 0.2 rad/s²): {len(t_f)} amostras puramente semi-estáticas mantidas.")
    
    return (torch.tensor(t_f, dtype=torch.float32).unsqueeze(1),
            torch.tensor(u_f, dtype=torch.float32).unsqueeze(1),
            torch.tensor(y_f, dtype=torch.float32).unsqueeze(1),
            torch.tensor(v_f, dtype=torch.float32).unsqueeze(1),
            torch.tensor(a_f, dtype=torch.float32).unsqueeze(1))

class StribeckStaticModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.m1, self.L1 = 0.122, 0.39
        self.m2, self.L2 = 0.055, 0.347
        self.g = 9.81
        
        # Inércia como parâmetro livre (inicializada perto do valor que o NODE já identificou)
        self.log_J = nn.Parameter(torch.log(torch.tensor(0.03)))
        
        self.log_Gu_pos = nn.Parameter(torch.log(torch.tensor(1.0)))
        self.log_Tc_pos = nn.Parameter(torch.log(torch.tensor(0.01))) 
        self.log_Ts_pos = nn.Parameter(torch.log(torch.tensor(0.02))) 
        self.log_vs_pos = nn.Parameter(torch.log(torch.tensor(0.1)))  
        self.log_b_pos  = nn.Parameter(torch.log(torch.tensor(0.05)), requires_grad=False) # Viscosidade travada
        
        self.log_Gu_neg = nn.Parameter(torch.log(torch.tensor(1.0)))
        self.log_Tc_neg = nn.Parameter(torch.log(torch.tensor(0.01)))
        self.log_Ts_neg = nn.Parameter(torch.log(torch.tensor(0.02)))
        self.log_vs_neg = nn.Parameter(torch.log(torch.tensor(0.1)))
        self.log_b_neg  = nn.Parameter(torch.log(torch.tensor(0.05)), requires_grad=False) # Viscosidade travada

    def forward(self, theta, theta_dot, theta_ddot, u):
        J = torch.exp(self.log_J)
        
        Gu_pos = torch.exp(self.log_Gu_pos)
        Tc_pos = torch.exp(self.log_Tc_pos)
        Ts_pos = Tc_pos + torch.exp(self.log_Ts_pos) 
        vs_pos = torch.exp(self.log_vs_pos)
        b_pos  = torch.exp(self.log_b_pos)
        
        Gu_neg = torch.exp(self.log_Gu_neg)
        Tc_neg = torch.exp(self.log_Tc_neg)
        Ts_neg = Tc_neg + torch.exp(self.log_Ts_neg)
        vs_neg = torch.exp(self.log_vs_neg)
        b_neg  = torch.exp(self.log_b_neg)

        sigma_v = torch.sigmoid(50.0 * theta_dot)
        sigma_u = torch.sigmoid(50.0 * u)
        
        Gu = sigma_u * Gu_pos + (1 - sigma_u) * Gu_neg
        Tc = sigma_v * Tc_pos + (1 - sigma_v) * Tc_neg
        Ts = sigma_v * Ts_pos + (1 - sigma_v) * Ts_neg
        vs = sigma_v * vs_pos + (1 - sigma_v) * vs_neg
        b  = sigma_v * b_pos  + (1 - sigma_v) * b_neg

        # Torque Efetivo Real = Motor - Gravidade - Inercial
        motor_torque = Gu * u * torch.abs(u)
        gravity_torque = (self.m1 * self.L1 - self.m2 * self.L2) * self.g * torch.sin(theta)
        inertial_torque = J * theta_ddot
        
        torque_efetivo = motor_torque - gravity_torque - inertial_torque
        
        # Modelo Stribeck
        stribeck_effect = Tc + (Ts - Tc) * torch.exp(- (theta_dot / (vs + 1e-6))**2)
        sgn_v = torch.tanh(100.0 * theta_dot) 
        
        torque_atrito = stribeck_effect * sgn_v + b * theta_dot
        
        return torque_efetivo, torque_atrito

if __name__ == '__main__':
    url = "https://raw.githubusercontent.com/FelipeEduardoMarcondes/SYSTEM-IDENTIFICATION-AERO/main/experimentos/dados_curva_semi_estatica-2_0807_17-40.csv"
    
    print("1. Carregando dados quase-estáticos...")
    t_ten, u_ten, theta_ten, v_ten, a_ten = carregar_semi_estatica(url)
    
    t_ten = t_ten.to(device)
    u_ten = u_ten.to(device)
    theta_ten = theta_ten.to(device)
    v_ten = v_ten.to(device)
    a_ten = a_ten.to(device)
    
    model = StribeckStaticModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    epochs = 100000
    print("\n2. Otimizando Stribeck (com correção Inercial)...")
    for epoch in range(epochs+1):
        optimizer.zero_grad()
        
        torque_efetivo, torque_atrito = model(theta_ten, v_ten, a_ten, u_ten)
        loss = torch.mean((torque_efetivo - torque_atrito)**2)
        loss.backward()
        optimizer.step()
        
        if epoch % 5000 == 0:
            print(f"Epoch {epoch:5d} | Loss (MSE): {loss.item():.8f}")
            
    print("\n3. Parâmetros Ideais Encontrados:")
    print(f"Inércia (J): {torch.exp(model.log_J).item():.5f} kg.m²")
    print(f"\n--- Lado Positivo (Subida) ---")
    print(f"Gu : {torch.exp(model.log_Gu_pos).item():.4f}")
    print(f"Tc : {torch.exp(model.log_Tc_pos).item():.4f} N.m")
    print(f"Ts : {(torch.exp(model.log_Tc_pos) + torch.exp(model.log_Ts_pos)).item():.4f} N.m")
    print(f"vs : {torch.exp(model.log_vs_pos).item():.4f} rad/s")
    print(f"b  : {torch.exp(model.log_b_pos).item():.4f} N.m.s/rad")
    
    print(f"\n--- Lado Negativo (Descida) ---")
    print(f"Gu : {torch.exp(model.log_Gu_neg).item():.4f}")
    print(f"Tc : {torch.exp(model.log_Tc_neg).item():.4f} N.m")
    print(f"Ts : {(torch.exp(model.log_Tc_neg) + torch.exp(model.log_Ts_neg)).item():.4f} N.m")
    print(f"vs : {torch.exp(model.log_vs_neg).item():.4f} rad/s")
    print(f"b  : {torch.exp(model.log_b_neg).item():.4f} N.m.s/rad")
    
    # Avaliação do R²
    torque_efetivo, torque_atrito = model(theta_ten, v_ten, a_ten, u_ten)
    
    ss_res = torch.sum((torque_efetivo - torque_atrito)**2)
    ss_tot = torch.sum((torque_efetivo - torch.mean(torque_efetivo))**2)
    r2 = 1.0 - (ss_res / ss_tot)
    
    print(f"\nVariância explicada (R²) do ajuste: {r2.item():.4f}")
    
    # Gráfico
    print("\n4. Gerando gráfico stribeck_fit_v2.png...")
    v_np = v_ten.cpu().detach().numpy().flatten()
    torque_app_np = torque_efetivo.cpu().detach().numpy().flatten()
    torque_fric_np = torque_atrito.cpu().detach().numpy().flatten()
    
    plt.figure(figsize=(12, 7))
    plt.scatter(v_np, torque_app_np, s=2, alpha=0.4, label='Torque Efetivo (Motor - Grav. - J·a)', color='gray')
    
    idx = np.argsort(v_np)
    plt.plot(v_np[idx], torque_fric_np[idx], 'r-', linewidth=3.0, label='Curva de Stribeck Identificada')
    
    plt.title(f'Ajuste da Curva de Stribeck com Correção Inercial\nR² = {r2.item():.4f} | J = {torch.exp(model.log_J).item():.4f} kg.m²')
    plt.xlabel('Velocidade Angular (rad/s)')
    plt.ylabel('Torque de Atrito (N.m)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig('stribeck_fit_v3.png', dpi=150)
    print("Gráfico salvo como stribeck_fit_v3.png")
