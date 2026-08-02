import json

notebook = {
    "cells": [],
    "metadata": {},
    "nbformat": 4,
    "nbformat_minor": 4
}

def add_md(text):
    notebook["cells"].append({"cell_type": "markdown", "metadata": {}, "source": text})

def add_code(text):
    notebook["cells"].append({"cell_type": "code", "metadata": {}, "source": text, "execution_count": None, "outputs": []})

add_md([
    "# Identificação do Aeropêndulo - BLS Activity\n",
    "This notebook implements the requirements of the BLS activity, using the real-world dataset `sinal3_swept_sine_malha_fechada.csv` from the aeropendulum."
])

add_code([
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "from sklearn.metrics import mean_squared_error, r2_score\n",
    "from scipy.signal import welch"
])

add_md(["## 1. Load Real-World Data and Preprocess (Detrending)"])
add_code([
    "file_path = 'sinal3_swept_sine_malha_fechada.csv'\n",
    "df = pd.read_csv(file_path)\n",
    "\n",
    "t = df['tempo_ms'].values / 1000.0\n",
    "u_raw = df['u_pct'].values\n",
    "y_raw = df['angulo_deg'].values\n",
    "\n",
    "N = len(u_raw)\n",
    "split_idx = int(0.7 * N)\n",
    "\n",
    "u_train_raw, y_train_raw = u_raw[:split_idx], y_raw[:split_idx]\n",
    "u_test_raw, y_test_raw = u_raw[split_idx:], y_raw[split_idx:]\n",
    "t_train, t_test = t[:split_idx], t[split_idx:]\n",
    "\n",
    "# Detrending (removing operating point using training data mean)\n",
    "u_mean, y_mean = np.mean(u_train_raw), np.mean(y_train_raw)\n",
    "\n",
    "u_train = u_train_raw - u_mean\n",
    "y_train = y_train_raw - y_mean\n",
    "u_test = u_test_raw - u_mean\n",
    "y_test = y_test_raw - y_mean\n",
    "\n",
    "Ts = np.mean(np.diff(t))\n",
    "print(f'Sample time Ts: {Ts:.3f} s')"
])

add_md(["## 2. Implement a Regression Matrix Function"])
add_code([
    "def build_regression_matrix(u, y, na, nb):\n",
    "    \"\"\"\n",
    "    Constructs the regression matrix Phi and target vector Y.\n",
    "    \"\"\"\n",
    "    max_delay = max(na, nb)\n",
    "    N_data = len(y)\n",
    "    \n",
    "    Phi = []\n",
    "    Y_target = []\n",
    "    \n",
    "    for k in range(max_delay, N_data):\n",
    "        row = []\n",
    "        # Autoregressive part: -y(k-1) to -y(k-na)\n",
    "        for i in range(1, na + 1):\n",
    "            row.append(-y[k - i])\n",
    "            \n",
    "        # Exogenous part: u(k-1) to u(k-nb)\n",
    "        for i in range(1, nb + 1):\n",
    "            row.append(u[k - i])\n",
    "            \n",
    "        Phi.append(row)\n",
    "        Y_target.append(y[k])\n",
    "        \n",
    "    return np.array(Phi), np.array(Y_target)"
])

add_md(["## 3. Implement a Free-Run Simulation Function"])
add_code([
    "def free_run_simulation(theta, u, y_initial, na, nb):\n",
    "    \"\"\"\n",
    "    Performs a free-run simulation using estimated parameters.\n",
    "    \"\"\"\n",
    "    max_delay = max(na, nb)\n",
    "    y_sim = np.zeros(len(u))\n",
    "    \n",
    "    # Set initial conditions\n",
    "    for i in range(max_delay):\n",
    "        y_sim[i] = y_initial[i]\n",
    "        \n",
    "    for k in range(max_delay, len(u)):\n",
    "        reg_vector = []\n",
    "        # Use simulated past outputs\n",
    "        for i in range(1, na + 1):\n",
    "            reg_vector.append(-y_sim[k - i])\n",
    "            \n",
    "        # Use actual past inputs\n",
    "        for i in range(1, nb + 1):\n",
    "            reg_vector.append(u[k - i])\n",
    "            \n",
    "        y_sim[k] = np.dot(reg_vector, theta)\n",
    "        \n",
    "    return y_sim"
])

add_md(["## 4. Model Estimation and Evaluation (Comparing Orders)"])
add_code([
    "orders_to_test = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)]\n",
    "results = []\n",
    "models = {}\n",
    "\n",
    "for na, nb in orders_to_test:\n",
    "    # Training\n",
    "    Phi_TRA, Y_target_TRA = build_regression_matrix(y_train, u_train, na, nb)\n",
    "    theta_hat, _, _, _ = np.linalg.lstsq(Phi_TRA, Y_target_TRA, rcond=None)\n",
    "    \n",
    "    # Validation matrices\n",
    "    Phi_TEST, Y_target_TEST = build_regression_matrix(y_test, u_test, na, nb)\n",
    "    \n",
    "    # 1. One-Step-Ahead (OSA) Prediction on Test Set\n",
    "    yhat_OSA_TEST = Phi_TEST @ theta_hat\n",
    "    # Padding the beginning to align with original array length\n",
    "    yhat_OSA_full = np.concatenate((y_test[:max(na, nb)], yhat_OSA_TEST))\n",
    "    \n",
    "    # 2. Free-Run (FR) Simulation on Test Set\n",
    "    yhat_FR_full = free_run_simulation(theta_hat, u_test, y_test[:max(na, nb)], na, nb)\n",
    "    \n",
    "    # Calculate Metrics (only for the valid predicted portion avoiding initial conditions)\n",
    "    y_true_valid = y_test[max(na, nb):]\n",
    "    y_osa_valid = yhat_OSA_full[max(na, nb):]\n",
    "    y_fr_valid = yhat_FR_full[max(na, nb):]\n",
    "    \n",
    "    rmse_osa = np.sqrt(mean_squared_error(y_true_valid, y_osa_valid))\n",
    "    r2_osa = r2_score(y_true_valid, y_osa_valid)\n",
    "    \n",
    "    rmse_fr = np.sqrt(mean_squared_error(y_true_valid, y_fr_valid))\n",
    "    r2_fr = r2_score(y_true_valid, y_fr_valid)\n",
    "    \n",
    "    results.append({\n",
    "        'na': na, 'nb': nb,\n",
    "        'RMSE_OSA': rmse_osa, 'R2_OSA': r2_osa,\n",
    "        'RMSE_FR': rmse_fr, 'R2_FR': r2_fr\n",
    "    })\n",
    "    \n",
    "    # Store simulations to compare best/worst later\n",
    "    models[(na, nb)] = {\n",
    "        'y_osa': yhat_OSA_full + y_mean,\n",
    "        'y_fr': yhat_FR_full + y_mean,\n",
    "        'error_osa': y_test - yhat_OSA_full,\n",
    "        'error_fr': y_test - yhat_FR_full\n",
    "    }\n",
    "\n",
    "df_results = pd.DataFrame(results)\n",
    "display(df_results)"
])

add_md(["## 5. Visualizing the Best and Worst Models\n", "We sort the results by Free-Run R2 to pick the best and worst."])
add_code([
    "best_row = df_results.loc[df_results['R2_FR'].idxmax()]\n",
    "worst_row = df_results.loc[df_results['R2_FR'].idxmin()]\n",
    "\n",
    "best_order = (int(best_row['na']), int(best_row['nb']))\n",
    "worst_order = (int(worst_row['na']), int(worst_row['nb']))\n",
    "\n",
    "print(f\"Best Model: na={best_order[0]}, nb={best_order[1]} (R2 FR: {best_row['R2_FR']:.4f})\")\n",
    "print(f\"Worst Model: na={worst_order[0]}, nb={worst_order[1]} (R2 FR: {worst_row['R2_FR']:.4f})\")\n",
    "\n",
    "# Plotting Best Model Predictions\n",
    "plt.figure(figsize=(12, 5))\n",
    "plt.plot(t_test, y_test_raw, label='Real Measured Output')\n",
    "plt.plot(t_test, models[best_order]['y_osa'], label='OSA Prediction', linestyle='--')\n",
    "plt.plot(t_test, models[best_order]['y_fr'], label='Free-Run Simulation', linestyle=':')\n",
    "plt.title(f'Best Model Predictions (na={best_order[0]}, nb={best_order[1]})')\n",
    "plt.xlabel('Time (s)')\n",
    "plt.ylabel('Angle (deg)')\n",
    "plt.legend()\n",
    "plt.grid()\n",
    "plt.show()"
])

add_md(["## 6. Spectrum Comparison\n", "We compare the spectrum of the measured signal against predictions and errors."])
add_code([
    "def plot_spectrum(signals, labels, title):\n",
    "    plt.figure(figsize=(10, 5))\n",
    "    for sig, label in zip(signals, labels):\n",
    "        f, Pxx = welch(sig, fs=1/Ts, nperseg=1024)\n",
    "        plt.semilogy(f, Pxx, label=label)\n",
    "    plt.title(title)\n",
    "    plt.xlabel('Frequency (Hz)')\n",
    "    plt.ylabel('Power Spectral Density')\n",
    "    plt.legend()\n",
    "    plt.grid()\n",
    "    plt.show()\n",
    "\n",
    "best_y_fr = models[best_order]['y_fr'] - y_mean\n",
    "best_y_osa = models[best_order]['y_osa'] - y_mean\n",
    "best_err_fr = models[best_order]['error_fr']\n",
    "best_err_osa = models[best_order]['error_osa']\n",
    "\n",
    "worst_y_fr = models[worst_order]['y_fr'] - y_mean\n",
    "worst_y_osa = models[worst_order]['y_osa'] - y_mean\n",
    "\n",
    "# Comparing Best Model Predictions\n",
    "plot_spectrum([y_test, best_y_osa, best_y_fr], \n",
    "              ['Measured (y)', 'OSA Prediction', 'Free-Run Sim'], \n",
    "              'Spectrum: Measured vs Best Model Predictions')\n",
    "\n",
    "# Comparing Best Model Errors\n",
    "plot_spectrum([best_err_osa, best_err_fr], \n",
    "              ['OSA Error', 'Free-Run Error'], \n",
    "              'Spectrum: Best Model Errors')\n",
    "\n",
    "# Comparing Best vs Worst Free-Run\n",
    "plot_spectrum([y_test, best_y_fr, worst_y_fr], \n",
    "              ['Measured (y)', f'Best FR (na={best_order[0]})', f'Worst FR (na={worst_order[0]})'], \n",
    "              'Spectrum: Measured vs Best and Worst FR Simulations')"
])

with open('identificação-collab/3_bls_activity.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)
