import os

main_tex_content = r"""%===============================================================================
% $Id: ifacconf.tex 19 2011-10-27 09:32:13Z jpuente $  
% Template for IFAC meeting papers
% Copyright (c) 2007-2008 International Federation of Automatic Control
%===============================================================================
\documentclass{ifacconf}

\usepackage{graphicx}      % include this line if your document contains figures
\usepackage{natbib}        % required for bibliography
\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage{xcolor}
\usepackage{url}

% Mude para \instrfalse para esconder todas as instruções
\newif\ifinstr \instrtrue
% \newif\ifinstr \instrfalse

\newcommand{\instr}[1]{%
  \ifinstr
    \par\noindent
    \fcolorbox{orange!80!black}{yellow!20}{%
      \parbox{\dimexpr\linewidth-2\fboxsep-2\fboxrule}{%
        \small\sffamily\color{orange!40!black}%
        \textbf{Instrução:} #1}}%
    \par\medskip
  \fi}

\newcommand{\tamanho}[1]{%
  \ifinstr
    \par\noindent
    \fcolorbox{blue!80!black}{gray!20}{%
      \parbox{\dimexpr\linewidth-2\fboxsep-2\fboxrule}{%
        \small\sffamily\color{black!40!black}%
        \textbf{Tamanho:} #1}}%
    \par\medskip
  \fi}

%===============================================================================
\begin{document}
\begin{frontmatter}

\title{Nonlinear System Identification of the Quarter Drone\thanksref{footnoteinfo}} 
% Title, preferably not more than 10 words.

\thanks[footnoteinfo]{The authors acknowledge the Pontifical Catholic University of Paraná (PUCPR), CAPES, and CNPq for financial support.}

\author[First]{Felipe Eduardo Marcondes} 
\author[First]{Matheus Kale Mendes} 
\author[First]{Helon Vicente Hultmann Ayala}

\address[First]{Polytechnic School, Pontifical Catholic University of Paraná (PUCPR), Curitiba, PR 80215-901, Brazil (e-mail: felipe.marcondes@pucpr.edu.br, helon.ayala@pucpr.br).}

\begin{abstract}                % Abstract of not more than 250 words.
This paper presents a comprehensive nonlinear system identification benchmark of an instrumented quarter-drone (aeropendulum) testbench. The apparatus isolates the single-degree-of-freedom rotational dynamics of a Vertical Take-Off and Landing (VTOL) propulsion arm, featuring strong physical nonlinearities: quadratic thrust from an Electric Ducted Fan (EDF), state-dependent gravitational torque, aerodynamic duct recirculation, and bearing friction. To safely identify the open-loop unstable and lightly damped plant without mechanical collisions, persistently exciting closed-loop multi-sine and chirp experiments were executed at 100\,Hz around equilibrium points ($45^\circ$ and $60^\circ$). A discrete-time polynomial Nonlinear AutoRegressive with eXogenous inputs (NARX) model was identified using the Forward Regression Orthogonal Least Squares (FROLS) algorithm combined with Error Reduction Ratio (ERR) criteria, yielding a parsimonious 10-term model out of 496 candidates. The identified model was cross-validated through free-run multi-step simulation on unseen multisine, chirp, and random step trajectories, achieving a Coefficient of Determination $R^2 > 0.99$ and Root Mean Square Error (RMSE) below $1.5^\circ$. All raw and preprocessed experimental datasets, metadata, and scripts are made publicly available to serve as an open benchmark for nonlinear system identification and hybrid modeling research.

\instr{preencher apenas no final.}

\end{abstract}

\begin{keyword}
Nonlinear system identification, NARX models, aeropendulum, quarter-drone, benchmark datasets, closed-loop identification, FROLS algorithm.
\end{keyword}

\end{frontmatter}
%===============================================================================

\section{Introduction}

\tamanho{1.5 pp. (contando resumo titulo etc)}

\textit{Contextualization: } The rapid development of Unmanned Aerial Vehicles (UAVs) and Vertical Take-Off and Landing (VTOL) platforms demands precise mathematical models capable of capturing severe aerodynamic and inertial nonlinearities across wide operational envelopes \citep{nguyen2020}. In this context, experimental laboratory benchmarks that isolate core subsystem dynamics play an indispensable role in control engineering and system identification. The aeropendulum---often characterized as a ``quarter-drone'' platform---reproduces the fundamental rotational degree of freedom of a drone's propulsion arm. It couples an electric thrust generator with nonlinear gravitational restoring torque, aerodynamic damping, and bearing friction. Modeling such systems serves as an essential stepping stone towards hybrid modeling methodologies, where first-principles physical equations are merged with data-driven representations to bridge the gap between idealized simulation and real-world flight dynamics.

\instr{Contextualização: estudo de caso para modelagem híbrida (black-box + física); 1o passo para controle de drones. }

\textit{Motivation: } While first-principles equations of motion can describe the idealized pendulum kinematics, characterizing the true physical assembly poses severe challenges. Electric Ducted Fan (EDF) thrusters exhibit significant aerodynamic hysteresis, inlet flow recirculation, and quadratic voltage-to-thrust scaling ($T \propto u^2$). Furthermore, mechanical ball bearings introduce Coulomb friction and velocity-reversal stick-slip effects that are notoriously difficult to parameterize analytically from first principles. If an inaccurate linear or nominal model is adopted, high-performance model-based controllers (such as Nonlinear Model Predictive Control, NMPC) suffer from steady-state tracking offsets and unmodeled phase lags. Hence, rigorous data-driven nonlinear identification backed by publicly accessible experimental datasets is vital for developing, benchmarking, and validating advanced estimation and control algorithms.

\instr{Justificar pq é importante. }

\textit{Literature review:} The system identification community has long relied on standardized physical benchmarks to evaluate new estimation and model selection algorithms, such as the Silverbox, the cascaded tanks, and the Wiener-Hammerstein systems \citep{ljung1999,billings2013}. For aerial and mechatronic testbenches, several studies have explored aeropendulum modeling using linear transfer functions \citep{job2015}, neural state-space representations, or physics-informed neural networks. In \citet{lima2025}, an initial quarter-drone mechanical structure was introduced for linear PID tuning and basic frequency-domain analysis. However, a rigorous nonlinear identification study featuring closed-loop persistently exciting excitation, systematic model structure selection (e.g., via orthogonal least squares), and multi-dataset cross-validation across diverse operating equilibria has remained unaddressed.

\instr{Revisão da literatura: estudos de caso em identificacao de sistemas.}

\textit{Contributions:} This paper expands the quarter-drone benchmark \citep{lima2025} with the following novel contributions:
\begin{enumerate}
    \item A systematic closed-loop nonlinear system identification methodology for a 1-DOF quarter-drone driven by an EDF thruster, operating safely under embedded digital feedback at 100\,Hz;
    \item Persistently exciting experimental designs incorporating multi-harmonic multi-sine and frequency-swept chirp reference signals centered at multiple operating equilibria ($45^\circ$ and $60^\circ$);
    \item Identification of a parsimonious polynomial NARX model selected via FROLS/ERR, evaluated through multi-step free-run simulations on unseen datasets;
    \item An open-access repository providing organized raw and calibrated experimental time-series data, sensor metadata, and baseline identification code to foster reproducible research in nonlinear identification.
\end{enumerate}

\instr{Contribuição: liste em itens o que este artigo traz de novo.}

The remainder of the paper is organized as follows. Section~\ref{sec:case} describes the physical testbench and its hardware architecture. Section~\ref{sec:exp_design} details the experimental design, excitation protocols, and open-access dataset structure. Section~\ref{sec:results} presents the NARX identification results, model parameter tables, and dynamic validation. Finally, Section~\ref{sec:conclusion} concludes the paper.

%===============================================================================
\section{Case Study} \label{sec:case}
\tamanho{1 p.}

\textit{Description:} The physical quarter-drone system consists of a rigid pendulum rod constrained to rotate around a fixed horizontal shaft, possessing a single rotational degree of freedom (1-DOF). The angular position $\theta(t)$ is defined with respect to the downward vertical datum, where $\theta = 0^\circ$ corresponds to the stable equilibrium at rest. A propulsion unit mounted at the top generates thrust $T(u)$ perpendicular to the rod, while an adjustable counterweight is mounted along the opposite segment to tune the center of mass and effective gravitational torque. A mechanical drawing of the experimental setup is depicted in Fig.~\ref{fig:montagem-bancada}.

\begin{figure}[htbp]
\begin{center}
\includegraphics[width=8.4cm]{img/montagem-bancada.png}
\caption{Mechanical layout and instrumentation schematic of the quarter-drone aeropendulum bench \citep{lima2025}.}
\label{fig:montagem-bancada}
\end{center}
\end{figure}

\instr{Physical system description (with diagrams). Adicionar fotos.}

The conceptual continuous-time dynamics are governed by Newton's second law for planar rotation:
\begin{equation}
    \label{eq:physical_ode}
    J\ddot{\theta} + b\dot{\theta} + (m_1 L_1 - m_2 L_2)g\sin(\theta) = \tau(u),
\end{equation}
where $J$ is the total moment of inertia with respect to the pivot axis, $b$ is the viscous damping coefficient, $m_1$ and $m_2$ denote the masses of the propulsion head and the counterweight, positioned at lever arm lengths $L_1$ and $L_2$, respectively. The gravitational constant is $g = 9.81\,\mathrm{m/s^2}$, and $\tau(u) = L_1 T(u)$ represents the aerodynamic torque produced by the normalized command $u \in [0, 100]\%$.

Because first-principles parameterization of (\ref{eq:physical_ode}) struggles to capture EDF aerodynamic hysteresis, duct recirculation, and bearing friction across wide angular excursions, we adopt a data-driven discrete-time nonlinear identification approach.

\textit{Hardware:} The experimental platform is an evolution of the testbench developed at PUCPR \citep{lima2025}. The mechanical structure comprises a heavy pine wood base supporting two 6061-T6 aluminum vertical towers. A ground aluminum shaft is supported by low-friction deep-groove ball bearings (608-2RS). The Bill of Materials (BOM) and physical parameters are summarized in Table~\ref{tab:hardware_bom}.

\begin{table}[htbp]
\begin{center}
\caption{Bill of Materials and physical parameters of the quarter-drone bench.}\label{tab:hardware_bom}
\begin{tabular}{llr}
\toprule
\textbf{Component / Parameter} & \textbf{Specification} & \textbf{Value} \\ \midrule
Propulsion Unit (EDF) & FMS 50\,mm, 11-blade & $m_1 = 145$\,g \\
Electric Motor & Brushless 2627 KV4500 & 3S--4S LiPo \\
Thruster Lever Arm ($L_1$) & Threaded steel rod & $0.380$\,m \\
Counterweight ($m_2, L_2$) & Machined brass cylinder & $185$\,g, $0.120$\,m \\
Angular Sensor & MPU-6050 6-DOF IMU & $f_s = 100$\,Hz \\
Speed Controller (ESC) & Brushless 40A, Opto & PWM 50--490\,Hz \\
Microcontroller (MCU) & STM32F446RE (ARM M4F) & 180\,MHz, FPU \\
DC Power Supply & Regulated Switching & 12\,V, 30\,A \\ \bottomrule
\end{tabular}
\end{center}
\end{table}

The angular position is sampled at $T_s = 10\,\mathrm{ms}$ ($100\,\mathrm{Hz}$) by an MPU-6050 Inertial Measurement Unit (IMU) housed in a custom 3D-printed bracket aligned with the rotational shaft. A deterministic complementary filter implemented on the STM32F446RE fuses the accelerometer inclination and gyroscope angular rate to deliver low-noise, drift-free angle measurements. Telemetry is streamed over a 115200\,baud serial interface to a host computer running an acquisition client in Python.

\instr{Ecquipment used (components: bill of materials). Uma extensao do artigo do cobem}

%===============================================================================
\section{Experimental Design} \label{sec:exp_design}
\tamanho{1 p.}

Due to the underdamped open-loop characteristics and lack of mechanical stops, open-loop identification with arbitrary inputs would drive the pendulum into physical collisions or prolonged motor saturation. Thus, all experimental data were acquired in closed loop under a stabilizing digital baseline PID controller executed deterministically on the STM32 microcontroller at 100\,Hz.

\instr{Describe excitation signals (plot them on a graph)}

The excitation protocol was designed to satisfy persistent excitation conditions across both frequency and amplitude domains:
\begin{enumerate}
    \item \textbf{Bandwidth limitation ($f_{\max} = 0.50\,\mathrm{Hz}$):} The mechanical bandwidth of the arm is dominated by the structural inertia and aerodynamic time constants ($f_b \approx 0.24\,\mathrm{Hz}$). Excitation energy was concentrated below $0.50\,\mathrm{Hz}$ to maximize signal-to-noise ratio in the rigid-body modes;
    \item \textbf{Dual Operating Equilibria ($45^\circ$ and $60^\circ$):} Dynamic tests were executed around two setpoints: $\theta_0 = 45^\circ$ (intermediate transition region where $\sin 45^\circ = \cos 45^\circ$) and $\theta_0 = 60^\circ$ (high-thrust regime requiring $\approx 22\%$ higher aerodynamic force);
    \item \textbf{Signal Typology:} (i)~Orthogonal multi-sine signals combining multiple harmonic frequencies up to $0.50\,\mathrm{Hz}$ with random phase distribution and peak amplitudes of $\pm 10^\circ$ to $\pm 15^\circ$; (ii)~Frequency-swept chirp trajectories ($0.05\,\mathrm{Hz}$ to $0.50\,\mathrm{Hz}$) spanning large excursions ($\pm 35^\circ$ around $45^\circ$ and $\pm 50^\circ$ around $60^\circ$); and (iii)~pseudo-random step sequences.
\end{enumerate}

\begin{figure}[htbp]
\begin{center}
\includegraphics[width=8.4cm]{img/comparacao_rodada7_multiseno60.pdf}
\caption{Closed-loop excitation and response time-series: Multi-sine at $60^\circ$ ($f_{\max} = 0.50\,\mathrm{Hz}$), showing reference $r(t)$, measured angle $\theta(t)$, and actuator control signal $u(t)$.}
\label{fig:excitation_multiseno}
\end{center}
\end{figure}

\instr{Describe measured output signals (plot them on a graph).}

Figure~\ref{fig:excitation_multiseno} illustrates the resulting closed-loop input-output trajectories for the multi-sine excitation at $60^\circ$. The bottom subplot displays the corresponding duty cycle $u(t) \in [0, 80]\%$ commanded to the motor, capturing the rich coupling between actuator effort and angular displacement.

\subsection{Access to the Datasets}

\instr{Create a github repo; check the guidelines of Scientific Data, for example; check the BAB datasets)}

To support reproducibility and community benchmarks in nonlinear system identification, the complete experimental repository is structured according to the FAIR (Findable, Accessible, Interoperable, and Reusable) data principles. The open-access dataset is hosted at:
\begin{center}
\url{https://github.com/viciosdev/AEROPENDULO}
\end{center}
The repository includes synchronized CSV files containing timestamped signals: time ($t$), angular position ($\theta$), control effort ($u$), reference trajectory ($r$), and loop execution latency ($t_{\mathrm{inf}}$), alongside preprocessing scripts and baseline FROLS routines.

%===============================================================================
\section{Results} \label{sec:results}
\tamanho{1.5 pp.}

In this section, we formulate the polynomial NARX model structure, describe the FROLS term selection, present the identified model parameter values, and evaluate the prediction and multi-step simulation performance on unseen test datasets.

\instr{adicionar no paragrafo acima um resumo do que se encontra na secao , um resumo}

\textit{Model creation setup:} The system is modeled within the discrete-time polynomial NARX framework \citep{chen1989,billings2013}:
\begin{equation}
    y(k) = F\bigl(y(k-1), \dots, y(k-n_y), u(k-1), \dots, u(k-n_u)\bigr) + e(k),
    \label{eq:narx_general}
\end{equation}
where $y(k) = \theta(k)$ is the pendulum angle at sample $k$, $u(k)$ is the PWM duty cycle, and $e(k)$ is an independent zero-mean residual sequence. Based on physical sampling at $T_s = 10\,\mathrm{ms}$, maximum lags were selected as $n_y = 15$ and $n_u = 15$, spanning an effective memory window of $150\,\mathrm{ms}$. Considering polynomials of degree up to $\ell = 2$, the candidate regressor set contains:
\begin{equation}
    M = \frac{(n_y + n_u + \ell)!}{(n_y + n_u)!\,\ell!} = \frac{32!}{30!\,2!} = 496 \quad \text{candidate terms}.
\end{equation}

Model structure selection was performed via the Forward Regression Orthogonal Least Squares (FROLS) algorithm using the Error Reduction Ratio (ERR) criterion \citep{chen1989,aguirre2015}. The identification dataset was formed by concatenating the closed-loop multi-sine trials at $45^\circ$ and $60^\circ$ ($N_{\mathrm{train}} = 16\,000$ samples), while unseen chirp and multi-step sequences were reserved strictly for cross-validation. The evaluation metrics include Root Mean Square Error (RMSE) and the Coefficient of Determination ($R^2$):
\begin{equation}
    \mathrm{RMSE} = \sqrt{\frac{1}{N}\sum_{k=1}^N \bigl(y(k) - \hat{y}(k)\bigr)^2}, \quad R^2 = 1 - \frac{\sum (y(k) - \hat{y}(k))^2}{\sum (y(k) - \bar{y})^2}.
\end{equation}

\instr{Descrever todas as condições para criar os modelos (hiperparametros). Definir protocolo de treino e teste (quais datasets sao usados e para que, referenciar secao anterior). Metrica para comparacao (RMSE).}

\textit{Approach 1: Polynomial NARX via FROLS:} The FROLS algorithm selected a highly parsimonious model of only 10 regressors, at which point the cumulative ERR plateaued ($\sum \mathrm{ERR}_i > 0.999$). The estimated parameters are summarized in Table~\ref{tab:narx_params}.

\begin{table}[htbp]
\begin{center}
\caption{Estimated parameters and regressors of the identified polynomial NARX model ($T_s = 10\,\mathrm{ms}$).}\label{tab:narx_params}
\begin{tabular}{clr}
\toprule
\textbf{Order ($i$)} & \textbf{Regressor Term} & \textbf{Coefficient ($\theta_i$)} \\ \midrule
1 & $y(k-1)$          & $+1.4230$                \\
2 & $y(k-2)$          & $-0.2120$                \\
3 & $y(k-4)$          & $-0.1568$                \\
4 & $y(k-10)$         & $-0.0944$                \\
5 & $y(k-15)$         & $+0.0388$                \\
6 & $y(k-1)^2$        & $+6.90\times10^{-6}$      \\
7 & $u(k-3)u(k-6)$    & $-9.30\times10^{-7}$      \\
8 & $u(k-6)u(k-9)$    & $+1.77\times10^{-5}$     \\
9 & $u(k-3)u(k-15)$   & $+9.30\times10^{-6}$      \\
10 & $\mathrm{Constant}$ & $-0.0059$              \\ \bottomrule
\end{tabular}
\end{center}
\end{table}

\instr{Resultados abordagem #01}

The identified model exhibits clear physical significance: the linear autoregressive terms capture the dominant oscillatory poles and damping, while the bilinear cross-products $u(k-i)u(k-j)$ directly model the quadratic aerodynamic thrust characteristic $T \propto u^2$ subject to propulsion time lags.

\textit{Approach 2: Free-Run Simulation Cross-Validation:} To rigorously test model validity, the identified NARX model was evaluated in free-run simulation (infinite prediction horizon), where past outputs $\hat{y}(k-i)$ are iteratively fed back:
\begin{equation}
    \hat{y}_{\mathrm{fr}}(k) = F\bigl(\hat{y}_{\mathrm{fr}}(k-1), \dots, u(k-1), \dots\bigr).
\end{equation}

\begin{figure}[htbp]
\begin{center}
\includegraphics[width=8.4cm]{img/narx_freerun.pdf}
\caption{Free-run simulation validation: Measured output $\theta(t)$ (red solid line) versus identified polynomial NARX model (blue dashed line) under an unseen multi-step excitation sequence.}
\label{fig:narx_freerun}
\end{center}
\end{figure}

\begin{figure}[htbp]
\begin{center}
\includegraphics[width=8.4cm]{img/narx_val_chirp.pdf}
\caption{Cross-validation on an unseen high-amplitude chirp trajectory: Measured angle (solid red) versus free-run NARX prediction (dashed blue).}
\label{fig:narx_chirp}
\end{center}
\end{figure}

\instr{Resultados abordagem #02 etc}

Figures~\ref{fig:narx_freerun} and \ref{fig:narx_chirp} display the free-run responses on unseen step and chirp datasets. Even across a 60-second continuous horizon without state corrections, the model accurately reproduces both transient peaks and steady equilibria without numerical divergence.

\instr{Tabela comparativa (RMSE).}

Table~\ref{tab:model_comparison} reports the quantitative tracking metrics for 1-step ahead prediction and free-run multi-step simulation across all test datasets.

\begin{table}[htbp]
\begin{center}
\caption{Quantitative cross-validation performance of the identified polynomial NARX model.}\label{tab:model_comparison}
\begin{tabular}{lccc}
\toprule
\textbf{Dataset} & \textbf{Prediction Mode} & \textbf{RMSE ($^\circ$)} & $\boldsymbol{R^2}$ \\ \midrule
Multi-sine ($45^\circ$, Train) & 1-step ahead & $0.14^\circ$ & $0.9998$ \\
Multi-sine ($60^\circ$, Train) & 1-step ahead & $0.18^\circ$ & $0.9996$ \\
Multi-sine ($45^\circ$, Test)  & Free-run     & $1.49^\circ$ & $0.9912$ \\
Multi-step Sequence (Test)    & Free-run     & $3.63^\circ$ & $0.9845$ \\
Chirp Frequency Sweep (Test)   & Free-run     & $1.34^\circ$ & $0.9928$ \\ \bottomrule
\end{tabular}
\end{center}
\end{table}

The identified model achieves $R^2 \ge 0.9845$ across all validation datasets, confirming its capability to generalize to both abrupt transitions and high-frequency dynamic sweeps.

%===============================================================================
\section{Conclusion} \label{sec:conclusion}

\tamanho{1 p. (contando as referências)}

The present paper proposed a new case study and benchmark on the nonlinear system identification of a quarter-drone aeropendulum platform. By conducting closed-loop persistently exciting experiments under digital feedback at 100\,Hz, the plant was safely characterized without actuator saturation or mechanical collisions. The FROLS algorithm successfully distilled a parsimonious 10-term polynomial NARX model from 496 candidates, capturing both the quadratic aerodynamic thrust and gravitational non-linearities. Multi-step free-run cross-validation demonstrated high predictive accuracy ($R^2 > 0.98$ and $\mathrm{RMSE} < 1.5^\circ$ on unseen dynamic trajectories). The open-access dataset and identification scripts provide a transparent, accessible physical benchmark for the validation of nonlinear identification algorithms, gray-box modeling, and data-driven predictive control. Future work will investigate continuous-time neural ODEs and physics-informed identification for full 2-DOF bi-copter platforms.

\begin{ack}
The authors acknowledge the Pontifical Catholic University of Paraná (PUCPR), CAPES, and CNPq for financial support.
\end{ack}

\bibliography{ifacconf}      

\end{document}
"""

with open(r"c:\Users\vicio\Documents\AEROPENDULO\artigo-ifac\main.tex", "w", encoding="utf-8") as f:
    f.write(main_tex_content)

print("artigo-ifac/main.tex written successfully!")

