import numpy as np
import time
import os
import matplotlib
matplotlib.use("Agg")   # use non-GUI backend
import matplotlib.pyplot as plt
from simulation_and_control import pb, MotorCommands, PinWrapper, feedback_lin_ctrl, SinusoidalReference 
import xml.etree.ElementTree as ET

def rpy_to_R(roll, pitch, yaw):
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[cy, -sy, 0],[sy, cy, 0],[0, 0, 1]])
    Ry = np.array([[cp, 0, sp],[0, 1, 0],[-sp, 0, cp]])
    Rx = np.array([[1, 0, 0],[0, cr, -sr],[0, sr, cr]])
    return Rz @ Ry @ Rx  # XYZ fixed-axis roll->pitch->yaw

def extract_link10_from_urdf(urdf_path, link_name):
    """Return [m, m*cx, m*cy, m*cz, Ixx, Ixy, Ixz, Iyy, Iyz, Izz] in the LINK frame."""
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    link = None
    for lk in root.findall('link'):
        if lk.get('name') == link_name:
            link = lk
            break
    if link is None:
        raise ValueError(f"Link '{link_name}' not found in URDF.")

    inertial = link.find('inertial')
    mass_tag = inertial.find('mass')
    m = float(mass_tag.attrib['value'])

    origin = inertial.find('origin')
    if origin is not None:
        xyz = [float(v) for v in origin.attrib.get('xyz', '0 0 0').split()]
        rpy = [float(v) for v in origin.attrib.get('rpy', '0 0 0').split()]
    else:
        xyz = [0.0, 0.0, 0.0]; rpy = [0.0, 0.0, 0.0]
    cx, cy, cz = xyz
    R_li = rpy_to_R(*rpy)

    I = inertial.find('inertia').attrib
    I_in = np.array([[float(I['ixx']), float(I['ixy']), float(I['ixz'])],
                     [float(I['ixy']), float(I['iyy']), float(I['iyz'])],
                     [float(I['ixz']), float(I['iyz']), float(I['izz'])]])
    I_link = R_li @ I_in @ R_li.T
    Ixx, Ixy, Ixz = I_link[0,0], I_link[0,1], I_link[0,2]
    Iyy, Iyz, Izz = I_link[1,1], I_link[1,2], I_link[2,2]
    return np.array([m, m*cx, m*cy, m*cz, Ixx, Ixy, Ixz, Iyy, Iyz, Izz], dtype=float)

def build_full_param_vector(n_dof, per_link_params):
    a_full = np.zeros(10 * n_dof)
    for link_idx, a10 in per_link_params.items():
        s = 10*(link_idx-1); a_full[s:s+10] = np.asarray(a10).reshape(10,)
    return a_full


# ---------- helpers (numerically-stable LS + metrics) ----------
def pinv_stable(A, lam=1e-8):
    """Tikhonov-regularized pseudo-inverse for stability."""
    AtA = A.T @ A
    n = AtA.shape[0]
    return np.linalg.solve(AtA + lam * np.eye(n), A.T)

def regression_metrics(y, y_hat, p):
    """Return RSS, TSS, R2, R2_adj, F-stat."""
    M = y.size
    residual = y - y_hat
    RSS = float(residual @ residual)
    TSS = float(((y - y.mean()) ** 2).sum()) if M > 0 else np.nan
    R2 = 1.0 - (RSS / TSS if TSS > 0 else np.nan)
    R2_adj = 1.0 - ((RSS / max(M - p, 1)) / (TSS / max(M - 1, 1)) if TSS > 0 else np.nan)
    num = (TSS - RSS) / p if TSS > RSS else 0.0
    den = (RSS / max(M - p - 1, 1))
    F = num / den if den > 0 else np.nan
    return {"RSS": RSS, "TSS": TSS, "R2": R2, "R2_adj": R2_adj, "F": F, "M": M, "p": p}

def main():
    # Configuration for the simulation
    conf_file_name = "pandaconfig.json"  # Configuration file for the robot
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    sim = pb.SimInterface(conf_file_name, conf_file_path_ext=cur_dir)  # Initialize simulation interface

    # Get active joint names from the simulation
    ext_names = sim.getNameActiveJoints()
    ext_names = np.expand_dims(np.array(ext_names), axis=0)  # Adjust the shape for compatibility

    source_names = ["pybullet"]  # Define the source for dynamic modeling

    # Create a dynamic model of the robot
    dyn_model = PinWrapper(conf_file_name, "pybullet", ext_names, source_names, False, 0, cur_dir)
    num_joints = dyn_model.getNumberofActuatedJoints()

    # Print initial joint angles
    print(f"Initial joint angles: {sim.GetInitMotorAngles()}")

    # Sinusoidal reference
    amplitudes = [np.pi/4, np.pi/6, np.pi/4, np.pi/4, np.pi/4, np.pi/4, np.pi/4]
    frequencies = [0.4, 0.5, 0.4, 0.4, 0.4, 0.4, 0.4]
    amplitude = np.array(amplitudes)
    frequency = np.array(frequencies)
    ref = SinusoidalReference(amplitude, frequency, sim.GetInitMotorAngles())
    
    # Simulation parameters
    time_step = sim.GetTimeStep()
    current_time = 0.0
    max_time = 10.0  # seconds
    
    # Command and control loop
    cmd = MotorCommands()
    kp = 1000
    kd = 100

    # Initialize data storage
    tau_mes_all = []      # list of (n,)
    regressor_all = []    # list of (n, 10n)
    times = []

    # Data collection loop
    while current_time < max_time:
        # Measure current state
        q_mes = sim.GetMotorAngles(0)
        qd_mes = sim.GetMotorVelocities(0)
        qdd_mes = sim.ComputeMotorAccelerationTMinusOne(0)
        
        # Compute sinusoidal reference trajectory
        q_d, qd_d = ref.get_values(current_time)  # Desired position and velocity
        
        # Control command
        tau_cmd = feedback_lin_ctrl(dyn_model, q_mes, qd_mes, q_d, qd_d, kp, kd)
        cmd.SetControlCmd(tau_cmd, ["torque"]*num_joints)
        sim.Step(cmd, "torque")

        # Get measured torque
        tau_mes = sim.GetMotorTorques(0)

        if dyn_model.visualizer: 
            for index in range(len(sim.bot)):  # Conditionally display the robot model
                q = sim.GetMotorAngles(index)
                dyn_model.DisplayModel(q)

        # Exit logic with 'q' key
        keys = sim.GetPyBulletClient().getKeyboardEvents()
        qKey = ord('q')
        if qKey in keys and keys[qKey] and sim.GetPyBulletClient().KEY_WAS_TRIGGERED:
            break

        # ---------- TODO #1: Compute regressor and store it ----------
        Y_t = dyn_model.ComputeDynamicRegressor(q_mes, qd_mes, qdd_mes)  # shape: (n, 10n)
        regressor_all.append(Y_t)
        tau_mes_all.append(np.asarray(tau_mes))
        times.append(current_time)
        
        current_time += time_step
        print(f"Current time in seconds: {current_time:.2f}")

    # Safety check: need at least a few samples
    if len(regressor_all) == 0:
        print("No data collected. Exiting.")
        return

    # ----- Warm-up by time (drop ~1s) instead of hard 1000 -----
    N_warm = int(1.0 / time_step)  # drop first 1 second
    regressor_all = regressor_all[N_warm:]
    tau_mes_all   = tau_mes_all[N_warm:]
    times         = times[N_warm:]

    n = num_joints

    # ----- Build known a_full for links 1..6 from your URDF -----
    urdf_path = "/home/terry/LAB_SESSION_COMP_0245_2025_PUBLIC/week_1_2/models/panda_description/panda.urdf"     # adjust if needed
    link_names = ["panda_link1","panda_link2","panda_link3",
                "panda_link4","panda_link5","panda_link6","panda_link7"]

    per_link_params_known = {}
    for i in range(1, 7):  # links 1..6 known
        per_link_params_known[i] = extract_link10_from_urdf(urdf_path, link_names[i-1])

    a_known_full = build_full_param_vector(n, per_link_params_known)

    # ----- Form the reduced system y = X a7 -----
    s7, e7 = 10*(7-1), 10*(7-1)+10  # slice for link-7 columns
    X_blocks, y_blocks = [], []

    for Y_t, tau_t in zip(regressor_all, tau_mes_all):
        y_t = tau_t - (Y_t @ a_known_full)   # subtract links 1..6
        X_t = Y_t[:, s7:e7]                  # keep only link-7 columns
        X_blocks.append(X_t)
        y_blocks.append(y_t)

    X = np.vstack(X_blocks)                  # (T*n, 10)
    y = np.hstack(y_blocks)                  # (T*n,)

    condX = np.linalg.cond(X)
    print(f"cond(X) = {condX:.3e}")

    # ----- Solve with small ridge for stability -----
    lam = 1e-6 if condX < 1e6 else 1e-4
    a7_hat = np.linalg.solve(X.T @ X + lam*np.eye(10), X.T @ y)

    print("\nEstimated parameters for link 7 (reduced LS):")
    print(a7_hat)

    # ----- Compare to URDF truth for link 7 -----
    a7_true = extract_link10_from_urdf(urdf_path, link_names[6])
    abs_err = np.abs(a7_hat - a7_true)
    rel_err = np.where(a7_true != 0, 100*np.abs((a7_hat - a7_true)/a7_true), np.nan)
    print("\na7_true:", a7_true)
    print("a7_hat :", a7_hat)
    print("abs err:", abs_err)
    print("rel %  :", rel_err)

    # ----- Rebuild full vector with estimated link-7 for predictions/metrics/plots -----
    a_full_hat = a_known_full.copy()
    a_full_hat[s7:e7] = a7_hat

    # Metrics treat only 10 free params (link 7)
    Y_stack = np.vstack(regressor_all)        # (T*n, 10n)
    tau_stack = np.hstack(tau_mes_all)        # (T*n,)
    tau_hat_stack = Y_stack @ a_full_hat
    mets = regression_metrics(tau_stack, tau_hat_stack, p=10)
    print("\nMetrics using reduced model (10 params for link 7):")
    for k, v in mets.items():
        print(f"  {k}: {v}")

    # ---------- Plot torque prediction error for each joint ----------
    u_hat_time = [Y @ a_full_hat for Y in regressor_all]   # list of (n,)
    tau_array  = np.vstack(tau_mes_all)                    # (T, n)
    uhat_array = np.vstack(u_hat_time)                     # (T, n)
    err_array  = uhat_array - tau_array                    # (T, n)

    T = tau_array.shape[0]
    t = np.array(times[:T])

    fig, axes = plt.subplots(n, 1, figsize=(10, 2.2*n), sharex=True)
    if n == 1:
        axes = [axes]
    for j in range(n):
        ax = axes[j]
        ax.plot(t, tau_array[:, j], label=f"τ{j+1} measured")
        ax.plot(t, uhat_array[:, j], linestyle="--", label=f"τ{j+1} predicted")
        ax.plot(t, err_array[:, j], linestyle=":", label=f"error τ{j+1}")
        ax.set_ylabel(f"Joint {j+1}")
        ax.grid(True, alpha=0.3)
        if j == 0:
            ax.legend(loc="upper right", ncol=3, fontsize=8)
    axes[-1].set_xlabel("Time [s]")
    fig.suptitle("Torque prediction & error (reduced model: link 7 only)")

    # =========================
    # === PART 2: ALL LINKS UNKNOWN (append-only)
    # =========================

    print("\n" + "="*72)
    print("PART 2: Estimating ALL links (1..7) parameters simultaneously")
    print("="*72)

    # Stack after warm-up (we already trimmed regressor_all/tau_mes_all/times earlier)
    Y_all = np.vstack(regressor_all)            # shape: (T*n, 10n)
    tau_all = np.hstack(tau_mes_all)            # shape: (T*n,)
    M_all, p_all = Y_all.shape                  # p_all = 10*n

    # Conditioning diagnosis
    condY = np.linalg.cond(Y_all)
    rankY = np.linalg.matrix_rank(Y_all)
    print(f"[diag] cond(Y_all) = {condY:.3e}   rank(Y_all) = {rankY}/{p_all}")

    # Solve full 70-parameter LS with small ridge (auto-tuned from conditioning)
    lam_all = 1e-6 if condY < 1e6 else 1e-4
    XtX_all = Y_all.T @ Y_all
    a_hat_all = np.linalg.solve(XtX_all + lam_all*np.eye(p_all), Y_all.T @ tau_all)

    print("\nEstimated full parameter vector a_hat_all (len={}):".format(a_hat_all.size))
    # print(a_hat_all)  # uncomment if you want the raw 70 numbers

    # Metrics (now the model has p_all free params)
    tau_pred_all = Y_all @ a_hat_all
    mets_all = regression_metrics(tau_all, tau_pred_all, p=p_all)
    print("\nOverall metrics (ALL links unknown):")
    for k, v in mets_all.items():
        print(f"  {k}: {v}")

    # ---- Compare against URDF truth per link ----
    true_params_all = {}
    for i in range(1, 8):
        true_params_all[i] = extract_link10_from_urdf(urdf_path, link_names[i-1])
    a_true_all = build_full_param_vector(n, true_params_all)

    abs_err_all = np.abs(a_hat_all - a_true_all)
    rel_err_all = np.where(a_true_all != 0, 100*np.abs((a_hat_all - a_true_all)/a_true_all), np.nan)

    print("\nPer-link error summary (|abs| and rel % by link, grouped in chunks of 10):")
    for i in range(1, 8):
        s = 10*(i-1); e = 10*i
        abs_err_i = abs_err_all[s:e]
        rel_err_i = rel_err_all[s:e]
        print(f"\nLink {i}:")
        print("  abs error: ", np.round(abs_err_i, 6))
        print("  rel  %   : ", np.round(rel_err_i, 2))

    # ---- (Optional) 95% CIs for parameters (may be noisy if Y is ill-conditioned) ----
    sigma2_all = float(((tau_all - tau_pred_all)**2).sum()) / max(M_all - p_all, 1)
    XtX_pinv = np.linalg.pinv(XtX_all)
    cov_all = sigma2_all * XtX_pinv
    stderr_all = np.sqrt(np.clip(np.diag(cov_all), 0, np.inf))
    print("\nNoise variance estimate (sigma^2):", sigma2_all)
    print("Typical stderr (median over 70 params):", float(np.median(stderr_all)))

    # ---- Quick notes you can paste into the report ----
    print("\nREPORT NOTES / CHALLENGES (Part 2):")
    print("- Joint estimation couples parameters across links; columns in Y are correlated.")
    print("- Conditioning (cond ~ {:.2e}) and rank {} can limit identifiability; ridge λ = {} used."
        .format(condY, rankY, lam_all))
    print("- Rich/aperiodic excitation is critical; incommensurate joint frequencies reduce collinearity.")
    print("- Acceleration noise biases inertia estimates; smoothing q̇ then differentiating helps.")
    print("- Some parameters are weakly excited (e.g., off-diagonal inertias), leading to wide CIs.")

plt.tight_layout()
plt.savefig("part1_torque.png", dpi=150)   # instead of plt.show()

if __name__ == '__main__':
    main()
