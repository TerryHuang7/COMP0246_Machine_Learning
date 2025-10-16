import numpy as np
import time
import os
import matplotlib.pyplot as plt
from simulation_and_control import pb, MotorCommands, PinWrapper, feedback_lin_ctrl, SinusoidalReference 

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
    # Specify different amplitude values for each joint
    amplitudes = [np.pi/4, np.pi/6, np.pi/4, np.pi/4, np.pi/4, np.pi/4, np.pi/4]  # Example amplitudes for joints
    # Specify different frequency values for each joint
    frequencies = [0.4, 0.5, 0.4, 0.4, 0.4, 0.4, 0.4]  # Example frequencies for joints

    # Convert lists to NumPy arrays for easier manipulation in computations
    amplitude = np.array(amplitudes)
    frequency = np.array(frequencies)
    ref = SinusoidalReference(amplitude, frequency, sim.GetInitMotorAngles())  # Initialize the reference
    
    
    # Simulation parameters
    time_step = sim.GetTimeStep()
    current_time = 0
    max_time = 10  # seconds
    
    # Command and control loop
    cmd = MotorCommands()  # Initialize command structure for motors
    # PD controller gains
    kp = 1000
    kd = 100

    # Initialize data storage
    tau_mes_all = []
    regressor_all = []

    # Data collection loop
    while current_time < max_time:
        # Measure current state
        q_mes = sim.GetMotorAngles(0)
        qd_mes = sim.GetMotorVelocities(0)
        qdd_mes = sim.ComputeMotorAccelerationTMinusOne(0)
        
        # Compute sinusoidal reference trajectory
        q_d, qd_d = ref.get_values(current_time)  # Desired position and velocity
        
        # Control command
        tau_cmd = feedback_lin_ctrl(dyn_model, q_mes, qd_mes, q_d, qd_d, kp, kd)  # Zero torque command
        cmd.SetControlCmd(tau_cmd, ["torque"]*7)  # Set the torque com
        sim.Step(cmd, "torque")

        # Get measured torque
        tau_mes = sim.GetMotorTorques(0)

        if dyn_model.visualizer: 
            for index in range(len(sim.bot)):  # Conditionally display the robot model
                q = sim.GetMotorAngles(index)
                dyn_model.DisplayModel(q)  # Update the display of the robot model

        # Exit logic with 'q' key
        keys = sim.GetPyBulletClient().getKeyboardEvents()
        qKey = ord('q')
        if qKey in keys and keys[qKey] and sim.GetPyBulletClient().KEY_WAS_TRIGGERED:
            break

        
        # TODO Compute regressor and store it
        Y_t = dyn_model.ComputeDynamicRegressor(q_mes, qd_mes, qdd_mes)     # (7, 70)
        regressor_all.append(Y_t)
        tau_mes_all.append(np.asarray(tau_mes))
        
        current_time += time_step
        # Optional: print current time
        print(f"Current time in seconds: {current_time:.2f}")

    # TODO After data collection, stack all the regressor and all the torque and compute the parameters 'a'  using pseudoinverse for all the joint
    if len(regressor_all) == 0:
        print("No data collected. Exiting."); return
    Y = np.vstack(regressor_all)                      # (T*7, 70)
    u = np.hstack(tau_mes_all)                        # (T*7,)

    # 小岭稳定：避免 (Y^T Y) 病态
    XtX = Y.T @ Y
    lam_all = 1e-6 if np.linalg.cond(XtX) < 1e6 else 1e-4
    a_hat_all = np.linalg.solve(XtX + lam_all*np.eye(XtX.shape[0]), Y.T @ u)
    print("\n[Part2] estimated full parameter vector (len={}): ridge λ={}".format(a_hat_all.size, lam_all))

    # TODO reshape the regressor and the torque vector to isolate the last joint and find the its dynamical parameters
    # 用 “1–6 已知” 的残差法，仅估计 link7 的 10 个参数
    urdf_path = os.path.join(cur_dir, "models", "panda_description", "panda.urdf")
    link_names = [f"panda_link{i}" for i in range(1,8)]
    per_link_params_known = {i: extract_link10_from_urdf(urdf_path, link_names[i-1]) for i in range(1,7)}
    a_known_full = build_full_param_vector(7, per_link_params_known)

    s7, e7 = 10*(7-1), 10*(7-1)+10
    r = u - (Y @ a_known_full)           # 去掉 1..6 链节贡献
    X7 = Y[:, s7:e7]                     # 只取 link7 的 10 列
    lam7 = 1e-6 if np.linalg.cond(X7.T @ X7) < 1e6 else 1e-4
    a7_hat = np.linalg.solve(X7.T @ X7 + lam7*np.eye(10), X7.T @ r)

    a7_true = extract_link10_from_urdf(urdf_path, link_names[6])
    print("\n[Part1] link7 params:")
    print("  a7_hat :", np.round(a7_hat, 6))
    print("  a7_true:", np.round(a7_true, 6))
    print("  abs err:", np.round(np.abs(a7_hat - a7_true), 6))

    # TODO compute the metrics (R-squared adjusted etc...) for the linear model on a different file 
    # 这里先在当前数据上给出指标（若要严格分训练/验证，可把这段复制到另一个评估脚本中）
    a_full_hat = a_known_full.copy(); a_full_hat[s7:e7] = a7_hat
    u_pred_part1 = Y @ a_full_hat
    mets1 = regression_metrics(u, u_pred_part1, p=10, has_intercept=False)
    print("\n[Part1] metrics:", mets1)

    u_pred_part2 = Y @ a_hat_all
    mets2 = regression_metrics(u, u_pred_part2, p=Y.shape[1], has_intercept=False)
    print("[Part2] metrics:", mets2)
    
    # TODO plot the torque prediction error for each joint (optional)
    T = len(tau_mes_all)
    t_axis = np.arange(T) * time_step
    tau_mat  = np.vstack(tau_mes_all)                   # (T, 7)
    uhat_mat = np.vstack([Yt @ a_full_hat for Yt in regressor_all])
    err_mat  = uhat_mat - tau_mat

    fig, axes = plt.subplots(7, 1, figsize=(10, 14), sharex=True)
    for j in range(7):
        ax = axes[j]
        ax.plot(t_axis, tau_mat[:, j], label=f"τ{j+1} meas")
        ax.plot(t_axis, uhat_mat[:, j], linestyle="--", label=f"τ{j+1} pred")
        ax.plot(t_axis, err_mat[:, j], linestyle=":", label=f"err{j+1}")
        ax.set_ylabel(f"J{j+1}")
        ax.grid(True, alpha=0.3)
        if j == 0:
            ax.legend(loc="upper right", ncol=3, fontsize=8)
    axes[-1].set_xlabel("Time [s]")
    fig.suptitle("Torque prediction & error (Part1: link7 estimated)")
    plt.tight_layout()
    plt.savefig("part1_torque.png", dpi=150)
    print("[OK] saved plot -> part1_torque.png")

if __name__ == '__main__':
    main()

# ============ helpers (追加；不改老师框架) ============
def regression_metrics(y, y_hat, p, has_intercept=False):
    """Return RSS, TSS, R2, R2_adj, F-stat with consistent DOF."""
    M = y.size
    residual = y - y_hat
    RSS = float(residual @ residual)
    TSS = float(((y - y.mean()) ** 2).sum()) if M > 0 else np.nan
    R2 = 1.0 - (RSS / TSS if TSS > 0 else np.nan)
    df_resid = max(M - p - (1 if has_intercept else 0), 1)
    df_adj_den = max(M - (1 if has_intercept else 0), 1)
    R2_adj = 1.0 - ((RSS / df_resid) / (TSS / df_adj_den) if TSS > 0 else np.nan)
    SSR = max(TSS - RSS, 0.0)
    num = (SSR / max(p,1))
    den = (RSS / df_resid)
    F = (num / den) if den > 0 else np.nan
    return {"RSS": RSS, "TSS": TSS, "R2": R2, "R2_adj": R2_adj, "F": F, "M": M, "p": p}

def rpy_to_R(roll, pitch, yaw):
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[cy, -sy, 0],[sy, cy, 0],[0, 0, 1]])
    Ry = np.array([[cp, 0, sp],[0, 1, 0],[-sp, 0, cp]])
    Rx = np.array([[1, 0, 0],[0, cr, -sr],[0, sr, cr]])
    return Rz @ Ry @ Rx

def extract_link10_from_urdf(urdf_path, link_name):
    """Parse URDF and return [m, m*cx, m*cy, m*cz, Ixx, Ixy, Ixz, Iyy, Iyz, Izz] (in link frame)."""
    import xml.etree.ElementTree as ET  # 局部导入，避免改动顶部 import
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    link = None
    for lk in root.findall('link'):
        if lk.get('name') == link_name:
            link = lk; break
    if link is None:
        raise ValueError(f"Link '{link_name}' not found in {urdf_path}")
    inertial = link.find('inertial')
    m = float(inertial.find('mass').attrib['value'])
    origin = inertial.find('origin')
    if origin is not None:
        xyz = [float(v) for v in origin.attrib.get('xyz', '0 0 0').split()]
        rpy = [float(v) for v in origin.attrib.get('rpy', '0 0 0').split()]
    else:
        xyz = [0.0,0.0,0.0]; rpy = [0.0,0.0,0.0]
    cx, cy, cz = xyz
    R = rpy_to_R(*rpy)
    I = inertial.find('inertia').attrib
    I_body = np.array([[float(I['ixx']), float(I['ixy']), float(I['ixz'])],
                       [float(I['ixy']), float(I['iyy']), float(I['iyz'])],
                       [float(I['ixz']), float(I['iyz']), float(I['izz'])]])
    I_link = R @ I_body @ R.T
    Ixx, Ixy, Ixz = I_link[0,0], I_link[0,1], I_link[0,2]
    Iyy, Iyz, Izz = I_link[1,1], I_link[1,2], I_link[2,2]
    return np.array([m, m*cx, m*cy, m*cz, Ixx, Ixy, Ixz, Iyy, Iyz, Izz], dtype=float)

def build_full_param_vector(n_dof, per_link_params):
    a_full = np.zeros(10*n_dof)
    for link_idx, a10 in per_link_params.items():
        s = 10*(link_idx-1); a_full[s:s+10] = np.asarray(a10).reshape(10,)
        # 其余未提供的链节保持 0（按“已知 1..6；未知 7”的设定）
    return a_full
