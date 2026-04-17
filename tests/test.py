import numpy as np

K = np.array([
    [-1.60726408e-01, -2.84179159e-02, -3.65660122e-02, -4.50309823e-03,
     -9.80783279e-12, -1.64742953e-12, -2.11325886e-12, -2.56434923e-13,
      1.00751452e-01,  2.32470854e-12],
    [-3.63120350e-12, -6.37239102e-13, -8.18350158e-13, -1.00801477e-13,
     -1.60726408e-01, -2.84179159e-02, -3.65660122e-02, -4.50309823e-03,
      2.32470854e-12,  1.00751452e-01]
])

u_prev = np.array([5e-3, 0.0])
u_ref_lqr = np.array([0.0, 0.0])

state = np.array([5e-3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
x_ref_lqr = np.array([0.0, 0.0, np.deg2rad(-1.0), 0.0, 0.0, 0.0, 0.0, 0.0])

x_err = state - x_ref_lqr
u_err = u_prev
xi_err = np.concatenate([x_err, u_err])

print("K shape:", K.shape)
print("xi_err shape:", xi_err.shape)

delta_u = -(K @ xi_err).ravel()
u = u_prev + delta_u

print("x_err:", x_err)
print("u_err:", u_err)
print(f"u: {np.round(u*1000, 2)} mm | delta_u: {np.round(delta_u*1000, 2)} mm")
print("row 0 termwise:", -(K[0] * xi_err))
print("row 1 termwise:", -(K[1] * xi_err))