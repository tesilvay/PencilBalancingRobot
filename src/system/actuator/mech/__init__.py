from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np
from src.shared import Spec
from src.system.actuator.mech.transform import FiveBarTransform
from src.system.actuator.mech.mechanism import FiveBarMechanism

# WARNING
# ALL VALUES ARE IN MM FOR THIS CLASS

@dataclass
class MechanismParams:
    O:  np.ndarray   # origin point
    B:  np.ndarray   # base point
    la: float        # link a length
    lb: float        # link b length
    calibration_file: str | None = "src/system/actuator/mech/five_bar_calibration.json"

MECHANISM_PRESETS = {
    "default": {
        "O":  np.array([128.77, 178.13]),
        "B":  np.array([101.77, 210.13]),
        "la": 175,
        "lb": 175,
        "calibration_file": "src/system/actuator/mech/five_bar_calibration.json",
    }
}

class Mechanism:
    def __init__(self, params: MechanismParams):
        tf = FiveBarTransform(params.O, params.B)
        self._mech = FiveBarMechanism(tf, la=params.la, lb=params.lb)
        self.workspace_offset = (0.0, 0.0)
        self.calibration_enabled = True
        self.calibration_path = self._resolve_calibration_path(params.calibration_file)
        self._calibration_points: dict[str, dict[str, np.ndarray]] = {}
        self._affine_matrix = np.eye(2, dtype=float)
        self._affine_offset = np.zeros(2, dtype=float)
        self.load_calibration()

    @property
    def tf(self):
        """Transform shared with :class:`FiveBarMechanism` (O_g, B_g, …)."""
        return self._mech.tf

    def set_workspace_offset(self, dx: float, dy: float) -> None:
        self.workspace_offset = (float(dx), float(dy))

    def set_calibration_enabled(self, enabled: bool) -> None:
        self.calibration_enabled = bool(enabled)

    def _resolve_calibration_path(self, calibration_file: str | None) -> Path | None:
        if calibration_file is None:
            return None
        path = Path(calibration_file)
        if not path.is_absolute():
            path = Path.cwd() / path
        return path

    def _fit_affine_from_points(self) -> None:
        if len(self._calibration_points) < 3:
            self._affine_matrix = np.eye(2, dtype=float)
            self._affine_offset = np.zeros(2, dtype=float)
            return

        names = sorted(self._calibration_points)
        desired = np.array([self._calibration_points[name]["desired_xy_m"] for name in names], dtype=float)
        servo = np.array([self._calibration_points[name]["servo_xy_m"] for name in names], dtype=float)
        X = np.column_stack([desired, np.ones(len(desired), dtype=float)])
        coeff, *_ = np.linalg.lstsq(X, servo, rcond=None)
        self._affine_matrix = coeff[:2, :].T
        self._affine_offset = coeff[2, :]

    def load_calibration(self) -> bool:
        if self.calibration_path is None or not self.calibration_path.exists():
            self._calibration_points = {}
            self._fit_affine_from_points()
            return False

        data = json.loads(self.calibration_path.read_text(encoding="utf-8"))
        raw_points = data.get("points", {})
        points: dict[str, dict[str, np.ndarray]] = {}
        for name, payload in raw_points.items():
            desired_xy = np.asarray(payload["desired_xy_m"], dtype=float).reshape(2)
            servo_xy = np.asarray(payload["servo_xy_m"], dtype=float).reshape(2)
            points[str(name)] = {
                "desired_xy_m": desired_xy,
                "servo_xy_m": servo_xy,
            }

        self._calibration_points = points
        self._fit_affine_from_points()
        return bool(points)

    def save_calibration(self, points: dict[str, dict[str, np.ndarray]]) -> None:
        clean_points: dict[str, dict[str, np.ndarray]] = {}
        for name, payload in points.items():
            clean_points[str(name)] = {
                "desired_xy_m": np.asarray(payload["desired_xy_m"], dtype=float).reshape(2),
                "servo_xy_m": np.asarray(payload["servo_xy_m"], dtype=float).reshape(2),
            }

        self._calibration_points = clean_points
        self._fit_affine_from_points()

        if self.calibration_path is None:
            return

        self.calibration_path.parent.mkdir(parents=True, exist_ok=True)
        serializable = {
            "version": 1,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "points": {
                name: {
                    "desired_xy_m": payload["desired_xy_m"].astype(float).tolist(),
                    "servo_xy_m": payload["servo_xy_m"].astype(float).tolist(),
                }
                for name, payload in sorted(self._calibration_points.items())
            },
        }
        self.calibration_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")

    def calibration_points(self) -> dict[str, dict[str, np.ndarray]]:
        return {
            name: {
                "desired_xy_m": payload["desired_xy_m"].copy(),
                "servo_xy_m": payload["servo_xy_m"].copy(),
            }
            for name, payload in self._calibration_points.items()
        }

    def _calibration_seed(self, name: str, desired_xy_m: np.ndarray) -> np.ndarray:
        saved = self._calibration_points.get(name)
        if saved is not None:
            return saved["servo_xy_m"].copy()
        return np.asarray(desired_xy_m, dtype=float).reshape(2)

    def calibration_targets(
        self,
        *,
        x_ref: float,
        y_ref: float,
        cardinal_delta_m: float,
        safe_radius: float | None,
    ) -> list[dict[str, np.ndarray | str]]:
        delta = float(safe_radius) if safe_radius is not None else float(cardinal_delta_m)
        targets = [
            ("center", np.array([x_ref, y_ref], dtype=float)),
            ("up", np.array([x_ref, y_ref + delta], dtype=float)),
            ("right", np.array([x_ref + delta, y_ref], dtype=float)),
            ("down", np.array([x_ref, y_ref - delta], dtype=float)),
            ("left", np.array([x_ref - delta, y_ref], dtype=float)),
        ]

        result: list[dict[str, np.ndarray | str]] = []
        for name, desired_xy in targets:
            desired_xy = np.asarray(desired_xy, dtype=float).reshape(2)
            if safe_radius is not None and name != "center":
                delta_xy = desired_xy - np.array([x_ref, y_ref], dtype=float)
                dist = float(np.linalg.norm(delta_xy))
                if dist > safe_radius and dist > 0.0:
                    desired_xy = np.array([x_ref, y_ref], dtype=float) + delta_xy * (safe_radius / dist)

            result.append(
                {
                    "name": name,
                    "desired_xy_m": desired_xy,
                    "seed_servo_xy_m": self._calibration_seed(name, desired_xy),
                }
            )
        return result

    def _apply_calibration(self, desired_xy_m: np.ndarray) -> np.ndarray:
        return self._affine_matrix @ desired_xy_m + self._affine_offset

    def state_geometry(
        self, px_m: float, py_m: float
    ) -> tuple[np.ndarray, tuple[float, float]]:
        """
        IK+FK solve for a workspace state point without command-side corrections.

        This uses the state position directly in the mechanism global frame:
        no workspace offset and no calibration affine are applied.
        """
        target_mm = np.array([float(px_m), float(py_m)], dtype=float) * 1000.0
        theta1, theta4, A_g, C_g, P_g = self._mech.solve(target_mm)
        joints = np.stack(
            [
                np.asarray(A_g, dtype=float).reshape(-1)[:2],
                np.asarray(C_g, dtype=float).reshape(-1)[:2],
                np.asarray(P_g, dtype=float).reshape(-1)[:2],
            ],
            axis=0,
        )
        theta_deg = (float(np.rad2deg(theta1)), float(np.rad2deg(theta4)))
        return joints, theta_deg

    def command_geometry(
        self, command
    ) -> tuple[np.ndarray, tuple[float, float]]:
        """
        Single IK+FK solve for the table command (meters → mm internally).

        Returns
        -------
        joints_mm : ndarray, shape (3, 2)
            Rows: A, C, P in global mm.
        (theta1_deg, theta4_deg) : tuple[float, float]
            Servo angles in degrees (same convention as :meth:`command_to_angles`).
        """
        desired_xy = np.array(
            [
                float(command.px_cmd) + self.workspace_offset[0],
                float(command.py_cmd) + self.workspace_offset[1],
            ],
            dtype=float,
        )
        target_xy = self._apply_calibration(desired_xy) if self.calibration_enabled else desired_xy
        target_mm = target_xy * 1000.0
        theta1, theta4, A_g, C_g, P_g = self._mech.solve(target_mm)
        joints = np.stack(
            [
                np.asarray(A_g, dtype=float).reshape(-1)[:2],
                np.asarray(C_g, dtype=float).reshape(-1)[:2],
                np.asarray(P_g, dtype=float).reshape(-1)[:2],
            ],
            axis=0,
        )
        theta_deg = (float(np.rad2deg(theta1)), float(np.rad2deg(theta4)))
        return joints, theta_deg

    def command_to_angles(self, command) -> tuple[float, float]:
        _, theta_deg = self.command_geometry(command)
        return theta_deg

MECHANISM_REGISTRY = {
    "five_bar": Spec(Mechanism, MechanismParams, MECHANISM_PRESETS),
}


@dataclass
class MechanismTPSParams:
    O:  np.ndarray
    B:  np.ndarray
    la: float
    lb: float
    calibration_file: str | None = "src/system/actuator/mech/five_bar_tps_calibration.json"


MECHANISM_TPS_PRESETS = {
    "default": {
        "O":  np.array([128.77, 178.13]),
        "B":  np.array([101.77, 210.13]),
        "la": 175,
        "lb": 175,
        "calibration_file": "src/system/actuator/mech/five_bar_tps_calibration.json",
    }
}


class MechanismTPS:
    """
    TPS actuator model: maps (x_des, y_des) → (θ₁, θ₄) via RBF interpolation.

    Replaces FK/IK for the real system. Falls back to FK/IK when no calibration
    file exists. Simulation is unaffected (uses Mechanism with FK/IK).
    """

    def __init__(self, params: MechanismTPSParams):
        tf = FiveBarTransform(params.O, params.B)
        self._mech = FiveBarMechanism(tf, la=params.la, lb=params.lb)
        self.workspace_offset = (0.0, 0.0)
        self.calibration_path = self._resolve_calibration_path(params.calibration_file)
        self._rbf_theta1 = None
        self._rbf_theta4 = None
        self._calibration_xy = None
        self._workspace_radius = 0.07
        self.load_calibration()

    @property
    def tf(self):
        return self._mech.tf

    def set_workspace_offset(self, dx: float, dy: float) -> None:
        self.workspace_offset = (float(dx), float(dy))

    def _resolve_calibration_path(self, calibration_file: str | None):
        if calibration_file is None:
            return None
        from pathlib import Path
        path = Path(calibration_file)
        if not path.is_absolute():
            path = Path.cwd() / path
        return path

    def load_calibration(self) -> bool:
        if self.calibration_path is None or not self.calibration_path.exists():
            return False

        import json
        from scipy.interpolate import RBFInterpolator

        data = json.loads(self.calibration_path.read_text(encoding="utf-8"))
        if data.get("version") != "v1_tps_actuator":
            return False

        points = data["points"]
        desired_xy = np.array([p["desired_xy_m"] for p in points], dtype=float)
        theta1     = np.array([p["theta1_deg"]   for p in points], dtype=float)
        theta4     = np.array([p["theta4_deg"]   for p in points], dtype=float)

        self._rbf_theta1 = RBFInterpolator(desired_xy, theta1, kernel="thin_plate_spline")
        self._rbf_theta4 = RBFInterpolator(desired_xy, theta4, kernel="thin_plate_spline")
        self._calibration_xy = desired_xy
        self._workspace_radius = float(data.get("workspace_radius_m", 0.07))
        return True

    def _fk_ik_solve(
        self, x_m: float, y_m: float
    ) -> tuple[np.ndarray, tuple[float, float]]:
        target_mm = np.array([x_m, y_m]) * 1000.0
        theta1, theta4, A_g, C_g, P_g = self._mech.solve(target_mm)
        joints = np.stack(
            [
                np.asarray(A_g, dtype=float).reshape(-1)[:2],
                np.asarray(C_g, dtype=float).reshape(-1)[:2],
                np.asarray(P_g, dtype=float).reshape(-1)[:2],
            ],
            axis=0,
        )
        return joints, (float(np.rad2deg(theta1)), float(np.rad2deg(theta4)))

    def command_geometry(
        self, command
    ) -> tuple[np.ndarray, tuple[float, float]]:
        x = float(command.px_cmd) + self.workspace_offset[0]
        y = float(command.py_cmd) + self.workspace_offset[1]

        if self._rbf_theta1 is None:
            return self._fk_ik_solve(x, y)

        r = float(np.hypot(x, y))
        if r > self._workspace_radius:
            scale = self._workspace_radius / r
            x, y = x * scale, y * scale

        pt = np.array([[x, y]])
        theta1_deg = float(self._rbf_theta1(pt)[0])
        theta4_deg = float(self._rbf_theta4(pt)[0])

        A_l, C_l, P1_l, P2_l = self._mech.fk(
            np.deg2rad(theta1_deg), np.deg2rad(theta4_deg)
        )
        O_l, B_l = self._mech.tf.bases_local()
        P_l = P1_l if self._mech.valid_config(O_l, B_l, A_l, C_l, P1_l) else P2_l

        joints = np.stack(
            [
                self._mech.tf.l2g(A_l).reshape(-1)[:2],
                self._mech.tf.l2g(C_l).reshape(-1)[:2],
                self._mech.tf.l2g(P_l).reshape(-1)[:2],
            ],
            axis=0,
        )
        return joints, (theta1_deg, theta4_deg)

    def command_to_angles(self, command) -> tuple[float, float]:
        _, theta_deg = self.command_geometry(command)
        return theta_deg

    def state_geometry(
        self, px_m: float, py_m: float
    ) -> tuple[np.ndarray, tuple[float, float]]:
        return self._fk_ik_solve(float(px_m), float(py_m))


# Re-register with TPS now defined
MECHANISM_REGISTRY["five_bar_tps"] = Spec(MechanismTPS, MechanismTPSParams, MECHANISM_TPS_PRESETS)
