import numpy as np
from scipy.spatial import Delaunay
from shapely.geometry import Polygon, Point, MultiLineString
from shapely.ops import polygonize
from tqdm import tqdm
import shapely


class FiveBarWorkspace:

    def __init__(self, mech):
        self.mech = mech

    def _cartesian_bounds(self):
        """Bounding box for the Cartesian grid (first quadrant; max extends past bases by la+lb)."""
        O_g, B_g = self.mech.tf.bases_global()
        la, lb = self.mech.la, self.mech.lb
        x_min = 0.0
        y_min = 0.0
        x_max = float(max(O_g[0], B_g[0])) + la + lb
        y_max = float(max(O_g[1], B_g[1])) + la + lb
        return x_min, y_min, x_max, y_max

    def _make_reachability_checker(self):
        """Return a fast reachability predicate based on distances to the two bases."""
        O_l, B_l = self.mech.tf.bases_local()
        la, lb = self.mech.la, self.mech.lb
        r_min = max(0.0, abs(la - lb) * 0.95)
        r_max = (la + lb) * 1.05

        def reachable_fast(pt_g):
            pt_l = self.mech.tf.g2l(pt_g)
            r_O = np.linalg.norm(pt_l - O_l)
            r_B = np.linalg.norm(pt_l - B_l)
            if r_O < r_min or r_O > r_max or r_B < r_min or r_B > r_max:
                return False
            return True

        return reachable_fast

    def sweep_cartesian(self, x_res=100, y_res=None):
        """Adaptive workspace sampling: coarse grid + local refinement around boundary points."""
        if y_res is None:
            y_res = x_res

        x_min, y_min, x_max, y_max = self._cartesian_bounds()

        # Fixed coarse resolution: very rough pass to find approximate workspace.
        coarse_res = min(40, x_res)  # do not exceed requested fine resolution

        reachable_fast = self._make_reachability_checker()

        # ---- Coarse sweep over full bounds ----
        xs_coarse = np.linspace(x_min, x_max, coarse_res)
        ys_coarse = np.linspace(y_min, y_max, coarse_res)
        dx = (x_max - x_min) / (coarse_res - 1) if coarse_res > 1 else 0.0
        dy = (y_max - y_min) / (coarse_res - 1) if coarse_res > 1 else 0.0

        coarse_valid = np.zeros((coarse_res, coarse_res), dtype=bool)
        coarse_points = []
        total_coarse = coarse_res * coarse_res
        with tqdm(total=total_coarse, desc="Workspace coarse sweep", unit="pt") as pbar:
            for i, x in enumerate(xs_coarse):
                for j, y in enumerate(ys_coarse):
                    pt = np.array([x, y])
                    if reachable_fast(pt):
                        try:
                            self.mech.solve(pt)
                            coarse_valid[i, j] = True
                            coarse_points.append(pt)
                        except ValueError:
                            pass
                    pbar.update(1)

        if not coarse_points:
            return np.empty((0, 2))

        coarse_points = np.vstack(coarse_points)

        # ---- Find boundary points in the coarse grid ----
        boundary_indices = []
        for i in range(coarse_res):
            for j in range(coarse_res):
                if not coarse_valid[i, j]:
                    continue
                # 4-neighborhood; missing neighbors treated as outside (boundary).
                neighbors = [
                    (i - 1, j),
                    (i + 1, j),
                    (i, j - 1),
                    (i, j + 1),
                ]
                has_hole_neighbor = False
                for ni, nj in neighbors:
                    if ni < 0 or ni >= coarse_res or nj < 0 or nj >= coarse_res:
                        has_hole_neighbor = True
                        break
                    if not coarse_valid[ni, nj]:
                        has_hole_neighbor = True
                        break
                if has_hole_neighbor:
                    boundary_indices.append((i, j))

        # ---- Local refinement around boundary points ----
        refined_points = []
        # Subdivision factor per coarse cell, based on desired fine resolution.
        sub_div = max(2, x_res // max(1, coarse_res))

        total_refined = len(boundary_indices) * (sub_div * sub_div)
        with tqdm(total=total_refined, desc="Workspace boundary refinement", unit="pt") as pbar:
            for i, j in boundary_indices:
                x_center = xs_coarse[i]
                y_center = ys_coarse[j]

                # Define a small box around the coarse point, within one coarse cell.
                x0 = max(x_min, x_center - dx / 2.0)
                x1 = min(x_max, x_center + dx / 2.0)
                y0 = max(y_min, y_center - dy / 2.0)
                y1 = min(y_max, y_center + dy / 2.0)

                xs_local = np.linspace(x0, x1, sub_div)
                ys_local = np.linspace(y0, y1, sub_div)

                for xl in xs_local:
                    for yl in ys_local:
                        pt = np.array([xl, yl])
                        if not reachable_fast(pt):
                            pbar.update(1)
                            continue
                        try:
                            self.mech.solve(pt)
                            refined_points.append(pt)
                        except ValueError:
                            pass
                        pbar.update(1)

        if refined_points:
            refined_points = np.vstack(refined_points)
            all_points = np.vstack([coarse_points, refined_points])
        else:
            all_points = coarse_points

        return all_points

    def sweep_cartesian_full(self, x_res=70, y_res=None):
        """Brute-force uniform grid sweep at a given Cartesian resolution."""
        if y_res is None:
            y_res = x_res

        x_min, y_min, x_max, y_max = self._cartesian_bounds()
        xs = np.linspace(x_min, x_max, x_res)
        ys = np.linspace(y_min, y_max, y_res)

        reachable_fast = self._make_reachability_checker()

        points = []
        total = x_res * y_res
        with tqdm(total=total, desc="Workspace full sweep", unit="pt") as pbar:
            for x in xs:
                for y in ys:
                    pt = np.array([x, y])
                    if not reachable_fast(pt):
                        pbar.update(1)
                        continue
                    try:
                        self.mech.solve(pt)
                        points.append(pt)
                    except ValueError:
                        pass
                    pbar.update(1)

        if not points:
            return np.empty((0, 2))

        return np.vstack(points)

    def sweep_cartesian_adaptive(self, max_res=70, min_res=10, samples_per_cell=3):
        """Multi-stage adaptive sweep: refine only boundary cells up to an effective max resolution."""
        from collections import deque

        if min_res < 2:
            raise ValueError("min_res must be at least 2")
        if samples_per_cell < 2:
            raise ValueError("samples_per_cell must be at least 2")

        x_min, y_min, x_max, y_max = self._cartesian_bounds()

        # Target spacing corresponding to a uniform grid of max_res.
        if max_res < 2:
            raise ValueError("max_res must be at least 2")
        dx_target = (x_max - x_min) / (max_res - 1)
        dy_target = (y_max - y_min) / (max_res - 1)

        # Initial coarse grid edges for seeding cells.
        xs0 = np.linspace(x_min, x_max, min_res)
        ys0 = np.linspace(y_min, y_max, min_res)

        # Cell representation: (x0, x1, y0, y1, level)
        queue = deque()
        for i in range(min_res - 1):
            for j in range(min_res - 1):
                x0, x1 = xs0[i], xs0[i + 1]
                y0, y1 = ys0[j], ys0[j + 1]
                queue.append((x0, x1, y0, y1, 0))

        # Maximum depth derived from desired refinement ratio.
        import math

        max_depth = math.ceil(math.log2(max_res / float(min_res)))

        reachable_fast = self._make_reachability_checker()
        points_adaptive = []

        def add_representative_point(cell_points, x0, x1, y0, y1):
            """Add a single representative point for this cell, near its center."""
            if not cell_points:
                return
            cx = 0.5 * (x0 + x1)
            cy = 0.5 * (y0 + y1)
            best_pt = None
            best_d2 = None
            for pt in cell_points:
                dxp = float(pt[0]) - cx
                dyp = float(pt[1]) - cy
                d2 = dxp * dxp + dyp * dyp
                if best_d2 is None or d2 < best_d2:
                    best_d2 = d2
                    best_pt = pt
            if best_pt is not None:
                points_adaptive.append(best_pt)

        with tqdm(desc="Workspace adaptive sweep", unit="cell") as pbar:
            while queue:
                x0, x1, y0, y1, level = queue.popleft()
                pbar.update(1)

                dx_cell = x1 - x0
                dy_cell = y1 - y0

                xs = np.linspace(x0, x1, samples_per_cell)
                ys = np.linspace(y0, y1, samples_per_cell)

                num_reachable = 0
                num_unreachable = 0
                cell_points = []

                for x in xs:
                    for y in ys:
                        pt = np.array([x, y])
                        if not reachable_fast(pt):
                            num_unreachable += 1
                            continue
                        try:
                            self.mech.solve(pt)
                            num_reachable += 1
                            cell_points.append(pt)
                        except ValueError:
                            num_unreachable += 1

                if num_reachable == 0:
                    # Exterior cell: nothing to do.
                    continue

                if num_unreachable == 0:
                    # Interior cell: keep one representative sample, do not refine.
                    add_representative_point(cell_points, x0, x1, y0, y1)
                    continue

                # Boundary cell: decide whether this is a leaf; if so, keep samples.
                is_leaf = (dx_cell <= dx_target and dy_cell <= dy_target) or level >= max_depth
                if is_leaf:
                    # Leaf boundary cell: keep one representative sample.
                    add_representative_point(cell_points, x0, x1, y0, y1)
                    continue

                # Otherwise, refine boundary cell further.
                xm = 0.5 * (x0 + x1)
                ym = 0.5 * (y0 + y1)
                next_level = level + 1
                queue.append((x0, xm, y0, ym, next_level))
                queue.append((xm, x1, y0, ym, next_level))
                queue.append((x0, xm, ym, y1, next_level))
                queue.append((xm, x1, ym, y1, next_level))

        if not points_adaptive:
            return np.empty((0, 2))

        points_adaptive = np.vstack(points_adaptive)
        # Remove exact duplicates if any.
        points_adaptive = np.unique(points_adaptive, axis=0)

        # Densify sampling along the coordinate axes so alpha-shape can
        # correctly wrap the workspace where it meets x=0 and y=0.
        points_adaptive = self._augment_points_on_axes(max_res, points_adaptive)
        return points_adaptive

    def _augment_points_on_axes(self, max_res, points):
        """Densify sampling along x=0 and y=0.

        Uses a resolution higher than max_res along the axes to ensure there are
        no visible gaps in the boundary sampling for alpha-shape.
        """
        x_min, y_min, x_max, y_max = self._cartesian_bounds()
        reachable_fast = self._make_reachability_checker()

        # Match axis sampling density to the target resolution.
        axis_res = max_res

        xs_axis = np.linspace(x_min, x_max, axis_res)
        ys_axis = np.linspace(y_min, y_max, axis_res)

        axis_points = []

        # Sweep along x-axis (y = 0).
        for x in xs_axis:
            pt = np.array([x, 0.0])
            if not reachable_fast(pt):
                continue
            try:
                self.mech.solve(pt)
                axis_points.append(pt)
            except ValueError:
                pass

        # Sweep along y-axis (x = 0).
        for y in ys_axis:
            pt = np.array([0.0, y])
            if not reachable_fast(pt):
                continue
            try:
                self.mech.solve(pt)
                axis_points.append(pt)
            except ValueError:
                pass

        if not axis_points:
            return points

        axis_points = np.vstack(axis_points)
        all_points = np.vstack([points, axis_points])
        return np.unique(all_points, axis=0)

    def compare_adaptive_to_full(self, max_res=70, alpha=None, min_res=10, samples_per_cell=3):
        """Run full and adaptive sweeps and compare their alpha-shape workspaces.

        Returns a dict with timing and geometric similarity metrics.
        """
        import time

        if alpha is None:
            # Simple heuristic: scale alpha with workspace size.
            x_min, y_min, x_max, y_max = self._cartesian_bounds()
            span = max(x_max - x_min, y_max - y_min)
            alpha = 1.0 / max(span * 0.1, 1e-6)

        t0 = time.perf_counter()
        pts_full = self.sweep_cartesian_full(max_res)
        t_full = (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        pts_adapt = self.sweep_cartesian_adaptive(max_res=max_res, min_res=min_res, samples_per_cell=samples_per_cell)
        t_adapt = (time.perf_counter() - t1) * 1000.0

        if pts_full.shape[0] == 0 or pts_adapt.shape[0] == 0:
            return {
                "pts_full": pts_full,
                "pts_adapt": pts_adapt,
                "poly_full": None,
                "poly_adapt": None,
                "time_full_ms": t_full,
                "time_adaptive_ms": t_adapt,
                "area_rel_error": None,
                "sym_diff_ratio": None,
                "hausdorff_distance": None,
            }

        poly_full = self.alpha_shape(pts_full, alpha)
        poly_adapt = self.alpha_shape(pts_adapt, alpha)

        area_full = poly_full.area
        area_adapt = poly_adapt.area
        area_diff = abs(area_full - area_adapt)
        area_rel_error = area_diff / area_full if area_full > 0 else None

        sym_diff = poly_full.symmetric_difference(poly_adapt)
        sym_diff_ratio = sym_diff.area / area_full if area_full > 0 else None

        hausdorff = poly_full.boundary.hausdorff_distance(poly_adapt.boundary)

        return {
            "pts_full": pts_full,
            "pts_adapt": pts_adapt,
            "poly_full": poly_full,
            "poly_adapt": poly_adapt,
            "time_full_ms": t_full,
            "time_adaptive_ms": t_adapt,
            "area_rel_error": area_rel_error,
            "sym_diff_ratio": sym_diff_ratio,
            "hausdorff_distance": hausdorff,
        }

    def alpha_shape(self, points, alpha):
        """Compute an alpha-shape (concave hull) polygon for the given points.

        This is made robust against degenerate point sets and Qhull failures by
        falling back to the convex hull when needed.
        """
        from shapely.geometry import MultiPoint

        # Guard against pathological inputs.
        if points is None or len(points) == 0:
            return Polygon()

        if len(points) < 4:
            # With very few points, just return the convex hull.
            return MultiPoint(points).convex_hull

        # Ensure alpha is a positive float.
        if alpha is None or alpha <= 0:
            raise ValueError("alpha must be a positive float for alpha_shape.")

        try:
            tri = Delaunay(points)
        except Exception:
            # Fall back to convex hull if Delaunay fails (e.g., nearly collinear points).
            return MultiPoint(points).convex_hull

        edges = set()

        for ia, ib, ic in tri.simplices:

            pa = points[ia]
            pb = points[ib]
            pc = points[ic]

            a = np.linalg.norm(pb - pa)
            b = np.linalg.norm(pc - pb)
            c = np.linalg.norm(pa - pc)

            s = (a + b + c) / 2
            area = np.sqrt(max(s * (s - a) * (s - b) * (s - c), 0))

            if area == 0:
                continue

            circum_r = a * b * c / (4 * area)

            if circum_r < 1 / alpha:
                edges.add(tuple(sorted((ia, ib))))
                edges.add(tuple(sorted((ib, ic))))
                edges.add(tuple(sorted((ic, ia))))

        if not edges:
            # If no edges are selected by the alpha criterion, fall back to convex hull.
            return MultiPoint(points).convex_hull

        edge_lines = [(points[i], points[j]) for i, j in edges]

        m = MultiLineString(edge_lines)
        polys = list(polygonize(m))

        if not polys:
            # As a final safety, return convex hull if polygonization fails.
            return MultiPoint(points).convex_hull

        return shapely.ops.unary_union(polys)

    def largest_inscribed_circle(self, poly, res=200):

        xmin,ymin,xmax,ymax = poly.bounds

        xs = np.linspace(xmin,xmax,res)
        ys = np.linspace(ymin,ymax,res)

        best_center=None
        best_r=0

        boundary = poly.boundary

        for x in xs:
            for y in ys:

                p = Point(x,y)

                if poly.contains(p):

                    d = p.distance(boundary)

                    if d>best_r:
                        best_r=d
                        best_center=(x,y)

        return best_center,best_r
    
    def safe_workspace_circle(self, poly):
        """Largest circle inscribed in poly (polygon already encodes safe margin via valid_config)."""
        center, r = self.largest_inscribed_circle(poly)
        return center, r