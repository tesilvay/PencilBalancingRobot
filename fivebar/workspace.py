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

    def sweep_cartesian(self, x_res=100, y_res=None):
        """Adaptive workspace sampling: coarse grid + boundary refinement via IK/solve."""
        if y_res is None:
            y_res = x_res

        # Derive a coarse resolution from the requested fine resolution.
        coarse_res = max(10, x_res // 2)

        x_min, y_min, x_max, y_max = self._cartesian_bounds()

        # Precompute geometry for early rejection in local frame.
        O_l, B_l = self.mech.tf.bases_local()
        la, lb = self.mech.la, self.mech.lb
        r_min = max(0.0, abs(la - lb) * 0.95)
        r_max = (la + lb) * 1.05

        def reachable_fast(pt_g):
            """Cheap reachability check based on distances from the two bases."""
            pt_l = self.mech.tf.g2l(pt_g)
            r_O = np.linalg.norm(pt_l - O_l)
            r_B = np.linalg.norm(pt_l - B_l)
            if r_O < r_min or r_O > r_max or r_B < r_min or r_B > r_max:
                return False
            return True

        # ---- Coarse sweep over full bounds ----
        xs_coarse = np.linspace(x_min, x_max, coarse_res)
        ys_coarse = np.linspace(y_min, y_max, coarse_res)

        coarse_points = []
        total_coarse = coarse_res * coarse_res
        with tqdm(total=total_coarse, desc="Workspace coarse sweep", unit="pt") as pbar:
            for x in xs_coarse:
                for y in ys_coarse:
                    pt = np.array([x, y])
                    if reachable_fast(pt):
                        try:
                            self.mech.solve(pt)
                            coarse_points.append(pt)
                        except ValueError:
                            pass
                    pbar.update(1)

        if not coarse_points:
            return np.empty((0, 2))

        coarse_points = np.vstack(coarse_points)

        # Build coarse polygon.
        coarse_poly = self.alpha_shape(coarse_points, alpha=0.05)

        # ---- Boundary-focused refinement ----
        xs_fine = np.linspace(x_min, x_max, x_res)
        ys_fine = np.linspace(y_min, y_max, y_res)

        # Narrow band around the polygon where we refine.
        band_width = 5.0  # mm
        band_region = coarse_poly.buffer(band_width)

        refined_points = []
        total_fine = x_res * y_res
        with tqdm(total=total_fine, desc="Workspace boundary refinement", unit="pt") as pbar:
            for x in xs_fine:
                for y in ys_fine:
                    pt = np.array([x, y])
                    if not band_region.contains(Point(pt[0], pt[1])):
                        pbar.update(1)
                        continue
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

    def alpha_shape(self, points, alpha):

        tri = Delaunay(points)
        edges = set()

        for ia, ib, ic in tri.simplices:

            pa = points[ia]
            pb = points[ib]
            pc = points[ic]

            a = np.linalg.norm(pb-pa)
            b = np.linalg.norm(pc-pb)
            c = np.linalg.norm(pa-pc)

            s = (a+b+c)/2
            area = np.sqrt(max(s*(s-a)*(s-b)*(s-c),0))

            if area == 0:
                continue

            circum_r = a*b*c/(4*area)

            if circum_r < 1/alpha:
                edges.add(tuple(sorted((ia,ib))))
                edges.add(tuple(sorted((ib,ic))))
                edges.add(tuple(sorted((ic,ia))))

        edge_lines = [(points[i], points[j]) for i,j in edges]

        m = MultiLineString(edge_lines)
        polys = list(polygonize(m))

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