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
        """Sample workspace by testing Cartesian grid points via IK/solve; returns valid points only."""
        if y_res is None:
            y_res = x_res
        x_min, y_min, x_max, y_max = self._cartesian_bounds()
        xs = np.linspace(x_min, x_max, x_res)
        ys = np.linspace(y_min, y_max, y_res)
        xx, yy = np.meshgrid(xs, ys)
        grid_points = np.column_stack([xx.ravel(), yy.ravel()])
        valid_points = []
        total = len(grid_points)
        with tqdm(total=total, desc="Sweeping workspace (Cartesian)", unit="pt") as pbar:
            for pt in grid_points:
                try:
                    self.mech.solve(pt)
                    valid_points.append(pt)
                except (ValueError, Exception):
                    pass
                pbar.update(1)
        return np.array(valid_points) if valid_points else np.empty((0, 2))

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