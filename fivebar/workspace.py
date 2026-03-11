import numpy as np
from scipy.spatial import Delaunay
from shapely.geometry import Polygon, Point, MultiLineString
from shapely.ops import polygonize
import shapely


class FiveBarWorkspace:

    def __init__(self, mech):
        self.mech = mech

    def sweep_joint_space(self, theta_res=100):

        theta_vals = np.linspace(0, np.pi, theta_res)

        valid_angles = []
        valid_points = []

        O_l, B_l = self.mech.tf.bases_local()

        for t1 in theta_vals:
            for t4 in theta_vals:

                try:

                    A_l, C_l, P1_l, P2_l = self.mech.fk(t1, t4)

                    if self.mech.valid_config(O_l, B_l, A_l, C_l, P1_l):
                        P_l = P1_l
                    elif self.mech.valid_config(O_l, B_l, A_l, C_l, P2_l):
                        P_l = P2_l
                    else:
                        continue

                    P_g = self.mech.tf.l2g(P_l)

                    valid_angles.append((t1, t4))
                    valid_points.append(P_g)

                except:
                    pass

        return valid_angles, np.array(valid_points)

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
    
    def safe_workspace_circle(self, poly, buffer=0.7):

        center, r = self.largest_inscribed_circle(poly)
        
        r_safe = r * buffer

        return center, r_safe