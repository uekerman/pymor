import math
import numpy as np
from IPython.core.display import display
from ipywidgets import widgets, Layout
from matplotlib.cm import get_cmap

import ngsolve as ngs
from ngsolve.webgui import encodeData, NGSWebGuiWidget, render_js_code, html_template

from pymor.discretizers.builtin.grids.referenceelements import triangle, square
from pymor.discretizers.builtin.grids.constructions import flatten_grid


try:
    __IPYTHON__
    _IN_IPYTHON = True
except NameError:
    _IN_IPYTHON = False

try:
    import google.colab

    _IN_GOOGLE_COLAB = True
except ImportError:
    _IN_GOOGLE_COLAB = False


class WebGLScene(widgets.VBox):
    def __init__(self, cf, mesh, order, min_, max_, draw_vol, draw_surf, autoscale, deformation, interpolate_multidim,
                 animate):
        self.cf = cf
        self.mesh = mesh
        self.order = order
        self.min = min_
        self.max = max_
        self.draw_vol = draw_vol
        self.draw_surf = draw_surf
        self.autoscale = autoscale
        self.interpolate_multidim = interpolate_multidim
        self.animate = animate

        self.deformation = deformation
        self.widget = NGSWebGuiWidget()
        self.widget.value = self.GetData()

        super().__init__(children=[self.widget])

    def GetData(self, set_minmax=True):
        d = BuildRenderData(self.mesh, self.cf, self.order)

        if set_minmax:
            if self.min is not None:
                d['funcmin'] = self.min
            if self.max is not None:
                d['funcmax'] = self.max
            d['autoscale'] = self.autoscale
        return d

    def Redraw(self):
        d = self.GetData(set_minmax=False)
        self.widget.value = d


bezier_trig_trafos = {}  # cache trafos for different orders


def BuildRenderData(grid, u, order=1):
    d = {}
    d['ngsolve_version'] = ngs.__version__
    d['mesh_dim'] = 2

    d['order2d'] = order

    d['draw_vol'] = False
    d['draw_surf'] = True

    d['show_wireframe'] = False
    d['show_mesh'] = False
    d['funcdim'] = 1
    if order > 0:
        og = order
        d['show_wireframe'] = True
        d['show_mesh'] = True

        # transform point-values to Bernsteinbasis
        def Binomial(n, i):
            return math.factorial(n) / math.factorial(i) / math.factorial(n - i)

        def Bernstein(x, i, n):
            return Binomial(n, i) * x ** i * (1 - x) ** (n - i)

        Bvals = ngs.Matrix(og + 1, og + 1)
        for i in range(og + 1):
            for j in range(og + 1):
                Bvals[i, j] = Bernstein(i / og, j, og)
        iBvals = Bvals.I

        Bezier_points = []

        # TODO: Quads
        #         ipts = [(i/og,0) for i in range(og+1)] + [(0, i/og) for i in range(og+1)] + [(i/og,1.0-i/og) for i in range(og+1)]
        #         ir_trig = ngs.IntegrationRule(ipts, [0,]*len(ipts))
        #         ipts = [(i/og,0) for i in range(og+1)] + [(0, i/og) for i in range(og+1)] + [(i/og,1.0) for i in range(og+1)] + [(1.0, i/og) for i in range(og+1)]
        #         ir_quad = ngs.IntegrationRule(ipts, [0,]*len(ipts))

        #         vb = [ngs.VOL, ngs.BND][mesh.dim-2]
        #         cf = func1 if draw_surf else func0
        #         pts = mesh.MapToAllElements({ngs.ET.TRIG: ir_trig, ngs.ET.QUAD: ir_quad}, vb)
        #         pmat = cf(pts)

        subentities, coordinates, entity_map = flatten_grid(grid)

        codim = 2
        if grid.reference_element == triangle:
            if codim == 2:
                vertices = np.zeros((len(coordinates), 3))
                vertices[:, :-1] = coordinates
                indices = subentities
            else:
                vertices = np.zeros((len(subentities) * 3, 3))
                VERTEX_POS = coordinates[subentities]
                vertices[:, 0:2] = VERTEX_POS.reshape((-1, 2))
                indices = np.arange(len(subentities) * 3, dtype=np.uint32)
        else:
            if codim == 2:
                vertices = np.zeros((len(coordinates), 3))
                vertices[:, :-1] = coordinates
                indices = np.vstack((subentities[:, 0:3], subentities[:, [0, 2, 3]]))
            else:
                num_entities = len(subentities)
                vertices = np.zeros((num_entities * 6, 3))
                VERTEX_POS = coordinates[subentities]
                vertices[0:num_entities * 3, 0:2] = VERTEX_POS[:, 0:3, :].reshape((-1, 2))
                vertices[num_entities * 3:, 0:2] = VERTEX_POS[:, [0, 2, 3], :].reshape((-1, 2))
                indices = np.arange(len(subentities) * 6, dtype=np.uint32)

        #         print(vertices.shape)
        #         print(indices.shape)
        # todo: quads
        ne = len(indices)

        pmat = np.zeros((3 * ne, 2, 4))
        for i in range(ne):
            for j in range(3):
                pmat[3 * i + j][0][:3] = vertices[indices[i][j]]
                pmat[3 * i + j][1][:3] = vertices[indices[i][(j + 1) % 3]]

        # pmat = pmat.reshape(-1, og+1, 4)
        #         print('pmat', pmat)
        BezierPnts = np.tensordot(iBvals.NumPy(), pmat, axes=(1, 1))
        #         print(BezierPnts)
        for i in range(og + 1):
            Bezier_points.append(encodeData(BezierPnts[i]))

        d['Bezier_points'] = Bezier_points
        d['edges'] = Bezier_points

        ndtrig = int((og + 1) * (og + 2) / 2)

        if og in bezier_trig_trafos.keys():
            iBvals_trig = bezier_trig_trafos[og]
        else:
            def BernsteinTrig(x, y, i, j, n):
                return math.factorial(n) / math.factorial(i) / math.factorial(j) / math.factorial(n - i - j) \
                       * x ** i * y ** j * (1 - x - y) ** (n - i - j)

            Bvals = ngs.Matrix(ndtrig, ndtrig)
            ii = 0
            for ix in range(og + 1):
                for iy in range(og + 1 - ix):
                    jj = 0
                    for jx in range(og + 1):
                        for jy in range(og + 1 - jx):
                            Bvals[ii, jj] = BernsteinTrig(ix / og, iy / og, jx, jy, og)
                            jj += 1
                    ii += 1
            iBvals_trig = Bvals.I
            bezier_trig_trafos[og] = iBvals_trig

        # Bezier_points = [ [] for i in range(ndtrig) ]
        Bezier_points = []
        values = u.to_numpy()[0][entity_map]
        #         values = u[0][entity_map]
        pmat = np.zeros((ne, 3, 4))
        for i in range(ne):
            for j in range(3):
                pmat[i][j][:3] = vertices[indices[i][j]]
                pmat[i][j][3] = values[indices[i][j]]
        funcmin = np.min(pmat[:, :, 3])
        funcmax = np.max(pmat[:, :, 3])

        #         pmat = pmat.reshape(-1, len(ir_trig), 4)
        BezierPnts = np.tensordot(iBvals_trig.NumPy(), pmat, axes=(1, 1))

        for i in range(ndtrig):
            Bezier_points.append(encodeData(BezierPnts[i]))

        d['Bezier_trig_points'] = Bezier_points
        bb = grid.bounding_box()
        center = np.zeros(3)
        center[0:2] = (bb[1] - bb[0]) / 2
        d['mesh_center'] = tuple(center)
        d['mesh_radius'] = np.linalg.norm(center) * 1.1

    d['funcmin'] = funcmin
    d['funcmax'] = funcmax
    #     print(d)
    return d
# BuildRenderData(data['grid'], U, order=1)


def visualize_ngsolve(grid, U, bounding_box=([0, 0], [1, 1]), codim=2, title=None, legend=None,
                    separate_colorbars=False, rescale_colorbars=False, columns=2,
         color_map=get_cmap('viridis')):
    if isinstance(U, tuple):
        raise NotImplementedError('tuples of VectorArrays cannot yet be visualized with the ngsolve backend')

    return WebGLScene(U, grid, order=1, min_=None, max_=None, draw_vol=True, draw_surf=True,
                       autoscale=True,
                       deformation=False, interpolate_multidim=False, animate=False)