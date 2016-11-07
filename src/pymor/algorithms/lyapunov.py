# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import numpy as np

from pymor.algorithms.to_matrix import to_matrix
from pymor.operators.interfaces import OperatorInterface
from pymor.operators.constructions import IdentityOperator, LincombOperator


try:
    import pymess

    class LyapunovEquation(pymess.equation):
        r"""Lyapunov equation class for pymess

        Represents a Lyapunov equation

        .. math::
            A X + X A^T + B B^T = 0

        if E is `None`, otherwise a generalized Lyapunov equation

        .. math::
            A X E^T + E X A^T + B B^T = 0.

        For the dual Lyapunov equation

        .. math::
            A^T X + X A + B^T B = 0, \\
            A^T X E + E^T X A + B^T B = 0,

        `opt.type` needs to be `pymess.MESS_OP_TRANSPOSE`.

        Parameters
        ----------
        opt
            pymess options structure.
        A
            The |Operator| A.
        E
            The |Operator| E or `None`.
        B
            The |Operator| B.
        """
        def __init__(self, opt, A, E, B):
            super().__init__(name='lyap_eqn', opt=opt, dim=A.source.dim)

            self.A = A
            self.E = E
            self.RHS = to_matrix(B)
            if opt.type == pymess.MESS_OP_TRANSPOSE:
                self.RHS = self.RHS.T
            self.p = []

        def AX_apply(self, op, y):
            y = self.A.source.from_data(np.array(y).T)
            if op == pymess.MESS_OP_NONE:
                x = self.A.apply(y)
            else:
                x = self.A.apply_adjoint(y)
            return np.matrix(x.data).T

        def EX_apply(self, op, y):
            if self.E is None:
                return y

            y = self.A.source.from_data(np.array(y).T)
            if op == pymess.MESS_OP_NONE:
                x = self.E.apply(y)
            else:
                x = self.E.apply_adjoint(y)
            return np.matrix(x.data).T

        def AINV_apply(self, op, y):
            y = self.A.source.from_data(np.array(y).T)
            if op == pymess.MESS_OP_NONE:
                x = self.A.apply_inverse(y)
            else:
                x = self.A.apply_inverse_adjoint(y)
            return np.matrix(x.data).T

        def EINV_apply(self, op, y):
            if self.E is None:
                return y

            y = self.A.source.from_data(np.array(y).T)
            if op == pymess.MESS_OP_NONE:
                x = self.E.apply_inverse(y)
            else:
                x = self.E.apply_inverse_adjoint(y)
            return np.matrix(x.data).T

        def ApEX_apply(self, op, p, idx_p, y):
            y = self.A.source.from_data(np.array(y).T)
            if op == pymess.MESS_OP_NONE:
                x = self.A.apply(y)
                if self.E is None:
                    x += p * y
                else:
                    x += p * self.E.apply(y)
            else:
                x = self.A.apply_adjoint(y)
                if self.E is None:
                    x += p.conjugate() * y
                else:
                    x += p.conjugate() * self.E.apply_adjoint(y)
            return np.matrix(x.data).T

        def ApEINV_apply(self, op, p, idx_p, y):
            y = self.A.source.from_data(np.array(y).T)
            E = IdentityOperator(self.A.source) if self.E is None else self.E

            if p.imag == 0:
                ApE = LincombOperator((self.A, E), (1, p.real))
            else:
                ApE = LincombOperator((self.A, E), (1, p))

            if op == pymess.MESS_OP_NONE:
                x = ApE.apply_inverse(y)
            else:
                x = ApE.apply_inverse_adjoint(y)
            return np.matrix(x.data).T

        def parameter(self, arp_p, arp_m, B=None, K=None):
            return None
except ImportError:
    pass


def solve_lyap(A, E, B, trans=False, me_solver=None, tol=None):
    """Find a factor of the solution of a Lyapunov equation

    Returns factor :math:`Z` such that :math:`Z Z^T` is approximately
    the solution :math:`X` of a Lyapunov equation (if E is `None`)

    .. math::
        A X + X A^T + B B^T = 0

    or generalized Lyapunov equation

    .. math::
        A X E^T + E X A^T + B B^T = 0.

    If trans is `True`, then solve (if E is `None`)

    .. math::
        A^T X + X A + B^T B = 0

    or

    .. math::
        A^T X E + E^T X A + B^T B = 0.

    Parameters
    ----------
    A
        The |Operator| A.
    E
        The |Operator| E or `None`.
    B
        The |Operator| B.
    trans
        If the dual equation needs to be solved.
    me_solver
        Method to use ('scipy', 'slycot', 'pymess_lyap', 'pymess_lradi').

        If `me_solver` is `None`, a method is chosen according to availability and priority:

            'pymess_lradi' (for bigger problems) > 'pymess_lyap' (for smaller problems) > 'slycot' > 'scipy'.
    tol
        Tolerance parameter.

    Returns
    -------
    Z
        Low-rank factor of the Lyapunov equation solution, |VectorArray| from `A.source`.
    """
    assert isinstance(A, OperatorInterface) and A.linear
    assert A.source == A.range
    assert isinstance(B, OperatorInterface) and B.linear
    assert not trans and B.range == A.source or trans and B.source == A.source
    assert E is None or isinstance(E, OperatorInterface) and E.linear and E.source == E.range == A.source
    assert me_solver is None or me_solver in ('scipy', 'slycot', 'pymess_lyap', 'pymess_lradi')

    if me_solver is None:
        import imp
        try:
            imp.find_module('pymess')
            if A.source.dim >= 1000:
                me_solver = 'pymess_lradi'
            else:
                me_solver = 'pymess_lyap'
        except ImportError:
            try:
                imp.find_module('slycot')
                me_solver = 'slycot'
            except ImportError:
                me_solver = 'scipy'

    if me_solver == 'scipy':
        if E is not None:
            raise NotImplementedError()
        import scipy.linalg as spla
        A_mat = to_matrix(A)
        B_mat = to_matrix(B)
        if not trans:
            X = spla.solve_lyapunov(A_mat, -B_mat.dot(B_mat.T))
        else:
            X = spla.solve_lyapunov(A_mat.T, -B_mat.T.dot(B_mat))
        from pymor.algorithms.cholp import cholp
        Z = cholp(X, copy=False)
    elif me_solver == 'slycot':
        import slycot
        A_mat = to_matrix(A)
        if E is not None:
            E_mat = to_matrix(E)
        B_mat = to_matrix(B)

        n = A_mat.shape[0]
        if not trans:
            C = -B_mat.dot(B_mat.T)
            trana = 'T'
        else:
            C = -B_mat.T.dot(B_mat)
            trana = 'N'
        dico = 'C'

        if E is None:
            U = np.zeros((n, n))
            X, scale, _, _, _ = slycot.sb03md(n, C, A_mat, U, dico, trana=trana)
        else:
            job = 'X'
            fact = 'N'
            Q = np.zeros((n, n))
            Z = np.zeros((n, n))
            uplo = 'L'
            X = C
            _, _, _, _, X, scale, _, _, _, _, _ = slycot.sg03ad(dico, job, fact, trana, uplo, n, A_mat, E_mat,
                                                                Q, Z, X)

        from pymor.algorithms.cholp import cholp
        Z = cholp(X, copy=False)
    elif me_solver == 'pymess_lyap':
        import pymess
        A_mat = to_matrix(A) if A.source.dim < 1000 else to_matrix(A, format='csc')
        if E is not None:
            E_mat = to_matrix(E) if E.source.dim < 1000 else to_matrix(E, format='csc')
        B_mat = to_matrix(B)
        if not trans:
            if E is None:
                Z = pymess.lyap(A_mat, None, B_mat)
            else:
                Z = pymess.lyap(A_mat, E_mat, B_mat)
        else:
            if E is None:
                Z = pymess.lyap(A_mat.T, None, B_mat.T)
            else:
                Z = pymess.lyap(A_mat.T, E_mat.T, B_mat.T)
    elif me_solver == 'pymess_lradi':
        import pymess
        opts = pymess.options()
        opts.adi.shifts.paratype = pymess.MESS_LRCFADI_PARA_ADAPTIVE_V
        if trans:
            opts.type = pymess.MESS_OP_TRANSPOSE
        if tol is not None:
            opts.rel_change_tol = tol
            opts.adi.res2_tol = tol
            opts.adi.res2c_tol = tol
        eqn = LyapunovEquation(opts, A, E, B)
        Z, status = pymess.lradi(eqn, opts)

    Z = A.source.from_data(np.array(Z).T)

    return Z
