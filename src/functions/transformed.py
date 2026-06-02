import numpy as np


class TransformedFunction:
    """
    Affine transformed objective.

        f(x) = base(Ax + b)
    grad:

        ∇f(x)=Aᵀ ∇base(Ax+b)
    """

    def __init__(
        self,
        base,
        dim=2,
        shift=None,
        rotation_deg=0.0,
        scale=None,
        A=None,
        b=None,
        randomize=False,
        seed=42,
        random_shift_range=5.0,
        random_scale_range=(0.25, 4.0),
    ):

        self.base = base

        rng = np.random.default_rng(seed)

        if randomize:
            shift = rng.uniform(
                -random_shift_range,
                random_shift_range,
                size=dim,
            )

            rotation_deg = rng.uniform(
                0.0,
                360.0,
            )

            log_min = np.log(random_scale_range[0])

            log_max = np.log(random_scale_range[1])

            scale = np.exp(
                rng.uniform(
                    log_min,
                    log_max,
                    size=dim,
                )
            )

        if shift is None:
            shift = np.zeros(dim)

        if scale is None:
            scale = np.ones(dim)

        self.shift = np.asarray(
            shift,
            dtype=float,
        )

        self.scale = np.asarray(
            scale,
            dtype=float,
        )

        self.rotation_deg = float(rotation_deg)

        if A is not None:
            self.A = np.asarray(
                A,
                dtype=float,
            )

        else:
            self.A = self._build_matrix(
                dim,
                self.rotation_deg,
                self.scale,
            )

        if b is not None:
            self.b = np.asarray(
                b,
                dtype=float,
            )

        else:
            self.b = self.shift

        self.AT = self.A.T

    def _build_matrix(
        self,
        dim,
        rotation_deg,
        scale,
    ):

        S = np.diag(scale)

        if dim == 2:
            theta = np.deg2rad(rotation_deg)

            c = np.cos(theta)
            s = np.sin(theta)

            R = np.array(
                [
                    [c, -s],
                    [s, c],
                ]
            )

            return R @ S

        if rotation_deg != 0:
            rng = np.random.default_rng(int(rotation_deg * 1000))

            Q, _ = np.linalg.qr(rng.normal(size=(dim, dim)))

            return Q @ S

        return S

    def transform(
        self,
        x,
    ):

        return self.A @ x + self.b

    def f(
        self,
        x,
    ):

        z = self.transform(x)

        return self.base.f(z)

    def grad(
        self,
        x,
    ):

        z = self.transform(x)

        grad_z = self.base.grad(z)

        return self.AT @ grad_z
