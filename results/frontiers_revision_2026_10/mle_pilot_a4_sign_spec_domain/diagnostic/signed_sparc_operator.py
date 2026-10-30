"""Isolated signed SPARC baryonic operator. Not a likelihood and not activated.

Published component convention, with the frozen one-common-Upsilon restriction:
    B = Upsilon * (Vdisk*|Vdisk| + Vbulge*|Vbulge|) + Vgas*|Vgas|.
Negative B is returned as a signed squared speed. It is not passed to sqrt.
"""
from __future__ import annotations

import numpy as np

STATUS_DEFINED = "DEFINED_ON_FROZEN_DICTIONARY"
STATUS_INVALID = "INVALID_REAL_CIRCULAR_SPEED"


def signed_square(velocity_kms):
    """Component contribution v*|v| in (km/s)^2. Negative velocity stays negative."""
    v = np.asarray(velocity_kms, dtype=float)
    return v * np.abs(v)


def baryonic_signed_v2(upsilon, v_disk_kms, v_bulge_kms, v_gas_kms):
    """Signed net baryonic squared speed. One shared Upsilon multiplies disk and bulge."""
    u = float(upsilon)
    stellar = signed_square(v_disk_kms) + signed_square(v_bulge_kms)
    return u * stellar + signed_square(v_gas_kms)


def baryonic_radial_acceleration(signed_v2_kms2, r_kpc, *, kpc_m, km_m):
    """Signed Newtonian acceleration v^2/r in m/s^2. No floor and no abs."""
    v2 = np.asarray(signed_v2_kms2, dtype=float)
    r_m = np.asarray(r_kpc, dtype=float) * float(kpc_m)
    out = np.full(np.broadcast(v2, r_m).shape, np.nan, dtype=float)
    ok = np.isfinite(v2) & np.isfinite(r_m) & (r_m > 0.0)
    out[ok] = (v2[ok] * (float(km_m) ** 2)) / r_m[ok]
    return out


def model_circular_speed_domain(total_signed_v2_kms2):
    """Real nonnegative circular speed only where the summed v^2 is nonnegative.

    Negative or nonfinite totals are marked invalid. No sqrt(abs) and no max(v2, 0).
    """
    total = np.asarray(total_signed_v2_kms2, dtype=float)
    defined = np.isfinite(total) & (total >= 0.0)
    speed = np.full(total.shape, np.nan, dtype=float)
    speed[defined] = np.sqrt(total[defined])
    status = np.where(defined, STATUS_DEFINED, STATUS_INVALID)
    return {
        "signed_v2_kms2": total,
        "defined": defined,
        "circular_speed_kms": speed,
        "status": status,
    }


def distance_scaled_signed_square(velocity_kms, distance_scale):
    """(sqrt(s)*v)*|sqrt(s)*v| for s>0, returned next to s*v*|v|."""
    s = float(distance_scale)
    if not (np.isfinite(s) and s > 0.0):
        raise ValueError("distance scale must be positive and finite")
    v = np.asarray(velocity_kms, dtype=float)
    scaled = np.sqrt(s) * v
    return signed_square(scaled), s * signed_square(v)
