#!/usr/bin/env python3
# Total distance travelled by the Beetle
# Reads N and then 3N numbers (x y z) in order; prints total distance per rules.

import sys
from decimal import Decimal, getcontext, ROUND_HALF_UP
from math import sqrt

getcontext().prec = 50
PI = Decimal('3.14159265358979323846264338327950288419716939937510')

L = 10.0  # cube side length
EPS = 1e-9

# Face identifiers
T, Lf, Rf, Ff, Bk = 0, 1, 2, 3, 4

# Face properties: normal, origin, u, v (as 3D vectors)
faces = {
    T:  {"n": (0.0, 0.0, 1.0), "o": (0.0, 0.0, 10.0), "u": (1.0, 0.0, 0.0), "v": (0.0, 1.0, 0.0)},
    Lf: {"n": (-1.0, 0.0, 0.0), "o": (0.0, 0.0, 0.0), "u": (0.0, 1.0, 0.0), "v": (0.0, 0.0, 1.0)},
    Rf: {"n": (1.0, 0.0, 0.0), "o": (10.0, 0.0, 0.0), "u": (0.0, 1.0, 0.0), "v": (0.0, 0.0, 1.0)},
    Ff: {"n": (0.0, -1.0, 0.0), "o": (0.0, 0.0, 0.0), "u": (1.0, 0.0, 0.0), "v": (0.0, 0.0, 1.0)},
    Bk: {"n": (0.0, 1.0, 0.0), "o": (0.0, 10.0, 0.0), "u": (1.0, 0.0, 0.0), "v": (0.0, 0.0, 1.0)},
}

neighbors = {
    T:  {Lf, Rf, Ff, Bk},
    Lf: {T, Ff, Bk},
    Rf: {T, Ff, Bk},
    Ff: {T, Lf, Rf},
    Bk: {T, Lf, Rf},
}

# Shared edge axes for adjacent faces (unordered pairs)
# axis defined by a point p0 on the edge and a unit direction vector u
from itertools import permutations

def axis_for(fa, fb):
    a, b = fa, fb
    if (a, b) not in adj_pairs:
        if (b, a) not in adj_pairs:
            raise ValueError("Faces not adjacent")
        a, b = b, a
    return adj_pairs[(a, b)]

adj_pairs = {
    # T with sides
    (T, Ff): ((0.0, 0.0, 10.0), (1.0, 0.0, 0.0)),
    (T, Bk): ((0.0, 10.0, 10.0), (1.0, 0.0, 0.0)),
    (T, Lf): ((0.0, 0.0, 10.0), (0.0, 1.0, 0.0)),
    (T, Rf): ((10.0, 0.0, 10.0), (0.0, 1.0, 0.0)),
    # side-side adjacencies
    (Lf, Ff): ((0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    (Lf, Bk): ((0.0, 10.0, 0.0), (0.0, 0.0, 1.0)),
    (Rf, Ff): ((10.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    (Rf, Bk): ((10.0, 10.0, 0.0), (0.0, 0.0, 1.0)),
}

# Vector operations

def dot(a, b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

def cross(a, b):
    return (
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0],
    )

def add(a, b):
    return (a[0]+b[0], a[1]+b[1], a[2]+b[2])

def sub(a, b):
    return (a[0]-b[0], a[1]-b[1], a[2]-b[2])

def mul(a, s):
    return (a[0]*s, a[1]*s, a[2]*s)

def norm(a):
    return sqrt(dot(a, a))

def unit(a):
    n = norm(a)
    if n == 0:
        return a
    return (a[0]/n, a[1]/n, a[2]/n)

# Rotate point p around axis (p0, u) by ±90 degrees. Sign +1 => +90°, -1 => -90°
# Using simplified Rodrigues with cos=0, sin=±1: v' = v_par + sgn * (u x v)

def rotate90(p, p0, u, sgn):
    v = sub(p, p0)
    u = unit(u)
    v_par = mul(u, dot(u, v))
    uxv = cross(u, v)
    v_rot = add(v_par, mul(uxv, sgn))
    return add(p0, v_rot)

# Choose rotation sign so that rotating normal n_from around axis by ±90 brings it to n_to

def choose_sign_to_align(n_from, n_to, axis_u):
    # test both signs, pick with larger dot to n_to
    rpos = rotate90(add(n_from, (0.0,0.0,0.0)), (0.0,0.0,0.0), axis_u, +1)
    rneg = rotate90(add(n_from, (0.0,0.0,0.0)), (0.0,0.0,0.0), axis_u, -1)
    if dot(rpos, n_to) > dot(rneg, n_to):
        return +1
    else:
        return -1

# Map 3D point to 2D (u,v) in base face's plane coordinates

def to_face_coords(face_id, p):
    props = faces[face_id]
    o = props["o"]
    u = props["u"]
    v = props["v"]
    w = sub(p, o)
    return (dot(w, u), dot(w, v))

# Determine which face a point lies on (excluding bottom/edges per constraints)

def which_face(p):
    x, y, z = p
    if abs(z - 10.0) < EPS:
        return T
    if abs(x - 0.0) < EPS:
        return Lf
    if abs(x - 10.0) < EPS:
        return Rf
    if abs(y - 0.0) < EPS:
        return Ff
    if abs(y - 10.0) < EPS:
        return Bk
    raise ValueError("Point not on allowed faces")

# Distance along same face: arc with 60 degrees, chord length times pi/3

def arc_distance_same_face(p1, p2, face_id):
    u1, v1 = to_face_coords(face_id, p1)
    u2, v2 = to_face_coords(face_id, p2)
    chord = Decimal(str(sqrt((u1-u2)**2 + (v1-v2)**2)))
    return chord * (PI / Decimal(3))

# Unfold f2 (and optional mid) onto f1 plane and compute 2D distance

def unfolded_distance(p1, f1, p2, f2, mid=None):
    # Base plane is f1 as-is. Rotate p2 into f1 plane via optional mid.
    q = p2
    if mid is not None:
        # rotate from f2 onto mid
        p0, ax = axis_for(mid, f2)
        n_from = faces[f2]["n"]
        n_to = faces[mid]["n"]
        sgn = choose_sign_to_align(n_from, n_to, ax)
        q = rotate90(q, p0, ax, sgn)
        # rotate from mid onto f1
        p0, ax = axis_for(f1, mid)
        n_from = faces[mid]["n"]
        n_to = faces[f1]["n"]
        sgn = choose_sign_to_align(n_from, n_to, ax)
        q = rotate90(q, p0, ax, sgn)
    else:
        # directly rotate from f2 onto f1
        p0, ax = axis_for(f1, f2)
        n_from = faces[f2]["n"]
        n_to = faces[f1]["n"]
        sgn = choose_sign_to_align(n_from, n_to, ax)
        q = rotate90(q, p0, ax, sgn)
    # Now both p1 and q lie in f1 plane: compute 2D distance
    a_u, a_v = to_face_coords(f1, p1)
    b_u, b_v = to_face_coords(f1, q)
    d = Decimal(str(sqrt((a_u-b_u)**2 + (a_v-b_v)**2)))
    return d

# Compute shortest surface distance (excluding bottom) between points on different faces

def surface_distance(p1, f1, p2, f2):
    # consider 1-edge if adjacent, else all 2-edge paths through common neighbors
    best = None
    # 1-edge
    if f2 in neighbors[f1]:
        d = unfolded_distance(p1, f1, p2, f2, mid=None)
        best = d if best is None else min(best, d)
    # 2-edges via allowed mids
    for mid in neighbors[f1].intersection(neighbors[f2]):
        d = unfolded_distance(p1, f1, p2, f2, mid=mid)
        best = d if best is None else min(best, d)
    if best is None:
        # Should not happen among allowed faces
        raise RuntimeError("No valid path found between faces")
    return best

# Round per-leg to 2 decimals, HALF_UP

def round2(x: Decimal) -> Decimal:
    return x.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)


def solve(data_iter):
    tokens = []
    for tok in data_iter:
        if tok.strip() != '':
            tokens.append(tok)
    if not tokens:
        return
    it = iter(tokens)
    try:
        N = int(next(it))
    except StopIteration:
        return
    coords = []
    for _ in range(N):
        try:
            x = float(next(it)); y = float(next(it)); z = float(next(it))
        except StopIteration:
            raise SystemExit("Insufficient coordinates provided")
        coords.append( (x,y,z) )
    total = Decimal('0')
    for i in range(N-1):
        p1 = coords[i]
        p2 = coords[i+1]
        f1 = which_face(p1)
        f2 = which_face(p2)
        if f1 == f2:
            d = arc_distance_same_face(p1, p2, f1)
        else:
            d = surface_distance(p1, f1, p2, f2)
        total += round2(d)
    print(f"{total.quantize(Decimal('0.00'), rounding=ROUND_HALF_UP)}")


if __name__ == '__main__':
    data = sys.stdin.read().split()
    solve(iter(data))
