#!/usr/bin/env python3
# Product Test - count distinct sets of advancers (finalists and wildcards)
# Counts distinct pairs (F, W) with |F|=c1, |W|=c2 achievable by assigning
# each player a score within [lo_i, hi_i] (lo_i = challenge, hi_i = challenge + sum Aij),
# then ranking by score desc, tie by smaller seed.
# Algorithm: sweep thresholds t (max outside low-key) and y (max wildcard key),
# classify players relative to (t, y) using only extremes (lo/hi), and count
# assignments via 2D polynomial DP. Enforce uniqueness by requiring there exists
# an outside with A==t not eligible (ext_ok) or else subtract assignments that
# include all A==t eligible. Also enforce y uniqueness by subtracting assignments
# with no wildcard achieving exactly y.

import sys
from math import comb

# Helpers for 2D polynomial DP over x (finalists) and y (wildcards)

def shift_monomial(dp, fx, wy):
    H = len(dp)
    W = len(dp[0])
    if fx >= H or wy >= W:
        return [[0]*W for _ in range(H)]
    ndp = [[0]*W for _ in range(H)]
    for i in range(H-fx):
        row = dp[i]
        rown = ndp[i+fx]
        for j in range(W-wy):
            if row[j]:
                rown[j+wy] += row[j]
    return ndp


def mul_xy_pow(dp, m):
    if m <= 0:
        return dp
    H = len(dp); W = len(dp[0])
    for _ in range(m):
        ndp = [[0]*W for _ in range(H)]
        for i in range(H):
            row = dp[i]
            for j in range(W):
                v = row[j]
                if not v:
                    continue
                if i + 1 < H:
                    ndp[i+1][j] += v
                if j + 1 < W:
                    ndp[i][j+1] += v
        dp = ndp
    return dp


def mul_1x_pow(dp, p):
    if p <= 0:
        return dp
    H = len(dp); W = len(dp[0])
    # binomial along x
    combs = [comb(p, k) for k in range(p+1)]
    ndp = [[0]*W for _ in range(H)]
    for i in range(H):
        for j in range(W):
            v = dp[i][j]
            if not v:
                continue
            maxk = min(p, H-1 - i)
            for k in range(maxk+1):
                ndp[i+k][j] += v * combs[k]
    return ndp


def mul_1y_pow(dp, q):
    if q <= 0:
        return dp
    H = len(dp); W = len(dp[0])
    combs = [comb(q, k) for k in range(q+1)]
    ndp = [[0]*W for _ in range(H)]
    for i in range(H):
        row = dp[i]
        for j in range(W):
            v = row[j]
            if not v:
                continue
            maxk = min(q, W-1 - j)
            for k in range(maxk+1):
                ndp[i][j+k] += v * combs[k]
    return ndp


def coef_pipeline(c1, c2,
                  mF_only, mW_only, mFlex,
                  opt_F_only, opt_W_only,
                  force_mAyFlex_to_F=0,
                  remove_eBy=0, remove_oBy=0,
                  extra_shift_x=0, extra_shift_y=0):
    # Build dp and return coefficient at x^{c1} y^{c2}
    H = c1 + 1
    W = c2 + 1
    dp = [[0]*W for _ in range(H)]
    dp[0][0] = 1
    # base shift
    fx = mF_only + extra_shift_x
    wy = mW_only + extra_shift_y
    if fx > c1 or wy > c2:
        return 0
    dp = shift_monomial(dp, fx, wy)
    # mandatory flex: (x + y)^{mFlex}
    if force_mAyFlex_to_F:
        # move those to shift; reduce flex count
        if fx + force_mAyFlex_to_F > c1:
            return 0
        dp = shift_monomial(dp, force_mAyFlex_to_F, 0)
        mFlex_eff = mFlex - force_mAyFlex_to_F
    else:
        mFlex_eff = mFlex
    dp = mul_xy_pow(dp, mFlex_eff)
    # optional factors
    dp = mul_1x_pow(dp, opt_F_only)
    # reduce W-only by removing y-equal contributors
    if remove_eBy or remove_oBy:
        if remove_eBy > opt_W_only:
            # we'll cap but this should not happen if counts consistent
            remove_eBy = min(remove_eBy, opt_W_only)
        opt_W_only_eff = opt_W_only - remove_eBy - remove_oBy
    else:
        opt_W_only_eff = opt_W_only
    dp = mul_1y_pow(dp, opt_W_only_eff)
    return dp[c1][c2]


def solve(stdin: str) -> str:
    it = iter(stdin.strip().split())
    try:
        n = int(next(it)); m = int(next(it)); c1 = int(next(it)); c2 = int(next(it))
    except StopIteration:
        return ''
    players = []  # (seed, lo, hi)
    for _ in range(n):
        seed = int(next(it))
        A = [int(next(it)) for _ in range(m)]
        C = int(next(it))
        lo = C
        hi = C + sum(A)
        players.append((seed, lo, hi))
    K = c1 + c2

    # Keys: A = (lo, -seed), B = (hi, -seed)
    A_keys = []
    B_keys = []
    for seed, lo, hi in players:
        A_keys.append((lo, -seed))
        B_keys.append((hi, -seed))

    t_values = sorted(set(A_keys))
    y_values = sorted(set(A_keys) | set(B_keys))

    total_count = 0

    for t in t_values:
        for y in y_values:
            if not (t < y):
                continue
            # classification counts
            mF_only = mW_only = mFlex = 0
            mAyFlex = mAyWonly = 0
            mByCount = 0

            eF_only = eW_only = 0
            eByCount = 0

            oF_only = oW_only = 0
            oByCount = 0

            E_F_only = E_W_only = 0  # for force-include case
            ext_ok_init = False

            invalid = False
            for idx in range(n):
                seed, lo, hi = players[idx]
                a = A_keys[idx]
                b = B_keys[idx]
                inCt = b > t
                if a == t and not inCt:
                    ext_ok_init = True
                if not inCt:
                    continue  # outside only
                if a > t:
                    # mandatory in S
                    if b <= y:
                        mW_only += 1
                        if b == y:
                            mByCount += 1
                        if a == y:
                            mAyWonly += 1
                    elif a > y:
                        mF_only += 1
                    else:
                        # a <= y < b
                        mFlex += 1
                        if a == y:
                            mAyFlex += 1
                elif a == t:
                    # optional in S
                    if b <= y:
                        eW_only += 1
                        if b == y:
                            eByCount += 1
                        E_W_only += 1
                    else:  # b > y
                        eF_only += 1
                        E_F_only += 1
                else:  # a < t
                    if b <= t:
                        continue  # outside only
                    if b <= y:
                        oW_only += 1
                        if b == y:
                            oByCount += 1
                    else:
                        oF_only += 1

            # base capacity checks
            if mF_only > c1 or mW_only > c2:
                continue
            if mF_only + mW_only + mFlex > K:
                continue

            # total coefficient counting all y cases
            coef_total = coef_pipeline(
                c1, c2,
                mF_only, mW_only, mFlex,
                eF_only + oF_only, eW_only + oW_only,
                force_mAyFlex_to_F=0,
                remove_eBy=0, remove_oBy=0,
                extra_shift_x=0, extra_shift_y=0,
            )
            if coef_total == 0:
                continue

            # subtract assignments with no y-hit if needed
            # y-hit forced if any mandatory W-only with A==y or any mandatory with B==y
            if mAyWonly == 0 and mByCount == 0:
                coef_no_y = coef_pipeline(
                    c1, c2,
                    mF_only, mW_only, mFlex,
                    eF_only + oF_only, eW_only + oW_only,
                    force_mAyFlex_to_F=mAyFlex,
                    remove_eBy=eByCount, remove_oBy=oByCount,
                    extra_shift_x=0, extra_shift_y=0,
                )
                coef_total -= coef_no_y
                if coef_total <= 0:
                    continue

            # enforce outside uniqueness: if no outsider with A==t and not inCt, then
            # subtract assignments where all A==t eligible are included in S
            if not ext_ok_init and (E_F_only + E_W_only) > 0:
                inv_total = coef_pipeline(
                    c1, c2,
                    mF_only + E_F_only, mW_only + E_W_only, mFlex,
                    eF_only + oF_only - E_F_only, eW_only + oW_only - E_W_only,
                    force_mAyFlex_to_F=0,
                    remove_eBy=0, remove_oBy=0,
                    extra_shift_x=0, extra_shift_y=0,
                )
                if mAyWonly == 0 and mByCount == 0:
                    # attempt to remove y-hit: if E_W_only includes any with b==y, cannot avoid y
                    # Count how many among A==t and inCt True have b==y
                    eBy_in_E = eByCount  # all eBy belong to A==t & inCt True
                    if eBy_in_E > 0:
                        inv_no_y = 0
                    else:
                        inv_no_y = coef_pipeline(
                            c1, c2,
                            mF_only + E_F_only, mW_only + E_W_only, mFlex,
                            eF_only + oF_only - E_F_only, eW_only + oW_only - E_W_only,
                            force_mAyFlex_to_F=mAyFlex,
                            remove_eBy=eByCount, remove_oBy=oByCount,
                            extra_shift_x=0, extra_shift_y=0,
                        )
                else:
                    inv_no_y = 0
                coef_total -= (inv_total - inv_no_y)
                if coef_total <= 0:
                    continue

            total_count += coef_total

    return str(total_count)


if __name__ == '__main__':
    print(solve(sys.stdin.read()))
