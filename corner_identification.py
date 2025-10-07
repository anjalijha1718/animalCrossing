#!/usr/bin/env python3
# Corner identification (edge detection on RLE images)
# Efficient streaming using row-boundary analysis; avoids pixel expansion

import sys
from bisect import bisect_right


def read_images(data_iter):
    while True:
        try:
            w = int(next(data_iter))
        except StopIteration:
            return
        if w == 0:
            return
        runs = []
        total = 0
        while True:
            v = int(next(data_iter)); L = int(next(data_iter))
            if v == 0 and L == 0:
                break
            runs.append((v, L))
            total += L
        if total % w != 0:
            raise SystemExit("Invalid image: total pixels not multiple of width")
        yield w, runs


def prefix_sums(runs):
    ps = [0]
    s = 0
    for _, L in runs:
        s += L
        ps.append(s)
    return ps  # len = len(runs)+1, ps[i] = sum of first i lengths


def value_at_pos(runs, ps, pos):
    # pos in [0..ps[-1)-1]
    i = bisect_right(ps, pos) - 1
    return runs[i][0]


def build_row_events(width, runs, ps):
    # Map row index -> list of (col, new_value) for boundaries inside the row
    events = {}
    P = ps[-1]
    # consider boundary at each run end except final
    for i in range(1, len(ps)-0):
        if i == len(ps)-1:
            break
        pos = ps[i]
        c = pos % width
        if c != 0:
            r = pos // width
            ev = events.setdefault(r, [])
            ev.append((c, runs[i][0]))  # new value after boundary
    # sort columns per row
    for r in events:
        events[r].sort()
    return events


def row_segments(width, runs, ps, r, row_events):
    # Returns list of (start_col, end_col, value) covering [0,width)
    start_pos = r * width
    cur_val = value_at_pos(runs, ps, start_pos)
    evs = row_events.get(r, [])
    segs = []
    last = 0
    for c, new_val in evs:
        if c > last:
            segs.append((last, c, cur_val))
        cur_val = new_val
        last = c
    if last < width:
        segs.append((last, width, cur_val))
    return segs


def value_from_segments(segs, col):
    # segs are disjoint sorted; binary search
    lo, hi = 0, len(segs)
    while lo < hi:
        mid = (lo + hi) // 2
        s, e, v = segs[mid]
        if col < s:
            hi = mid
        elif col >= e:
            lo = mid + 1
        else:
            return v
    # shouldn't happen
    return segs[-1][2]


def collect_breaks(width, segs_top, segs_mid, segs_bot):
    pts = set([0, width])
    for segs in (segs_top, segs_mid, segs_bot):
        for s, e, _ in segs:
            for b in (s, e):
                for d in (-1, 0, 1):
                    x = b + d
                    if 0 <= x <= width:
                        pts.add(x)
    br = sorted(pts)
    # merge duplicates implicitly; return consecutive pairs as intervals
    return br


def compute_interval_value(width, r, s, e, segs_top, segs_mid, segs_bot):
    # Pick representative c in [s, e-1]
    if s >= e:
        return 0
    c = s
    V = value_from_segments(segs_mid, c)
    # neighbor helper with boundary handling
    def val_top(x):
        if r == 0:
            return V
        if x < 0 or x >= width:
            return V
        return value_from_segments(segs_top, x)
    def val_bot(x):
        # height unknown here; handled by caller via segs_bot presence; when bottom missing, segs_bot empty -> treat as V
        if not segs_bot:
            return V
        if x < 0 or x >= width:
            return V
        return value_from_segments(segs_bot, x)
    def val_mid(x):
        if x < 0 or x >= width:
            return V
        return value_from_segments(segs_mid, x)
    vals = [
        abs(V - val_top(c-1)), abs(V - val_top(c)), abs(V - val_top(c+1)),
        abs(V - val_mid(c-1)), abs(V - val_mid(c+1)),
        abs(V - val_bot(c-1)), abs(V - val_bot(c)), abs(V - val_bot(c+1)),
    ]
    return max(vals)


def emit_run(out_runs, val, length):
    if length <= 0:
        return
    if out_runs and out_runs[-1][0] == val:
        out_runs[-1] = (val, out_runs[-1][1] + length)
    else:
        out_runs.append((val, length))


def aligned_boundary_rows(width, ps):
    rows = set()
    for i in range(1, len(ps)-0):
        if i == len(ps)-1:
            break
        pos = ps[i]
        if pos % width == 0:
            r = pos // width
            rows.add(r)
    return rows


def process_image(width, runs):
    ps = prefix_sums(runs)
    P = ps[-1]
    H = P // width

    row_events = build_row_events(width, runs, ps)
    interesting_rows = set(row_events.keys())
    rows_to_process = set()
    for r in list(interesting_rows):
        for dr in (-1, 0, 1):
            rr = r + dr
            if 0 <= rr < H:
                rows_to_process.add(rr)

    aligned_rows = aligned_boundary_rows(width, ps)
    split_rows = set(rows_to_process)
    split_rows.update(aligned_rows)
    split_rows.update({r-1 for r in aligned_rows if 0 <= r-1 < H})
    # Also split at the row just after an aligned boundary, because only two adjacent rows around
    # the boundary can have non-zero vertical-difference contributions.
    split_rows.update({r+1 for r in aligned_rows if r+1 < H})
    split_points = sorted({0, H} | split_rows)

    out_runs = []

    def segs_for_row(r):
        # returns segments; for nonexistent top/bottom, return [] to signal boundary
        return row_segments(width, runs, ps, r, row_events)

    # iterate over row intervals between split points
    for idx in range(len(split_points)-1):
        a = split_points[idx]
        b = split_points[idx+1]
        # process rows [a, b)
        cur = a
        while cur < b:
            if cur in rows_to_process:
                # complex row: compute using segments overlay
                segs_mid = segs_for_row(cur)
                segs_top = segs_for_row(cur-1) if cur-1 >= 0 else []
                segs_bot = segs_for_row(cur+1) if cur+1 < H else []
                br = collect_breaks(width, segs_top if segs_top else [(0, width, 0)], segs_mid, segs_bot if segs_bot else [(0, width, 0)])
                for j in range(len(br)-1):
                    s, e = br[j], br[j+1]
                    if s >= e:
                        continue
                    val = compute_interval_value(width, cur, s, e, segs_top, segs_mid, segs_bot)
                    emit_run(out_runs, val, e - s)
                cur += 1
            else:
                # boring block up to next special row or b
                nxt = cur + 1
                while nxt < b and (nxt not in rows_to_process):
                    nxt += 1
                # rows [cur, nxt) are uniform with uniform neighbors; output per row is constant and identical across rows
                # Compute for representative row cur
                V = value_at_pos(runs, ps, cur * width)
                if cur == 0:
                    Vtop = V
                else:
                    Vtop = value_at_pos(runs, ps, (cur-1) * width)
                if cur+1 >= H:
                    Vbot = V
                else:
                    Vbot = value_at_pos(runs, ps, (cur+1) * width)
                val = max(abs(V - Vtop), abs(V - Vbot))
                length = (nxt - cur) * width
                emit_run(out_runs, val, length)
                cur = nxt

    return out_runs


def solve(stdin: str) -> str:
    it = iter(stdin.strip().split())
    out_lines = []
    for w, runs in read_images(it):
        out_runs = process_image(w, runs)
        out_lines.append(str(w))
        for v, L in out_runs:
            out_lines.append(f"{v} {L}")
        out_lines.append("0 0")
    out_lines.append("0")
    return "\n".join(out_lines)


if __name__ == '__main__':
    data = sys.stdin.read()
    print(solve(data))
