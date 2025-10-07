#!/usr/bin/env python3
# Kronecker's Knumbers - self-inventorying numbers

import sys
from collections import Counter


def inventory(s: str) -> str:
    # count digits 0..9 in s
    cnt = Counter(s)
    items = []
    for d in range(10):
        c = cnt.get(str(d), 0)
        if c:
            items.append(str(c))
            items.append(str(d))
    return ''.join(items) if items else '0'


def classify(n_str: str) -> str:
    seen = {n_str: 0}
    seq = [n_str]
    if inventory(n_str) == n_str:
        return f"{n_str} is self-inventorying"
    cur = n_str
    for i in range(1, 16):
        nxt = inventory(cur)
        # If the newly produced number is a fixed point, then after i steps we reached a self-inventorying number
        if inventory(nxt) == nxt:
            return f"{n_str} is self-inventorying after {i} steps"
        if nxt in seen:
            k = i - seen[nxt]
            if k == 1:
                # Would imply immediate repeat (fixed point), but we already handled above
                return f"{n_str} is self-inventorying after {seen[nxt]} steps"
            return f"{n_str} enters an inventory loop of length {k}"
        seen[nxt] = i
        seq.append(nxt)
        cur = nxt
    return f"{n_str} can not be classified after 15 iterations"


def solve(stdin: str):
    out_lines = []
    for line in stdin.splitlines():
        line = line.strip()
        if not line:
            continue
        if line == '-1':
            break
        out_lines.append(classify(line))
    return '\n'.join(out_lines)


if __name__ == '__main__':
    data = sys.stdin.read()
    print(solve(data))
