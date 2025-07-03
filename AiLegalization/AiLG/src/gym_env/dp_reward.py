import math


def preprocess(w, x, y, origin_x, origin_y):
    tu = [(_x, _y, _w, o_x, o_y) for _w, _x, _y, o_x, o_y in zip(w, x, y, origin_x, origin_y)]
    tu.sort(key=lambda t: (math.floor(t[1] / 8 + 0.5), t[0] + t[2] // 2))
    res = []
    for i in tu:
        if len(res) == 0 or math.floor(res[-1][0][1] / 8 + 0.5) != math.floor(i[1] / 8 + 0.5):
            res.append([i])
        else:
            res[-1].append(i)
    return res


def dp_solve(n, tu, origin_tu, M, N):
    w, x, y = [], [], []
    origin_x = []
    origin_y = []
    for i, (_x, _y, _w) in enumerate(tu):
        x.append(_x)
        y.append(_y)
        w.append(_w)
        origin_x.append(origin_tu[i][0])
        origin_y.append(origin_tu[i][1])

    def dp_on_row(xx):
        # print(len(xx)* M)
        dp = [0 for _ in range(M)]
        res = [[-1 for _ in range(M)] for _ in range(len(xx))]
        minn = 0
        t = [0 for _ in range(M)]
        for i, (_x, _y, _w, o_x, o_y) in enumerate(xx):
            for k in range(len(dp)):
                t[k] = dp[k]
                dp[k] = float('inf')
            # print(t)
            min_value = -1
            pre_pos = -1
            for pos in range(math.floor(minn + _w), M):
                if min_value == -1 or (min_value != -1 and min_value > t[math.floor(pos - _w)]):
                    min_value = t[math.floor(pos - _w)]
                    pre_pos = pos - _w
                if dp[pos] > min_value + (pos - _w - o_x) ** 2:
                    dp[pos] = min_value + (pos - _w - o_x) ** 2
                    res[i][pos] = pre_pos
            minn = minn + _w
        min_all = float('inf')
        res_pos = -1
        for pos in range(math.floor(minn), M):
            if min_all > dp[pos]:
                min_all = dp[pos]
                res_pos = pos
        # print(min_all)
        res_xx = []
        for i in reversed(range(len(xx))):
            res_xx.append((res_pos - xx[i][2], xx[i][1], xx[i][2], xx[i][3], xx[i][4]))
            res_pos = res[i][math.floor(res_pos)]
            # print(res_pos,)
        return res_xx

    tp = preprocess(w, x, y, origin_x, origin_y)
    # tp = [(_x, _y, _w) for _w, _x, _y in zip(w, x, y)]
    res_x = []
    res_y = []
    ind = 0
    origin_x = []
    origin_y = []
    for i, xx in enumerate(tp):
        res_xx = dp_on_row(xx)
        for i, (_x, _y, _w, o_x, o_y) in enumerate(res_xx):
            res_x.append(_x)
            res_y.append(math.floor(_y / 8 + 0.5) * 8)
            origin_x.append(o_x)
            origin_y.append(o_y)
            w[ind] = _w
            ind += 1
    # print(tp)

    return list(zip(origin_x, origin_y, w)), list(zip(res_x, res_y, w))
