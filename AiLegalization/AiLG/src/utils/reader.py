'''
N: y的约束，PLB阵列的列数
M: x的约束，PLB阵列的行数
x: 高度
y: 宽度
w: macro的高度

'''
import math


def get_list(d, key=lambda _a: (math.floor(_a[2] / 8 + 0.5) * 8, _a[1] + _a[0] / 2)):
    lis = []
    d.sort(key=key)
    j = 0
    pre = -1
    for i in range(len(d)):
        key = math.floor(d[i][2] / 8 + 0.5) * 8
        if key == pre:
            lis[j].append(d[i])
        else:
            lis.append([d[i]])
    return lis


def get_dic(d, key=lambda _a: (math.floor(_a[2] / 8 + 0.5), _a[1] + _a[0] / 2)):
    dic = {}
    d.sort(key=key)
    for i in range(len(d)):
        key = math.floor(d[i][2] / 8 + 0.5) * 8
        if key not in dic:
            dic[key] = [(i, d[i])]
        else:
            dic[key].append((i, d[i]))
    return dic


def read_data(file_name):
    f = open(file_name, 'r')
    # print(f.readline().split(' '))
    s = f.readline().split(' ')
    # print(s)
    sites_num = int(s[0]) * 8
    rows_num = int(s[1])
    n = int(s[2].strip())
    x = []
    y = []
    w = []
    h = []
    for i in range(n):
        s = f.readline().split(' ')
        w.append(float(s[0]))
        y.append(float(s[1]))
        x.append(float(s[2].strip()))
        h.append(8)
    tu = list(zip(x, y, w, h))
    # tu.sort(key=lambda t: (math.floor(t[1] / 8 + 0.5), t[0] + t[2] // 2))
    f.close()
    return tu, n, rows_num, sites_num
