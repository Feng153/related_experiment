import math


def cal_distance(p1, p2):
    return math.sqrt(math.pow((p2[0] - p1[0]), 2) + math.pow((p2[1] - p1[1]), 2))


p1 = [0, 0]
p2 = [1, 2]
dis = cal_distance(p1, p2)
print(dis)
