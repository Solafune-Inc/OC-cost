import pulp
from collections import defaultdict
from pulp import LpMinimize
import numpy as np


class OCOpt:

    d = list()
    s = list()

    def __init__(self, n: int, m: int, beta: float, name="sample"):
        self.prob = pulp.LpProblem(name, sense=LpMinimize)
        self.n = n
        self.m = m
        self.variable = defaultdict(dict)
        self.cost = np.zeros((m, n))
        self.beta = beta

        for i in range(m - 1):
            self.s.append(1)
        self.s.append(n - 1)

        for j in range(n - 1):
            self.d.append(1)
        self.d.append(m - 1)

    def setVariable(self):
        # 変数の定義
        self.variable = [[pulp.LpVariable(f'pi_{i}_{j}', lowBound=0) for i in range(
            self.m)] for j in range(self.n)]

    def setObjective(self):
        self.prob += pulp.lpDot(self.cost, self.variable)

    def setConstrain(self):
        for j in range(self.n):
            self.prob += pulp.lpSum(self.variable[j]) == self.d[j]
        for i in range(self.m):
            self.prob += pulp.lpSum(self.variable[j][i]
                                    for j in range(self.n)) == self.s[i]

        self.prob += pulp.lpSum(self.s) == pulp.lpSum(self.d)

    def set_cost_matrix(self, cost: np.array):
        cost = np.insert(cost, self.n - 1, self.beta, axis=0)
        cost = np.insert(cost, self.m - 1, self.beta, axis=1)
        self.cost = cost


if __name__ == '__main__':
    opt = OCOpt(3, 4, 0.3)
    cost = np.array([[0.3, 0.4],
                     [0.2, 0.5],
                     [0.2, 0.1], ])

    opt.set_cost_matrix(cost)
    opt.setVariable()
    opt.setObjective()
    opt.setConstrain()

    result = opt.prob.solve()

    print('objective value: {}'.format(pulp.value(opt.prob.objective)))
    print('solution')
    for i in range(opt.m):
        for j in range(opt.n):
            print(f'{opt.variable[i][j]} = {pulp.value(opt.variable[i][j])}')
