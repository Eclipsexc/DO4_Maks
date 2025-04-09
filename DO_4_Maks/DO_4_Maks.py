import numpy as np
from collections import Counter

def check_problem_type(ai, bj, cij):
    total_ai = sum(ai)
    total_bj = sum(bj)

    if total_ai == total_bj:
        print("Задача є збалансованою.")
        return ai, bj, cij, False, None

    if total_ai < total_bj:
        diff = total_bj - total_ai
        print(f"Задача є незбалансована. Додається фіктивний Ai із запасом {diff}.")
        ai = np.append(ai, diff)
        fake_row = np.zeros((1, cij.shape[1]))
        cij = np.vstack([cij, fake_row])
        return ai, bj, cij, True, 'supplier'
    else:
        diff = total_ai - total_bj
        print(f"Задача є відкритою. Додається фіктивний Bj з потребою {diff}.")
        bj = np.append(bj, diff)
        fake_col = np.zeros((cij.shape[0], 1))
        cij = np.hstack([cij, fake_col])
        return ai, bj, cij, True, 'consumer'

def print_plan_costs_format(plan, cij, ai, bj, title):
    m, n = plan.shape
    header = ["     "] + [f"B{j+1}" for j in range(n)] + [" aᵢ"]
    rows = []
    for i in range(m):
        row = [f"A{i+1}"]
        for j in range(n):
            if plan[i, j] > 0:
                row.append(f"{int(cij[i, j])}[{int(plan[i, j])}]")
            else:
                row.append("")
        row.append(f"{int(ai[i])}")
        rows.append(row)
    last_row = ["bⱼ"] + [f"{int(d)}" for d in bj] + [""]

    col_widths = [max(len(str(item)) for item in col) + 2 for col in zip(*([header] + rows + [last_row]))]

    def format_row(row):
        return "".join(str(item).center(w) for item, w in zip(row, col_widths))

    print(f"\n{title}\n")
    print(format_row(header))
    for row in rows:
        print(format_row(row))
    print(format_row(last_row))

def print_cost_expression(plan, cij):
    terms = []
    total = 0
    m, n = plan.shape
    for i in range(m):
        for j in range(n):
            amount = int(plan[i, j])
            cost = int(cij[i, j])
            if amount > 0:
                terms.append(f"{cost}*{amount}")
                total += cost * amount
    expression = " + ".join(terms)
    print(f"Z = {expression} = {total}")

def method_northwest_corner(ai, bj):
    ai = ai.copy()
    bj = bj.copy()
    m, n = len(ai), len(bj)
    x = np.zeros((m, n), dtype=int)
    i = j = 0
    while i < m and j < n:
        value = min(ai[i], bj[j])
        x[i, j] = value
        ai[i] -= value
        bj[j] -= value
        if ai[i] == 0 and bj[j] == 0:
            if j + 1 < n:
                j += 1
            elif i + 1 < m:
                i += 1
            else:
                break
        elif ai[i] == 0:
            i += 1
        elif bj[j] == 0:
            j += 1
    return x

def method_least_cost(cij, ai, bj):
    ai = ai.copy()
    bj = bj.copy()
    m, n = len(ai), len(bj)
    x = np.zeros((m, n), dtype=int)

    fake_supplier_index = np.where(~cij.any(axis=1))[0]
    has_fake_supplier = len(fake_supplier_index) > 0
    fake_supplier_index = fake_supplier_index[0] if has_fake_supplier else -1

    cost_indices = [(i, j) for i in range(m) for j in range(n)]
    cost_indices_main = [(i, j) for (i, j) in cost_indices if i != fake_supplier_index]
    cost_indices_fake = [(i, j) for (i, j) in cost_indices if i == fake_supplier_index]

    cost_indices_main.sort(key=lambda x: cij[x[0], x[1]])
    cost_indices_fake.sort(key=lambda x: cij[x[0], x[1]])

    cost_indices = cost_indices_main + cost_indices_fake

    for i, j in cost_indices:
        if ai[i] == 0 or bj[j] == 0:
            continue
        value = min(ai[i], bj[j])
        x[i, j] = value
        ai[i] -= value
        bj[j] -= value
    return x

def метод_потенціалів(ai, bj, cij, initial_plan):
    C = np.copy(cij)
    X = initial_plan.astype(float)
    n, m = C.shape

    _x, _y = np.where(X > 0)
    nonzero = list(zip(_x, _y))
    if len(nonzero) < n + m - 1:
        for i in range(n):
            for j in range(m):
                if X[i, j] == 0:
                    X[i, j] = 0.00001
                    nonzero.append((i, j))
                    if len(nonzero) >= n + m - 1:
                        break
            if len(nonzero) >= n + m - 1:
                break

    while True:
        u = np.array([np.nan] * n)
        v = np.array([np.nan] * m)
        S = np.zeros((n, m))

        _x, _y = np.where(X > 0)
        nonzero = list(zip(_x, _y))
        u[nonzero[0][0]] = 0

        changed = True
        while changed:
            changed = False
            for i, j in nonzero:
                if np.isnan(u[i]) and not np.isnan(v[j]):
                    u[i] = C[i, j] - v[j]
                    changed = True
                elif not np.isnan(u[i]) and np.isnan(v[j]):
                    v[j] = C[i, j] - u[i]
                    changed = True

        for i in range(n):
            for j in range(m):
                S[i, j] = C[i, j] - u[i] - v[j]

        if np.all(S >= 0):
            break

        i, j = np.unravel_index(np.argmin(S), S.shape)
        start = (i, j)
        T = np.copy(X)
        T[start] = 1

        max_iters = 1000
        iter_count = 0
        while True:
            iter_count += 1
            if iter_count > max_iters:
                return X, np.sum(X * C)

            _xs, _ys = np.nonzero(T)
            xcount, ycount = Counter(_xs), Counter(_ys)
            for x, count in xcount.items():
                if count <= 1:
                    T[x, :] = 0
            for y, count in ycount.items():
                if count <= 1:
                    T[:, y] = 0
            if all(count > 1 for count in xcount.values()) and all(count > 1 for count in ycount.values()):
                break

        dist = lambda a, b: abs(a[0] - b[0]) + abs(a[1] - b[1])
        fringe = set(tuple(p) for p in np.argwhere(T > 0))
        size = len(fringe)
        path = [start]
        while len(path) < size:
            last = path[-1]
            if last in fringe:
                fringe.remove(last)
            next_candidates = [f for f in fringe if f[0] == last[0] or f[1] == last[1]]
            if not next_candidates:
                return X, np.sum(X * C)
            next_step = min(next_candidates, key=lambda x: dist(last, x))
            path.append(next_step)

        neg = path[1::2]
        pos = path[::2]

        if not neg:
            return X, np.sum(X * C)

        q = min([X[i, j] for i, j in neg])
        for i, j in neg:
            X[i, j] -= q
        for i, j in pos:
            X[i, j] += q

    return X, np.sum(X * C)

def print_initial_table(matrix, ai, bj, title):
    m, n = matrix.shape
    header = ["     "] + [f"B{j+1}" for j in range(n)] + [" aᵢ"]
    rows = []
    for i in range(m):
        row = [f"A{i+1}"] + [f"{int(matrix[i][j])}" for j in range(n)] + [f"{int(ai[i])}"]
        rows.append(row)
    last_row = ["bₕ"] + [f"{int(d)}" for d in bj] + [""]

    col_widths = [max(len(str(item)) for item in col) + 2 for col in zip(*([header] + rows + [last_row]))]

    def format_row(row):
        return "".join(str(item).center(w) for item, w in zip(row, col_widths))

    print(f"\n{title}\n")
    print(format_row(header))
    for row in rows:
        print(format_row(row))
    print(format_row(last_row))

if __name__ == '__main__':
    ai = np.array([40, 60, 50])
    bj = np.array([30, 10, 45, 75])
    cij = np.array([[1, 7, 2, 5],
                    [3, 8, 4, 1],
                    [6, 3, 5, 3]])

    ai, bj, cij, is_open, added_type = check_problem_type(ai, bj, cij)

    print_initial_table(cij, ai, bj, "Таблиця транспортної задачі:")

    print("\nОберіть метод для побудови початкового опорного плану:")
    print("1) Метод північно-західного кута")
    print("2) Метод найменшої вартості")
    method = input("Введіть (1-2): ")

    if method == '1':
        initial_plan = method_northwest_corner(ai, bj)
        method_name = "Метод північно-західного кута"
    elif method == '2':
        initial_plan = method_least_cost(cij, ai, bj)
        method_name = "Метод найменшої вартості"
    else:
        print("Некоректний вибір методу.")
        exit()

    print_plan_costs_format(initial_plan, cij, ai, bj, f"Початковий опорний план ({method_name}):")
    print_cost_expression(initial_plan, cij)

    plan, total_cost = метод_потенціалів(ai, bj, cij, initial_plan)

    if plan is not None:
        plan_int = np.round(plan).astype(int)
        print_plan_costs_format(plan_int, cij, ai, bj, "Оптимальний опорний план:")
        print_cost_expression(plan_int, cij)