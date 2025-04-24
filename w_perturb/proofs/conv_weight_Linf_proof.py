from cvc5.pythonic import *

def setup_element(name: str):
    return Real(name)

def setup_array(name: str, n: int):
    return [Real(f'{name}[{i}]') for i in range(n)]

def element_expand(x: Real, n: int):
    return [x for _ in range(n)]

def element_clamp_pos(x: list[Real]):
    return [If(xi >= 0, xi, 0) for xi in x]

def element_clamp_neg(x: list[Real]):
    return [If(xi <= 0, xi, 0) for xi in x]

def element_conv(x: list[Real], w: list[Real]):
    assert len(x) == len(w)

    return Sum([x[i] * w[i] for i in range(len(x))])

def element_add(x: list[Real], y: list[Real]):
    assert len(x) == len(y)

    return [xi + yi for xi, yi in zip(x, y)]

def element_lte(x: list[Real], y: list[Real]):
    assert len(x) == len(y)

    return And([xi <= yi for xi, yi in zip(x, y)])

solver = Solver()

# Kernel size (when flattened)
n = 9

# Base Conditions

x = setup_array('x', n)
w = setup_array('w', n)

y = element_conv(x, w)

# Bounds Proof

x_pos = element_clamp_pos(x)
x_neg = element_clamp_neg(x)

w_c = setup_array('w_c', n)
w_d = setup_element('w_d') # half eps - assumes Linf norm!

lw = element_add(w_c, element_expand(-w_d, n))
uw = element_add(w_c, element_expand(w_d, n))

y_c = element_conv(x, w_c)

dev_pos = element_conv(x_pos, element_expand(w_d, n))
dev_neg = element_conv(x_neg, element_expand(-w_d, n))

deviation = dev_pos + dev_neg

ub = y_c + deviation
lb = y_c - deviation

solver.add(And(
    element_lte(lw, uw),
    element_lte(lw, w),
    element_lte(w, uw)
))

solver.add(Not(And(
    lb <= y,
    y <= ub
)))

if solver.check() == sat:
    print("Found counter example:")

    model = solver.model()
    print(model)
else:
    print("Proved, result =", solver.check())
