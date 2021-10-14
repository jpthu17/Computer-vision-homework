import torch


def get_f(x, y, w):
    relu = torch.nn.ReLU()
    dis = (relu(x @ w) - y) * (relu(x @ w) - y)
    return dis.sum()


def varepsilon(x):
    if x > 0:
        return 1
    elif x == 0:
        return 0.5
    else:
        return 0


def get_fx(x, y, w):
    result = torch.zeros_like(x)
    relu = torch.nn.ReLU()
    I, J = y.size()
    _, K = x.size()

    for i in range(I):
        for k in range(K):
            for j in range(J):
                result[i,k] += 2*(relu(x @ w) - y)[i,j]*varepsilon((x @ w)[i,j])*w[k,j]
    return result


def get_fy(x, y, w):
    result = torch.zeros_like(y)
    relu = torch.nn.ReLU()
    I, J = y.size()
    _, K = x.size()

    for i in range(I):
        for j in range(J):
            result[i, j] = -2 * (relu(x @ w) - y)[i, j]
    return result


def get_fw(x, y, w):
    result = torch.zeros_like(w)
    relu = torch.nn.ReLU()
    I, J = y.size()
    _, K = x.size()

    for k in range(K):
        for j in range(J):
            for i in range(I):
                result[k, j] += 2 * (relu(x @ w) - y)[i, j] * varepsilon((x @ w)[i, j]) * x[i, k]
    return result


if __name__ == '__main__':
    torch.manual_seed(0)
    x = torch.randn(10, 4, requires_grad=True)
    w = torch.randn(4, 4, requires_grad=True)
    y = torch.randn(10, 4, requires_grad=True)

    loss = get_f(x, y, w)

    loss.backward()

    print("x.grad:\n pytorch:\n {}\n ours:\n {}".format(x.grad, get_fx(x, y, w)))
    print("y.grad:\n pytorch:\n {}\n ours:\n {}".format(y.grad, get_fy(x, y, w)))
    print("w.grad:\n pytorch:\n {}\n ours:\n {}".format(w.grad, get_fw(x, y, w)))
