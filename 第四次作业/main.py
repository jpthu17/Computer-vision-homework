import torch


def get_f(x, y, w):
    relu = torch.nn.ReLU()
    dis = (relu(x @ w) - y) * (relu(x @ w) - y)
    return dis.sum()


def varepsilon(x):
    result = torch.zeros_like(x)
    result[x > 0] = 1
    result[x == 0] = 0.5
    return result


def get_fx(x, y, w):
    relu = torch.nn.ReLU()
    result = 2 * (relu(x @ w) - y) * varepsilon(x @ w) @ w.t()
    return result


def get_fy(x, y, w):
    relu = torch.nn.ReLU()
    result = -2 * (relu(x @ w) - y)
    return result


def get_fw(x, y, w):
    relu = torch.nn.ReLU()
    result = 2 * x.t() @ ((relu(x @ w) - y) * varepsilon(x @ w))
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
