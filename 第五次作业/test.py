import torch
import torch.nn.functional as F


def get_h_sigmoid(x, w1, b1, w2, b2):
    sigmoid = torch.nn.Sigmoid()
    return sigmoid(x @ w1 + b1)


def get_y(x, w1, b1, w2, b2):
    return get_h_sigmoid(x, w1, b1, w2, b2) @ w2 + b2


def get_fw1(x, w1, b1, w2, b2, y):
    return 2*x.t()@(((get_y(x, w1, b1, w2, b2)-y)@w2.t())*(get_h_sigmoid(x, w1, b1, w2, b2)*(1-get_h_sigmoid(x, w1, b1, w2, b2))))


def get_fw2(x, w1, b1, w2, b2, y):
    return 2*get_h_sigmoid(x, w1, b1, w2, b2).t()@(get_y(x, w1, b1, w2, b2)-y)


def get_fb1(x, w1, b1, w2, b2, y):
    return 2*((get_y(x, w1, b1, w2, b2)-y)@w2.t()*(get_h_sigmoid(x, w1, b1, w2, b2)*(1-get_h_sigmoid(x, w1, b1, w2, b2))))


def get_fb2(x, w1, b1, w2, b2, y):
    return 2*(get_y(x, w1, b1, w2, b2)-y)


if __name__ == '__main__':
    torch.manual_seed(0)
    x = torch.randn(10, 4, requires_grad=True)
    w1 = torch.randn(4, 4, requires_grad=True)
    b1 = torch.randn(10, 4, requires_grad=True)
    w2 = torch.randn(4, 4, requires_grad=True)
    b2 = torch.randn(10, 4, requires_grad=True)
    y = torch.randn(10, 4, requires_grad=True)

    loss = ((get_y(x, w1, b1, w2, b2) - y) * (get_y(x, w1, b1, w2, b2) - y)).sum()

    loss.backward()

    print("w1.grad:\n pytorch:\n {}\n ours:\n {}".format(w1.grad, get_fw1(x, w1, b1, w2, b2, y)))
    print("w2.grad:\n pytorch:\n {}\n ours:\n {}".format(w2.grad, get_fw2(x, w1, b1, w2, b2, y)))
    print("b1.grad:\n pytorch:\n {}\n ours:\n {}".format(b1.grad, get_fb1(x, w1, b1, w2, b2, y)))
    print("b2.grad:\n pytorch:\n {}\n ours:\n {}".format(b2.grad, get_fb2(x, w1, b1, w2, b2, y)))