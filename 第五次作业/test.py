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


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.prediction = torch.nn.Sequential(
                            torch.nn.Linear(n_feature, n_hidden, bias=True),
                            torch.nn.Sigmoid(),
                            torch.nn.Linear(n_hidden, n_output, bias=True)
                            )

    def forward(self, x):
        x = self.prediction(x)
        return x


if __name__ == '__main__':

    net = Net(n_feature=1, n_hidden=20, n_output=1)     # define the network
    print(net)  # net architecture

    torch.manual_seed(0)
    x = torch.randn(1, 1, requires_grad=True)
    y = torch.randn(1, 1, requires_grad=True)
    prediction = net(x)     # input x and predict based on x
    loss = ((prediction - y)*(prediction - y)).sum()     # must be (1. nn output, 2. target)

    loss.backward()         # backpropagation, compute gradients

    w1 = net.prediction[0].weight.t()
    w2 = net.prediction[2].weight.t()
    b1 = net.prediction[0].bias
    b2 = net.prediction[2].bias

    print("w1.grad:\n pytorch:\n {}\n ours:\n {}".format(net.prediction[0].weight.grad.t(), get_fw1(x, w1, b1, w2, b2, y)))
    print("w2.grad:\n pytorch:\n {}\n ours:\n {}".format(net.prediction[2].weight.grad.t(), get_fw2(x, w1, b1, w2, b2, y)))
    print("b1.grad:\n pytorch:\n {}\n ours:\n {}".format(net.prediction[0].bias.grad, get_fb1(x, w1, b1, w2, b2, y)))
    print("b2.grad:\n pytorch:\n {}\n ours:\n {}".format(net.prediction[2].bias.grad, get_fb2(x, w1, b1, w2, b2, y)))