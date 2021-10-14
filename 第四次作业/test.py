import torch


if __name__ == '__main__':
    relu = torch.nn.ReLU()
    x = torch.zeros(1, requires_grad=True)
    loss = relu(x)
    loss.backward()
    print(x.grad)