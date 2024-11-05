import torch
import pywt

class WaveletConv1D(torch.nn.Module):
    def __init__(self, wavelet_name, input_dim, output_dim):
        super(WaveletConv1D, self).__init__()
        self.wavelet = pywt.Wavelet(wavelet_name)
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.linear = torch.nn.Linear(int(input_dim/2), output_dim)

    def forward(self, x):
        coeffs = pywt.wavedec(x.cpu().detach().numpy(), self.wavelet, mode='symmetric', level=1)
        wavelet_coeffs = torch.tensor(coeffs[0], dtype=torch.float32).to(x.device)
        x = self.linear(wavelet_coeffs)
        return x

class WNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, wavelet_name='db1'):
        super(WNN, self).__init__()
        self.wavelet_conv = WaveletConv1D(wavelet_name, input_dim, hidden_dim)
        self.hidden = torch.nn.Linear(hidden_dim, hidden_dim)
        self.out = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.wavelet_conv(x)
        x = torch.tanh(self.hidden(x))
        x = self.out(x)
        return x

model = WNN(10, 20, 10)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
def train(model, x, y, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    y_pred = model(x)
    loss = criterion(y_pred, y)
    loss.backward()
    optimizer.step()
    return loss


x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], dtype=torch.float32)
y = torch.tensor([1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0], dtype=torch.float32)

for i in range(1000):
    loss = train(model, x, y, optimizer, criterion)
    if i % 100 == 0:
        print(f'Epoch {i}, Loss: {loss.item()}')

