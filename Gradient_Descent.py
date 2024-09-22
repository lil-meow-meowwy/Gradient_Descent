import torch
import numpy as np

inputs = np.array([[73, 67, 43],
                   [91, 88, 64],
                   [87, 134, 58],
                   [102, 43, 37],
                   [69, 96, 70]], dtype='float32')

targets = np.array([[56, 70],
                    [81, 101],
                    [119, 133],
                    [22, 37],
                    [103, 119]], dtype='float32')

inputs = torch.from_numpy(inputs)
targets = torch.from_numpy(targets)

w = torch.randn(2, 3, requires_grad=True)
b = torch.randn(2, requires_grad=True)

def model(x):
    return x @ w.t() + b

def mse(t1, t2):
    diff = t1 - t2
    return torch.sum(diff * diff) / diff.numel()

# Навчання моделі
learning_rate = 1e-5
epochs = 1000

for epoch in range(epochs):
    # Обчислюємо прогнози
    preds = model(inputs)

    # Обчислюємо похибку
    loss = mse(preds, targets)

    # Виконуємо обчислення градієнтів
    loss.backward()

    # Оновлюємо ваги та зсуви
    with torch.no_grad():
        w -= w.grad * learning_rate
        b -= b.grad * learning_rate

        # Очищуємо градієнти після кожної ітерації
        w.grad.zero_()
        b.grad.zero_()

    # Виводимо поточну похибку кожні 100 епох
    if (epoch + 1) % 100 == 0:
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')

# Останній прогноз та MSE після навчання
preds = model(inputs)
loss = mse(preds, targets)
print(f'\nFinal Loss after {epochs} epochs: {loss.item()}')
print(f'Final Predictions: \n{preds}')
print(f'Actual Targets: \n{targets}')
