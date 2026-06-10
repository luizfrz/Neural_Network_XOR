import numpy as np

def tanh(x):
     return np.tanh(x)

def derivative(x):
     return 1 - x**2

input = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

output = np.array([
    [0],
    [1],
    [1],
    [0]
])

np.random.seed(1)

w0 = 2 * np.random.random((2, 3)) - 1

w1 = 2 * np.random.random((3, 1)) - 1

epochs = 10000

learning_rate = 0.1

for i in range(epochs):

    hidden_input = np.dot(input, w0)
    hidden_layer = tanh(hidden_input)

    output_input = np.dot(hidden_layer, w1)
    output_layer = tanh(output_input)

    error = output - output_layer

    if i % 1000 == 0:
        print("Capturando erros...")
        print(f"Época {i} - Erro médio: {np.mean(np.abs(error))}")

    # Backpropagation
    delta_output = error * derivative(output_layer)

    error_hidden = delta_output.dot(w1.T)
    delta_hidden = error_hidden * derivative(hidden_layer)

    # Ajuste dos pesos
    w1 += hidden_layer.T.dot(delta_output) * learning_rate
    w0 += input.T.dot(delta_hidden) * learning_rate

print("\nResultado binário:")
print((output_layer > 0.5).astype(int))

print("\nSaída esperada:")
print(output)

# Resultado final
print("\nResultado final:")
print(output_layer)

