import pandas as pd
import numpy as np


df = pd.read_csv('data.csv')
df = df.iloc[np.random.permutation(len(df))]
y = df.iloc[0:100, 4].values
y = np.where(y == "Iris-setosa", 0,y)
y = np.where(y == "Iris-versicolor", 1, y)
y = np.where(y == "Iris-virginica", 2, y)

X = df.iloc[0:100, [0, 2]].values

input_size = X.shape[1]
neurons_hidden_layer= 10
neurons_output_layer = 3
print('input_size:', input_size)
print('neurons_hidden_layer:',neurons_hidden_layer)
print('neurons_output_layer:',neurons_output_layer)

# матрица весов и порого первого слоя
W_1 = np.zeros((1 + input_size, neurons_hidden_layer))
# пороговые значения
W_1[0, :] = (np.random.randint(0, 3, size = (neurons_hidden_layer)))
# веса
W_1[1:, :] = (np.random.randint(-1, 2, size = (input_size, neurons_hidden_layer)))
print('W_1:',W_1)

# матрица весов и порогов второго слоя
W_2 = np.random.randint(0, 2, size = (1 + neurons_hidden_layer, neurons_output_layer)).astype(np.float64)
print('W_2:', W_2)

def predict(X):
    W_1_out = np.where((np.dot(X, W_1[1:, :]) + W_1[0, :]) >= 0.0, 1, -1).astype(np.float64)
    W_2_out = np.where((np.dot(W_1_out, W_2[1:, :]) + W_2[0, :]) >= 0.0, 1, -1).astype(np.float64)
    return W_2_out, W_1_out


def errors(result, expected_class):
    expected_vector = np.array([-1, -1, -1])
    expected_vector[expected_class] = 1
    sum_errors = sum(expected_vector -result)
    return sum_errors


# количетсво обучающих итераций
n_iter=0
# шаг обучения
step = 0.01
#количество
check_iter = 5

# список для хранения  матрицы весов второго слоя
list_w_2_weights= [];

#обучение
while(True):
    print('iteration:',n_iter)
    n_iter+=1

    for x_input, expected, j in zip(X, y, range(X.shape[0])):
        W_2_out, W_1_out = predict(x_input)
        expected_vector = np.array([-1,-1,-1])
        expected_vector[expected] = 1
        W_2[1:] += ((step * (expected_vector - W_2_out)) * W_1_out.reshape(-1, 1))
        W_2[0] += step * (expected_vector - W_2_out)

    #список матрицы весов второго слоя
    list_w_2_weights.append(W_2.tobytes())

    W_2_out, W_1_out = predict(X)
    sum_errors=0
    for i in range(len(y)):
        sum_errors+=errors(W_2_out[i], y[i])
    print('sum_errors', sum_errors)
    if (sum_errors == 0):
        print('Все примеры обучающей выборки решены:')
        break

    #проверка наличия дубликатов в списке весов
    if ((n_iter % check_iter )==0):
        break_out_flag = False
        for item in list_w_2_weights:
            if list_w_2_weights.count(item) > 1:
                print('Повторение весов:')
                break_out_flag = True
                break
        if break_out_flag:
            break
