import numpy as np;
import math;
import matplotlib.pyplot as plot;

# Configuração da rede neural
tamanho_entrada = 1   # 1 entrada (valor de x)
tamanho_oculto_1 = 40  # tamanho da primeira camada oculta (melhor resultado 40)
tamanho_oculto_2 = 20  # tamanho da segunda camada oculta (melhor resultado 20)
tamanho_saida = 1  # 1 saída (valor de sin(x))
taxa_aprendizado = 0.001  # Taxa de aprendizado
epocas = 10000  # Número de épocas (10000 ja é o suficiente)
tolerancia = 1e-7  # Critério de tolerância para early stopping
paciencia = 1000  # Número de épocas sem melhoria antes de parar 

# Funções matematicas:
def tangente_hiperbolica(x): # Função de ativação tangente hiperbólica
    return np.tanh(x)

def deriva_tanH(x): #Função de derivação da tangente hiperbólica
    return 1 - np.tanh(x) ** 2

def erro_quadratico_medio(y_real, y_predito): #Função do erro quadratico médio (custo)
    return np.mean((y_real - y_predito) ** 2)

#Geração e Normalização dos dados
# Gera dados de treinamento
X_treino = np.linspace(0, 2 * np.pi, 20).reshape(-1, 1) # Retorna X numeros espaçados igualmente entre o intervalo dado
y_treino = np.sin(X_treino)
#Gera dados de teste
X_teste = np.linspace(0, 2 * np.pi, 150).reshape(-1, 1)
y_teste = np.sin(X_teste)
#Garante que os dados estão entre -1 e 1
X_treino = (X_treino - np.pi) / np.pi
X_teste = (X_teste - np.pi) / np.pi

# Inicialização dos pesos 
W1 = np.random.randn(tamanho_entrada, tamanho_oculto_1) * np.sqrt(2. / tamanho_entrada) #Weight
b1 = np.zeros((1, tamanho_oculto_1)) #Bias
W2 = np.random.randn(tamanho_oculto_1, tamanho_oculto_2) * np.sqrt(2. / tamanho_oculto_1)
b2 = np.zeros((1, tamanho_oculto_2))
W3 = np.random.randn(tamanho_oculto_2, tamanho_saida) * np.sqrt(2. / tamanho_oculto_2)
b3 = np.zeros((1, tamanho_saida))

# Algoritmo de otimização Adam (com acumuladores de momento)
mW1, mW2, mW3 = 0, 0, 0
vW1, vW2, vW3 = 0, 0, 0
mb1, mb2, mb3 = 0, 0, 0
vb1, vb2, vb3 = 0, 0, 0
beta1, beta2 = 0.9, 0.999
epsilon = 1e-8

# Função para atualizar pesos com Adam
def atualizar_adam(parametro, gradiente, m, v, t, taxa_aprendizado):
    m = beta1 * m + (1 - beta1) * gradiente
    v = beta2 * v + (1 - beta2) * (gradiente ** 2)
    m_corrigido = m / (1 - beta1 ** t)
    v_corrigido = v / (1 - beta2 ** t)
    parametro += taxa_aprendizado * m_corrigido / (np.sqrt(v_corrigido) + epsilon)
    return parametro, m, v

# Treinamento da rede neural com backpropagation e Adam
for epoca in range(epocas):
    # Foward Pass
    entrada_camada_oculta_1 = np.dot(X_treino, W1) + b1
    saida_camada_oculta_1 = tangente_hiperbolica(entrada_camada_oculta_1)

    entrada_camada_oculta_2 = np.dot(saida_camada_oculta_1, W2) + b2
    saida_camada_oculta_2 = tangente_hiperbolica(entrada_camada_oculta_2)

    entrada_camada_saida = np.dot(saida_camada_oculta_2, W3) + b3
    saida_predita = tangente_hiperbolica(entrada_camada_saida)  # Aplica tanh na saída

    # Calcular o erro (loss)
    erro = erro_quadratico_medio(y_treino, saida_predita)
    
    # Backpropagation
    erro_gradiente = y_treino - saida_predita

    # Gradiente para a camada de saída
    d_saida = erro_gradiente * deriva_tanH(saida_predita)

    # Gradiente para a segunda camada escondida
    d_camada_oculta_2 = d_saida.dot(W3.T) * deriva_tanH(entrada_camada_oculta_2)

    # Gradiente para a primeira camada escondida
    d_camada_oculta_1 = d_camada_oculta_2.dot(W2.T) * deriva_tanH(entrada_camada_oculta_1)

    # Atualização dos pesos e bias com Adam
    W3, mW3, vW3 = atualizar_adam(W3, saida_camada_oculta_2.T.dot(d_saida), mW3, vW3, epoca + 1, taxa_aprendizado)
    b3, mb3, vb3 = atualizar_adam(b3, np.sum(d_saida, axis=0, keepdims=True), mb3, vb3, epoca + 1, taxa_aprendizado)
    W2, mW2, vW2 = atualizar_adam(W2, saida_camada_oculta_1.T.dot(d_camada_oculta_2), mW2, vW2, epoca + 1, taxa_aprendizado)
    b2, mb2, vb2 = atualizar_adam(b2, np.sum(d_camada_oculta_2, axis=0, keepdims=True), mb2, vb2, epoca + 1, taxa_aprendizado)
    W1, mW1, vW1 = atualizar_adam(W1, X_treino.T.dot(d_camada_oculta_1), mW1, vW1, epoca + 1, taxa_aprendizado)
    b1, mb1, vb1 = atualizar_adam(b1, np.sum(d_camada_oculta_1, axis=0, keepdims=True), mb1, vb1, epoca + 1, taxa_aprendizado)

    # Imprimir o erro a cada 5000 épocas
    if (epoca + 1) % 1000 == 0:
        print(f"Época {epoca+1}/{epocas}, Erro: {erro:.10f}")

# Após o treinamento, testar o modelo com os dados de teste
entrada_camada_oculta_1_teste = np.dot(X_teste, W1) + b1
saida_camada_oculta_1_teste = tangente_hiperbolica(entrada_camada_oculta_1_teste)

entrada_camada_oculta_2_teste = np.dot(saida_camada_oculta_1_teste, W2) + b2
saida_camada_oculta_2_teste = tangente_hiperbolica(entrada_camada_oculta_2_teste)

entrada_camada_saida_teste = np.dot(saida_camada_oculta_2_teste, W3) + b3
y_predito_teste = tangente_hiperbolica(entrada_camada_saida_teste)  # Usando tanh para a saída

# Calcular o erro no conjunto de teste
erro_teste = erro_quadratico_medio(y_teste, y_predito_teste)
print(f"Erro no conjunto de teste: {erro_teste:.10f}")

# Plotar os resultados no conjunto de teste, com o eixo X em graus
X_teste_graus = (X_teste * np.pi + np.pi) * 180 / np.pi  # Converter de radianos para graus

plot.plot(X_teste_graus, y_teste, label='Seno Real')
plot.plot(X_teste_graus, y_predito_teste, label='Seno Predito (Teste)', linestyle='--')
plot.title('Seno Real vs Seno Predito (Conjunto de Teste)')
plot.xlabel('Graus')
plot.ylabel('seno(x)')
plot.legend()
plot.show()