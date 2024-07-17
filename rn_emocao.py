import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

 
frases = ["Cacá já mandou 3 gol contra, chega uma hora que não é mais azar, é cabacisse mesmo",
          "Caca tem que começar a ser punido por isso plmrddeus como é que pode",
          "Eu nunca vi um zagueiro fazer tanto gol contra bicho",
          "PÉSSIMO... EU TO CANSADO",
          "GRANDE HUGO KKKKKKKKKKKKKKKK",
          "Já pode fechar as portas porque esse time está fadado ao fracasso.",
          "TAVA DEMORANDO MSM KKKKKKKKKKKKLKKKKKKKKKK INÚTEIS",
          "o que ta acontecendo com meu time mds",
          "nao conseguimos passar do meio campo contra o CRICIUMA",
          "ainda pensei em apostar no timão"]
 
vectorizer = CountVectorizer()
 
bow = vectorizer.fit_transform(frases)
vocabulario = vectorizer.get_feature_names_out()
 
vetores_bag = bow.toarray()
 
 
entradas = np.array(vetores_bag)
saidas = np.array([[1, 0, 0],
                   [1, 0, 0],
                   [0, 1, 0],
                   [1, 0, 0],
                   [0, 1, 0],
                   [1, 0, 0],
                   [1, 0, 0],
                   [0, 1, 0],
                   [0, 1, 0],
                   [0, 1, 0]])
 
entradas_size = entradas.shape[1]
camada_oculta_size = 37
saidas_sizes = saidas.shape[1]
taxa_aprendizado = 0.01
epocas = 1000000
 
w_entrada_oculta = np.random.uniform(-1, 1, (entradas_size, camada_oculta_size))
b_camada_oculta = np.random.uniform(-1, 1, camada_oculta_size)
w_oculta_saida = np.random.uniform(-1, 1, (camada_oculta_size, saidas_sizes))
b_saida = np.random.uniform(-1, 1, saidas_sizes)


 
def relu(z):
    return np.maximum(0, z)


 
def relu_derivative(z):
    return np.where(z > 0, 1, 0)


 
def softmax(z):
    if z.ndim > 1:
        e_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return e_z / np.sum(e_z, axis=1, keepdims=True)
    else:
        e_z = np.exp(z - np.max(z))
        return e_z / np.sum(e_z)


 
for epoca in range(epocas):
    for i in range(len(entradas)):
        z_oculta = np.dot(entradas[i], w_entrada_oculta) + b_camada_oculta
        a_oculta = relu(z_oculta)
        z_saida = np.dot(a_oculta, w_oculta_saida) + b_saida
        a_saida = softmax(z_saida)

        erro_saida = saidas[i] - a_saida

        delta_saida = erro_saida
        gradiente_w_oculta_saida = np.outer(a_oculta, delta_saida)
        gradiente_b_saida = delta_saida
        delta_oculta = np.dot(delta_saida, w_oculta_saida.T) * relu_derivative(z_oculta)
        gradiente_w_entrada_oculta = np.outer(entradas[i], delta_oculta)
        gradiente_b_oculta = delta_oculta

        w_oculta_saida += taxa_aprendizado * gradiente_w_oculta_saida
        b_saida += taxa_aprendizado * gradiente_b_saida
        w_entrada_oculta += taxa_aprendizado * gradiente_w_entrada_oculta
        b_camada_oculta += taxa_aprendizado * gradiente_b_oculta

    if epoca % 100000 == 0:
        erro_medio = np.mean(np.square(erro_saida))
        print(f'Época {epoca}, Erro médio: {erro_medio}')


 
def interpretar_saida(resultado_bin):
    if resultado_bin[0] == 1:
        return 'Negativo'
    elif resultado_bin[1] == 1:
        return 'Neutro'
    elif resultado_bin[2] == 1:
        return 'Positivo'
    else:
        return 'Indefinido'


 
for j in range(len(entradas)):
    z_oculta = np.dot(entradas[j], w_entrada_oculta) + b_camada_oculta
    a_oculta = relu(z_oculta)
    z_saida = np.dot(a_oculta, w_oculta_saida) + b_saida
    a_saida = softmax(z_saida.reshape(1, -1))

    resultado_bin = np.round(a_saida).astype(int)[0]

    interpretacao = interpretar_saida(resultado_bin)

    print(f'Tweet: "{frases[j]}"')
    print(f'Saída Esperada: {saidas[j]}, Resultado: {resultado_bin} ({interpretacao})')
 
print('--- Pesos finais ---')
print('Pesos da camada de entrada para oculta:')
print(w_entrada_oculta)
print('Bias da camada oculta:')
print(b_camada_oculta)
print('Pesos da camada oculta para saída:')
print(w_oculta_saida)
print('Bias da camada de saída:')
print(b_saida)
 
nova_frase = ["Time inútil estou cansado"]

vectorizer = CountVectorizer(vocabulary=vocabulario)

nova_entrada = vectorizer.transform(nova_frase).toarray()

z_oculta = np.dot(nova_entrada, w_entrada_oculta) + b_camada_oculta
a_oculta = relu(z_oculta)
z_saida = np.dot(a_oculta, w_oculta_saida) + b_saida
a_saida = softmax(z_saida)

resultado_bin = np.round(a_saida).astype(int)[0]

interpretacao = interpretar_saida(resultado_bin)

print(f'Novo Tweet: "{nova_frase[0]}"')
print(f'Resultado: {resultado_bin} ({interpretacao})')
