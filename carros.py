import streamlit as st  # Biblioteca para criar aplicações web interativas
import pandas as pd  # Biblioteca para manipulação de dados
from sklearn.model_selection import train_test_split  # Função para dividir os dados em treino e teste
from sklearn.preprocessing import OrdinalEncoder  # Codificador para variáveis categóricas
from sklearn.naive_bayes import CategoricalNB  # Algoritmo de classificação Naive Bayes para dados categóricos
from sklearn.metrics import accuracy_score  # Métrica para avaliar a acurácia do modelo

st.set_page_config(
    page_title="Classificação de Veículos", 
    layout="wide"
)  # Configurações da página do Streamlit
@st.cache_data  # Cacheia os dados e o modelo para acelerar a aplicação
def load_data_and_model():
    # Carrega o dataset de veículos
    carros = pd.read_csv('carros.csv', sep=',')
    
    # Inicializa o codificador para variáveis categóricas
    encoder = OrdinalEncoder()

    # Garante que todas as colunas (exceto 'class') sejam do tipo string
    for col in carros.columns.drop('class'):
        carros[col] = carros[col].astype(str)
        encoder = OrdinalEncoder()
    
    # Codifica as variáveis independentes (X)
    X_encoded = encoder.fit_transform(carros.drop('class', axis=1))
    # Codifica a variável alvo (y)
    y = carros['class'].astype('category').cat.codes
    
    # Divide os dados em conjuntos de treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3, random_state=42)

    # Cria e treina o modelo Naive Bayes categórico
    modelo = CategoricalNB()
    modelo.fit(X_train, y_train)

    # Faz previsões e calcula a acurácia do modelo
    y_pred = modelo.predict(X_test)
    accuracia = accuracy_score(y_test, y_pred)

    # Retorna o modelo treinado, o codificador, a acurácia e o dataframe original
    return modelo, encoder, accuracia, carros

