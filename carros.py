import streamlit as st  # Biblioteca para criar aplicações web interativas
import pandas as pd  # Biblioteca para manipulação de dados
from sklearn.model_selection import train_test_split  # Função para dividir os dados em treino e teste
from sklearn.preprocessing import OrdinalEncoder  # Codificador para variáveis categóricas
from sklearn.naive_bayes import CategoricalNB  # Algoritmo de classificação Naive Bayes para dados categóricos
from sklearn.metrics import accuracy_score  # Métrica para avaliar a acurácia do modelo

st.set_page_config(
    page_title="Classificação de Veículos", 
    layout="wide",
    initial_sidebar_state="expanded"
)  

# Configurações da página do Streamlit
@st.cache_data  # Cacheia os dados e o modelo para acelerar a aplicação
def load_data_and_model():
    # Carrega o dataset de veículos
    carros = pd.read_csv('carros.csv', sep=',')
    
    # Inicializa o codificador para variáveis categóricas
    encoder = OrdinalEncoder()

    # Garante que todas as colunas (exceto 'classe') sejam do tipo string
    for col in carros.columns.drop('classe'):
        carros[col] = carros[col].astype(str)
    
    # Codifica as variáveis independentes (X)
    X_encoded = encoder.fit_transform(carros.drop('classe', axis=1))
    # Codifica a variável alvo (y)
    y = carros['classe'].astype('category').cat.codes

    # Divide os dados em conjuntos de treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3, random_state=42)

    # Cria e treina o modelo Naive Bayes categórico
    modelo = CategoricalNB()
    modelo.fit(X_train, y_train)

    # Faz previsões e calcula a acurácia do modelo
    y_pred = modelo.predict(X_test)
    acuracia = accuracy_score(y_test, y_pred)

    # Retorna o modelo treinado, o codificador, a acurácia e o dataframe original
    return modelo, encoder, acuracia, carros



# Carrega os dados e o modelo
modelo, encoder, acuracia, carros = load_data_and_model()

# Cria abas para navegação
tab1, tab2, tab3, tab4 = st.tabs(["Classificação", "Explicabilidade", "Documentação", "Desenvolvedor"])



# Aba 1: Classificação
with tab1:
    st.title("Previsão da Qualidade de Veículos")
    st.markdown("""
    <style>
    .stButton>button {background-color: #0099ff; color: white; font-weight: bold;}
    .stTabs [data-baseweb="tab"] {font-size: 18px;}
    </style>
    """, unsafe_allow_html=True)
    st.write(f"Acurácia do modelo: <b>{(acuracia*100):.2f}%</b>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        input_features = {
            "preco": st.selectbox("Preço", options=carros['preco'].unique(), help="Faixa de preço do veículo (ex: baixo, alto, muito_alto)"),
            "manutencao": st.selectbox("Manutenção", options=carros['manutencao'].unique(), help="Custo de manutenção do veículo (ex: baixa, alta, muito_alto)"),
            "portas": st.selectbox("Número de portas", options=carros['portas'].unique(), help="Quantidade de portas do veículo (ex: 2, 3, 4, 5_ou_mais)"),
        }
    with col2:
        input_features.update({
            "pessoas": st.selectbox("Capacidade de Passageiros", options=carros['pessoas'].unique(), help="Número de pessoas que o veículo comporta (ex: 2, 4, mais)"),
            "porta_malas": st.selectbox("Capacidade do porta-malas", options=carros['porta_malas'].unique(), help="Tamanho do porta-malas (ex: pequeno, médio, grande)"),
            "seguranca": st.selectbox("Segurança", options=carros['seguranca'].unique(), help="Nível de segurança do veículo (ex: baixa, média, alta)")
        })
    if st.button("Processar"):
        input_df = pd.DataFrame([input_features], columns=carros.columns.drop('classe'))
        input_encoded = encoder.transform(input_df)
        prediction = modelo.predict(input_encoded)
        classes = carros['classe'].astype('category').cat.categories
        predicted_class = classes[prediction[0]]
        st.success(f"Resultado da Previsão: {predicted_class}")

# Aba 2: Explicabilidade
with tab2:
    st.subheader("Explicabilidade do Modelo")
    st.write("Como o modelo Naive Bayes é probabilístico e categórico, ele não gera importâncias de variáveis diretamente. Podemos, porém, mostrar a frequência das variáveis por classe para ajudar na interpretação.")
    import matplotlib.pyplot as plt
    import seaborn as sns
    # Exemplo de visualização: Frequência das variáveis por classe em layout 2x4
    cols = list(carros.columns.drop('classe'))
    grid = [st.columns(2) for _ in range(4)]
    idx = 0
    explicacoes = {
        "preco": "Faixa de preço do veículo.",
        "manutencao": "Custo de manutenção do veículo.",
        "portas": "Quantidade de portas do veículo.",
        "pessoas": "Número de pessoas que o veículo comporta.",
        "porta_malas": "Tamanho do porta-malas.",
        "seguranca": "Nível de segurança do veículo."
    }
    for row in grid:
        for col_box in row:
            if idx < len(cols):
                col_name = cols[idx]
                with col_box:
                    st.write(f"Distribuição de '{col_name}' por classe:")
                    st.caption(explicacoes.get(col_name, ""))
                    fig, ax = plt.subplots(figsize=(6, 3))
                    sns.countplot(x=col_name, hue='classe', data=carros, ax=ax)
                    st.pyplot(fig)
                idx += 1


# Aba 3: Documentação
with tab3:
    st.subheader("Instruções de Uso")
    st.markdown("""
    1. Selecione as características do veículo nas caixas à esquerda.
    2. Clique em **Processar** para obter a previsão da qualidade do veículo.
    3. Veja a explicação do modelo e a influência das variáveis na aba **Explicabilidade**.
    4. Para dúvidas sobre o dataset, consulte abaixo:
    """)
    st.write("**Sobre o Dataset:**")
    st.write("O dataset contém informações categóricas sobre veículos, como preço, manutenção, número de portas, capacidade de passageiros, tamanho do porta-malas e segurança. A variável alvo é a qualidade do veículo ('class').")
    st.dataframe(carros.head())

# Aba 4: Desenvolvedor
with tab4:
    st.subheader("Sobre o Desenvolvedor")
    st.markdown("""
    **Giulliano Veiga**  
    Cientista de Dados  
    WhatsApp: [+55 85 98170-8027](https://wa.me/5585981708027)  
    Instagram: [@giullianoveiga](https://instagram.com/giullianoveiga)  
    GitHub: [giullianoveiga](https://github.com/giullianoveiga)  
    LinkedIn: [Giulliano Veiga](https://www.linkedin.com/in/giulliano-veiga/)
    """)