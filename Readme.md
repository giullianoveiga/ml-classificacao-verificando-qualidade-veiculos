# Classificação de Veículos com Machine Learning

Este projeto utiliza Streamlit para criar uma aplicação web interativa que prevê a qualidade de veículos com base em características categóricas, utilizando o algoritmo Naive Bayes.

## Funcionalidades

- Interface intuitiva para seleção das características do veículo
- Previsão da qualidade do veículo (inaceitável, aceitável, bom, muito bom)
- Visualização da influência das variáveis por meio de gráficos
- Ajuda contextual nos campos e gráficos
- Documentação e instruções de uso
- Aba do desenvolvedor com contatos e redes sociais

## História de Uso Fictícia

Imagine que João trabalha em uma concessionária e precisa avaliar rapidamente a qualidade de diversos veículos usados que chegam para revenda. Antes, ele gastava horas analisando manualmente cada ficha técnica, correndo o risco de cometer erros ou ser influenciado por opiniões pessoais.

Com esta aplicação, João simplesmente insere as características principais de cada carro (preço, manutenção, número de portas, capacidade de passageiros, porta-malas e segurança) e recebe instantaneamente uma previsão confiável sobre a qualidade do veículo. Assim, ele pode tomar decisões mais rápidas e seguras, garantindo que apenas veículos de boa qualidade sejam oferecidos aos clientes, aumentando a reputação da loja e a satisfação dos compradores.

Esta solução automatiza o processo de avaliação, reduz o tempo de análise e traz mais confiança para o negócio de João!

Projeto para fins educacionais e demonstração de técnicas de classificação com dados categóricos.

## Como executar

1. Instale as dependências:
   ```bash
   pip install streamlit pandas scikit-learn matplotlib seaborn
   ```
2. Execute o aplicativo:
   ```bash
   streamlit run carros.py
   ```
3. Acesse o navegador no endereço indicado pelo Streamlit.

## Estrutura dos arquivos

- `carros.py`: Código principal da aplicação
- `carros.csv` ou `carros_traduzido.csv`: Dataset de veículos
- `Readme.md`: Este arquivo de documentação

## Sobre o Dataset

O dataset contém informações categóricas sobre veículos:

- **preco**: Faixa de preço do veículo
- **manutencao**: Custo de manutenção
- **portas**: Número de portas
- **pessoas**: Capacidade de passageiros
- **porta_malas**: Tamanho do porta-malas
- **seguranca**: Nível de segurança
- **classe**: Qualidade do veículo (alvo)

## Desenvolvedor

- **Giulliano Veiga**
- Cientista de Dados
- WhatsApp: [+55 85 98170-8027](https://wa.me/5585981708027)
- Instagram: [@giullianoveiga](https://instagram.com/giullianoveiga)
- GitHub: [giullianoveiga](https://github.com/giullianoveiga)
- LinkedIn: [Giulliano Veiga](https://www.linkedin.com/in/giulliano-veiga/)