# Sistema Analítico Ames Housing - Análise de Dados Aplicada à Computação 

## Acesse o Dashboard
Explore a versão interativa do projeto diretamente no Streamlit:

**[Abrir Dashboard Ames Housing](https://avaliacaoc3.streamlit.app/)**

Use os filtros laterais para navegar pelos bairros, comparar preços e visualizar os resultados de regressão, classificação, clusterização, PCA e detecção de outliers.

## Sobre o Projeto
Este repositório contém a solução desenvolvida para o Desafio de Análise de Dados e Machine Learning (Hackathon de Preços de Casas). O objetivo do projeto é aplicar o pipeline completo de Ciência de Dados — desde a Análise Exploratória (EDA) e Feature Engineering até a criação de modelos de Aprendizado Supervisionado e Não Supervisionado — para prever e entender a dinâmica dos preços de imóveis.

## Estrutura do Repositório
* `notebooks/trabalho_analise_c3.ipynb`: Jupyter Notebook contendo todo o código de tratamento de dados, modelagem matemática (Random Forest, K-Means, PCA, Isolation Forest) e o Storytelling analítico do projeto.
* `outputs/app.py`: Código-fonte do Dashboard interativo em Streamlit.
* `data/train.csv` / `data/test.csv`: Bases de dados originais utilizadas no desafio.

## Tecnologias e Bibliotecas Utilizadas
* **Linguagem:** Python
* **Manipulação e Tratamento de Dados:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn (Random Forest Regressor/Classifier, K-Means, PCA, Isolation Forest)
* **Visualização:** Matplotlib, Seaborn, Plotly
* **Frontend / Dashboard:** Streamlit

## Como Executar o Dashboard Localmente
Caso o avaliador deseje rodar o visualizador interativo, basta seguir os passos abaixo no terminal:

1. Clone este repositório para a sua máquina.
2. Certifique-se de ter o Python instalado e instale as dependências executando:
   ```bash
   python -m pip install -r requirements.txt
   ```
3. Inicie o servidor local da aplicação com o comando:
   ```bash
   python -m streamlit run outputs/app.py
   ```

Versão publicada no Streamlit: [https://avaliacaoc3.streamlit.app/](https://avaliacaoc3.streamlit.app/)

Autores / Membros do Grupo
Paulo Henrique Nascimento

Murilo Daré 

Eduardo Gobii

Luis Felipe Andrade

João Emanoel
