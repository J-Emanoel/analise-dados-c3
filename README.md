# Sistema Analítico Ames Housing - Análise de Dados Aplicada à Computação 

## Sobre o Projeto
Este repositório contém a solução desenvolvida para o Desafio de Análise de Dados e Machine Learning (Hackathon de Preços de Casas). O objetivo do projeto é aplicar o pipeline completo de Ciência de Dados — desde a Análise Exploratória (EDA) e Feature Engineering até a criação de modelos de Aprendizado Supervisionado e Não Supervisionado — para prever e entender a dinâmica dos preços de imóveis.

## Estrutura do Repositório
* `trabalho_final_limpo.ipynb`: Jupyter Notebook contendo todo o código de tratamento de dados, modelagem matemática (Random Forest, K-Means, PCA, Isolation Forest) e o Storytelling analítico do projeto.
* `app.py`: Código-fonte do Dashboard interativo construído para apoiar a visualização dos resultados de forma dinâmica.
* `train.csv` / `test.csv`: Bases de dados originais utilizadas no desafio.

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
   pip install pandas numpy scikit-learn matplotlib seaborn plotly streamlit
  3.Inicie o servidor local da aplicação com o comando:
     streamlit run app.py
