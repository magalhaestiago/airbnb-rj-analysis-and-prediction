# Rio de Janeiro Airbnb Price Prediction

Este projeto é uma solução de Machine Learning desenvolvida para prever preços de diárias de imóveis no Rio de Janeiro utilizando dados do [Inside Airbnb](http://insideairbnb.com/).

O objetivo é auxiliar anfitriões e viajantes a entenderem o valor justo de uma estadia com base em características como localização, número de quartos, comodidades e avaliações.

## Projeto

### Problema
O dataset original possui desafios significativos, incluindo:
* Dados faltantes (NaN) em colunas críticas.
* Necessidade de engenharia de features para converter, remover features redundantes ou desnecessárias, ou então criar novas.

### Solução
Desenvolvemos um pipeline completo que:
1.  **Processa os Dados:** Limpeza, tratamento de nulos e conversão de tipos.
2.  **Treina Modelos:** Comparação entre **XGBoost** e **Random Forest**.
3.  **Disponibiliza via API:** Uma interface RESTful usando **FastAPI** para realizar previsões em tempo real.


![folium-example](folium_example.gif)

---

## Estrutura do Repositório

A organização das pastas segue o padrão abaixo:

```bash
├── backend/
│   └── api.py             # Código fonte da API (FastAPI)
├── csvs/
│   └── listings.csv
├── models/
│   ├── model_metadata.json  # Variáveis utilizadas para treinar o modelo
│   ├── xgboost_model.joblib # Modelo treinado (XGBoost)
│   └── random_forest.joblib # Modelo treinado (Random Forest)
├── official.ipynb          # Notebook de EDA, Pré-processamento e Treino
├── requirements.txt        # Dependências do projeto
└── README.md               # Documentação
```

## Instalação

```
git clone https://github.com/magalhaestiago/airbnb-rj-analysis-and-prediction.git

cd airbnb-rj-analysis-and-prediction

python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

pip install -r requirements.txt
```

## Execução

Observação: Pelo tamanho do modelo, não foi possível subi-lo para o repositório GitHub. Então é necessário rodar o notebook `Airbnb analysis and Prediction.ipynb` para gerar os arquivos de modelo na pasta `models/`.

```
cd backend

fastapi run
```


