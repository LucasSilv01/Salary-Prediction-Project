
# Salary Prediction Project

## Visão geral
Projeto para análise e modelagem de previsão de salários (regressão e classificação) a partir do dataset `salary_prepared_with_targets.csv`.

O repositório inclui:
- EDA (análise exploratória)
- Modelos: Regressão Linear e Regressão Logística
- Pipeline simples de pré-processamento
- Scripts organizados em `src/`
- Notebook e relatórios gerados

## Estrutura do projeto
```
salary_project/
├── main.py
├── requirements.txt
├── setup.py
├── README.md
└── src/
    ├── eda.py
    ├── model_regression.py
    ├── model_classification.py
    ├── preprocessing.py
    └── utils.py
```

## Como usar (local / VS Code)
1. Coloque `salary_prepared_with_targets.csv` na raiz do projeto (mesmo nível de `main.py`).
2. Instale dependências:
```bash
pip install -r requirements.txt
```
3. Rodar app:
```bash
python main.py
```

## Instalando como pacote (opcional)
Para instalar localmente como pacote:
```bash
pip install .
```
Isso instalará o pacote `salary_project` e você poderá importar funções no Python.
