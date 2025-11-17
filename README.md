
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


# Resumo Executivo: Previsão de Salários - Regressão vs. Classificação

## 📋 Síntese da Atividade

Este projeto explorou dois paradigmas complementares para análise de dados salariais usando o dataset "Salary Prediction Classification" do Kaggle (32.561 registros). A atividade comparou a abordagem de **regressão linear** (prever salário contínuo) com a de **regressão logística** (classificar em categorias de salário), investigando quando cada método é mais apropriado.

---

## 🔍 Principais Descobertas

### **Dataset e Preparação**
- **Tamanho**: 32.561 observações com 17 variáveis (6 numéricas, 8 categóricas)
- **Variável Alvo Original**: `salary` (binária: ≤50K vs >50K)
- **Engenharia de Features**: 
  - Criação de `salary_numeric_proxy` para regressão (valores contínuos: 30.000 e 70.000)
  - Manutenção de `salary_class` (0/1) para classificação
  - Imputação de valores faltantes (mediana para numéricos, moda para categóricos)
  - Codificação one-hot das 8 variáveis categóricas
  - Padronização das variáveis numéricas (StandardScaler)
- **Divisão Treino/Teste**: 70% / 30% (estratificada na classificação)

---

## 📊 Resultados dos Modelos

### **Modelo 1: Regressão Linear**
| Métrica | Valor |
|---------|-------|
| MAE (Erro Absoluto Médio) | $10.579,89 |
| RMSE (Raiz do Erro Quadrático Médio) | $13.612,42 |
| R² (Coeficiente de Determinação) | 0,3593 |

**Interpretação**: O modelo explica apenas ~36% da variância nos salários. O erro médio de $10.579 significa que as previsões desviam, em média, dessa quantia do valor real. Isso indica que variáveis contínuas sozinhas não capturam toda a complexidade do problema.

### **Modelo 2: Regressão Logística (Classificação)**
| Métrica | Valor |
|---------|-------|
| Acurácia | 85,45% |
| Precisão | 73,86% |
| Recall (Sensibilidade) | 61,27% |
| F1-Score | 66,98% |
| AUC-ROC | 0,9088 |

**Interpretação**: O modelo classifica corretamente 85% dos casos. Com 91% AUC-ROC, demonstra excelente capacidade discriminativa. A precisão de 74% significa que quando prediz "salário alto", acerta 74% das vezes. O recall de 61% indica que identifica apenas 61% dos casos de salário alto (limitação importante).

---

## ⚖️ Comparação Entre Abordagens

| Critério | Regressão Linear | Regressão Logística |
|----------|------------------|-------------------|
| **Objetivo** | Prever valor contínuo | Classificar em categorias |
| **Desempenho** | Modesto (R²=0,36) | Forte (Acurácia=85%) |
| **Interpretabilidade** | Simples: coeficientes = mudança em $Y por unidade de $X$ | Simples: odds ratio para cada feature |
| **Uso Prático** | Salarial estimado individual | Decisões binárias (contratar/benefícios) |
| **Sensibilidade** | Afetado por outliers e escala | Robusto a outliers, baseado em probabilidades |
| **Métricas Apropriadas** | MAE, RMSE, R² | Acurácia, Precisão, Recall, F1, AUC |

### **Por que a Classificação Superou?**

1. **Alinhamento com os Dados**: O dataset foi originalmente criado com uma **classificação binária** (≤50K vs >50K). A regressão forçou valores contínuos (30K e 70K), perdendo informação real.

2. **Natureza do Problema**: Decisões de RH (benefícios, categorias salariais) são tipicamente binárias. Classificação é mais alinhada ao caso de uso real.

3. **Limitações da Regressão Linear**: 
   - R² baixo indica que relacionamento linear não captura padrões
   - Regressão não respeita a natureza discreta do problema original
   - Previsões podem ficar fora do intervalo esperado (ex: salário negativo)

---

## 💡 Abordagem Recomendada

### **Escolha: Regressão Logística**

**Justificativa:**

1. **Desempenho Superior**: 85% de acurácia vs. R²=36% da regressão (incomparáveis, mas a classificação é muito mais efetiva)

2. **Alinhamento Conceitual**: O problema é naturalmente binário (faixa de salário). Forçar regressão é um mau-uso do método.

3. **Probabilidades Interpretáveis**: Fornece $P(\text{salário alto})$ para cada pessoa — informação acionável.

4. **Métricas Adequadas**:
   - AUC-ROC de 0,91 = modelo excelente em discriminar classes
   - Precisão 74% = confiável para identificar salários altos
   - Recall 61% = detecta maioria dos altos salários (room para melhoria)

5. **Contexto Real**: HR e negócios pensam em categorias, não em valores contínuos.

---

## 🚀 Melhorias Propostas

Para otimizar ainda mais a **regressão logística**:

1. **Feature Engineering**:
   - Criar interações entre variáveis (ex: idade × educação)
   - Polinômios para variáveis numéricas
   - Agregações de categorias categóricas raramente frequentes

2. **Regularização**:
   - Introduzir L1 (Lasso) ou L2 (Ridge) para evitar overfitting
   - Validação cruzada k-fold para seleção de hiperparâmetros

3. **Balanceamento de Classes**:
   - Dataset é desbalanceado (76% classe 0, 24% classe 1)
   - Usar SMOTE, class_weight ou threshold ajustado

4. **Modelos Alternativos**:
   - Random Forest / Gradient Boosting (melhor captura de não-linearidades)
   - Ensemble de múltiplos modelos

5. **Análise de Erros**:
   - Investigar os 39% de falsos negativos (salários altos preditos como baixos)
   - Pode haver clusters não capturados

---

## 📌 Conclusão

A **regressão logística é a abordagem recomendada** para este dataset. Enquanto regressão linear fornece interpretabilidade, ela é inadequada para um problema fundamentalmente classificatório. A regressão logística combina interpretabilidade (importante), desempenho robusto (AUC=0,91) e alinhamento com o contexto real (decisões categóricas em RH).

---

**Data**: 17 de novembro de 2025  
**Dataset**: Salary Prediction Classification (Kaggle)  
**Ferramentas**: Python, scikit-learn, pandas  
**Tempo de Teste**: 70% treino / 30% teste
