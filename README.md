
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

## 📊 Análise Exploratória dos Dados (EDA)

### **Características do Dataset**
- **Tamanho**: 32.561 observações com 17 variáveis
- **Composição**: 8 variáveis numéricas e 9 categóricas
- **Valores Faltantes**: Nenhum (após limpeza)

### **Variáveis Numéricas - Estatísticas Descritivas**

| Variável | Média | Desvio Padrão | Mínimo | Q1 | Mediana | Q3 | Máximo |
|----------|-------|---------------|--------|-----|---------|-----|--------|
| **age** | 38,58 | 13,64 | 17 | 28 | 37 | 48 | 90 |
| **fnlwgt** | 189.778 | 105.550 | 12.285 | 117.827 | 178.356 | 237.051 | 1.484.705 |
| **education-num** | 10,09 | 2,57 | 1 | 9 | 10 | 13 | 16 |
| **capital-gain** | 1.077,65 | 7.385,29 | 0 | 0 | 0 | 0 | 99.999 |
| **capital-loss** | 87,30 | 402,96 | 0 | 0 | 0 | 0 | 4.356 |
| **hours-per-week** | 40,44 | 12,39 | 1 | 40 | 40 | 45 | 99 |

**Insights**: A idade média de ~39 anos reflete uma força de trabalho madura. A mediana de horas semanais (40h) é o padrão, mas há variação considerável. Capital-gain/loss são zeros em maioria (75% quartil inferior = 0), indicando alta assimetria.

### **Distribuição da Variável Target**

**Classe 0 (Salário ≤ $50K)**: 24.720 casos (75,92%)  
**Classe 1 (Salário > $50K)**: 7.841 casos (24,08%)

⚠️ **Dataset Desbalanceado**: A proporção 75%-25% indica desbalanceamento moderado. Modelos tendem a priorizar a classe majoritária, afetando recall da classe minoritária.

### **Correlação com Target** (salary_numeric_proxy)

| Variável | Correlação |
|----------|-----------|
| education-num | +0,335 |
| age | +0,234 |
| hours-per-week | +0,229 |
| capital-gain | +0,223 |
| capital-loss | +0,151 |
| fnlwgt | -0,009 |

**Achados principais**:
- **Educação** é o preditor mais forte (r=0,34): mais anos de educação correlacionam com salário mais alto
- **Idade** tem correlação positiva (r=0,23): experiência influencia salário
- **Horas trabalhadas** (r=0,23): trabalhar mais associa-se a salários mais altos
- **Ganhos de capital** (r=0,22): investimentos influenciam classe salarial
- **Peso da amostra (fnlwgt)** quase não correlaciona: variável técnica, não preditora

### **Variáveis Categóricas - Distribuição Principal**

| Variável | Categoria Principal | Frequência |
|----------|---------------------|-----------|
| **workclass** | Private | 22.696 (69,7%) |
| **education** | HS-grad | 10.501 (32,2%) |
| **marital-status** | Married-civ-spouse | 14.976 (46,0%) |
| **occupation** | Prof-specialty | 4.140 (12,7%) |
| **race** | White | 27.816 (85,4%) |
| **sex** | Male | 21.790 (66,9%) |
| **native-country** | United-States | 29.170 (89,6%) |

**Observações**:
- Setor **privado** domina (70% dos dados)
- Maioria com educação **ensino médio** (~32%) ou alguns cursos superiores (~22%)
- **Casados** constituem 46% da amostra
- Distribuição **desequilibrada por gênero** (67% homens vs 33% mulheres) — pode introduzir viés
- Amostra **predominantemente norte-americana e caucasiana** (89% EUA, 85% brancos) — restringe generalização

### **Implicações para Modelagem**

1. **Variáveis Numéricas**: Distribuições assimétricas (ex: capital-gain) justificam padronização (StandardScaler)
2. **Categorias Raras**: Algumas categorias em 'native-country' e 'occupation' aparecem <1% — one-hot encoding com `drop='first'` evita multicolinearidade
3. **Desbalanceamento**: Importância em considerar métricas além de acurácia (recall, F1, AUC)
4. **Potencial Viés**: Dados não representam mulheres e minorias proporcionalmente — resultados com ressalva

---

## 🔍 Principais Descobertas

### **Dataset e Preparação dos Dados**
- **Divisão Treino/Teste**: 70% / 30% (estratificada na classificação para manter proporções de classes)
- **Imputação**: Valores faltantes tratados com mediana (numéricas) e moda (categóricas)
- **Codificação**: One-hot encoding das 9 variáveis categóricas com `drop='first'` para evitar multicolinearidade
- **Padronização**: StandardScaler aplicado às 8 variáveis numéricas para garantir igualdade de escala

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

## 🔬 Análise de Coeficientes e Interpretação

### **Regressão Linear - Interpretabilidade**
- **Coeficientes Positivos**: Aumentam o salário predito
  - Educação: +correlação forte → cada ano adicional aumenta salário esperado
  - Idade: +correlação → experiência adiciona valor
  - Horas de trabalho: +correlação → mais horas = salário maior
- **Limitação**: Com R²=0,36, coeficientes explicam pouco da variação total
- **Implicação Prática**: Modelo inadequado para prever salários individuais com confiança

### **Regressão Logística - Interpretabilidade (Odds Ratio)**
- **Coeficientes** representam mudança na probabilidade de "salário alto"
- **Exemplo Interpretativo**: Se coeficiente de educação é +0,15, cada ano adicional de educação **multiplica as odds de salário alto por e^0,15 ≈ 1,16 (+16%)**
- **Vantagem**: Probabilidades são diretas e acionáveis
  - Resultado: "Probabilidade de salário >50K para esta pessoa: 72%"
  - Aplicável em decisões de negócio (elegibilidade para benefícios, promoções)

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

### **Resumo Executivo Final**

**O que foi feito:**
Desenvolvemos dois modelos preditivos para classificar salários em faixas (≤$50K vs >$50K) usando dataset de 32.561 registros com 17 variáveis (econômicas, demográficas e laborais).

**O que descobrimos:**
1. **Regressão Linear** (prever valor contínuo): R²=0,36, RMSE=$13.612 → modelo fraco, inadequado
2. **Regressão Logística** (classificação binária): Acurácia=85%, AUC=0,91 → modelo forte, confiável
3. **Fatores Preditivos**: Educação (r=0,34), idade (r=0,23), horas trabalhadas (r=0,23) são os melhores indicadores
4. **Dataset Desbalanceado**: 76% salários baixos, 24% salários altos → explica recall moderado (61%)

**Qual abordagem escolhemos:**
**Regressão Logística**. Razões:
- Alinha-se com natureza binária do problema (categorizar pessoas em faixas salariais)
- Fornece probabilidades interpretáveis ("essa pessoa tem 72% de chance de ganhar >$50K")
- Desempenho superior (AUC=0,91 vs R²=0,36)
- Aplicável diretamente em decisões RH (benefícios, promoções, recrutamento)
- Coeficientes interpretáveis como odds ratio (mudança de probabilidade por unidade de feature)

**Limitações Atuais & Próximos Passos:**
- Recall de 61% deixa 39% dos salários altos não-identificados → usar SMOTE para balancear classes
- Dataset enviesado (67% homens, 85% EUA) → testar em dados mais diversos
- Explorar regularização (L1/L2) e validação cruzada para melhorar generalização
- Considerar Gradient Boosting ou Random Forest para capturar não-linearidades

**Impacto Estimado**: Modelo final pode economizar tempo em triagem de candidatos, com 91% de confiança na discriminação entre salários alto/baixo.

---

**Data**: 17 de novembro de 2025  
**Dataset**: Salary Prediction Classification (Kaggle)  
**Ferramentas**: Python, scikit-learn, pandas  
**Divisão**: 70% treino / 30% teste (estratificada)  
**Status**: ✅ Concluído

