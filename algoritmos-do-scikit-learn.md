# 🧠 Guia dos Algoritmos Essenciais do Scikit-Learn

> **Foco Prático:** No Scikit-Learn existem dezenas de algoritmos, mas no mercado de trabalho real **mais de 95% dos problemas são resolvidos por 2 a 3 famílias**. Este guia compila o que você realmente precisa saber para tomar decisões rápidas e fundamentadas.

---

## Sumário

1. [Onde entra o KNN? (O Algoritmo da Vizinhança)](#1-onde-entra-o-knn-o-algoritmo-da-vizinhança)
2. [Família 1: Modelos Lineares & Regularização](#2-família-1-modelos-lineares--regularização)
3. [Família 2: Árvores & Ensembles (Os Reis de Dados Tabulares)](#3-família-2-árvores--ensembles-os-reis-de-dados-tabulares)
4. [Família 3: Margens & Suporte (SVM)](#4-família-3-margens--suporte-svm)
5. [Tabela Comparativa de Decisão](#5-tabela-comparativa-de-decisão)
6. [O Fluxograma Mental do Mundo Real](#6-o-fluxograma-mental-do-mundo-real)
7. [Snippets Práticos Prontos para Pipeline](#7-snippets-práticos-prontos-para-pipeline)

---

## 1. Onde entra o KNN? (O Algoritmo da Vizinhança)

O **KNN (K-Nearest Neighbors / K-Vizinhos Mais Próximos)** opera sob o princípio popular: *"Diga-me com quem andas e te direi quem és"*.

```
                (Classe B)
          ●       ●
             \   /
      ● ─── [NOVO PONTO] ─── ▲ (Classe A)
             /   \
            ▲     ▲
```

### Como ele funciona:
* **No `.fit(X, y)`:** Ele é um *lazy learner* (aprendiz preguiçoso). **Não calcula pesos nem retas**. Ele apenas memoriza as coordenadas dos dados no hiperespaço.
* **No `.predict(X_novo)`:** Ele calcula a distância matemática (euclidiana) do novo dado contra **todas as linhas do dataset** e seleciona os $K$ vizinhos mais próximos:
  * **Regressão (`KNeighborsRegressor`):** A previsão é a **média** dos valores dos $K$ vizinhos.
  * **Classificação (`KNeighborsClassifier`):** A previsão é a **moda** (voto majoritário) dos $K$ vizinhos.

### Por que ele raramente é o modelo principal em produção?
1. **Latência de Inferência Horrível:** A cada nova previsão na API, ele precisa recalcular a distância contra todo o banco de dados. Em uma base com 500.000 linhas, uma única predição pode demorar segundos em vez de milissegundos.
2. **A Maldição da Dimensionalidade:** Em tabelas com mais de 15 a 20 colunas, todos os pontos ficam matematicamente "longe" uns dos outros no hiperespaço. O conceito geométrico de "vizinho próximo" perde a eficácia.
3. **Dependência Crítica de Escala:** Se uma coluna for salário (R$ 15.000) e outra for número de quartos (3), a distância euclidiana ignorará completamente os quartos se você não aplicar `StandardScaler`.

### Onde o KNN brilha no mundo real?
* **Sistemas de Recomendação Simples:** Encontrar os 5 usuários com comportamento de compra mais próximo de um cliente específico.
* **Imputação Inteligente (`KNNImputer`):** Em vez de preencher valores nulos com a média global, o `KNNImputer` preenche o nulo com a média das linhas mais parecidas geometricamente.

---

## 2. Família 1: Modelos Lineares & Regularização

*Ideal para baselines ultra-rápidos, auditoria, conformidade regulatória e problemas onde a explicabilidade é mandatória.*

```
     y
     │         / (Reta do Modelo)
     │       /  ●
     │   ● /
     │   / ●
     │ /
     └───────────── x
```

### 1. `LinearRegression` / `LogisticRegression`
* **Conceito:** Encontra a reta ou hiperplano que minimiza a soma dos erros quadráticos (regressão) ou separa com uma curva sigmoide (classificação).
* **Vantagens:** Instantâneo para treinar, fácil de exportar, altamente interpretável (pesos diretos).
* **Fraquezas:** Sofre gravemente com multicolinearidade (colunas correlacionadas) e não aprende não-linearidades sem engenharia manual.
* **Exige Scaler?** **Sim**, obrigatoriamente.

### 2. `Ridge` (Regularização $L_2$)
* **O Problema que resolve:** Em bases com muitas colunas ou correlações altas, a `LinearRegression` pode explodir os pesos para valores gigantescos tentando compensar pequenas variações (overfitting).
* **Como funciona:** Adiciona uma penalidade proporcional ao quadrado dos pesos ($\alpha \sum w_i^2$). Isso força os coeficientes a encolherem suavemente em direção a zero, tornando o modelo estável.
* **Hiperparâmetro Chave:** `alpha` (quanto maior, mais o modelo resiste ao ruído).

### 3. `Lasso` (Regularização $L_1$)
* **O Problema que resolve:** Você tem uma base com 100 colunas e suspeita que a maioria é inútil ou irrelevante.
* **Como funciona:** Adiciona uma penalidade proporcional ao valor absoluto dos pesos ($\alpha \sum |w_i|$). O efeito matemático é fascinante: **o Lasso zera completamente os pesos das colunas inúteis!**
* **O Superpoder:** Funciona como um **seletor automático de features**.
* **Hiperparâmetro Chave:** `alpha` (quanto maior, mais colunas são eliminadas).

---

## 3. Família 2: Árvores & Ensembles (Os Reis de Dados Tabulares)

*Dominam as competições de dados estruturados e os sistemas corporativos modernos. São imunes a escalas e capturam qualquer não-linearidade.*

```
               [ Renda > 50.000? ]
                  /           \
               Sim             Não
              /                   \
      [ Quartos > 3? ]      [ Idade > 30? ]
```

### 1. `RandomForestRegressor` / `Classifier` (Bagging)
* **Como pensa:** *"A sabedoria das multidões independentes"*.
* **Mecanismo:** Treina centenas de árvores de decisão **em paralelo**. Cada árvore recebe uma amostra aleatória das linhas (*bootstrap*) e um sorteio aleatório de colunas. A previsão final é a média (ou votação) de todas as árvores.
* **Vantagens:** Praticamente impossível de sofrer overfitting severo sem aviso. Robusta a outliers e ruído.
* **Hiperparâmetros Chave:**
  * `n_estimators`: Número de árvores (ex: 100 a 500).
  * `max_depth`: Profundidade máxima de cada árvore.
* **Exige Scaler?** **Não.**

### 2. `HistGradientBoostingRegressor` / `Classifier` (Boosting)
* **Como pensa:** *"A melhoria contínua e sequencial"*.
* **Mecanismo:** Treina árvores **em fila indiana sequencial**:
  1. A Árvore 1 faz previsões preliminares (e erra bastante).
  2. A Árvore 2 é treinada **especificamente para prever os resíduos (erros) da Árvore 1**.
  3. A Árvore 3 treina para prever os erros restantes da Árvore 2, e assim por diante.
* **Vantagens:** O algoritmo de maior performance nativo do Scikit-Learn. Trata valores nulos (`NaN`) nativamente sem precisar de imputer e é ultra-rápido (usa *binning* com histogramas, similar ao LightGBM).
* **Hiperparâmetros Chave:**
  * `learning_rate`: Taxa de aprendizado de cada árvore (ex: 0.05 a 0.1).
  * `max_iter`: Quantidade máxima de rodadas de árvores.
  * `min_samples_leaf`: Mínimo de amostras por folha (evita decorar casos isolados).
* **Exige Scaler?** **Não.**

---

## 4. Família 3: Margens & Suporte (SVM)

### `SVC` / `SVR` (Support Vector Machines)
* **Como pensa:** Procura o hiperplano que garante a maior "margem de segurança" (distância) entre os pontos mais difíceis das classes (os vetores de suporte).
* **O "Truque do Kernel" (Kernel Trick):** Usando funções matemáticas (como o kernel `rbf`), o SVM consegue projetar os dados para um espaço de infinitas dimensões onde dados antes inseparáveis se tornam separáveis por um plano reto.
* **Quando Usar:** Datasets com **poucas linhas ($N < 50.000$) e muitas colunas complexas** (bioinformática, genética, análise espectral).
* **Por que não é o padrão de mercado?** O custo computacional escala entre $O(N^2)$ e $O(N^3)$. Se o dataset tiver 100.000 linhas, ele pode travar a máquina por horas.
* **Exige Scaler?** **Sim**, obrigatoriamente.

---

## 5. Tabela Comparativa de Decisão

| Algoritmo | Família | Exige Scaler? | Imune a Outliers? | Velocidade de Treino | Latência de Predição (API) | Interpretabilidade | Caso de Uso Típico |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Linear / Logistic** | Linear | **Sim** | 🔴 Não | ⚡ Instantâneo | ⚡ Instantânea | 🟢 Total (pesos diretos) | Baselines, modelos regulados por lei |
| **Ridge** | Linear ($L_2$) | **Sim** | 🔴 Não | ⚡ Instantâneo | ⚡ Instantânea | 🟢 Total | Evitar overfitting em modelos lineares |
| **Lasso** | Linear ($L_1$) | **Sim** | 🔴 Não | ⚡ Instantâneo | ⚡ Instantânea | 🟢 Total | Seleção automática de variáveis |
| **Random Forest** | Árvores | **Não** | 🟢 Sim | 🟡 Médio | 🟢 Rápida | 🟡 Média (importâncias) | Modelo "pau pra toda obra" sem tunar |
| **HistGradientBoosting** | Árvores | **Não** | 🟢 Sim | 🟢 Rápido | 🟢 Rápida | 🔴 Baixa (Caixa-preta) | **Máxima nota e performance em produção** |
| **KNN** | Distância | **Sim** | 🔴 Não | ⚡ Instantâneo | 🔴 Lenta | 🟢 Intuitivo | Sistemas de recomendação e `KNNImputer` |
| **SVM (SVC/SVR)** | Margens | **Sim** | 🟡 Média | 🔴 Lento em $N$ grande | 🟡 Média | 🔴 Baixa | Poucas linhas com dezenas de colunas |

---

## 6. O Fluxograma Mental do Mundo Real

Sempre que receber uma base nova para modelagem, siga esta estratégia de 3 passos:

```
                  [ Recebeu os Dados ]
                            │
                            ▼
              ┌───────────────────────────┐
              │ 1. Crie o Baseline Linear │
              │    (Ridge ou Logistic)    │
              └─────────────┬─────────────┘
                            │ (Anota métricas e tempo)
                            ▼
              ┌───────────────────────────┐
              │ 2. Rode Gradient Boosting │
              │  (HistGradientBoosting)   │
              └─────────────┬─────────────┘
                            │ (Descobre o teto de performance)
                            ▼
                   [ Comparar as Notas ]
                  /                     \
        Diferença Desprezível        Diferença Significativa
       (ex: 70.1% vs 70.5%)            (ex: 64% vs 83%)
                 │                               │
                 ▼                               ▼
       Fique com o Linear!            Fique com o Boosting!
    (Mais simples, leve e seguro)    (O ganho justifica o modelo)
```

---

## 7. Snippets Práticos Prontos para Pipeline

### Troca Rápida de Estimador na Mesma Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor

# 1. Pipeline base configurada:
pipeline = Pipeline(steps=[
    ("preparador", preparador_de_colunas),
    ("modelo", Ridge(alpha=1.0)) # <-- Começa com Ridge
])

# 2. Testar Lasso (zera colunas inúteis):
pipeline.set_params(modelo=Lasso(alpha=0.1))
pipeline.fit(X_train, y_train)

# 3. Testar Random Forest:
pipeline.set_params(modelo=RandomForestRegressor(n_estimators=200, random_state=42))
pipeline.fit(X_train, y_train)

# 4. Testar o Campeão de Performance (HistGradientBoosting):
pipeline.set_params(modelo=HistGradientBoostingRegressor(random_state=42))
pipeline.fit(X_train, y_train)
```
