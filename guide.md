# ⚡ Guia Prático de Scikit-Learn: Os 20% Que Resolvem 80% dos Problemas

> **"Domine o fluxo de trabalho essencial antes de se perder em dezenas de algoritmos complexos."**
> No dia a dia de Ciência de Dados, 80% dos resultados vêm de: entender a estrutura dos dados ($X$ e $y$), tratar variáveis corretamente, dividir treino/teste sem vazamento de dados, usar pipelines e avaliar com as métricas certas.
>
> Este guia adota a **Regressão Linear (`LinearRegression`)** como base didática central para você dominar todos os conceitos fundamentais do Scikit-Learn.

---

## Sumário

1. [O Modelo Mental do Scikit-Learn (O Básico do Básico)](#1-o-modelo-mental-do-scikit-learn-o-básico-do-básico)
2. [A Estrutura dos Dados: $X$ vs $y$ (Onde a Maioria Erra)](#2-a-estrutura-dos-dados-x-vs-y-onde-a-maioria-erra)
3. [Primeiro Modelo Prático: Passo a Passo com `LinearRegression`](#3-primeiro-modelo-prático-passo-a-passo-com-linearregression)
4. [Divisão Treino e Teste (`train_test_split`)](#4-divisão-treino-e-teste-traintestsplit)
5. [Pré-processamento: Tratando Dados Reais](#5-pré-processamento-tratando-dados-reais)
6. [O Maestro dos Dados: `ColumnTransformer`](#6-o-maestro-dos-dados-columntransformer)
7. [Pipelines: O Segredo de um Código Limpo e Profissional](#7-pipelines-o-segredo-de-um-código-limpo-e-profissional)
8. [Métricas de Avaliação para Regressão (Como Saber se o Modelo é Bom)](#8-métricas-de-avaliação-para-regressão)
9. [Validação Cruzada (`cross_val_score`)](#9-validação-cruzada-crossvalscore)
10. [Salvando e Carregando o Modelo (`joblib`)](#10-salvando-e-carregando-o-modelo-joblib)
11. [Template Mestre Ponta a Ponta (Código 100% Executável)](#11-template-mestre-ponta-a-ponta-código-100-executável)
12. [Top 5 Armadilhas Comuns e Como Evitá-las](#12-top-5-armadilhas-comuns-e-como-evitá-las)

---

## 1. O Modelo Mental do Scikit-Learn (O Básico do Básico)

O Scikit-Learn segue um padrão de orientação a objetos rigoroso e uniforme. Se você aprender o funcionamento de um objeto, saberá usar quase todos da biblioteca.

Existem 3 tipos principais de classes:

| Tipo                  | Métodos Principais                          | O que faz                                           | Exemplos                                                 |
| --------------------- | -------------------------------------------- | --------------------------------------------------- | -------------------------------------------------------- |
| **Estimator**   | `.fit(X, y)`                               | Aprende parâmetros a partir dos dados              | `LinearRegression`, `StandardScaler`                 |
| **Transformer** | `.fit_transform(X, y)` / `.transform(X)` | Modifica, normaliza ou codifica características    | `StandardScaler`, `OneHotEncoder`, `SimpleImputer` |
| **Predictor**   | `.predict(X)`                              | Faz previsões em novos dados usando o que aprendeu | `LinearRegression`, `Ridge`                          |

```
                 ┌────────────────────────┐
                 │  Dados de Treino (X, y)│
                 └───────────┬────────────┘
                             │
                         .fit(X, y)   <-- O modelo aprende pesos e intercepto
                             │
                             ▼
                 ┌────────────────────────┐
                 │    Modelo Ajustado     │
                 └───────────┬────────────┘
                             │
                      .predict(X_novos)   <-- Gera as previsões
                             │
                             ▼
                 ┌────────────────────────┐
                 │       Previsões        │
                 └────────────────────────┘
```

---

## 2. A Estrutura dos Dados: $X$ vs $y$ (Onde a Maioria Erra)

Para o Scikit-Learn funcionar sem erros de dimensão (*shape errors*), seus dados precisam ter formatos bem específicos:

* **$X$ (Matriz de Características / Features)**: Deve ser **bidimensional (2D)** com formato `(n_linhas, n_colunas)`.
  * Em Pandas: sempre selecione com uma lista de colunas $\rightarrow$ `df[['coluna1', 'coluna2']]` ou `df[['coluna_unica']]` (note os **colchetes duplos**).
* **$y$ (Vetor Alvo / Target)**: Deve ser **unidimensional (1D)** com formato `(n_linhas,)`.
  * Em Pandas: selecione como Series $\rightarrow$ `df['alvo']` (com **colchetes simples**).

### Exemplo em Código:

```python
import pandas as pd

# Criando um dataset simples de exemplo
df = pd.DataFrame({
    "area_m2": [50, 70, 90, 120, 150],
    "quartos": [1, 2, 2, 3, 4],
    "preco": [180000, 250000, 310000, 420000, 530000]
})

# X: 2D (Tabela com as características explicativas)
X = df[["area_m2", "quartos"]]

# y: 1D (Vetor com o valor que queremos prever)
y = df["preco"]

print("Formato de X (2D):", X.shape) # (5, 2)
print("Formato de y (1D):", y.shape) # (5,)
```

> ⚠️ **Atenção:** Se você fizer `X = df['area_m2']`, o Scikit-Learn vai lançar o erro:
> `ValueError: Expected 2D array, got 1D array instead`.
> Para corrigir uma coluna única, use sempre `df[['area_m2']]`!

---

## 3. Primeiro Modelo Prático: Passo a Passo com `LinearRegression`

Vamos construir e dissecar um exemplo básico com a Regressão Linear:

```python
import pandas as pd
from sklearn.linear_model import LinearRegression

# 1. Dados de exemplo
dados = pd.DataFrame({
    "area_m2": [40, 60, 80, 100, 120],
    "quartos": [1, 2, 2, 3, 3],
    "preco": [150000, 220000, 290000, 360000, 430000]
})

X = dados[["area_m2", "quartos"]]
y = dados["preco"]

# 2. Instanciar o modelo
modelo = LinearRegression()

# 3. Treinar o modelo (ajustar a reta / hiperplano)
modelo.fit(X, y)

# 4. Fazer previsões para um novo imóvel
novo_imovel = pd.DataFrame({
    "area_m2": [75],
    "quartos": [2]
})

previsao = modelo.predict(novo_imovel)
print(f"Previsão de preço: R$ {previsao[0]:,.2f}")
```

### O que o modelo aprendeu internamente?

No Scikit-Learn, qualquer parâmetro aprendido pelo modelo após o `.fit()` termina com um sublinhado (`_`):

```python
# Coeficientes angulares (pesos de cada coluna de X):
print("Pesos (coeficientes):", modelo.coef_)
# Exemplo de saída: [3500., 0.] -> Cada m² acrescenta ~R$ 3.500

# Intercepto (ponto de partida quando X = 0):
print("Intercepto (b):", modelo.intercept_)

# A equação matemática aprendida é:
# preco = (coef_[0] * area_m2) + (coef_[1] * quartos) + intercept_
```

---

## 4. Divisão Treino e Teste (`train_test_split`)

Você **nunca** deve testar o modelo nos mesmos dados em que ele treinou. Isso seria como um aluno fazer uma prova com as exatas questões que decorou do gabarito.

```python
from sklearn.model_selection import train_test_split

# Dividindo 80% para treino e 20% para teste
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y,
    test_size=0.2,      # Proporção do teste (20%)
    random_state=42     # Garante reprodutibilidade (mesmo resultado sempre)
)

print("Tamanho do treino:", X_treino.shape[0], "linhas")
print("Tamanho do teste :", X_teste.shape[0], "linhas")
```

Treinando e avaliando a divisão:

```python
modelo = LinearRegression()
modelo.fit(X_treino, y_treino)

# Previsões nos dados que o modelo NUNCA viu:
previsoes_teste = modelo.predict(X_teste)
```

---

## 5. Pré-processamento: Tratando Dados Reais

Na vida real, seus dados contêm valores ausentes (`NaN`), textos/categorias e variáveis em escalas muito desproporcionais.

### A) Valores Ausentes: `SimpleImputer`

Substitui valores ausentes pela média, mediana ou moda:

```python
from sklearn.impute import SimpleImputer

imputador = SimpleImputer(strategy="median")

# Ajusta a mediana no treino e preenche os nulos
X_treino_imputado = imputador.fit_transform(X_treino)

# Apenas aplica a mesma mediana no teste (SEM chamar .fit!)
X_teste_imputado = imputador.transform(X_teste)
```

### B) Escala Numérica: `StandardScaler`

Transforma os dados para que tenham média 0 e desvio padrão 1 ($z = \frac{x - \mu}{\sigma}$). Essencial para algoritmos sensíveis à escala:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# Aprende média e desvio do treino e aplica
X_treino_escalado = scaler.fit_transform(X_treino)

# Aplica a mesma média e desvio no teste
X_teste_escalado = scaler.transform(X_teste)
```

### C) Variáveis Categóricas (Texto): `OneHotEncoder`

Computadores entendem apenas números. Para categorias (ex: `"bairro": ["Centro", "Sul"]`), criamos colunas binárias (0 ou 1):

```python
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
# handle_unknown='ignore': se aparecer uma nova categoria no teste, preenche com 0 em vez de travar!
```

---

## 6. O Maestro dos Dados: `ColumnTransformer`

Em vez de fazer cada transformação manualmente no Pandas, o `ColumnTransformer` aplica transformações específicas em cada grupo de colunas simultaneamente:

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

colunas_numericas = ["area_m2", "quartos"]
colunas_categoricas = ["bairro", "tipo_imovel"]

# 1. Pipeline para variáveis numéricas
pipeline_num = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# 2. Pipeline para variáveis categóricas
pipeline_cat = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

# 3. Agrupando tudo em um único pré-processador
preparador = ColumnTransformer(transformers=[
    ("num", pipeline_num, colunas_numericas),
    ("cat", pipeline_cat, colunas_categoricas)
])
```

---

## 7. Pipelines: O Segredo de um Código Limpo e Profissional

O `Pipeline` junta o pré-processamento e o algoritmo de machine learning (`LinearRegression`) em um **único objeto integrado**.

### Por que o Pipeline é obrigatório em projetos reais:

1. **Zero Vazamento de Dados (Data Leakage):** Cálculos de média e desvio só ocorrem dentro do conjunto de treino.
2. **Código Enxuto:** Uma única chamada de `.fit()` e uma única de `.predict()`.
3. **Pronto para Produção:** Ao salvar o pipeline, você exporta o pré-processamento e o modelo juntos.

```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

# Criando o Pipeline completo: Pré-processamento + Modelo
pipeline_modelo = Pipeline([
    ("preparador", preparador),
    ("regressor", LinearRegression())
])

# Treina TUDO (imputa, escala, codifica e ajusta a regressão)
pipeline_modelo.fit(X_treino, y_treino)

# Faz a previsão direta em dados brutos!
previsoes = pipeline_modelo.predict(X_teste)
```

---

## 8. Métricas de Avaliação para Regressão

Em problemas de regressão (previsão de números contínuos), **não existe acurácia** (o modelo nunca vai acertar o centavo exato). Usamos métricas de erro e ajuste:

```python
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

# 1. MAE (Erro Médio Absoluto)
# Significado: "Em média, o modelo erra em R$ X para mais ou para menos"
mae = mean_absolute_error(y_teste, previsoes)

# 2. RMSE (Raiz do Erro Quadrático Médio)
# Significado: Semelhante ao MAE, mas penaliza com muito mais peso os grandes erros
rmse = root_mean_squared_error(y_teste, previsoes)

# 3. R² (Coeficiente de Determinação)
# Significado: Varia geralmente de 0 a 1 (1.0 é a perfeição).
# 0.85 significa que o modelo explica 85% da variação dos preços.
r2 = r2_score(y_teste, previsoes)

print(f"MAE : R$ {mae:,.2f}")
print(f"RMSE: R$ {rmse:,.2f}")
print(f"R²  : {r2:.4f}")
```

---

## 9. Validação Cruzada (`cross_val_score`)

Avaliar o modelo em apenas uma divisão de treino/teste pode gerar conclusões enviesadas (por sorte ou azar daquela fatia de dados). A **Validação Cruzada** divide os dados em $K$ partes (geralmente 5) e testa em cada uma delas:

```python
from sklearn.model_selection import cross_val_score

# Avalia o pipeline em 5 fatias diferentes usando a métrica R²
scores = cross_val_score(pipeline_modelo, X_treino, y_treino, cv=5, scoring="r2")

print("Pontuações em cada dobra:", scores)
print(f"R² Médio: {scores.mean():.3f} (+/- {scores.std():.3f})")
```

---

## 10. Salvando e Carregando o Modelo (`joblib`)

Depois de treinar seu pipeline, você pode salvá-lo em disco para usá-lo em um backend, script ou dashboard:

### Salvar

```python
import joblib

joblib.dump(pipeline_modelo, "modelo_regressao_casas.joblib")
print("Modelo salvo com sucesso!")
```

### Carregar e Prever em Produção

```python
import joblib
import pandas as pd

# Carregando o pipeline completo
modelo_carregado = joblib.load("modelo_regressao_casas.joblib")

# Novos dados brutos vindos de um formulário ou API:
novo_cliente = pd.DataFrame([{
    "area_m2": 85,
    "quartos": 3,
    "bairro": "Centro",
    "tipo_imovel": "Apartamento"
}])

# O pipeline trata os nulos, escala, faz one-hot e prevê automaticamente!
preco_estimado = modelo_carregado.predict(novo_cliente)
print(f"Preço estimado: R$ {preco_estimado[0]:,.2f}")
```

---

## 11. Template Mestre Ponta a Ponta (Código 100% Executável)

Este exemplo utiliza o dataset de gorjetas (`tips`) do Seaborn. O objetivo é prever o valor da gorjeta (`tip`) com base no valor da conta (`total_bill`), tamanho da mesa (`size`), dia da semana, sexo, etc., usando **`LinearRegression`**.

Execute diretamente no seu terminal com `uv run python script.py`:

```python
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
import joblib

# 1. Carregando dados
df = sns.load_dataset("tips")

# 2. Separando Características (X) e Alvo (y)
colunas_num = ["total_bill", "size"]
colunas_cat = ["sex", "smoker", "day", "time"]

X = df[colunas_num + colunas_cat]
y = df["tip"]

# 3. Dividindo Treino e Teste
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Construindo o Pré-processador
pipeline_num = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

pipeline_cat = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preparador = ColumnTransformer(transformers=[
    ("num", pipeline_num, colunas_num),
    ("cat", pipeline_cat, colunas_cat)
])

# 5. Criando o Pipeline com LinearRegression
modelo = Pipeline([
    ("preparador", preparador),
    ("regressor", LinearRegression())
])

# 6. Treinando o modelo completo
modelo.fit(X_treino, y_treino)

# 7. Fazendo previsões e avaliando
y_pred = modelo.predict(X_teste)

print("=== Métricas de Desempenho (Teste) ===")
print(f"MAE  (Erro Médio)        : $ {mean_absolute_error(y_teste, y_pred):.2f}")
print(f"RMSE (Penaliza desvios)  : $ {root_mean_squared_error(y_teste, y_pred):.2f}")
print(f"R²   (Qualidade do ajuste): {r2_score(y_teste, y_pred):.4f}")

# 8. Salvando o Pipeline
joblib.dump(modelo, "pipeline_linear_regression.joblib")
print("\nModelo exportado com sucesso em 'pipeline_linear_regression.joblib'!")
```

---

## 12. Top 5 Armadilhas Comuns e Como Evitá-las

| #           | Erro Comum                                                     | Consequência                                                                                       | A Solução Certa                                                                                                   |
| ----------- | -------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| **1** | Chamar`.fit_transform()` nos dados de teste                  | **Vazamento de Dados (Data Leakage)**: O modelo "espia" a média/escala do conjunto de teste. | Use`.fit_transform()` **apenas no treino**. No teste, use apenas `.transform()`, ou utilize `Pipeline`. |
| **2** | Passar`df['coluna']` para $X$ em vez de `df[['coluna']]` | `ValueError: Expected 2D array, got 1D array instead`.                                            | $X$ sempre deve ter duas dimensões: use colchetes duplos `[['coluna']]`.                                       |
| **3** | Usar`OneHotEncoder` sem `handle_unknown='ignore'`          | O código quebra em produção caso apareça um valor de categoria não visto no treino.            | Configure sempre`OneHotEncoder(handle_unknown='ignore')`.                                                         |
| **4** | Tentar calcular`accuracy_score` em Regressão                | Gera erro ou métrica sem sentido, pois a acurácia é exclusiva para classificação.              | Para regressão, use sempre`mean_absolute_error`, `root_mean_squared_error` e `r2_score`.                     |
| **5** | Esquecer o`random_state` no `train_test_split`             | Seus resultados mudam a cada execução, impossibilitando comparar melhorias.                       | Sempre defina uma semente numérica fixa, ex:`random_state=42`.                                                   |

---

## 🎯 Resumo Rápido para Consulta

```python
# 1. Importações Básicas
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import root_mean_squared_error, r2_score
import joblib

# 2. Divisão
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Treinar e Prever
pipeline.fit(X_train, y_train)
previsoes = pipeline.predict(X_test)

# 4. Avaliar
print("R²:", r2_score(y_test, previsoes))
print("RMSE:", root_mean_squared_error(y_test, previsoes))

# 5. Salvar / Carregar
joblib.dump(pipeline, "modelo.joblib")
modelo_salvo = joblib.load("modelo.joblib")
```
