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
13. [O Essencial de Feature Engineering (Além do Básico)](#13-o-essencial-de-feature-engineering-além-do-básico)

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

# X: 2D (Tabela com as características lexplicativas)
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

## 13. O Essencial de Feature Engineering (Além do Básico)

Depois de dominar o ciclo básico, você eventualmente encontrará cenários onde apenas imputar e escalar não é suficiente.

Em vez de se perder em dezenas de técnicas matemáticas complexas, existem **5 ferramentas essenciais** que resolvem 95% das necessidades práticas de projetos reais.

---

### A) Criando Features na Pipeline com `FunctionTransformer`

#### 1. O que é o `FunctionTransformer`? (O Modelo Mental)

Para que qualquer etapa possa entrar dentro de uma `Pipeline` do Scikit-Learn, ela **obrigatoriamente precisa ser um objeto que possui dois métodos**:

1. `.fit(X, y)`: Aprende parâmetros a partir dos dados de treino (ex: média e desvio no `StandardScaler`, mediana no `SimpleImputer`).
2. `.transform(X)`: Aplica a transformação nos dados.

Uma função comum em Python (ex: `def calcular_razoes(df): ...`) **não possui** esses métodos. Se você tentar passar uma função pura para o `Pipeline`, o Scikit-Learn lançará um erro do tipo:

> `TypeError: All intermediate steps should be transformers and implement fit and transform...`

O **`FunctionTransformer`** é o **adaptador (wrapper)** do Scikit-Learn. Ele "veste" a sua função Python comum com a casca de um Transformador do Scikit-Learn:

* No **`.fit(X)`**: Como funções customizadas (ex: divisões matemáticas) geralmente não precisam calcular médias nem desvios, o `FunctionTransformer` **não faz nada** (é *stateless* / sem estado). Ele apenas retorna a si mesmo sem aprender pesos.
* No **`.transform(X)`**: Ele simplesmente chama a sua função Python passando os dados atuais ($X$) como argumento e devolve o resultado para a próxima etapa da pipeline!

---

#### 2. O Problema Real: Por que NÃO criar colunas soltas no Pandas?

Imagine que no seu Jupyter Notebook você faça:

```python
# ❌ ERRO CONCEITUAL: Engenharia de Features "Solta" no Pandas
dados["comodos_por_casa"] = dados["total_comodos"] / dados["total_casas"]
dados["populacao_por_casa"] = dados["total_populacao"] / dados["total_casas"]
```

Isso parece inofensivo no notebook, mas **quebra a solução inteira em produção**. Veja o porquê:

1. **Quebra na Predição:** Quando você salvar seu modelo com `joblib` e colocar em uma API (ex: FastAPI ou Flask), a requisição que chega de um usuário ou aplicativo contém apenas os dados brutos da casa (`total_comodos: 1200`, `total_casas: 300`).
2. Se essas colunas derivadas não estiverem dentro da `Pipeline`, o modelo tentará rodar `.predict()` nos dados brutos, não encontrará a coluna `comodos_por_casa` e lançará um erro fatal.
3. Você seria forçado a duplicar o código de engenharia de features no seu backend da API, gerando **duplicação de código** e risco altíssimo de inconsistências (se mudar a fórmula no treino, o backend prevê com a fórmula antiga).

**A Solução Definitiva:** Fazer a própria `Pipeline` calcular as novas colunas no exato instante em que receber os dados brutos!

---

#### 3. As Duas Formas de Implantar o `FunctionTransformer`

Dependendo do tipo de cálculo que você quer fazer, existem duas formas corretas de posicionar o `FunctionTransformer`:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ FORMA 1: Como Primeiro Passo da Pipeline Mestra                             │
│ (Ideal para criar colunas relacionando 2 ou mais características)           │
│                                                                             │
│ [ DataFrame Bruto (X) ]                                                     │
│         │                                                                   │
│         ▼                                                                   │
│ ┌───────────────────────────────────────┐                                   │
│ │ 1. FunctionTransformer                │ <── Cria 'comodos_por_casa'       │
│ └─────────────────┬─────────────────────┘     usando comodos e casas        │
│                   │ [ DataFrame com colunas originais + derivadas ]         │
│                   ▼                                                         │
│ ┌───────────────────────────────────────┐                                   │
│ │ 2. ColumnTransformer                  │ <── Imputa e escala todas as      │
│ └─────────────────┬─────────────────────┘     colunas (originais e novas!)  │
│                   │ [ Matriz Numérica Pronta ]                              │
│                   ▼                                                         │
│ ┌───────────────────────────────────────┐                                   │
│ │ 3. Modelo Estimador                   │ <── Treina (.fit) ou Prevê        │
│ └───────────────────────────────────────┘                                   │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ FORMA 2: Diretamente Dentro do ColumnTransformer                            │
│ (Ideal para transformações matemáticas coluna a coluna, como Log ou Raiz)   │
│                                                                             │
│ ColumnTransformer([                                                         │
│     ("log_renda", FunctionTransformer(np.log1p), ["media_renda"]),          │
│     ("scaler_num", StandardScaler(), ["idade_mediana_casas"]),              │
│     ...                                                                     │
│ ])                                                                          │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

#### 4. Passo a Passo Completo: Implementando a Forma 1 (Pipeline Mestra)

Este é o padrão de ouro para projetos reais onde criamos taxas e razões entre variáveis.

##### Passo 1: Escrever a Função Pura em Python

A função precisa obedecer a **três regras de ouro**:

1. **Sempre fazer cópia:** Use `.copy()` no DataFrame de entrada para evitar modificar os dados originais fora da pipeline por referência.
2. **Evitar divisão por zero:** Adicione uma constante microscópica no denominador (ex: `+ 1e-6`) ou trate com condições.
3. **Retornar o DataFrame expandido:** Mantenha as colunas originais e adicione as novas.

```python
import pandas

def criar_caracteristicas_derivadas(dados: pandas.DataFrame) -> pandas.DataFrame:
    """
    Recebe um DataFrame bruto e cria novas razões de negócio.
    Retorna o DataFrame enriquecido com as novas colunas.
    """
    dataframe_expandido = dados.copy()
  
    # 1. Média de cômodos por domicílio:
    dataframe_expandido["comodos_por_casa"] = (
        dataframe_expandido["total_comodos"] / (dataframe_expandido["total_casas"] + 1e-6)
    )
  
    # 2. Média de moradores por domicílio:
    dataframe_expandido["populacao_por_casa"] = (
        dataframe_expandido["total_populacao"] / (dataframe_expandido["total_casas"] + 1e-6)
    )
  
    # 3. Proporção de quartos em relação ao total de cômodos:
    dataframe_expandido["proporcao_quartos"] = (
        dataframe_expandido["total_quartos"] / (dataframe_expandido["total_comodos"] + 1e-6)
    )
  
    return dataframe_expandido
```

##### Passo 2: Empacotar com `FunctionTransformer`

```python
from sklearn.preprocessing import FunctionTransformer

transformador_engenharia = FunctionTransformer(criar_caracteristicas_derivadas)
```

##### Passo 3: Registrar as Novas Colunas no `ColumnTransformer`

> ⚠️ **A Pegadinha Mais Comum:** O `ColumnTransformer` só processa as colunas que você disser explicitamente para ele processar. Portanto, as **novas colunas criadas no passo 1 devem constar na lista de colunas numéricas** do `ColumnTransformer`!

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Lista contendo tanto as colunas originais quanto as 3 novas que serão criadas pelo FunctionTransformer:
colunas_numericas = [
    "longitude",
    "latitude",
    "idade_mediana_casas",
    "total_comodos",
    "total_quartos",
    "total_populacao",
    "total_casas",
    "media_renda",
    "comodos_por_casa",     # <-- Coluna derivada
    "populacao_por_casa",   # <-- Coluna derivada
    "proporcao_quartos"     # <-- Coluna derivada
]

colunas_categoricas = ["proximidade_oceano"]

pipeline_numerica = Pipeline(steps=[
    ("imputador", SimpleImputer(strategy="median")),
    ("escalador", StandardScaler())
])

pipeline_categorica = Pipeline(steps=[
    ("imputador", SimpleImputer(strategy="most_frequent")),
    ("codificador", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preparador_colunas = ColumnTransformer(transformers=[
    ("numerico", pipeline_numerica, colunas_numericas),
    ("categorico", pipeline_categorica, colunas_categoricas)
])
```

##### Passo 4: Conectar Tudo na Pipeline Mestra

Agora organizamos a sequência na `Pipeline`:

1. `engenharia`: Aplica o `FunctionTransformer` no DataFrame bruto e gera o DataFrame com as 3 novas colunas.
2. `preparador`: Recebe o DataFrame expandido, imputa nulos, padroniza as numéricas e codifica as categóricas.
3. `modelo`: Recebe a matriz final pronta e ajusta a reta ou floresta.

```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

pipeline_completa = Pipeline(steps=[
    ("engenharia", transformador_engenharia),
    ("preparador", preparador_colunas),
    ("modelo", LinearRegression())
])
```

##### Passo 5: Treinar e Testar a Mágica em Produção

```python
# Treinando tudo com um único comando:
pipeline_completa.fit(X_train, y_train)

# Simulando um dado NOVO que acabou de chegar (dados brutos, sem as colunas derivadas!):
novo_imovel_bruto = pandas.DataFrame([{
    "longitude": -122.23,
    "latitude": 37.88,
    "idade_mediana_casas": 41.0,
    "total_comodos": 880.0,
    "total_quartos": 129.0,
    "total_populacao": 322.0,
    "total_casas": 126.0,
    "media_renda": 8.3252,
    "proximidade_oceano": "NEAR BAY"
}])

# A Pipeline calcula automaticamente as razões, imputa, escala e devolve o preço:
preco_predito = pipeline_completa.predict(novo_imovel_bruto)
print(f"Preço Previsto: $ {preco_predito[0]:,.2f}")
```

---

#### 5. Como Testar e Inspecionar o `FunctionTransformer` Isoladamente

Se você estiver em dúvida se a sua função funcionou antes de rodar a pipeline inteira, você pode testá-la individualmente chamando `.transform()` diretamente nela:

```python
# Teste isolado:
dados_transformados = transformador_engenharia.transform(X_train.head(3))

# Verificando se as novas colunas realmente nasceram:
print(dados_transformados[["comodos_por_casa", "populacao_por_casa", "proporcao_quartos"]])
```

Se as novas colunas aparecerem com valores calculados corretamente, sua função está 100% pronta para rodar dentro da pipeline!

---

### B) Proteção Anti-Crash com `handle_unknown='ignore'` no `OneHotEncoder`

#### O Problema:

Se a sua base de treino contém apenas as cidades `"São Paulo"`, `"Campinas"` e `"Santos"`, o que acontece se no conjunto de teste ou em produção entrar uma casa com `"Curitiba"` ou com um erro de digitação como `"Sao Paulo"`?

Sem configuração, o Scikit-Learn lança o erro fatal:

> `ValueError: Found unknown categories ['Curitiba'] in column 0 during transform`

#### A Solução:

Ao configurar **`handle_unknown="ignore"`**, o `OneHotEncoder` simplesmente preenche todas as colunas binárias daquela categoria com **`0`**, permitindo que o modelo continue funcionando sem travar o sistema:

```python
from sklearn.preprocessing import OneHotEncoder

# Proteção essencial para qualquer ambiente real:
encoder_blindado = OneHotEncoder(
    handle_unknown="ignore",  # Ignora categorias inéditas com zeros
    sparse_output=False
)
```

> 💡 **Dica Moderna (Agrupando Categorias Raras):** Se você tiver uma coluna com dezenas de categorias, adicione `min_frequency=0.02`. Qualquer categoria que aparecer em menos de 2% das linhas será automaticamente agrupada em uma categoria `"infrequent"`, evitando sobreajuste (*overfitting*):
>
> ```python
> encoder_otimizado = OneHotEncoder(min_frequency=0.02, handle_unknown="infrequent_if_exist", sparse_output=False)
> ```

---

### C) Decomposição de Datas em Pipeline

Modelos de Machine Learning não aceitam strings de data (ex: `"2026-09-14"`). Uma data pura precisa ser decomposta em números informativos:

```python
import pandas as pd
from sklearn.preprocessing import FunctionTransformer

def decompor_datas(dados: pd.DataFrame) -> pd.DataFrame:
    df_novo = dados.copy()
    # Converte para datetime caso ainda seja string
    datas = pd.to_datetime(df_novo["data_venda"])
  
    # Extrai partes numéricas fundamentais:
    df_novo["ano_venda"] = datas.dt.year
    df_novo["mes_venda"] = datas.dt.month
    df_novo["dia_da_semana"] = datas.dt.dayofweek  # 0 = Segunda-feira, 6 = Domingo
    df_novo["fim_de_semana"] = (datas.dt.dayofweek >= 5).astype(int)
  
    # Remove a coluna original de data (já que texto/timestamp não pode ir para o modelo)
    return df_novo.drop(columns=["data_venda"])

transformador_datas = FunctionTransformer(decompor_datas)
```

Você pode plugar esse `transformador_datas` na sua pipeline exatamente da mesma forma que o `transformador_features` do item A.

---

### D) Proteção Contra Outliers Extremos com `RobustScaler`

#### Quando usar?

* O `StandardScaler` usa a **média** e o **desvio padrão**. Se houver um bilionário ou um valor digitado errado (ex: R$ 999.999.999), a média e o desvio explodem, comprimindo todos os outros dados normais em um intervalo minúsculo.
* O **`RobustScaler`** usa a **Mediana** e o **Intervalo Interquartil (IQR: Percentil 75 - Percentil 25)**. Como a mediana não se deixa abalar por valores extremos, a escala fica perfeita para o restante dos dados.

#### Como aplicar:

Basta substituir o `StandardScaler()` pelo `RobustScaler()` dentro da pipeline numérica:

```python
from sklearn.preprocessing import RobustScaler

pipeline_num_robusta = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", RobustScaler())  # <-- Troca direta de 1 linha!
])
```

---

### E) Classes Desbalanceadas em Classificação (`class_weight='balanced'`)

#### O Problema da "Acurácia Mentirosa":

Em problemas reais como detecção de fraudes (99% compras legítimas, 1% fraude) ou cancelamento de clientes (*churn*):

* Se o modelo chutar "Não é fraude" para 100% dos clientes, ele terá **99% de acurácia**, mas o modelo é completamente inútil para o negócio.

#### A Solução:

Adicione o parâmetro **`class_weight="balanced"`** no estimador. O algoritmo aumentará matematicamente a penalidade caso ele erre a classe rara:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# O algoritmo dá pesos proporcionais à raridade de cada classe:
modelo_classificacao = LogisticRegression(class_weight="balanced")
# ou:
modelo_floresta = RandomForestClassifier(class_weight="balanced", random_state=42)
```

> ⚠️ **Lembre-se:** Em problemas desbalanceados, nunca avalie por acurácia! Use o `classification_report(y_true, y_pred)` e observe a **Precisão** e o **Recall** da classe rara.

---

### F) O "Atalho das Árvores": `HistGradientBoosting`

Se a sua regressão linear não estiver alcançando um bom $R^2$ e você não quiser perder horas tentando transformações matemáticas complexas (logaritmos, potências, etc.), o Scikit-Learn possui um algoritmo moderno baseado em árvores que resolve a maioria desses problemas sozinho:

```python
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier

# Para prever números (Preços, Vendas, Demandas):
modelo_arvores = HistGradientBoostingRegressor(random_state=42)

# Para prever categorias (Sim/Não, Fraude, Churn):
modelo_classif_arvores = HistGradientBoostingClassifier(random_state=42)
```

#### Por que ele é o "Atalho"?

1. **Imune a Outliers:** Árvores cortam os dados por limiares (ex: `Area > 100`), então um número absurdo de 1 bilhão não distorce as outras divisões.
2. **Dispensam Scaling:** Árvores não calculam distâncias nem produtos escalares; o `StandardScaler` ou `MinMaxScaler` são completamente opcionais.
3. **Nativamente não-linear:** Aprende curvas, interações e degraus automaticamente.
4. **Trata nulos (`NaN`):** O `HistGradientBoosting` consegue inclusive lidar com valores faltantes nativamente sem quebrar!

---

### 🌟 Exemplo Integrado Completo (Código 100% Executável)

Aqui está um exemplo que conecta **todas** as peças essenciais em uma única Pipeline pronta para produção:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder, FunctionTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# 1. Base com dados brutos (incluindo nulos, categorias e texto)
df = pd.DataFrame({
    "Area": [48.0, 60.0, 73.0, 52.0, 85.0, None, 92.0, 45.0, 110.0, 68.0],
    "Quartos": [3.0, 2.0, 4.0, 2.0, 3.0, 2.0, 3.0, 1.0, 4.0, 2.0],
    "Banheiros": [1.0, 2.0, 3.0, 1.0, 2.0, 1.0, None, 1.0, 3.0, 2.0],
    "Tem_Garagem": ["Nao", "Sim", "Sim", "Nao", "Sim", "Nao", "Sim", "Nao", "Sim", "Nao"],
    "Valor": [240000, 320000, 460000, 260000, 410000, 270000, 450000, 210000, 580000, 330000]
})

X = df.drop(columns=["Valor"])
y = df["Valor"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Engenharia de Features (Razões de Negócio)
def adicionar_features(dados: pd.DataFrame) -> pd.DataFrame:
    d = dados.copy()
    d["area_por_quarto"] = d["Area"] / (d["Quartos"] + 1e-6)
    d["banheiro_por_quarto"] = d["Banheiros"] / (d["Quartos"] + 1e-6)
    return d

transformador_features = FunctionTransformer(adicionar_features)

# 3. Pré-processamento com ColumnTransformer
colunas_num = ["Area", "Quartos", "Banheiros", "area_por_quarto", "banheiro_por_quarto"]
colunas_cat = ["Tem_Garagem"]

preparador = ColumnTransformer(transformers=[
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()) # Imune a outliers
    ]), colunas_num),
  
    ("cat", Pipeline([
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)) # Imune a novas categorias
    ]), colunas_cat)
])

# 4. Pipeline Mestra (Engenharia + Pré-processamento + Modelo Potente)
pipeline_producao = Pipeline(steps=[
    ("engenharia", transformador_features),
    ("preparador", preparador),
    ("modelo", HistGradientBoostingRegressor(random_state=42))
])

# 5. Treinamento e Avaliação
pipeline_producao.fit(X_train, y_train)
previsoes = pipeline_producao.predict(X_test)

print(f"MAE : R$ {mean_absolute_error(y_test, previsoes):,.2f}")
print(f"R²  : {r2_score(y_test, previsoes):.4f}")

# 6. Salvar em Disco para Produção
joblib.dump(pipeline_producao, "pipeline_completa.joblib")
print("Pipeline salva com sucesso!")
```

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
