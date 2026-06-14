# Rastreabilidade de Dados e Dicionário (Etapa 1)

Este projeto usa **duas bases públicas do Kaggle** como camada factual: uma com
os **metadados dos jogos** (preço, data, gêneros) e outra com o **texto das
avaliações** dos usuários. As duas são unidas pelo `app_id`/`appid` no pipeline
de extração semântica.

> Os arquivos `.csv` brutos **não são versionados** no Git (são grandes). Baixe-os
> pelos links abaixo e coloque-os nesta pasta (`data/kaggle/`).

---

## 1. Fontes

### 1.1. Metadados dos jogos — `kaggle_steam_game.csv`
- **Dataset:** Steam Games Dataset (artermiloff)
- **Link:** https://www.kaggle.com/datasets/artermiloff/steam-games-dataset
- **Arquivo usado:** `games_march2025_cleaned.csv` (renomeado para `kaggle_steam_game.csv`)
- **Volume:** ~89.6 mil jogos, 47 colunas
- **Licença:** MIT
- **Versão / referência temporal:** dados coletados em março/2025

### 1.2. Avaliações dos usuários — `kaggle_steam_reviews.csv`
- **Dataset:** Steam Reviews (andrewmvd)
- **Link:** https://www.kaggle.com/datasets/andrewmvd/steam-reviews
- **Arquivo usado:** renomeado para `kaggle_steam_reviews.csv`
- **Volume:** ~6.4 milhões de avaliações, 5 colunas
- **Licença:** Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)

### Instruções de download
1. Faça login no Kaggle e acesse os dois links acima.
2. Baixe `games_march2025_cleaned.csv` e o CSV de reviews.
3. Renomeie e coloque os arquivos nesta pasta:
   - `data/kaggle/kaggle_steam_game.csv`
   - `data/kaggle/kaggle_steam_reviews.csv`

---

## 2. Dicionário de Dados

### 2.1. `kaggle_steam_reviews.csv` (avaliações)

| Coluna | Tipo | Descrição | Uso no pipeline |
| :--- | :--- | :--- | :--- |
| `app_id` | int | ID do jogo na Steam (chave de junção). | Chave do `merge` com os metadados. |
| `app_name` | str | Nome do jogo. | Exibição e logs. |
| `review_text` | str | Texto livre da avaliação do usuário. | **Entrada do pipeline de NLP** (consumido em batch, depois descartado). |
| `review_score` | int | Indica se a avaliação é positiva/negativa. | **Descartado** (vazamento — ver §3). |
| `review_votes` | int | Quantos usuários acharam a avaliação útil. | **Descartado** (vazamento — ver §3). |

### 2.2. `kaggle_steam_game.csv` (metadados dos jogos)

Esta base tem 47 colunas. O pipeline usa apenas um subconjunto pequeno; as demais
são descritivas ou métricas pós-fato (não usadas na decisão).

**Colunas efetivamente usadas:**

| Coluna | Tipo | Descrição | Uso no pipeline |
| :--- | :--- | :--- | :--- |
| `appid` | int | ID do jogo na Steam. | Renomeada para `app_id`; chave de junção. |
| `name` | str | Nome comercial do jogo. | Renomeada para `app_name`; exibição. |
| `release_date` | str | Data de lançamento. | Preservada como metadado do jogo. |
| `price` | float | Preço de tabela. | Base para cálculo de incentivos/descontos. |
| `reviews` | str | Trecho de review/curadoria da loja. | Usada como fallback de texto via *coalesce* com `review_text`. |

**Colunas não usadas (exemplos):** `required_age`, `dlc_count`,
`detailed_description`, `about_the_game`, `short_description`, `header_image`,
`website`, `windows`/`mac`/`linux`, `metacritic_score`, `achievements`,
`recommendations`, `supported_languages`, `developers`, `publishers`,
`categories`, `genres`, `tags`, `positive`, `negative`, `estimated_owners`,
`average_playtime_forever`, `peak_ccu`, `num_reviews_total`, `pct_pos_total`,
entre outras.

### 2.3. Saída — `data/processed/games_metadata_with_embeddings.json`

Catálogo semântico gerado pela LLM local (Llama 3.1 via Ollama).

| Campo | Tipo | Descrição |
| :--- | :--- | :--- |
| `game_id` | int | ID do jogo de origem. |
| `game_name` | str | Nome do jogo. |
| `base_price` | float | Preço de tabela vindo do Kaggle. |
| `release_date` | str | Data de lançamento. |
| `primary_genre` | str | Gênero principal inferido pela LLM (ontologia fechada). |
| `semantic_profile` | List[{feature_name, weight}] | Características marcantes do gameplay e seus pesos (0.0–1.0). |

---

## 3. Decisão sobre Vazamento de Dados (Data Leakage)

O modelo de decisão (Multi-Armed Bandit) escolhe uma oferta **antes** de saber o
resultado. Portanto, qualquer coluna que reflita o **resultado/sucesso futuro**
do jogo não pode entrar na decisão, sob pena de vazamento.

**Colunas descartadas / não usadas por vazamento ou risco de viés:**

- `review_score` e `review_votes` (base de reviews): refletem o desfecho da
  avaliação — **descartadas explicitamente** no `merge` do pipeline.
- Métricas de popularidade/sucesso acumulado da base de jogos: `positive`,
  `negative`, `recommendations`, `user_score`, `metacritic_score`,
  `num_reviews_total`, `pct_pos_total`, `estimated_owners`, `peak_ccu`,
  `average_playtime_forever` — são **pós-fato** e não são usadas no contexto de
  decisão.
- `review_text`: usado **apenas offline/batch** na extração semântica e depois
  descartado do ambiente de produção (LGPD + redução de latência).

---

## 4. Limitações Conhecidas

- **Reviews escassas:** jogos com poucas avaliações podem gerar perfis semânticos
  simplistas ou incompletos.
- **Viés linguístico:** a base de reviews é majoritariamente em inglês; nuances de
  reviews traduzidas podem se perder.
- **Cobertura da junção:** usamos *inner join* por `app_id`, então jogos sem
  correspondência nas duas bases ficam de fora (priorizamos integridade sobre
  volume).
- **Restrição de licença (uso não comercial):** a base de reviews é
  **CC BY-NC 4.0**, ou seja, permite uso **não comercial** com atribuição. O uso
  neste projeto é estritamente **acadêmico** (Datathon/FIAP). Qualquer aplicação
  comercial real exigiria outra fonte de dados ou licenciamento adequado.
