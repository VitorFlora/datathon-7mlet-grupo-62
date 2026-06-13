# Rastreabilidade de Dados e Dicionário Semântico (Etapa 1)

Dataset de Referência: Steam Store Games (Kaggle Fact-Base)
Origem: Kaggle (Base histórica de metadados públicos da Steam)
Licença: CC0: Public Domain

1. Rastreabilidade da Fonte Fact-Base

Este projeto utiliza uma base factual de dados de jogos extraída publicamente para servir de âncora real aos perfis semânticos do nosso catálogo. O processamento primário lê os metadados brutos e consolida avaliações históricas para enriquecimento semântico.

Decisão Técnica de Mitigação de Vazamento de Dados (Data Leakage)

Durante o Sanity Check da base bruta, as seguintes variáveis foram explicitamente descartadas ou desconsideradas no pipeline de decisão:

global_rating_score: Notas acumuladas de avaliações gerais futuras do jogo. Se o modelo soubesse o sucesso futuro de um jogo ao decidir uma oferta no passado, haveria vazamento de informação.

total_downloads_2026 e future_revenue: Métricas acumuladas pós-fato que enviesariam o algoritmo de exploração.

review_text (pós-extração): O texto bruto é consumido estritamente de forma assíncrona/batch e descartado do ambiente de produção em tempo real para fins de LGPD e otimização de latência.

2. Dicionário de Dados

2.1. Base de Entrada Raw (steam_games_kaggle.csv)

Coluna

Tipo

Descrição

Tratamento / Decisão MLOps

id

Integer

ID único do jogo na Steam.

Preservado como âncora factual.

game_name

String

Nome comercial do jogo.

Preservado para exibição e logs auditáveis.

review_text

String

Texto consolidado das principais avaliações dos usuários.

Consumido no pipeline de NLP offline e dropado para produção.

price

Float

Preço original de tabela do jogo.

Preservado para cálculo de margem e incentivos.

platforms

String

Plataformas suportadas separadas por ; (ex: PC;Xbox).

Convertido em vetor categórico de contexto.

2.2. Catálogo Semântico Processado (games_metadata_with_embeddings.json)

O output consolidado pelo modelo de linguagem local (Llama 3.1) mapeia o consenso qualitativo dos jogos em representações numéricas e categóricas utilizáveis pelo Bandit.

Campo JSON

Tipo

Descrição

game_id

Integer

ID determinístico do jogo de origem.

game_name

String

Nome determinístico do jogo de origem.

primary_genre

String

Gênero conceitual sintetizado pelo LLM (ex: RPG, Sandbox).

semantic_profile

List[Dict]

Lista de atributos compostos por feature_name (Mecânica) e weight (Força, de 0.0 a 1.0).

3. Limitações Conhecidas

Representatividade de Nicho: Jogos muito novos ou extremamente obscuros com menos de 20 reviews na base bruta podem ter perfis semânticos imprecisos devido à escassez de dados para consenso.

Viés Linguístico: O LLM local foi instruído a normalizar as saídas semânticas em inglês e português, mas reviews traduzidas automaticamente podem apresentar perda sutil de nuance de apelo de jogabilidade.