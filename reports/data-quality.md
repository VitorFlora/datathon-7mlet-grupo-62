# Relatório de Qualidade de Dados (Etapa 1)

**Projeto:** Datathon 7-MLET — Plataforma de Experimentação Adaptativa (Grupo 62)
**Data da análise:** 2026-06-14
**Artefato analisado:** `data/processed/games_metadata_with_embeddings.json`
**Modelo de extração:** Llama 3.1 8B (local, via Ollama) · `temperature=0.0` · `num_ctx=16384`

> Este relatório documenta a qualidade tanto da base factual do Kaggle quanto do
> catálogo semântico gerado pelo pipeline offline. Ele serve de evidência para a
> Etapa 1 (rastreabilidade, EDA e qualidade) e baliza o uso dos dados nas etapas
> seguintes (enriquecimento sintético e Multi-Armed Bandit).

---

## 1. Fontes e Rastreabilidade

Detalhes completos (links, versões, licenças, dicionário) em
[`data/kaggle/README.md`](../data/kaggle/README.md).

| Base | Origem | Volume | Licença |
| :--- | :--- | :--- | :--- |
| Metadados de jogos | Kaggle — artermiloff/steam-games-dataset (`games_march2025_cleaned.csv`) | ~89.618 jogos × 47 colunas | MIT |
| Avaliações | Kaggle — andrewmvd/steam-reviews | ~6.417.106 reviews × 5 colunas | CC BY-NC 4.0 |

A base de avaliações cobre **9.972 jogos distintos** (app_id únicos), enquanto a
base de metadados cobre os ~89,6 mil. A junção é feita por `app_id` via
*inner join* (ver §3).

## 2. Tratamento de Vazamento de Dados (Data Leakage)

O Bandit decide a oferta **antes** de observar o resultado, então colunas
pós-fato não entram na decisão. Foram descartadas / não utilizadas:

- **Reviews:** `review_score`, `review_votes` (refletem o desfecho da avaliação).
- **Metadados:** métricas de sucesso acumulado (`positive`, `negative`,
  `recommendations`, `metacritic_score`, `num_reviews_total`, `estimated_owners`,
  `peak_ccu`, `average_playtime_*`, etc.) — sequer são lidas (`usecols`).
- **Texto de review:** usado apenas no processamento offline e descartado da
  camada de produção (LGPD + latência).

## 3. Pipeline de Geração e Decisões

1. **Leitura enxuta:** apenas as colunas necessárias são carregadas (`usecols`),
   e bytes inválidos no CSV de reviews são tolerados (`encoding_errors='replace'`).
2. **Coalesce:** o texto de review é unificado entre as duas fontes
   (`combine_first`), priorizando a base de avaliações.
3. **Inner join por `app_id`:** prioriza **integridade sobre volume** — jogos sem
   correspondência nas duas bases ficam de fora.
4. **Extração semântica com saída estruturada:** o schema Pydantic
   (`GameSemantics`) é imposto ao modelo, garantindo gênero válido (ontologia
   fechada de 17 valores) e de 3 a 5 features com peso em [0, 1].

## 4. Qualidade do Catálogo Processado

Catálogo com **499 jogos** (de um alvo de 500; ver §5).

> **Reprodutível por linha de comando.** Todas as métricas desta seção são
> geradas pelo script [`src/features/validate_catalog.py`](../src/features/validate_catalog.py):
> ```bash
> python src/features/validate_catalog.py
> ```
> O script lê apenas o JSON processado, então serve também como comando rápido de validação do repositório.

### 4.1. Completude e Integridade — ✅

| Verificação | Resultado |
| :--- | :--- |
| Jogos com todos os campos esperados | 499 / 499 (100%) |
| Perfis semânticos vazios | **0** |
| `game_id` duplicados | **0** |
| `release_date` ausente/`Unknown` | **0** |
| Pesos fora do intervalo [0, 1] | **0** |

### 4.2. Distribuição de Features por Jogo

| Nº de features | Jogos |
| :--- | :--- |
| 5 | 459 |
| 4 | 14 |
| 3 | 26 |

Total de **2.429 features** extraídas. A faixa de 3–5 (em vez de exigir
exatamente 5) evita que jogos com reviews pobres sejam forçados a "inventar"
tags adicionais.

### 4.3. Distribuição de Gêneros (ontologia fechada)

| Gênero | Jogos | | Gênero | Jogos |
| :--- | :--- | :--- | :--- | :--- |
| RPG | 163 | | RTS | 19 |
| Indie | 66 | | Strategy | 12 |
| Simulation | 51 | | MOBA | 3 |
| Puzzle | 42 | | Horror | 2 |
| Action | 41 | | Hack and slash | 2 |
| Adventure | 39 | | | |
| FPS | 35 | | | |
| Racing | 24 | | | |

Todos os gêneros pertencem à ontologia fechada (sem alucinação). Há concentração
em RPG (33%), esperada pela ordem de leitura dos jogos; gêneros como Sports,
Exploration, Fantasy e Arcade não apareceram nesta amostra inicial.

## 5. Problemas Conhecidos e Severidade

| # | Achado | Severidade | Status / Mitigação |
| :--- | :--- | :--- | :--- |
| 1 | **Regras de texto das tags não são plenamente respeitadas pela LLM**: ~24% das features (579/2.429) contêm palavra "banida" (ex.: "Gameplay", "Graphics"); ~12% têm mais de 3 palavras; ~10% contêm underscore. | Média | **Limitação conhecida do modelo local (Llama 3.1 8B).** A saída estruturada garante o *schema* (tipos, gênero, faixa de peso), mas não regras de texto livre. Um ambiente de produção com mais poder computacional usaria um modelo mais forte, com maior aderência a instruções. |
| 2 | **1 jogo descartado** ("Build-A-Lot 4"): a LLM retornou peso negativo (−0.6) nas 2 tentativas. | Baixa | **Mitigado**: adicionado um *clamp* que ajusta pesos para [0, 1] em vez de descartar o jogo (vale para futuras execuções). |
| 3 | **42 jogos com `base_price = 0`** (free-to-play). | Baixa | Não é erro (preço real). Atenção na Etapa 2: aplicar cupom de desconto sobre jogo grátis não faz sentido como braço — tratar na geração do catálogo de ofertas. |
| 4 | **Cobertura de gênero desigual** (concentração em RPG; gêneros raros com 2–3 jogos). | Baixa | Esperado na amostra inicial; aumentar `max_games` e diversificar a seleção de jogos amplia a cobertura. |

## 6. Conclusão

O catálogo processado está **estruturalmente íntegro** (100% dos campos, sem
perfis vazios, sem duplicatas, pesos válidos) e **rastreável** até as bases
factuais do Kaggle, com tratamento explícito de vazamento. A principal
oportunidade de melhoria é a **qualidade textual das tags** (achado #1), que será
endereçada antes da modelagem final.

Os dados estão **aptos a avançar** para a Etapa 2 (enriquecimento sintético),
observando os cuidados dos achados #1 e #3.

---

### Como reproduzir esta análise

- **Métricas do catálogo (§4):** `python src/features/validate_catalog.py`
  (lê `data/processed/games_metadata_with_embeddings.json`).
- **Geração do catálogo:** [`src/features/extract_semantics.py`](../src/features/extract_semantics.py).
- **Análise exploratória da base bruta:** [`notebooks/01_eda_and_quality.ipynb`](../notebooks/01_eda_and_quality.ipynb).
