# Geração da Camada Sintética de Experimentação (Etapa 2)

**Projeto:** Datathon 7-MLET — Grupo 62
**Script:** [`src/features/generate_synthetic_data.py`](../src/features/generate_synthetic_data.py)
**Entrada:** `data/processed/games_metadata_with_embeddings.json` (499 jogos)
**Saída:** `data/synthetic_enrichment/`

> Esta camada é **sintética** e fica **fisicamente separada** da base factual do
> Kaggle. Ela cria o ambiente de experimentação (braços, contexto de usuário,
> impressões e recompensas atrasadas) sobre o qual o Multi-Armed Bandit será
> treinado e avaliado nas etapas seguintes.

## 1. Reprodutibilidade

- **Semente única:** `SEED = 42` (`np.random.seed`). Rodar o script de novo
  produz exatamente os mesmos arquivos.
- **Data de referência fixa:** `START_DATE = 2026-01-01`. Não usamos
  `datetime.now()` justamente para não quebrar a reprodutibilidade dos timestamps.

## 2. Arquivos gerados e schema

### 2.1. `offer_catalog.json` — os braços do Bandit

Cada braço é a combinação **jogo × incentivo**.

| Campo | Tipo | Descrição |
| :--- | :--- | :--- |
| `arm_id` | int | Identificador sequencial do braço. |
| `game_id`, `game_name` | int, str | Jogo de origem (catálogo processado). |
| `primary_genre` | str | Gênero (ontologia fechada). |
| `semantic_features` | list | Lista de `{feature_name, weight}` herdada do catálogo. |
| `incentive_id` | str | `FULL_PRICE`, `COUPON_10`, `COUPON_20` ou `VIP_PASS`. |
| `base_price` | float | Preço de tabela (do Kaggle). |
| `final_price` | float | `base_price × (1 − desconto)`. |
| `platforms` | list | **Sintético** — plataformas onde o jogo está disponível (ver §4). |

**Regra de negócio:** jogos **gratuitos** (`base_price = 0`) recebem **apenas**
`FULL_PRICE` — desconto sobre jogo grátis não faz sentido como braço. Total:
**1.870 braços** (457 jogos pagos × 4 incentivos + 42 jogos grátis × 1).

### 2.2. `user_profiles.json` — contexto dos usuários

| Campo | Tipo | Descrição |
| :--- | :--- | :--- |
| `user_id` | int | ID do usuário (1000–2999). |
| `preferred_platform` | str | Plataforma preferida (PC/Console/Mobile). |
| `favorite_genre` | str | Gênero favorito, sorteado **entre os gêneros que existem no catálogo**. |
| `price_sensitivity` | float | 0.1–1.0; acima de 0.7 = "caçador de ofertas". |
| `created_at` | str (ISO) | Data de criação da conta. |

**2.000 usuários.** Este arquivo é o **vetor de contexto** que o Bandit usará na decisão.

### 2.3. `offer_events.jsonl` — impressões (uma por linha)

| Campo | Tipo | Descrição |
| :--- | :--- | :--- |
| `event_id` | str | `evt_000000`... |
| `timestamp` | str (ISO) | Momento da impressão. |
| `user_id` | int | Usuário que recebeu a oferta. |
| `device_context` | str | Plataforma usada no momento do clique. |
| `arm_id`, `game_name`, `incentive_id` | — | Braço apresentado. |
| `clicked` | int (0/1) | Se houve clique (recompensa imediata). |

**50.000 impressões** ao longo de **90 dias** (horizonte temporal).

### 2.4. `delayed_rewards.jsonl` — recompensas atrasadas (uma por clique)

| Campo | Tipo | Descrição |
| :--- | :--- | :--- |
| `event_id` | str | Liga ao evento de impressão correspondente. |
| `reward_timestamp` | str (ISO) | Quando a recompensa "chegou" (5 min a 2 dias após o clique). |
| `user_id` | int | Usuário. |
| `bought` | int (0/1) | Se houve compra. |
| `gameplay_hours` | float | Horas jogadas (proxy de satisfação / risco de reembolso). |
| `accumulated_reward` | float | Recompensa consolidada do evento. |

## 3. Modelo de recompensa (ground-truth da simulação)

A propensão é um **ground-truth artificial**, usado só para gerar recompensas
plausíveis — **não** entra no modelo de decisão (o Bandit terá de aprendê-lo).

**Probabilidade de clique** (limitada a [0.05, 0.9]):
```
0.2 (base)
+ 0.4  se o gênero do braço == gênero favorito do usuário
+ afinidade de preço:
    caçador de ofertas (sens > 0.7): +0.5 se oferta barata (COUPON_20/VIP/grátis), senão −0.3
    usuário padrão: +0.1 se FULL_PRICE
+ 0.1  se device_context == plataforma preferida do usuário
```

**Funil de recompensa** (após o clique):
```
clique           → +0.1
  compra         → +0.4
    gameplay ≥ 2h → +0.5   (recompensa consolidada; passou do limite de reembolso)
    gameplay < 2h → −0.3   (risco de reembolso / abandono precoce)
```
As horas jogadas seguem uma distribuição exponencial com média 15h quando o
gênero agrada o usuário e 1.5h caso contrário.

## 4. Plataformas sintéticas — nota de transparência

A base do Kaggle é de jogos da Steam (essencialmente PC) e **não** contém a
dimensão PC/Console/Mobile usada pelo nosso marketplace cross-platform. Por isso,
a disponibilidade de plataforma de cada jogo é **gerada sinteticamente** (de forma
determinística, com a seed fixa): **todo jogo tem "PC"**, e "Console"/"Mobile"
entram com ~50%/~35% de chance. É um atributo de contexto da camada de
experimentação, **não um dado real do Steam**.

## 5. Hipóteses assumidas

- Afinidade de gênero é o principal motor de conversão e de retenção (horas jogadas).
- Usuários sensíveis a preço reagem a descontos agressivos; demais preferem o preço cheio.
- 2h de gameplay é o limiar de reembolso (regra padrão de lojas digitais).
- O contexto (gênero favorito, sensibilidade a preço, plataforma) é estável no horizonte simulado.

## 6. Limitações e riscos

- **Dados 100% simulados:** servem para validar o pipeline do Bandit, não
  refletem comportamento real de mercado.
- **Plataforma sintética** (ver §4): pode ser substituída por dado real no futuro.
- **Ground-truth conhecido:** como nós definimos a regra de recompensa, o risco de
  *reward hacking* é controlado aqui; em produção a recompensa real precisaria de
  guardrails (tratado nas etapas de governança).
- **Cobertura de gênero herda o catálogo** (concentração em RPG), o que enviesa a
  frequência de afinidade positiva.

## 7. Como reproduzir

```bash
cd src/features
python generate_synthetic_data.py
```
