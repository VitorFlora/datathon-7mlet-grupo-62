# Datathon 7-MLET — Plataforma de Experimentação Adaptativa para Marketplace de Jogos

**Fase:** 05 — Deploy Avançado de IA Generativa
**Grupo:** Grupo 62 · **Turma:** MLET7
**Caso de Negócio:** Marketplace de Jogos e Assinaturas Cross-Platform

## 1. Visão Geral do Problema

No cenário competitivo do entretenimento digital, marketplaces de jogos têm
dificuldade de engajar usuários de forma personalizada. **Regras estáticas de
recomendação e testes A/B longos são ineficientes**: demoram a reagir a mudanças
de comportamento, desperdiçam tráfego e ignoram o contexto imediato do usuário.

Para resolver isso, o projeto implementa uma **Plataforma de Experimentação
Adaptativa baseada em Contextual Multi-Armed Bandits (MAB)**. O sistema decide,
em tempo real, **qual oferta/incentivo** apresentar a cada jogador, equilibrando
exploração e explotação e aprendendo com as respostas observadas.

## 2. Aderência ao Desafio (Domínio)

O desafio oficial é enquadrado em uma **instituição financeira digital** (decidir
qual oferta, mensagem ou próximo passo apresentar a cada cliente elegível). O
edital permite o uso de **outra base, desde que o grupo justifique a aderência**.

Optamos por aplicar a **mesma estrutura de problema** a um **marketplace de
jogos**, mantendo o paralelo conceitual:

| Conceito do edital (financeiro) | Equivalente neste projeto (jogos) |
| :--- | :--- |
| Cliente elegível | Jogador da plataforma |
| Oferta / mensagem / próximo passo | Incentivo comercial (preço cheio, cupom 10%/20%, assinatura VIP) |
| Conversão | Clique → compra → tempo de jogo |
| Suitability (adequação) | Adequação da oferta ao perfil e ao risco de reembolso |
| Política comercial regulada | Regras de desconto e de assinatura |
| Recompensa atrasada | Horas jogadas após a compra (proxy de satisfação/reembolso) |

A natureza do problema (decisão adaptativa de oferta em canal digital, com
exploração vs. explotação, recompensas atrasadas e governança) é idêntica; muda
apenas o vertical de negócio.

## 3. Escopo da Solução (3 pipelines de MLOps)

1. **Feature Pipeline (offline / batch):** consome dados factuais do Kaggle
   (Steam), unifica fontes via *coalesce* e usa uma LLM local
   (**Llama 3.1 8B via Ollama**) para gerar perfis semânticos padronizados,
   mapeando gêneros com ontologia estrita (Pydantic).
2. **Inference Pipeline (online / real-time):** API que recebe o contexto do
   usuário e aciona o MAB (*Thompson Sampling*, *UCB/LinUCB*) para decidir a
   oferta em milissegundos. *(planejado — Etapas 3 e 5)*
3. **Delayed Feedback Loop (assíncrono):** captura a recompensa atrasada com base
   no comportamento pós-conversão (horas jogadas). *(simulado — Etapa 2)*

## 4. Decisões de Design

- **LLM local (Ollama + Llama 3.1 8B):** custo zero de inferência (FinOps) e
  conformidade com a LGPD (sem trafegar dados brutos para APIs de terceiros).
- **Ontologia de gêneros fechada (Pydantic `Literal`):** bloqueia alucinação do
  LLM e reduz a dimensionalidade para o MAB.
- **Recompensa atrasada baseada em reembolso:** regra padrão das lojas digitais
  (até 2h de gameplay como limite de reembolso), mitigando *reward hacking*.
- **Anti-vazamento de dados:** colunas pós-fato (notas futuras, downloads
  acumulados) são descartadas do pipeline de decisão.

### Créditos e reuso open-source

Este repositório herda e adapta a lógica de NLP e extração semântica baseada em
conteúdo do projeto open-source **NextSteamGame**, estendendo-a de recomendação
estática para um sistema adaptativo de precificação e ofertas.

## 5. Pré-requisitos

- Python 3.10 ou superior (recomendado: Python 3.14 via Miniconda)
- [Ollama](https://ollama.com) instalado no sistema
- NVIDIA GPU com CUDA (opcional, acelera a LLM local)

## 6. Instalação e Execução Local

```bash
# 1. Crie e ative o ambiente (Conda)
conda create -n datathon-7mlet python=3.14 -y
conda activate datathon-7mlet

# 2. Instale o projeto e suas dependências (a partir da raiz do repositório)
pip install .

# 3. Baixe o modelo da LLM local
ollama pull llama3.1

# 4. Coloque as bases do Kaggle em data/kaggle/ (ver data/kaggle/README.md)

# 5. Execute o pipeline de extração semântica
#    (rode a partir da pasta src/features/ por causa dos caminhos relativos)
cd src/features
python extract_semantics.py
```

Para a Análise Exploratória (EDA), abra o notebook
`notebooks/01_eda_and_quality.ipynb` no VSCode ou no Jupyter.

## 7. Mapa de Pastas

```
datathon-7mlet-grupo-62/
├── .env.example                 # Template de variáveis de ambiente (sem segredos)
├── .gitignore
├── LICENSE                      # Licença MIT
├── pyproject.toml               # Dependências, versão de Python e pontos de entrada
├── README.md
├── data/
│   ├── kaggle/                  # Base factual do Kaggle (CSVs não versionados)
│   │   └── README.md            # Rastreabilidade, licença e dicionário de dados
│   ├── processed/               # Catálogo semântico gerado pela LLM
│   │   └── games_metadata_with_embeddings.json
│   └── synthetic_enrichment/    # Camada sintética do Bandit
│       ├── offer_catalog.json   # Braços (jogo × incentivo)
│       ├── offer_events.jsonl   # Eventos de impressão simulados
│       └── delayed_rewards.jsonl
├── notebooks/
│   ├── 01_eda_and_quality.ipynb # EDA e relatório de qualidade
│   └── test_ids.py              # Sanity check do join por app_id
├── src/
│   ├── __init__.py
│   └── features/
│       ├── __init__.py
│       ├── extract_semantics.py        # Pipeline de merge + coalesce + LLM
│       └── generate_synthetic_data.py  # Geração dos dados sintéticos do Bandit
└── docs/
    └── architecture-azure.md    # Arquitetura-alvo Azure (planejado — Etapa 6)
```

## 8. Comandos Principais

| Comando | Função |
| :--- | :--- |
| `ollama run llama3.1` | Inicia o prompt interativo da LLM local. |
| `python src/features/extract_semantics.py` | Executa o pipeline de dados (merge + coalesce + Ollama). |
| `python src/features/generate_synthetic_data.py` | Gera o catálogo e os eventos sintéticos do Bandit. |
| `pytest` | Executa a suíte de testes automatizados (planejado — Etapa 5). |

## 9. Limitações Conhecidas

- **Reviews escassas:** jogos com poucas avaliações podem gerar perfis semânticos
  simplistas. *Mitigação:* fallback baseado na categoria padrão do produto.
- **Cold-start de novos jogos:** jogos recém-lançados não têm avaliações
  consolidadas. *Mitigação:* herdar tags do publisher até atingir um mínimo de
  reviews.
- **Cold-start do Ollama:** a primeira requisição pode demorar a recarregar o
  modelo na memória. *Mitigação:* em produção (Azure), manter instâncias mínimas
  ativas.

## 10. Licença

Distribuído sob a licença MIT. Veja o arquivo [LICENSE](LICENSE).
