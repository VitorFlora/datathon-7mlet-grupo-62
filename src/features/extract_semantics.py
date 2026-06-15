import os
import json
import pandas as pd
import ollama
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Literal
import time

ALLOWED_GENRES = [
    "FPS", "RPG", "RTS", "Indie", "Simulation", "Action", "Adventure",
    "Sports", "Puzzle", "Strategy", "MOBA", "Hack and slash", "Horror",
    "Arcade", "Exploration", "Fantasy", "Racing",
]

# Modelo da LLM local. Em maquinas sem GPU dedicada, troque por um modelo menor
# e bem mais rapido, por exemplo "llama3.2:3b" (lembre de rodar: ollama pull llama3.2:3b).
MODEL_NAME = "llama3.1" #"llama3.2:3b" 
KEEP_ALIVE = "10m"
GEN_OPTIONS = {"temperature": 0.0, "num_ctx": 16384, "num_predict": 384}
max_games = 500
max_review = 100
max_review_chars = 600

class GameFeature(BaseModel):
    feature_name: str = Field(max_length=40, description="Tag curta (no máximo 3 palavras) sobre a experiência do jogador.")
    weight: float = Field(ge=0.0, le=1.0, description="Relevância da feature para o jogo, de 0.0 a 1.0.")

    # Ajusta o peso para o intervalo [0, 1] em vez de descartar o jogo
    # quando a LLM devolve um valor fora da faixa (ex.: -0.6).
    @field_validator("weight", mode="before")
    @classmethod
    def clamp_weight(cls, value):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return value
        return min(1.0, max(0.0, value))

class GameSemantics(BaseModel):
    primary_genre: Literal["FPS", "RPG", "RTS", "Indie", "Simulation", "Action", "Adventure", "Sports", "Puzzle", "Strategy", "MOBA", "Hack and slash", "Horror", "Arcade", "Exploration", "Fantasy", "Racing"] = Field(
        description="Gênero principal inferido com base nas reviews."
    )
    semantic_profile: List[GameFeature] = Field(
        min_length=3, max_length=5,
        description="Lista de 3 a 5 características marcantes do gameplay."
    )

def merge_and_clean_datasets(path_ds1: str, path_ds2: str) -> pd.DataFrame:
    print("Carregando datasets do Kaggle...")
    start_time = time.perf_counter()
    # Lemos apenas as colunas necessarias (evita carregar as 47 colunas dos jogos,
    # varias com textos enormes, e as colunas de vazamento das reviews).
    # encoding_errors='replace' tolera bytes invalidos no CSV de reviews.
    df1 = pd.read_csv(path_ds1, usecols=['app_id', 'app_name', 'review_text'], encoding_errors='replace')
    df2 = pd.read_csv(path_ds2, usecols=['appid', 'name', 'release_date', 'price', 'reviews'], encoding_errors='replace')
    end_time = time.perf_counter()
    print(f"Datasets carregados em {end_time - start_time:.2f} segundos.")

    if 'name' in df2.columns:
        df2 = df2.rename(columns={'name': 'app_name'})
    if 'reviews' in df2.columns:
        df2 = df2.rename(columns={'reviews': 'review_text_ds2'})
    if 'appid' in df2.columns:
        df2 = df2.rename(columns={'appid': 'app_id'})
    if 'review_text' in df1.columns:
        df1 = df1.rename(columns={'review_text': 'review_text_ds1'})

    df2 = df2.drop_duplicates(subset=['app_id'])

    df_merged = pd.merge(df1, df2[['app_id', 'release_date', 'price', 'review_text_ds2']], on='app_id', how='inner')

    df_merged['review_text'] = df_merged['review_text_ds1'].combine_first(df_merged['review_text_ds2'])
    df_merged = df_merged.dropna(subset=['review_text', 'price', 'release_date'])
    
    leakage_columns = ["review_score", "review_votes", "review_text_ds1", "review_text_ds2"]
    df_cleaned = df_merged.drop(columns=[col for col in leakage_columns if col in df_merged.columns])
    
    print(f"Base unificada possui {len(df_cleaned)} avaliações válidas.")
    return df_cleaned

SYSTEM_PROMPT = f"""You analyze player reviews of a video game and return ONE structured JSON object.

The JSON object has exactly two fields:
- "primary_genre": a single value chosen from this list: {", ".join(ALLOWED_GENRES)}.
- "semantic_profile": a list of 3 to 5 items. Each item has:
    - "feature_name": a short tag describing something concrete about THIS game, drawn
      from its reviews: the dominant feeling, the signature mechanic, the social mode,
      the difficulty, or the art/mood. Derive every tag from the reviews below.
    - "weight": a number from 0.0 to 1.0 showing how strongly the reviews support that feature.

Rules for every "feature_name":
1. Maximum 3 words. No underscores. No hyphens unless truly necessary.
2. Do NOT use these filler words: gameplay, storyline, graphics, AI, achievements,
   fun, good, bad, gaming, addictive, play, loop, mechanics, theme, vibe, sound, audio.
3. Do NOT summarize the plot or the store-page description; focus on the player experience.

Reply ONLY with the JSON object. No markdown, no explanations."""

def extract_semantics_from_reviews(game_name: str, aggregated_reviews: str, max_attempts: int = 2) -> Optional[dict]:
    for attempt in range(1, max_attempts + 1):
        try:
            response = ollama.chat(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Jogo: {game_name}\nCompilado de Reviews:\n{aggregated_reviews}"}
                ],
                format=GameSemantics.model_json_schema(),
                options=GEN_OPTIONS,
                keep_alive=KEEP_ALIVE
            )
            parsed = GameSemantics.model_validate_json(response['message']['content'])
            return parsed.model_dump()
        except Exception as e:
            print(f"  Tentativa {attempt}/{max_attempts} falhou para '{game_name}': {e}")
    return None

def main():
    start_time = time.perf_counter()
    path_ds1 = "../../data/kaggle/kaggle_steam_reviews.csv"
    path_ds2 = "../../data/kaggle/kaggle_steam_game.csv"
    
    output_dir = "../../data/processed"
    output_path = os.path.join(output_dir, "games_metadata_with_embeddings.json")


    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(path_ds1) or not os.path.exists(path_ds2):
        print("Certifique-se de que os datasets estão na pasta data/kaggle/")
        return

    try:
        ollama.generate(model=MODEL_NAME, prompt='', keep_alive=KEEP_ALIVE, options=GEN_OPTIONS)
    except:
        pass

    df_cleaned = merge_and_clean_datasets(path_ds1, path_ds2)

    # Agregando reviews
    df_top_reviews = df_cleaned.groupby('app_id').head(max_review)
    
    aggregated_data = []
    grouped = df_top_reviews.groupby(['app_id', 'app_name'])
    
    for (game_id, game_name), group_df in grouped:
        reviews_list = group_df['review_text'].dropna().astype(str).tolist()
        
        # Estruturação em XML para melhorar a Atenção (Attention Mechanism) do LLM.
        # Cada review é truncada para nao estourar a janela de contexto do modelo.
        xml_reviews = []
        for idx, text in enumerate(reviews_list, 1):
            short_text = text[:max_review_chars]
            xml_reviews.append(f'<review id="{idx}">\n{short_text}\n</review>')
        combined_text = "\n".join(xml_reviews)
        
        raw_date = group_df['release_date'].dropna().iloc[0] if not group_df['release_date'].dropna().empty else "Unknown"
        raw_price = group_df['price'].dropna().iloc[0] if not group_df['price'].dropna().empty else 0.0

        aggregated_data.append({
            'app_id': game_id,
            'app_name': game_name,
            'release_date': raw_date,
            'price': raw_price,
            'review_text': combined_text
        })
        
    df_aggregated = pd.DataFrame(aggregated_data)
    
    sample_size = min(max_games, len(df_aggregated))
    df_sample = df_aggregated.head(sample_size)
    
    processed_games = []
    print(f"\nIniciando extração em batch para {sample_size} jogos...")
    
    for i, row in df_sample.iterrows():
        true_game_id = int(row['app_id'])
        true_game_name = row['app_name']

        print(f"[{i+1}/{sample_size}] Processando game: {true_game_name}...")
        semantics = extract_semantics_from_reviews(true_game_name, row['review_text'])

        if semantics is not None:
            semantics_with_id = {
                "game_id": true_game_id,
                "game_name": true_game_name,
                "base_price": float(row.get('price', 0.0)) if not pd.isna(row.get('price')) else 0.0,
                "release_date": str(row.get('release_date', 'Unknown')),
                **semantics
            }
            processed_games.append(semantics_with_id)
        else:
            print(f"Falha ao extrair dados de {true_game_name}.")
            
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(processed_games, f, indent=4, ensure_ascii=False)
        
    print(f"\nPipeline concluído. Arquivo salvo em {output_path}")

    # Libera a VRAM ao final do batch.
    try:
        ollama.generate(model=MODEL_NAME, prompt='', keep_alive=0)
    except:
        pass

    end_time = time.perf_counter()
    print(f"Tempo total gasto: ({(end_time - start_time)/60:.2f} minutos) carregando e processando {sample_size} jogos.")

if __name__ == "__main__":
    main()