import os
import json
import pandas as pd
import ollama
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
import time

class GameFeature(BaseModel):
    feature_name: str = Field(description="Nome da característica semântica.")
    weight: float = Field(description="Peso ou relevância desta feature para o jogo, de 0.0 a 1.0.")

class GameSemantics(BaseModel):
    primary_genre: Literal["FPS", "RPG", "RTS", "Indie", "Simulation", "Action", "Adventure", "Sports", "Puzzle", "Strategy", "MOBA", "Hack and slash", "Horror", "Arcade", "Exploration", "Fantasy", "Racing"] = Field(
        description="Gênero principal inferido com base na review."
    )
    semantic_profile: List[GameFeature] = Field(description="Lista de no máximo 5 características marcantes do gameplay.")

def merge_and_clean_datasets(path_ds1: str, path_ds2: str) -> pd.DataFrame:
    print("Carregando datasets do Kaggle...")
    start_time = time.perf_counter()
    df1 = pd.read_csv(path_ds1)
    df2 = pd.read_csv(path_ds2)
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

def extract_semantics_from_reviews(game_name: str, aggregated_reviews: str) -> Optional[dict]:
    try:
        response = ollama.chat(
            model='llama3.1',
            messages=[
                {
                    "role": "system", 
                    "content": """Extract exactly 6 highly specific semantic tags capturing the core player experience, psychological vibe, and mechanical depth found in these reviews.

                                    RULES:
                                    1. DO NOT summarize the game's plot, basic premise, or store-page description. 
                                    2. FOCUS ON EXPERIENCE: Capture what players actually feel, suffer, or master (e.g., "oppressive environmental isolation", "conveyor logistics loop").
                                    3. IDENTITY & DIFFERENTIATORS: Look for what uniquely distinguishes this game from its closest competitors. Why do fans stay?
                                    4. BANNED FILLER WORDS: Gameplay, storyline, graphics, AI, achievements, fun, good, bad, gaming, addictive, play, loop, mechanics, theme, vibe, sound, audio.
                                    5. FORMAT: Maximum 3 words per tag. No underscores. No hyphens unless necessary.

                                    Reply ONLY with a valid JSON array of strings. Do not add markdown or explanations."""
                },
                {
                    "role": "user", 
                    "content": f"Jogo: {game_name}\nCompilado de Reviews:\n{aggregated_reviews}"
                }
            ],
            format=GameSemantics.model_json_schema(),
            options={"temperature": 0.05, "num_ctx": 3072}
        )
        return json.loads(response['message']['content'])        
    except Exception as e:
        print(f"Erro ao processar o jogo {game_name}: {str(e)}")
        return None

def main():
    start_time = time.perf_counter()
    path_ds1 = "../../data/kaggle/kaggle_steam_reviews.csv"
    path_ds2 = "../../data/kaggle/kaggle_steam_game.csv"
    
    output_dir = "../../data/processed"
    output_path = os.path.join(output_dir, "games_metadata_with_embeddings.json")
    max_games = 10
    max_review = 200

    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(path_ds1) or not os.path.exists(path_ds2):
        print("Certifique-se de que os datasets estão na pasta data/kaggle/")
        return

    try:
        ollama.generate(model='llama3.1', prompt='', keep_alive=0)
    except:
        pass

    df_cleaned = merge_and_clean_datasets(path_ds1, path_ds2)

    # Agregando reviews
    df_top_reviews = df_cleaned.groupby('app_id').head(max_review)
    
    aggregated_data = []
    grouped = df_top_reviews.groupby(['app_id', 'app_name'])
    
    for (game_id, game_name), group_df in grouped:
        reviews_list = group_df['review_text'].dropna().astype(str).tolist()
        
        # Estruturação em XML para melhorar a Atenção (Attention Mechanism) do LLM
        xml_reviews = []
        for idx, text in enumerate(reviews_list, 1):
            xml_reviews.append(f'<review id="{idx}">\n{text}\n</review>')
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

    try:
        ollama.generate(model='llama3.1', prompt='', keep_alive=0)
    except:
        pass

    end_time = time.perf_counter()
    print(f"Tempo total gasto: ({(end_time - start_time)/60:.2f} minutos) carregando e processando {sample_size} jogos.")

if __name__ == "__main__":
    main()