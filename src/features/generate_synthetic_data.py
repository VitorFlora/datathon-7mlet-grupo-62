import os
import json
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Configura sementes para garantir reprodutibilidade (Exigência da Etapa 2)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

def load_processed_catalog(path: str) -> list:
    """Carrega o catálogo semântico gerado pelo nosso pipeline do Ollama."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Catálogo processado não encontrado em {path}. Execute o extract_semantics.py primeiro.")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def generate_synthetic_data():
    processed_catalog_path = "../../data/processed/games_metadata_with_embeddings.json"
    enrichment_dir = "../../data/synthetic_enrichment"
    os.makedirs(enrichment_dir, exist_ok=True)
    
    # 1. Carrega os dados reais processados
    games_catalog = load_processed_catalog(processed_catalog_path)
    num_games = len(games_catalog)
    print(f"Carregados {num_games} jogos para servir de braços base.")

    # 2. Define o Catálogo de Incentivos (Braços do MAB)
    # No nosso marketplace, o Bandit decide a combinação de Jogo + Nível de Desconto/Incentivo
    incentives = [
        {"incentive_id": "FULL_PRICE", "discount_pct": 0.0, "description": "Sem desconto."},
        {"incentive_id": "COUPON_10", "discount_pct": 0.10, "description": "Desconto de 10% no checkout."},
        {"incentive_id": "COUPON_20", "discount_pct": 0.20, "description": "Desconto de 20% no checkout."},
        {"incentive_id": "VIP_PASS", "discount_pct": 1.00, "description": "Incluso na Assinatura VIP (Custo 100% subsidiado)."}
    ]

    # Salva o offer_catalog oficial separado fisicamente do Kaggle original (Etapa 2)
    offer_catalog = []
    arm_id = 0
    for game in games_catalog:
        for inc in incentives:
            # Preço final simulado
            base_price = float(game.get("base_price", 59.90))
            final_price = base_price * (1.0 - inc["discount_pct"])
            
            offer_catalog.append({
                "arm_id": arm_id,
                "game_id": game["game_id"],
                "game_name": game["game_name"],
                "primary_genre": game["primary_genre"],
                "semantic_features": game["semantic_profile"],
                "incentive_id": inc["incentive_id"],
                "final_price": round(final_price, 2),
                "platforms": game.get("platforms", ["PC"])
            })
            arm_id += 1

    with open(os.path.join(enrichment_dir, "offer_catalog.json"), "w", encoding="utf-8") as f:
        json.dump(offer_catalog, f, indent=4, ensure_ascii=False)
    print(f"Salvo catálogo de ofertas sintéticas com {len(offer_catalog)} combinações de braços.")

    # 3. Geração de Usuários Sintéticos e Contextos de Decisão
    # Simula 500 usuários ativos na nossa plataforma cross-platform
    num_users = 500
    user_profiles = []
    
    platforms_pool = ["PC", "Console", "Mobile"]
    
    # Criamos preferências semânticas para os usuários (para o bandit ter o que aprender!)
    for user_id in range(1000, 1000 + num_users):
        user_profiles.append({
            "user_id": user_id,
            "preferred_platform": np.random.choice(platforms_pool, p=[0.4, 0.4, 0.2]),
            # Afinidade a gêneros de jogos
            "favorite_genre": np.random.choice(["RPG", "Action", "Strategy", "Indie", "Simulation"]),
            # Sensibilidade a preço (usuários com alta sensibilidade só compram com cupons agressivos)
            "price_sensitivity": round(np.random.uniform(0.1, 1.0), 2),
            "created_at": (datetime.now() - timedelta(days=np.random.randint(10, 100))).isoformat()
        })

    # 4. Geração de Eventos de Impressão (offer_events) e Recompensas Atrasadas (delayed_rewards)
    # Simularemos 5000 rodadas (decisões) de ofertas ocorrendo ao longo de 30 dias
    num_rounds = 5000
    offer_events = []
    delayed_rewards = []
    
    start_date = datetime.now() - timedelta(days=30)
    
    for round_id in range(num_rounds):
        # Seleciona aleatoriamente um usuário e seu contexto de acesso
        user = np.random.choice(user_profiles)
        device_context = np.random.choice(platforms_pool, p=[0.5, 0.3, 0.2]) # Aparelho usado na hora do clique
        
        # Filtra os braços do catálogo elegíveis para a plataforma atual
        eligible_arms = [arm for arm in offer_catalog if device_context in arm["platforms"]]
        if not eligible_arms:
            eligible_arms = offer_catalog # Fallback de segurança
            
        # Simula o braço selecionado (no log histórico, combinamos aleatoriedade + propensão para simular exploração)
        selected_arm = np.random.choice(eligible_arms)
        
        # --- Cálculo Semântico de Propensão à Conversão (Ground Truth para simulação) ---
        # 1. Afinidade de Gênero: ganha bônus se o gênero bate com a preferência do usuário
        genre_bonus = 0.4 if selected_arm["primary_genre"] == user["favorite_genre"] else 0.0
        
        # 2. Afinidade de Preço: cupons altos ajudam usuários muito sensíveis a preço
        price_affinity = 0.0
        if user["price_sensitivity"] > 0.7:  # Usuário "caçador de ofertas"
            if selected_arm["incentive_id"] in ["COUPON_20", "VIP_PASS"]:
                price_affinity = 0.5
            else:
                price_affinity = -0.3
        else: # Usuário padrão
            if selected_arm["incentive_id"] == "FULL_PRICE":
                price_affinity = 0.1
                
        # Calcula probabilidade final combinada de clique e compra
        click_probability = min(0.9, max(0.05, 0.2 + genre_bonus + price_affinity))
        
        # Simula se clicou ou não
        clicked = int(np.random.binomial(1, click_probability))
        
        # Se clicou, simula a probabilidade de compra baseada nas horas que ele vai gastar
        bought = 0
        gameplay_hours = 0.0
        reward = 0.0
        
        if clicked:
            reward += 0.1  # Recompensa imediata de clique
            buy_probability = min(0.8, max(0.02, 0.1 + (genre_bonus * 1.5)))
            bought = int(np.random.binomial(1, buy_probability))
            
            if bought:
                reward += 0.4  # Recompensa de compra
                # Simula gameplay (delayed reward) baseada no alinhamento de preferência
                if selected_arm["primary_genre"] == user["favorite_genre"]:
                    gameplay_hours = round(np.random.exponential(15.0), 1)  # Ama o jogo, joga muito
                else:
                    gameplay_hours = round(np.random.exponential(1.5), 1)   # Não curtiu, joga pouco (risco reembolso)
                
                # Se jogou mais de 2 horas, valida a recompensa cheia. Se não, reembolso simula perda
                if gameplay_hours >= 2.0:
                    reward += 0.5  # Recompensa máxima consolidada (1.0 total)
                else:
                    reward -= 0.3  # Penalidade de reembolso / abandono precoce
        
        timestamp = start_date + timedelta(seconds=round_id * 500)
        
        # Registra evento de impressão (Etapa 2)
        offer_events.append({
            "event_id": f"evt_{round_id:06d}",
            "timestamp": timestamp.isoformat(),
            "user_id": user["user_id"],
            "device_context": device_context,
            "arm_id": selected_arm["arm_id"],
            "game_name": selected_arm["game_name"],
            "incentive_id": selected_arm["incentive_id"],
            "clicked": clicked
        })
        
        # Registra o evento de recompensa atrasada (Delayed Reward)
        # Em produção, esse evento chega minutos, horas ou dias depois do clique!
        if clicked:
            delay_seconds = np.random.randint(300, 86400 * 2) # Delay simulado de 5 min a 2 dias
            reward_timestamp = timestamp + timedelta(seconds=delay_seconds)
            delayed_rewards.append({
                "event_id": f"evt_{round_id:06d}",
                "reward_timestamp": reward_timestamp.isoformat(),
                "user_id": user["user_id"],
                "bought": bought,
                "gameplay_hours": gameplay_hours,
                "accumulated_reward": round(reward, 2)
            })

    # Escrita dos arquivos de eventos e delayed rewards no formato JSONL (Exigência técnica)
    with open(os.path.join(enrichment_dir, "offer_events.jsonl"), "w", encoding="utf-8") as f:
        for ev in offer_events:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")
            
    with open(os.path.join(enrichment_dir, "delayed_rewards.jsonl"), "w", encoding="utf-8") as f:
        for dr in delayed_rewards:
            f.write(json.dumps(dr, ensure_ascii=False) + "\n")

    print(f"Geração Sintética Concluída com sucesso!")
    print(f"Arquivos salvos em: {enrichment_dir}")
    print(f"offer_catalog.json ({len(offer_catalog)} ofertas mapeadas)")
    print(f"offer_events.jsonl ({len(offer_events)} registros de impressões)")
    print(f"delayed_rewards.jsonl ({len(delayed_rewards)} recompensas tardias registradas)")

if __name__ == "__main__":
    generate_synthetic_data()