import os
import json
import numpy as np
from datetime import datetime, timedelta

# Semente unica para tornar TODA a geracao reproduzivel (exigencia da Etapa 2).
SEED = 42
np.random.seed(SEED)

# Caminhos resolvidos a partir deste arquivo (funciona rodando de qualquer pasta).
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CATALOG_PATH = os.path.join(BASE_DIR, "..", "..", "data", "processed", "games_metadata_with_embeddings.json")
ENRICHMENT_DIR = os.path.join(BASE_DIR, "..", "..", "data", "synthetic_enrichment")

# Escala da simulacao.
NUM_USERS = 2000
NUM_ROUNDS = 50000
HORIZON_DAYS = 90                   # janela temporal
START_DATE = datetime(2026, 1, 1)   # data fixa para não quebrar a idempotência temporal

PLATFORMS = ["PC", "Console", "Mobile"]

INCENTIVES = [
    {"incentive_id": "FULL_PRICE", "discount_pct": 0.00},
    {"incentive_id": "COUPON_10", "discount_pct": 0.10},
    {"incentive_id": "COUPON_20", "discount_pct": 0.20},
    {"incentive_id": "VIP_PASS", "discount_pct": 1.00},
]


def load_processed_catalog(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Catalogo nao encontrado em {path}. Rode o extract_semantics.py primeiro.")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def assign_platforms():
    """Plataformas SINTETICAS de um jogo. PC sempre presente; Console/Mobile por sorteio."""
    platforms = ["PC"]
    if np.random.random() < 0.50:
        platforms.append("Console")
    if np.random.random() < 0.35:
        platforms.append("Mobile")
    return platforms


def build_offer_catalog(games):
    """Cada braco = jogo x incentivo. Jogos gratis (base_price == 0) so recebem FULL_PRICE,
    pois desconto sobre jogo gratuito nao faz sentido como braco."""
    offer_catalog = []
    arm_id = 0
    for game in games:
        base_price = float(game.get("base_price", 0.0))
        platforms = assign_platforms()
        is_free = base_price == 0.0
        incentives = [INCENTIVES[0]] if is_free else INCENTIVES

        for inc in incentives:
            final_price = round(base_price * (1.0 - inc["discount_pct"]), 2)
            offer_catalog.append({
                "arm_id": arm_id,
                "game_id": game["game_id"],
                "game_name": game["game_name"],
                "primary_genre": game["primary_genre"],
                "semantic_features": game["semantic_profile"],
                "incentive_id": inc["incentive_id"],
                "base_price": base_price,
                "final_price": final_price,
                "platforms": platforms,          # SINTETICO (ver reports/data-generation.md)
            })
            arm_id += 1
    return offer_catalog


def build_user_profiles(num_users, catalog_genres):
    """Usuarios sinteticos. O genero favorito vem dos generos que existem no catalogo,
    para o Bandit ter sinal real de afinidade para aprender."""
    profiles = []
    for user_id in range(1000, 1000 + num_users):
        profiles.append({
            "user_id": user_id,
            "preferred_platform": str(np.random.choice(PLATFORMS, p=[0.4, 0.4, 0.2])),
            "favorite_genre": str(np.random.choice(catalog_genres)),
            "price_sensitivity": round(float(np.random.uniform(0.1, 1.0)), 2),
            "created_at": (START_DATE - timedelta(days=int(np.random.randint(10, 100)))).isoformat(),
        })
    return profiles


def conversion_propensity(arm, user, device_context):
    """Ground-truth da simulacao: probabilidade de clique combinando afinidade de
    genero, de preco e de plataforma. NAO entra no modelo de decisao; serve so para
    gerar recompensas plausiveis."""
    genre_bonus = 0.4 if arm["primary_genre"] == user["favorite_genre"] else 0.0

    is_cheap = arm["incentive_id"] in ("COUPON_20", "VIP_PASS") or arm["final_price"] == 0.0
    if user["price_sensitivity"] > 0.7:           # cacador de ofertas
        price_affinity = 0.5 if is_cheap else -0.3
    else:                                          # usuario padrao
        price_affinity = 0.1 if arm["incentive_id"] == "FULL_PRICE" else 0.0

    platform_affinity = 0.1 if device_context == user["preferred_platform"] else 0.0

    return min(0.9, max(0.05, 0.2 + genre_bonus + price_affinity + platform_affinity)), genre_bonus


def simulate_rounds(offer_catalog, user_profiles):
    offer_events = []
    delayed_rewards = []
    step_seconds = HORIZON_DAYS * 86400 / NUM_ROUNDS

    for round_id in range(NUM_ROUNDS):
        user = user_profiles[np.random.randint(len(user_profiles))]
        device_context = str(np.random.choice(PLATFORMS, p=[0.5, 0.3, 0.2]))

        eligible_arms = [arm for arm in offer_catalog if device_context in arm["platforms"]]
        if not eligible_arms:
            eligible_arms = offer_catalog
        selected_arm = eligible_arms[np.random.randint(len(eligible_arms))]

        click_probability, genre_bonus = conversion_propensity(selected_arm, user, device_context)
        clicked = int(np.random.binomial(1, click_probability))

        bought = 0
        gameplay_hours = 0.0
        reward = 0.0

        if clicked:
            reward += 0.1
            buy_probability = min(0.8, max(0.02, 0.1 + genre_bonus * 1.5))
            bought = int(np.random.binomial(1, buy_probability))
            if bought:
                reward += 0.4
                # Recompensa atrasada: Quem curte o genero joga mais.
                mean_hours = 15.0 if selected_arm["primary_genre"] == user["favorite_genre"] else 1.5
                gameplay_hours = round(float(np.random.exponential(mean_hours)), 1)
                if gameplay_hours >= 2.0:
                    reward += 0.5                # consolida (limite de reembolso superado)
                else:
                    reward -= 0.3                # risco de reembolso / abandono precoce

        timestamp = START_DATE + timedelta(seconds=round(round_id * step_seconds))
        event_id = f"evt_{round_id:06d}"

        offer_events.append({
            "event_id": event_id,
            "timestamp": timestamp.isoformat(),
            "user_id": user["user_id"],
            "device_context": device_context,
            "arm_id": selected_arm["arm_id"],
            "game_name": selected_arm["game_name"],
            "incentive_id": selected_arm["incentive_id"],
            "clicked": clicked,
        })

        if clicked:
            delay_seconds = int(np.random.randint(300, 86400 * 2))   # 5 min a 2 dias
            delayed_rewards.append({
                "event_id": event_id,
                "reward_timestamp": (timestamp + timedelta(seconds=delay_seconds)).isoformat(),
                "user_id": user["user_id"],
                "bought": bought,
                "gameplay_hours": gameplay_hours,
                "accumulated_reward": round(reward, 2),
            })

    return offer_events, delayed_rewards


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def save_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def generate_synthetic_data():
    os.makedirs(ENRICHMENT_DIR, exist_ok=True)

    games = load_processed_catalog(CATALOG_PATH)
    print(f"Carregados {len(games)} jogos do catalogo.")

    offer_catalog = build_offer_catalog(games)
    catalog_genres = sorted({g["primary_genre"] for g in games})
    user_profiles = build_user_profiles(NUM_USERS, catalog_genres)
    offer_events, delayed_rewards = simulate_rounds(offer_catalog, user_profiles)

    save_json(os.path.join(ENRICHMENT_DIR, "offer_catalog.json"), offer_catalog)
    save_json(os.path.join(ENRICHMENT_DIR, "user_profiles.json"), user_profiles)
    save_jsonl(os.path.join(ENRICHMENT_DIR, "offer_events.jsonl"), offer_events)
    save_jsonl(os.path.join(ENRICHMENT_DIR, "delayed_rewards.jsonl"), delayed_rewards)

    clicks = sum(e["clicked"] for e in offer_events)
    print("Geracao sintetica concluida!")
    print(f"  offer_catalog.json   : {len(offer_catalog)} bracos")
    print(f"  user_profiles.json   : {len(user_profiles)} usuarios")
    print(f"  offer_events.jsonl   : {len(offer_events)} impressoes ({clicks} cliques)")
    print(f"  delayed_rewards.jsonl: {len(delayed_rewards)} recompensas atrasadas")


if __name__ == "__main__":
    generate_synthetic_data()
