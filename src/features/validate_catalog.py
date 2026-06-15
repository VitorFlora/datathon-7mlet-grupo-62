import os
import json
from collections import Counter

# Caminho do catalogo, resolvido a partir da localizacao deste arquivo
# (funciona rodando de qualquer pasta).
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CATALOG_PATH = os.path.join(BASE_DIR, "..", "..", "data", "processed", "games_metadata_with_embeddings.json")

EXPECTED_FIELDS = {"game_id", "game_name", "base_price", "release_date", "primary_genre", "semantic_profile"}

# Mesmas palavras proibidas do prompt de extracao (extract_semantics.py).
banido_WORDS = {
    "gameplay", "storyline", "graphics", "ai", "achievements", "fun", "good",
    "bad", "gaming", "addictive", "play", "loop", "mechanics", "theme", "vibe",
    "sound", "audio",
}


def load_catalog(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def check_structure(catalog):
    """Verifica estrutura do catalogo"""
    missing_fields = sum(1 for g in catalog if not EXPECTED_FIELDS.issubset(g))
    empty_profiles = sum(1 for g in catalog if not g.get("semantic_profile"))
    bad_weights = sum(
        1 for g in catalog for f in g.get("semantic_profile", [])
        if not (0.0 <= f.get("weight", -1) <= 1.0)
    )
    ids = [g.get("game_id") for g in catalog]
    duplicate_ids = len(ids) - len(set(ids))
    missing_dates = sum(1 for g in catalog if not g.get("release_date") or g["release_date"] == "Unknown")

    return {
        "missing_fields": missing_fields,
        "empty_profiles": empty_profiles,
        "bad_weights": bad_weights,
        "duplicate_ids": duplicate_ids,
        "missing_dates": missing_dates,
    }


def check_tag_rules(catalog):
    """Regras de texto das tags"""
    underline = words_over_3 = banido = total = 0
    for g in catalog:
        for f in g.get("semantic_profile", []):
            name = f.get("feature_name", "")
            total += 1
            if "_" in name:
                underline += 1
            if len(name.split()) > 3:
                words_over_3 += 1
            tokens = {t.strip(".,!").lower() for t in name.split()}
            if tokens & banido_WORDS:
                banido += 1
    return {"qtd_features": total, "underline": underline, "words_over_3": words_over_3, "banido": banido}


def main():
    if not os.path.exists(CATALOG_PATH):
        print(f"Catalogo nao encontrado em {CATALOG_PATH}")
        print("Rode antes o extract_semantics.py para gerar o catalogo.")
        return

    catalog = load_catalog(CATALOG_PATH)
    total = len(catalog)

    print("=" * 60)
    print("RELATORIO DE QUALIDADE DO CATALOGO PROCESSADO")
    print("=" * 60)
    print(f"Total de jogos: {total}\n")

    structure = check_structure(catalog)
    print("--- Estrutura e integridade (esperado: tudo 0) ---")
    print(f"Jogos sem todos os campos esperados : {structure['missing_fields']}")
    print(f"Perfis semanticos vazios            : {structure['empty_profiles']}")
    print(f"Pesos fora de [0, 1]                : {structure['bad_weights']}")
    print(f"game_id duplicados                  : {structure['duplicate_ids']}")
    print(f"release_date ausente/Unknown        : {structure['missing_dates']}")

    print("\n--- Distribuicao de features por jogo ---")
    feat_dist = Counter(len(g.get("semantic_profile", [])) for g in catalog)
    for n in sorted(feat_dist):
        print(f"{n} features: {feat_dist[n]} jogos")

    print("\n--- Distribuicao de generos ---")
    genres = Counter(g.get("primary_genre") for g in catalog)
    for genre, count in genres.most_common():
        print(f"{genre:18s}: {count}")

    print("\n--- Qualidade das tags (avisos) ---")
    tags = check_tag_rules(catalog)
    tot = tags["qtd_features"] or 1
    print(f"Total de features            : {tags['qtd_features']}")
    print(f"Com palavra proibida         : {tags['banido']} ({100*tags['banido']/tot:.1f}%)")
    print(f"Com mais de 3 palavras       : {tags['words_over_3']} ({100*tags['words_over_3']/tot:.1f}%)")
    print(f"Com underline               : {tags['underline']} ({100*tags['underline']/tot:.1f}%)")

    free_games = sum(1 for g in catalog if not g.get("base_price"))
    print(f"\nJogos free-to-play (base_price = 0): {free_games}")

if __name__ == "__main__":
    main()
