import pandas as pd
import json


def calculate_elo_and_winpct(file_path):
    # ELO hesaplama parametreleri
    K = 32
    initial_elo = 1200

    # CSV dosyasını okuma
    df = pd.read_csv(file_path)

    # ELO ve kazanma yüzdesi hesaplamaları için oyuncu verilerini saklama
    players = {}

    def expected_score(player_elo, opponent_elo):
        return 1 / (1 + 10 ** ((opponent_elo - player_elo) / 400))

    def update_elo(player, opponent, result):
        player_elo = players[player]["elo"]
        opponent_elo = players[opponent]["elo"]

        expected = expected_score(player_elo, opponent_elo)
        if result == 1:
            score = 1
        elif result == 0:
            score = 0.5
        else:
            score = 0

        new_elo = player_elo + K * (score - expected)
        players[player]["elo"] = new_elo
        players[player]["games_played"] += 1
        if result == 1:
            players[player]["wins"] += 1
        elif result == 0:
            players[player]["draws"] += 1

        # Aynı işlemleri rakip için de tersine yapalım
        opponent_result = 2 if result == 1 else 1 if result == 2 else 0
        expected_opponent = expected_score(opponent_elo, player_elo)
        opponent_score = (
            1 if opponent_result == 1 else 0.5 if opponent_result == 0 else 0
        )
        new_opponent_elo = opponent_elo + K * (opponent_score - expected_opponent)
        players[opponent]["elo"] = new_opponent_elo
        players[opponent]["games_played"] += 1
        if opponent_result == 1:
            players[opponent]["wins"] += 1
        elif opponent_result == 0:
            players[opponent]["draws"] += 1

    # Her maç için ELO ve WinPct hesapla
    for _, row in df.iterrows():
        player1, player2, result = row["oyuncu1"], row["oyuncu2"], row["mac_sonucu"]

        if player1 not in players:
            players[player1] = {
                "elo": initial_elo,
                "games_played": 0,
                "wins": 0,
                "draws": 0,
            }
        if player2 not in players:
            players[player2] = {
                "elo": initial_elo,
                "games_played": 0,
                "wins": 0,
                "draws": 0,
            }

        update_elo(player1, player2, result)

    # Sonuçları bir DataFrame'e çevirme
    results = []
    for player, stats in players.items():
        win_pct = (
            (stats["wins"] / stats["games_played"]) * 100
            if stats["games_played"] > 0
            else 0
        )
        results.append([player, stats["elo"], win_pct])

    result_df = pd.DataFrame(results, columns=["Oyuncu", "ELO", "WinPct"])

    return result_df


# CSV dosyasını işle ve sonuçları göster
file_path = "oyun_sonuclari.csv"
results = calculate_elo_and_winpct(file_path)
print(results)
with open("elo_winpct.json", "w") as f:
    json.dump(results.to_dict(orient="records"), f, indent=4)
