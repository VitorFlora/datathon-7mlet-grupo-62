import pandas as pd
import time

start_time = time.perf_counter()
df1 = pd.read_csv("../data/kaggle/kaggle_steam_reviews.csv")
df2 = pd.read_csv("../data/kaggle/kaggle_steam_game.csv")
end_time = time.perf_counter()
print(f"Datasets loaded in {end_time - start_time:.2f} seconds.")

df_filtered1 = df1[df1['app_id'] == 360]
df_filtered2 = df2[df2['appid'] == 360]

print(df_filtered1[['app_id', 'app_name', 'review_text']].head(5))
print(df_filtered2[['appid', 'name', 'reviews']].head(5))