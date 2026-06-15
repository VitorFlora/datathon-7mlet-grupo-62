import pandas as pd
import time

path_ds1 = "../data/kaggle/kaggle_steam_reviews.csv"
path_ds2 = "../data/kaggle/kaggle_steam_game.csv"

start_time = time.perf_counter()
df1 = pd.read_csv(path_ds1, usecols=['app_id', 'app_name', 'review_text'], encoding_errors='replace')
df2 = pd.read_csv(path_ds2, usecols=['appid', 'name', 'release_date', 'price', 'reviews'], encoding_errors='replace')
end_time = time.perf_counter()
print(f"Datasets loaded in {end_time - start_time:.2f} seconds.")

df_filtered1 = df1[df1['app_id'] == 99830]
df_filtered2 = df2[df2['appid'] == 99830]

print(df_filtered1[['app_id', 'app_name', 'review_text']].head(5))
print(df_filtered2[['appid', 'name', 'reviews']].head(5))