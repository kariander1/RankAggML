import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Setup
sns.set(style="whitegrid")
os.makedirs("outputs/data", exist_ok=True)

# Load dataset
df = pd.read_csv("data/vgsales.csv")

# --- 1. Histogram of games by release year ---
plt.figure(figsize=(10, 6))
df['Year'].dropna().astype(int).value_counts().sort_index().plot(kind='bar')
plt.title("Number of Games Released by Year")
plt.xlabel("Year")
plt.ylabel("Number of Games")
plt.tight_layout()
plt.savefig("outputs/data/games_by_year.jpg")
plt.close()

# --- 2. Games released on multiple platforms ---
multi_platform_counts = df.groupby('Name')['Platform'].nunique()
multi_platform = multi_platform_counts[multi_platform_counts > 0]

plt.figure(figsize=(8, 6))
multi_platform.value_counts().sort_index().plot(kind='bar')
plt.title("Number of Games Released on Multiple Platforms")
plt.xlabel("Number of Platforms")
plt.ylabel("Number of Games")
plt.tight_layout()
plt.savefig("outputs/data/multi_platform_games.jpg")
plt.close()

# --- 3. Histogram of game genres ---
plt.figure(figsize=(10, 6))
df['Genre'].value_counts().plot(kind='bar')
plt.title("Distribution of Game Genres")
plt.xlabel("Genre")
plt.ylabel("Number of Games")
plt.tight_layout()
plt.savefig("outputs/data/genre_histogram.jpg")
plt.close()

# --- 4. Top 10 publishers by number of games ---
top_publishers = df['Publisher'].value_counts().head(10)

plt.figure(figsize=(10, 6))
top_publishers.plot(kind='bar')
plt.title("Top 10 Publishers by Number of Games")
plt.xlabel("Publisher")
plt.ylabel("Number of Games")
plt.tight_layout()
plt.savefig("outputs/data/top_10_publishers.jpg")
plt.close()

# --- 5. Top 10 best-selling games in each region ---
region_cols = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']
top_games_by_region = {}

for region in region_cols:
    top_games = df[['Name', region]].dropna().groupby('Name').sum()
    top_games = top_games.sort_values(by=region, ascending=False).head(10)
    top_games_by_region[region] = top_games

    # Plot each
    plt.figure(figsize=(10, 6))
    top_games[region].plot(kind='barh')
    plt.title(f"Top 10 Best-Selling Games in {region.replace('_Sales', '')}")
    plt.xlabel("Sales (millions)")
    plt.ylabel("Game")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f"outputs/data/top_10_{region.lower()}.jpg")
    plt.close()

#


# Compute total global sales
df['Global_Sales'] = df[region_cols].sum(axis=1)

# Select top 10 global games
top_global = df.groupby('Name')[region_cols + ['Global_Sales']].sum()
top10 = top_global.sort_values(by='Global_Sales', ascending=False).head(10)

# Drop global sales (only need per-region for stacking)
top10 = top10[region_cols]

# Plot
plt.figure(figsize=(12, 7))
top10.plot(kind='bar', stacked=True, colormap='tab20', figsize=(12, 7))

plt.title("Top 10 Best-Selling Games by Region (Stacked Bar)")
plt.ylabel("Sales (millions)")
plt.xlabel("Game")
plt.xticks(rotation=45, ha='right')
plt.legend(title="Region")
plt.tight_layout()
plt.savefig("outputs/data/top_10_games_stacked_regions.jpg")
plt.close()
