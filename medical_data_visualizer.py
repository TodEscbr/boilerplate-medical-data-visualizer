import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. Import data
df = pd.read_csv('medical_examination.csv')

# 2. Tambah kolum overweight
df['overweight'] = ((df['weight'] / ((df['height'] / 100) ** 2)) > 25).astype(int)

# 3. Normalisasi cholesterol dan gluc (0 = baik, 1 = tidak baik)
df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
df['gluc'] = (df['gluc'] > 1).astype(int)

# 4
def draw_cat_plot():
    # 5. Ubah ke long format
    df_cat = pd.melt(
        df,
        id_vars='cardio',
        value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight']
    )

    # 6. Group dan kira jumlah setiap kombinasi
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')

    # 7. Lukis catplot
    fig = sns.catplot(
        data=df_cat,
        x='variable',
        y='total',
        hue='value',
        col='cardio',
        kind='bar'
    ).fig

    # 8. Simpan ke fail dan return
    fig.savefig('catplot.png')
    return fig

# 10
def draw_heat_map():
    # 11. Bersihkan data
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # 12. Kira correlation matrix
    corr = df_heat.corr()

    # 13. Buat mask untuk segitiga atas
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 14. Setup figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # 15. Lukis heatmap
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt='.1f',
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5}
    )

    # 16. Simpan ke fail dan return
    fig.savefig('heatmap.png')
    return fig
