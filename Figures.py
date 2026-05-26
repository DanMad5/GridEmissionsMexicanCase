import pandas as pd
import numpy as np
import os.path
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from matplotlib.patches import Patch
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib as mpl

base_font = 14
mpl.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman"],
    "font.size": base_font,
    "axes.titlesize": base_font + 4, "axes.labelsize": base_font + 1,
    "xtick.labelsize": base_font - 1, "ytick.labelsize": base_font - 1,
    "axes.linewidth": 1.2, "grid.linewidth": 0.3
})

def calculateFC(year, cap_df, tech_names):
    gen_path = f"Data/CompleteGeneration/complete-{str(year)[-2:]}.csv"
    gen_df = pd.read_csv(gen_path)
    tech_gen = gen_df[tech_names]
    cap_year = cap_df[cap_df['Year'] == year].iloc[0]
    return tech_gen.div(cap_year[tech_names])

#Fig 2
def emissionFactorYears(year1=2018, year2=2024):
    base_font = 16
    mpl.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman"],
        "font.size": base_font,
        "axes.titlesize": base_font + 2, "axes.labelsize": base_font + 1,
        "xtick.labelsize": base_font - 1, "ytick.labelsize": base_font - 1,
        "axes.linewidth": 1.2, "legend.fontsize": base_font - 1
    })

    data_to_plot = {}
    target_years = [year1, year2]
    colors = {year1: '#4CAF50', year2: 'blue'}

    for y in target_years:
        file_path = f"Data/EmissionFactors/EFe/EFe-{str(y)[2:]}.csv"
        try:
            df = pd.read_csv(file_path)
            data_to_plot[y] = df.iloc[:, 0]
        except FileNotFoundError:
            print(f"Warning: File {file_path} not found.")
            return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    for y in target_years:
        ax1.plot(data_to_plot[y].values, label=str(y), color=colors[y], alpha=0.7)
    
    ax1.set_ylabel('Emission Factor [tCO$_2$e/MWh]', fontweight='normal')
    ax1.set_title('(a) Hourly Emission Factors')
    ax1.legend(loc='upper right', frameon=True)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 8760)

    for y in target_years:
        ma = data_to_plot[y].rolling(window=24, min_periods=1).mean()
        ax2.plot(ma.values, label=str(y), color=colors[y], alpha=0.7)
    
    ax2.set_xlabel('Time [hour]')
    ax2.set_ylabel('Emission Factor [tCO$_2$e/MWh]')
    ax2.set_title('(b) 24-hour Moving Average')
    ax2.legend(loc='upper right', frameon=True)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 8760)

    plt.tight_layout()
    plt.show()

#Fig 3
def heatmap(title="Monthly Average Emission Factors - Adjusted (2017–2024)"):
    base_font = 14
    mpl.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman"],
        "font.size": base_font, "font.weight": "bold",
        "axes.titlesize": base_font + 4, "axes.labelsize": base_font + 1,
        "xtick.labelsize": base_font - 1, "ytick.labelsize": base_font - 1,
        "axes.linewidth": 1.2, "grid.linewidth": 0.3
    })

    years = [str(y) for y in range(2017, 2025)]
    months_labels = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    
    matriz = np.full((len(years), len(months_labels)), np.nan)

    for i, year in enumerate(years):
        file_path = f"Data/EmissionFactors/EF-MA-Adj/EF-MA-Adj-{year[2:]}.csv"
        
        try:
            df_year = pd.read_csv(file_path)
            data_col = df_year.iloc[:, -1] 
            
            start_date = pd.Timestamp(f'{year}-01-01')
            time_index = pd.date_range(start=start_date, periods=len(data_col), freq='h')
            temp_df = pd.DataFrame({'val': data_col.values}, index=time_index)
            
            monthly_means = temp_df.resample('ME').mean()
            
            for j in range(min(12, len(monthly_means))):
                matriz[i, j] = round(float(monthly_means.iloc[j, 0]), 4)
                    
        except FileNotFoundError:
            print(f"Warning: {file_path} not found. Skipping {year}.")
            continue

    promedios = np.nanmean(matriz, axis=1)
    matriz_ext = np.hstack([matriz, promedios.reshape(-1, 1)]) # 8x13

    vmin_global = np.nanmin(matriz_ext)
    vmax_global = np.nanmax(matriz_ext)
    
    cmap = LinearSegmentedColormap.from_list(
        "custom_blues",
        ["#eaf3fb", "#b7d5ea", "#7fb3d4", "#4f86b8", "#2f5f9b", "#1f3f7a"],
        256
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(matriz_ext, aspect='auto', cmap=cmap, vmin=vmin_global, vmax=vmax_global)

    ax.set_xticks(np.arange(13))
    ax.set_xticklabels(months_labels + ['Average'], rotation=45, ha='right')
    ax.set_yticks(np.arange(len(years)))
    ax.set_yticklabels(years)
    ax.set_xlabel("Month")
    ax.set_ylabel("Year")
    ax.set_title(title)

    for i in range(matriz_ext.shape[0]):
        for j in range(matriz_ext.shape[1]):
            val = matriz_ext[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=9, color="black")

    ax.axvline(x=11.5, color="black", linestyle="--", linewidth=1.2)

    for j in range(matriz_ext.shape[1]):
        col_data = matriz_ext[:, j]
        if np.all(np.isnan(col_data)):
            continue

        idx_max = int(np.nanargmax(col_data))
        idx_min = int(np.nanargmin(col_data))

        ax.add_patch(Rectangle(
            (j - 0.5, idx_max - 0.5), 1, 1,
            fill=False, edgecolor="red", linewidth=2.2
        ))

        ax.add_patch(Rectangle(
            (j - 0.5, idx_min - 0.5), 1, 1,
            fill=False, edgecolor="green", linewidth=2.2
        ))

    cbar = fig.colorbar(im, ax=ax, shrink=0.95, pad=0.02)
    cbar.set_label("Emission Factor [tCO$_2$e/MWh]", rotation=270, labelpad=18)
    
    ax.tick_params(direction="in", length=4, width=1)
    for spine in ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(1.2)

    plt.tight_layout()
    plt.show()

#Fig 4-5 
def boxAndViolin():
    base_font = 16
    mpl.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman"],
        "font.size": base_font, "font.weight": "bold",
        "axes.titlesize": base_font + 2, "axes.labelsize": base_font + 1,
        "xtick.labelsize": base_font - 1, "ytick.labelsize": base_font - 1,
        "axes.linewidth": 1.2, "legend.fontsize": base_font - 1
    })

    alpha_boxes = 0.70
    color_blue = "#0000FF"
    color_green = "#4CAF50"
    blue_rgba = (0/255, 0/255, 255/255, alpha_boxes)
    green_rgba = (76/255, 175/255, 80/255, alpha_boxes)

    years = [str(y) for y in range(2017, 2025)]
    raw_arrays = []
    adj_arrays = []
    valid_years = []

    for year in years:
        file_raw = f"Data/EmissionFactors/EFe/EFe-{year[2:]}.csv"
        file_adj = f"Data/EmissionFactors/EF-Adj/EF-Adj-{year[2:]}.csv"
        
        try:
            df_raw = pd.read_csv(file_raw)
            df_adj = pd.read_csv(file_adj)
            
            raw_arrays.append(df_raw.iloc[:, 0].dropna().values)
            adj_arrays.append(df_adj.iloc[:, 0].dropna().values)
            valid_years.append(year)
        except FileNotFoundError:
            print(f"Warning: File missing for year {year}. Skipping.")
            continue

    if not valid_years:
        print("Error: No data files were successfully loaded.")
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    base_pos = np.arange(len(valid_years))
    offset = 0.18
    pos_raw = base_pos - offset
    pos_adj = base_pos + offset

    v_raw = ax.violinplot(raw_arrays, positions=pos_raw, showmeans=False, showmedians=False, showextrema=False, widths=0.30)
    for body in v_raw["bodies"]:
        body.set_facecolor(blue_rgba)
        body.set_edgecolor("black")
        body.set_linewidth(0.8)
        body.set_alpha(0.35)

    v_adj = ax.violinplot(adj_arrays, positions=pos_adj, showmeans=False, showmedians=False, showextrema=False, widths=0.30)
    for body in v_adj["bodies"]:
        body.set_facecolor(green_rgba)
        body.set_edgecolor("black")
        body.set_linewidth(0.8)
        body.set_alpha(0.35)

    flier_blue = dict(marker='o', markerfacecolor=color_blue, markeredgecolor='black', markersize=3.5, alpha=0.5)
    flier_green = dict(marker='o', markerfacecolor=color_green, markeredgecolor='black', markersize=3.5, alpha=0.5)

    bp1 = ax.boxplot(raw_arrays, positions=pos_raw, widths=0.10, patch_artist=True, showfliers=True, 
                     flierprops=flier_blue, medianprops=dict(color='black', linewidth=1.4), 
                     boxprops=dict(linewidth=1.4, edgecolor="black", facecolor=blue_rgba), 
                     whiskerprops=dict(linewidth=1.2, color=color_blue), capprops=dict(linewidth=1.2, color=color_blue))

    bp2 = ax.boxplot(adj_arrays, positions=pos_adj, widths=0.10, patch_artist=True, showfliers=True, 
                     flierprops=flier_green, medianprops=dict(color='black', linewidth=1.4), 
                     boxprops=dict(linewidth=1.4, edgecolor="black", facecolor=green_rgba), 
                     whiskerprops=dict(linewidth=1.2, color=color_green), capprops=dict(linewidth=1.2, color=color_green))

    ref_vals = [0.582, 0.527, 0.505, 0.494, 0.423, 0.435, 0.438, 0.444]
    ref_dict = dict(zip([str(y) for y in range(2017, 2025)], ref_vals))
    valid_refs = [ref_dict[y] for y in valid_years]
    ax.scatter(base_pos, valid_refs, marker='D', s=45, facecolor='red', edgecolor='black', zorder=5, label='Reported EF')

    # Formatting Aesthetics
    ax.set_xticks(base_pos)
    ax.set_xticklabels(valid_years)
    ax.set_xlabel("Year")
    ax.set_ylabel("Emission factor [tCO$_2$e/MWh]")
    ax.set_ylim(0.2, 0.75)
    ax.set_yticks(np.arange(0.2, 0.76, 0.1))
    
    ax.set_axisbelow(True)
    ax.tick_params(axis='both', direction='in', length=4, width=1)
    ax.minorticks_on()
    ax.grid(which='major', axis='y', linestyle='--', alpha=0.35, linewidth=0.8)
    ax.grid(which='minor', axis='y', linestyle=':', alpha=0.15, linewidth=0.5)

    for spine in ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(1.2)

    legend_elements = [
        Patch(facecolor=blue_rgba, edgecolor="black", label='Raw data (no adjustment)'),
        Patch(facecolor=green_rgba, edgecolor="black", label='Adjusted (multiple regression)'),
        mpl.lines.Line2D([0], [0], marker='D', linestyle='None', markersize=6, markerfacecolor='red', markeredgecolor='black', label='Reported EF')
    ]
    leg = ax.legend(handles=legend_elements, loc='upper right', frameon=True)
    leg.get_frame().set_edgecolor("black")
    leg.get_frame().set_linewidth(1.2)
    leg.get_frame().set_facecolor("white")

    plt.tight_layout()
    plt.show()

#Fig 6
def energyTech(year=2023):
    base_font = 14
    mpl.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman"],
        "font.size": base_font, "font.weight": "bold",
        "axes.titlesize": base_font + 3, "axes.labelsize": base_font + 1,
        "xtick.labelsize": base_font - 1, "ytick.labelsize": base_font - 1,
        "axes.linewidth": 1.0, "figure.dpi": 150
    })

    techNames = ['EO','FV','BIO','CE','CC','CI','GEO','HE','NU','TC','TG']
    techNames_english = [
        'Wind', 'Solar PV', 'Biomass', 'Coal', 'Combined Cycle', 'Internal Combustion',
        'Geothermal', 'Hydroelectric', 'Nuclear', 'Conventional Thermal', 'Gas Turbine'
    ]

    year_short = str(year)[-2:]
    file_path = f"Data/CompleteGeneration/completeBrute-{year_short}.csv"
    
    try:
        generation = pd.read_csv(file_path)
        generacion_por_tech = {}
        
        for i, tech in enumerate(techNames):
            if tech in generation.columns:
                generacion_por_tech[techNames_english[i]] = generation[tech].sum() / 1000  # GWh
        
        df = (pd.DataFrame.from_dict(generacion_por_tech, orient="index", columns=["GWh"])
              .sort_values("GWh", ascending=False))
        
        total = df["GWh"].sum()
        df["Pct"] = 100 * df["GWh"] / total
        
        umbral = 5.0
        principales = df[df["Pct"] > umbral].copy()
        minor = df[df["Pct"] <= umbral].copy()
        
        if len(minor) > 0:
            principales.loc["Other technologies"] = [minor["GWh"].sum(), minor["Pct"].sum()]
        
        labels = principales.index.tolist()
        values = principales["GWh"].values
        
        fig, ax = plt.subplots(figsize=(10.5, 7.5))
        colors = plt.cm.tab20(np.linspace(0, 1, len(labels)))
        if "Other technologies" in labels:
            colors[labels.index("Other technologies")] = (0.75, 0.75, 0.75, 1.0)
            
        wedges, _ = ax.pie(
            values,
            startangle=90,
            colors=colors,
            wedgeprops=dict(width=0.38, edgecolor="black", linewidth=1.0)
        )
        
        center_face = (0.92, 0.94, 0.98)  
        ax.add_artist(plt.Circle((0, 0), 0.58, facecolor=center_face, edgecolor="black", linewidth=1.0, zorder=10))
        
        ax.text(0, 0.03, f"{total:,.0f}", ha="center", va="center", fontsize=base_font + 4, fontweight="bold", zorder=20)
        ax.text(0, -0.10, "GWh", ha="center", va="center", fontsize=base_font + 1, fontweight="bold", zorder=20)
        ax.set_title(f"Energy Generation Share - {year}", pad=20)

        for w, lab, val in zip(wedges, labels, values):
            pct = 100 * val / total
            if pct < 2.0: continue

            ang = (w.theta2 + w.theta1) / 2.0
            x = np.cos(np.deg2rad(ang))
            y = np.sin(np.deg2rad(ang))

            xy = (0.92 * x, 0.92 * y)
            x_text = 1.35 * np.sign(x)
            y_text = 1.20 * y

            ax.annotate(
                f"{lab}\n{pct:.1f}%",
                xy=xy,
                xytext=(x_text, y_text),
                ha="left" if x >= 0 else "right",
                va="center",
                fontsize=base_font-2,
                fontweight="bold",
                arrowprops=dict(arrowstyle="-", lw=0.9, color="black", connectionstyle="angle3,angleA=0,angleB=90")
            )

        ax.axis("equal")
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error processing data for {year}: {e}")

#Fig 7
def capacityFactor(year1=2018, year2=2023):
    tech_names = ['EO','FV','BIO','CE','CC','CI','GEO','HE','NU','TC','TG']
    cap_df = pd.read_csv('Data/InstalledCapacity2018-2023.csv')
    
    fc1 = calculateFC(year1, cap_df, tech_names)
    fc2 = calculateFC(year2, cap_df, tech_names)

    name = 'CC'
    figName = 'Combined Cycle'

    plt.figure(figsize=(14, 6))
    plt.plot(fc2[name], color='blue', alpha=0.8, linewidth=1, label=f'{year2} - ' + figName)
    plt.plot(fc1[name], color='#4CAF50', alpha=0.8, linewidth=1, label=f'{year1} - ' + figName)
    plt.axhline(y=0.35, color='red', linestyle='--', linewidth=1.5, label='Capacity factor limit 35%')
    
    plt.xlabel('Hours of the year')
    plt.ylabel('Capacity factor [p.u.]')
    plt.title(f'Hourly Capacity Factor Comparison: CC ({year1} vs {year2})')
    plt.xlim(0, len(fc2['CC']))
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

#Fig 8
def capacityFactorDistribution(year1=2018, year2=2023):
    tech_names = ['EO','FV','BIO','CE','CC','CI','GEO','HE','NU','TC','TG']
    cap_df = pd.read_csv('Data/InstalledCapacity2018-2023.csv')
    
    name = 'CC'
    figName = 'Combined Cycle'

    fc1 = calculateFC(year1, cap_df, tech_names)[name].dropna()
    fc2 = calculateFC(year2, cap_df, tech_names)[name].dropna()

    print(f"\n{'='*70}\nCOMBINED CYCLE ANALYSIS: {year1} vs {year2}\n{'='*70}")
    stats_rows = [
        ('Mean', fc1.mean(), fc2.mean()),
        ('Median', fc1.median(), fc2.median()),
        ('Max', fc1.max(), fc2.max()),
        ('Min', fc1.min(), fc2.min()),
        ('Std Dev', fc1.std(), fc2.std())
    ]
    
    print(f"{'Statistic':<20} {year1:<12} {year2:<12} {'Diff':<12} {'Change %':<12}")
    print("-" * 70)
    for name, v1, v2 in stats_rows:
        diff = v2 - v1
        pct = (diff / v1 * 100) if v1 != 0 else 0
        print(f"{name:<20} {v1:.4f}       {v2:.4f}       {diff:+.4f}      {pct:+.1f}%")

    """ fc1 = fc1[fc1 >= 0.1]
    fc2 = fc2[fc2 >= 0.1] """

    plt.figure(figsize=(12, 6))
    plt.hist(fc1, bins=50, alpha=0.6, label=str(year1), color='#4CAF50', density=True, edgecolor='black')
    plt.hist(fc2, bins=50, alpha=0.6, label=str(year2), color='blue', density=True, edgecolor='black')
    plt.axvline(fc1.mean(), color='#2E7D32', linestyle='--', label=f'Avg {year1}', linewidth=2)
    plt.axvline(fc2.mean(), color='darkblue', linestyle='--', label=f'Avg {year2}', linewidth=2)
    
    plt.title('Capacity Factor Distribution - ' + figName)
    plt.xlabel('Capacity Factor [p.u.]')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

#Fig 9

#Fig 10
def emissionFactorForecast():
    base_font = 14
    mpl.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman"],
        "font.size": base_font, "font.weight": "bold",
        "axes.titlesize": base_font + 2, "axes.labelsize": base_font + 1,
        "xtick.labelsize": base_font - 1, "ytick.labelsize": base_font - 1,
        "axes.linewidth": 1.2, "legend.fontsize": base_font - 1
    })

    years = [str(y) for y in range(2017, 2025)]
    combined_data = []

    for year in years:
        file_path = f"Data/EmissionFactors/EF-MA-Adj/EF-MA-Adj-{year[2:]}.csv"
        try:
            df_year = pd.read_csv(file_path)
            combined_data.append(df_year)
        except FileNotFoundError:
            print(f"Warning: File {file_path} missing. Skipping from master compilation.")
            continue

    if not combined_data:
        print("Error: No individual EF-MA-Adj files found to generate the consolidated file.")
        return

    master_df = pd.concat(combined_data, axis=0).reset_index(drop=True)
    master_file_path = f"Data/MovingAverageCompleteAdjusted.csv"
    master_df.to_csv(master_file_path, index=False)

    data_raw = pd.read_csv(master_file_path)

    TARGET_COL = "EFe" if "EFe" in data_raw.columns else data_raw.columns[-1]
    TRAIN_FRACTION = 0.9
    
    y = data_raw[TARGET_COL].astype(float).reset_index(drop=True)
    n = len(y)
    k = np.arange(1, n + 1)

    data = pd.DataFrame({"Emissions": y, "k": k})
    LAGS = [3, 6, 12, 24]
    for lag in LAGS:
        data[f"lag_{lag}"] = data["Emissions"].shift(lag)
        
    data = data.dropna().reset_index(drop=True)

    split_idx = int(len(data) * TRAIN_FRACTION)
    train_df = data.iloc[:split_idx]
    test_df = data.iloc[split_idx:]

    y_train = train_df["Emissions"].values
    y_test = test_df["Emissions"].values

    feature_cols = [c for c in data.columns if c != "Emissions"]
    X_train = train_df[feature_cols].values
    X_test = test_df[feature_cols].values

    model = XGBRegressor(
        n_estimators=800,
        learning_rate=0.01,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    offset = y_train[-1] - y_pred[0]
    y_pred = y_pred + offset

    MSE = np.mean(np.square(y_test - y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mask = y_test != 0
    mape = np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100

    fig, ax = plt.subplots(figsize=(12, 6))
    
    k_all = data["k"].to_numpy()
    k_test = test_df["k"].to_numpy()

    ax.plot(k_all, data["Emissions"], color="#0072B2", label="Real Data", linewidth=1.8)
    ax.plot(k_test, y_pred, linestyle=":", color="#E69F00", linewidth=2.2, label="XGBoost Forecast")
    ax.axvline(k_test[0], linestyle="--", color="gray", linewidth=1.2, label="Train/Forecast split")
    
    ax.set_xlabel("Time [hour]")
    ax.set_ylabel(r"Emission factor [tCO$_2$e/MWh]")
    ax.set_title("Emission Factor Forecasting via XGBoost Method")
    ax.set_axisbelow(True)
    ax.tick_params(direction="in", length=4, width=1)
    ax.minorticks_on()
    ax.grid(which='major', axis='both', linestyle='--', alpha=0.35, linewidth=0.8)
    ax.grid(which='minor', axis='both', linestyle=':', alpha=0.15, linewidth=0.5)
    
    leg = ax.legend(loc="upper right", frameon=True)
    leg.get_frame().set_edgecolor("black")
    leg.get_frame().set_linewidth(1.2)

    for spine in ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(1.2)

    plt.tight_layout()
    plt.show()