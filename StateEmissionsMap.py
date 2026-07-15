import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import folium
import io
import base64
import webbrowser
import os

def generateEmissionsMap():
    base_font = 12
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman"],
        "font.size": base_font, "font.weight": "bold",
        "axes.linewidth": 1.2
    })

    capacidades = pd.read_csv('Data/Map/CapacityCoordinates.csv', encoding='utf8')
    df_gen = pd.read_csv('Data/CompleteGeneration/complete-23.csv', encoding='utf8')
    emisiones = pd.read_csv('Data/Emissions/emissions-23.csv', encoding='utf8')

    gen = df_gen.iloc[:, 1:12] 
    tot_tecnologia = gen.sum() 
    gen_tot = tot_tecnologia.sum(axis=0)
    proporcion = tot_tecnologia / gen_tot

    emisiones = emisiones.iloc[:, 1:12]
    emisiones_anuales = emisiones.sum()

    FE_adj = 0.4367 
    FE_x = 0.3427
    FE = FE_adj / FE_x

    capacity_edo = capacidades.groupby(['Estado', 'Technology'])['Capacidad_MW'].sum().reset_index()
    cap_tot = capacidades.groupby(['Technology'])['Capacidad_MW'].sum()
    edos = capacity_edo['Estado'].unique()

    fosiles = ['BIO', 'CE', 'CC', 'CI', 'TC', 'TG']
    techs = gen.columns

    resultados = [] 
    for edo in edos:
        df_edo = capacity_edo[capacity_edo['Estado'] == edo] 
        for tecnologia in df_edo['Technology'].unique():
            df_edo_tech = df_edo[df_edo['Technology'] == tecnologia]['Capacidad_MW'].sum() 
            cap_total_tech = cap_tot.loc[tecnologia] 
            fraccion = df_edo_tech / cap_total_tech 
            
            prop_tech = proporcion.loc[tecnologia] 
            gen_tech = tot_tecnologia.loc[tecnologia] 
            energia = fraccion * gen_tech 

            K = FE * prop_tech 
            K_old = prop_tech
            emisiones_k = (energia * K * FE_adj) / K_old 
            
            resultados.append({ 
                'Estado': edo,
                'Technology': tecnologia,
                'Capacidad(MW) Estatal': df_edo_tech,
                'Proporción Capacidad': fraccion,
                'Energía (MWh) estimada': energia,
                'tCO2e': emisiones_k,
                'valor K': K
            })
            
    df_resultados = pd.DataFrame(resultados)
    df_resultados['Limpia'] = ~df_resultados['Technology'].isin(fosiles) 
    
    df_resultados['Emisiones_Calculadas'] = df_resultados.apply( 
        lambda row: row['tCO2e'] if not row['Limpia'] 
        else -row['Energía (MWh) estimada'] * FE_adj, 
        axis=1
    )

    capitales_mexico = {
        "Aguascalientes": (21.8853, -102.2916), "Baja California": (32.6245, -115.4523),
        "Baja California Sur": (24.1426, -110.3128), "Campeche": (19.8301, -90.5349),
        "Coahuila de Zaragoza": (26.90687, -101.42056), "Colima": (19.2452, -103.7241),
        "Chiapas": (16.7569, -93.1292), "Chihuahua": (28.6330, -106.0691),
        "Estado de México": (19.4326, -99.1332), "Durango": (24.0277, -104.6532),
        "Guanajuato": (21.0190, -101.2574), "Guerrero": (17.5514, -99.5008),
        "Hidalgo": (20.1011, -98.7624), "Jalisco": (20.6597, -103.3496),
        "México": (19.2826, -99.6557), "Michoacán": (19.7008, -101.1844),
        "Morelos": (18.9242, -99.2216), "Nayarit": (21.5040, -104.8946),
        "Nuevo León": (25.6866, -100.3161), "Oaxaca": (17.0732, -96.7266),
        "Puebla": (19.0414, -98.2063), "Querétaro": (20.5888, -100.3899),
        "Quintana Roo": (18.5081, -88.3038), "San Luis Potosí": (22.1565, -100.9855),
        "Sinaloa": (24.8059, -107.3944), "Sonora": (29.0730, -110.9559),
        "Tabasco": (17.9895, -92.9475), "Tamaulipas": (23.7369, -99.1411),
        "Tlaxcala": (19.3139, -98.2404), "Veracruz de Ignacio de la Llave": (19.5438, -96.9272),
        "Yucatán": (20.9674, -89.5926), "Zacatecas": (22.7709, -102.5830)
    }

    paleta_fosil = {
        'CC': "#888A85", 'TC': "#E1D7D7", 'TG': "#000000",
        'CI': "#514444", 'CE': "#4F8154", 'BIO': '#B59B6A'
    }

    m = folium.Map(location=[23.6345, -102.5528], zoom_start=5, tiles='CartoDB positron')
    df_mapa = df_resultados[~df_resultados['Limpia']].copy()

    for estado, (lat, lon) in capitales_mexico.items():
        edo = df_mapa[df_mapa['Estado'] == estado]
        if edo.empty:
            continue

        total_emi = edo['tCO2e'].sum()
        max_emi = df_mapa['tCO2e'].max()
        tamaño_min, tamaño_max = 25, 60
        
        tamaño = tamaño_min + (total_emi / max_emi) * (tamaño_max - tamaño_min) if total_emi > 0 and max_emi > 0 else tamaño_min
        
        sizes = edo['tCO2e'].abs()
        labels = edo['Technology']
        colores_edo = [paleta_fosil[tech] for tech in labels]

        fig, ax = plt.subplots(figsize=(1, 1))
        fig.patch.set_alpha(0)
        ax.pie(sizes, colors=colores_edo, startangle=90)
        ax.axis('equal')
        
        buff = io.BytesIO()
        plt.savefig(buff, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
        plt.close(fig)
        buff.seek(0)
        
        b64_img = base64.b64encode(buff.read()).decode('utf-8')
        icon_url = f"data:image/png;base64,{b64_img}"
        
        df_popup = edo[['Technology', 'tCO2e']].rename(columns={'tCO2e': 'tCO<sub>2</sub>e'})
        tabla_html = df_popup.to_html(classes='table table-striped', index=False, float_format='{:,.2f}'.format, escape=False)
        popup_html = f"<h4>{estado}</h4><p>Total: {total_emi:,.2f} tCO<sub>2</sub>e</p>{tabla_html}"
        
        icon = folium.CustomIcon(icon_url, icon_size=(int(tamaño), int(tamaño)))
        folium.Marker(
            location=[lat, lon],
            icon=icon,
            tooltip=f"{estado}: {total_emi:,.0f} tCO<sub>2</sub>e",
            popup=folium.Popup(popup_html, max_width=300)
        ).add_to(m)

    leyenda_html = '''
     <div style="position: fixed; bottom: 50px; left: 50px; width: 150px; height: auto; 
     border:2px solid grey; z-index:9999; font-size:12px; background-color:white; padding: 10px; opacity: 0.9;">
     <b>Technologies</b><br>
     '''
    for tech, color in paleta_fosil.items():
        leyenda_html += f'<i style="background:{color};width:10px;height:10px;display:inline-block;"></i> {tech}<br>'
    
    leyenda_html += "<br><b>Units:</b> tCO<sub>2</sub>e"
    leyenda_html += "</div>"
    m.get_root().html.add_child(folium.Element(leyenda_html))

    output_file = 'StateEmissionsMap.html'
    m.save(output_file)
    print(f"Map saved succesfully in {output_file}")
    webbrowser.open(output_file)

    df_totales = df_resultados.groupby('Technology').sum(numeric_only=True).reset_index()
    df_totales['Limpia'] = ~df_totales['Technology'].isin(fosiles)
    df_totales['emisiones_x'] = np.where(~df_totales['Limpia'], df_totales['Energía (MWh) estimada'] * FE_x, 0)
    df_totales['emisiones_y'] = np.where(~df_totales['Limpia'], df_totales['Energía (MWh) estimada'] * FE_adj, 0)
    df_totales['emisiones_z'] = np.where(~df_totales["Limpia"], df_totales['tCO2e'], 0)
    
    df_totales = df_totales[~df_totales['Limpia']].copy()
    
"""     fig, ax = plt.subplots(figsize=(8, 5))
    df_totales.set_index("Technology")[["emisiones_x", "emisiones_y", "emisiones_z"]].plot(kind="bar", ax=ax)
    
    ax.set_ylabel("Emissions (tCO2e)")
    ax.set_xlabel("Technology")
    ax.ticklabel_format(style='plain', axis='y')
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.0f}'))
    
    ax.set_axisbelow(True)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    
    for spine in ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(1.2)
        
    plt.tight_layout()
    plt.show() """