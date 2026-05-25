import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import folium
import io
import base64
import webbrowser
import matplotlib.ticker as mticker


capacidades = pd.read_csv('Data/Map/CapacityCoordinates.csv', encoding='utf8')
df_gen = pd.read_csv('Data/CompleteGeneration/complete-24.csv', encoding='utf8')
emisiones = pd.read_csv('Data/Emissions/emissions-24.csv', encoding='utf8')

#data prep
gen = df_gen.iloc[:, 1:12] #generación MWh
tot_tecnologia = gen.sum() #generación total por tecnología
gen_tot=tot_tecnologia.sum(axis=0)
proporcion=tot_tecnologia/gen_tot

emisiones = emisiones.iloc[:, 1:12]
emisiones_anuales = emisiones.sum() #emisiones totales por tecnología

FE_adj = 0.4367 #tabla ajustado COPAR 2024
FE_x = 0.3427
FE = FE_adj/FE_x
print("Factor de Emisiones del Sistema (tCO2e/MWh):", FE_adj)
print('MAX BIO:', gen['BIO'].max())

#capacidad instalada por edo.
capacity_edo = (capacidades.groupby(['Estado', 'Tecnología 2'])['Capacidad_MW'].sum().reset_index()) #hacer dataframe con esta info
#capacidad por tecnología
cap_tot = capacidades.groupby(['Tecnología 2'])['Capacidad_MW'].sum()
#lsita de estados para ciclo
edos = capacity_edo['Estado'].unique()
#distinguir entre techs para f.e. y colores

fosiles = ['BIO', 'CE', 'CC', 'CI', 'TC', 'TG']
techs = gen.columns
limpias = [t for t in techs if t not in fosiles]

#aportación por tech por edo 
def operacion(capacity_edo, tot_tecnologia, emisiones_anuales, proporcion):
    resultados = [] #append resultados
    for edo in edos:
        df_edo = capacity_edo[capacity_edo['Estado'] == edo] #df edo tiene la info de cada estado para tener la info por estado iterado(esa info es capacidad y tecnología), lo acumulado esta en df_resultados
        for tecnologia in df_edo['Tecnología 2'].unique():

            df_edo_tech = df_edo[df_edo['Tecnología 2'] == tecnologia]['Capacidad_MW'].sum() #capacidad total para esa tech en ese edo.
            cap_total_tech = cap_tot.loc[tecnologia] #cap MW nacional para esa tech
            fraccion = df_edo_tech / cap_total_tech #proporción adimensional capacidad
            
            #filtar por tecnología
            prop_tech = proporcion.loc[tecnologia] #proporción de generación para esa tecnología
            gen_tech = tot_tecnologia.loc[tecnologia] #gen total para esa tecnología para ese estado
            # emisiones_tech = emisiones_anuales.loc[tecnologia] #total anual
            energia = fraccion * gen_tech #energía estimada por tech por edo

#METODOLOGÍA K:
            K=FE*prop_tech #new
            K_old = prop_tech
            emisiones_k = (energia*K*FE_adj)/K_old #emisiones con ese approach tco2e
            resultados.append({ 'Estado': edo,
            'Tecnología': tecnologia,
            'Capacidad(MW) Estatal': df_edo_tech,
            'Proporción Capacidad': fraccion,
            'Energía (MWh) estimada': energia,
            'tCO2e': emisiones_k,
            'valor K': K})
    df_resultados = pd.DataFrame(resultados)
#segregar por tipo de tecnología, limpias usan factor ajustado solamente, no K
    df_resultados['Limpia'] = ~df_resultados['Tecnología'].isin(fosiles) #boolean
    #info de emisiones evitadasa (<0) para hacer el pay
    df_resultados['Emisiones_Calculadas'] = df_resultados.apply( #apply aplica función para filas o columnas
        lambda row: row['tCO2e'] if not row['Limpia'] #argument : expresión, si la fila NO es limpia return resultados tco2e
        else -row['Energía (MWh) estimada'] * FE_adj, #si es limpia usar FE adj 
        axis=1) #filas axis 0 columnas
    return df_resultados
resultado = operacion(capacity_edo, tot_tecnologia, emisiones_anuales, proporcion)
#resultado.to_csv("resultados.csv")

#comparar resultados
k=(resultado.loc[~resultado['Limpia'], 'tCO2e'].sum()
    /gen_tot) #resultado.loc[~resultado['Limpia'], 'Energía (MWh) estimada'].sum())
print('FE_z:', k)

emisiones_k = resultado.loc[~resultado['Limpia'], 'tCO2e'].sum()
print('emisiones con factor k:', emisiones_k)

evitadas = resultado.loc[resultado['Limpia'], 'Emisiones_Calculadas'].sum()
print('emisiones evitadas:', evitadas)

emi_reales = tot_tecnologia.sum()*FE_adj
print('emisiones anuales:', emi_reales)

# K_tech = (resultado.groupby(['Tecnología'])['valor K'].unique().reset_index()) #hacer dataframe con esta info
# print('factor k por tecnología:', K_tech)

error = abs(k - FE_adj) / FE_adj * 100
error_emisiones = abs(emisiones_k - emi_reales) / emi_reales * 100

print('porcentaje de error factor:', error)
print('porcentaje de error emisiones:', error_emisiones)

#capitales de cada estado
capitales_mexico = {
    "Aguascalientes": (21.8853, -102.2916),
    "Baja California": (32.6245, -115.4523),
    "Baja California Sur": (24.1426, -110.3128),
    "Campeche": (19.8301, -90.5349),
    "Coahuila de Zaragoza": (26.90687, -101.42056),
    "Colima": (19.2452, -103.7241),
    "Chiapas": (16.7569, -93.1292),
    "Chihuahua": (28.6330, -106.0691),
    "Estado de México": (19.4326, -99.1332),
    "Durango": (24.0277, -104.6532),
    "Guanajuato": (21.0190, -101.2574),
    "Guerrero": (17.5514, -99.5008),
    "Hidalgo": (20.1011, -98.7624),
    "Jalisco": (20.6597, -103.3496),
    "México": (19.2826, -99.6557),
    "Michoacán": (19.7008, -101.1844),
    "Morelos": (18.9242, -99.2216),
    "Nayarit": (21.5040, -104.8946),
    "Nuevo León": (25.6866, -100.3161),
    "Oaxaca": (17.0732, -96.7266),
    "Puebla": (19.0414, -98.2063),
    "Querétaro": (20.5888, -100.3899),
    "Quintana Roo": (18.5081, -88.3038),
    "San Luis Potosí": (22.1565, -100.9855),
    "Sinaloa": (24.8059, -107.3944),
    "Sonora": (29.0730, -110.9559),
    "Tabasco": (17.9895, -92.9475),
    "Tamaulipas": (23.7369, -99.1411),
    "Tlaxcala": (19.3139, -98.2404),
    "Veracruz de Ignacio de la Llave": (19.5438, -96.9272),
    "Yucatán": (20.9674, -89.5926),
    "Zacatecas": (22.7709, -102.5830),
}

#diccionarios colores
paleta_fosil = {
    'CC':  "#888A85",
    'TC':  "#E1D7D7",   
    'TG':  "#000000",   
    'CI':  "#514444",   
    'CE':  "#4F8154",   
    'BIO': '#B59B6A'    
}

paleta_limpia = {
    'EO': '#1B9E77',
    'FV': "#E2E44C",
    'HE': "#4B88B7",
    'GEO': "#024B1F",
    'NU': "#0BA11A"}
palette_tecnologias = {**paleta_fosil, **paleta_limpia} #doble asterisco sirve para combianr los dos diccionarios

def Graficar_Mapa_Pastel(resultado, capital):
    #Configuración inicial del mapa
    m = folium.Map(location=[23.6345, -102.5528], zoom_start=5, tiles='CartoDB positron')
    
    resultado = resultado[~resultado['Limpia']].copy()

    #colores consistentes para las tecnologías
    colores = paleta_fosil

    #Iterar sobre cada estado
    for estado, (lat, lon) in capital.items():
        #Filtrar datos del estado actual        
        edo = resultado[resultado['Estado'] == estado]
        if edo.empty:
            continue

        #datos para normalizar tamaño pay
        total_emi = edo['tCO2e'].sum()
        max_emi = resultado['tCO2e'].max()
        tamaño_min = 25 #marcadores límites
        tamaño_max = 60
        if total_emi > 0 and max_emi > 0:
            tamaño = tamaño_min+(total_emi/max_emi)*(tamaño_max-tamaño_min)
        else:
            tamaño = tamaño_min
        
        #Datos para el grafico de pastel
        sizes = edo['tCO2e'].abs() #tamaño de cada porción del pastel
        labels = edo['Tecnología'] 
        #Mapear colores
        colores_edo = [colores[tech] for tech in labels]

        #Generar el icono a base del grafico de pastel (matplotlib a imagen Base64)
        fig, ax = plt.subplots(figsize=(1, 1)) #Tamaño del icono
        fig.patch.set_alpha(0) #Fondo transparente
        
        #Crear pastel sin etiquetas, solo colores
        ax.pie(sizes, colors=colores_edo, startangle=90)
        ax.axis('equal')
        
        #Guardar iconos
        buff = io.BytesIO()
        plt.savefig(buff, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
        plt.close(fig)
        buff.seek(0)
        
        #base64 con HTML
        b64_img = base64.b64encode(buff.read()).decode('utf-8')
        icon_url = f"data:image/png;base64,{b64_img}"
        
        #crear un popup (tabla de información)
        #Crear una tabla HTML simple para ver los detalles al dar clic
        tabla_html = edo[['Tecnología', 'tCO2e']].to_html(classes='table table-striped', index=False, float_format="%.2f")
        popup_html = f"<h4>{estado}</h4><p>Total: {total_emi:,.2f} tCO2e</p>{tabla_html}"
        
        #Agregar popup al mapa
        icon = folium.CustomIcon(icon_url, icon_size=(int(tamaño), int(tamaño)))
        
        folium.Marker(
            location=[lat, lon],
            icon=icon,
            tooltip=f"{estado}: {total_emi:,.0f} tCO2e", #Hover simple
            popup=folium.Popup(popup_html, max_width=300) #Click detallado
        ).add_to(m)

    #Agregar leyenda de colores 
    leyenda_html = '''
     <div style="position: fixed; bottom: 50px; left: 50px; width: 150px; height: auto; 
     border:2px solid grey; z-index:9999; font-size:12px; background-color:white; padding: 10px; opacity: 0.9;">
     <b>Technologies</b><br>
     '''
    for tech, color in colores.items():
        leyenda_html += f'<i style="background:{color};width:10px;height:10px;display:inline-block;"></i> {tech}<br>'
    leyenda_html += "</div>"
    m.get_root().html.add_child(folium.Element(leyenda_html))

    #Guardar y abrir
    output_file = 'Mapa_Pasteles_Mexico.html'
    m.save(output_file)
    print(f"Mapa guardado exitosamente en {output_file}")
    webbrowser.open(output_file)

Graficar_Mapa_Pastel(resultado, capitales_mexico)

#dataframae emisiones
df_totales = resultado.groupby('Tecnología').sum(numeric_only=True).reset_index()

df_totales['Limpia'] = ~df_totales['Tecnología'].isin(fosiles)
df_totales['emisiones_x'] = np.where(~df_totales['Limpia'], df_totales['Energía (MWh) estimada'] * FE_x, 0)
df_totales['emisiones_y'] = np.where(~df_totales['Limpia'], df_totales['Energía (MWh) estimada'] * FE_adj, 0)
df_totales['emisiones_z'] = np.where(~df_totales["Limpia"], df_totales['tCO2e'], 0)
df_totales['emisiones_evitadas'] = np.where(
    df_totales['Limpia'],
    df_totales['Emisiones_Calculadas']*-1, 0)

df_totales = df_totales[['Limpia', 'Tecnología', 'emisiones_x', 'emisiones_y', 'emisiones_z']]
#df_totales.to_csv("emisiones.csv", index=False)
# emisiones_z=df_totales['emisiones_z'].sum()
# print(emisiones_z)

#gráfica
def barras(df_totales, solo_fosil=True, figsize=(8,5)):
    if solo_fosil:
        df_totales = df_totales[~df_totales['Limpia']].copy()
    
    # graficar
    fig, ax = plt.subplots(figsize=figsize)

    df_totales.set_index("Tecnología")[[
        "emisiones_x",
        "emisiones_y",
        "emisiones_z"
    ]].plot(kind="bar", ax=ax)
    
    ax.set_ylabel("Emisiones (tCO2e)")
    ax.set_xlabel("Tecnología")

    ax.ticklabel_format(style='plain', axis='y')
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.0f}'))
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
barras(df_totales, solo_fosil=True, figsize=(8,5))