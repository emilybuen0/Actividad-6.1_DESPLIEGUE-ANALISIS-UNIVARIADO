###############################################
# Importamos librerías
###############################################
import streamlit as st 
import plotly.express as px
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title='Dashboard Wuppi 🎮', layout="wide")

@st.cache_resource
def load_data():
    df = pd.read_csv("DataAnalytics.csv")
    df['Usuario'] = df['Usuario'].str.strip().str.upper()
    df = df.bfill().ffill()  # Limpiar nulos
    Lista = ['color presionado', 'mini juego', 'dificultad', 'Juego']
    Usuario = df['Usuario'].unique().tolist()
    return df, Lista, Usuario

# Cargar datos
df, Lista, Usuario = load_data()

# Sidebar imagen y selección de vista
st.sidebar.image("Wsidebar1.jpg", width=200)
st.sidebar.title("ANÁLISIS UNIVARIADO WUPPI 🔍")

View = st.sidebar.selectbox(
    label="Tipo de Análisis", 
    options=["Extracción de Características 📝", "Regresión Lineal Simple 📈👤", "Regresión Lineal Múltiple 📉👤" , "Regresión No Lineal", "Regresión Logistica", "ANOVA"]
)

# Vista 1: Extracción de características
if View == "Extracción de Características 📝":
    Variable_Cat = st.sidebar.selectbox(label="Variable a analizar", options=Lista)
    usuarios_seleccionados = st.sidebar.multiselect("Selecciona hasta 4 usuarios:", options=Usuario, max_selections=4)

    st.title("Extracción de Características por Usuario 📝")
    st.subheader("Variable seleccionada: " + Variable_Cat)

    if usuarios_seleccionados:
        df_filtrado = df[df['Usuario'].isin(usuarios_seleccionados)]

        filas = [usuarios_seleccionados[i:i+2] for i in range(0, len(usuarios_seleccionados), 2)]
        for fila in filas:
            columnas = st.columns(len(fila))
            for idx, usuario in enumerate(fila):
                with columnas[idx]:
                    df_usuario = df_filtrado[df_filtrado['Usuario'] == usuario]
                    st.markdown(f"**Usuario: {usuario}**")
                    tabla = df_usuario[Variable_Cat].value_counts().reset_index()
                    tabla.columns = ['categorias', 'frecuencia']

                    if Variable_Cat == 'color presionado':
                        fig = px.bar(tabla, x='categorias', y='frecuencia', title=f"{Variable_Cat} - {usuario}",
                                     color='categorias', color_discrete_map={color: color for color in tabla['categorias']})
                        st.plotly_chart(fig, use_container_width=True)

                    elif Variable_Cat == 'mini juego':
                        fig = px.pie(tabla, names='categorias', values='frecuencia', hole=0.4, title=f"{Variable_Cat} - {usuario}")
                        st.plotly_chart(fig, use_container_width=True)

                    elif Variable_Cat == 'dificultad':
                        fig = px.pie(tabla, names='categorias', values='frecuencia', title=f"{Variable_Cat} - {usuario}")
                        st.plotly_chart(fig, use_container_width=True)

                    elif Variable_Cat == 'Juego':
                        fig = px.scatter(tabla, x='categorias', y='frecuencia', size='frecuencia',
                                         title=f"Juegos más utilizados - {usuario}", color='categorias')
                        fig.update_layout(height=350)
                        st.plotly_chart(fig, use_container_width=True)

        mostrar_numericas = st.checkbox("**¿Mostrar variables numéricas por usuario?**", key="num_checkbox")
        if mostrar_numericas:
            st.subheader("Tablas de variables numéricas por usuario")
            filas_tablas = [usuarios_seleccionados[i:i+2] for i in range(0, len(usuarios_seleccionados), 2)]
            for fila_tablas in filas_tablas:
                columnas_tablas = st.columns(len(fila_tablas))
                for i, usuario in enumerate(fila_tablas):
                    with columnas_tablas[i]:
                        df_usuario = df[df['Usuario'] == usuario]
                        df_numericas = df_usuario.select_dtypes(include='number')
                        if not df_numericas.empty:
                            st.markdown(f"**{usuario}**")
                            st.dataframe(df_numericas)
    else:
        st.warning("Por favor selecciona al menos un usuario para visualizar los análisis.")


#############################################################
#############################################################
#PARTE 2

##############################################
# VISTA 2 - REGRESIÓN LINEAL  o MÚLTIPLE 
##############################################
# Vista: Regresión Lineal Simple y Heatmap

from sklearn.linear_model import LinearRegression
if View == "Regresión Lineal Simple 📈👤":
    st.title("Regresión Lineal por Usuario 📈👤")
    from sklearn.linear_model import LinearRegression
    df = pd.read_csv("DataAnalytics_color_minijuego_dificultad_juego_encoded.csv")
    df['Usuario'] = df['Usuario'].str.strip().str.upper()
    df = df.bfill().ffill()

    usuarios_regresion = df["Usuario"].astype(str).str.strip().str.upper().dropna().unique().tolist()
    usuarios_regresion.sort()

    Lista_numericas = [
        'botón correcto', 'tiempo de interacción', 'número de interacción', 'auto push',
        'tiempo de lección', 'tiempo de sesión', 'color presionado', 'mini juego', 'dificultad', 'Juego'
    ]
    Lista_numericas = [col for col in Lista_numericas if col in df.columns]

    tab_simple, tab_heatmap = st.tabs(["Regresión Lineal Simple 📈", "Heatmap 🔥"])

    with tab_simple:
        st.subheader("Comparación entre Usuarios - Regresión Lineal Simple")

        with st.sidebar:
            with st.expander("📈 Parámetros - Regresión Simple"):
                usuario_a = st.selectbox("Usuario A (Simple)", options=usuarios_regresion, key="usuario_a_simple")
                usuario_b = st.selectbox("Usuario B (Simple)", options=usuarios_regresion, key="usuario_b_simple")

                disponibles_a = [col for col in Lista_numericas if col in df.columns and df[df['Usuario'] == usuario_a][col].sum() > 0]
                disponibles_b = [col for col in Lista_numericas if col in df.columns and df[df['Usuario'] == usuario_b][col].sum() > 0]
                comunes = list(set(disponibles_a).intersection(disponibles_b))
                comunes.sort()

                Variable_y = st.selectbox("Variable Objetivo (Y)", options=comunes, key="y_simple")
                Variable_x = st.selectbox("Variable Independiente (X)", options=comunes, key="x_simple")

        df_a = df[df["Usuario"].str.strip().str.upper() == usuario_a]
        df_b = df[df["Usuario"].str.strip().str.upper() == usuario_b]

        Contenedor_A, Contenedor_B = st.columns(2)

        with Contenedor_A:
            st.markdown(f"**Usuario A: {usuario_a}**")
            if Variable_x in df_a.columns and Variable_y in df_a.columns:
                if not df_a[[Variable_x, Variable_y]].isnull().values.any():
                    model_a = LinearRegression()
                    model_a.fit(df_a[[Variable_x]], df_a[Variable_y])
                    r2_a = model_a.score(df_a[[Variable_x]], df_a[Variable_y])
                    r_a = np.sqrt(r2_a)
                    st.metric(label="Correlación", value=round(r_a, 4))
                    fig_a = px.scatter(df_a, x=Variable_x, y=Variable_y,
                                       title=f"Regresión Simple - {usuario_a}",
                                       labels={'x': Variable_x, 'y': Variable_y})
                    fig_a.update_layout(
                        paper_bgcolor='black',
                        plot_bgcolor='black',
                        font=dict(color='white'),
                        margin=dict(t=50, l=20, r=20, b=30),
                        height=400,
                        xaxis=dict(title_font=dict(color='white'), tickfont=dict(color='white')),
                        yaxis=dict(title_font=dict(color='white'), tickfont=dict(color='white')),
                        legend=dict(title_font=dict(color='white'), font=dict(color='white')),
                    )
                    st.plotly_chart(fig_a, key=f"plot_simple_a_{usuario_a}")
                    st.caption(f"Número de datos analizados: {len(df_a)}")
                else:
                    st.warning("Datos nulos para Usuario A")
            else:
                st.warning("Una o ambas variables seleccionadas no existen en los datos de Usuario A")

        with Contenedor_B:
            st.markdown(f"**Usuario B: {usuario_b}**")
            if Variable_x in df_b.columns and Variable_y in df_b.columns:
                if not df_b[[Variable_x, Variable_y]].isnull().values.any():
                    model_b = LinearRegression()
                    model_b.fit(df_b[[Variable_x]], df_b[Variable_y])
                    r2_b = model_b.score(df_b[[Variable_x]], df_b[Variable_y])
                    r_b = np.sqrt(r2_b)
                    st.metric(label="Correlación", value=round(r_b, 4))
                    fig_b = px.scatter(df_b, x=Variable_x, y=Variable_y,
                                       title=f"Regresión Simple - {usuario_b}",
                                       labels={'x': Variable_x, 'y': Variable_y})
                    fig_b.update_layout(
                        paper_bgcolor='black',
                        plot_bgcolor='black',
                        font=dict(color='white'),
                        margin=dict(t=50, l=20, r=20, b=30),
                        height=400,
                        xaxis=dict(title_font=dict(color='white'), tickfont=dict(color='white')),
                        yaxis=dict(title_font=dict(color='white'), tickfont=dict(color='white')),
                        legend=dict(title_font=dict(color='white'), font=dict(color='white')),
                    )
                    st.plotly_chart(fig_b, key=f"plot_simple_b_{usuario_b}")
                    st.caption(f"Número de datos analizados: {len(df_b)}")
                else:
                    st.warning("Datos nulos para Usuario B")
            else:
                st.warning("Una o ambas variables seleccionadas no existen en los datos de Usuario B")
        st.markdown("""
            ℹ️ Codificación de variables:
         - **'Juego'**: representa los episodios como valores numéricos.
             - 0 = Astro
            - 1 = Cadetes
            - **'dificultad'**: niveles representados numéricamente:
            - 0 = Episodio 1
             - 1 = Episodio 2
            - 2 = Episodio 3
            - 3 = Episodio 4
            """)
    # TAB 2: HEATMAP
    with tab_heatmap:
        st.subheader("Mapa de Calor de Correlaciones")

        with st.sidebar:
            with st.expander("Parámetros - Heatmap 🔥"):
                usuario_a_h = st.selectbox("Usuario A (Heatmap)", options=usuarios_regresion, key="usuario_a_heat")
                usuario_b_h = st.selectbox("Usuario B (Heatmap)", options=usuarios_regresion, key="usuario_b_heat")

        df_a_h = df[df["Usuario"].str.strip().str.upper() == usuario_a_h]
        df_b_h = df[df["Usuario"].str.strip().str.upper() == usuario_b_h]

        st.markdown("### Heatmap General del Dataset 🔥")
        valid_cols_all = [col for col in Lista_numericas if col in df.columns and df[col].nunique() > 1 and not df[col].isna().all()]
        if valid_cols_all:
            corr_matrix_all = df[valid_cols_all].corr()
            fig_all = px.imshow(corr_matrix_all, text_auto=True, color_continuous_scale='Viridis', aspect='auto')
            fig_all.update_layout(
                paper_bgcolor='black',
                plot_bgcolor='black',
                font=dict(color='white'),
                margin=dict(t=50, l=20, r=20, b=30),
                height=500
            )
            st.plotly_chart(fig_all, use_container_width=True, key="heatmap_general")
        else:
            st.info("No hay suficientes variables numéricas para el heatmap general.")

        col_h1, col_h2 = st.columns(2)
        with col_h1:
            st.markdown(f"### Heatmap de Usuario A 🔥 - {usuario_a_h}")
            valid_cols_a = [col for col in Lista_numericas if col in df_a_h.columns and df_a_h[col].nunique() > 1 and not df_a_h[col].isna().all()]
            if valid_cols_a:
                corr_matrix_a = df_a_h[valid_cols_a].corr()
                fig_corr_a = px.imshow(corr_matrix_a, text_auto=True, color_continuous_scale='Viridis', aspect='auto')
                fig_corr_a.update_layout(
                    paper_bgcolor='black',
                    plot_bgcolor='black',
                    font=dict(color='white'),
                    margin=dict(t=50, l=20, r=20, b=30),
                    height=500
                )
                st.plotly_chart(fig_corr_a, use_container_width=True, key=f"heatmap_a_{usuario_a_h}")
            else:
                st.info("No hay suficientes variables numéricas válidas para mostrar el heatmap de Usuario A.")

        with col_h2:
            st.markdown(f"### Heatmap de Usuario B 🔥 - {usuario_b_h}")
            valid_cols_b = [col for col in Lista_numericas if col in df_b_h.columns and df_b_h[col].nunique() > 1 and not df_b_h[col].isna().all()]
            if valid_cols_b:
                corr_matrix_b = df_b_h[valid_cols_b].corr()
                fig_corr_b = px.imshow(corr_matrix_b, text_auto=True, color_continuous_scale='Viridis', aspect='auto')
                fig_corr_b.update_layout(
                    paper_bgcolor='black',
                    plot_bgcolor='black',
                    font=dict(color='white'),
                    margin=dict(t=50, l=20, r=20, b=30),
                    height=500
                )
                st.plotly_chart(fig_corr_b, use_container_width=True, key=f"heatmap_b_{usuario_b_h}")
            else:
                st.info("No hay suficientes variables numéricas válidas para mostrar el heatmap de Usuario B.")


#######################################################
# VISTA 3 - REGRESIÓN LINEAL MÚLTIPLE
#######################################################
from sklearn.linear_model import LinearRegression

if View == "Regresión Lineal Múltiple 📉👤":
    st.title("Regresión Lineal Múltiple por Usuario 📉👤")
    # Cargar el CSV codificado con los valores numéricos para color y minijuego
    df = pd.read_csv("DataAnalytics_color_minijuego_dificultad_juego_encoded.csv")
    df['Usuario'] = df['Usuario'].str.strip().str.upper()
    df = df.bfill().ffill()

    usuarios_regresion = df["Usuario"].astype(str).str.strip().str.upper().dropna().unique().tolist()
    usuarios_regresion.sort()

    # Lista manual de variables numéricas actualizadas (valores codificados)
    Lista_numericas = [
        'botón correcto', 'tiempo de interacción', 'número de interacción', 'auto push',
        'tiempo de lección', 'tiempo de sesión', 'color presionado', 'mini juego', 'dificultad', 'Juego'
    ]
    Lista_numericas = [col for col in Lista_numericas if col in df.columns]

    st.subheader("Comparación entre Usuarios - Regresión Lineal Múltiple")

    with st.sidebar:
        st.markdown("### Parámetros - Regresión Múltiple")
        usuario_a_m = st.selectbox("Usuario A (Múltiple)", options=usuarios_regresion, key="usuario_a_multiple")
        usuario_b_m = st.selectbox("Usuario B (Múltiple)", options=usuarios_regresion, key="usuario_b_multiple")
        Variable_y_m = st.selectbox("Variable Objetivo (Y)", options=Lista_numericas, key="y_multiple")
        Variables_x_m = st.multiselect("Variables Independientes (X)", options=Lista_numericas, key="x_multiple")

    df_a_m = df[df["Usuario"].str.strip().str.upper() == usuario_a_m]
    df_b_m = df[df["Usuario"].str.strip().str.upper() == usuario_b_m]

    Contenedor_A_m, Contenedor_B_m = st.columns(2)

    with Contenedor_A_m:
        st.markdown(f"**Usuario A: {usuario_a_m}**")
        if Variables_x_m and Variable_y_m in df_a_m.columns and all(var in df_a_m.columns for var in Variables_x_m):
            if not df_a_m[Variables_x_m + [Variable_y_m]].isnull().values.any():
                model_a_m = LinearRegression()
                model_a_m.fit(df_a_m[Variables_x_m], df_a_m[Variable_y_m])
                r2_a_m = model_a_m.score(df_a_m[Variables_x_m], df_a_m[Variable_y_m])
                r_a_m = np.sqrt(r2_a_m)
                st.metric(label="Correlación Múltiple", value=round(r_a_m, 4))
                fig_a_m = px.scatter(df_a_m, x=Variables_x_m[0], y=Variable_y_m,
                                     title=f"Regresión Múltiple - {usuario_a_m}")
                fig_a_m.update_layout(
                    xaxis_title="Variables Independientes (x)",
                    yaxis_title=Variable_y_m,
                    paper_bgcolor='black',
                    plot_bgcolor='black',
                    font=dict(color='white'),
                    margin=dict(t=50, l=20, r=20, b=30),
                    height=400,
                    xaxis=dict(title_font=dict(color='white'), tickfont=dict(color='white')),
                    yaxis=dict(title_font=dict(color='white'), tickfont=dict(color='white')),
                    legend=dict(title_font=dict(color='white'), font=dict(color='white')),)
                st.plotly_chart(fig_a_m, key=f"plot_multiple_a_{usuario_a_m}")
                st.caption(f"Número de datos analizados: {len(df_a_m)}")
            else:
                st.warning("Datos insuficientes para Usuario A")
        else:
            st.warning("Una o más variables seleccionadas no existen para Usuario A")

    with Contenedor_B_m:
        st.markdown(f"**Usuario B: {usuario_b_m}**")
        if Variables_x_m and Variable_y_m in df_b_m.columns and all(var in df_b_m.columns for var in Variables_x_m):
            if not df_b_m[Variables_x_m + [Variable_y_m]].isnull().values.any():
                model_b_m = LinearRegression()
                model_b_m.fit(df_b_m[Variables_x_m], df_b_m[Variable_y_m])
                r2_b_m = model_b_m.score(df_b_m[Variables_x_m], df_b_m[Variable_y_m])
                r_b_m = np.sqrt(r2_b_m)
                st.metric(label="Correlación Múltiple", value=round(r_b_m, 4))
                fig_b_m = px.scatter(df_b_m, x=Variables_x_m[0], y=Variable_y_m,
                                     title=f"Regresión Múltiple - {usuario_b_m}")
                fig_b_m.update_layout(
                    xaxis_title="Variables Independientes (x)",
                    yaxis_title=Variable_y_m,
                    paper_bgcolor='black',
                    plot_bgcolor='black',
                    font=dict(color='white'),
                    margin=dict(t=50, l=20, r=20, b=30),
                    height=400,
                    xaxis=dict(title_font=dict(color='white'), tickfont=dict(color='white')),
                    yaxis=dict(title_font=dict(color='white'), tickfont=dict(color='white')),
                    legend=dict(title_font=dict(color='white'), font=dict(color='white')),)
                st.plotly_chart(fig_b_m, key=f"plot_multiple_b_{usuario_b_m}")
                st.caption(f"Número de datos analizados: {len(df_b_m)}")
            else:
                st.warning("Datos insuficientes para Usuario B")
        else:
            st.warning("Una o más variables seleccionadas no existen para Usuario B")
        # Codificación de variables al final
    st.markdown("""
        ℹ️ Codificación de variables:
        - **'Juego'**: representa los episodios como valores numéricos.
          - 0 = Astro
          - 1 = Cadetes
        - **'dificultad'**: niveles representados numéricamente:
          - 0 = Episodio 1
          - 1 = Episodio 2
          - 2 = Episodio 3
          - 3 = Episodio 4
    """)



###### DETERMINACION DE ESTILOS Y FORMATO
st.markdown("""
    <style>
        /* Fondo blanco para el main */
        .main {
            background-color: #0F1116;
        }
        
        /* Fondo con imagen para el sidebar */
        section[data-testid="stSidebar"] {
            background-image: url("https://img.freepik.com/vector-gratis/fondo-nube-turquesa_91008-163.jpg?semt=ais_hybrid&w=740");
            background-size: cover;
        }

        /* Texto blanco en sidebar */
        section[data-testid="stSidebar"] * {
            color: white;
        }

        /* Texto blanco dentro de selectbox/multiselect en sidebar */
        section[data-testid="stSidebar"] .css-1d391kg, 
        section[data-testid="stSidebar"] .css-1cpxqw2, 
        section[data-testid="stSidebar"] .css-1c2gdn6 {
            color: white !important;
        }
    </style>
""", unsafe_allow_html=True)





