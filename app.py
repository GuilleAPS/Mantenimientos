import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.linear_model import LinearRegression
from datetime import timedelta, date

st.set_page_config(page_title="Dashboard Mantenimiento por Categoría", layout="wide")
st.title("📊 Dashboard Predictivo por Categoría de Mantenimiento")

# 📤 Subir archivo
archivo = st.file_uploader("Sube tu archivo Excel de mantenimiento", type=["xlsx"])

if archivo is not None:
    df = pd.read_excel(archivo)
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    df['Fecha_ordinal'] = df['Fecha'].map(pd.Timestamp.toordinal)

    vehiculos = df['Vehiculo'].unique()
    tipos_mantenimiento = df['Tipo'].unique()

    # 🔧 Intervalos personalizados
    intervalos_km = { "servicioc": 5000,  "serviciot": 10000, "llantas": 50000, "baterias": 50000 }

    # 📋 Resumen de mantenimientos futuros
    resumen = []
    for (vehiculo, tipo), sub_df in df.groupby(["Vehiculo", "Tipo"]):
        sub_df = sub_df.sort_values('Fecha')
        if len(sub_df) < 2:
            continue

        X = sub_df[['Fecha_ordinal']]
        y = sub_df['Km']
        modelo = LinearRegression().fit(X, y)

        coef = modelo.coef_[0]
        if coef == 0:
            continue

        ultimo = sub_df.iloc[-1]
        tipo_normalizado = tipo.strip().lower()
        km_intervalo = intervalos_km.get(tipo_normalizado, 3000)
        km_objetivo = ultimo['Km'] + km_intervalo
        fecha_ordinal = (km_objetivo - modelo.intercept_) / coef

        if np.isfinite(fecha_ordinal):
            try:
                fecha_estimado = pd.Timestamp.fromordinal(int(fecha_ordinal)).date()
                resumen.append({
                    "Vehiculo": vehiculo,
                    "Tipo": tipo,
                    "Última Fecha": ultimo["Fecha"].date(),
                    "Último KM": int(ultimo["Km"]),
                    "Próximo KM": int(km_objetivo),
                    "Próxima Fecha Estimada": fecha_estimado
                })
            except:
                continue

    if resumen:
        st.subheader("📌 Resumen de Próximos Mantenimientos")
        resumen_df = pd.DataFrame(resumen)

        def colorear_fechas(fila):
            fecha = fila["Próxima Fecha Estimada"]
            hoy = date.today()
            dias = (fecha - hoy).days
            if dias < 7:
                color = "background-color: red"
            elif dias < 30:
                color = "background-color: yellow"
            else:
                color = "background-color: lightgreen"
            return ["" if col != "Próxima Fecha Estimada" else color for col in fila.index]

        st.dataframe(resumen_df.style.apply(colorear_fechas, axis=1))

    # 🔽 Selección
    vehiculo_seleccionado = st.selectbox("Selecciona un vehículo", vehiculos)
    tipo_seleccionado = st.selectbox("Selecciona tipo de mantenimiento", tipos_mantenimiento)

    sub_df = df[
        (df['Vehiculo'] == vehiculo_seleccionado) & 
        (df['Tipo'] == tipo_seleccionado)
    ].sort_values('Fecha')

    if len(sub_df) < 2:
        st.warning("No hay suficientes registros para hacer predicción.")
    else:
        X = sub_df[['Fecha_ordinal']]
        y = sub_df['Km']
        modelo = LinearRegression().fit(X, y)
        sub_df['Km_predicho'] = modelo.predict(X)

        coef = modelo.coef_[0]
        if coef != 0:
            ultimo = sub_df.iloc[-1]
            tipo_normalizado = tipo_seleccionado.strip().lower()
            km_intervalo = intervalos_km.get(tipo_normalizado, 3000)
            km_objetivo = ultimo['Km'] + km_intervalo
            fecha_ordinal = (km_objetivo - modelo.intercept_) / coef

            if np.isfinite(fecha_ordinal):
                fecha_estimada = pd.Timestamp.fromordinal(int(fecha_ordinal))
                st.subheader("📅 Mantenimiento estimado")
                st.markdown(
                    f"""
                    - Último KM: **{int(ultimo['Km'])}**  
                    - Última fecha: **{ultimo['Fecha'].date()}**  
                    - Próximo mantenimiento estimado en **{int(km_objetivo)} km** (cada {km_intervalo} km)  
                    - Estimado para la fecha: **{fecha_estimada.date()}**
                    """
                )
        else:
            st.warning("El modelo no puede predecir con coeficiente cero.")

        st.subheader("📈 Gráfico de km vs fecha")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(sub_df['Fecha'], sub_df['Km'], marker='o', label='Real')
        ax.plot(sub_df['Fecha'], sub_df['Km_predicho'], linestyle='--', label='Predicho')
        ax.set_xlabel("Fecha")
        ax.set_ylabel("Kilometraje")
        ax.set_title(f"{vehiculo_seleccionado} - {tipo_seleccionado}")
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)

        st.subheader("📋 Datos utilizados")
        st.dataframe(sub_df)

else:
    st.info("Por favor, sube un archivo Excel con columnas: Vehiculo, Fecha, Km, Tipo, Costo.")

