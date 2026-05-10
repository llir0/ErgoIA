"""Panel Streamlit de ErgoIA para historial, resumen y registro de hidratacion."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st

from config import CONFIG
from storage import CsvStorage
from utils import format_duration


st.set_page_config(page_title="ErgoIA", layout="wide")

storage = CsvStorage(CONFIG.history_path, CONFIG.hydration_log_path)

st.title("ErgoIA")
st.caption("Panel local para revisar alertas, registrar hidratacion y consultar la configuracion.")

col_a, col_b, col_c = st.columns(3)

alerts_df = storage.read_alerts()
hydration_df = storage.read_hydration()
last_hydration = storage.last_hydration_datetime()

with col_a:
    st.metric("Alertas registradas", len(alerts_df))
with col_b:
    if last_hydration:
        elapsed = (datetime.now() - last_hydration).total_seconds()
        st.metric("Tiempo sin registrar agua", format_duration(elapsed))
    else:
        st.metric("Tiempo sin registrar agua", "Sin datos")
with col_c:
    if alerts_df.empty:
        st.metric("Tipo mas frecuente", "Sin alertas")
    else:
        st.metric("Tipo mas frecuente", alerts_df["tipo_alerta"].mode().iloc[0])

st.divider()

left, right = st.columns([1, 2])

with left:
    st.subheader("Hidratacion")
    if st.button("Registrar que tome agua", use_container_width=True):
        storage.log_hydration()
        st.success("Hidratacion registrada.")
        st.rerun()

    st.subheader("Tiempos configurados")
    st.write(f"Mala postura: **{CONFIG.bad_posture_seconds} segundos**")
    st.write(f"Pausa activa: **{CONFIG.sitting_break_minutes} minutos**")
    st.write(f"Hidratacion: **{CONFIG.hydration_minutes} minutos**")
    st.info("Para cambiar estos valores de forma permanente, edita config.py.")

with right:
    st.subheader("Historial de alertas")
    if alerts_df.empty:
        st.write("Todavia no hay alertas registradas.")
    else:
        st.dataframe(alerts_df.sort_index(ascending=False), use_container_width=True)

st.divider()

tab1, tab2 = st.tabs(["Resumen", "Hidratacion"])

with tab1:
    if alerts_df.empty:
        st.write("Ejecuta `python infer.py --source 0` para empezar una sesion.")
    else:
        grouped = alerts_df.groupby("tipo_alerta").size().reset_index(name="cantidad")
        st.bar_chart(grouped, x="tipo_alerta", y="cantidad")
        total_duration = int(pd.to_numeric(alerts_df["duracion_segundos"], errors="coerce").fillna(0).sum())
        st.write(f"Duracion acumulada asociada a alertas: **{format_duration(total_duration)}**")

with tab2:
    if hydration_df.empty:
        st.write("No hay registros de hidratacion.")
    else:
        st.dataframe(hydration_df.sort_index(ascending=False), use_container_width=True)
