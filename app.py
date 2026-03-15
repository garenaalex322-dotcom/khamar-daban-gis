import streamlit as st
import folium
from streamlit_folium import st_folium
import branca.colormap as cm
import rasterio
from rasterio.warp import transform_bounds, transform, reproject, Resampling
from rasterio.transform import rowcol
import numpy as np
import matplotlib.pyplot as plt
import io
import os
import base64
import joblib
import tempfile
import requests
import json

st.set_page_config(page_title="ГИС Хамар-Дабан", layout="wide", initial_sidebar_state="expanded")

# ВАШ КЛЮЧ (Не забудьте скрыть его в Streamlit Secrets перед публикацией на GitHub!)
OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]

# ==========================================
# ФУНКЦИЯ ДЛЯ АНАЛИЗА КАРТИНОК И ГРАФИКОВ (VISION)
# ==========================================
def analyze_with_vision(stats_text, img_b64, model, api_key):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    system_prompt = """Ты главный эксперт-лесопатолог. Твоя задача детально проанализировать предоставленную статистику и визуальный график (гистограмму RWI).
    Обрати внимание на форму распределения на графике (скошенное, нормальное, бимодальное). Свяжи форму графика со статистикой и сделай глубокий вывод о состоянии леса на этой территории."""

    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": stats_text},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
            ]}
        ]
    }

    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            return f"Ошибка API: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Сетевая ошибка: {e}"


# ==========================================
# 🤖 ИИ-АССИСТЕНТ (БОКОВАЯ ПАНЕЛЬ ЧАТА)
# ==========================================
st.sidebar.title("🤖 ИИ-Аналитик")

selected_model = st.sidebar.selectbox(
    "🧠 Выберите нейросеть:",
    options=["anthropic/claude-3.5-sonnet", "openai/gpt-4o", "google/gemini-1.5-pro"],
    format_func=lambda x: "Claude 3.5 Sonnet 🏆" if "claude" in x else ("GPT-4o ⚡" if "gpt" in x else "Gemini 1.5 Pro 🌐")
)

st.sidebar.markdown("💡 **Быстрые действия:**")
col1, col2 = st.sidebar.columns(2)
col3, col4 = st.sidebar.columns(2)

quick_prompt = None
if col1.button("📝 Отчет", use_container_width=True):
    quick_prompt = "Напиши строгий экспертный отчет (3 абзаца) о важности мониторинга индекса RWI на Хамар-Дабане."
if col2.button("💧 Про NDMI", use_container_width=True):
    quick_prompt = "Объясни понятным языком, почему для оценки стресса деревьев мы используем индекс NDMI, а не NDVI."
if col3.button("🔥 Риски", use_container_width=True):
    quick_prompt = "Какие риски (пожары, вредители) возникают, если RWI падает ниже 0.85?"
if col4.button("🌲 LST", use_container_width=True):
    quick_prompt = "Что такое индекс LST, как он связан с транспирацией дерева?"

st.sidebar.divider()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.sidebar.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.sidebar.chat_input("Задайте свой вопрос...")
prompt = quick_prompt if quick_prompt else user_input

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.sidebar.chat_message("user"):
        st.markdown(prompt)

    with st.sidebar.chat_message("assistant"):
        with st.spinner("⏳ Нейросеть думает..."):
            try:
                system_prompt = "Ты главный эксперт-лесопатолог. Анализируй данные модели RWI лесов Хамар-Дабана. RWI < 0.85 (сильный стресс), LST (температура поверхности), NDMI (влага в хвое)."
                headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
                api_messages = [{"role": "system", "content": system_prompt}] + st.session_state.messages
                data = {"model": selected_model, "messages": api_messages}

                response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)

                if response.status_code == 200:
                    ai_reply = response.json()["choices"][0]["message"]["content"]
                    st.session_state.messages.append({"role": "assistant", "content": ai_reply})
                    st.rerun()
                else:
                    st.error(f"Ошибка API: {response.status_code}")
            except Exception as e:
                st.error(f"Сетевая ошибка: {e}")

# ==========================================
# ОСНОВНОЙ ИНТЕРФЕЙС ГИС
# ==========================================
st.title("🌲 Пространственная ГИС-модель радиального прироста")

MODEL_PATH = 'true_spatial_baikal_rf_model.joblib'
MAPS_DIR = 'Final_RWI_Maps'
MASK_PATH = 'Khamar_Daban_Forest_Mask_30m.tif'
GEOJSON_PATH = 'baikal_reserve.geojson'
PIXEL_AREA_HA = 0.09  # 1 пиксель 30х30м = 900 м2 = 0.09 га


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    model_data = load_model()
    rf_model = model_data['model']
    imputer = model_data['imputer']
except FileNotFoundError:
    st.error(f"❌ Файл {MODEL_PATH} не найден! Убедитесь, что он загружен на GitHub.")
    st.stop()

tab1, tab2, tab3 = st.tabs([
    "🗺️ Интерактивный Атлас (1986-2018)",
    "📁 Загрузить свой растр (Прогноз)",
    "🧮 Калькулятор (Ручной ввод)"
])

# ==========================================
# ВКЛАДКА 1: ИНТЕРАКТИВНЫЙ АТЛАС
# ==========================================
with tab1:
    def get_available_years():
        years = []
        if os.path.exists(MAPS_DIR):
            for f in os.listdir(MAPS_DIR):
                if f.endswith('.tif') and f.startswith('RWI_Map_'):
                    try:
                        years.append(int(f.replace('RWI_Map_', '').replace('.tif', '')))
                    except:
                        pass
        return sorted(years)


    years = get_available_years()

    if not years:
        st.warning(f"В папке {MAPS_DIR} пока нет готовых карт.")
    else:
        selected_year = st.select_slider("Выберите год для анализа:", options=years, value=years[-1])
        file_path = os.path.join(MAPS_DIR, f'RWI_Map_{selected_year}.tif')

        with rasterio.open(file_path) as src:
            img = src.read(1)
            bounds = src.bounds
            min_lon, min_lat, max_lon, max_lat = transform_bounds(src.crs, 'EPSG:4326', *bounds)
            valid_pixels = img[~np.isnan(img)]
            total_pixels = len(valid_pixels)

            if total_pixels > 0:
                mean_rwi = np.mean(valid_pixels)
                stress_count = np.sum(valid_pixels < 0.85)
                optimal_count = np.sum(valid_pixels > 1.15)
                stress_percent = (stress_count / total_pixels) * 100
                optimal_percent = (optimal_count / total_pixels) * 100
                stress_ha = stress_count * PIXEL_AREA_HA
                optimal_ha = optimal_count * PIXEL_AREA_HA
                total_ha = total_pixels * PIXEL_AREA_HA
            else:
                mean_rwi, stress_percent, optimal_percent, stress_ha, optimal_ha, total_ha = 0, 0, 0, 0, 0, 0

            norm = plt.Normalize(vmin=0.5, vmax=1.5)
            cmap_mpl = plt.get_cmap('RdYlGn')
            rgba_img = cmap_mpl(norm(img))
            rgba_img[np.isnan(img), 3] = 0
            img_buf = io.BytesIO()
            plt.imsave(img_buf, rgba_img, format='png')
            img_buf.seek(0)
            b64_img = base64.b64encode(img_buf.read()).decode('utf-8')
            image_url = f"data:image/png;base64,{b64_img}"

        col_map, col_stats = st.columns([7, 3], gap="large")

        with col_map:
            st.markdown(f"**Карта стресса древостоев за {selected_year} год**")
            center_lat, center_lon = (min_lat + max_lat) / 2, (min_lon + max_lon) / 2
            m = folium.Map(location=[center_lat, center_lon], zoom_start=10, control_scale=True)
            m.get_root().header.add_child(
                folium.Element("<style>.leaflet-control-attribution {display: none !important;}</style>"))
            folium.TileLayer('OpenTopoMap', attr=' ').add_to(m)
            colormap = cm.LinearColormap(colors=['#d73027', '#fdae61', '#ffffbf', '#a6d96a', '#1a9850'], vmin=0.5,
                                         vmax=1.5, caption='Индекс RWI')
            m.add_child(colormap)
            folium.raster_layers.ImageOverlay(image=image_url, bounds=[[min_lat, min_lon], [max_lat, max_lon]],
                                              opacity=0.8).add_to(m)
            try:
                folium.GeoJson(GEOJSON_PATH, name='Границы ООПТ',
                               style_function=lambda x: {'color': '#2b2b2b', 'weight': 2.5, 'fillOpacity': 0,
                                                         'dashArray': '6, 6'}).add_to(m)
            except:
                pass

            map_data = st_folium(m, width=800, height=550, key="map1")
            if map_data and map_data.get('last_clicked'):
                clat, clon = map_data['last_clicked']['lat'], map_data['last_clicked']['lng']
                with rasterio.open(file_path) as src:
                    xs, ys = transform('EPSG:4326', src.crs, [clon], [clat])
                    try:
                        row, col = src.index(xs[0], ys[0])
                        val = img[row, col]
                        if np.isnan(val):
                            st.warning(f"📍 Вне зоны леса")
                        else:
                            st.success(f"📍 RWI: {val:.3f}")
                    except:
                        pass

        with col_stats:
            st.markdown("### 📊 Аналитика года")
            st.metric(label="Средний RWI", value=f"{mean_rwi:.2f}")
            st.markdown(f"🔴 Сильный стресс: `{stress_percent:.1f}%` ({stress_ha:,.0f} га)".replace(',', ' '))
            st.markdown(f"🟢 Активный рост: `{optimal_percent:.1f}%` ({optimal_ha:,.0f} га)".replace(',', ' '))
            st.markdown(f"*Общая площадь: {total_ha:,.0f} га*".replace(',', ' '))

            fig_hist, ax_hist = plt.subplots(figsize=(5, 4))
            ax_hist.hist(valid_pixels, bins=40, color='#2ca02c', alpha=0.7, edgecolor='black', linewidth=0.5)
            ax_hist.axvline(mean_rwi, color='red', linestyle='dashed', linewidth=2)
            ax_hist.axvline(1.0, color='black', linestyle='solid', linewidth=1)
            ax_hist.set_title('Распределение RWI', fontsize=12)

            buf_hist = io.BytesIO()
            fig_hist.savefig(buf_hist, format="png", bbox_inches='tight', dpi=100)
            buf_hist.seek(0)
            st.image(buf_hist)
            plt.close(fig_hist)

            # --- КНОПКА ИИ-АНАЛИЗА ВО ВКЛАДКЕ 1 ---
            st.markdown("---")
            if st.button("👁️ ИИ-Анализ графиков и карты", key=f"ai_btn1_{selected_year}", use_container_width=True,
                         type="primary"):
                with st.spinner("ИИ изучает гистограмму и статистику..."):
                    # Собираем данные
                    stats_text = f"Анализ за {selected_year} год. Средний RWI: {mean_rwi:.2f}. Площадь сильного стресса: {stress_ha:,.0f} га ({stress_percent:.1f}%). Площадь активного роста: {optimal_ha:,.0f} га ({optimal_percent:.1f}%). Посмотри на гистограмму. Опиши форму распределения пикселей и сделай подробный лесопатологический вывод."

                    # Фотографируем график
                    buf_hist.seek(0)
                    hist_b64 = base64.b64encode(buf_hist.read()).decode('utf-8')

                    # Отправляем в ИИ
                    reply = analyze_with_vision(stats_text, hist_b64, selected_model, OPENROUTER_API_KEY)
                    st.info(reply)

            with open(file_path, "rb") as file:
                st.download_button(label=f"💾 Скачать растр {selected_year}", data=file,
                                   file_name=f"RWI_{selected_year}.tif", mime="image/tiff", use_container_width=True)

# ==========================================
# ВКЛАДКА 2: МГНОВЕННАЯ ЗАГРУЗКА РАСТРА
# ==========================================
with tab2:
    st.markdown("Загрузите файл `Predictor_Stack.tif`, чтобы построить прогноз и проанализировать его ИИ.")
    uploaded_file = st.file_uploader("Выберите TIF файл", type=['tif', 'tiff'])

    if uploaded_file is not None:
        if "last_file" not in st.session_state or st.session_state.last_file != uploaded_file.name:
            st.session_state.last_file = uploaded_file.name
            st.info("🔄 Нейросеть обрабатывает пространственные данные...")
            with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            try:
                with rasterio.open(tmp_path) as src:
                    out_meta = src.meta.copy()
                    out_meta.update({"count": 1, "dtype": 'float32', "nodata": np.nan})
                    img = src.read()
                    n_bands, height, width = img.shape
                    bounds = src.bounds
                    min_lon, min_lat, max_lon, max_lat = transform_bounds(src.crs, 'EPSG:4326', *bounds)

                    if n_bands != 7: st.error(f"❌ Ошибка: {n_bands} каналов вместо 7!"); st.stop()

                    img_2d = img.reshape(n_bands, -1).T
                    valid_mask = ~np.isnan(img_2d).any(axis=1) & (img_2d[:, 0] != src.nodata)

                    if os.path.exists(MASK_PATH):
                        with rasterio.open(MASK_PATH) as msk:
                            if msk.width == width and msk.height == height:
                                mask_1d = msk.read(1).flatten()
                            else:
                                mask_data = np.zeros((height, width), dtype='uint8')
                                reproject(source=rasterio.band(msk, 1), destination=mask_data,
                                          src_transform=msk.transform, src_crs=msk.crs, dst_transform=src.transform,
                                          dst_crs=src.crs, resampling=Resampling.nearest)
                                mask_1d = mask_data.flatten()
                            valid_mask = valid_mask & (mask_1d == 1)

                    if valid_mask.any():
                        X_valid = img_2d[valid_mask]
                        X_imp = imputer.transform(X_valid)
                        preds = rf_model.predict(X_imp)

                        pred_map = np.full((height * width), np.nan)
                        pred_map[valid_mask] = preds
                        pred_map_2d = pred_map.reshape((height, width))

                        st.session_state.t2_pred_map = pred_map_2d
                        st.session_state.t2_crs = src.crs
                        st.session_state.t2_transform = src.transform
                        st.session_state.t2_bounds = [[min_lat, min_lon], [max_lat, max_lon]]
                        st.session_state.t2_preds = preds

                        norm = plt.Normalize(vmin=0.5, vmax=1.5)
                        cmap_mpl = plt.get_cmap('RdYlGn')
                        rgba_img = cmap_mpl(norm(pred_map_2d))
                        rgba_img[np.isnan(pred_map_2d), 3] = 0
                        img_buf = io.BytesIO()
                        plt.imsave(img_buf, rgba_img, format='png')
                        img_buf.seek(0)
                        st.session_state.t2_img_url = f"data:image/png;base64,{base64.b64encode(img_buf.read()).decode('utf-8')}"

                        with rasterio.MemoryFile() as memfile:
                            with memfile.open(**out_meta) as dest: dest.write(pred_map_2d.astype('float32'), 1)
                            st.session_state.t2_tiff_bytes = memfile.read()
                    else:
                        st.error("❌ Растр пуст.");
                        st.stop()
            finally:
                os.remove(tmp_path)

        if "t2_pred_map" in st.session_state:
            preds = st.session_state.t2_preds
            total_pixels = len(preds)
            mean_rwi = np.mean(preds)
            stress_count = np.sum(preds < 0.85)
            optimal_count = np.sum(preds > 1.15)
            stress_percent = (stress_count / total_pixels) * 100
            optimal_percent = (optimal_count / total_pixels) * 100
            stress_ha = stress_count * PIXEL_AREA_HA
            optimal_ha = optimal_count * PIXEL_AREA_HA
            total_ha = total_pixels * PIXEL_AREA_HA

            col_map2, col_stats2 = st.columns([7, 3], gap="large")

            with col_map2:
                st.markdown("**Ваш загруженный прогноз RWI**")
                min_lat, min_lon = st.session_state.t2_bounds[0]
                max_lat, max_lon = st.session_state.t2_bounds[1]
                center_lat, center_lon = (min_lat + max_lat) / 2, (min_lon + max_lon) / 2
                m2 = folium.Map(location=[center_lat, center_lon], zoom_start=10, control_scale=True)
                m2.get_root().header.add_child(
                    folium.Element("<style>.leaflet-control-attribution {display: none !important;}</style>"))
                folium.TileLayer('OpenTopoMap', attr=' ').add_to(m2)
                colormap2 = cm.LinearColormap(colors=['#d73027', '#fdae61', '#ffffbf', '#a6d96a', '#1a9850'], vmin=0.5,
                                              vmax=1.5, caption='Индекс RWI')
                m2.add_child(colormap2)
                folium.raster_layers.ImageOverlay(image=st.session_state.t2_img_url, bounds=st.session_state.t2_bounds,
                                                  opacity=0.8).add_to(m2)
                try:
                    folium.GeoJson(GEOJSON_PATH, name='Границы ООПТ',
                                   style_function=lambda x: {'color': '#2b2b2b', 'weight': 2.5, 'fillOpacity': 0,
                                                             'dashArray': '6, 6'}).add_to(m2)
                except:
                    pass

                map_data2 = st_folium(m2, width=800, height=550, key="map2")
                if map_data2 and map_data2.get('last_clicked'):
                    clat, clon = map_data2['last_clicked']['lat'], map_data2['last_clicked']['lng']
                    xs, ys = transform('EPSG:4326', st.session_state.t2_crs, [clon], [clat])
                    try:
                        row, col = rowcol(st.session_state.t2_transform, xs[0], ys[0])
                        val = st.session_state.t2_pred_map[row, col]
                        if np.isnan(val):
                            st.warning(f"📍 Вне зоны леса")
                        else:
                            st.success(f"📍 Прогноз RWI: {val:.3f}")
                    except:
                        pass

            with col_stats2:
                st.markdown("### 📊 Аналитика прогноза")
                st.metric(label="Средний RWI", value=f"{mean_rwi:.2f}")
                st.markdown(f"🔴 Сильный стресс: `{stress_percent:.1f}%` ({stress_ha:,.0f} га)".replace(',', ' '))
                st.markdown(f"🟢 Активный рост: `{optimal_percent:.1f}%` ({optimal_ha:,.0f} га)".replace(',', ' '))

                fig_hist2, ax_hist2 = plt.subplots(figsize=(5, 4))
                ax_hist2.hist(preds, bins=40, color='#2ca02c', alpha=0.7, edgecolor='black', linewidth=0.5)
                ax_hist2.axvline(mean_rwi, color='red', linestyle='dashed', linewidth=2)
                ax_hist2.axvline(1.0, color='black', linestyle='solid', linewidth=1)
                ax_hist2.set_title('Распределение RWI', fontsize=12)

                buf_hist2 = io.BytesIO()
                fig_hist2.savefig(buf_hist2, format="png", bbox_inches='tight', dpi=100)
                buf_hist2.seek(0)
                st.image(buf_hist2)
                plt.close(fig_hist2)

                # --- КНОПКА ИИ-АНАЛИЗА ВО ВКЛАДКЕ 2 ---
                st.markdown("---")
                if st.button("👁️ ИИ-Анализ графиков и карты", key="ai_btn2", use_container_width=True, type="primary"):
                    with st.spinner("ИИ изучает гистограмму и статистику..."):
                        stats_text = f"Анализ пользовательского прогноза. Средний RWI: {mean_rwi:.2f}. Площадь сильного стресса: {stress_ha:,.0f} га ({stress_percent:.1f}%). Площадь активного роста: {optimal_ha:,.0f} га ({optimal_percent:.1f}%). Посмотри на гистограмму. Сделай подробный лесопатологический вывод."
                        buf_hist2.seek(0)
                        hist_b64 = base64.b64encode(buf_hist2.read()).decode('utf-8')
                        reply = analyze_with_vision(stats_text, hist_b64, selected_model, OPENROUTER_API_KEY)
                        st.info(reply)

                st.download_button(label="💾 Скачать прогноз", data=st.session_state.t2_tiff_bytes,
                                   file_name="Custom_RWI.tif", mime="image/tiff", use_container_width=True)

# ==========================================
# ВКЛАДКА 3: КАЛЬКУЛЯТОР
# ==========================================
with tab3:
    st.markdown("Посмотрите, как изменение климатических и спутниковых параметров влияет на прирост.")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Климатические факторы")
        v1 = st.number_input("Early_Temp_Diff (Шок май-июнь, °C)", value=-1.30, step=0.1)
        v2 = st.number_input("Early_DEF (Дефицит влаги май-июнь, мм)", value=55.0, step=5.0)
        v3 = st.number_input("Late_DEF (Дефицит влаги июль-авг, мм)", value=25.0, step=5.0)
        v4 = st.number_input("Winter_Precip (Осадки зимы, мм)", value=60.0, step=5.0)
        v5 = st.number_input("Late_Temp_Diff (Шок июль-авг, °C)", value=-1.80, step=0.1)
    with col2:
        st.subheader("Спутниковые факторы (Landsat)")
        v6 = st.slider("Landsat_NDMI (Влагосодержание хвои)", min_value=-0.50, max_value=0.60, value=0.24, step=0.01)
        v7 = st.slider("Landsat_LST (Температура поверхности, °C)", min_value=5.0, max_value=35.0, value=15.0, step=0.5)

    if st.button("🚀 Рассчитать RWI", type="primary"):
        X_manual = np.array([[v1, v2, v3, v4, v5, v6, v7]])
        X_imp = imputer.transform(X_manual)
        prediction = rf_model.predict(X_imp)[0]
        st.markdown("---")
        if prediction < 0.8:
            st.error(f"### Прогноз RWI: {prediction:.3f} (Экстремальный стресс / Усыхание)")
        elif prediction > 1.2:
            st.success(f"### Прогноз RWI: {prediction:.3f} (Благоприятные условия / Бурный рост)")
        else:
            st.warning(f"### Прогноз RWI: {prediction:.3f} (Нормальный радиальный прирост)")