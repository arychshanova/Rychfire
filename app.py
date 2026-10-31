"""Rychfire v0.2: real MODIS detections; run with streamlit run app.py."""
from pathlib import Path
import hashlib

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'fire_archive_M-C61_833601.csv'
SHA256 = '447713ef898fc003def3acd489d71c443a874a181efdb5969d1c0dbf03e3039c'
AOI = 'Северный Казахстан — пользовательский AOI'
MONTHS = ['Янв', 'Фев', 'Мар', 'Апр', 'Май', 'Июн', 'Июл', 'Авг', 'Сен', 'Окт', 'Ноя', 'Дек']


def load_data(path=DATA):
    """Validate exact source; preserve every original field, row and multiplicity."""
    path = Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != SHA256:
        raise ValueError('Контрольная сумма CSV отличается от исходной. Нужен оригинальный файл.')
    original = pd.read_csv(path, dtype=str, keep_default_na=False)
    if len(original) != 55533:
        raise ValueError('Ожидалось 55 533 обнаружления.')
    df = original.copy()
    df['_date'] = pd.to_datetime(df['acq_date'], format='%Y-%m-%d', errors='raise')
    for field in ['latitude', 'longitude', 'confidence']:
        df['_' + field] = pd.to_numeric(df[field], errors='raise')
    if not df['_latitude'].between(-90, 90).all() or not df['_longitude'].between(-180, 180).all():
        raise ValueError('Недопустимые координаты; строки не удалялись.')
    if not df['_confidence'].between(0, 100).all():
        raise ValueError('Недопустимая confidence; строки не удалялись.')
    if set(df['instrument']) != {'MODIS'} or not df['_date'].dt.year.between(2006, 2025).all():
        raise ValueError('Ожидался только MODIS за 2006–2025 годы.')
    df['_year'] = df['_date'].dt.year
    df['_month'] = df['_date'].dt.month
    return original, df


def monthly_counts(df, first_date, last_date):
    """Zero means no rows in file, never confirmed absence of fire/coverage."""
    periods = pd.period_range('2006-01', '2025-12', freq='M')
    counts = df.groupby(df['_date'].dt.to_period('M')).size().reindex(periods, fill_value=0)
    out = pd.DataFrame({'date': periods.to_timestamp(), 'detections': counts.to_numpy()})
    out['year'] = out.date.dt.year
    out['month'] = out.date.dt.month
    first, last = pd.Timestamp(first_date), pd.Timestamp(last_date)
    out['status'] = [
        'outside' if p.end_time.normalize() < first or p.start_time > last
        else 'edge_partial' if p.start_time < first or p.end_time.normalize() > last
        else 'within_file_range' for p in periods
    ]
    return out


def add_anomalies(monthly):
    # Baseline is fixed, independent of chosen display years; edge year excluded.
    baseline = monthly[monthly.year.between(2007, 2024)].groupby('month').detections.median()
    out = monthly.copy()
    out['baseline_median_2007_2024'] = out.month.map(baseline)
    denom = out.baseline_median_2007_2024.replace(0, np.nan)
    out['anomaly_percent'] = (out.detections / denom - 1) * 100
    out.loc[out.status != 'within_file_range', 'anomaly_percent'] = np.nan
    return out


def daily_calendar(df, year, first_date, last_date):
    dates = pd.date_range(f'{year}-01-01', f'{year}-12-31')
    out = df.groupby('_date').size().reindex(dates, fill_value=0).rename('detections').to_frame()
    out.loc[(out.index < first_date) | (out.index > last_date), 'detections'] = np.nan
    out['month'] = out.index.month
    out['day'] = out.index.day
    return out.pivot(index='month', columns='day', values='detections').reindex(index=range(1, 13), columns=range(1, 32))


def map_sample(df, limit):
    return df.sample(n=min(len(df), int(limit)), random_state=42) if len(df) else df.copy()


def make_map(df, limit):
    import folium
    from folium.plugins import FastMarkerCluster
    sample = map_sample(df, limit)
    center = [float(df._latitude.mean()), float(df._longitude.mean())] if len(df) else [55.6, 65.6]
    m = folium.Map(location=center, zoom_start=6, tiles='OpenStreetMap')
    if len(sample):
        # Positions come only from rows of the source. No jitter or generated points.
        points = [[float(r._latitude), float(r._longitude)] for _, r in sample.iterrows()]
        FastMarkerCluster(points, name='Обнаружения MODIS').add_to(m)
        m.fit_bounds([[float(df._latitude.min()), float(df._longitude.min())],
                      [float(df._latitude.max()), float(df._longitude.max())]])
    folium.LayerControl().add_to(m)
    return m


def main():
    import streamlit as st
    import streamlit.components.v1 as components
    import plotly.express as px
    import plotly.graph_objects as go

    st.set_page_config(page_title='Rychfire v0.2 | MODIS', page_icon='🔥', layout='wide')
    st.title('🔥 Rychfire v0.2')
    st.caption(AOI)
    st.warning('Обнаружение ≠ отдельный пожар. Только MODIS. 2025 год неполный. '
               'AOI не является административной границей и не подтверждает охват всей Костанайской области.')
    try:
        original, df = st.cache_data(show_spinner='Проверка исходного CSV…')(load_data)()
    except (OSError, ValueError, KeyError) as exc:
        st.error(f'Не удалось прочитать исходные данные: {exc}')
        st.stop()
    first, last = df._date.min(), df._date.max()
    st.caption(f'В исходном файле: {len(df):,} строк · {first:%d.%m.%Y}–{last:%d.%m.%Y} · '
               'без пространственной фильтрации и удаления повторов.')
    years = st.sidebar.slider('Годы', 2006, 2025, (2006, 2025))
    confidence = st.sidebar.slider('Минимальная confidence MODIS', 0, 100, 0)
    satellites = st.sidebar.multiselect('Спутники', ['Aqua', 'Terra'], default=['Aqua', 'Terra'])
    daynight = st.sidebar.multiselect('Время наблюдения', ['D', 'N'], default=['D', 'N'])
    st.sidebar.caption('D — день, N — ночь. По умолчанию включены все исходные строки. '
                       'Фильтры меняют все расчёты и экспорт. Лимит карты меняет только карту.')
    population = df[df._confidence.ge(confidence) & df.satellite.isin(satellites) & df.daynight.isin(daynight)]
    selected = population[population._year.between(*years)]
    monthly = add_anomalies(monthly_counts(population, first, last))
    visible = monthly[monthly.year.between(*years)].copy()
    a, b, c = st.columns(3)
    a.metric('Обнаружений в выборке', f'{len(selected):,}')
    b.metric('Дней с обнаружениями', str(selected._date.nunique()))
    c.metric('Годы в выборке', f'{years[0]}–{years[1]}')
    if selected.empty:
        st.info('Нет обнаружений с выбранными параметрами. Экспорт содержит только заголовки.')
    st.subheader('Календарь реальных обнаружений')
    st.caption('Числа — количество строк CSV. 0 означает отсутствие записей, а не доказанное отсутствие '
               'пожаров. Пропуски — вне диапазона дат файла или несуществующие дни. '
               'Первая и последняя даты обнаружений не доказывают начало и конец спутникового покрытия.')
    grid_source = visible.copy()
    grid_source.loc[grid_source.status.eq('outside'), 'detections'] = np.nan
    grid = grid_source.pivot(index='year', columns='month', values='detections').reindex(columns=range(1, 13))
    st.plotly_chart(px.imshow(grid, x=MONTHS, aspect='auto', color_continuous_scale='YlOrRd',
                            labels={'x': 'Месяц', 'y': 'Год', 'color': 'Обнаружения'}), use_container_width=True)
    calendar_year = st.selectbox('Год дневного календаря', list(range(years[0], years[1] + 1)), index=years[1] - years[0])
    daily = daily_calendar(population, calendar_year, first, last)
    st.plotly_chart(px.imshow(daily, y=MONTHS, aspect='auto', color_continuous_scale='YlOrRd',
                            labels={'x': 'День месяца', 'y': 'Месяц', 'color': 'Обнаружения'}), use_container_width=True)
    st.subheader('Годовая и месячная сезонность')
    annual = visible.groupby('year', as_index=False).detections.sum()
    annual['period'] = annual.year.map(lambda y: 'Крайний год файла' if y in [2006, 2025] else 'Внутри диапазона файла')
    st.plotly_chart(px.bar(annual, x='year', y='detections', color='period',
                           labels={'year': 'Год', 'detections': 'Обнаружения', 'period': 'Период'}), use_container_width=True)
    st.caption('2006 и 2025 — крайние годы файла; 2025 неполный. Полнота остальных лет не подтверждена. '
               'Не сравнивайте неполный год с полными годовыми суммами как изменение пожарной активности.')
    season = monthly[monthly.year.between(2007, 2024)].groupby('month', as_index=False).detections.agg(['mean', 'median']).reset_index()
    st.plotly_chart(go.Figure([
        go.Bar(x=MONTHS, y=season['mean'], name='Среднее'),
        go.Scatter(x=MONTHS, y=season['median'], name='Медиана', mode='lines+markers')
    ]).update_layout(yaxis_title='Обнаружений за месяц', xaxis_title='Месяц · фиксированная база 2007–2024'), use_container_width=True)
    st.subheader('Месячные отклонения от медианы')
    st.caption('База: медиана числа обнаружений для того же календарного месяца за 2007–2024, '
               'с учётом выбранных спутников, confidence и дня/ночи, но независимо от выбранных лет. '
               'Отклонение = (число / медиана − 1) × 100%. При медиане 0 и для крайних неполных месяцев '
               'значение не рассчитывается. Это описательная статистика, не значимость, не прогноз и не оценка риска.')
    st.plotly_chart(px.bar(visible, x='date', y='anomaly_percent',
                           labels={'date': 'Месяц', 'anomaly_percent': 'Отклонение, %'}), use_container_width=True)
    st.subheader('Карта обнаружений MODIS')
    limit = st.slider('Максимум точек на карте', 100, 10000, 3000, 100)
    st.caption(f'Показано {min(len(selected), limit):,} из {len(selected):,} обнаружений. '
               'При превышении лимита используется воспроизводимая случайная выборка реальных строк (seed=42). '
               'Кластеры считают только показанные точки. Все расчёты и CSV используют полную выбранную выборку. '
               'Подложка и библиотеки карты требуют интернета.')
    components.html(make_map(selected, limit).get_root().render(), height=530, scrolling=False)
    st.subheader('Экспорт CSV')
    st.download_button('Все исходные обнаружения · оригинальный CSV', DATA.read_bytes(),
                       file_name=DATA.name, mime='text/csv')
    st.download_button('Выбранные обнаружения', original.loc[selected.index].to_csv(index=False).encode('utf-8-sig'),
                       file_name='rychfire_selected_detections.csv', mime='text/csv')
    st.download_button('Месячные суммы и отклонения', visible.to_csv(index=False).encode('utf-8-sig'),
                       file_name='rychfire_monthly.csv', mime='text/csv')
    st.download_button('Годовые суммы', annual.to_csv(index=False).encode('utf-8-sig'),
                       file_name='rychfire_annual.csv', mime='text/csv')
    with st.expander('Методика и научные ограничения'):
        st.markdown('''
Каждая строка — спутниковое обнаружение тепловой аномалии. Один пожар может давать много пикселей
и повторных обнаружений при разных пролётах; источники тепла могут не быть лесными пожарами.
Число обнаружений не является числом пожаров или площадью выгорания.

Используется только MODIS на Terra/Aqua. VIIRS и объединения сенсоров нет. Облачность, дым,
время пролёта, размер пикселя, пропуски наблюдений и изменения работы спутников влияют на ряды.
Confidence — показатель алгоритма MODIS, не вероятность отдельного пожара.
Даты агрегируются по acq_date NASA FIRMS (UTC), без пересчёта в местное время.

2006 начинается в файле с 21 марта, 2025 заканчивается 21 ноября и неполный.
Полнота покрытия не подтверждена даже для внутренних лет. Нули — только отсутствие строк в файле.
Название архива содержит M-C61, но поле version содержит 6.03 и 61.03; значения сохранены
без подмены. Однородность обработки всего временного ряда не установлена.

Территория: Северный Казахстан — пользовательский AOI. Использованы все координаты исходного
файла, без обрезки административными границами; полного охвата Костанайской области не заявляется.
Контур AOI не восстановлен: экстент точек не равен исходной границе запроса.

Источник: [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/),
[описание полей MODIS](https://firms.modaps.eosdis.nasa.gov/download/Readme.txt).
''')


if __name__ == '__main__':
    main()
