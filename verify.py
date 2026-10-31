"""Read-only checks of real source and analytical invariants; no fake observations."""
import hashlib
import json
from pathlib import Path
import py_compile
import numpy as np
import pandas as pd
import app


def verify():
    py_compile.compile(str(Path(__file__).with_name('app.py')), doraise=True)
    original, df = app.load_data()
    first, last = df._date.min(), df._date.max()
    monthly = app.add_anomalies(app.monthly_counts(df, first, last))
    assert len(df) == 55533 and monthly.detections.sum() == 55533
    assert len(monthly) == 240
    assert str(first.date()) == '2006-03-21' and str(last.date()) == '2025-11-21'
    expected = [5421, 2737, 10763, 3077, 6815, 1638, 4591, 711, 1676, 1067,
                1113, 1951, 1624, 2180, 697, 4038, 2761, 1977, 290, 406]
    assert df.groupby('_year').size().tolist() == expected
    for year in range(2006, 2026):
        grid = app.daily_calendar(df, year, first, last)
        assert grid.sum().sum() == expected[year - 2006]
        assert np.isnan(grid.loc[2, 30])
    assert not np.isnan(app.daily_calendar(df, 2024, first, last).loc[2, 29])
    assert np.isnan(app.daily_calendar(df, 2023, first, last).loc[2, 29])
    assert monthly.loc[monthly.status.ne('within_file_range'), 'anomaly_percent'].isna().all()
    assert monthly.loc[monthly.baseline_median_2007_2024.eq(0), 'anomaly_percent'].isna().all()
    april = monthly[(monthly.year == 2008) & (monthly.month == 4)].iloc[0]
    baseline = np.median([len(df[(df._year == y) & (df._month == 4)]) for y in range(2007, 2025)])
    assert baseline == 721.5
    assert np.isclose(april.anomaly_percent, (april.detections / baseline - 1) * 100)
    sample = app.map_sample(df, 3000)
    assert len(sample) == 3000 and sample.index.is_unique and sample.index.isin(df.index).all()
    assert sample.index.equals(app.map_sample(df, 3000).index)
    empty = df[df.satellite.isin([])]
    assert app.map_sample(empty, 3000).empty
    assert app.add_anomalies(app.monthly_counts(empty, first, last)).anomaly_percent.isna().all()
    assert app.monthly_counts(empty, first, last).detections.sum() == 0
    from io import StringIO
    exported = pd.read_csv(StringIO(original.to_csv(index=False)), dtype=str, keep_default_na=False)
    pd.testing.assert_frame_equal(original, exported)
    return {
        'source_sha256': hashlib.sha256(app.DATA.read_bytes()).hexdigest(),
        'source_rows': len(df), 'monthly_sum': int(monthly.detections.sum()),
        'date_min': str(first.date()), 'date_max': str(last.date()),
        'instruments': df.instrument.value_counts().to_dict(),
        'satellites': df.satellite.value_counts().to_dict(),
        'versions': df.version.value_counts().to_dict(),
        'annual_counts': dict(zip(range(2006, 2026), expected)),
        'syntax_and_analytical_checks': 'passed',
        'source_bytes_preserved': True,
        'spatial_filter': False, 'added_observations': 0,
        'full_streamlit_plotly_folium_runtime': 'not tested: packages absent; installation declined',
    }


if __name__ == '__main__':
    print(json.dumps(verify(), indent=2, ensure_ascii=False))
