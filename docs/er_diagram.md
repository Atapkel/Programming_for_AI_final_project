# ER diagram — `data/macro.db`

```mermaid
erDiagram
    sources ||--o{ indicators : provides
    indicators ||--o{ observations : has

    sources {
        INTEGER source_id PK
        TEXT name UK
        TEXT url
    }
    indicators {
        INTEGER indicator_id PK
        TEXT code UK "usd_kzt, base_rate, cpi_index, brent, ..."
        TEXT name
        TEXT unit
        TEXT frequency "daily, monthly, annual, decision"
        TEXT source_code "series id at the source, e.g. DCOILBRENTEU"
        INTEGER source_id FK
    }
    observations {
        INTEGER indicator_id PK, FK
        TEXT date PK "YYYY-MM-DD"
        REAL value
        INTEGER is_filled "1 = interpolated"
    }
    monthly_macro {
        TEXT month PK "YYYY-MM"
        REAL usd_kzt_avg
        REAL usd_kzt_eom
        INTEGER fx_filled_days
        REAL brent_avg
        REAL usd_kzt_change_pct
        REAL brent_change_pct
        REAL cpi_index
        REAL inflation_mom
        REAL inflation_yoy
        REAL base_rate
        REAL real_rate
        TEXT regime
    }
```

- `sources`, `indicators`, `observations` are the normalized core: every series, whatever
  its frequency, is stored as rows of `(indicator, date, value)`.
- `monthly_macro` is a derived analysis table built by `src/preprocessing.py`
  (all series aggregated to months and joined); it is used for modelling and the dashboard.
