"""Norway-specific external-data fetching with a cache-first contract.

This packages the Part-B fetch layer from the notebook: it probes the public,
no-auth Norges Bank / SSB / ECB / Riksbank endpoints and assembles a raw daily
panel of NOR-relevant series, alongside a connectivity report describing what
each source returned.

Network access is never triggered at import time — only inside the fetch
functions — so the module is safe to import offline.

Cache-first contract
---------------------
:func:`load_norway_raw` defaults to ``live=False``: it reads the committed
cache CSV so two people on different machines reproduce the same panel. A live
fetch is opt-in via ``live=True``; on success it overwrites the cache. If the
live fetch fails or returns empty, it falls back to the cache; if neither is
available it returns an empty frame plus an explanatory report entry.

Report tuples
-------------
Every report row is a 7-tuple ``(label, status, note, freq, n, lo, hi)`` where
``status`` is one of ``OK`` / ``SKIP`` / ``FAIL``, ``freq`` is the native
sampling frequency, ``n`` the observation count, and ``lo``/``hi`` the coverage
bounds (``None`` when empty). :func:`print_connectivity_report` renders them.
"""

import io
import json
import os
import urllib.request

import pandas as pd

NB_API  = 'https://data.norges-bank.no/api/data'
SSB_API = 'https://data.ssb.no/api/v0/en/table'
_HDRS   = {'User-Agent': 'master-thesis-research'}


def _http_get(url: str) -> str:
    return urllib.request.urlopen(urllib.request.Request(url, headers=_HDRS),
                                  timeout=30).read().decode('utf-8', 'replace')


def _http_post(url: str, payload: dict) -> str:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={**_HDRS, 'Content-Type': 'application/json'})
    return urllib.request.urlopen(req, timeout=30).read().decode('utf-8', 'replace')


def _nb_series(flow: str, key: str, start: str, end: str) -> pd.Series:
    """Norges Bank SDMX -> pd.Series(OBS_VALUE) indexed by date."""
    url = f'{NB_API}/{flow}/{key}?format=csv&startPeriod={start}&endPeriod={end}'
    d = pd.read_csv(io.StringIO(_http_get(url)), sep=';')
    s = pd.Series(pd.to_numeric(d['OBS_VALUE'], errors='coerce').values,
                  index=pd.to_datetime(d['TIME_PERIOD'])).dropna().sort_index()
    return s[~s.index.duplicated(keep='last')]


def _ssb_monthly(table: str, content: str, filters: list, start: str, end: str) -> pd.Series:
    """SSB PxWebApi monthly -> pd.Series indexed by reference-month END timestamp."""
    query = [{'code': c, 'selection': {'filter': 'item', 'values': v}} for c, v in filters]
    query.append({'code': 'ContentsCode', 'selection': {'filter': 'item', 'values': [content]}})
    js  = json.loads(_http_post(f'{SSB_API}/{table}',
                                {'query': query, 'response': {'format': 'json-stat2'}}))
    idx, vals = js['dimension']['Tid']['category']['index'], js['value']
    out = {}
    for m, pos in sorted(idx.items(), key=lambda kv: kv[1]):
        if vals[pos] is None:
            continue
        out[pd.Timestamp(int(m[:4]), int(m[5:7]), 1) + pd.offsets.MonthEnd(0)] = float(vals[pos])
    s = pd.Series(out).sort_index()
    return s[(s.index >= start) & (s.index <= end)]


def _ecb_dfr(start: str) -> pd.Series:
    url = ('https://data-api.ecb.europa.eu/service/data/FM/'
           f'B.U2.EUR.4F.KR.DFR.LEV?format=csvdata&startPeriod={start}')
    d = pd.read_csv(io.StringIO(_http_get(url)))
    return pd.Series(pd.to_numeric(d['OBS_VALUE'], errors='coerce').values,
                     index=pd.to_datetime(d['TIME_PERIOD'])).dropna().sort_index()


def _riksbank(start: str) -> pd.Series:
    j = json.loads(_http_get(f'https://api.riksbank.se/swea/v1/Observations/SECBREPOEFF/{start}'))
    return pd.Series({pd.Timestamp(o['date']): float(o['value']) for o in j}).sort_index()


def fetch_norway_data(start, end) -> tuple:
    """Probe every Norway source. Returns (raw_df, report). Each source is wrapped:
    on failure it is logged and skipped, never fabricated."""
    a, b   = pd.Timestamp(start).strftime('%Y-%m-%d'), pd.Timestamp(end).strftime('%Y-%m-%d')
    raw, report = {}, []

    def _try(label, fn, freq):
        try:
            s = fn()
            d = s.dropna() if s is not None else pd.Series(dtype=float)
            if len(d) == 0:
                report.append((label, 'SKIP', 'empty', freq, 0, None, None)); return None
            report.append((label, 'OK', '', freq, len(d), d.index.min().date(), d.index.max().date()))
            return s
        except Exception as e:
            report.append((label, 'FAIL', repr(e)[:60], freq, 0, None, None)); return None

    raw['nb_eurnok'] = _try('NB EUR/NOK',  lambda: _nb_series('EXR', 'B.EUR.NOK.SP', a, b), 'daily')
    raw['nb_usdnok'] = _try('NB USD/NOK',  lambda: _nb_series('EXR', 'B.USD.NOK.SP', a, b), 'daily')
    raw['nb_i44']    = _try('NB I-44',     lambda: _nb_series('EXR', 'B.I44.NOK.SP', a, b), 'daily')
    raw['nb_polrate']= _try('NB policy rate', lambda: _nb_series('IR', 'B.KPRA.SD.R', a, b), 'daily')
    raw['nb_nowa']   = _try('NB NOWA',     lambda: _nb_series('SHORT_RATES', 'B.NOWA.ON.R', a, b), 'daily')
    for ten in ['3Y', '5Y', '10Y']:
        raw[f'nb_govt_{ten.lower()}'] = _try(
            f'NB govt {ten}', lambda t=ten: _nb_series('GOVT_GENERIC_RATES', f'B.{t}.GBON', a, b), 'daily')
    raw['ssb_kpi_yoy']   = _try('SSB KPI YoY', lambda: _ssb_monthly(
        '03013', 'Tolvmanedersendring', [('Konsumgrp', ['TOTAL'])], a, b), 'monthly')
    raw['ssb_kpijae_yoy']= _try('SSB KPI-JAE YoY', lambda: _ssb_monthly(
        '05327', 'Tolvmanedersendring', [('Konsumgrp', ['JAE_TOTAL'])], a, b), 'monthly')
    raw['ssb_unemp']     = _try('SSB LFS unemp', lambda: _ssb_monthly(
        '13760', 'Arbeidsledige',
        [('Kjonn', ['0']), ('Alder', ['15-74']), ('Justering', ['S'])], a, b), 'monthly')
    raw['ecb_dfr']    = _try('ECB deposit rate', lambda: _ecb_dfr(a), 'event')
    raw['rb_polrate'] = _try('Riksbank policy rate', lambda: _riksbank(a), 'daily')

    raw = {k: v for k, v in raw.items() if v is not None}
    return (pd.DataFrame(raw).sort_index() if raw else pd.DataFrame()), report


def load_norway_raw(start, end, cache_path: str, *, live: bool = False) -> tuple:
    """Load the raw Norway panel, cache-first.

    Parameters
    ----------
    start, end :
        Date bounds passed to :func:`fetch_norway_data` for a live fetch.
    cache_path :
        CSV path used as the reproducible cache (written on a successful live
        fetch, read otherwise).
    live :
        If ``True``, attempt a live fetch first and overwrite the cache on
        success. Defaults to ``False`` (cache-only) so runs are reproducible
        across machines from the committed cache.

    Returns
    -------
    (df, report) :
        ``df`` is the raw panel (possibly empty); ``report`` is a list of
        7-tuples ``(label, status, note, freq, n, lo, hi)``.
    """
    nor_raw = pd.DataFrame()
    report: list = []
    if live:
        try:
            nor_raw, report = fetch_norway_data(start, end)
            if not nor_raw.empty:
                nor_raw.to_csv(cache_path)
                return nor_raw, report
        except Exception as e:
            print(f'[WARN] live fetch failed ({repr(e)[:80]}); trying cache')

    # live=False, or the live fetch failed / returned empty -> fall back to cache
    if os.path.exists(cache_path):
        nor_raw = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        report = report or [('cache load', 'OK', cache_path, 'mixed', len(nor_raw), None, None)]
        return nor_raw, report

    # neither a live fetch nor a cache produced data
    report = report or []
    report.append(('norway data', 'FAIL',
                   f'no cache at {cache_path}; pass live=True to fetch', 'mixed', 0, None, None))
    return pd.DataFrame(), report


def print_connectivity_report(report) -> None:
    """Render a connectivity report (list of 7-tuples) exactly as the notebook did."""
    print('=== PART B — connectivity report ===')
    print(f'  {"source":24s} {"stat":4s} {"freq":7s} {"n_obs":>6s}  coverage')
    print('  ' + '-' * 70)
    for label, status, note, freq, n, lo, hi in report:
        cov = f'{lo}..{hi}' if lo else ''
        print(f'  {label:24s} {status:4s} {freq:7s} {n:6d}  {cov}  {note}')
    print('  SKIP Brent / Brent-WTI / EIA gas  (needs API key) -> fall back to oil_mom_*/natgas_mom_1m')
