#!/usr/bin/env python3
"""Stage 1: inspect historical XAUUSD M30 OHLC data, no trade simulation."""
import csv, json, sys
from collections import Counter
from datetime import datetime, timezone, timedelta
from pathlib import Path

DATA = Path('data/XAUUSD_M30.csv')
OUT = Path('results')

def parse_time(value):
    value = value.strip().replace('Z', '+00:00')
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        dt = datetime.strptime(value, '%Y-%m-%d %H:%M:%S')
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)

def main():
    if not DATA.exists():
        sys.exit('ERROR: data/XAUUSD_M30.csv fehlt. CSV mit UTC-Zeit und OHLC hochladen.')
    with DATA.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        required = {'time','open','high','low','close'}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            sys.exit('ERROR: CSV-Spalten erforderlich: time,open,high,low,close (volume optional)')
        rows=[]
        for line,r in enumerate(reader,2):
            try:
                t=parse_time(r['time']); o,h,l,c=(float(r[k]) for k in ('open','high','low','close'))
                if not (l <= min(o,c) <= max(o,c) <= h) or l<=0:
                    raise ValueError('OHLC inkonsistent')
                rows.append((t,o,h,l,c))
            except (ValueError,TypeError) as e:
                sys.exit(f'ERROR: Zeile {line}: {e}')
    if not rows: sys.exit('ERROR: Leere CSV')
    rows.sort(key=lambda x:x[0])
    duplicates=sum(rows[i][0]==rows[i-1][0] for i in range(1,len(rows)))
    gaps=[]
    for a,b in zip(rows,rows[1:]):
        diff=b[0]-a[0]
        if diff>timedelta(minutes=30):
            gaps.append({'from':a[0].isoformat(),'to':b[0].isoformat(),'missing_intervals':int(diff.total_seconds()//1800)-1})
    by_year=Counter(str(r[0].year) for r in rows)
    report={'phase':'DATA_AUDIT_ONLY','strategy_backtested':False,'time_zone':'UTC',
            'bars':len(rows),'first_bar':rows[0][0].isoformat(), 'last_bar':rows[-1][0].isoformat(),
            'duplicate_timestamps':duplicates,'gaps_over_30m':len(gaps),
            'note':'Wochenenden, Feiertage und Brokerpausen koennen normale Luecken verursachen.',
            'bars_by_year':dict(sorted(by_year.items())),'first_100_gaps':gaps[:100]}
    OUT.mkdir(exist_ok=True)
    (OUT/'audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='first_100_gaps'},indent=2))
    if duplicates: sys.exit('ERROR: Doppelte Zeitstempel erkannt. Bitte Daten bereinigen.')

if __name__=='__main__': main()
