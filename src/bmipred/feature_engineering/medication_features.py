#!/usr/bin/env python3

import numpy as np
import pandas as pd
import heapq
from collections import Counter, defaultdict
import warnings

warnings.filterwarnings('ignore') # deactivate warnings



def normalize_atc_codes(df: pd.DataFrame,
                        atc_col: str = "ATC") -> pd.DataFrame:
    out = df.copy()
    atc = out[atc_col].astype("string").str.strip().str.upper()
    out[atc_col] = atc.replace({
        "": pd.NA, 
        "NAN": pd.NA, 
        "NONE": pd.NA,
        "NAT": pd.NA, 
        "<NA>": pd.NA, 
        "NA": pd.NA,
    })
    return out



def medication_history_features(df: pd.DataFrame,
                                patientid_col: str = "PatientDurableKey",
                                atc_col: str = "ATC",
                                start_col: str = "StartInstant",
                                discontinued_col: str = "DiscontinuedInstant",) -> pd.DataFrame:
    
    
    ANTIPSYCHOTICS_ATC_CODES = [
        "N05AD01", "N05AD08", "N05AH02", "N05AH03", "N05AH04", "N05AX08", "N05AX12",
        "N05AF05", "N05AX16", "N05AF03", "N05AE05", "N05AE03", "N05AG02", "N05AX13",
        "N05AA02", "N05AE04", "N05AF01", "N05AL05", "N05AX", "N05AB03", "N05AX15",
        "N05AC01", "N05AL01", "N05AD05", "N05AB04", "N05AD03", "N05AG03", "N05AH05",
        "N05AA01", "N05AC02"
    ]
    
    ANTIDEPRESSANTS_ATC_CODES = [
        "N06AX11", "N06AB04", "N06AX16", "N06AX26", "N06AB06", "N06AX21", "N06AA10",
        "N06AX03", "N06AB05", "N06AA09", "N06AB03", "N06AA02", "N06AB10", "N06AA04",
        "N06AX22", "N06AX12", "N06AF01", "N06AA16", "N06AG02", "N06AB08", "N06AX18",
        "N06AA12", "N06AX25", "N06AX27", "N06AA21"
    ]
    
    ANXIOLYTICS_ATC_CODES = [
        "N05BA04", "N05BA01", "N05BB01", "N05BA06", "N05BA02",
        "N05BA12", "N05BA09", "N05BA08", "N05BE01"
    ]
    
    ATC_SETS = {
        "past_antipsychotic_treatments": ANTIPSYCHOTICS_ATC_CODES,
        "past_antidepressant_treatments": ANTIDEPRESSANTS_ATC_CODES,
        "past_anxiolytic_treatments": ANXIOLYTICS_ATC_CODES,
    }

    out = df.copy()
    out[start_col] = pd.to_datetime(out[start_col], errors="coerce")
    out[discontinued_col] = pd.to_datetime(out[discontinued_col], errors="coerce")

    sets = {name: set(codes) for name, codes in ATC_SETS.items()}

    # Precompute set membership per distinct code
    flags = {
        code: {name: code in codes for name, codes in sets.items()}
        for code in out[atc_col].dropna().unique()
    }

    n = len(out)
    idx_to_pos = pd.Series(np.arange(n), index=out.index)

    active_count       = np.zeros(n, dtype=np.int32)
    active_list        = np.full(n, "", dtype=object)
    past_unique_count  = np.zeros(n, dtype=np.int32)
    past_unique_list   = np.full(n, "", dtype=object)
    times_current      = np.zeros(n, dtype=np.int32)
    set_buffers        = {name: np.zeros(n, dtype=np.int32) for name in sets}

    for _, g in out.groupby(patientid_col, dropna=False, sort=False):
        g = g.dropna(subset=[start_col]).sort_values(start_col)
        if g.empty:
            continue

        active_atc_counts = Counter()
        end_heap = []
        ever = set()
        code_history = defaultdict(int)               # strictly-earlier count per code
        set_history  = {name: 0 for name in sets} # strictly-earlier count per set

        for t, batch in g.groupby(start_col, sort=False):

            # 1) Expire meds whose end <= t (end-exclusive)
            while end_heap and end_heap[0][0] <= t:
                _, code = heapq.heappop(end_heap)
                active_atc_counts[code] -= 1
                if active_atc_counts[code] <= 0:
                    del active_atc_counts[code]

            positions   = idx_to_pos.loc[batch.index].values
            batch_codes = batch[atc_col].values
            batch_ends  = batch[discontinued_col].values

            # 2) PAST features (strictly before t) — read BEFORE any updates
            past_unique_count[positions] = len(ever)
            past_unique_list[positions]  = ",".join(sorted(ever)) if ever else ""
            for name, count in set_history.items():
                set_buffers[name][positions] = count
            for pos, code in zip(positions, batch_codes):
                if pd.notna(code):
                    times_current[pos] = code_history[code]

            # 3) ACTIVE features — same-time starts included, row's own code excluded
            starting_atcs = {c for c in batch_codes if pd.notna(c)}
            codes_active_now = set(active_atc_counts) | starting_atcs
            for pos, code in zip(positions, batch_codes):
                row_active = codes_active_now - {code} if pd.notna(code) else codes_active_now
                active_count[pos] = len(row_active)
                active_list[pos]  = ",".join(sorted(row_active)) if row_active else ""

            # 4) Update persistent state AFTER batch
            for code, end in zip(batch_codes, batch_ends):
                if pd.isna(code):
                    continue
                ever.add(code)
                code_history[code] += 1
                for name, in_set in flags[code].items():
                    set_history[name] += in_set
                if pd.notna(end) and end > t:
                    active_atc_counts[code] += 1
                    heapq.heappush(end_heap, (end, code))
                elif pd.isna(end):
                    active_atc_counts[code] += 1   # never expires

    out["active_atc_codes_count"]      = active_count
    out["active_atc_codes_csv"]        = active_list
    out["past_unique_atc_codes_count"] = past_unique_count
    out["past_unique_atc_codes_csv"]   = past_unique_list
    out["past_atc_code_treatments"]    = times_current
    for name, buf in set_buffers.items():
        out[name] = buf

    return out



'''
def retrospectively_overlapping_atcs(df: pd.DataFrame,
                                     patientid_col: str = "PatientDurableKey",
                                     atc_col: str = "ATC",
                                     start_col: str = "StartInstant",
                                     discontinued_col: str = "DiscontinuedInstant",
                                     count_col: str = "active_overlapping_period_atc_codes_count",
                                     list_col: str = "active_overlapping_period_atc_codes_list",) -> pd.DataFrame:
    out = df.copy()
    out[start_col] = pd.to_datetime(out[start_col], errors="coerce")
    out[discontinued_col] = pd.to_datetime(out[discontinued_col], errors="coerce")

    out[count_col] = 0
    out[list_col] = ""

    for _, group in out.groupby(patientid_col, dropna=False, sort=False):
        group = group.dropna(subset=[start_col]).copy()

        if group.empty:
            continue

        atc_codes = group[atc_col].to_dict()
        start_times = group[start_col].to_dict()
        end_times = group[discontinued_col].to_dict()

        events = []

        for idx in group.index:
            start_time = start_times[idx]
            end_time = end_times[idx]

            if pd.isna(start_time):
                continue

            if pd.notna(end_time) and end_time <= start_time:
                continue

            events.append((start_time, "start", idx))

            if pd.notna(end_time):
                events.append((end_time, "end", idx))

        # End before start at same timestamp = end-exclusive intervals
        events.sort(key=lambda x: (x[0], 0 if x[1] == "end" else 1))

        active_indices = set()
        overlapping_atcs = {idx: set() for idx in group.index}

        for event_time, event_type, idx in events:
            current_atc = atc_codes[idx]

            if event_type == "start":
                for other_idx in active_indices:
                    other_atc = atc_codes[other_idx]

                    if pd.notna(current_atc) and pd.notna(other_atc) and other_atc != current_atc:
                        overlapping_atcs[idx].add(other_atc)
                        overlapping_atcs[other_idx].add(current_atc)

                active_indices.add(idx)

            else:
                active_indices.discard(idx)

        for idx, codes in overlapping_atcs.items():
            out.loc[idx, count_col] = len(codes)
            out.loc[idx, list_col] = ", ".join(sorted(map(str, codes))) if codes else ""

    out[count_col] = out[count_col].astype(np.int32)
    return out
'''