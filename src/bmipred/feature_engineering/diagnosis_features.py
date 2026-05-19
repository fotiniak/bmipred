#!/usr/bin/env python3

import numpy as np
import pandas as pd
import warnings
import heapq
from collections import Counter, defaultdict

warnings.filterwarnings('ignore') # deactivate warnings


def normalize_sks_codes(df: pd.DataFrame,
                        code_col: str = "SKSCode") -> pd.DataFrame:
    out = df.copy()
    code = out[code_col].astype("string").str.strip().str.upper()
    out[code_col] = code.replace({
        "": pd.NA, 
        "NAN": pd.NA, 
        "NONE": pd.NA,
        "NAT": pd.NA, 
        "<NA>": pd.NA, 
        "NA": pd.NA,
    })
    return out


def diagnosis_history_features(df: pd.DataFrame,
                               patientid_col: str = "PatientDurableKey",
                               code_col: str = "SKSCode",
                               start_col: str = "DiagnosisStartDate",
                               end_col: str = "DiagnosisEndDate",) -> pd.DataFrame:
    
    SCHIZOPHRENIA_PREFIXES = ("DF2",)
    BIPOLAR_PREFIXES = ("DF31",)
    CANCER_PREFIXES = ("DC", "DD0", "DD1", "DD2", "DD3", "DD4")
    
    SKS_PREFIX_SETS = {
        "active_schizophrenia": SCHIZOPHRENIA_PREFIXES,
        "active_bipolar": BIPOLAR_PREFIXES,
        "active_cancer": CANCER_PREFIXES,
    }

    out = df.copy()
    out[start_col] = pd.to_datetime(out[start_col], errors="coerce")
    out[end_col] = pd.to_datetime(out[end_col], errors="coerce")

    prefix_sets = {name: tuple(p.upper() for p in prefixes) for name, prefixes in SKS_PREFIX_SETS.items()}

    # Precompute prefix membership per distinct code -> avoids repeated startswith
    flags = {
        code: {name: code.startswith(prefixes) for name, prefixes in prefix_sets.items()}
        for code in out[code_col].dropna().unique()
    }

    n = len(out)
    idx_to_pos = pd.Series(np.arange(n), index=out.index)

    active_count   = np.zeros(n, dtype=np.int32)
    active_list    = np.full(n, "", dtype=object)
    past_count     = np.zeros(n, dtype=np.int32)
    past_list      = np.full(n, "", dtype=object)
    times_before   = np.zeros(n, dtype=np.int32)
    prefix_buffers = {name: np.zeros(n, dtype=np.int32) for name in prefix_sets}

    for _, g in out.groupby(patientid_col, dropna=False, sort=False):
        g = g.dropna(subset=[start_col]).sort_values(start_col)
        if g.empty:
            continue

        active = Counter()
        end_heap = []
        ever = set()
        prefix_counters = {name: 0 for name in prefix_sets}
        code_history = defaultdict(int)            # strictly-earlier count per code

        for t, batch in g.groupby(start_col, sort=False):

            # 1) Expire codes whose end < t (end-inclusive)
            while end_heap and end_heap[0][0] < t:
                _, code = heapq.heappop(end_heap)
                active[code] -= 1
                if active[code] == 0:
                    del active[code]
                for name, has_prefix in flags[code].items():
                    prefix_counters[name] -= has_prefix

            positions   = idx_to_pos.loc[batch.index].values
            batch_codes = batch[code_col].values
            batch_ends  = batch[end_col].values

            # 2) Read history BEFORE incrementing -> same-time ties share value
            for pos, code in zip(positions, batch_codes):
                if pd.notna(code):
                    times_before[pos] = code_history[code]

            # 3) Add current batch to active / ever / heap / prefix counters
            for code, end in zip(batch_codes, batch_ends):
                if pd.isna(code):
                    continue
                active[code] += 1
                ever.add(code)
                for name, has_prefix in flags[code].items():
                    prefix_counters[name] += has_prefix
                if pd.notna(end):
                    heapq.heappush(end_heap, (end, code))

            # 4) Write batch outputs (all rows in batch share these values)
            active_count[positions] = len(active)
            active_list[positions]  = ",".join(sorted(active))
            past_count[positions]   = len(ever)
            past_list[positions]    = ",".join(sorted(ever)) if ever else ""
            for name, count in prefix_counters.items():
                prefix_buffers[name][positions] = int(count > 0)

            # 5) Bump history AFTER processing batch
            for code in batch_codes:
                if pd.notna(code):
                    code_history[code] += 1

    out["active_sks_codes_count"]              = active_count
    out["active_sks_codes_csv"]                = active_list
    out["past_unique_sks_codes_count"]         = past_count
    out["past_sks_codes_csv"]                  = past_list
    out["times_current_sks_diagnosed_before"]  = times_before
    for name, buf in prefix_buffers.items():
        out[name] = buf

    return out
