from __future__ import annotations

import io
import json
import os
import random
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Dict, List

import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go

import folium
from streamlit_folium import st_folium

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader


# =========================
# CONFIG
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BUILD_TAG = "v1.6-compare-hover"

GEOJSON_COMUNAS_PATH = os.path.join(BASE_DIR, "comunas_corregimientos.geojson")
GEOJSON_BARRIOS_PATH = os.path.join(BASE_DIR, "barrios_veredas.geojson")

GEOJSON_COMUNAS_FIELD = "comuna_corregimiento"
GEOJSON_BARRIOS_FIELD = "barrio_vereda"

CALI_CENTER = (3.4516, -76.5320)

SOURCES = ["SIVIGILA", "COMISARIAS", "CASA_MATRIA", "EAE"]
MONTHS_ES = ["ENERO","FEBRERO","MARZO","ABRIL","MAYO","JUNIO","JULIO","AGOSTO","SEPTIEMBRE","OCTUBRE","NOVIEMBRE","DICIEMBRE"]

SOURCE_COLORS = {
    "CASA_MATRIA": "#c8a6d8",
    "COMISARIAS":  "#6a1bb1",
    "EAE":         "#79c34a",
    "SIVIGILA":    "#1f0a3d",
}

RELATIONS = ["PAREJA", "PADRE", "OTRO", "NINGUNO", "MADRE", "HIJO(A)", "FAMILIAR", "EX-PAREJA"]
AGE_KEYS = ["60+", "51-59", "40-50", "35-39", "30-34", "18-29"]
AGE_LEGEND = ["60 años y más", "51 a 59 años", "40 a 50 años", "35 a 39 años", "30 a 34 años", "18 a 29 años"]

AGE_COLORS = {
    "18-29": "#1f0a3d",
    "30-34": "#3a0a7a",
    "35-39": "#a44ac7",
    "40-50": "#c8a6d8",
    "51-59": "#e5cfe9",
    "60+":   "#79c34a",
}

AGE_LABELS_G3 = ["60 años y más", "51 a 59 años", "40 a 50 años", "35 a 39 años", "30 a 34 años", "18 a 29 años"]
VIOL_KEYS = ["VERBAL", "SEXUAL", "PSICOLOGICA", "FISICA", "ECONOMICA"]
VIOL_LEGEND = ["VERBAL", "SEXUAL", "PSICOLÓGICA", "FÍSICA", "ECONÓMICA Y ABANDONO"]

VIOL_COLORS = {
    "VERBAL": "#79c34a",
    "SEXUAL": "#a44ac7",
    "PSICOLOGICA": "#c8a6d8",
    "FISICA": "#6a1bb1",
    "ECONOMICA": "#1f0a3d",
}


CONTEXT_KEYS = ["VIF", "DELITOS_SEXUALES", "HOMICIDIOS_MUJER", "FEMINICIDIOS"]

CONTEXT_LABELS = {
    "VIF": "Violencia intrafamiliar (VIF)",
    "DELITOS_SEXUALES": "Delitos sexuales",
    "HOMICIDIOS_MUJER": "Homicidios de mujeres",
    "FEMINICIDIOS": "Feminicidios",
}

# Paleta consistente para Contexto (mismos tonos morado/verde del resto)
CONTEXT_COLORS = {
    "VIF": SOURCE_COLORS["SIVIGILA"],          # morado oscuro
    "DELITOS_SEXUALES": SOURCE_COLORS["COMISARIAS"],  # morado
    "HOMICIDIOS_MUJER": SOURCE_COLORS["CASA_MATRIA"], # lila
    "FEMINICIDIOS": SOURCE_COLORS["EAE"],      # verde
}

# =========================
# RIESGOS (mock determinístico)
# =========================

RISK_LEVELS = ["HAY_RIESGO", "NO_RIESGO"]
RISK_LEVEL_LABELS = {
    "HAY_RIESGO": "HAY RIESGO",
    "NO_RIESGO": "NO HAY RIESGO",
}

# Compatibilidad: si en session_state quedaron niveles antiguos (BAJO/MEDIO/ALTO/EXTREMO)
# los normalizamos al esquema binario solicitado.
LEGACY_RISK_LEVEL_MAP = {
    "BAJO": "NO_RIESGO",
    "MEDIO": "HAY_RIESGO",
    "ALTO": "HAY_RIESGO",
    "EXTREMO": "HAY_RIESGO",
}

def normalize_risk_levels(levels: List[str]) -> List[str]:
    if not levels:
        return []
    out: List[str] = []
    for lvl in levels:
        lvl2 = LEGACY_RISK_LEVEL_MAP.get(lvl, lvl)
        if lvl2 in RISK_LEVELS and lvl2 not in out:
            out.append(lvl2)
    return out

RISK_TYPES = [
    "VIF_REINCIDENCIA",
    "DELITO_SEXUAL",
    "RIESGO_FEMINICIDA",
    "RIESGO_LETALIDAD",
    "RIESGO_SALUD_MENTAL",
]
RISK_TYPE_LABELS = {
    "VIF_REINCIDENCIA": "Reincidencia VIF",
    "DELITO_SEXUAL": "Delito sexual",
    "RIESGO_FEMINICIDA": "Riesgo feminicida",
    "RIESGO_LETALIDAD": "Riesgo de letalidad",
    "RIESGO_SALUD_MENTAL": "Riesgo salud mental",
}


# =========================
# UTILIDADES
# =========================

def clamp_int(x: float, lo: int = 0) -> int:
    return max(lo, int(round(x)))

def dirichlet_like(rng: random.Random, weights: List[float], sharpness: float = 25.0) -> List[float]:
    alphas = [max(0.001, w) * sharpness for w in weights]
    samples = [rng.gammavariate(a, 1.0) for a in alphas]
    s = sum(samples) or 1.0
    return [v / s for v in samples]

def to_percentages(counts: List[int]) -> List[int]:
    total = sum(counts)
    if total <= 0:
        return [0] * len(counts)

    raw = [c * 100.0 / total for c in counts]
    rounded = [int(round(x)) for x in raw]
    diff = 100 - sum(rounded)

    residuals = [raw[i] - int(raw[i]) for i in range(len(raw))]
    order = sorted(range(len(counts)), key=lambda i: residuals[i], reverse=(diff > 0))

    i = 0
    while diff != 0 and i < 10000:
        idx = order[i % len(order)]
        if diff > 0:
            rounded[idx] += 1
            diff -= 1
        else:
            if rounded[idx] > 0:
                rounded[idx] -= 1
                diff += 1
        i += 1

    return rounded

def fmt_int(n: int) -> str:
    return f"{n:,}".replace(",", ".")


# =========================
# MODELO DE DATOS
# =========================

@dataclass
class DailyRecord:
    d: date
    counts_sources: Dict[str, int]
    rel_age_counts: Dict[str, List[int]]
    age_viol_counts: Dict[str, List[int]]
    context_counts: Dict[str, int]

class DataModel:
    def __init__(self):
        self.records: List[DailyRecord] = []
        self._generate_mock_data()

    def _generate_mock_data(self):
        rng = random.Random(2025)

        start = date(2025, 1, 1)
        end = date(2025, 12, 16)
        n_days = (end - start).days + 1

        base_src = {"SIVIGILA": 12, "COMISARIAS": 11, "CASA_MATRIA": 4, "EAE": 2}

        rel_age_profile = {
            "PAREJA":    [0.06, 0.07, 0.18, 0.13, 0.18, 0.38],
            "PADRE":     [0.06, 0.06, 0.16, 0.10, 0.10, 0.52],
            "OTRO":      [0.20, 0.05, 0.20, 0.10, 0.12, 0.33],
            "NINGUNO":   [0.08, 0.06, 0.12, 0.12, 0.14, 0.48],
            "MADRE":     [0.07, 0.08, 0.14, 0.06, 0.12, 0.53],
            "HIJO(A)":   [0.55, 0.22, 0.15, 0.05, 0.02, 0.01],
            "FAMILIAR":  [0.28, 0.18, 0.16, 0.07, 0.08, 0.23],
            "EX-PAREJA": [0.03, 0.06, 0.16, 0.14, 0.22, 0.39],
        }

        age_viol_profile = {
            "60 años y más": [0.05, 0.05, 0.40, 0.30, 0.20],
            "51 a 59 años":  [0.07, 0.08, 0.34, 0.44, 0.07],
            "40 a 50 años":  [0.06, 0.08, 0.33, 0.46, 0.07],
            "35 a 39 años":  [0.05, 0.08, 0.31, 0.49, 0.07],
            "30 a 34 años":  [0.05, 0.09, 0.29, 0.51, 0.06],
            "18 a 29 años":  [0.03, 0.17, 0.22, 0.54, 0.04],
        }

        def seasonal_factor(dt: date) -> float:
            if dt.month in (7, 8): return 1.25
            if dt.month == 11: return 1.10
            if dt.month == 12: return 0.65
            return 1.0

        def daily_case_pool(total_src: int) -> int:
            return max(5, int(total_src * rng.uniform(0.55, 0.85)))

        key_to_label_g3 = {
            "60+": "60 años y más",
            "51-59": "51 a 59 años",
            "40-50": "40 a 50 años",
            "35-39": "35 a 39 años",
            "30-34": "30 a 34 años",
            "18-29": "18 a 29 años",
        }

        self.records.clear()

        for i in range(n_days):
            dt = start + timedelta(days=i)
            f = seasonal_factor(dt)
            spike = 1.0 + (0.5 if rng.random() < 0.03 else 0.0)

            counts_sources = {}
            for s in SOURCES:
                val = (base_src[s] * f * spike) + rng.randint(0, base_src[s] + 6)
                counts_sources[s] = clamp_int(val)

            total_src = sum(counts_sources.values())
            pool = daily_case_pool(total_src)

            rel_weights = [0.22, 0.08, 0.12, 0.14, 0.10, 0.06, 0.18, 0.10]
            rel_props = dirichlet_like(rng, rel_weights, sharpness=18.0)
            rel_counts = [max(0, int(round(pool * p))) for p in rel_props]

            diff = pool - sum(rel_counts)
            for _ in range(abs(diff)):
                idx = rng.randrange(len(rel_counts))
                if diff > 0: rel_counts[idx] += 1
                else:
                    if rel_counts[idx] > 0: rel_counts[idx] -= 1

            rel_age_counts = {}
            for rel, n_rel in zip(RELATIONS, rel_counts):
                prof = rel_age_profile[rel]
                props = dirichlet_like(rng, prof, sharpness=30.0)
                age_counts = [max(0, int(round(n_rel * p))) for p in props]

                d2 = n_rel - sum(age_counts)
                for _ in range(abs(d2)):
                    j = rng.randrange(len(age_counts))
                    if d2 > 0: age_counts[j] += 1
                    else:
                        if age_counts[j] > 0: age_counts[j] -= 1

                rel_age_counts[rel] = age_counts

            age_weights_for_pool = [0.10, 0.10, 0.16, 0.12, 0.16, 0.36]
            age_props = dirichlet_like(rng, age_weights_for_pool, sharpness=22.0)
            age_pool_counts = [max(0, int(round(pool * p))) for p in age_props]

            d3 = pool - sum(age_pool_counts)
            for _ in range(abs(d3)):
                j = rng.randrange(len(age_pool_counts))
                if d3 > 0: age_pool_counts[j] += 1
                else:
                    if age_pool_counts[j] > 0: age_pool_counts[j] -= 1

            age_viol_counts = {}
            for age_key, n_age in zip(AGE_KEYS, age_pool_counts):
                label = key_to_label_g3[age_key]
                prof = age_viol_profile[label]
                props = dirichlet_like(rng, prof, sharpness=35.0)
                viol_counts = [max(0, int(round(n_age * p))) for p in props]

                d4 = n_age - sum(viol_counts)
                for _ in range(abs(d4)):
                    k = rng.randrange(len(viol_counts))
                    if d4 > 0: viol_counts[k] += 1
                    else:
                        if viol_counts[k] > 0: viol_counts[k] -= 1

                age_viol_counts[label] = viol_counts

            context_counts = {
                "VIF": max(0, int(rng.uniform(6, 18) * f)),
                "DELITOS_SEXUALES": max(0, int(rng.uniform(1, 6) * f)),
                "HOMICIDIOS_MUJER": 1 if rng.random() < (0.06 * f) else 0,
                "FEMINICIDIOS": 1 if rng.random() < (0.015 * f) else 0,
            }

            self.records.append(
                DailyRecord(
                    d=dt,
                    counts_sources=counts_sources,
                    rel_age_counts=rel_age_counts,
                    age_viol_counts=age_viol_counts,
                    context_counts=context_counts,
                )
            )

    def aggregate(self, d_from: date, d_to: date):
        rows = [r for r in self.records if d_from <= r.d <= d_to]
        if not rows:
            return {
                "kpis": {},
                "monthly": {s: [0]*12 for s in SOURCES},
                "total_monthly": [0]*12,
                "distribution": {s: 0 for s in SOURCES},
                "meta": {"days": 0, "peak_month": None, "peak_value": 0},
                "g2_pct": {rel: [0]*len(AGE_KEYS) for rel in RELATIONS},
                "g3_pct": {age: [0]*len(VIOL_KEYS) for age in AGE_LABELS_G3},
                "context_totals": {k: 0 for k in CONTEXT_KEYS},
                "context_monthly": {k: [0]*12 for k in CONTEXT_KEYS},
            }

        totals_src = {s: 0 for s in SOURCES}
        monthly = {s: [0]*12 for s in SOURCES}
        total_monthly = [0]*12

        for r in rows:
            mi = r.d.month - 1
            for s in SOURCES:
                totals_src[s] += r.counts_sources[s]
                monthly[s][mi] += r.counts_sources[s]
                total_monthly[mi] += r.counts_sources[s]

        total_all = sum(totals_src.values())
        days = len(rows)
        avg_daily = total_all / days if days else 0.0

        peak_value = max(total_monthly) if total_monthly else 0
        peak_month_idx = total_monthly.index(peak_value) if peak_value > 0 else None
        peak_month = MONTHS_ES[peak_month_idx] if peak_month_idx is not None else None

        g2_counts = {rel: [0]*len(AGE_KEYS) for rel in RELATIONS}
        for r in rows:
            for rel in RELATIONS:
                vec = r.rel_age_counts[rel]
                for i in range(len(AGE_KEYS)):
                    g2_counts[rel][i] += vec[i]
        g2_pct = {rel: to_percentages(g2_counts[rel]) for rel in RELATIONS}

        g3_counts = {age: [0]*len(VIOL_KEYS) for age in AGE_LABELS_G3}
        for r in rows:
            for age in AGE_LABELS_G3:
                vec = r.age_viol_counts[age]
                for i in range(len(VIOL_KEYS)):
                    g3_counts[age][i] += vec[i]
        g3_pct = {age: to_percentages(g3_counts[age]) for age in AGE_LABELS_G3}

        context_totals = {k: 0 for k in CONTEXT_KEYS}
        context_monthly = {k: [0]*12 for k in CONTEXT_KEYS}
        for r in rows:
            mi = r.d.month - 1
            for kx in CONTEXT_KEYS:
                v = r.context_counts.get(kx, 0)
                context_totals[kx] += v
                context_monthly[kx][mi] += v

        return {
            "kpis": {
                "TOTAL_ATENCIONES": total_all,
                "PROMEDIO_DIARIO": avg_daily,
                "SIVIGILA": totals_src["SIVIGILA"],
                "COMISARIAS": totals_src["COMISARIAS"],
                "CASA_MATRIA": totals_src["CASA_MATRIA"],
                "EAE": totals_src["EAE"],
            },
            "monthly": monthly,
            "total_monthly": total_monthly,
            "distribution": totals_src,
            "meta": {"days": days, "peak_month": peak_month, "peak_value": peak_value},
            "g2_pct": g2_pct,
            "g3_pct": g3_pct,
            "context_totals": context_totals,
            "context_monthly": context_monthly,
        }

    def aggregate_by_area_keys(self, d_from: date, d_to: date, metric_key: str, area_keys: List[str]) -> Dict[str, int]:
        if not area_keys:
            return {}

        seed = (
            int(d_from.strftime("%Y%m%d")) * 31
            + int(d_to.strftime("%Y%m%d")) * 17
            + sum(ord(c) for c in metric_key)
            + len(area_keys) * 13
        )
        rng = random.Random(seed)

        agg = self.aggregate(d_from, d_to)
        base_total = int(agg.get("kpis", {}).get("TOTAL_ATENCIONES", 0))
        ctx_total = int(agg.get("context_totals", {}).get(metric_key, 0)) if metric_key in CONTEXT_KEYS else 0

        base = ctx_total if metric_key in CONTEXT_KEYS else base_total
        base = max(base, 1)

        weights = [rng.uniform(0.6, 1.6) for _ in area_keys]
        s = sum(weights) or 1.0
        raw = [base * (w / s) for w in weights]
        vals = [int(round(x)) for x in raw]

        diff = base - sum(vals)
        for _ in range(abs(diff)):
            i = rng.randrange(len(vals))
            if diff > 0:
                vals[i] += 1
            else:
                if vals[i] > 0:
                    vals[i] -= 1

        return {k: v for k, v in zip(area_keys, vals)}

    def aggregate_risks_by_area_keys(
        self,
        d_from: date,
        d_to: date,
        selected_types: List[str],
        selected_levels: List[str],
        area_keys: List[str],
    ) -> Dict[str, int]:
        """Distribuye riesgos (mock determinístico) por áreas.

        - Calcula primero el total de riesgos del rango y tipos (aggregate_risks).
        - Se queda con los niveles seleccionados (p.ej. ALTO+EXTREMO).
        - Distribuye ese total entre las áreas de forma determinística.

        Retorna {area_key: valor}.
        """
        if not area_keys:
            return {}

        selected_types = selected_types or list(RISK_TYPES)
        selected_levels = normalize_risk_levels(selected_levels or list(RISK_LEVELS))

        risks = self.aggregate_risks(d_from, d_to, selected_types=selected_types)
        totals_by_level = risks.get("totals_by_level", {})
        base = sum(int(totals_by_level.get(lvl, 0)) for lvl in selected_levels)
        base = max(int(base), 0)

        # Seed determinístico por (rango, tipos, niveles, tamaño)
        seed = (
            int(d_from.strftime("%Y%m%d")) * 41
            + int(d_to.strftime("%Y%m%d")) * 29
            + sum(ord(c) for c in "|".join(selected_types))
            + sum(ord(c) for c in "|".join(selected_levels))
            + len(area_keys) * 17
            + 2025
        )
        rng = random.Random(seed)

        if base <= 0:
            return {k: 0 for k in area_keys}

        # Pesos por área (variación moderada)
        weights = [rng.uniform(0.6, 1.8) for _ in area_keys]
        s = sum(weights) or 1.0
        raw = [base * (w / s) for w in weights]
        vals = [int(round(x)) for x in raw]

        diff = base - sum(vals)
        for _ in range(abs(diff)):
            i = rng.randrange(len(vals))
            if diff > 0:
                vals[i] += 1
            else:
                if vals[i] > 0:
                    vals[i] -= 1

        return {k: v for k, v in zip(area_keys, vals)}


    def aggregate_risks(self, d_from: date, d_to: date, selected_types: List[str] | None = None) -> dict:
        """
        Agregado mock determinístico de riesgos.
        - No depende del GeoJSON.
        - Es determinístico por (rango de fechas, tipos seleccionados) para que al mover fechas cambie.

        Retorna:
          - totals_by_level: {nivel: total}
          - totals_by_type: {tipo: total}
          - matrix: {tipo: {nivel: total}}
          - monthly_by_level: {nivel: [12]}
        """
        if selected_types is None or len(selected_types) == 0:
            selected_types = list(RISK_TYPES)
        # Asegura que no queden niveles legacy en memoria/caché
        _ = normalize_risk_levels(list(RISK_LEVELS))

        # Base: lo amarramos al volumen general del rango para que tenga coherencia
        agg = self.aggregate(d_from, d_to)
        base_total = int(agg.get("kpis", {}).get("TOTAL_ATENCIONES", 0))
        days = int(agg.get("meta", {}).get("days", 0))

        # Seed determinístico
        seed = (
            int(d_from.strftime("%Y%m%d")) * 97
            + int(d_to.strftime("%Y%m%d")) * 53
            + sum(ord(c) for c in "|".join(selected_types))
            + 1337
        )
        rng = random.Random(seed)

        # Total de "eventos de riesgo" (proporción del total)
        # Si hay pocos casos, garantizamos un mínimo para que el dashboard no quede vacío
        risk_pool = max(20, int(base_total * rng.uniform(0.35, 0.65)))
        if days <= 0:
            risk_pool = 0

        # Distribución binaria: Hay riesgo vs No hay riesgo
        level_weights = {
            "HAY_RIESGO": 0.45,
            "NO_RIESGO": 0.55,
        }

        # Preferencias por tipo (puedes ajustar)
        type_weights = {
            "VIF_REINCIDENCIA": 0.28,
            "DELITO_SEXUAL": 0.16,
            "RIESGO_FEMINICIDA": 0.20,
            "RIESGO_LETALIDAD": 0.22,
            "RIESGO_SALUD_MENTAL": 0.14,
        }

        # Filtramos a los seleccionados y renormalizamos
        tw = [type_weights.get(t, 0.1) for t in selected_types]
        tw_sum = sum(tw) or 1.0
        tw = [v / tw_sum for v in tw]

        # Repartimos el pool entre tipos
        type_props = dirichlet_like(rng, tw, sharpness=22.0)
        type_counts = [max(0, int(round(risk_pool * p))) for p in type_props]
        diff = risk_pool - sum(type_counts)
        for _ in range(abs(diff)):
            i = rng.randrange(len(type_counts))
            if diff > 0:
                type_counts[i] += 1
            else:
                if type_counts[i] > 0:
                    type_counts[i] -= 1

        matrix: Dict[str, Dict[str, int]] = {t: {lvl: 0 for lvl in RISK_LEVELS} for t in selected_types}
        totals_by_level = {lvl: 0 for lvl in RISK_LEVELS}
        totals_by_type = {t: 0 for t in selected_types}

        # Dentro de cada tipo, distribuimos por nivel con leve variación
        base_level_vec = [level_weights.get(l, 0.0) for l in RISK_LEVELS]
        for t, n_t in zip(selected_types, type_counts):
            # Variación suave por tipo
            props = dirichlet_like(rng, base_level_vec, sharpness=28.0)
            lvl_counts = [max(0, int(round(n_t * p))) for p in props]
            d2 = n_t - sum(lvl_counts)
            for _ in range(abs(d2)):
                j = rng.randrange(len(lvl_counts))
                if d2 > 0:
                    lvl_counts[j] += 1
                else:
                    if lvl_counts[j] > 0:
                        lvl_counts[j] -= 1

            for lvl, v in zip(RISK_LEVELS, lvl_counts):
                matrix[t][lvl] += v
                totals_by_level[lvl] += v
                totals_by_type[t] += v

        # Tendencia mensual (mock coherente con seasonality del modelo)
        monthly_by_level = {lvl: [0] * 12 for lvl in RISK_LEVELS}
        if days > 0:
            # Usamos el total mensual del agregado como señal
            total_monthly = agg.get("total_monthly", [0] * 12)
            for mi in range(12):
                month_base = max(0, int(total_monthly[mi] * rng.uniform(0.25, 0.55)))
                # Repartimos month_base por niveles
                props_m = dirichlet_like(rng, base_level_vec, sharpness=24.0)
                vals = [max(0, int(round(month_base * p))) for p in props_m]
                d3 = month_base - sum(vals)
                for _ in range(abs(d3)):
                    j = rng.randrange(len(vals))
                    if d3 > 0:
                        vals[j] += 1
                    else:
                        if vals[j] > 0:
                            vals[j] -= 1
                for lvl, v in zip(RISK_LEVELS, vals):
                    monthly_by_level[lvl][mi] += v

        return {
            "selected_types": selected_types,
            "risk_pool": risk_pool,
            "totals_by_level": totals_by_level,
            "totals_by_type": totals_by_type,
            "matrix": matrix,
            "monthly_by_level": monthly_by_level,
        }


# =========================
# CHARTS (Matplotlib)
# =========================

def fig_grouped_bars(monthly: Dict[str, List[int]]):
    fig, ax = plt.subplots(figsize=(10, 4))
    x = list(range(12))
    width = 0.18
    series = ["CASA_MATRIA", "COMISARIAS", "EAE", "SIVIGILA"]
    offsets = [-1.5, -0.5, 0.5, 1.5]

    for s, off in zip(series, offsets):
        ax.bar([i + off * width for i in x], monthly[s], width=width, label=s, color=SOURCE_COLORS[s])

    ax.set_title("Gráfico 1. Cantidad de mujeres atendidas por organismo y mes")
    ax.set_ylabel("Número de casos")
    ax.set_xticks(x)
    ax.set_xticklabels(MONTHS_ES, rotation=45, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=8)
    fig.tight_layout()
    return fig

# === Plotly version: grouped bars with hover (same palette) ===
def fig_grouped_bars_hover(monthly: Dict[str, List[int]]):
    """Barras agrupadas interactivas (hover): mujeres atendidas por organismo y mes."""
    x = MONTHS_ES

    # Orden fijo para que sea consistente con leyenda/paleta
    series = ["CASA_MATRIA", "COMISARIAS", "EAE", "SIVIGILA"]
    series_labels = {
        "CASA_MATRIA": "CASA MATRIA",
        "COMISARIAS": "COMISARÍAS",
        "EAE": "EAE",
        "SIVIGILA": "SIVIGILA",
    }

    fig = go.Figure()
    for s in series:
        y = [int(v) for v in (monthly.get(s, [0] * 12) or [0] * 12)]
        fig.add_trace(
            go.Bar(
                x=x,
                y=y,
                name=series_labels.get(s, s),
                marker=dict(color=SOURCE_COLORS.get(s, "#64748b"), line=dict(color="white", width=0.6)),
                hovertemplate="%{x}<br>Organismo: "
                + series_labels.get(s, s)
                + "<br>Mujeres atendidas: %{y:,}<extra></extra>",
            )
        )

    fig.update_layout(
        title="Gráfico 1. Cantidad de mujeres atendidas por organismo y mes",
        barmode="group",
        height=420,
        margin=dict(l=0, r=0, t=60, b=0),
        hovermode="x unified",
        legend=dict(orientation="v", x=1.02, y=0.5, xanchor="left", yanchor="middle"),
    )
    fig.update_xaxes(tickangle=-45)
    fig.update_yaxes(title_text="Número de casos", gridcolor="rgba(148,163,184,0.25)")
    return fig

def fig_total_line_static(total_monthly: List[int]):
    fig, ax = plt.subplots(figsize=(10, 3.8))
    x = list(range(12))
    ax.plot(x, total_monthly, marker="o")
    ax.set_title("Tendencia mensual (Total atenciones)")
    ax.set_xticks(x)
    ax.set_xticklabels(MONTHS_ES, rotation=45, ha="right", fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig

# === Plotly version: interactive two-line chart with hover ===
def fig_total_line_hover(total_monthly: List[int]):
    """Dos líneas interactivas (hover): Total atenciones vs Mujeres atendidas (estimado)."""
    x = MONTHS_ES

    month_factors = [0.84, 0.83, 0.835, 0.832, 0.838, 0.834, 0.845, 0.846, 0.836, 0.839, 0.833, 0.825]
    women_monthly = [int(round(v * month_factors[i])) for i, v in enumerate(total_monthly)]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=total_monthly,
            mode="lines+markers",
            name="Total atenciones",
            line=dict(color=SOURCE_COLORS["SIVIGILA"], width=3),
            marker=dict(color=SOURCE_COLORS["SIVIGILA"], size=7),
            hovertemplate="%{x}<br>Total atenciones: %{y:,}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=women_monthly,
            mode="lines+markers",
            name="Mujeres atendidas (estimado)",
            line=dict(color=SOURCE_COLORS["COMISARIAS"], width=3),
            marker=dict(color=SOURCE_COLORS["COMISARIAS"], size=7),
            hovertemplate="%{x}<br>Mujeres atendidas (est.): %{y:,}<extra></extra>",
        )
    )

    # Formato miles con punto (ES) en hover: reemplazo simple en frontend no aplica;
    # dejamos coma de Plotly, pero el valor es claro. Si quieres punto, lo ajustamos luego.
    fig.update_layout(
        title="Tendencia mensual: total atenciones vs mujeres atendidas",
        margin=dict(l=0, r=0, t=50, b=0),
        height=360,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(tickangle=-45)
    fig.update_yaxes(title_text="Número de casos")
    return fig


def fig_donut(distribution: Dict[str, int]):
    fig, ax = plt.subplots(figsize=(6, 4))

    labels = list(distribution.keys())
    values = [int(distribution.get(k, 0)) for k in labels]
    colors = [SOURCE_COLORS.get(k, None) for k in labels]

    total = sum(values)
    if total <= 0:
        # evita división por cero y mantiene la visual aunque no haya datos
        values = [1 for _ in values]
        total = sum(values)

    def autopct_func(pct: float) -> str:
        # muestra solo segmentos visibles
        return f"{pct:.1f}%" if pct >= 1 else ""

    wedges, texts, autotexts = ax.pie(
        values,
        colors=colors,
        startangle=90,
        autopct=autopct_func,
        pctdistance=0.78,
        wedgeprops={"width": 0.35, "edgecolor": "white"},
    )

    # Estilo de los porcentajes para que SIEMPRE se vean
    for t in autotexts:
        t.set_fontsize(9)
        t.set_weight("bold")
        t.set_color("white")
        # borde/halo oscuro para legibilidad en segmentos claros
        try:
            import matplotlib.patheffects as pe
            t.set_path_effects([pe.withStroke(linewidth=2, foreground="#111827")])
        except Exception:
            pass

    ax.set_title("Distribución por organismo")
    ax.legend(
        wedges,
        labels,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
        fontsize=8,
    )
    ax.axis("equal")

    fig.tight_layout()
    return fig


# =========================
# DONUT INTERACTIVO (Plotly)
# =========================

def fig_donut_hover(distribution: Dict[str, int]):
    """Dona interactiva: NO muestra etiquetas; porcentajes aparecen al pasar el mouse."""
    labels = list(distribution.keys())
    values = [int(distribution.get(k, 0)) for k in labels]
    colors = [SOURCE_COLORS.get(k, "#64748b") for k in labels]

    # Evita gráfico vacío
    if sum(values) <= 0:
        values = [1 for _ in values]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.65,
                marker=dict(colors=colors, line=dict(color="white", width=1)),
                textinfo="none",  # <- no texto visible
                hovertemplate="%{label}<br>%{percent:.1%} (%{value})<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title_text="Distribución por organismo",
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(orientation="v", x=1.02, y=0.5, xanchor="left", yanchor="middle"),
        height=360,
    )
    return fig

# =========================
# G2 / G3 INTERACTIVOS (Plotly - hover)
# =========================

def fig_g2_hover(g2_pct: Dict[str, List[int]]):
    """Barras horizontales apiladas 100% (hover): relación -> % por edad."""
    y = RELATIONS  # categorías
    fig = go.Figure()

    # Para apilado 100%, g2_pct ya viene en % enteros que suman 100 por fila.
    for idx, age_key in enumerate(AGE_KEYS):
        vals = [int(g2_pct.get(rel, [0]*len(AGE_KEYS))[idx]) for rel in y]
        fig.add_trace(
            go.Bar(
                y=y,
                x=vals,
                orientation="h",
                name=AGE_LEGEND[idx],
                marker=dict(color=AGE_COLORS[age_key]),
                hovertemplate="Relación: %{y}<br>" + AGE_LEGEND[idx] + ": %{x}%<extra></extra>",
            )
        )

    fig.update_layout(
        title="Gráfico 2. Distribución de rangos de edad según la relación familiar con la víctima",
        barmode="stack",
        height=520,
        margin=dict(l=0, r=0, t=55, b=0),
        legend=dict(orientation="v", x=1.02, y=0.5, xanchor="left", yanchor="middle"),
    )
    fig.update_xaxes(title_text="Porcentaje", range=[0, 100], gridcolor="rgba(148,163,184,0.25)")
    fig.update_yaxes(autorange="reversed")
    return fig


def fig_g3_hover(g3_pct: Dict[str, List[int]]):
    """Barras horizontales apiladas 100% (hover): edad -> % por tipo de violencia."""
    y = AGE_LABELS_G3
    fig = go.Figure()

    for idx, key in enumerate(VIOL_KEYS):
        vals = [int(g3_pct.get(age, [0]*len(VIOL_KEYS))[idx]) for age in y]
        fig.add_trace(
            go.Bar(
                y=y,
                x=vals,
                orientation="h",
                name=VIOL_LEGEND[idx],
                marker=dict(color=VIOL_COLORS[key]),
                hovertemplate="Edad: %{y}<br>" + VIOL_LEGEND[idx] + ": %{x}%<extra></extra>",
            )
        )

    fig.update_layout(
        title="Gráfico 3. Distribución de rangos de edad según el tipo de violencia",
        barmode="stack",
        height=520,
        margin=dict(l=0, r=0, t=55, b=0),
        legend=dict(orientation="v", x=1.02, y=0.5, xanchor="left", yanchor="middle"),
    )
    fig.update_xaxes(title_text="Porcentaje", range=[0, 100], gridcolor="rgba(148,163,184,0.25)")
    fig.update_yaxes(autorange="reversed")
    return fig


# === NUEVA FUNCIÓN DE COMPARACIÓN DE DOS MESES POR ORGANISMO ===
def fig_compare_sources_two_months(monthly: Dict[str, List[int]], month_a_idx: int, month_b_idx: int):
    """Compara atenciones por organismo entre dos meses (barras lado a lado)."""
    fig, ax = plt.subplots(figsize=(10, 4))

    sources = ["SIVIGILA", "COMISARIAS", "CASA_MATRIA", "EAE"]
    labels = ["SIVIGILA", "COMISARÍAS", "CASA MATRIA", "EAE"]

    a_vals = [int(monthly.get(s, [0]*12)[month_a_idx]) for s in sources]
    b_vals = [int(monthly.get(s, [0]*12)[month_b_idx]) for s in sources]

    x = list(range(len(sources)))
    width = 0.36

    ax.bar(
        [i - width/2 for i in x],
        a_vals,
        width=width,
        label=MONTHS_ES[month_a_idx],
        color=SOURCE_COLORS["SIVIGILA"],  # morado oscuro
        edgecolor="white",
        linewidth=0.6,
    )
    ax.bar(
        [i + width/2 for i in x],
        b_vals,
        width=width,
        label=MONTHS_ES[month_b_idx],
        color=SOURCE_COLORS["EAE"],  # verde
        edgecolor="white",
        linewidth=0.6,
    )

    ax.set_title("Comparación: mujeres atendidas por organismo (mes vs mes)")
    ax.set_ylabel("Número de casos")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig

# === Plotly version: interactive two-month comparison with hover ===
def fig_compare_sources_two_months_hover(monthly: Dict[str, List[int]], month_a_idx: int, month_b_idx: int):
    """Comparación interactiva (hover): organismo en dos meses."""
    sources = ["SIVIGILA", "COMISARIAS", "CASA_MATRIA", "EAE"]
    labels_x = ["SIVIGILA", "COMISARÍAS", "CASA MATRIA", "EAE"]

    a_vals = [int(monthly.get(s, [0]*12)[month_a_idx]) for s in sources]
    b_vals = [int(monthly.get(s, [0]*12)[month_b_idx]) for s in sources]

    name_a = MONTHS_ES[month_a_idx]
    name_b = MONTHS_ES[month_b_idx]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=labels_x,
            y=a_vals,
            name=name_a,
            marker=dict(color=SOURCE_COLORS["SIVIGILA"], line=dict(color="white", width=0.6)),
            hovertemplate="Organismo: %{x}<br>" + name_a + ": %{y:,}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=labels_x,
            y=b_vals,
            name=name_b,
            marker=dict(color=SOURCE_COLORS["EAE"], line=dict(color="white", width=0.6)),
            hovertemplate="Organismo: %{x}<br>" + name_b + ": %{y:,}<extra></extra>",
        )
    )

    fig.update_layout(
        title="Comparación: mujeres atendidas por organismo (mes vs mes)",
        barmode="group",
        height=420,
        margin=dict(l=0, r=0, t=55, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_yaxes(title_text="Número de casos", gridcolor="rgba(148,163,184,0.25)")
    return fig

# === NUEVA FUNCIÓN DE COMPARACIÓN DE DOS RANGOS POR ORGANISMO ===
def fig_compare_sources_two_ranges(agg_a: dict, agg_b: dict, label_a: str, label_b: str):
    """Compara atenciones por organismo entre dos rangos (barras lado a lado)."""
    fig, ax = plt.subplots(figsize=(10, 4))

    sources = ["SIVIGILA", "COMISARIAS", "CASA_MATRIA", "EAE"]
    labels = ["SIVIGILA", "COMISARÍAS", "CASA MATRIA", "EAE"]

    k_a = agg_a.get("kpis", {})
    k_b = agg_b.get("kpis", {})

    a_vals = [int(k_a.get(s, 0)) for s in sources]
    b_vals = [int(k_b.get(s, 0)) for s in sources]

    x = list(range(len(sources)))
    width = 0.36

    ax.bar(
        [i - width/2 for i in x],
        a_vals,
        width=width,
        label=label_a,
        color=SOURCE_COLORS["SIVIGILA"],  # morado oscuro
        edgecolor="white",
        linewidth=0.6,
    )
    ax.bar(
        [i + width/2 for i in x],
        b_vals,
        width=width,
        label=label_b,
        color=SOURCE_COLORS["EAE"],  # verde
        edgecolor="white",
        linewidth=0.6,
    )

    ax.set_title("Comparación: mujeres atendidas por organismo (rango vs rango)")
    ax.set_ylabel("Número de casos")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig

# === Plotly version: interactive range-vs-range comparison with hover ===
def fig_compare_sources_two_ranges_hover(agg_a: dict, agg_b: dict, label_a: str, label_b: str):
    """Comparación interactiva (hover): organismo en dos rangos."""
    sources = ["SIVIGILA", "COMISARIAS", "CASA_MATRIA", "EAE"]
    labels_x = ["SIVIGILA", "COMISARÍAS", "CASA MATRIA", "EAE"]

    k_a = agg_a.get("kpis", {}) or {}
    k_b = agg_b.get("kpis", {}) or {}

    a_vals = [int(k_a.get(s, 0)) for s in sources]
    b_vals = [int(k_b.get(s, 0)) for s in sources]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=labels_x,
            y=a_vals,
            name=label_a,
            marker=dict(color=SOURCE_COLORS["SIVIGILA"], line=dict(color="white", width=0.6)),
            hovertemplate="Organismo: %{x}<br>Periodo A: %{y:,}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=labels_x,
            y=b_vals,
            name=label_b,
            marker=dict(color=SOURCE_COLORS["EAE"], line=dict(color="white", width=0.6)),
            hovertemplate="Organismo: %{x}<br>Periodo B: %{y:,}<extra></extra>",
        )
    )

    fig.update_layout(
        title="Comparación: mujeres atendidas por organismo (rango vs rango)",
        barmode="group",
        height=420,
        margin=dict(l=0, r=0, t=55, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_yaxes(title_text="Número de casos", gridcolor="rgba(148,163,184,0.25)")
    return fig

def fig_g2(g2_pct: Dict[str, List[int]]):
    fig, ax = plt.subplots(figsize=(10, 5))
    y = list(range(len(RELATIONS)))
    left = [0] * len(RELATIONS)
    for idx, age_key in enumerate(AGE_KEYS):
        vals = [g2_pct[rel][idx] for rel in RELATIONS]
        bars = ax.barh(
            y,
            vals,
            left=left,
            color=AGE_COLORS[age_key],
            edgecolor="none",
            height=0.6,
            label=AGE_LEGEND[idx],
        )

        # Agregar porcentajes dentro de cada segmento (si es visible)
        for i, (bar, v) in enumerate(zip(bars, vals)):
            if v >= 5:  # solo mostramos si el segmento es suficientemente grande
                ax.text(
                    left[i] + v / 2,
                    bar.get_y() + bar.get_height() / 2,
                    f"{v}%",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white",
                    fontweight="bold",
                )

        left = [l + v for l, v in zip(left, vals)]
    ax.set_xlim(0, 100)
    ax.set_yticks(y)
    ax.set_yticklabels(RELATIONS)
    ax.set_xlabel("Porcentaje")
    ax.set_title("Gráfico 2. Distribución de rangos de edad según la relación familiar con la víctima")
    ax.grid(axis="x", alpha=0.2)
    ax.invert_yaxis()
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=8)
    fig.tight_layout()
    return fig

def fig_g3(g3_pct: Dict[str, List[int]]):
    fig, ax = plt.subplots(figsize=(10, 5))
    rows = AGE_LABELS_G3
    y = list(range(len(rows)))
    left = [0] * len(rows)
    for idx, key in enumerate(VIOL_KEYS):
        vals = [g3_pct[age][idx] for age in rows]
        bars = ax.barh(
            y,
            vals,
            left=left,
            color=VIOL_COLORS[key],
            edgecolor="none",
            height=0.6,
            label=VIOL_LEGEND[idx],
        )

        # Agregar porcentajes dentro de cada segmento (si es visible)
        for i, (bar, v) in enumerate(zip(bars, vals)):
            if v >= 5:
                ax.text(
                    left[i] + v / 2,
                    bar.get_y() + bar.get_height() / 2,
                    f"{v}%",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white",
                    fontweight="bold",
                )

        left = [l + v for l, v in zip(left, vals)]
    ax.set_xlim(0, 100)
    ax.set_yticks(y)
    ax.set_yticklabels(rows)
    ax.set_xlabel("Porcentaje")
    ax.set_title("Gráfico 3. Distribución de rangos de edad según el tipo de violencia")
    ax.grid(axis="x", alpha=0.2)
    ax.invert_yaxis()
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


def fig_context_bar(ct: Dict[str, int]):
    fig, ax = plt.subplots(figsize=(10, 3.8))
    labels = [CONTEXT_LABELS[k] for k in CONTEXT_KEYS]
    values = [ct.get(k, 0) for k in CONTEXT_KEYS]
    ax.bar(labels, values)
    ax.set_title("Contexto: totales en el rango")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", labelrotation=20)
    fig.tight_layout()
    return fig

def fig_context_lines(cm: Dict[str, List[int]]):
    fig, ax = plt.subplots(figsize=(10, 3.8))
    x = list(range(12))
    for k in CONTEXT_KEYS:
        ax.plot(x, cm.get(k, [0]*12), marker="o", label=CONTEXT_LABELS[k])
    ax.set_title("Contexto: tendencia mensual")
    ax.set_xticks(x)
    ax.set_xticklabels(MONTHS_ES, rotation=45, ha="right", fontsize=8)
    ax.grid(alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=8)
    fig.tight_layout()
    return fig

# =========================
# CONTEXTO INTERACTIVO (Plotly - hover)
# =========================

def fig_context_bar_hover(ct: Dict[str, int]):
    """Barras interactivas (hover): totales de contexto en el rango."""
    labels = [CONTEXT_LABELS[k] for k in CONTEXT_KEYS]
    values = [int(ct.get(k, 0)) for k in CONTEXT_KEYS]
    colors = [CONTEXT_COLORS.get(k, "#64748b") for k in CONTEXT_KEYS]

    fig = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=values,
                marker=dict(color=colors, line=dict(color="white", width=0.6)),
                hovertemplate="%{x}<br>Valor: %{y:,}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title="Contexto: totales en el rango",
        height=360,
        margin=dict(l=0, r=0, t=55, b=0),
    )
    fig.update_yaxes(title_text="Número de casos", gridcolor="rgba(148,163,184,0.25)")
    fig.update_xaxes(tickangle=-20)
    return fig


def fig_context_lines_hover(cm: Dict[str, List[int]]):
    """Líneas interactivas (hover): tendencia mensual de contexto."""
    x = MONTHS_ES
    fig = go.Figure()

    for kx in CONTEXT_KEYS:
        y = cm.get(kx, [0] * 12) or [0] * 12
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines+markers",
                name=CONTEXT_LABELS.get(kx, kx),
                line=dict(color=CONTEXT_COLORS.get(kx, "#64748b"), width=3),
                marker=dict(color=CONTEXT_COLORS.get(kx, "#64748b"), size=7),
                hovertemplate="%{x}<br>%{fullData.name}: %{y:,}<extra></extra>",
            )
        )

    fig.update_layout(
        title="Contexto: tendencia mensual",
        height=420,
        margin=dict(l=0, r=0, t=55, b=0),
        hovermode="x unified",
        legend=dict(orientation="v", x=1.02, y=0.5, xanchor="left", yanchor="middle"),
    )
    fig.update_xaxes(tickangle=-45)
    fig.update_yaxes(title_text="Número de casos", gridcolor="rgba(148,163,184,0.25)")
    return fig


def fig_risk_levels_bar(totals_by_level: Dict[str, int]):
    fig, ax = plt.subplots(figsize=(10, 3.6))

    labels = [RISK_LEVEL_LABELS[l] for l in RISK_LEVELS]
    values = [int(totals_by_level.get(l, 0)) for l in RISK_LEVELS]

    # Paleta consistente (morado/verde)
    risk_colors = {
        "HAY_RIESGO": SOURCE_COLORS["SIVIGILA"],  # morado oscuro
        "NO_RIESGO": SOURCE_COLORS["EAE"],        # verde
    }
    colors = [risk_colors.get(lvl, "#64748b") for lvl in RISK_LEVELS]

    ax.bar(labels, values, color=colors, edgecolor="white", linewidth=0.6)
    ax.set_title("Riesgos: distribución por nivel")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def fig_risk_types_bar(totals_by_type: Dict[str, int], selected_types: List[str]):
    fig, ax = plt.subplots(figsize=(10, 3.8))
    labels = [RISK_TYPE_LABELS.get(t, t) for t in selected_types]
    values = [totals_by_type.get(t, 0) for t in selected_types]
    ax.bar(labels, values)
    ax.set_title("Riesgos: distribución por tipología")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", labelrotation=20)
    fig.tight_layout()
    return fig


def fig_risk_trend(monthly_by_level: Dict[str, List[int]]):
    fig, ax = plt.subplots(figsize=(10, 3.8))
    x = list(range(12))

    # Paleta consistente (morado/verde)
    risk_colors = {
        "HAY_RIESGO": SOURCE_COLORS["SIVIGILA"],  # morado oscuro
        "NO_RIESGO": SOURCE_COLORS["EAE"],        # verde
    }

    for lvl in RISK_LEVELS:
        y = monthly_by_level.get(lvl, [0] * 12)
        ax.plot(
            x,
            y,
            marker="o",
            label=RISK_LEVEL_LABELS[lvl],
            color=risk_colors.get(lvl, "#64748b"),
            linewidth=2.5,
            markersize=5,
        )

    ax.set_title("Riesgos: tendencia mensual por nivel")
    ax.set_xticks(x)
    ax.set_xticklabels(MONTHS_ES, rotation=45, ha="right", fontsize=8)
    ax.grid(alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=9)
    fig.tight_layout()
    return fig


# =========================
# RIESGOS INTERACTIVOS (Plotly - hover)
# =========================

def fig_risk_levels_bar_hover(totals_by_level: Dict[str, int]):
    """Barras interactivas (hover): distribución por nivel (HAY RIESGO / NO HAY RIESGO)."""
    labels = [RISK_LEVEL_LABELS[l] for l in RISK_LEVELS]
    values = [int(totals_by_level.get(l, 0)) for l in RISK_LEVELS]

    risk_colors = {
        "HAY_RIESGO": SOURCE_COLORS["SIVIGILA"],  # morado oscuro
        "NO_RIESGO": SOURCE_COLORS["EAE"],        # verde
    }
    colors = [risk_colors.get(lvl, "#64748b") for lvl in RISK_LEVELS]

    fig = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=values,
                marker=dict(color=colors, line=dict(color="white", width=0.6)),
                hovertemplate="%{x}<br>Valor: %{y:,}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title="Riesgos: distribución por nivel",
        height=360,
        margin=dict(l=0, r=0, t=55, b=0),
    )
    fig.update_yaxes(title_text="Número de casos", gridcolor="rgba(148,163,184,0.25)")
    return fig


def fig_risk_trend_hover(monthly_by_level: Dict[str, List[int]]):
    """Líneas interactivas (hover): tendencia mensual por nivel."""
    x = MONTHS_ES

    risk_colors = {
        "HAY_RIESGO": SOURCE_COLORS["SIVIGILA"],  # morado oscuro
        "NO_RIESGO": SOURCE_COLORS["EAE"],        # verde
    }

    fig = go.Figure()
    for lvl in RISK_LEVELS:
        y = monthly_by_level.get(lvl, [0] * 12) or [0] * 12
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines+markers",
                name=RISK_LEVEL_LABELS.get(lvl, lvl),
                line=dict(color=risk_colors.get(lvl, "#64748b"), width=3),
                marker=dict(color=risk_colors.get(lvl, "#64748b"), size=7),
                hovertemplate="%{x}<br>%{fullData.name}: %{y:,}<extra></extra>",
            )
        )

    fig.update_layout(
        title="Riesgos: tendencia mensual por nivel",
        height=420,
        margin=dict(l=0, r=0, t=55, b=0),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(tickangle=-45)
    fig.update_yaxes(title_text="Número de casos", gridcolor="rgba(148,163,184,0.25)")
    return fig


def fig_to_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


# =========================
# MAPA (Folium choropleth estilo)
# =========================

@st.cache_data(show_spinner=False)
def load_geojson(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_map(geojson: dict, area_field: str, values_by_area: Dict[str, int], metric_label: str):
    m = folium.Map(location=CALI_CENTER, zoom_start=11, tiles="CartoDB positron")

    # Inject values
    for feat in geojson.get("features", []):
        props = feat.get("properties", {}) or {}
        key = str(props.get(area_field, ""))
        props["__valor__"] = int(values_by_area.get(key, 0))
        feat["properties"] = props

    values = [feat.get("properties", {}).get("__valor__", 0) for feat in geojson.get("features", [])]
    vmax = max(values) if values else 1

    try:
        from branca.colormap import linear
        colormap = linear.YlOrRd_09.scale(0, max(1, vmax))
        colormap.caption = metric_label
        colormap.add_to(m)
    except Exception:
        colormap = None

    def style_fn(feat):
        v = feat.get("properties", {}).get("__valor__", 0)
        fill = colormap(v) if colormap is not None else "#f59e0b"
        return {"fillColor": fill, "color": "#334155", "weight": 0.8, "fillOpacity": 0.75}

    folium.GeoJson(
        geojson,
        name="areas",
        style_function=style_fn,
        tooltip=folium.GeoJsonTooltip(
            fields=[area_field, "__valor__"],
            aliases=["Área", metric_label],
            labels=True,
            sticky=True,
        ),
    ).add_to(m)

    return m


# =========================
# PDF EXPORT (ReportLab)
# =========================

def build_pdf_bytes(title: str, subtitle: str, kpi_lines: List[str], figures_png: List[bytes]) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    w, h = A4

    # Header
    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, h - 50, title)
    c.setFont("Helvetica", 10)
    c.drawString(40, h - 68, subtitle)

    y = h - 95
    c.setFont("Helvetica", 10)
    for line in kpi_lines:
        c.drawString(40, y, line)
        y -= 14
        if y < 80:
            c.showPage()
            y = h - 50

    for img_bytes in figures_png:
        img = ImageReader(io.BytesIO(img_bytes))
        # Fit image to page width with margins
        max_w = w - 80
        max_h = h - 140
        iw, ih = img.getSize()
        scale = min(max_w / iw, max_h / ih)
        dw, dh = iw * scale, ih * scale

        if y - dh < 50:
            c.showPage()
            y = h - 50

        c.drawImage(img, 40, y - dh, width=dw, height=dh)
        y = y - dh - 18

    c.showPage()
    c.save()
    return buf.getvalue()


# =========================
# STREAMLIT UI
# =========================

st.set_page_config(page_title="Reporte CIMU (Streamlit)", layout="wide")

# Nota: evitamos cache_resource aquí para no quedarnos con una instancia "vieja"
# cuando agregamos métodos nuevos (p. ej. aggregate_risks). Para mantener performance,
# guardamos en session_state.

def get_model() -> DataModel:
    if "_cimu_model" not in st.session_state:
        st.session_state["_cimu_model"] = DataModel()
    return st.session_state["_cimu_model"]

model = get_model()

st.sidebar.title("REPORTE CIMU")
page = st.sidebar.radio("Secciones", ["📊 Dashboard", "📈 Análisis", "🗺️ Mapa", "⚠️ Riesgos", "🧾 Contexto"], index=0)
st.sidebar.caption(f"Build: {BUILD_TAG}")

# Global date filter
st.sidebar.subheader("Filtro de fechas")
dmin = date(2025, 1, 1)
dmax = date(2025, 12, 16)

d_from = st.sidebar.date_input("Desde", value=dmin, min_value=dmin, max_value=dmax)
d_to = st.sidebar.date_input("Hasta", value=dmax, min_value=dmin, max_value=dmax)
if d_from > d_to:
    d_from, d_to = d_to, d_from

agg = model.aggregate(d_from, d_to)

st.caption(f"Rango: {d_from.isoformat()} → {d_to.isoformat()} • Datos: mock dinámico")

# Common helpers
k = agg.get("kpis", {})
meta = agg.get("meta", {})
ct = agg.get("context_totals", {})
cm = agg.get("context_monthly", {})

# =========================
# DASHBOARD
# =========================

if page == "📊 Dashboard":
    st.header("Dashboard")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("TOTAL ATENCIONES", fmt_int(int(k.get("TOTAL_ATENCIONES", 0))))
    c2.metric("PROMEDIO DIARIO", f"{float(k.get('PROMEDIO_DIARIO', 0)):.1f}")
    c3.metric("DÍAS EN RANGO", str(meta.get("days", 0)))
    pm = meta.get("peak_month") or "—"
    pv = int(meta.get("peak_value", 0))
    c4.metric("MES PICO (TOTAL)", f"{pm} ({fmt_int(pv)})")

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("SIVIGILA", fmt_int(int(k.get("SIVIGILA", 0))))
    d2.metric("COMISARÍAS", fmt_int(int(k.get("COMISARIAS", 0))))
    d3.metric("CASA MATRIA", fmt_int(int(k.get("CASA_MATRIA", 0))))
    d4.metric("EAE", fmt_int(int(k.get("EAE", 0))))

    # =========================================
    # NUEVO: Total mujeres atendidas (rango) y discriminado por organismo
    # =========================================
    st.subheader("Total mujeres atendidas en el rango (discriminado por organismo)")

    total_mujeres = int(k.get("TOTAL_ATENCIONES", 0))
    org_labels = ["SIVIGILA", "COMISARÍAS", "CASA MATRIA", "EAE"]
    org_values = [
        int(k.get("SIVIGILA", 0)),
        int(k.get("COMISARIAS", 0)),
        int(k.get("CASA_MATRIA", 0)),
        int(k.get("EAE", 0)),
    ]

    fig_total_org = go.Figure()
    fig_total_org.add_trace(
        go.Bar(
            x=org_labels,
            y=org_values,
            marker=dict(
                color=[
                    SOURCE_COLORS["SIVIGILA"],
                    SOURCE_COLORS["COMISARIAS"],
                    SOURCE_COLORS["CASA_MATRIA"],
                    SOURCE_COLORS["EAE"],
                ],
                line=dict(color="white", width=0.6),
            ),
            hovertemplate="Organismo: %{x}<br>Mujeres atendidas: %{y:,}<extra></extra>",
        )
    )

    fig_total_org.update_layout(
        title=f"Total mujeres atendidas en el rango: {fmt_int(total_mujeres)}",
        height=380,
        margin=dict(l=0, r=0, t=60, b=0),
    )
    fig_total_org.update_yaxes(
        title_text="Número de mujeres atendidas",
        gridcolor="rgba(148,163,184,0.25)",
    )

    st.plotly_chart(fig_total_org, use_container_width=True)

    st.plotly_chart(fig_grouped_bars_hover(agg["monthly"]), use_container_width=True)

    a, b = st.columns([2, 1])
    with a:
        st.plotly_chart(fig_total_line_hover(agg["total_monthly"]), use_container_width=True)
    with b:
        st.plotly_chart(fig_donut_hover(agg["distribution"]), use_container_width=True)

    st.divider()
    st.subheader("Comparar dos rangos")
    st.caption("Ejemplo: 01–31 ENE 2025 vs 01–31 MAR 2025. Estos rangos son independientes del filtro global.")

    # Rango A
    rA1, rA2, rB1, rB2 = st.columns([1, 1, 1, 1])
    with rA1:
        a_from = st.date_input(
            "Periodo A - Desde",
            value=date(2025, 1, 1),
            min_value=dmin,
            max_value=dmax,
            key="cmp_range_a_from",
        )
    with rA2:
        a_to = st.date_input(
            "Periodo A - Hasta",
            value=date(2025, 1, 31),
            min_value=dmin,
            max_value=dmax,
            key="cmp_range_a_to",
        )

    # Rango B
    with rB1:
        b_from = st.date_input(
            "Periodo B - Desde",
            value=date(2025, 3, 1),
            min_value=dmin,
            max_value=dmax,
            key="cmp_range_b_from",
        )
    with rB2:
        b_to = st.date_input(
            "Periodo B - Hasta",
            value=date(2025, 3, 31),
            min_value=dmin,
            max_value=dmax,
            key="cmp_range_b_to",
        )

    if a_from > a_to:
        a_from, a_to = a_to, a_from
    if b_from > b_to:
        b_from, b_to = b_to, b_from

    agg_a = model.aggregate(a_from, a_to)
    agg_b = model.aggregate(b_from, b_to)

    label_a = f"A: {a_from.isoformat()} → {a_to.isoformat()}"
    label_b = f"B: {b_from.isoformat()} → {b_to.isoformat()}"

    st.plotly_chart(fig_compare_sources_two_ranges_hover(agg_a, agg_b, label_a, label_b), use_container_width=True)

    # Tabla de comparación + delta
    sources = ["SIVIGILA", "COMISARIAS", "CASA_MATRIA", "EAE"]
    labels = {"SIVIGILA": "SIVIGILA", "COMISARIAS": "COMISARÍAS", "CASA_MATRIA": "CASA MATRIA", "EAE": "EAE"}

    k_a = agg_a.get("kpis", {})
    k_b = agg_b.get("kpis", {})

    rows = []
    for s in sources:
        a_val = int(k_a.get(s, 0))
        b_val = int(k_b.get(s, 0))
        delta = b_val - a_val
        pct = (delta / a_val * 100.0) if a_val > 0 else None
        rows.append({
            "Organismo": labels.get(s, s),
            "Periodo A": a_val,
            "Periodo B": b_val,
            "Δ (B-A)": delta,
            "%Δ": (f"{pct:.1f}%" if pct is not None else "—"),
        })

    st.dataframe(rows, use_container_width=True, hide_index=True)

    # KPIs rápidos del total
    tA = int(k_a.get("TOTAL_ATENCIONES", 0))
    tB = int(k_b.get("TOTAL_ATENCIONES", 0))
    dT = tB - tA
    pT = (dT / tA * 100.0) if tA > 0 else None

    t1, t2, t3 = st.columns(3)
    t1.metric("TOTAL A", fmt_int(tA))
    t2.metric("TOTAL B", fmt_int(tB))
    t3.metric("Δ TOTAL (B-A)", fmt_int(dT), (f"{pT:.1f}%" if pT is not None else None))

# =========================
# ANALISIS
# =========================
elif page == "📈 Análisis":
    st.header("Análisis (Gráficos)")

    st.plotly_chart(fig_g2_hover(agg["g2_pct"]), use_container_width=True)
    st.plotly_chart(fig_g3_hover(agg["g3_pct"]), use_container_width=True)

    if st.button("Exportar PDF (Análisis)"):
        figs = [
            fig_to_png_bytes(fig_g2(agg["g2_pct"])),
            fig_to_png_bytes(fig_g3(agg["g3_pct"])),
        ]
        kpi_lines = [
            f"Rango: {d_from.isoformat()} → {d_to.isoformat()}",
            f"TOTAL ATENCIONES: {fmt_int(int(k.get('TOTAL_ATENCIONES', 0)))}",
        ]
        pdf = build_pdf_bytes(
            "Reporte CIMU - Análisis",
            f"Rango: {d_from.isoformat()} → {d_to.isoformat()}",
            kpi_lines,
            figs,
        )
        st.download_button(
            "Descargar PDF",
            data=pdf,
            file_name=f"reporte_cimu_analisis_{d_from.isoformat()}_a_{d_to.isoformat()}.pdf",
            mime="application/pdf",
        )

# =========================
# MAPA
# =========================
elif page == "🗺️ Mapa":
    st.header("Mapa coroplético")

    left, right = st.columns([1, 2])

    with left:
        nivel = st.selectbox("Nivel territorial", ["Comunas y corregimientos", "Barrios y veredas"], index=0)
        metric_options = [("TOTAL_ATENCIONES", "Mujeres atendidas")] + [(k, CONTEXT_LABELS[k]) for k in CONTEXT_KEYS]
        metric_key = st.selectbox("Métrica", metric_options, format_func=lambda x: x[1])[0]

        if nivel == "Barrios y veredas":
            geo_path = GEOJSON_BARRIOS_PATH
            expected_field = GEOJSON_BARRIOS_FIELD
        else:
            geo_path = GEOJSON_COMUNAS_PATH
            expected_field = GEOJSON_COMUNAS_FIELD

        st.caption(f"GeoJSON esperado: {geo_path}")
        st.caption(f"Campo esperado: {expected_field}")

        # Si no existe el archivo, permitimos subirlo desde la UI (muy común en Mac/VSCode)
        geojson = None
        if not os.path.exists(geo_path):
            st.warning("No encontré el GeoJSON en esa ruta. Puedes subirlo aquí (una vez) para probar.")
            up = st.file_uploader("Sube el GeoJSON de esta capa", type=["geojson", "json"], key=f"uploader_{nivel}")
            if up is None:
                st.stop()
            geojson = json.load(up)
        else:
            geojson = load_geojson(geo_path)

        # Diagnóstico rápido
        nfeat = len(geojson.get("features", []) or [])
        st.caption(f"Features cargadas: {nfeat}")

        # Auto-detect del campo de nombre: si el esperado no existe, buscamos uno similar
        area_field = expected_field
        if nfeat > 0:
            props0 = (geojson.get("features", [])[0].get("properties", {}) or {})
            if area_field not in props0:
                candidates = list(props0.keys())
                # preferimos campos que contengan palabras clave
                for kw in ("barrio", "vereda", "comuna", "correg", "nombre", "name"):
                    hit = next((k for k in candidates if kw in k.lower()), None)
                    if hit:
                        area_field = hit
                        break

        st.caption(f"Campo usado: {area_field}")

        # Construimos llaves de área
        area_keys = []
        for feat in geojson.get("features", []):
            props = feat.get("properties", {}) or {}
            kx = props.get(area_field)
            if kx is not None:
                area_keys.append(str(kx))

        if not area_keys:
            st.error(
                "No pude leer nombres/llaves desde el GeoJSON. "
                f"Revisa que exista properties['{expected_field}'] (o cambia el campo en el código)."
            )
            # mostramos las keys del primer feature para que se vea qué viene realmente
            if nfeat > 0:
                props0 = (geojson.get("features", [])[0].get("properties", {}) or {})
                st.code(list(props0.keys()))
            st.stop()

        values = model.aggregate_by_area_keys(d_from, d_to, metric_key, area_keys)
        metric_label = dict(metric_options).get(metric_key, metric_key)

        # Download HTML map
        st.download_button(
            "Descargar mapa (HTML)",
            data=build_map(geojson, area_field, values, metric_label).get_root().render().encode("utf-8"),
            file_name=f"mapa_{'barrios' if nivel=='Barrios y veredas' else 'comunas'}_{metric_key}_{d_from}_a_{d_to}.html",
            mime="text/html",
            key=f"dl_mapa_{nivel}_{metric_key}_{d_from.isoformat()}_{d_to.isoformat()}"
        )

    with right:
        m = build_map(geojson, area_field, values, dict(metric_options).get(metric_key, metric_key))
        st_folium(m, width=None, height=650, key=f"mapa_{nivel}_{metric_key}_{d_from.isoformat()}_{d_to.isoformat()}")

#
# =========================
# RIESGOS
# =========================
elif page == "⚠️ Riesgos":
    st.header("Riesgos")

    st.caption("Clasificación de riesgo según aportes de talleres (Sí hay riesgo / No hay riesgo).")

    # Filtros específicos de la sección

    # Fijar todas las tipologías sin filtro
    selected_types = list(RISK_TYPES)

    # Niveles (solo dos opciones: HAY RIESGO / NO HAY RIESGO)
    # Limpieza de estado: si antes existían BAJO/MEDIO/ALTO/EXTREMO, Streamlit puede recordarlos.
    if "risk_levels" in st.session_state:
        st.session_state["risk_levels"] = normalize_risk_levels(st.session_state.get("risk_levels") or [])

    selected_levels = st.multiselect(
        "Niveles de riesgo",
        options=RISK_LEVELS,
        default=list(RISK_LEVELS),
        format_func=lambda l: RISK_LEVEL_LABELS.get(l, l),
        key="risk_levels",
    )

    # Normaliza por si llegan valores inesperados
    selected_levels = normalize_risk_levels(selected_levels)

    # Si el usuario deja vacío, interpretamos como "todos"
    effective_levels = selected_levels if selected_levels else list(RISK_LEVELS)

    st.divider()
    st.subheader("Riesgos por territorio")
    nivel_territorial = st.selectbox(
        "Nivel territorial",
        ["Comunas y corregimientos", "Barrios y veredas"],
        index=0,
        key="riesgos_nivel_territorial",
    )

    risks = model.aggregate_risks(d_from, d_to, selected_types=list(RISK_TYPES))

    # --- Aplicar filtros de niveles a TODO lo que se muestra (KPIs, gráficas, tabla, territorio) ---
    # `aggregate_risks` genera todo en ambos niveles; aquí nos quedamos con los seleccionados.

    def _filter_risks_by_levels(risks_dict: dict, levels: List[str]) -> dict:
        matrix0 = risks_dict.get("matrix", {}) or {}
        monthly0 = risks_dict.get("monthly_by_level", {}) or {}

        # Totales por nivel (solo niveles seleccionados)
        totals_by_level_sel = {lvl: 0 for lvl in levels}
        for t, row in matrix0.items():
            for lvl in levels:
                totals_by_level_sel[lvl] += int((row or {}).get(lvl, 0))

        # Totales por tipo (sumando solo niveles seleccionados)
        totals_by_type_sel: Dict[str, int] = {}
        for t, row in matrix0.items():
            totals_by_type_sel[t] = sum(int((row or {}).get(lvl, 0)) for lvl in levels)

        # Matriz filtrada
        matrix_sel: Dict[str, Dict[str, int]] = {
            t: {lvl: int((row or {}).get(lvl, 0)) for lvl in levels}
            for t, row in matrix0.items()
        }

        # Tendencia mensual filtrada (para niveles no seleccionados = 0)
        monthly_sel = {lvl: (monthly0.get(lvl, [0] * 12) or [0] * 12) for lvl in levels}

        risk_pool_sel = sum(totals_by_level_sel.values())

        return {
            "risk_pool": int(risk_pool_sel),
            "totals_by_level": totals_by_level_sel,
            "totals_by_type": totals_by_type_sel,
            "matrix": matrix_sel,
            "monthly_by_level": monthly_sel,
        }

    risks_filtered = _filter_risks_by_levels(risks, effective_levels)

    # Para mantener las gráficas consistentes (orden fijo), rellenamos con ceros los niveles NO seleccionados.
    totals_by_level = {lvl: int(risks_filtered["totals_by_level"].get(lvl, 0)) if lvl in effective_levels else 0 for lvl in RISK_LEVELS}
    monthly_by_level = {lvl: (risks_filtered["monthly_by_level"].get(lvl, [0] * 12) if lvl in effective_levels else [0] * 12) for lvl in RISK_LEVELS}



    # KPIs
    k1, k2, k3 = st.columns(3)
    total_pool = int(risks_filtered.get("risk_pool", 0))
    hay_riesgo_total = int(totals_by_level.get("HAY_RIESGO", 0))
    no_riesgo_total = int(totals_by_level.get("NO_RIESGO", 0))

    k1.metric("TOTAL EVENTOS EVALUADOS", fmt_int(total_pool))
    k2.metric("HAY RIESGO", fmt_int(hay_riesgo_total))
    k3.metric("NO HAY RIESGO", fmt_int(no_riesgo_total))

    # Gráficos (hover)
    st.plotly_chart(fig_risk_levels_bar_hover(totals_by_level), use_container_width=True)
    st.caption("Los resultados cambian según los niveles seleccionados arriba.")
    st.plotly_chart(fig_risk_trend_hover(monthly_by_level), use_container_width=True)

    # ===== Territorio (mapa + ranking) =====
    if nivel_territorial == "Barrios y veredas":
        geo_path = GEOJSON_BARRIOS_PATH
        expected_field = GEOJSON_BARRIOS_FIELD
    else:
        geo_path = GEOJSON_COMUNAS_PATH
        expected_field = GEOJSON_COMUNAS_FIELD

    st.caption(f"GeoJSON esperado: {geo_path}")
    st.caption(f"Campo esperado: {expected_field}")

    geojson = None
    if not os.path.exists(geo_path):
        st.warning("No encontré el GeoJSON en esa ruta. Puedes subirlo aquí para ver el mapa.")
        up = st.file_uploader("Sube el GeoJSON de esta capa", type=["geojson", "json"], key=f"uploader_riesgos_{nivel_territorial}")
        if up is None:
            st.stop()
        geojson = json.load(up)
    else:
        geojson = load_geojson(geo_path)

    nfeat = len(geojson.get("features", []) or [])
    st.caption(f"Features cargadas: {nfeat}")

    area_field = expected_field
    if nfeat > 0:
        props0 = (geojson.get("features", [])[0].get("properties", {}) or {})
        if area_field not in props0:
            candidates = list(props0.keys())
            for kw in ("barrio", "vereda", "comuna", "correg", "nombre", "name"):
                hit = next((k for k in candidates if kw in k.lower()), None)
                if hit:
                    area_field = hit
                    break

    st.caption(f"Campo usado: {area_field}")

    # Llaves de áreas
    area_keys = []
    for feat in geojson.get("features", []):
        props = feat.get("properties", {}) or {}
        kx = props.get(area_field)
        if kx is not None:
            area_keys.append(str(kx))

    if not area_keys:
        st.error(
            "No pude leer nombres/llaves desde el GeoJSON. "
            f"Revisa que exista properties['{expected_field}'] (o cambia el campo en el código)."
        )
        if nfeat > 0:
            props0 = (geojson.get("features", [])[0].get("properties", {}) or {})
            st.code(list(props0.keys()))
        st.stop()

    # Valores de riesgo por área según filtros (tipos + niveles)
    values_by_area = model.aggregate_risks_by_area_keys(
        d_from,
        d_to,
        selected_types=risks["selected_types"],
        selected_levels=effective_levels,
        area_keys=area_keys,
    )

    lvl_txt = "+".join(RISK_LEVEL_LABELS.get(l, l) for l in (selected_levels or []))
    metric_label = "Riesgos" + (f" ({lvl_txt})" if lvl_txt else "")

    # Ranking Top 10
    st.subheader("Top 10 territorios (por valor)")
    items = sorted(values_by_area.items(), key=lambda kv: kv[1], reverse=True)
    top10 = items[:10]
    st.dataframe(
        [{"Territorio": k, "Valor": int(v)} for k, v in top10],
        use_container_width=True,
        hide_index=True,
    )

    # Mapa
    st.subheader("Mapa coroplético")
    m = build_map(geojson, area_field, values_by_area, metric_label)
    st_folium(m, width=None, height=650, key=f"riesgos_mapa_{nivel_territorial}_{'-'.join(selected_levels) if selected_levels else 'todos'}_{d_from.isoformat()}_{d_to.isoformat()}")

    # Descarga HTML del mapa
    st.download_button(
        "Descargar mapa (HTML)",
        data=m.get_root().render().encode("utf-8"),
        file_name=f"mapa_riesgos_{'barrios' if nivel_territorial=='Barrios y veredas' else 'comunas'}_{'_'.join(selected_levels) or 'todos'}_{d_from}_a_{d_to}.html",
        mime="text/html",
        key=f"dl_riesgos_mapa_{nivel_territorial}_{'-'.join(selected_levels) if selected_levels else 'todos'}_{d_from.isoformat()}_{d_to.isoformat()}"
    )

    # Export PDF
    if st.button("Exportar PDF (Riesgos)"):
        figs = [
            fig_to_png_bytes(fig_risk_levels_bar(totals_by_level)),
            fig_to_png_bytes(fig_risk_trend(monthly_by_level)),
        ]
        kpi_lines = [
            f"Rango: {d_from.isoformat()} → {d_to.isoformat()}",
            f"TOTAL EVENTOS EVALUADOS: {fmt_int(int(risks_filtered.get('risk_pool', 0)))}",
            f"HAY RIESGO: {fmt_int(int(totals_by_level.get('HAY_RIESGO', 0)))}",
            f"NO HAY RIESGO: {fmt_int(int(totals_by_level.get('NO_RIESGO', 0)))}",
        ]
        pdf = build_pdf_bytes(
            "Reporte CIMU - Riesgos",
            f"Rango: {d_from.isoformat()} → {d_to.isoformat()}",
            kpi_lines,
            figs,
        )
        st.download_button(
            "Descargar PDF",
            data=pdf,
            file_name=f"reporte_cimu_riesgos_{d_from.isoformat()}_a_{d_to.isoformat()}.pdf",
            mime="application/pdf",
        )

# =========================
# CONTEXTO
# =========================
else:
    st.header("Contexto y fuentes")

    # =========================
    # Comparativo con año anterior (mock determinístico)
    # =========================
    import random

    seed_prev = (
        int(d_from.strftime("%Y%m%d")) * 19
        + int(d_to.strftime("%Y%m%d")) * 23
        + 2024
    )
    rng_prev = random.Random(seed_prev)

    prev_ct = {}
    for kx in CONTEXT_KEYS:
        actual = int(ct.get(kx, 0))
        # Variación entre -15% y +15% para simular año anterior
        factor = rng_prev.uniform(0.85, 1.15)
        prev_ct[kx] = max(0, int(round(actual * factor)))

    st.subheader("Comparativo con año anterior")

    c1, c2, c3, c4 = st.columns(4)

    def metric_with_delta(label, key):
        actual = int(ct.get(key, 0))
        prev = int(prev_ct.get(key, 0))
        delta = actual - prev
        pct = (delta / prev * 100.0) if prev > 0 else None
        delta_str = f"{fmt_int(delta)} ({pct:.1f}%)" if pct is not None else fmt_int(delta)
        return label, fmt_int(actual), delta_str

    l1, v1, d1 = metric_with_delta("VIF (rango)", "VIF")
    l2, v2, d2 = metric_with_delta("Delitos sexuales (rango)", "DELITOS_SEXUALES")
    l3, v3, d3 = metric_with_delta("Homicidios mujeres (rango)", "HOMICIDIOS_MUJER")
    l4, v4, d4 = metric_with_delta("Feminicidios (rango)", "FEMINICIDIOS")

    c1.metric(l1, v1, d1)
    c2.metric(l2, v2, d2)
    c3.metric(l3, v3, d3)
    c4.metric(l4, v4, d4)

    # Tabla resumen comparativa
    st.divider()
    st.subheader("Resumen comparativo (actual vs año anterior)")

    rows_cmp = []
    for kx in CONTEXT_KEYS:
        actual = int(ct.get(kx, 0))
        prev = int(prev_ct.get(kx, 0))
        delta = actual - prev
        pct = (delta / prev * 100.0) if prev > 0 else None
        rows_cmp.append({
            "Indicador": CONTEXT_LABELS.get(kx, kx),
            "Año actual": actual,
            "Año anterior": prev,
            "Diferencia": delta,
            "% Variación": (f"{pct:.1f}%" if pct is not None else "—"),
        })

    st.dataframe(rows_cmp, use_container_width=True, hide_index=True)

    st.plotly_chart(fig_context_bar_hover(ct), use_container_width=True)
    st.plotly_chart(fig_context_lines_hover(cm), use_container_width=True)

    st.subheader("Fuentes y notas")
    st.write("• Policía Nacional – SIEDCO (VIF, delitos sexuales).")
    st.write("• Comité Interinstitucional de Muerte por Causa Externa – CIMCE (homicidios, feminicidios).")
    st.write("• Este prototipo usa datos mock para demostración; la estructura está lista para conectar CSV/BD/API.")

    if st.button("Exportar PDF (Contexto)"):
        figs = [
            fig_to_png_bytes(fig_context_bar(ct)),
            fig_to_png_bytes(fig_context_lines(cm)),
        ]
        kpi_lines = [
            f"VIF (rango): {fmt_int(int(ct.get('VIF', 0)))}",
            f"Delitos sexuales (rango): {fmt_int(int(ct.get('DELITOS_SEXUALES', 0)))}",
            f"Homicidios mujeres (rango): {ct.get('HOMICIDIOS_MUJER', 0)}",
            f"Feminicidios (rango): {ct.get('FEMINICIDIOS', 0)}",
        ]
        pdf = build_pdf_bytes(
            "Reporte CIMU - Contexto",
            f"Rango: {d_from.isoformat()} → {d_to.isoformat()}",
            kpi_lines,
            figs,
        )
        st.download_button(
            "Descargar PDF",
            data=pdf,
            file_name=f"reporte_cimu_contexto_{d_from.isoformat()}_a_{d_to.isoformat()}.pdf",
            mime="application/pdf",
        )