"""
Couche PRÉSENTATION — dashboard décisionnel (Streamlit).

Pensé pour un décideur NON technique : chaque onglet commence par un verdict en
langage simple ; les détails techniques (∫I dt, p-values, IC95…) sont repliés
dans des expanders. Lancement :
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Isolation des imports : garantit que CE projet (et non un projet voisin
# homonyme) est résolu en premier, quel que soit le CWD ou le PYTHONPATH ambiant.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import streamlit as st

from api.deps import AppState
from domain.epidemiology import basic_reproduction_number
from presentation.plots import (
    benchmark_figure, controls_figure, fit_figure, reff_figure, seir_figure,
)

st.set_page_config(page_title="Décision · Santé Publique v2", layout="wide")

STRAT_FR = {"baseline": "Sans intervention", "lqr": "LQR (ajustement continu)",
            "pmp": "PMP (plan pré-optimisé)", "hjb": "HJB-PINN (IA)"}


def fmt_people(n: float) -> str:
    """1234567 -> « 1,23 million » ; 12500 -> « 12 500 » (lisible pour un décideur)."""
    n = float(n)
    if n >= 1e6:
        return f"{n / 1e6:.2f} million".replace(".", ",") + ("s" if n >= 2e6 else "")
    return f"{n:,.0f}".replace(",", " ")


def pct_pop(n: float, N: float) -> str:
    return f"{100.0 * n / N:.1f} %".replace(".", ",")


@st.cache_resource
def get_state() -> AppState:
    s = AppState()
    s.ensure_calibrated()
    return s


@st.cache_data(show_spinner="Calcul du benchmark…")
def run_benchmark(beta, sigma, gamma, N, horizon, strategies):
    """Benchmark mis en cache : ne dépend QUE des paramètres calibrés, de l'horizon
    et des stratégies — pas des curseurs u1/u2. Évite de tout recalculer à chaque geste."""
    from domain.seir import SEIRParams
    from services.benchmark_service import BenchmarkService
    params = SEIRParams(beta=beta, sigma=sigma, gamma=gamma, N=N)
    return BenchmarkService(params).run(list(strategies), horizon)


state = get_state()
p = state.params
R0 = basic_reproduction_number(p)

st.title("🦠 Plateforme d'Aide à la Décision — Santé Publique")
st.caption("Compare des stratégies d'intervention (vaccination, confinement) sur un modèle "
           "épidémique SEIR calibré, et recommande quand et comment agir.")

with st.sidebar:
    st.header("🎛️ Tester une politique")
    u1 = st.slider("Vaccination (u₁)", 0.0, 1.0, 0.1, 0.01,
                   help="Fraction de la population susceptible vaccinée chaque jour.")
    u2 = st.slider("Confinement (u₂)", 0.0, 1.0, 0.3, 0.01,
                   help="Réduction des contacts sociaux (0 = aucune, 1 = isolement total).")
    horizon = st.slider("Horizon (jours)", 60, 365, 180, 10)
    st.divider()
    st.caption("Stratégies à comparer")
    from pathlib import Path as _P
    from core.config import settings as _cfg
    _hjb_ok = _P(_cfg.hjb_weights_path).exists()
    inc_hjb = st.checkbox("HJB-PINN (IA)", value=_hjb_ok, disabled=not _hjb_ok,
                          help="Contrôle optimal appris par réseau de neurones (rapide si poids entraînés).")
    inc_pmp = st.checkbox("PMP (précis, mais lent)", value=False,
                          help="Plan d'intervention pré-optimisé — peut prendre quelques minutes.")
    st.divider()
    st.metric("Contagiosité R₀", f"{R0:.2f}",
              help="Nombre moyen de personnes contaminées par chaque malade. "
                   "Au-dessus de 1, l'épidémie progresse.")
    st.caption(f"≈ chaque malade contamine **{R0:.1f}** personnes.")
    with st.expander("Paramètres du modèle (technique)"):
        st.metric("β (transmission)", f"{p.beta:.3f}")
        st.metric("γ (guérison)", f"{p.gamma:.3f}")
        st.metric("σ (incubation)", f"{p.sigma:.3f}")
        st.metric("Population N", fmt_people(p.N))

# Benchmark calculé une fois, partagé par tous les onglets.
strategies = ["baseline", "lqr"] + (["pmp"] if inc_pmp else []) + (["hjb"] if inc_hjb else [])
bench = run_benchmark(p.beta, p.sigma, p.gamma, p.N, float(horizon), tuple(strategies))
details = bench["details"]

# Meilleure stratégie = plus forte réduction du pic parmi les stratégies réussies.
base_m = details["baseline"]["metrics"]
_candidates = {k: v for k, v in bench["summary"].items()
               if k != "baseline" and "peak_reduction_pct" in v}
best = max(_candidates, key=lambda k: _candidates[k]["peak_reduction_pct"]) if _candidates else None
best_m = details[best]["metrics"] if best else None
avoided = (base_m["final_attack_rate"] - best_m["final_attack_rate"]) if best_m else 0.0

# Trajectoire de référence pour les recommandations (meilleure dispo, sinon baseline).
ref_strat = best or "baseline"
rec = state.recommendation_engine().recommend(details[ref_strat]["trajectory"], float(horizon))

# ---------------------------------------------------------------- l'essentiel
st.subheader("🧭 L'essentiel")
RISK_ICON = {"élevé": "🔴", "modéré": "🟠", "faible": "🟢"}
e1, e2, e3, e4 = st.columns(4)
e1.metric("Niveau de risque", f"{RISK_ICON.get(rec['risk_level'], '⚪')} {rec['risk_level'].capitalize()}",
          help="Basé sur la contagiosité R₀ estimée à partir des données.")
e2.metric("Sans intervention", f"{fmt_people(base_m['peak_I'])} malades",
          f"pic vers J+{base_m['peak_day']:.0f}", delta_color="off",
          help="Nombre maximal de personnes malades en même temps si on ne fait rien.")
if best_m:
    e3.metric(f"Avec la meilleure stratégie", f"{fmt_people(best_m['peak_I'])} malades",
              f"-{_candidates[best]['peak_reduction_pct']:.0f} % de pic", delta_color="inverse",
              help=f"Stratégie : {STRAT_FR.get(best, best)}.")
    e4.metric("Personnes épargnées", f"≈ {fmt_people(avoided)}",
              pct_pop(avoided, p.N) + " de la population", delta_color="off",
              help="Personnes qui ne seront jamais infectées grâce à l'intervention.")
st.success(f"**En une phrase :** risque {rec['risk_level']} (R₀ = {rec['R0']}). "
           f"En déclenchant **vaccination + confinement vers J+{rec['intervention_start_day']}** "
           f"({STRAT_FR.get(ref_strat, ref_strat)}), le pic est "
           + (f"réduit de **{_candidates[best]['peak_reduction_pct']:.0f} %** et "
              f"**≈ {fmt_people(avoided)} personnes** échappent à l'infection." if best_m
              else "à surveiller — sélectionner une stratégie de contrôle dans la barre latérale."))

tab_scn, tab_bench, tab_ctrl, tab_calib, tab_reco, tab_obs = st.tabs(
    ["📈 Tester une politique", "🏁 Comparer les stratégies", "🎛️ Effort à fournir",
     "🧬 Fiabilité du modèle", "📋 Recommandations", "📡 Surveillance"]
)

# ---------------------------------------------------------------- scénario
with tab_scn:
    st.markdown("**Question : que se passe-t-il si j'applique cette politique ?** "
                "(réglez les curseurs vaccination/confinement à gauche)")
    res = state.scenario_service().run(u1, u2, float(horizon))
    traj_scn = res["trajectory"]
    m = res["metrics"]

    k1, k2, k3 = st.columns(3)
    k1.metric("Pic de malades simultanés", fmt_people(m["peak_I"]),
              f"vers J+{m['peak_day']:.0f}", delta_color="off",
              help="Dimensionne la charge hospitalière maximale.")
    k2.metric("Personnes touchées au total", fmt_people(m["final_attack_rate"]),
              pct_pop(m["final_attack_rate"], p.N) + " de la population", delta_color="off")
    delta_vs_base = m["peak_I"] - base_m["peak_I"]
    k3.metric("Pic vs « ne rien faire »",
              f"{100 * delta_vs_base / base_m['peak_I']:+.0f} %",
              "mieux" if delta_vs_base < 0 else "équivalent/pire", delta_color="off")

    c1, c2 = st.columns(2)
    with c1:
        log_scn = st.toggle("Échelle log", value=False, key="log_scn",
                            help="Utile pour voir les petites valeurs (début/fin d'épidémie).")
        st.pyplot(seir_figure(traj_scn, f"Évolution de l'épidémie (u₁={u1:.2f}, u₂={u2:.2f})",
                              log=log_scn), width="stretch")
        st.caption("**Comment lire :** S = pas encore touchés, E = incubation, "
                   "I = malades, R = guéris/immunisés.")
    with c2:
        st.pyplot(reff_figure(traj_scn, R0, p.N, "L'épidémie progresse-t-elle encore ?"),
                  width="stretch")
        st.caption("**Comment lire :** au-dessus de la ligne rouge (R_eff > 1), chaque malade "
                   "contamine plus d'une personne → l'épidémie s'étend. L'objectif d'une bonne "
                   "politique est de passer **sous la ligne au plus tôt**.")
    with st.expander("Détails techniques"):
        st.write({"charge infectieuse ∫I dt": f"{m['total_infected_proxy']:,.0f}",
                  "paramètres": p.as_dict()})

# ---------------------------------------------------------------- benchmark
with tab_bench:
    if best:
        st.success(f"🏆 **Meilleure stratégie : {STRAT_FR.get(best, best)}** — pic réduit de "
                   f"**{_candidates[best]['peak_reduction_pct']:.0f} %**, "
                   f"**≈ {fmt_people(avoided)} personnes épargnées** par rapport à « ne rien faire ».")
    c1, c2 = st.columns([3, 2])
    with c1:
        fig = benchmark_figure(details, strategies, f"Comparaison des stratégies (R₀ = {R0:.2f})")
        st.pyplot(fig, width="stretch")
        st.caption("**Comment lire :** plus la courbe est basse, mieux c'est. "
                   "Une courbe qui touche le bas (I = 1) signifie que l'épidémie est **éteinte**. "
                   "L'échelle est logarithmique : chaque graduation = ×10.")
    with c2:
        rows = []
        for name in strategies:
            det = details.get(name, {})
            if "metrics" not in det:
                continue
            mm = det["metrics"]
            summ = bench["summary"].get(name, {})
            rows.append({
                "Stratégie": STRAT_FR.get(name, name),
                "Pic de malades": fmt_people(mm["peak_I"]),
                "Date du pic": f"J+{mm['peak_day']:.0f}",
                "Réduction du pic": f"{summ.get('peak_reduction_pct', 0.0):.0f} %",
                "Personnes épargnées": fmt_people(max(0.0, base_m["final_attack_rate"]
                                                      - mm["final_attack_rate"])),
            })
        st.dataframe(pd.DataFrame(rows).set_index("Stratégie"), width="stretch")
        errs = {k: v["error"] for k, v in details.items() if "error" in v}
        if errs:
            st.warning("Stratégies non calculées : " + ", ".join(errs))
        with st.expander("Détails techniques"):
            st.json(bench["summary"])

# ---------------------------------------------------------------- contrôles
with tab_ctrl:
    st.markdown("**Question : quel effort (vaccination, confinement) chaque stratégie demande-t-elle, "
                "et quand ?**")
    controlled = [s for s in strategies if s != "baseline"]
    if any("controls" in details.get(s, {}) for s in controlled):
        c1, c2 = st.columns([3, 2])
        with c1:
            st.pyplot(controls_figure(details, controlled, "Effort demandé au cours du temps"),
                      width="stretch")
            st.caption("**Comment lire :** 0 = aucun effort, 1 = effort maximal. Un bon contrôle "
                       "agit **tôt et brièvement** plutôt que tard et longtemps.")
        with c2:
            st.markdown("**Coût social de chaque stratégie** "
                        "(jours-équivalents d'effort maximal) :")
            for s in controlled:
                d = details.get(s, {})
                if "controls" not in d:
                    continue
                eff = (np.trapezoid(d["controls"]["u1"], d["trajectory"]["t"])
                       + np.trapezoid(d["controls"]["u2"], d["trajectory"]["t"]))
                st.metric(STRAT_FR.get(s, s), f"{eff:.0f} jours d'effort",
                          help="Somme de l'intensité de vaccination + confinement sur la période. "
                               "Plus c'est bas, moins la stratégie pèse sur la société.")
            st.caption("Un confinement total de 10 jours « coûte » 10 jours d'effort ; "
                       "un demi-confinement de 20 jours aussi.")
    else:
        st.info("Cochez au moins une stratégie (LQR/PMP/HJB) dans la barre latérale.")

# ---------------------------------------------------------------- calibration
with tab_calib:
    st.markdown("**Question : peut-on faire confiance aux prévisions ?** Le modèle a été ajusté "
                "sur les données observées — plus la ligne suit les points, plus il est fiable.")
    latest = state.registry.get_latest("seir_params")
    c1, c2 = st.columns([3, 2])
    with c1:
        if state.data is not None:
            from domain.seir import simulate
            t_obs = np.asarray(state.data["t"], float)
            I_obs = np.asarray(state.data["y"], float)[:, 2]
            model = simulate(p, np.asarray(state.data["y0"], float),
                             (0, float(t_obs.max())), len(t_obs))
            st.pyplot(fit_figure(t_obs, I_obs, model, "Données observées vs modèle"),
                      width="stretch")
    with c2:
        if latest:
            st.markdown(f"✅ Modèle calibré — méthode : *{latest['metadata'].get('method', '—')}*, "
                        f"version **{latest['version']}**.")
            uq = latest["metadata"].get("uncertainty")
            if uq and "beta" in uq:
                lo, hi = uq["beta"]["ci95"]
                g = p.gamma
                st.metric("Contagiosité R₀ (avec marge d'erreur)",
                          f"{R0:.2f}", f"entre {lo / g:.2f} et {hi / g:.2f}", delta_color="off",
                          help="Intervalle de confiance à 95 % issu de la quantification d'incertitude.")
            with st.expander("Détails techniques (incertitude, versions)"):
                if uq:
                    df = pd.DataFrame({
                        k: {"moyenne": v["mean"], "écart-type": v["std"],
                            "IC95 bas": v["ci95"][0], "IC95 haut": v["ci95"][1]}
                        for k, v in uq.items()
                    }).T
                    st.dataframe(df.style.format("{:.4f}"), width="stretch")
                versions = state.registry.list_versions("seir_params")
                if versions:
                    st.dataframe(
                        pd.DataFrame([
                            {"version": v["version"],
                             "méthode": v["metadata"].get("method", "—"),
                             "date": pd.Timestamp(v["created_at"], unit="s").strftime("%Y-%m-%d %H:%M")}
                            for v in versions
                        ]).set_index("version"),
                        width="stretch",
                    )

# ---------------------------------------------------------------- recommandations
with tab_reco:
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Niveau de risque", f"{RISK_ICON.get(rec['risk_level'], '⚪')} {rec['risk_level'].capitalize()}")
    r2.metric("Quand agir ?", f"J+{rec['intervention_start_day']}",
              help="Déclencher l'intervention ce jour-là : ~2 semaines avant le seuil d'alerte.")
    r3.metric("Pic attendu", f"J+{rec['projected_peak_day']}")
    r4.metric("Seuil d'alerte", f"{fmt_people(rec['alert_threshold_I'])} malades",
              help="1 % de la population malade en même temps.")
    st.info("📌 " + rec["advice"])
    st.caption(f"Recommandation basée sur la stratégie : {STRAT_FR.get(ref_strat, ref_strat)}.")

# ---------------------------------------------------------------- surveillance
with tab_obs:
    st.markdown("**Question : le modèle est-il toujours valable quand de nouvelles données arrivent ?** "
                "(simulez un changement avec le curseur)")
    c1, c2 = st.columns([3, 2])
    shift = c2.slider("Facteur de décalage des nouvelles données", 0.5, 5.0, 1.0, 0.1,
                      help="1,0 = données identiques à la calibration ; 2,0 = deux fois plus de cas.")
    ref = state.data["y"][:, 2]
    new = ref * shift
    drift = state.drift_monitor().check(new)
    with c1:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7.3, 3.6), dpi=170)
        t_obs = np.asarray(state.data["t"], float)
        ax.plot(t_obs, ref, color="#149E8E", lw=2.2, label="Données de calibration")
        ax.plot(t_obs, new, color="#E74C3C", lw=2.2, ls="--", label="Nouvelles données")
        ax.set_xlabel("Jours"); ax.set_ylabel("Malades I(t)")
        ax.set_title("Les nouvelles données ressemblent-elles aux anciennes ?", fontweight="bold")
        ax.legend(frameon=False); ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig, width="stretch")
    with c2:
        ok = not drift["drift_detected"]
        st.metric("Le modèle reste-t-il fiable ?", "Oui ✅" if ok else "Non ⚠️")
        st.markdown("**Action :** " + ("aucune — les prévisions restent valables."
                                       if ok else "**recalibrer le modèle** sur les nouvelles "
                                                  "données avant toute décision."))
        with st.expander("Détails techniques (test KS)"):
            st.write(drift)
