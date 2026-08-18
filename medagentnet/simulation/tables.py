"""
MedAgentNet - LaTeX table generation for the R1 results.

Every number printed here comes straight from the results dictionary, so the
tables in the manuscript can be regenerated from a run rather than transcribed.
"""
from __future__ import annotations


def _pct(x, digits=1):
    try:
        return f"{100 * float(x):.{digits}f}"
    except (TypeError, ValueError):
        return "--"


def _ci(d):
    if not isinstance(d, dict) or "ci95" not in d:
        return "--"
    lo, hi = d["ci95"]
    return f"{_pct(d['rate'])} [{_pct(lo)}, {_pct(hi)}]"


def _ms(d, key="mean", digits=3):
    if not isinstance(d, dict):
        return "--"
    try:
        return f"{d[key]:.{digits}f}"
    except (KeyError, TypeError, ValueError):
        return "--"


def _msd(d, digits=3):
    if not isinstance(d, dict) or "mean" not in d:
        return "--"
    return f"{d['mean']:.{digits}f} $\\pm$ {d.get('std', 0):.{digits}f}"


def table_tiers(e1) -> str:
    rows = e1.get("rows", {})
    L = [r"\begin{table}[!t]", r"\centering",
         r"\caption{Effect of the disclosure tier on utility and on measured "
         r"information exposure. Detection uses the strict, ground-truth-matched "
         r"criterion; exposure is the mean fraction of each responding "
         r"department's identifiable facts that crossed the boundary. "
         r"Mean $\pm$ s.d. over three seeds.}",
         r"\label{tab:r1_tiers}", r"\small",
         r"\begin{tabular}{lccc}", r"\toprule",
         r"\textbf{Metric} & \textbf{Tier 1} & \textbf{Tier 2} & \textbf{Tier 3} \\",
         r" & (Flag only) & (Clinical summary) & (Full context) \\",
         r"\midrule"]

    def cell(t, path):
        d = rows.get(f"tier_{t}", {})
        cur = d
        for p in path:
            cur = cur.get(p, {}) if isinstance(cur, dict) else {}
        return cur

    for label, path, fmt in [
        ("Conflict detection (\\%)", ("pooled", "conflict_detection/localised_or_better"), "ci"),
        ("Pattern detection (\\%)", ("pooled", "pattern_detection/localised_or_better"), "ci"),
        ("Precision", ("aggregate", "classification/precision"), "msd"),
        ("Recall", ("aggregate", "classification/recall"), "msd"),
        ("F1", ("aggregate", "classification/f1"), "msd"),
        ("Field exposure (\\%)", ("leakage", "mean_field_exposure"), "pctmsd"),
        ("Items disclosed per response", ("leakage", "mean_items_disclosed"), "msd"),
        ("Mean anonymity set", ("leakage", "mean_anonymity_set"), "msd"),
        ("Responses singling out the patient (\\%)", ("leakage", "singled_out_rate"), "pctmsd"),
        ("Latency (s)", ("aggregate", "latency_seconds/mean"), "msd"),
    ]:
        cells = []
        for t in (1, 2, 3):
            v = cell(t, path)
            # No response at this tier disclosed an identifiable item, so the
            # anonymity set is undefined rather than zero. Printing 0 would
            # read as "maximally identifying", the opposite of the truth.
            if label.startswith(("Mean anonymity", "Responses singling")) and \
                    isinstance(v, dict) and v.get("n", 0) == 0:
                cells.append("n/a (nothing disclosed)")
            elif fmt == "ci":
                cells.append(_ci(v))
            elif fmt == "pctmsd":
                cells.append(f"{_pct(v.get('mean', 0))} $\\pm$ {_pct(v.get('std', 0))}"
                             if isinstance(v, dict) else "--")
            else:
                cells.append(_msd(v))
        L.append(f"{label} & " + " & ".join(cells) + r" \\")

    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(L)


def table_consent(e2) -> str:
    L = [r"\begin{table}[!t]", r"\centering",
         r"\caption{Consent restriction sweep. At each restriction level, ten "
         r"restriction graphs were sampled independently; directed department "
         r"pairs are revoked uniformly at random per patient. Detection rates "
         r"are pooled over graphs with Wilson 95\% intervals.}",
         r"\label{tab:r1_consent}", r"\small",
         r"\begin{tabular}{lccccc}", r"\toprule",
         r"\textbf{Pairs revoked} & \textbf{Observed} & \textbf{Conflict det.} & "
         r"\textbf{Pattern det.} & \textbf{Recall} & \textbf{F1} \\",
         r" & \textbf{denial rate} & \textbf{(\%, CI)} & \textbf{(\%, CI)} & & \\",
         r"\midrule"]
    for row in e2.get("rows", []):
        L.append(
            f"{_pct(row['restriction_fraction'], 0)}\\% & "
            f"{_pct(row['observed_denial_rate'].get('mean', 0))}\\% & "
            f"{_ci(row['conflict_detection'])} & "
            f"{_ci(row['pattern_detection'])} & "
            f"{_msd(row['recall'])} & {_msd(row['f1'])} \\\\"
        )
    L += [r"\midrule",
          r"\multicolumn{6}{@{}l}{\textit{Targeted removal of the pairs carrying "
          r"the evidence, matched for the number of pairs removed}} \\"]
    for row in e2.get("targeted_removal", {}).get("rows", []):
        L.append(f"{row['removal'].replace('_', ' ')} & -- & "
                 f"{_ci(row['conflict_detection'])} & {_ci(row['pattern_detection'])} "
                 f"& -- & {row['f1']:.3f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(L)


def table_ablation(e4) -> str:
    pretty = {
        "medagentnet": "MedAgentNet (full)",
        "medagentnet_grounded_only": "\\quad grounded synthesis only",
        "medagentnet_llm_only": "\\quad model synthesis only",
        "ablate_synthesis": "-- cross-departmental synthesis",
        "ablate_orchestration": "-- orchestration (single department)",
        "ablate_relevance_routing": "-- relevance routing (broadcast)",
        "ablate_tiers": "-- tiered disclosure",
        "ablate_consent": "-- consent service",
        "ablate_structured_protocol": "-- structured query protocol",
        "ablate_freetext_parser": "-- free-text response rescue",
    }
    L = [r"\begin{table*}[!t]", r"\centering",
         r"\caption{Ablation matrix. Each row removes or replaces one component "
         r"and is otherwise identical: same patients, same queries, same scorer. "
         r"$p$ is a McNemar test against the full system on matched scenarios.}",
         r"\label{tab:r1_ablation}", r"\small",
         r"\begin{tabular}{lccccccc}", r"\toprule",
         r"\textbf{Configuration} & \textbf{Conflict det.} & \textbf{Pattern det.} & "
         r"\textbf{Precision} & \textbf{Recall} & \textbf{F1} & \textbf{Queries/} & "
         r"\textbf{$p$} \\",
         r" & \textbf{(\%, CI)} & \textbf{(\%, CI)} & & & & \textbf{scenario} & \\",
         r"\midrule"]
    for key, label in pretty.items():
        row = e4.get("rows", {}).get(key)
        if not row:
            continue
        p = row.get("vs_medagentnet", {}).get("p_value")
        p_txt = "--" if key == "medagentnet" else (
            "$<$0.001" if isinstance(p, float) and p < 0.001 else
            (f"{p:.3f}" if isinstance(p, float) else "--"))
        L.append(
            f"{label} & {_ci(row['conflict_detection'])} & "
            f"{_ci(row['pattern_detection'])} & {_msd(row['precision'])} & "
            f"{_msd(row['recall'])} & {_msd(row['f1'])} & "
            f"{_ms(row['queries_per_scenario'], 'mean', 1)} & {p_txt} \\\\"
        )
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    return "\n".join(L)


def table_baselines(e5) -> str:
    pretty = {
        "medagentnet": "MedAgentNet",
        "centralized_llm": "Centralized single agent (full record)",
        "centralized_rules": "Conventional CDSS on aggregated record",
        "direct_retrieval": "Federated retrieval, no reasoning layer",
    }
    L = [r"\begin{table}[!t]", r"\centering",
         r"\caption{Comparison against systems that differ in architecture rather "
         r"than in backend. All rows use identical patients, queries and scoring.}",
         r"\label{tab:r1_baselines}", r"\small",
         r"\begin{tabular}{lcccccc}", r"\toprule",
         r"\textbf{System} & \textbf{Data} & \textbf{Precision} & \textbf{Recall} & "
         r"\textbf{F1} & \textbf{Latency} & \textbf{$p$} \\",
         r" & \textbf{centralised} & & & & \textbf{(s)} & \\", r"\midrule"]
    for key, label in pretty.items():
        row = e5.get("rows", {}).get(key)
        if not row:
            continue
        p = row.get("vs_medagentnet", {}).get("p_value")
        p_txt = "--" if key == "medagentnet" else (
            "$<$0.001" if isinstance(p, float) and p < 0.001 else
            (f"{p:.3f}" if isinstance(p, float) else "--"))
        L.append(f"{label} & {'yes' if row.get('data_centralised') else 'no'} & "
                 f"{_msd(row['precision'])} & {_msd(row['recall'])} & "
                 f"{_msd(row['f1'])} & {_ms(row.get('latency_s', {}), 'mean', 4)} & "
                 f"{p_txt} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(L)


def table_backends(e6) -> str:
    L = [r"\begin{table*}[!t]", r"\centering",
         r"\caption{Backend comparison. The coordination, consent and disclosure "
         r"layers are unchanged across rows; only the reasoning backend differs.}",
         r"\label{tab:r1_backends}", r"\small",
         r"\begin{tabular}{lcccccc}", r"\toprule",
         r"\textbf{Backend} & \textbf{Precision} & \textbf{Recall} & \textbf{F1} & "
         r"\textbf{Format} & \textbf{Latency} & \textbf{Completion} \\",
         r" & & & & \textbf{failures} & \textbf{(s)} & \textbf{tokens} \\",
         r"\midrule"]
    for name, row in e6.get("rows", {}).items():
        L.append(
            f"{row.get('identity', name)} & {_msd(row['precision'])} & "
            f"{_msd(row['recall'])} & {_msd(row['f1'])} & "
            f"{_pct(row.get('format_failure_rate') or 0)}\\% & "
            f"{row.get('mean_latency_s') or 0:.2f} & "
            f"{row.get('approx_completion_tokens') or 0} \\\\"
        )
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    return "\n".join(L)


def table_scalability(e7) -> str:
    L = [r"\begin{table}[!t]", r"\centering",
         r"\caption{Scalability. Communication volume is the number of inter-agent "
         r"queries per scenario. Latency percentiles are end-to-end per scenario.}",
         r"\label{tab:r1_scalability}", r"\small",
         r"\begin{tabular}{llccccc}", r"\toprule",
         r"\textbf{Varied} & \textbf{Value} & \textbf{Queries/} & "
         r"\textbf{Throughput} & \textbf{p50} & \textbf{p95} & \textbf{F1} \\",
         r" & & \textbf{scenario} & \textbf{(scen./s)} & \textbf{(s)} & \textbf{(s)} & \\",
         r"\midrule"]
    for row in e7.get("by_patient_count", []):
        lat = row.get("latency", {})
        L.append(f"Patients & {row['patients']} & {row['queries_per_scenario']} & "
                 f"{row['scenarios_per_second']:.2f} & {lat.get('p50', 0):.2f} & "
                 f"{lat.get('p95', 0):.2f} & {row['f1']:.3f} \\\\")
    L.append(r"\midrule")
    for row in e7.get("by_department_count", []):
        lat = row.get("latency", {})
        L.append(f"Departments & {row['departments']} & {row['queries_per_scenario']} & "
                 f"{row['scenarios_per_second']:.2f} & {lat.get('p50', 0):.2f} & "
                 f"{lat.get('p95', 0):.2f} & {row['f1']:.3f} \\\\")
    L.append(r"\midrule")
    for row in e7.get("by_concurrency", []):
        lat = row.get("latency", {})
        L.append(f"Concurrency & {row['concurrency']} & {row['queries_per_scenario']} & "
                 f"{row['scenarios_per_second']:.2f} & {lat.get('p50', 0):.2f} & "
                 f"{lat.get('p95', 0):.2f} & {row['f1']:.3f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(L)


def table_adversarial(e8) -> str:
    L = [r"\begin{table*}[!t]", r"\centering",
         r"\caption{Adversarial and failure-mode evaluation. Each threat is run "
         r"with and without its mitigation on the same cohort.}",
         r"\label{tab:r1_adversarial}", r"\small",
         r"\begin{tabular}{p{3.1cm}p{4.6cm}p{3.4cm}p{3.4cm}}", r"\toprule",
         r"\textbf{Threat} & \textbf{Mitigation} & \textbf{Without} & \textbf{With} \\",
         r"\midrule"]

    a3 = e8.get("A3_prompt_injection", {})
    if a3:
        L.append(
            r"A3 instruction injection in free text & "
            r"schema-validated responses; notes withheld below Tier 3 & "
            f"{a3.get('injected', {}).get('marker_disclosures', 0)} out-of-policy "
            f"disclosures / {a3.get('injected', {}).get('responses', 0)} responses & "
            f"{a3.get('injected_with_mitigations', {}).get('marker_disclosures', 0)} / "
            f"{a3.get('injected_with_mitigations', {}).get('responses', 0)} \\\\")

    a2 = e8.get("A2_compromised_agent", {})
    if a2:
        ind = a2.get("independent_compromise", {})
        w = ind.get("without_mitigation", {})
        m = ind.get("with_corroboration_requirement", {})
        col = a2.get("colluding_compromise", {}).get(
            "with_corroboration_requirement", {})
        L.append(
            r"A2 compromised agent (independent) & "
            r"critical alerts require corroboration from a second department & "
            f"{w.get('fabricated_at_critical', 0)} fabricated critical alerts & "
            f"{m.get('fabricated_at_critical', 0)} fabricated critical alerts \\\\")
        L.append(
            r"A2 compromised agents (colluding) & as above & -- & "
            f"{col.get('fabricated_at_critical', 0)} fabricated critical alerts "
            r"(mitigation does not hold) \\")

    a1 = e8.get("A1_differencing", {})
    if a1:
        w = a1.get("unbounded", {})
        m = a1.get("with_query_budget", {})
        L.append(
            r"A1 repeated-query differencing & "
            f"per-pair query budget of {m.get('budget', 3)} & "
            f"{_pct(w.get('mean_final_reconstruction', 0))}\\% of record recovered & "
            f"{_pct(m.get('mean_final_reconstruction', 0))}\\% recovered \\\\")

    a4 = e8.get("A4_token_replay", {})
    if a4:
        wo = a4.get("results", {}).get("without_validation", {})
        wi = a4.get("results", {}).get("with_validation", {})
        n_wo = sum(1 for v in wo.values() if v.get("token_accepted"))
        n_wi = sum(1 for v in wi.values() if v.get("token_accepted"))
        L.append(
            r"A4 consent-token replay & "
            r"tokens bound to patient, pair and tier; single use; short expiry & "
            f"{n_wo}/{len(wo)} replay attempts accepted & "
            f"{n_wi}/{len(wi)} accepted \\\\")

    a5 = e8.get("A5_availability", {})
    if a5 and a5.get("rows"):
        rows = a5["rows"]
        base = rows[0]
        worst = rows[-2] if len(rows) > 2 else rows[-1]
        L.append(
            r"A5 unavailable agents & "
            r"partial results returned with an explicit coverage statement & "
            f"F1 {base.get('f1', 0):.3f} with all agents & "
            f"F1 {worst.get('f1', 0):.3f} with "
            f"{worst.get('agents_unavailable', 0)} unavailable \\\\")

    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    return "\n".join(L)


def table_context_audit(e9) -> str:
    r0 = e9.get("r0_construction", {})
    r1 = e9.get("r1_construction", {})
    L = [r"\begin{table}[!t]", r"\centering",
         r"\caption{Audit of the query context. A context field is "
         r"\emph{disallowed} if it is not one a requesting clinician supplies. "
         r"A \emph{label term} is a token drawn from the ground-truth entry for "
         r"that patient.}",
         r"\label{tab:r1_context_audit}", r"\small",
         r"\begin{tabular}{lcc}", r"\toprule",
         r"\textbf{Property of the query context} & \textbf{R0} & \textbf{R1} \\",
         r"\midrule",
         f"Fields per context & {r0.get('mean_keys', '--')} & {r1.get('mean_keys', '--')} \\\\",
         f"Disallowed fields per context & {r0.get('mean_disallowed_keys', '--')} & "
         f"{r1.get('mean_disallowed_keys', '--')} \\\\",
         f"Contexts carrying ground-truth text & "
         f"{r0.get('contexts_carrying_ground_truth', '--')} of {r0.get('n_contexts', '--')} & "
         f"{r1.get('contexts_carrying_ground_truth', '--')} of {r1.get('n_contexts', '--')} \\\\",
         f"Label terms per context & {r0.get('mean_label_terms_present', '--')} & "
         f"{r1.get('mean_label_terms_present', '--')} \\\\",
         r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(L)


def table_variance(e3) -> str:
    dev = e3.get("development_family", {}).get("across_runs", {})
    held = e3.get("heldout_family", {}).get("across_runs", {})
    devp = e3.get("development_family", {}).get("pooled", {})
    heldp = e3.get("heldout_family", {}).get("pooled", {})
    L = [r"\begin{table}[!t]", r"\centering",
         r"\caption{Stability across seeds and generalisation to a held-out "
         r"scenario family written after the prompts and rules were frozen. "
         r"Mean $\pm$ s.d. over five seeds; detection pooled with Wilson 95\% "
         r"intervals.}",
         r"\label{tab:r1_variance}", r"\small",
         r"\begin{tabular}{lcc}", r"\toprule",
         r"\textbf{Metric} & \textbf{Development family} & \textbf{Held-out family} \\",
         r"\midrule",
         f"Conflict detection (\\%) & "
         f"{_ci(devp.get('conflict_detection/localised_or_better', {}))} & "
         f"{_ci(heldp.get('conflict_detection/localised_or_better', {}))} \\\\",
         f"Pattern detection (\\%) & "
         f"{_ci(devp.get('pattern_detection/localised_or_better', {}))} & "
         f"{_ci(heldp.get('pattern_detection/localised_or_better', {}))} \\\\",
         f"False alarms, matched negatives (\\%) & "
         f"{_ci(devp.get('false_alarms/matched_negatives', {}))} & "
         f"{_ci(heldp.get('false_alarms/matched_negatives', {}))} \\\\",
         f"Precision & {_msd(dev.get('classification/precision', {}))} & "
         f"{_msd(held.get('classification/precision', {}))} \\\\",
         f"Recall & {_msd(dev.get('classification/recall', {}))} & "
         f"{_msd(held.get('classification/recall', {}))} \\\\",
         f"F1 & {_msd(dev.get('classification/f1', {}))} & "
         f"{_msd(held.get('classification/f1', {}))} \\\\",
         r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(L)


def generate_all_tables(results: dict) -> str:
    parts = ["% Generated by simulation/tables.py from a single results.json.",
             f"% Run: {results.get('meta', {}).get('timestamp', '')}",
             f"% Backend: {results.get('meta', {}).get('provider', '')}", ""]
    builders = [
        ("e9", table_context_audit), ("e1", table_tiers), ("e2", table_consent),
        ("e3", table_variance), ("e3", table_per_scenario),
        ("e4", table_ablation), ("e5", table_baselines),
        ("e6", table_backends), ("e7", table_scalability), ("e8", table_adversarial),
    ]
    for key, fn in builders:
        block = results.get(key)
        if isinstance(block, dict) and "error" not in block:
            try:
                parts.append(fn(block))
            except Exception as e:
                parts.append(f"% table for {key} failed: {e}\n")
    return "\n".join(parts)


def table_per_scenario(e3) -> str:
    """Per-scenario detection and per-control false alarms, pooled over seeds,
    plus the effect of record quality. This is the breakdown the reviewers asked
    for and it is generated from the same results file as everything else."""
    runs = (e3.get("development_family", {}) or {}).get("per_run", [])
    held = (e3.get("heldout_family", {}) or {}).get("per_run", [])
    if not runs:
        return ""

    def pool(key, num, den):
        acc = {}
        for r in runs + held:
            for name, d in (r.get(key, {}) or {}).items():
                a = acc.setdefault(name, [0, 0])
                a[0] += d.get(num, 0)
                a[1] += d.get(den, 0)
        return acc

    per_scen = pool("per_scenario", "localised", "n")
    per_neg = pool("per_negative", "false_alarms", "n")

    quality = {}
    for r in runs + held:
        for flag, d in (r.get("by_record_quality", {}) or {}).items():
            a = quality.setdefault(flag, [0, 0])
            a[0] += d.get("k", 0)
            a[1] += d.get("n", 0)

    def _rate(k, n):
        if not n:
            return "--"
        from simulation.evaluation import wilson
        p, lo, hi = wilson(k, n)
        return f"{_pct(p)} [{_pct(lo)}, {_pct(hi)}] ({k}/{n})"

    L = [r"\begin{table}[!t]", r"\centering",
         r"\caption{Per-scenario detection, per-control false alarms, and the "
         r"effect of record quality. Pooled over five seeds and over both "
         r"scenario families, with Wilson 95\% intervals and counts.}",
         r"\label{tab:r1_per_scenario}", r"\small",
         r"\begin{tabular}{lc}", r"\toprule",
         r"\textbf{Scenario} & \textbf{Detection (\%, CI, $k/n$)} \\", r"\midrule",
         r"\multicolumn{2}{@{}l}{\textit{Planted findings}} \\"]
    for name in sorted(per_scen):
        k, n = per_scen[name]
        L.append(f"\\quad {name.replace('_',' ')} & {_rate(k, n)} \\\\")
    L += [r"\midrule",
          r"\multicolumn{2}{@{}l}{\textit{Drug-matched negative controls "
          r"(false-alarm rate)}} \\"]
    for name in sorted(per_neg):
        k, n = per_neg[name]
        L.append(f"\\quad {name.replace('_',' ')} & {_rate(k, n)} \\\\")
    if quality:
        L += [r"\midrule",
              r"\multicolumn{2}{@{}l}{\textit{Detection by record quality}} \\"]
        pretty_q = {"clean_record": "no corruption applied",
                    "contradiction": "contradictory documentation",
                    "stale_medication": "stale prescription present",
                    "resolved_diagnosis": "resolved diagnosis present",
                    "duplicate_entry": "brand/generic duplicate",
                    "note_only_limb": "conflict limb only in free text"}
        for flag in sorted(quality):
            k, n = quality[flag]
            L.append(f"\\quad {pretty_q.get(flag, flag.replace('_',' '))} & "
                     f"{_rate(k, n)} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(L)
