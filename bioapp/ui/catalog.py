"""Display metadata for the dialogs: Prism-style analysis categories, the equation tree and equation texts.

Pure data (no Qt, no computation). Keys are ``AnalysisKind.id`` / ``Model.id``; tests check that every registered
kind and model is covered.
"""
from __future__ import annotations

# ------------------------------------------------------------------ Analyze dialog categories (Prism order)
CATEGORIES = ("Transform, Normalize", "XY analyses", "Column analyses", "Grouped analyses",
              "Contingency table analyses", "Survival analyses", "Nested analyses", "Multiple variable analyses")

KIND_CATEGORY = {
    "normalize": CATEGORIES[0], "subtract_baseline": CATEGORIES[0],
    "nonlin_fit": CATEGORIES[1], "linear_regression": CATEGORIES[1], "deming_regression": CATEGORIES[1],
    "simple_logistic_regression": CATEGORIES[1], "auc": CATEGORIES[1], "correlation": CATEGORIES[1],
    "descriptive": CATEGORIES[2], "normality": CATEGORIES[2], "outliers": CATEGORIES[2],
    "frequency_distribution": CATEGORIES[2], "t_test": CATEGORIES[2], "mann_whitney": CATEGORIES[2],
    "wilcoxon_paired": CATEGORIES[2], "one_way_anova": CATEGORIES[2], "kruskal_wallis": CATEGORIES[2],
    "friedman": CATEGORIES[2], "rm_one_way_anova": CATEGORIES[2],
    "two_way_anova": CATEGORIES[3], "rm_two_way_anova": CATEGORIES[3], "three_way_anova": CATEGORIES[3],
    "contingency": CATEGORIES[4],
    "survival": CATEGORIES[5],
    "nested_t_test": CATEGORIES[6], "nested_one_way_anova": CATEGORIES[6],
    "cox_regression": CATEGORIES[7], "multiple_linear_regression": CATEGORIES[7],
    "multiple_logistic_regression": CATEGORIES[7], "poisson_regression": CATEGORIES[7], "pca": CATEGORIES[7],
    "hierarchical_clustering": CATEGORIES[7], "kmeans": CATEGORIES[7],
}

TABLE_KIND_LABEL = {"XY": "XY", "Column": "Column", "Grouped": "Grouped", "Contingency": "Contingency",
                    "Survival": "Survival", "Nested": "Nested", "MultipleVariables": "Multiple variables"}

# ------------------------------------------------------------------ option sections of long forms
SECTION_ORDER = ("Fit", "Constraints", "Compare", "Clusters", "Output")
SECTION_OF = {
    **dict.fromkeys(("constraints", "shared"), "Constraints"),
    **dict.fromkeys(("compare", "compare_alpha", "multiple_comparisons", "mc_family", "control", "correction",
                     "family", "sphericity", "welch_bf"), "Compare"),
    **dict.fromkeys(("rows_method", "rows_count", "rows_cutoff", "cols_method", "cols_count", "cols_cutoff"), "Clusters"),
    **dict.fromkeys(("ci_level", "ci_method", "interpolate", "bands", "alpha", "prop_ci", "percent_of", "at_risk_times",
                     "distance_matrix", "simulations", "seed", "runs_test", "normality", "cutoff", "units", "relative",
                     "replicates", "mark"), "Output"),
}

# ------------------------------------------------------------------ equation tree (Prism's built-in list)
# family -> subgroup ("" = leaves directly under the family) -> model ids
EQUATION_TREE = {
    "Dose-response": {
        "Stimulation": ("log_agonist_variable", "log_agonist_standard", "log_agonist_normalized", "agonist_variable"),
        "Inhibition": ("log_inhibitor_variable", "log_inhibitor_standard", "log_inhibitor_normalized", "inhibitor_variable"),
        "Special": ("dr_5pl_log", "dr_5pl_dose", "dr_biphasic_log", "dr_biphasic_dose", "dr_bell_log", "dr_bell_dose",
                    "dr_operational_log", "dr_operational_global", "dr_operational_dose", "dr_schild_log",
                    "dr_schild_dose", "dr_schild_pa2_log", "dr_schild_pa2_dose", "dr_allosteric_log",
                    "dr_allosteric_dose", "dr_ecanything_log", "dr_ecanything_dose"),
    },
    "Binding": {
        "Saturation": ("sat_total", "sat_specific", "sat_total_ns", "sat_two_total_ns", "sat_two_specific", "sat_hill"),
        "Competitive": ("comp_logic50", "comp_ki", "comp_two_logic50", "comp_two_ki"),
        "Kinetics": ("kin_association", "kin_dissociation", "kin_assoc_dissoc"),
    },
    "Enzyme kinetics": {
        "Velocity": ("enz_mm", "enz_kcat", "enz_allosteric", "enz_substrate_inhibition"),
        "Inhibition": ("enz_competitive", "enz_noncompetitive", "enz_uncompetitive", "enz_mixed", "enz_morrison"),
    },
    "Exponential": {"": ("gen_exp_growth", "gen_decay1", "gen_assoc1", "gen_decay2", "gen_assoc2", "gen_plateau_decay")},
    "Lines": {"": ("line",)},
    "Polynomial": {"": ("gen_poly1", "gen_poly2", "gen_poly3", "gen_poly4")},
    "Gaussian": {"": ("gen_gaussian", "gen_lorentzian", "gen_lognormal")},
    "Sine waves": {"": ("gen_sine", "gen_sine_baseline", "gen_sine_damped")},
    "Growth": {"": ("gen_gompertz", "gen_logistic", "gen_beta")},
    "Linear-quadratic": {"": ("gen_lq_survival", "gen_lq_survival_pct", "gen_lq_survival_y0", "gen_lq_killed")},
    # Prism's standard-curve names for models listed above: (display name, model id)
    "Standard curves to interpolate": {"": (
        ("Sigmoidal, 4PL, X is concentration", "agonist_variable"),
        ("Sigmoidal, 4PL, X is log(concentration)", "log_agonist_variable"),
        ("Sigmoidal, 5PL, X is concentration", "dr_5pl_dose"),
        ("Sigmoidal, 5PL, X is log(concentration)", "dr_5pl_log"),
        ("Hyperbola", "sat_specific"), ("Straight line", "line"),
        ("Second order polynomial (quadratic)", "gen_poly2"), ("Third order polynomial (cubic)", "gen_poly3"),
        ("Fourth order polynomial", "gen_poly4"))},
}

_HILL = "Bottom + (Top-Bottom)/(1+10^((LogEC50-X)*HillSlope))"
_NORM = "100/(1+10^(({p}-X)*HillSlope))"
_POLY = "Y=" + " + ".join(["B0", "B1*X"] + [f"B{i}*X^{i}" for i in range(2, 5)])
_SCHILD = "antag = 1 + (B/10^(-pA2))^SchildSlope"
_LQ = "exp(-(A*X + B*X^2))"

EQUATIONS = {
    "log_agonist_variable": "Y=" + _HILL,
    "log_agonist_standard": "Y=Bottom + (Top-Bottom)/(1+10^(LogEC50-X))",
    "log_inhibitor_variable": "Y=Bottom + (Top-Bottom)/(1+10^((LogIC50-X)*HillSlope))   (HillSlope is negative)",
    "log_inhibitor_standard": "Y=Bottom + (Top-Bottom)/(1+10^(X-LogIC50))",
    "log_agonist_normalized": "Y=" + _NORM.format(p="LogEC50"),
    "log_inhibitor_normalized": "Y=" + _NORM.format(p="LogIC50") + "   (HillSlope is negative)",
    "agonist_variable": "Y=Bottom + X^HillSlope*(Top-Bottom)/(X^HillSlope + EC50^HillSlope)",
    "inhibitor_variable": "Y=Bottom + (Top-Bottom)/(1+(IC50/X)^HillSlope)   (HillSlope is negative)",
    "line": "Y=Slope*X + Y-intercept",
    "sat_total": "Y=Bmax*X/(Kd+X) + NS*X + Background",
    "sat_specific": "Y=Bmax*X/(Kd+X)",
    "sat_total_ns": "Data set A (total): Y=Bmax*X/(Kd+X) + NS*X + Background\nData set B (nonspecific): Y=NS*X + Background",
    "sat_two_total_ns": ("Data set A (total): Y=BmaxHi*X/(KdHi+X) + BmaxLo*X/(KdLo+X) + NS*X + Background\n"
                         "Data set B (nonspecific): Y=NS*X + Background"),
    "sat_two_specific": "Y=BmaxHi*X/(KdHi+X) + BmaxLo*X/(KdLo+X)",
    "sat_hill": "Y=Bmax*X^h/(Kd^h + X^h)",
    "comp_logic50": "Y=Bottom + (Top-Bottom)/(1+10^(X-LogIC50))",
    "comp_ki": "logEC50=LogKi + log10(1+HotNM/HotKdNM)\nY=Bottom + (Top-Bottom)/(1+10^(X-logEC50))",
    "comp_two_logic50": "Y=Bottom + (Top-Bottom)*(FractionHi/(1+10^(X-LogIC50Hi)) + (1-FractionHi)/(1+10^(X-LogIC50Lo)))",
    "comp_two_ki": ("logEC50Hi=LogKiHi + log10(1+HotNM/HotKdNMHi)\nlogEC50Lo=LogKiLo + log10(1+HotNM/HotKdNMLo)\n"
                    "Y=Bottom + (Top-Bottom)*(FractionHi/(1+10^(X-logEC50Hi)) + (1-FractionHi)/(1+10^(X-logEC50Lo)))"),
    "kin_association": "L=HotNM*1E-9\nKob=Kon*L + Koff\nY=Bmax*L/(L+Koff/Kon)*(1-exp(-Kob*X))",
    "kin_dissociation": "Y=(Y0-NS)*exp(-K*X) + NS",
    "kin_assoc_dissoc": ("L=HotNM*1E-9;  Kob=Kon*L + Koff;  Eq=Bmax*L/(L+Koff/Kon)\n"
                         "Y=Eq*(1-exp(-Kob*X)) + NS for X<Time0\nY=Eq*(1-exp(-Kob*Time0))*exp(-Koff*(X-Time0)) + NS for X>=Time0"),
    "enz_mm": "Y=Vmax*X/(Km+X)",
    "enz_kcat": "Y=Et*kcat*X/(Km+X)",
    "enz_allosteric": "Y=Vmax*X^h/(Khalf^h + X^h)",
    "enz_substrate_inhibition": "Y=Vmax*X/(Km + X*(1+X/Ki))",
    "enz_competitive": "Y=Vmax*X/(Km*(1+I/Ki) + X)",
    "enz_noncompetitive": "Y=Vmax/(1+I/Ki)*X/(Km+X)",
    "enz_uncompetitive": "f=1+I/Ki\nY=Vmax/f*X/(Km/f + X)",
    "enz_mixed": "a=1+I/(Alpha*Ki);  Vapp=Vmax/a;  Kapp=Km*(1+I/Ki)/a\nY=Vapp*X/(Kapp+X)",
    "enz_morrison": "b=Et + X + Ki*(1+S/Km)\nY=Vo*(1 - (b - sqrt(b^2 - 4*Et*X))/(2*Et))",
    "dr_5pl_log": "logXb=LogEC50 + log10(2^(1/S)-1)/HillSlope\nY=Bottom + (Top-Bottom)/(1+10^((logXb-X)*HillSlope))^S",
    "dr_5pl_dose": "Y=Bottom + (Top-Bottom)/(1+(2^(1/S)-1)*(EC50/X)^HillSlope)^S",
    "dr_biphasic_log": ("Y=Bottom + (Top-Bottom)*(Frac/(1+10^((LogEC50_1-X)*nH1)) + (1-Frac)/(1+10^((LogEC50_2-X)*nH2)))"),
    "dr_biphasic_dose": "Y=Bottom + (Top-Bottom)*(Frac/(1+(EC50_1/X)^nH1) + (1-Frac)/(1+(EC50_2/X)^nH2))",
    "dr_bell_log": "Y=Dip + (Plateau1-Dip)/(1+10^((LogEC50_1-X)*nH1)) + (Plateau2-Dip)/(1+10^((X-LogEC50_2)*nH2))",
    "dr_bell_dose": "Y=Dip + (Plateau1-Dip)/(1+(EC50_1/X)^nH1) + (Plateau2-Dip)/(1+(X/EC50_2)^nH2)",
    "dr_operational_log": "operate=((10^logKA + 10^X)/10^(logTau + X))^n\nY=Basal + (Effectmax-Basal)/(1+operate)",
    "dr_operational_global": ("Data set A (full agonist): Y=Basal + (Effectmax-Basal)/(1+10^((LogEC50-X)*n))\n"
                             "Other data sets (partial agonists): operate=((10^logKA + 10^X)/10^(logTau + X))^n\n"
                             "Y=Basal + (Effectmax-Basal)/(1+operate)"),
    "dr_operational_dose": "Y=Basal + (Effectmax-Basal)/(1+((KA+X)/(Tau*X))^n)",
    "dr_schild_log": _SCHILD + "\nY=Bottom + (Top-Bottom)/(1+10^((LogEC50 + log10(antag) - X)*HillSlope))",
    "dr_schild_dose": _SCHILD + "\nY=Bottom + (Top-Bottom)/(1+(EC50*antag/X)^HillSlope)",
    "dr_schild_pa2_log": "antag = 1 + B/10^(-pA2)\nY=Bottom + (Top-Bottom)/(1+10^((LogEC50 + log10(antag) - X)*HillSlope))",
    "dr_schild_pa2_dose": "antag = 1 + B/10^(-pA2)\nY=Bottom + (Top-Bottom)/(1+(EC50*antag/X)^HillSlope)",
    "dr_allosteric_log": ("antag = (1 + B/10^LogKB)/(1 + 10^LogAlpha*B/10^LogKB)\n"
                          "Y=Bottom + (Top-Bottom)/(1+10^((LogEC50 + log10(antag) - X)*HillSlope))"),
    "dr_allosteric_dose": "antag = (1 + B/KB)/(1 + Alpha*B/KB)\nY=Bottom + (Top-Bottom)/(1+(EC50*antag/X)^HillSlope)",
    "dr_ecanything_log": "logEC50=LogECF - log10(F/(100-F))/HillSlope\nY=Bottom + (Top-Bottom)/(1+10^((logEC50-X)*HillSlope))",
    "dr_ecanything_dose": "EC50=ECF/(F/(100-F))^(1/HillSlope)\nY=Bottom + (Top-Bottom)/(1+(EC50/X)^HillSlope)",
    "gen_exp_growth": "Y=Y0*exp(k*X)",
    "gen_decay1": "Y=(Y0-Plateau)*exp(-K*X) + Plateau",
    "gen_assoc1": "Y=Y0 + (Plateau-Y0)*(1-exp(-K*X))",
    "gen_decay2": ("Span=Y0-Plateau\nY=Plateau + Span*PercentFast*0.01*exp(-KFast*X) + "
                   "Span*(100-PercentFast)*0.01*exp(-KSlow*X)"),
    "gen_assoc2": ("Span=Plateau-Y0\nY=Y0 + Span*PercentFast*0.01*(1-exp(-KFast*X)) + "
                   "Span*(100-PercentFast)*0.01*(1-exp(-KSlow*X))"),
    "gen_plateau_decay": "Y=Y0 for X<X0\nY=Plateau + (Y0-Plateau)*exp(-K*(X-X0)) for X>=X0",
    "gen_poly1": "Y=B0 + B1*X",
    "gen_poly2": "Y=B0 + B1*X + B2*X^2",
    "gen_poly3": "Y=B0 + B1*X + B2*X^2 + B3*X^3",
    "gen_poly4": _POLY,
    "gen_gaussian": "Y=Amplitude*exp(-0.5*((X-Mean)/SD)^2)",
    "gen_lorentzian": "Y=Amplitude/(1+((X-Center)/Width)^2)",
    "gen_lognormal": "Y=A/X*exp(-0.5*(ln(X/GeoMean)/ln(GeoSD))^2)",
    "gen_sine": "Y=Amplitude*sin(2*pi*X/Wavelength + PhaseShift)",
    "gen_sine_baseline": "Y=Baseline + Amplitude*sin(2*pi*X/Wavelength + PhaseShift)",
    "gen_sine_damped": "Y=Amplitude*exp(-K*X)*sin(2*pi*X/Wavelength + PhaseShift)",
    "gen_gompertz": "Y=YM*(Y0/YM)^exp(-K*X)",
    "gen_logistic": "Y=YM*Y0/((YM-Y0)*exp(-k*X) + Y0)",
    "gen_beta": "Y=Ym*(1 + (Te-X)/(Te-Tm))*(X/Te)^(Te/(Te-Tm))",
    "gen_lq_survival": "Y=" + _LQ,
    "gen_lq_survival_pct": "Y=100*" + _LQ,
    "gen_lq_survival_y0": "Y=Y0*" + _LQ,
    "gen_lq_killed": "Y=1 - " + _LQ,
}

X_SCALE_TEXT = {"log": "X is log10 of the concentration", "dose": "X is the concentration (not logged)",
                "linear": "X is used as entered"}
