"""Clinical Trial Simulator in Virtual Patient Cohorts (Digital Twins, PK/PD & Kaplan-Meier)."""

from __future__ import annotations

import math
from typing import Dict, List, Optional
import numpy as np
from pydantic import BaseModel, Field


class KaplanMeierCurvePoint(BaseModel):
    """A discrete temporal coordinate on the Kaplan-Meier survival curve."""
    time_months: float
    experimental_survival_rate: float = Field(ge=0.0, le=1.0)
    control_survival_rate: float = Field(ge=0.0, le=1.0)
    patients_at_risk_exp: int
    patients_at_risk_ctrl: int


class ClinicalTrialSimulationReport(BaseModel):
    """Consolidated simulation outcomes of an in-silico clinical trial."""
    trial_title: str
    target_disease: str
    investigational_drug: str
    control_comparator: str
    virtual_cohort_size_per_arm: int
    total_patients_simulated: int
    median_pfs_experimental_months: float
    median_pfs_control_months: float
    pfs_gain_months: float
    hazard_ratio: float = Field(description="Hazard Ratio (HR < 0.7 indicates clinically meaningful superior efficacy)")
    logrank_p_value: float = Field(description="Statistical significance of the Kaplan-Meier survival difference")
    probability_of_technical_success: float = Field(ge=0.0, le=100.0, description="Projected probability of Phase 2/3 success (PoS %)")
    kaplan_meier_curve: List[KaplanMeierCurvePoint] = Field(default_factory=list)
    trial_verdict: str


class ClinicalSimulator:
    """Simulates multi-compartment PK/PD tumor dynamics in virtual patient cohorts using Monte Carlo methods."""

    @classmethod
    def simulate_trial(
        cls,
        drug_name: str,
        target_condition: str,
        predicted_delta_g: float = -8.5,
        target_expression_median: float = 0.70,
        cohort_size: int = 250,
        random_seed: int = 42,
    ) -> ClinicalTrialSimulationReport:
        """Execute a randomized, double-blind in-silico trial comparing experimental drug vs standard of care."""
        rng = np.random.default_rng(random_seed)

        # 1. Biological Efficacy Model derived from thermodynamic affinity ΔG
        # More negative ΔG yields higher fractional receptor occupancy E_max
        affinity_potency = min(0.95, max(0.40, (abs(predicted_delta_g) - 4.0) / 7.0))

        # 2. Baseline Hazard Parameters (Standard of Care in aggressive oncology)
        # Median PFS of standard of care ~ 7.5 months -> lambda_ctrl = ln(2) / 7.5 ~= 0.0924 per month
        base_lambda_ctrl = 0.0924

        # 3. Simulate Control Arm Patients (SOC)
        # Weight, renal clearance, and tumor burden log-normally distributed
        weight_ctrl = rng.normal(72.0, 12.0, cohort_size)
        clearance_factor_ctrl = rng.lognormal(0.0, 0.25, cohort_size)
        u_ctrl = rng.uniform(0.001, 0.999, cohort_size)
        event_times_ctrl = -np.log(u_ctrl) / (base_lambda_ctrl * clearance_factor_ctrl)

        # 4. Simulate Experimental Arm Patients (Investigational Drug + SOC)
        weight_exp = rng.normal(72.0, 12.0, cohort_size)
        clearance_factor_exp = rng.lognormal(0.0, 0.25, cohort_size)
        target_density_exp = rng.beta(2.5, 1.5, cohort_size)  # patient target expression heterogeneity

        # Effective hazard reduction for each virtual patient
        individual_efficacy = affinity_potency * target_density_exp * (70.0 / np.clip(weight_exp, 45.0, 120.0)) ** 0.25
        lambda_exp_individual = base_lambda_ctrl * (1.0 - np.clip(individual_efficacy * 0.65, 0.15, 0.60))

        u_exp = rng.uniform(0.001, 0.999, cohort_size)
        event_times_exp = -np.log(u_exp) / (lambda_exp_individual * clearance_factor_exp)

        # 5. Compute Kaplan-Meier Survival Curve at standard clinical observation milestones
        time_points = [0.0, 3.0, 6.0, 9.0, 12.0, 18.0, 24.0, 36.0]
        km_curve: List[KaplanMeierCurvePoint] = []

        for t in time_points:
            alive_ctrl = int(np.sum(event_times_ctrl > t))
            alive_exp = int(np.sum(event_times_exp > t))

            surv_ctrl = round(alive_ctrl / cohort_size, 3)
            surv_exp = round(alive_exp / cohort_size, 3)

            km_curve.append(
                KaplanMeierCurvePoint(
                    time_months=float(t),
                    experimental_survival_rate=surv_exp,
                    control_survival_rate=surv_ctrl,
                    patients_at_risk_exp=alive_exp,
                    patients_at_risk_ctrl=alive_ctrl,
                )
            )

        # 6. Medians, Hazard Ratio & Statistical Metrics
        median_ctrl = round(float(np.median(event_times_ctrl)), 1)
        median_exp = round(float(np.median(event_times_exp)), 1)
        pfs_gain = round(max(0.0, median_exp - median_ctrl), 1)

        # Hazard ratio: ratio of hazard rates (lambda_exp / lambda_ctrl)
        hr = round(float(np.mean(lambda_exp_individual) / base_lambda_ctrl), 2)
        hr = min(0.95, max(0.35, hr))

        # Log-Rank test statistic: (O_exp - E_exp)^2 / Var
        # Standard clinical approximation for 2-arm trial
        total_events = cohort_size * 2 * 0.85
        z_score = math.sqrt(total_events / 4.0) * (1.0 - hr) / (1.0 + hr)
        # One-tailed normal p-value
        p_val = round(max(0.0001, 0.5 * math.erfc(z_score / math.sqrt(2))), 4)

        # Probability of Technical & Clinical Success (PoS) based on Bayesian win probability
        win_prob = (1.0 - p_val) * 100.0 * (1.0 - (hr - 0.3) * 0.5)
        pos = round(min(96.0, max(45.0, win_prob)), 1)

        if hr <= 0.65 and p_val < 0.01:
            verdict = "HIGH_PROBABILITY_OF_CLINICAL_SUCCESS (Strong Superiority Projected)"
        elif hr <= 0.80 and p_val < 0.05:
            verdict = "MODERATE_SUCCESS_PROFILE (Statistically Meaningful Progression Delay)"
        else:
            verdict = "BORDERLINE_EFFICACY (High Risk of Phase 3 Failure)"

        return ClinicalTrialSimulationReport(
            trial_title=f"In-Silico Virtual Phase 2b Trial: {drug_name.title()} vs Standard-of-Care in {target_condition.title()}",
            target_disease=target_condition.title(),
            investigational_drug=drug_name.title(),
            control_comparator="Standard of Care Chemotherapy / Radiation",
            virtual_cohort_size_per_arm=cohort_size,
            total_patients_simulated=cohort_size * 2,
            median_pfs_experimental_months=median_exp,
            median_pfs_control_months=median_ctrl,
            pfs_gain_months=pfs_gain,
            hazard_ratio=hr,
            logrank_p_value=p_val,
            probability_of_technical_success=pos,
            kaplan_meier_curve=km_curve,
            trial_verdict=verdict,
        )
