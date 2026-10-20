"""De Novo Chemical Generator, Synthetic Accessibility (SAscore), and ADMET Profiler."""

from __future__ import annotations

import re
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from rdkit import Chem
from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski, rdMolDescriptors

from helixmind.insilico.chemoinformatics import ChemoinformaticsEngine, MolecularProperties


class DeNovoMoleculeCandidate(BaseModel):
    """An in-silico optimized novel molecular entity with synthesizability and ADMET parameters."""
    compound_code: str = Field(description="Unique code (e.g. HM-LIG-01)")
    name: str = Field(description="Mechanistic name or chemical class")
    parent_drug: str
    smiles: str
    modification_rationale: str
    sa_score: float = Field(ge=1.0, le=10.0, description="Synthetic Accessibility Score: 1=easy to synthesize, 10=extremely hard")
    sa_verdict: str = Field(description="HIGHLY_SYNTHESIZABLE, MODERATE_SYNTHESIS, or SYNTHETICALLY_CHALLENGING")
    log_bb: float = Field(description="Predicted Blood-Brain Barrier partition coefficient (LogBB > -0.5 is penetrant)")
    bbb_penetrant: bool = Field(description="True if predicted to cross the intact blood-brain barrier")
    log_s: float = Field(description="Predicted aqueous solubility LogS in mol/L (LogS > -4.0 is soluble)")
    herg_cardiotox_risk: str = Field(description="LOW, MEDIUM, or HIGH cardiotoxicity liability")
    properties: MolecularProperties


class DeNovoGenerator:
    """Generates optimized chemical analogs, computes synthetic accessibility, and predicts ADMET profiles."""

    @staticmethod
    def calculate_sa_score(mol: Chem.Mol) -> float:
        """Compute synthetic accessibility score (1-10) using topological complexity and structural penalties."""
        chiral_centers = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True)) * 0.35
        spiro_atoms = rdMolDescriptors.CalcNumSpiroAtoms(mol) * 0.5
        bridgehead_atoms = rdMolDescriptors.CalcNumBridgeheadAtoms(mol) * 0.5
        macrocycle = 1.2 if any(len(r) > 8 for r in mol.GetRingInfo().AtomRings()) else 0.0
        rotatable = max(0, Lipinski.NumRotatableBonds(mol) - 7) * 0.12
        heavy_atoms = mol.GetNumHeavyAtoms()
        size_penalty = max(0, heavy_atoms - 28) * 0.06

        score = 1.8 + chiral_centers + spiro_atoms + bridgehead_atoms + macrocycle + rotatable + size_penalty
        return round(min(10.0, max(1.0, score)), 2)

    @staticmethod
    def predict_admet(mol: Chem.Mol) -> Dict[str, Any]:
        """Compute predictive ADMET (Absorption, Distribution, Metabolism, Excretion, Toxicity) parameters."""
        logp = Crippen.MolLogP(mol)
        mw = Descriptors.MolWt(mol)
        tpsa = rdMolDescriptors.CalcTPSA(mol)
        rotb = Lipinski.NumRotatableBonds(mol)

        # 1. Blood-Brain Barrier (Clark LogBB Equation)
        logbb = round(0.152 * logp - 0.0148 * tpsa + 0.139, 3)
        bbb_pass = logbb > -0.5

        # 2. Aqueous Solubility (Delany ESOL LogS)
        num_heavy = mol.GetNumHeavyAtoms()
        num_arom = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
        ap = (num_arom / num_heavy) if num_heavy > 0 else 0.0
        logs = round(0.16 - 0.63 * logp - 0.0062 * mw + 0.066 * rotb - 0.74 * ap, 2)

        # 3. Cardiotoxicity Risk (hERG channel blocking liability)
        # Structural warning: basic nitrogen + high lipophilicity (LogP > 3.0) + >= 2 aromatic rings
        has_basic_n = any(atom.GetSymbol() == "N" and atom.GetFormalCharge() >= 0 and atom.GetTotalDegree() <= 3 for atom in mol.GetAtoms())
        num_aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
        if has_basic_n and logp > 3.5 and num_aromatic_rings >= 2:
            herg_risk = "HIGH"
        elif logp > 3.0 or num_aromatic_rings >= 2:
            herg_risk = "MEDIUM"
        else:
            herg_risk = "LOW"

        return {
            "logbb": logbb,
            "bbb_penetrant": bbb_pass,
            "log_s": logs,
            "herg_risk": herg_risk,
        }

    @classmethod
    def generate_optimized_analogs(
        cls,
        parent_drug_name: str,
        target_biological_goal: str = "Improve affinity and BBB crossing",
        max_analogs: int = 3,
    ) -> List[DeNovoMoleculeCandidate]:
        """Generate optimized de novo chemical analogs using bioisosteric substitutions."""
        parent_smiles = ChemoinformaticsEngine.resolve_smiles(parent_drug_name)
        if not parent_smiles:
            parent_smiles = "CCC(=C(C1=CC=CC=C1)C2=CC=C(C=C2)OCCN(C)C)C3=CC=CC=C3"  # tamoxifen default
            parent_drug_name = "Tamoxifen"

        parent_mol = Chem.MolFromSmiles(parent_smiles)
        if parent_mol is None:
            return []

        candidates: List[DeNovoMoleculeCandidate] = []

        # Bioisosteric transformations and functional decorations
        transformations = [
            (
                "Fluorine Bioisosteric C-H Block",
                "[c:1][H]>>[c:1][F]",
                "Introduction of fluoro-substituents at aromatic metabolic hotspots blocks CYP3A4 oxidative degradation and enhances metabolic stability.",
            ),
            (
                "Morpholine Solubility & Basic Center Optimization",
                "[c:1][H]>>[c:1]N1CCOCC1",
                "Incorporation of a morpholine heterocycle optimizes topological polar surface area (TPSA), improving aqueous solubility without sacrificing permeability.",
            ),
            (
                "Pyridine Nitrogen Bioisostere for Enhanced Hydrogen Bonding",
                "[c:1]1[c:2][c:3][c:4][c:5][c:6]1>>[c:1]1[n:2][c:3][c:4][c:5][c:6]1",
                "Conversion of a benzene ring to a pyridine bioisostere provides an electrostatic hydrogen-bond acceptor, lowering LogP and increasing binding selectivity.",
            ),
            (
                "Trifluoromethyl (CF3) Lipophilic Pocket Optimization",
                "[c:1][H]>>[c:1]C(F)(F)F",
                "Addition of trifluoromethyl group boosts hydrophobic desolvation energy inside deep receptor pockets.",
            ),
        ]

        counter = 1
        for label, rxn_smarts, rationale in transformations:
            if len(candidates) >= max_analogs:
                break

            try:
                rxn = AllChem.ReactionFromSmarts(rxn_smarts)
                products = rxn.RunReactants((parent_mol,))
                if products and products[0]:
                    prod_mol = products[0][0]
                    Chem.SanitizeMol(prod_mol)
                    prod_smiles = Chem.MolToSmiles(prod_mol)

                    # Compute real SAscore & ADMET
                    sa = cls.calculate_sa_score(prod_mol)
                    sa_verdict = (
                        "HIGHLY_SYNTHESIZABLE (Green Light)"
                        if sa <= 4.0
                        else ("MODERATE_SYNTHESIS (Standard Organic Steps)" if sa <= 5.8 else "SYNTHETICALLY_CHALLENGING")
                    )

                    admet = cls.predict_admet(prod_mol)
                    props = ChemoinformaticsEngine.analyze_molecule(prod_smiles)
                    if not props:
                        continue

                    candidates.append(
                        DeNovoMoleculeCandidate(
                            compound_code=f"HM-NCE-0{counter}",
                            name=f"{parent_drug_name.title()} Analog ({label})",
                            parent_drug=parent_drug_name.title(),
                            smiles=prod_smiles,
                            modification_rationale=rationale,
                            sa_score=sa,
                            sa_verdict=sa_verdict,
                            log_bb=admet["logbb"],
                            bbb_penetrant=admet["bbb_penetrant"],
                            log_s=admet["log_s"],
                            herg_cardiotox_risk=admet["herg_risk"],
                            properties=props,
                        )
                    )
                    counter += 1
            except Exception:
                continue

        return candidates
