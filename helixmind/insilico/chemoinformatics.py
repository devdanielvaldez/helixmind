"""Chemoinformatics and molecular property evaluation using RDKit."""

from __future__ import annotations

import math
from typing import Dict, Optional
from pydantic import BaseModel, Field
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, Crippen, rdMolDescriptors
from rdkit.Chem import QED

# Curated canonical SMILES for frequent anticancer, immunotherapeutic and experimental molecules
KNOWN_DRUG_SMILES: Dict[str, str] = {
    "tamoxifen": "CCC(=C(C1=CC=CC=C1)C2=CC=C(C=C2)OCCN(C)C)C3=CC=CC=C3",
    "afatinib": "CN(C)C/C=C/C(=O)NC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC(=C(C=C3)F)Cl)O[C@H]4CCOC4",
    "temozolomide": "CN1C(=O)N2C=NC(=C2N=N1)C(=O)N",
    "cisplatin": "N.N.[Cl-].[Cl-].[Pt+2]",
    "doxorubicin": "CC1C(C(CC(O1)OC2CC(CC3=C(C4=C(C(=C32)O)C(=O)C5=C(C4=O)C=CC=C5OC)O)(C(=O)CO)O)N)O",
    "paclitaxel": "CC1=C2C(C(=O)C3(C(CC4C(C3C(C(C2(C)C)(CC1OC(=O)C(C(C5=CC=CC=C5)NC(=O)C6=CC=CC=C6)O)O)OC(=O)C7=CC=CC=C7)(CO4)OC(=O)C)O)C)OC(=O)C",
    "palbociclib": "CC(=O)C1=C(NC2=NC(=NC=C12)NC3=NC=C(C=C3)N4CCNCC4)C5=CC(=O)N(C=C5)C6CCCC6",
    "erlotinib": "COCCOC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC=CC(=C3)C#C)OCCOC",
    "gefitinib": "COC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC(=C(C=C3)F)Cl)OCCCN4CCOCC4",
    "imatinib": "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5",
    "aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
    "selenium_nanoparticle": "[Se]",
    "curcumin": "COC1=C(C=CC(=C1)/C=C/C(=O)CC(=O)/C=C/C2=CC(=C(C=C2)O)OC)O",
    "resveratrol": "C1=CC(=CC=C1/C=C/C2=CC(=CC(=C2)O)O)O",
    "quercetin": "C1=CC(=C(C=C1C2=C(C(=O)C3=C(C=C(C=C3O2)O)O)O)O)O",
}


class MolecularProperties(BaseModel):
    """Calculated physicochemical properties and Lipinski Rule of 5 evaluation."""
    name: str
    smiles: str
    molecular_weight: float = Field(description="Molecular weight in g/mol (Lipinski ideal <= 500)")
    logp: float = Field(description="Octanol-water partition coefficient LogP (Lipinski ideal <= 5.0)")
    hbd: int = Field(description="Number of Hydrogen Bond Donors (Lipinski ideal <= 5)")
    hba: int = Field(description="Number of Hydrogen Bond Acceptors (Lipinski ideal <= 10)")
    tpsa: float = Field(description="Topological Polar Surface Area in Å² (permeability ideal <= 140)")
    rotatable_bonds: int = Field(description="Number of rotatable bonds (flexibility ideal <= 10)")
    lipinski_violations: int = Field(description="Number of Lipinski Ro5 violations (0 to 4)")
    lipinski_pass: bool = Field(description="True if violations <= 1 (drug-like oral bioavailability)")
    qed_score: float = Field(description="Quantitative Estimate of Druglikeness score (0.0 to 1.0)")
    chemical_formula: str = Field(description="Hill notation chemical formula (e.g. C26H29NO)")


class ChemoinformaticsEngine:
    """Evaluates drug compounds, chemical inhibitors, and small molecules with RDKit."""

    @staticmethod
    def resolve_smiles(name_or_smiles: str) -> Optional[str]:
        """Resolve a drug name to a canonical SMILES string or validate raw SMILES."""
        clean = name_or_smiles.strip().lower()
        if clean in KNOWN_DRUG_SMILES:
            return KNOWN_DRUG_SMILES[clean]

        # Check if the input is already a valid SMILES string
        mol = Chem.MolFromSmiles(name_or_smiles.strip())
        if mol is not None:
            return Chem.MolToSmiles(mol)

        # Dynamic online resolution via NCBI PubChem PUG REST API
        try:
            import urllib.parse
            import httpx
            encoded_name = urllib.parse.quote(name_or_smiles.strip())
            url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{encoded_name}/property/ConnectivitySMILES,IsomericSMILES,CanonicalSMILES/JSON"
            with httpx.Client(timeout=6.0) as client:
                res = client.get(url)
                if res.status_code == 200:
                    props = res.json().get("PropertyTable", {}).get("Properties", [])
                    if props:
                        smi = props[0].get("IsomericSMILES") or props[0].get("ConnectivitySMILES") or props[0].get("CanonicalSMILES")
                        if smi:
                            KNOWN_DRUG_SMILES[clean] = smi
                            return smi
        except Exception:
            pass

        # Partial matching in local cache
        for k, smi in KNOWN_DRUG_SMILES.items():
            if k in clean or clean in k:
                return smi

        return None

    @classmethod
    def analyze_molecule(cls, name_or_smiles: str) -> Optional[MolecularProperties]:
        """Compute complete Lipinski Rule of 5 and ADMET descriptors."""
        smiles = cls.resolve_smiles(name_or_smiles)
        if not smiles:
            return None

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        mw = round(float(Descriptors.MolWt(mol)), 2)
        logp = round(float(Crippen.MolLogP(mol)), 2)
        hbd = int(Lipinski.NumHDonors(mol))
        hba = int(Lipinski.NumHAcceptors(mol))
        tpsa = round(float(rdMolDescriptors.CalcTPSA(mol)), 2)
        rot_bonds = int(Lipinski.NumRotatableBonds(mol))

        # Lipinski Rule of 5 violations
        violations = 0
        if mw > 500:
            violations += 1
        if logp > 5.0:
            violations += 1
        if hbd > 5:
            violations += 1
        if hba > 10:
            violations += 1

        lipinski_pass = violations <= 1

        try:
            qed_score = round(float(QED.qed(mol)), 3)
        except Exception:
            qed_score = 0.5

        formula = rdMolDescriptors.CalcMolFormula(mol)
        display_name = name_or_smiles.strip().title()

        return MolecularProperties(
            name=display_name,
            smiles=smiles,
            molecular_weight=mw,
            logp=logp,
            hbd=hbd,
            hba=hba,
            tpsa=tpsa,
            rotatable_bonds=rot_bonds,
            lipinski_violations=violations,
            lipinski_pass=lipinski_pass,
            qed_score=qed_score,
            chemical_formula=formula,
        )
