"""Vector SVG visualizer for 2D chemical structures, DNA constructs, and genetic cassettes."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

from helixmind.config import get_settings
from helixmind.insilico.chemoinformatics import ChemoinformaticsEngine


class BioVisualizer:
    """Renders 2D molecular structures and DNA cassettes into publication-grade SVG graphics."""

    @classmethod
    def get_figures_dir(cls) -> Path:
        settings = get_settings()
        d = settings.data_dir / "figures"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @classmethod
    def render_molecule_svg(
        cls,
        name_or_smiles: str,
        width: int = 360,
        height: int = 280,
    ) -> Optional[str]:
        """Render a chemical molecule into a vector SVG using RDKit."""
        smiles = ChemoinformaticsEngine.resolve_smiles(name_or_smiles)
        if not smiles:
            return None

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
        opts = drawer.drawOptions()
        opts.clearBackground = False
        opts.bondLineWidth = 2.0
        opts.comicMode = False

        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        return drawer.GetDrawingText()

    @classmethod
    def render_dna_construct_svg(
        cls,
        construct_name: str,
        components: Optional[List[Dict[str, str]]] = None,
        width: int = 760,
        height: int = 140,
    ) -> str:
        """Render a publication-grade genetic cassette / plasmid construct in vector SVG."""
        default_components = components or [
            {"label": "hU6 Promoter", "type": "promoter", "color": "#0284c7"},
            {"label": "20-nt sgRNA", "type": "sgrna", "color": "#f59e0b"},
            {"label": "Cas9 Scaffold", "type": "scaffold", "color": "#8b5cf6"},
            {"label": "Poly-T Term", "type": "terminator", "color": "#ef4444"},
            {"label": "EF1a Promoter", "type": "promoter", "color": "#0284c7"},
            {"label": "EGFP / PuroR", "type": "marker", "color": "#10b981"},
        ]

        svg_parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" height="{height}" style="background:#090d16; border-radius:8px;">',
            f'  <text x="20" y="24" fill="#94a3b8" font-family="sans-serif" font-size="11" font-weight="700">GENETIC CONSTRUCT: {construct_name.upper()}</text>',
            f'  <!-- Backbone DNA Double Strand Line -->',
            f'  <line x1="20" y1="75" x2="{width - 20}" y2="75" stroke="#334155" stroke-width="4" stroke-linecap="round"/>',
        ]

        total_parts = len(default_components)
        spacing = (width - 60) / total_parts
        cur_x = 30

        for i, comp in enumerate(default_components):
            c_width = spacing - 8
            label = comp.get("label", f"Part {i}")
            color = comp.get("color", "#0284c7")
            ctype = comp.get("type", "feature")

            # Render feature block or bent promoter arrow
            if ctype == "promoter":
                svg_parts.extend([
                    f'  <!-- Promoter Arrow -->',
                    f'  <path d="M {cur_x} 75 L {cur_x} 48 L {cur_x + c_width - 8} 48 L {cur_x + c_width - 8} 40 L {cur_x + c_width} 52 L {cur_x + c_width - 8} 64 L {cur_x + c_width - 8} 56 L {cur_x} 56 Z" fill="{color}" stroke="#000" stroke-width="1"/>',
                ])
            else:
                svg_parts.extend([
                    f'  <!-- Feature Box -->',
                    f'  <rect x="{cur_x}" y="55" width="{c_width}" height="40" rx="6" fill="{color}" stroke="#000" stroke-width="1"/>',
                ])

            # Label
            text_x = cur_x + (c_width / 2)
            svg_parts.append(
                f'  <text x="{text_x}" y="115" fill="#f8fafc" font-family="sans-serif" font-size="10" font-weight="600" text-anchor="middle">{label}</text>'
            )
            cur_x += spacing

        svg_parts.append("</svg>")
        return "\n".join(svg_parts)

    @classmethod
    def save_figure(cls, svg_content: str, filename: str) -> Path:
        """Persist SVG figure to disk."""
        path = cls.get_figures_dir() / f"{filename}.svg"
        with open(path, "w", encoding="utf-8") as f:
            f.write(svg_content)
        return path
