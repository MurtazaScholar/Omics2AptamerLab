
# ============================================================
# Aptamer pipeline v8 (Google Colab single-script)
#
# Purpose
# -------
# Train a protein->aptamer interaction model from PDB complexes and predict
# high-specificity aptamer candidates plus their secondary structures for an
# input protein sequence or motif/domain sequence.
#
# Major design choices / justification
# ------------------------------------
# 1) 12 Å extraction window:
#    We extract all motif contacts up to 12 Å because nucleic-acid/protein
#    recognition often includes direct contacts plus second-shell electrostatic
#    and stacking effects. We still keep the exact minimum distance and a
#    distance-derived strength score, so close contacts remain more influential
#    than weak peripheral contacts.
#
# 2) Symmetric motif modeling on both sides:
#    We build interaction data for nucleotide motifs of length 1/2/3 against
#    amino-acid motifs of length 1/2/3:
#      nt1-aa1, nt1-aa2, nt1-aa3
#      nt2-aa1, nt2-aa2, nt2-aa3
#      nt3-aa1, nt3-aa2, nt3-aa3
#    This directly answers requests such as "single nucleotide to double a.a"
#    and "vice versa". It captures point contacts, local protein motifs, and
#    local aptamer context, all of which are important for specificity.
#
# 3) Leakage-safe evaluation:
#    The holdout split is done by PDB id before model tuning. Cross-validation
#    also groups by PDB id, reducing leakage from repeated rows extracted from
#    the same complex.
#
# 4) Sequence prediction strategy:
#    The ML model predicts interaction probability/score for motif pairs.
#    Aptamer sequences are then generated with an in-silico SELEX / genetic
#    search that optimizes the aggregate interaction score against the input
#    protein and folds each candidate with RNAfold to obtain secondary
#    structure.
#
# Outputs
# -------
# - interaction_data.csv
# - dna_sequences.csv / protein_sequences.csv
# - one example-PDB interaction table + summary
# - specificity long tables + matrices for all 9 nt/aa motif combinations
# - performance_summary.csv + heldout_report.txt + ROC/PR/confusion plots
# - feature importance plot
# - held-out sequence retrieval evaluation
# - predicted top-50 aptamers with binding motifs and RNAfold structures
# - SELEX round summaries and plots
# - final zip bundle
# ============================================================

import os
import gc
import re
import math
import json
import sys
import argparse
import zipfile
import shutil
import importlib
import subprocess
import warnings
from collections import defaultdict, Counter

# ------------------------------------------------------------
# Dependency bootstrap (Colab-friendly)
# ------------------------------------------------------------

def ensure_python_package(import_name: str, pip_name: str = None):
    pip_name = pip_name or import_name
    try:
        return importlib.import_module(import_name)
    except ImportError:
        print(f"[SETUP] Installing {pip_name} ...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pip_name], check=True)
        return importlib.import_module(import_name)

ensure_python_package("Bio", "biopython")
ensure_python_package("catboost", "catboost")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from Bio.PDB import PDBParser
from catboost import CatBoostClassifier
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay
from sklearn.isotonic import IsotonicRegression

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ------------------------------------------------------------
# Plot style
# ------------------------------------------------------------
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 15,
    "axes.labelsize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.figsize": (10, 8),
    "figure.dpi": 180,
    "savefig.dpi": 250,
    "savefig.format": "tiff",
    "savefig.bbox": "tight",
})

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
SEED = 42
rng = np.random.default_rng(SEED)
np.random.seed(SEED)

PDB_DIR = os.path.join(os.getcwd(), "PDB_files")
OUTPUT_DIR = os.path.join(os.getcwd(), "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

PDB_EXTS = (".pdb", ".ent")
PDB_SEARCH_DEPTH = 10

# Contact extraction
CONTACT_CUTOFF = 12.0          # requested extraction window
NEGATIVE_MIN_DIST = 16.0       # non-contact negatives
NEGATIVE_MAX_DIST = 28.0
NEGATIVE_RATIO = 1.2           # sampled negatives per positive
MAX_RESIDUES_PER_CHAIN = 1500

# Modeling / cleaning
TEST_SIZE = 0.20
GLOBAL_AMBIG_LOW = 0.45
GLOBAL_AMBIG_HIGH = 0.55
GLOBAL_AMBIG_MIN_SUPPORT = 8
MAX_TRAIN_ROWS = 900000

# RNAfold
USE_RNAFOLD = True
AUTO_INSTALL_VIENNARNA = False

# Prediction / SELEX
NUM_APTAMERS = 50
MAX_RESIDUES_TO_SCORE = 350
SELEX_POOL_SIZE = 1800
SELEX_ROUNDS = 7
SELEX_KEEP_FRAC = 0.12
SELEX_ELITE_FRAC = 0.03
SELEX_OFFSPRING_PER_PARENT = 10
SELEX_MUT_RATE = 0.07
SELEX_MIN_MUT_RATE = 0.015
SELEX_INDEL_RATE = 0.08
SELEX_CROSSOVER_FRAC = 0.28
SELEX_EXPLORATION_FRAC = 0.12
SELEX_RANDOM_INJECTION_FRAC = 0.08
SELEX_SEED_FROM_TRAIN_FRAC = 0.35
SELEX_LEN_MIN = 24
SELEX_LEN_MAX = 34
DIVERSITY_K = 3
DIVERSITY_MAX_JACCARD = 0.68
ENSEMBLE_UNCERTAINTY_PENALTY = 0.10
NEAR_DUPLICATE_TO_TRAIN_JACCARD = 0.92

# ------------------------------------------------------------
# Mappings and physicochemical features
# ------------------------------------------------------------
AA3_TO_1 = {
    "ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLU":"E","GLN":"Q","GLY":"G",
    "HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P","SER":"S",
    "THR":"T","TRP":"W","TYR":"Y","VAL":"V","MSE":"M","SEC":"C","PYL":"K"
}
DNA_RNA_BASE = {
    "DA":"A","DC":"C","DG":"G","DT":"T","DU":"T",
    "A":"A","C":"C","G":"G","U":"T","T":"T",
    "RA":"A","RC":"C","RG":"G","RU":"T"
}
VALID_AA_SET = set(list("ACDEFGHIKLMNPQRSTVWY"))
NUCLEOTIDES = np.array(["A","C","G","T"], dtype=object)

AA_CHARGE = {"D": -1.0, "E": -1.0, "K": 1.0, "R": 1.0, "H": 0.5}
AA_HYDRO = {
    "A": 1.8,  "C": 2.5,  "D": -3.5, "E": -3.5, "F": 2.8,
    "G": -0.4, "H": -3.2, "I": 4.5,  "K": -3.9, "L": 3.8,
    "M": 1.9,  "N": -3.5, "P": -1.6, "Q": -3.5, "R": -4.5,
    "S": -0.8, "T": -0.7, "V": 4.2,  "W": -0.9, "Y": -1.3
}
AA_AROMATIC = set(["F", "W", "Y", "H"])
PURINES = set(["A", "G"])

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def discover_pdb_files():
    if not os.path.isdir(PDB_DIR):
        raise FileNotFoundError(f"PDB_DIR not found: {PDB_DIR}")
    pdbs = []
    for root, dirs, files in os.walk(PDB_DIR):
        depth = root[len(PDB_DIR):].count(os.sep)
        if depth > PDB_SEARCH_DEPTH:
            dirs[:] = []
            continue
        for fn in files:
            if fn.lower().endswith(PDB_EXTS):
                pdbs.append(os.path.join(root, fn))
    return sorted(list(set(pdbs)))

def atom_to_tuple(atom):
    return (
        atom.get_name().strip(),
        float(atom.coord[0]),
        float(atom.coord[1]),
        float(atom.coord[2]),
        str(getattr(atom, "element", "") or "").upper(),
    )

def heavy_atom_coords(atom_list):
    coords = []
    for atom_name, x, y, z, element in atom_list:
        if atom_name.startswith("H") or element == "H":
            continue
        coords.append([x, y, z])
    if not coords:
        coords = [[a[1], a[2], a[3]] for a in atom_list]
    return np.array(coords, dtype=float)

def rep_coord(atom_list, prefer=("CA",), fallback=("C", "N", "O")):
    atom_map = {a[0]: a for a in atom_list}
    for name in prefer:
        if name in atom_map:
            _, x, y, z, _ = atom_map[name]
            return np.array([x, y, z], dtype=float)
    for name in fallback:
        if name in atom_map:
            _, x, y, z, _ = atom_map[name]
            return np.array([x, y, z], dtype=float)
    return heavy_atom_coords(atom_list).mean(axis=0)

def is_protein_res(resname, atoms):
    return resname in AA3_TO_1

def is_nucleic_res(resname, atoms):
    if resname in DNA_RNA_BASE:
        return True
    names = {a[0] for a in atoms}
    sugar = ("C1'" in names) or ("C1*" in names) or ("O4'" in names) or ("O4*" in names)
    return ("P" in names) and sugar

def base_from_resname(resname):
    if resname in DNA_RNA_BASE:
        return DNA_RNA_BASE[resname]
    last = resname[-1] if resname else ""
    if last in ("A", "C", "G", "T"):
        return last
    if last == "U":
        return "T"
    if resname.startswith("D") and len(resname) >= 2 and resname[1] in ("A", "C", "G", "T", "U"):
        return "T" if resname[1] == "U" else resname[1]
    for ch in ("A", "C", "G", "T", "U"):
        if ch in resname:
            return "T" if ch == "U" else ch
    return None

def parse_pdb_first_model(pdb_path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(os.path.basename(pdb_path), pdb_path)
    model = next(structure.get_models())
    residues = {}
    for chain in model:
        for residue in chain:
            resname = residue.get_resname().strip().upper()
            resseq = str(residue.id[1])
            icode = str(residue.id[2]).strip()
            chain_id = str(chain.id).strip()
            atoms = []
            for atom in residue.get_atoms():
                altloc = str(atom.get_altloc()).strip()
                if altloc not in ("", "A"):
                    continue
                atoms.append(atom_to_tuple(atom))
            if not atoms:
                continue
            key = (chain_id, resseq, icode, resname)
            residues[key] = atoms
    return residues

def extract_entities(residues):
    protein = defaultdict(list)
    nucleic = defaultdict(list)

    def sort_key(key):
        try:
            rnum = int(key[1])
        except Exception:
            rnum = 0
        return (rnum, key[2] or "")

    for key, atoms in residues.items():
        chain, resseq, icode, resname = key
        if is_protein_res(resname, atoms):
            aa = AA3_TO_1[resname]
            protein[chain].append({
                "key": key,
                "aa": aa,
                "rep": rep_coord(atoms, prefer=("CA",)),
                "heavy": heavy_atom_coords(atoms),
                "resname": resname,
            })
        elif is_nucleic_res(resname, atoms):
            base = base_from_resname(resname) or "N"
            nucleic[chain].append({
                "key": key,
                "base": base,
                "rep": rep_coord(atoms, prefer=("P", "C1'", "C1*"), fallback=()),
                "heavy": heavy_atom_coords(atoms),
                "resname": resname,
            })

    for ch in list(protein.keys()):
        protein[ch] = sorted(protein[ch], key=lambda r: sort_key(r["key"]))
        if len(protein[ch]) > MAX_RESIDUES_PER_CHAIN:
            protein[ch] = protein[ch][:MAX_RESIDUES_PER_CHAIN]
    for ch in list(nucleic.keys()):
        nucleic[ch] = sorted(nucleic[ch], key=lambda r: sort_key(r["key"]))

    return protein, nucleic

def choose_best_nucleic_chain(protein, nucleic):
    if not nucleic or not protein:
        return None
    prot_reps = []
    for _, lst in protein.items():
        prot_reps.extend([r["rep"] for r in lst])
    if not prot_reps:
        return None
    prot_reps = np.stack(prot_reps, axis=0)

    best = None
    best_score = 1e9
    for nch, nlist in nucleic.items():
        bases = [r["base"] for r in nlist]
        clean_len = sum(b in ("A", "C", "G", "T") for b in bases)
        if clean_len < 8:
            continue
        nreps = np.stack([r["rep"] for r in nlist], axis=0)
        diff = prot_reps[:, None, :] - nreps[None, :, :]
        d = np.sqrt(np.sum(diff * diff, axis=2))
        min_d = float(d.min())
        score = min_d - 0.002 * clean_len
        if score < best_score:
            best_score = score
            best = nch
    return best

def min_dist(A, B):
    diff = A[:, None, :] - B[None, :, :]
    d2 = np.sum(diff * diff, axis=2)
    return float(np.sqrt(d2.min()))

def residue_distance_matrix(prot_residues, nt_residues):
    M = len(prot_residues)
    N = len(nt_residues)
    D = np.zeros((M, N), dtype=np.float32)
    for i, pres in enumerate(prot_residues):
        A = pres["heavy"]
        for j, nres in enumerate(nt_residues):
            D[i, j] = min_dist(A, nres["heavy"])
    return D

def motif_distance(D, aa_start, aa_len, nt_start, nt_len):
    return float(D[aa_start:aa_start+aa_len, nt_start:nt_start+nt_len].min())

def nt_flank(seq, start, length):
    left = seq[start-1] if start > 0 else "^"
    right = seq[start + length] if (start + length) < len(seq) else "$"
    return left, right

def aa_flank(seq, start, length):
    left = seq[start-1] if start > 0 else "^"
    right = seq[start + length] if (start + length) < len(seq) else "$"
    return left, right

def shannon_entropy(s):
    if not s:
        return 0.0
    c = Counter(s)
    p = np.array([v / len(s) for v in c.values()], dtype=float)
    return float(-(p * np.log2(np.maximum(p, 1e-12))).sum())

def nt_features(seq, ss):
    gc = (seq.count("G") + seq.count("C")) / max(1, len(seq))
    pur = (seq.count("A") + seq.count("G")) / max(1, len(seq))
    paired_frac = sum(ch == "P" for ch in ss) / max(1, len(ss))
    return {
        "nt_len": len(seq),
        "nt_gc": gc,
        "nt_purine": pur,
        "nt_entropy": shannon_entropy(seq),
        "ss_paired_frac": paired_frac,
        "nt_is_homopolymer": int(len(set(seq)) == 1),
    }

def aa_features(seq):
    charge = np.mean([AA_CHARGE.get(a, 0.0) for a in seq]) if seq else 0.0
    hydro = np.mean([AA_HYDRO.get(a, 0.0) for a in seq]) if seq else 0.0
    arom = np.mean([1.0 if a in AA_AROMATIC else 0.0 for a in seq]) if seq else 0.0
    posf = np.mean([1.0 if a in ("K", "R", "H") else 0.0 for a in seq]) if seq else 0.0
    negf = np.mean([1.0 if a in ("D", "E") else 0.0 for a in seq]) if seq else 0.0
    return {
        "aa_len": len(seq),
        "aa_charge_mean": float(charge),
        "aa_hydro_mean": float(hydro),
        "aa_arom_frac": float(arom),
        "aa_pos_frac": float(posf),
        "aa_neg_frac": float(negf),
        "aa_entropy": shannon_entropy(seq),
    }

def distance_bin(d):
    if d <= 5.0:
        return "strong"
    if d <= 8.0:
        return "medium"
    if d <= 12.0:
        return "weak"
    return "none"

def interaction_score_from_distance(d):
    if d > CONTACT_CUTOFF:
        return 0.0
    return float(math.exp(-(d - 2.0) / 4.0))

def sample_weight_from_row(is_pos, d):
    if is_pos:
        if d <= 5.0:
            return 1.40
        if d <= 8.0:
            return 1.20
        return 1.00
    if d < 20.0:
        return 1.00
    return 0.70

def ensure_rnafold_available():
    global USE_RNAFOLD
    if not USE_RNAFOLD:
        return False
    if AUTO_INSTALL_VIENNARNA and shutil.which("RNAfold") is None:
        try:
            print("[SETUP] Installing ViennaRNA (RNAfold) ...")
            subprocess.run(["bash", "-lc", "apt-get -y update && apt-get -y install viennarna"], check=False)
        except Exception:
            pass
    ok = shutil.which("RNAfold") is not None
    if not ok:
        USE_RNAFOLD = False
        print("[INFO] RNAfold not available; using dot-bracket proxy.")
    else:
        print("[INFO] RNAfold available.")
    return ok

def ss_proxy_for_sequence(seq: str) -> str:
    seq = str(seq)
    ss = []
    for i in range(len(seq)):
        tri = seq[max(0, i-1):min(len(seq), i+2)]
        if tri in ("GGG", "CCC", "GCG", "CGC"):
            ss.append("P")
        else:
            ss.append("U")
    return "".join(ss)

def rnafold_ss_and_mfe(seq: str):
    seq = seq.strip().upper()
    if not seq:
        return "", np.nan, ""
    if not USE_RNAFOLD:
        ss = ss_proxy_for_sequence(seq)
        return ss, np.nan, "".join("." if c == "U" else "(" for c in ss)
    seq_rna = seq.replace("T", "U")
    try:
        p = subprocess.run(
            ["bash", "-lc", f"printf '{seq_rna}\n' | RNAfold --noPS"],
            capture_output=True,
            text=True
        )
        if p.returncode != 0:
            ss = ss_proxy_for_sequence(seq)
            return ss, np.nan, "".join("." if c == "U" else "(" for c in ss)
        lines = [ln.strip() for ln in p.stdout.splitlines() if ln.strip()]
        if len(lines) < 2:
            ss = ss_proxy_for_sequence(seq)
            return ss, np.nan, "".join("." if c == "U" else "(" for c in ss)
        struct_line = lines[1]
        m = re.search(r"([().]+)\s+\(([-0-9.]+)\)", struct_line)
        if m:
            dotb = m.group(1)
            mfe = float(m.group(2))
        else:
            parts = struct_line.split()
            dotb = parts[0]
            mm = re.search(r"\(([-0-9.]+)\)", struct_line)
            mfe = float(mm.group(1)) if mm else np.nan
        ss = "".join("P" if ch in ("(", ")") else "U" for ch in dotb)
        if len(ss) != len(seq):
            ss = ss_proxy_for_sequence(seq)
        return ss, mfe, dotb
    except Exception:
        ss = ss_proxy_for_sequence(seq)
        return ss, np.nan, "".join("." if c == "U" else "(" for c in ss)

# ------------------------------------------------------------
# Dataset extraction
# ------------------------------------------------------------

def build_interaction_rows_for_complex(pdb_id, protein, nuc_chain_id, nucleic):
    nt_res = [r for r in nucleic[nuc_chain_id] if r["base"] in ("A", "C", "G", "T")]
    if len(nt_res) < 8:
        return None, None

    aptamer_seq = "".join(r["base"] for r in nt_res)
    ss_str, mfe, dotb = rnafold_ss_and_mfe(aptamer_seq)

    rows = []
    protein_seq_records = []
    chain_summaries = []

    for prot_chain, prot_res in protein.items():
        if len(prot_res) < 1:
            continue
        prot_seq = "".join(r["aa"] for r in prot_res)
        protein_seq_records.append({
            "pdb_id": pdb_id,
            "prot_chain": prot_chain,
            "sequence": prot_seq,
            "length": len(prot_seq),
        })

        D = residue_distance_matrix(prot_res, nt_res)
        if float(D.min()) > (CONTACT_CUTOFF + 6.0):
            continue

        for aa_len in (1, 2, 3):
            if len(prot_seq) < aa_len:
                continue
            for nt_len in (1, 2, 3):
                if len(aptamer_seq) < nt_len:
                    continue

                positives = []
                negatives = []

                for ai in range(len(prot_seq) - aa_len + 1):
                    aa_motif = prot_seq[ai:ai+aa_len]
                    aa_left, aa_right = aa_flank(prot_seq, ai, aa_len)
                    for ni in range(len(aptamer_seq) - nt_len + 1):
                        nt_motif = aptamer_seq[ni:ni+nt_len]
                        ss_motif = ss_str[ni:ni+nt_len]
                        nt_left, nt_right = nt_flank(aptamer_seq, ni, nt_len)
                        d = motif_distance(D, ai, aa_len, ni, nt_len)
                        is_pos = int(d <= CONTACT_CUTOFF)
                        if not is_pos and not (NEGATIVE_MIN_DIST <= d <= NEGATIVE_MAX_DIST):
                            continue

                        base_row = {
                            "pdb_id": pdb_id,
                            "prot_chain": prot_chain,
                            "dna_chain": nuc_chain_id,
                            "aa_motif": aa_motif,
                            "aa_left": aa_left,
                            "aa_right": aa_right,
                            "aa_center": aa_motif[len(aa_motif)//2],
                            "aa_len": aa_len,
                            "nt_motif": nt_motif,
                            "nt_left": nt_left,
                            "nt_right": nt_right,
                            "nt_center": nt_motif[len(nt_motif)//2],
                            "nt_len": nt_len,
                            "ss_motif": ss_motif,
                            "combo": f"nt{nt_len}_aa{aa_len}",
                            "aa_start": ai,
                            "nt_start": ni,
                            "min_distance": round(float(d), 4),
                            "distance_bin": distance_bin(d),
                            "interaction": is_pos,
                            "interaction_score": round(interaction_score_from_distance(d), 6),
                            "sample_weight": sample_weight_from_row(is_pos, d),
                            "aptamer_length": len(aptamer_seq),
                            "protein_length": len(prot_seq),
                            "aptamer_mfe": mfe if not np.isnan(mfe) else np.nan,
                            "aptamer_dotbracket": dotb,
                        }
                        base_row.update(nt_features(nt_motif, ss_motif))
                        base_row.update(aa_features(aa_motif))

                        if is_pos:
                            positives.append(base_row)
                        else:
                            negatives.append(base_row)

                if positives:
                    rows.extend(positives)
                    if negatives:
                        take = min(len(negatives), max(1, int(round(len(positives) * NEGATIVE_RATIO))))
                        pick_idx = rng.choice(len(negatives), size=take, replace=False)
                        for idx in pick_idx:
                            rows.append(negatives[int(idx)])

        chain_summaries.append({
            "pdb_id": pdb_id,
            "prot_chain": prot_chain,
            "protein_length": len(prot_seq),
            "aptamer_chain": nuc_chain_id,
            "aptamer_length": len(aptamer_seq),
            "min_residue_distance": float(D.min()),
            "aptamer_mfe": mfe if not np.isnan(mfe) else np.nan,
        })

    if not rows:
        return None, None

    meta = {
        "aptamer_record": {
            "pdb_id": pdb_id,
            "dna_chain": nuc_chain_id,
            "sequence": aptamer_seq,
            "length": len(aptamer_seq),
            "secondary_structure": ss_str,
            "dotbracket": dotb,
            "mfe": mfe if not np.isnan(mfe) else np.nan,
        },
        "protein_records": protein_seq_records,
        "chain_summary": chain_summaries,
    }
    return pd.DataFrame(rows), meta

def generate_interaction_dataset_from_pdbs(pdb_files, out_csv):
    header_written = False
    skip_log = []
    dna_seq_records = []
    prot_seq_records = []
    complex_summary = []
    first_example_saved = False
    total_rows = 0

    for pdb_path in pdb_files:
        pdb_id = os.path.splitext(os.path.basename(pdb_path))[0]
        try:
            residues = parse_pdb_first_model(pdb_path)
            protein, nucleic = extract_entities(residues)

            if not protein:
                skip_log.append({"pdb_id": pdb_id, "reason": "no_protein_detected"})
                continue
            if not nucleic:
                skip_log.append({"pdb_id": pdb_id, "reason": "no_nucleic_detected"})
                continue

            best_nuc = choose_best_nucleic_chain(protein, nucleic)
            if best_nuc is None:
                skip_log.append({"pdb_id": pdb_id, "reason": "no_suitable_aptamer_chain"})
                continue

            df_rows, meta = build_interaction_rows_for_complex(pdb_id, protein, best_nuc, nucleic)
            if df_rows is None or df_rows.empty:
                skip_log.append({"pdb_id": pdb_id, "reason": "no_rows_after_contact_filter"})
                continue

            if not header_written:
                df_rows.to_csv(out_csv, index=False, mode="w")
                header_written = True
            else:
                df_rows.to_csv(out_csv, index=False, mode="a", header=False)

            total_rows += len(df_rows)
            dna_seq_records.append(meta["aptamer_record"])
            prot_seq_records.extend(meta["protein_records"])
            complex_summary.extend(meta["chain_summary"])

            if not first_example_saved:
                example_path = os.path.join(OUTPUT_DIR, "example_one_pdb_interactions.csv")
                df_rows.head(1500).to_csv(example_path, index=False)
                with open(os.path.join(OUTPUT_DIR, "example_one_pdb_summary.txt"), "w") as f:
                    f.write("Example data structure from one PDB complex\n")
                    f.write("=========================================\n")
                    f.write(f"PDB ID: {pdb_id}\n")
                    f.write(f"Aptamer chain: {meta['aptamer_record']['dna_chain']}\n")
                    f.write(f"Aptamer sequence: {meta['aptamer_record']['sequence']}\n")
                    f.write(f"Aptamer SS (P/U): {meta['aptamer_record']['secondary_structure']}\n")
                    f.write(f"Aptamer dot-bracket: {meta['aptamer_record']['dotbracket']}\n")
                    f.write(f"Aptamer MFE: {meta['aptamer_record']['mfe']}\n\n")
                    f.write("Example interaction row columns:\n")
                    f.write(", ".join(df_rows.columns.tolist()) + "\n\n")
                    f.write("Interpretation:\n")
                    f.write("- nt_motif / aa_motif: nucleotide and amino-acid motif windows\n")
                    f.write("- combo: motif-length pairing such as nt1_aa2 or nt3_aa1\n")
                    f.write("- min_distance: minimum heavy-atom distance between the two motif windows\n")
                    f.write("- interaction: 1 if min_distance <= 12 Å, else sampled non-contact negative\n")
                    f.write("- interaction_score: distance-derived strength score for positives\n")
                    f.write("- ss_motif: aptamer secondary structure labels aligned to nt_motif\n")
                first_example_saved = True

            del df_rows
            gc.collect()

        except Exception as e:
            skip_log.append({"pdb_id": pdb_id, "reason": f"exception:{type(e).__name__}"})

    pd.DataFrame(skip_log).to_csv(os.path.join(OUTPUT_DIR, "skip_log.csv"), index=False)
    pd.DataFrame(dna_seq_records).to_csv(os.path.join(OUTPUT_DIR, "dna_sequences.csv"), index=False)
    pd.DataFrame(prot_seq_records).to_csv(os.path.join(OUTPUT_DIR, "protein_sequences.csv"), index=False)
    pd.DataFrame(complex_summary).to_csv(os.path.join(OUTPUT_DIR, "complex_summary.csv"), index=False)

    print(f"[INFO] Processed complexes: {len(dna_seq_records)}")
    print(f"[INFO] Skipped complexes: {len(skip_log)}")
    print(f"[INFO] Total interaction rows: {total_rows:,}")
    return len(dna_seq_records)

# ------------------------------------------------------------
# Specificity matrices
# ------------------------------------------------------------

def plot_heatmap_from_matrix(mat_df, title, out_path, max_rows=32, max_cols=32):
    if mat_df.empty:
        return
    use = mat_df.copy()
    row_order = use.mean(axis=1).sort_values(ascending=False).index[:max_rows]
    col_order = use.mean(axis=0).sort_values(ascending=False).index[:max_cols]
    use = use.loc[row_order, col_order]
    plt.figure(figsize=(12, 10))
    sns.heatmap(use, cmap="viridis")
    plt.title(title)
    plt.xlabel("Amino-acid motif")
    plt.ylabel("Nucleotide motif")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def build_specificity_outputs(interaction_csv):
    df = pd.read_csv(interaction_csv, low_memory=False)
    combos = sorted(df["combo"].dropna().unique().tolist())

    combo_summary = []
    for combo in combos:
        sub = df[df["combo"] == combo].copy()
        if sub.empty:
            continue

        grp = sub.groupby(["nt_motif", "aa_motif"], as_index=False).agg(
            n_examples=("interaction", "size"),
            positive_rate=("interaction", "mean"),
            mean_contact_score=("interaction_score", "mean"),
            mean_min_distance=("min_distance", "mean"),
        )
        grp["specificity_score"] = grp["positive_rate"] * np.log1p(grp["n_examples"])
        grp = grp.sort_values(["specificity_score", "n_examples"], ascending=False)
        grp.to_csv(os.path.join(OUTPUT_DIR, f"specificity_{combo}_long.csv"), index=False)

        rate_mat = grp.pivot(index="nt_motif", columns="aa_motif", values="positive_rate").fillna(0.0)
        score_mat = grp.pivot(index="nt_motif", columns="aa_motif", values="specificity_score").fillna(0.0)

        rate_mat.to_csv(os.path.join(OUTPUT_DIR, f"specificity_{combo}_matrix_positive_rate.csv"))
        score_mat.to_csv(os.path.join(OUTPUT_DIR, f"specificity_{combo}_matrix_specificity_score.csv"))
        rate_mat.T.to_csv(os.path.join(OUTPUT_DIR, f"specificity_{combo}_matrix_positive_rate_transposed.csv"))
        score_mat.T.to_csv(os.path.join(OUTPUT_DIR, f"specificity_{combo}_matrix_specificity_score_transposed.csv"))

        plot_heatmap_from_matrix(
            rate_mat,
            f"{combo}: positive rate matrix",
            os.path.join(OUTPUT_DIR, f"specificity_{combo}_heatmap_positive_rate.tiff")
        )
        plot_heatmap_from_matrix(
            score_mat,
            f"{combo}: specificity score matrix",
            os.path.join(OUTPUT_DIR, f"specificity_{combo}_heatmap_specificity_score.tiff")
        )

        combo_summary.append({
            "combo": combo,
            "n_rows": len(sub),
            "n_unique_nt_motifs": int(sub["nt_motif"].nunique()),
            "n_unique_aa_motifs": int(sub["aa_motif"].nunique()),
            "positive_rate_overall": float(sub["interaction"].mean()),
            "mean_contact_score_overall": float(sub["interaction_score"].mean()),
        })

    pd.DataFrame(combo_summary).to_csv(os.path.join(OUTPUT_DIR, "specificity_combo_summary.csv"), index=False)

# ------------------------------------------------------------
# Train / eval
# ------------------------------------------------------------

FEATURE_COLS = [
    "combo",
    "nt_motif",
    "ss_motif",
    "nt_left",
    "nt_right",
    "nt_center",
    "aa_motif",
    "aa_left",
    "aa_right",
    "aa_center",
    "nt_len",
    "aa_len",
    "nt_gc",
    "nt_purine",
    "nt_entropy",
    "ss_paired_frac",
    "nt_is_homopolymer",
    "aa_charge_mean",
    "aa_hydro_mean",
    "aa_arom_frac",
    "aa_pos_frac",
    "aa_neg_frac",
    "aa_entropy",
]
CAT_COLS = [
    "combo", "nt_motif", "ss_motif", "nt_left", "nt_right", "nt_center",
    "aa_motif", "aa_left", "aa_right", "aa_center"
]
NUM_COLS = [c for c in FEATURE_COLS if c not in CAT_COLS]

def reduce_ambiguity(train_df, hold_df):
    key_cols = ["combo", "nt_motif", "ss_motif", "nt_left", "nt_right", "aa_motif", "aa_left", "aa_right"]
    train_tmp = train_df.copy()
    train_tmp["feature_key"] = train_tmp[key_cols].astype(str).agg("|".join, axis=1)

    amb = train_tmp.groupby("feature_key", as_index=False).agg(
        support=("interaction", "size"),
        pos_rate=("interaction", "mean")
    )
    amb = amb[(amb["support"] >= GLOBAL_AMBIG_MIN_SUPPORT) &
              (amb["pos_rate"] >= GLOBAL_AMBIG_LOW) &
              (amb["pos_rate"] <= GLOBAL_AMBIG_HIGH)]
    ambiguous_keys = set(amb["feature_key"].tolist())

    if ambiguous_keys:
        train_df = train_df[~train_tmp["feature_key"].isin(ambiguous_keys)].copy()

        hold_tmp = hold_df.copy()
        hold_tmp["feature_key"] = hold_tmp[key_cols].astype(str).agg("|".join, axis=1)
        hold_df = hold_df[~hold_tmp["feature_key"].isin(ambiguous_keys)].copy()

    return train_df.reset_index(drop=True), hold_df.reset_index(drop=True), len(ambiguous_keys)

def deduplicate_split(df):
    agg_cols = FEATURE_COLS + ["pdb_id"]
    out = df.groupby(agg_cols, as_index=False).agg(
        interaction=("interaction", "mean"),
        interaction_score=("interaction_score", "mean"),
        min_distance=("min_distance", "mean"),
        sample_weight=("sample_weight", "mean"),
        n_redundant=("interaction", "size"),
    )
    out["interaction"] = (out["interaction"] >= 0.5).astype(int)
    out["sample_weight"] = out["sample_weight"] * np.log1p(out["n_redundant"])
    return out

def threshold_search(y_true, proba):
    best_thr, best_mcc = 0.5, -1.0
    for thr in np.linspace(0.20, 0.80, 61):
        pred = (proba >= thr).astype(int)
        mcc = matthews_corrcoef(y_true, pred)
        if mcc > best_mcc:
            best_mcc = mcc
            best_thr = float(thr)
    return best_thr, best_mcc

def fit_catboost(X_train, y_train, w_train, X_val, y_val, cat_idx, params):
    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=SEED,
        verbose=False,
        allow_writing_files=False,
        **params
    )
    model.fit(
        X_train, y_train,
        sample_weight=w_train,
        cat_features=cat_idx,
        eval_set=(X_val, y_val),
        use_best_model=True,
    )
    return model

def cv_search_catboost(train_df):
    groups = train_df["pdb_id"].astype(str).values
    X = train_df[FEATURE_COLS].copy()
    y = train_df["interaction"].astype(int).values
    w = train_df["sample_weight"].astype(float).values

    cat_idx = [X.columns.get_loc(c) for c in CAT_COLS]

    param_grid = [
        {"depth": 6, "learning_rate": 0.05, "l2_leaf_reg": 6, "iterations": 700},
        {"depth": 7, "learning_rate": 0.05, "l2_leaf_reg": 8, "iterations": 850},
        {"depth": 8, "learning_rate": 0.04, "l2_leaf_reg": 10, "iterations": 950},
        {"depth": 8, "learning_rate": 0.03, "l2_leaf_reg": 12, "iterations": 1100},
    ]

    cv = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    results = []
    best_params = None
    best_thr = 0.5
    best_score = -1e9

    for params in param_grid:
        oof = np.zeros(len(train_df), dtype=float)
        fold_scores = []
        for fold_id, (tr_idx, va_idx) in enumerate(cv.split(X, y, groups=groups), start=1):
            X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
            y_tr, y_va = y[tr_idx], y[va_idx]
            w_tr = w[tr_idx]

            model = fit_catboost(X_tr, y_tr, w_tr, X_va, y_va, cat_idx, params)
            pred_va = model.predict_proba(X_va)[:, 1]
            oof[va_idx] = pred_va
            fold_scores.append({
                "fold": fold_id,
                "pr_auc": average_precision_score(y_va, pred_va),
                "roc_auc": roc_auc_score(y_va, pred_va),
            })

        thr, cv_mcc = threshold_search(y, oof)
        cv_pr = average_precision_score(y, oof)
        cv_auc = roc_auc_score(y, oof)
        cv_bacc = balanced_accuracy_score(y, (oof >= thr).astype(int))
        final_score = 0.40 * cv_pr + 0.25 * cv_auc + 0.25 * cv_mcc + 0.10 * cv_bacc
        results.append({
            "params": json.dumps(params),
            "cv_pr_auc": cv_pr,
            "cv_roc_auc": cv_auc,
            "cv_best_mcc": cv_mcc,
            "cv_best_balanced_accuracy": cv_bacc,
            "cv_best_threshold": thr,
            "selection_score": final_score,
            "fold_summary": json.dumps(fold_scores),
        })

        if final_score > best_score:
            best_score = final_score
            best_params = params
            best_thr = thr

    pd.DataFrame(results).to_csv(os.path.join(OUTPUT_DIR, "cv_model_search_results.csv"), index=False)

    # Refit best fold models for an ensemble; fit calibrator on train-only OOF predictions.
    ensemble_models = []
    oof_best = np.zeros(len(train_df), dtype=float)
    for fold_id, (tr_idx, va_idx) in enumerate(cv.split(X, y, groups=groups), start=1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        w_tr = w[tr_idx]
        model = fit_catboost(X_tr, y_tr, w_tr, X_va, y_va, cat_idx, best_params)
        ensemble_models.append(model)
        oof_best[va_idx] = model.predict_proba(X_va)[:, 1]

    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(oof_best, y)
    calibrated_oof = calibrator.predict(oof_best)
    cal_thr, cal_mcc = threshold_search(y, calibrated_oof)

    final_model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=SEED,
        verbose=False,
        allow_writing_files=False,
        **best_params
    )
    final_model.fit(X, y, sample_weight=w, cat_features=cat_idx)

    pd.DataFrame([{
        "threshold_raw_oof": best_thr,
        "threshold_calibrated_oof": cal_thr,
        "mcc_calibrated_oof": cal_mcc,
        "pr_auc_raw_oof": average_precision_score(y, oof_best),
        "roc_auc_raw_oof": roc_auc_score(y, oof_best),
        "pr_auc_calibrated_oof": average_precision_score(y, calibrated_oof),
        "roc_auc_calibrated_oof": roc_auc_score(y, calibrated_oof),
    }]).to_csv(os.path.join(OUTPUT_DIR, "oof_calibration_summary.csv"), index=False)

    return {
        "model": final_model,
        "ensemble_models": ensemble_models,
        "calibrator": calibrator,
        "best_params": best_params,
        "threshold": cal_thr,
        "cat_idx": cat_idx,
    }

def predict_proba_ensemble(model_bundle, X):
    models = model_bundle.get("ensemble_models", []) + [model_bundle["model"]]
    pred_matrix = np.column_stack([m.predict_proba(X)[:, 1] for m in models])
    raw_mean = pred_matrix.mean(axis=1)
    raw_std = pred_matrix.std(axis=1)
    calibrator = model_bundle.get("calibrator")
    if calibrator is not None:
        cal_mean = calibrator.predict(raw_mean)
    else:
        cal_mean = raw_mean
    return cal_mean, raw_std, raw_mean

def evaluate_predictions(y_true, proba, thr):
    pred = (proba >= thr).astype(int)
    return {
        "threshold": float(thr),
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "mcc": float(matthews_corrcoef(y_true, pred)),
        "f1": float(f1_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
    }, pred

def plot_eval_outputs(y_true, proba, pred, prefix="heldout"):
    conf = confusion_matrix(y_true, pred)
    report = classification_report(y_true, pred, digits=4)

    plt.figure()
    RocCurveDisplay.from_predictions(y_true, proba)
    plt.title(f"ROC ({prefix})")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{prefix}_roc.tiff"))
    plt.close()

    plt.figure()
    PrecisionRecallDisplay.from_predictions(y_true, proba)
    plt.title(f"Precision-Recall ({prefix})")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{prefix}_pr.tiff"))
    plt.close()

    plt.figure()
    sns.heatmap(conf, annot=True, fmt="d", cmap="Blues", cbar_kws={"label": "Count"})
    plt.title(f"Confusion Matrix ({prefix})")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{prefix}_confusion.tiff"))
    plt.close()

    with open(os.path.join(OUTPUT_DIR, f"{prefix}_report.txt"), "w") as f:
        f.write(report)
        f.write("\nConfusion matrix:\n")
        f.write(np.array2string(conf, separator=", "))

def plot_feature_importance(model, X):
    fi = pd.DataFrame({
        "feature": X.columns,
        "importance": model.get_feature_importance()
    }).sort_values("importance", ascending=False)
    fi.to_csv(os.path.join(OUTPUT_DIR, "feature_importance.csv"), index=False)

    plt.figure(figsize=(10, 8))
    top = fi.head(25).sort_values("importance", ascending=True)
    plt.barh(top["feature"], top["importance"])
    plt.title("Top feature importances")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance_top25.tiff"))
    plt.close()

def maybe_downsample_rows(df):
    if len(df) <= MAX_TRAIN_ROWS:
        return df
    pos = df[df["interaction"] == 1]
    neg = df[df["interaction"] == 0]
    target_pos = min(len(pos), MAX_TRAIN_ROWS // 2)
    target_neg = min(len(neg), MAX_TRAIN_ROWS - target_pos)
    pos_s = pos.sample(target_pos, random_state=SEED) if len(pos) > target_pos else pos
    neg_s = neg.sample(target_neg, random_state=SEED) if len(neg) > target_neg else neg
    out = pd.concat([pos_s, neg_s], axis=0).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    return out

def train_eval_save(interaction_csv):
    df = pd.read_csv(interaction_csv, low_memory=False)
    df = df[df["aa_motif"].astype(str).str.fullmatch(r"[ACDEFGHIKLMNPQRSTVWY]+", na=False)].copy()
    df = df[df["nt_motif"].astype(str).str.fullmatch(r"[ACGT]+", na=False)].copy()
    df = df[df["ss_motif"].astype(str).str.fullmatch(r"[PU]+", na=False)].copy()

    unique_pdb = sorted(df["pdb_id"].astype(str).unique().tolist())
    if len(unique_pdb) < 8:
        raise RuntimeError("Too few unique PDB complexes for grouped train/holdout evaluation.")

    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
    tr_idx, te_idx = next(gss.split(df, df["interaction"], groups=df["pdb_id"].astype(str)))
    train_df = df.iloc[tr_idx].copy().reset_index(drop=True)
    hold_df = df.iloc[te_idx].copy().reset_index(drop=True)

    train_df = deduplicate_split(train_df)
    hold_df = deduplicate_split(hold_df)
    train_df, hold_df, n_amb = reduce_ambiguity(train_df, hold_df)

    # Slightly downweight far negatives after split/cleaning.
    train_df.loc[train_df["interaction"] == 0, "sample_weight"] *= 0.92
    train_df = maybe_downsample_rows(train_df)

    print(f"[INFO] Train rows after cleaning: {len(train_df):,}")
    print(f"[INFO] Holdout rows after cleaning: {len(hold_df):,}")
    print(f"[INFO] Ambiguous train-derived feature patterns removed: {n_amb}")

    model_bundle = cv_search_catboost(train_df)

    X_train = train_df[FEATURE_COLS].copy()
    X_hold = hold_df[FEATURE_COLS].copy()
    y_hold = hold_df["interaction"].astype(int).values

    proba_hold, proba_std, proba_hold_raw = predict_proba_ensemble(model_bundle, X_hold)
    metrics, pred_hold = evaluate_predictions(y_hold, proba_hold, model_bundle["threshold"])
    metrics["n_train_rows"] = int(len(train_df))
    metrics["n_holdout_rows"] = int(len(hold_df))
    metrics["n_holdout_pdb"] = int(hold_df["pdb_id"].nunique())
    metrics["best_params"] = json.dumps(model_bundle["best_params"])
    metrics["holdout_pred_std_mean"] = float(np.mean(proba_std))
    metrics["holdout_pred_std_median"] = float(np.median(proba_std))

    pd.DataFrame([metrics]).to_csv(os.path.join(OUTPUT_DIR, "performance_summary.csv"), index=False)
    plot_eval_outputs(y_hold, proba_hold, pred_hold, prefix="heldout")
    plot_feature_importance(model_bundle["model"], X_train)

    calib_df = pd.DataFrame({
        "y_true": y_hold,
        "proba_calibrated": proba_hold,
        "proba_raw_mean": proba_hold_raw,
        "ensemble_std": proba_std,
        "pred": pred_hold,
    })
    calib_df.to_csv(os.path.join(OUTPUT_DIR, "heldout_predictions.csv"), index=False)

    joblib.dump({
        "model": model_bundle["model"],
        "ensemble_models": model_bundle["ensemble_models"],
        "calibrator": model_bundle["calibrator"],
        "feature_cols": FEATURE_COLS,
        "cat_cols": CAT_COLS,
        "threshold": model_bundle["threshold"],
        "best_params": model_bundle["best_params"],
    }, os.path.join(OUTPUT_DIR, "aptamer_model_bundle.pkl"))

    return {
        "bundle": {
            "model": model_bundle["model"],
            "ensemble_models": model_bundle["ensemble_models"],
            "calibrator": model_bundle["calibrator"],
            "feature_cols": FEATURE_COLS,
            "cat_cols": CAT_COLS,
            "threshold": model_bundle["threshold"],
            "best_params": model_bundle["best_params"],
        },
        "train_df": train_df,
        "hold_df": hold_df,
        "train_pdb_ids": sorted(train_df["pdb_id"].astype(str).unique().tolist()),
    }

# ------------------------------------------------------------
# Protein-side and candidate-side motif tables for scoring
# ------------------------------------------------------------

def subsample_residues(seq, max_n):
    if len(seq) <= max_n:
        return seq
    idx = np.linspace(0, len(seq)-1, max_n).astype(int)
    return "".join(seq[i] for i in idx)

def protein_motif_table(protein_seq):
    protein_seq = subsample_residues(protein_seq.strip().upper(), MAX_RESIDUES_TO_SCORE)
    rows = []
    for aa_len in (1, 2, 3):
        if len(protein_seq) < aa_len:
            continue
        for i in range(len(protein_seq) - aa_len + 1):
            aa_motif = protein_seq[i:i+aa_len]
            aa_left, aa_right = aa_flank(protein_seq, i, aa_len)
            row = {
                "aa_motif": aa_motif,
                "aa_left": aa_left,
                "aa_right": aa_right,
                "aa_center": aa_motif[len(aa_motif)//2],
                "aa_len": aa_len,
                "combo_suffix": f"aa{aa_len}",
                "count": 1.0,
            }
            row.update(aa_features(aa_motif))
            rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    grp_cols = ["aa_motif", "aa_left", "aa_right", "aa_center", "aa_len", "combo_suffix",
                "aa_charge_mean", "aa_hydro_mean", "aa_arom_frac", "aa_pos_frac", "aa_neg_frac", "aa_entropy"]
    agg = df.groupby(grp_cols, as_index=False).agg(count=("count", "sum"))
    agg["weight"] = agg["count"] / agg["count"].sum()
    return agg

def nt_token_table(seq):
    seq = seq.strip().upper()
    ss_str, mfe, dotb = rnafold_ss_and_mfe(seq)
    rows = []
    for nt_len in (1, 2, 3):
        if len(seq) < nt_len:
            continue
        for i in range(len(seq) - nt_len + 1):
            nt_motif = seq[i:i+nt_len]
            ss_motif = ss_str[i:i+nt_len]
            nt_left, nt_right = nt_flank(seq, i, nt_len)
            row = {
                "nt_motif": nt_motif,
                "ss_motif": ss_motif,
                "nt_left": nt_left,
                "nt_right": nt_right,
                "nt_center": nt_motif[len(nt_motif)//2],
                "nt_len": nt_len,
                "combo_prefix": f"nt{nt_len}",
                "count": 1.0,
                "nt_start": i,
            }
            row.update(nt_features(nt_motif, ss_motif))
            rows.append(row)
    df = pd.DataFrame(rows)
    return df, ss_str, mfe, dotb

def build_pair_feature_frame(protein_table, nt_table):
    rows = []
    nt_indices = []
    for _, prow in protein_table.iterrows():
        for nt_idx, nrow in nt_table.iterrows():
            if int(prow["aa_len"]) not in (1, 2, 3) or int(nrow["nt_len"]) not in (1, 2, 3):
                continue
            combo = f"nt{int(nrow['nt_len'])}_aa{int(prow['aa_len'])}"
            row = {
                "combo": combo,
                "nt_motif": nrow["nt_motif"],
                "ss_motif": nrow["ss_motif"],
                "nt_left": nrow["nt_left"],
                "nt_right": nrow["nt_right"],
                "nt_center": nrow["nt_center"],
                "aa_motif": prow["aa_motif"],
                "aa_left": prow["aa_left"],
                "aa_right": prow["aa_right"],
                "aa_center": prow["aa_center"],
                "nt_len": int(nrow["nt_len"]),
                "aa_len": int(prow["aa_len"]),
                "nt_gc": float(nrow["nt_gc"]),
                "nt_purine": float(nrow["nt_purine"]),
                "nt_entropy": float(nrow["nt_entropy"]),
                "ss_paired_frac": float(nrow["ss_paired_frac"]),
                "nt_is_homopolymer": int(nrow["nt_is_homopolymer"]),
                "aa_charge_mean": float(prow["aa_charge_mean"]),
                "aa_hydro_mean": float(prow["aa_hydro_mean"]),
                "aa_arom_frac": float(prow["aa_arom_frac"]),
                "aa_pos_frac": float(prow["aa_pos_frac"]),
                "aa_neg_frac": float(prow["aa_neg_frac"]),
                "aa_entropy": float(prow["aa_entropy"]),
                "pair_weight": float(prow["weight"] * nrow["count"]),
                "nt_index": int(nrow["nt_start"]),
            }
            rows.append(row)
    pair_df = pd.DataFrame(rows)
    return pair_df

def gaussian_pref(x, mu, sigma):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return 0.5
    return float(math.exp(-((x - mu) ** 2) / (2 * (sigma ** 2))))

def is_low_complexity(seq):
    if re.search(r"(A{5,}|C{5,}|G{5,}|T{5,})", seq):
        return True
    counts = Counter(seq)
    if max(counts.values()) / len(seq) > 0.58:
        return True
    return False

def g4_risk_score(seq):
    return 1.0 if re.search(r"G{3,}.{0,7}G{3,}.{0,7}G{3,}.{0,7}G{3,}", seq) else 0.0

def score_candidate_sequence(seq, protein_seq, model_bundle, train_seed_sets=None):
    model = model_bundle["model"]
    protein_table = protein_motif_table(protein_seq)
    if protein_table.empty:
        return {
            "sequence": seq,
            "score": 0.0,
            "model_mean": 0.0,
            "model_uncertainty": 1.0,
            "position_coverage": 0.0,
            "gc_content": 0.0,
            "mfe": np.nan,
            "secondary_structure": "",
            "dotbracket": "",
            "binding_motif": "",
            "binding_motif_top3": "",
        }

    nt_table, ss_str, mfe, dotb = nt_token_table(seq)
    pair_df = build_pair_feature_frame(protein_table, nt_table)
    if pair_df.empty:
        return {
            "sequence": seq,
            "score": 0.0,
            "model_mean": 0.0,
            "model_uncertainty": 1.0,
            "position_coverage": 0.0,
            "gc_content": (seq.count("G") + seq.count("C")) / max(1, len(seq)),
            "mfe": mfe,
            "secondary_structure": ss_str,
            "dotbracket": dotb,
            "binding_motif": "",
            "binding_motif_top3": "",
        }

    proba_mean, proba_std, proba_raw = predict_proba_ensemble(model_bundle, pair_df[FEATURE_COLS])
    pair_df = pair_df.copy()
    pair_df["proba"] = proba_mean
    pair_df["proba_std"] = proba_std
    pair_df["weighted_proba"] = pair_df["proba"] * pair_df["pair_weight"]

    model_mean = float(pair_df["weighted_proba"].sum() / max(1e-12, pair_df["pair_weight"].sum()))
    model_uncertainty = float((pair_df["proba_std"] * pair_df["pair_weight"]).sum() / max(1e-12, pair_df["pair_weight"].sum()))

    per_pos = pair_df.groupby("nt_index", as_index=False).agg(
        max_proba=("proba", "max"),
        mean_uncertainty=("proba_std", "mean")
    )
    position_coverage = float(per_pos["max_proba"].mean()) if not per_pos.empty else 0.0

    gc_frac = (seq.count("G") + seq.count("C")) / max(1, len(seq))
    gc_pref = gaussian_pref(gc_frac, mu=0.55, sigma=0.10)
    mfe_pref = gaussian_pref(mfe, mu=-8.0, sigma=4.0) if not np.isnan(mfe) else 0.5

    motif_contrib = (
        pair_df.groupby(["nt_motif"], as_index=False)
        .agg(contribution=("weighted_proba", "sum"))
        .sort_values("contribution", ascending=False)
    )
    top_motifs = motif_contrib.head(3)["nt_motif"].tolist()
    contrib = motif_contrib["contribution"].values.astype(float)
    if contrib.sum() > 0:
        p = contrib / contrib.sum()
        binding_entropy = float(-(p * np.log(p + 1e-12)).sum() / np.log(len(p) + 1e-12))
    else:
        binding_entropy = 0.0

    novelty_penalty = 0.0
    novelty_to_train = np.nan
    if train_seed_sets:
        ks = kmer_set(seq, DIVERSITY_K)
        max_j = max((jaccard(ks, prev) for prev in train_seed_sets), default=0.0)
        novelty_to_train = 1.0 - max_j
        if max_j >= NEAR_DUPLICATE_TO_TRAIN_JACCARD:
            novelty_penalty += 0.08

    penalty = 0.0
    if is_low_complexity(seq):
        penalty += 0.15
    if gc_frac < 0.30 or gc_frac > 0.75:
        penalty += 0.10
    penalty += 0.10 * g4_risk_score(seq)
    penalty += novelty_penalty

    score = (0.56 * model_mean) + (0.14 * position_coverage) + (0.08 * gc_pref) + (0.08 * mfe_pref) + (0.04 * binding_entropy) - (ENSEMBLE_UNCERTAINTY_PENALTY * model_uncertainty) - penalty

    return {
        "sequence": seq,
        "length": len(seq),
        "score": float(score),
        "model_mean": model_mean,
        "model_uncertainty": model_uncertainty,
        "position_coverage": position_coverage,
        "gc_content": gc_frac,
        "mfe": mfe,
        "secondary_structure": ss_str,
        "dotbracket": dotb,
        "binding_motif": top_motifs[0] if top_motifs else "",
        "binding_motif_top3": ",".join(top_motifs),
        "g4_risk": float(g4_risk_score(seq)),
        "low_complexity": int(is_low_complexity(seq)),
        "gc_pref": gc_pref,
        "mfe_pref": mfe_pref,
        "binding_entropy": binding_entropy,
        "novelty_to_train": novelty_to_train,
        "penalty": penalty,
    }

# ------------------------------------------------------------
# Held-out sequence retrieval evaluation
# ------------------------------------------------------------

def mutate(seq, rate):
    seq = list(seq)
    for i in range(len(seq)):
        if rng.random() < rate:
            choices = [b for b in "ACGT" if b != seq[i]]
            seq[i] = rng.choice(choices)
    return "".join(seq)

def random_seq(L):
    return "".join(rng.choice(NUCLEOTIDES, size=L))

def make_decoys(true_seq, n=49):
    decoys = set()
    while len(decoys) < n:
        if rng.random() < 0.5:
            s = random_seq(len(true_seq))
        else:
            s = mutate(true_seq, rate=0.35)
        if s != true_seq:
            decoys.add(s)
    return list(decoys)

def evaluate_sequence_retrieval(model_bundle, holdout_pdb_ids):
    dna = pd.read_csv(os.path.join(OUTPUT_DIR, "dna_sequences.csv"))
    prot = pd.read_csv(os.path.join(OUTPUT_DIR, "protein_sequences.csv"))

    dna["pdb_id"] = dna["pdb_id"].astype(str)
    prot["pdb_id"] = prot["pdb_id"].astype(str)

    hold_ids = set(map(str, holdout_pdb_ids))
    dna = dna[dna["pdb_id"].isin(hold_ids)].copy()
    prot = prot[prot["pdb_id"].isin(hold_ids)].copy()

    if dna.empty or prot.empty:
        return

    prot_merged = prot.groupby("pdb_id", as_index=False).agg(sequence=("sequence", lambda x: "".join(map(str, x))))
    eval_rows = []

    for _, drow in dna.iterrows():
        pdb_id = str(drow["pdb_id"])
        prow = prot_merged[prot_merged["pdb_id"] == pdb_id]
        if prow.empty:
            continue
        protein_seq = str(prow.iloc[0]["sequence"])
        true_seq = str(drow["sequence"])

        true_score = score_candidate_sequence(true_seq, protein_seq, model_bundle, train_seed_sets=None)["score"]
        decoys = make_decoys(true_seq, n=49)
        decoy_scores = [score_candidate_sequence(s, protein_seq, model_bundle, train_seed_sets=None)["score"] for s in decoys]
        all_scores = [true_score] + decoy_scores
        rank = 1 + sum(s > true_score for s in decoy_scores)
        z = (true_score - np.mean(decoy_scores)) / (np.std(decoy_scores) + 1e-8)
        eval_rows.append({
            "pdb_id": pdb_id,
            "true_sequence": true_seq,
            "true_score": true_score,
            "decoy_mean": float(np.mean(decoy_scores)),
            "decoy_std": float(np.std(decoy_scores)),
            "z_score": float(z),
            "rank_among_50": int(rank),
            "top1": int(rank == 1),
            "top5": int(rank <= 5),
            "top10": int(rank <= 10),
        })

    if not eval_rows:
        return

    seq_eval = pd.DataFrame(eval_rows)
    seq_eval.to_csv(os.path.join(OUTPUT_DIR, "heldout_sequence_retrieval.csv"), index=False)
    summary = pd.DataFrame([{
        "n_complexes": len(seq_eval),
        "top1_rate": float(seq_eval["top1"].mean()),
        "top5_rate": float(seq_eval["top5"].mean()),
        "top10_rate": float(seq_eval["top10"].mean()),
        "mean_rank_among_50": float(seq_eval["rank_among_50"].mean()),
        "median_z_score": float(seq_eval["z_score"].median()),
    }])
    summary.to_csv(os.path.join(OUTPUT_DIR, "heldout_sequence_retrieval_summary.csv"), index=False)

    plt.figure()
    sns.histplot(seq_eval["rank_among_50"], bins=20)
    plt.title("Held-out true aptamer rank among 49 decoys")
    plt.xlabel("Rank (1 is best)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "heldout_sequence_retrieval_rank_hist.tiff"))
    plt.close()

# ------------------------------------------------------------
# In-silico SELEX / candidate generation
# ------------------------------------------------------------

def kmer_set(seq, k=3):
    return {seq[i:i+k] for i in range(max(0, len(seq)-k+1))}

def jaccard(a, b):
    return len(a & b) / max(1, len(a | b)) if (a or b) else 0.0

def diversify_sorted(df, keep_n=200, max_jacc=DIVERSITY_MAX_JACCARD):
    kept = []
    seen = []
    for _, row in df.iterrows():
        if len(kept) >= keep_n:
            break
        s = row["sequence"]
        ks = kmer_set(s, DIVERSITY_K)
        if all(jaccard(ks, prev) <= max_jacc for prev in seen):
            kept.append(row.to_dict())
            seen.append(ks)
    return pd.DataFrame(kept)

def crossover(a, b):
    L = min(len(a), len(b))
    if L < 6:
        return a
    cut = int(rng.integers(1, L - 1))
    return a[:cut] + b[cut:L]

def mutate_with_indels(seq, base_rate, indel_rate=SELEX_INDEL_RATE):
    seq = list(seq)
    for i in range(len(seq)):
        if rng.random() < base_rate:
            cur = seq[i]
            seq[i] = rng.choice([b for b in "ACGT" if b != cur])
    if rng.random() < indel_rate and len(seq) < SELEX_LEN_MAX:
        pos = int(rng.integers(0, len(seq) + 1))
        seq.insert(pos, str(rng.choice(NUCLEOTIDES)))
    if rng.random() < indel_rate and len(seq) > SELEX_LEN_MIN:
        pos = int(rng.integers(0, len(seq)))
        seq.pop(pos)
    return "".join(seq)

def estimate_base_probs(seed_sequences):
    counts = Counter()
    total = 0
    for s in seed_sequences:
        counts.update(list(str(s).strip().upper()))
        total += len(str(s).strip())
    if total == 0:
        return np.array([0.25, 0.25, 0.25, 0.25], dtype=float)
    probs = np.array([counts.get(b, 0) for b in ["A", "C", "G", "T"]], dtype=float)
    probs = (probs + 1.0) / (probs.sum() + 4.0)
    return probs / probs.sum()

def random_seq_biased(L, base_probs):
    return "".join(rng.choice(NUCLEOTIDES, size=L, p=base_probs))

def load_training_seed_sequences(train_pdb_ids):
    path = os.path.join(OUTPUT_DIR, "dna_sequences.csv")
    if not os.path.exists(path):
        return []
    dna = pd.read_csv(path)
    dna["pdb_id"] = dna["pdb_id"].astype(str)
    seeds = dna[dna["pdb_id"].isin(set(map(str, train_pdb_ids)))]["sequence"].dropna().astype(str).tolist()
    seeds = [s.strip().upper() for s in seeds if len(s.strip()) >= 8 and set(s.strip().upper()) <= set("ACGT")]
    return sorted(list(set(seeds)))

def initial_pool(seed_sequences):
    base_probs = estimate_base_probs(seed_sequences)
    pool = []
    target_seeded = int(SELEX_POOL_SIZE * SELEX_SEED_FROM_TRAIN_FRAC)

    # Seeded exploitation from known aptamer families, but mutated so the search is not a memorizer.
    while len(pool) < target_seeded and seed_sequences:
        parent = str(rng.choice(seed_sequences))
        if len(parent) > SELEX_LEN_MAX:
            start = int(rng.integers(0, len(parent) - SELEX_LEN_MAX + 1))
            parent = parent[start:start + SELEX_LEN_MAX]
        if len(parent) < SELEX_LEN_MIN:
            parent = parent + random_seq_biased(SELEX_LEN_MIN - len(parent), base_probs)
        child = mutate_with_indels(parent, base_rate=0.12, indel_rate=0.10)
        child = child[:SELEX_LEN_MAX]
        if len(child) < SELEX_LEN_MIN:
            child += random_seq_biased(SELEX_LEN_MIN - len(child), base_probs)
        gc = (child.count("G") + child.count("C")) / max(1, len(child))
        if 0.28 <= gc <= 0.78 and not is_low_complexity(child):
            pool.append(child)

    # De novo exploration.
    while len(pool) < SELEX_POOL_SIZE:
        L = int(rng.integers(SELEX_LEN_MIN, SELEX_LEN_MAX + 1))
        s = random_seq_biased(L, base_probs) if rng.random() < 0.65 else random_seq(L)
        gc = (s.count("G") + s.count("C")) / max(1, len(s))
        if 0.28 <= gc <= 0.78 and not is_low_complexity(s):
            pool.append(s)
    return pool

def select_parents_weighted(df, n_needed):
    if df.empty:
        return []
    w = df["selection_weight"].values.astype(float)
    w = np.maximum(w, 1e-8)
    w = w / w.sum()
    idx = rng.choice(np.arange(len(df)), size=n_needed, replace=True, p=w)
    return df.iloc[idx]["sequence"].tolist()

def rescore_pool(pool, protein_sequence, model_bundle, train_seed_sets):
    unique_pool = list(dict.fromkeys(pool))
    rows = [score_candidate_sequence(s, protein_sequence, model_bundle, train_seed_sets=train_seed_sets) for s in unique_pool]
    df = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    if df.empty:
        return df
    rank = np.arange(1, len(df) + 1, dtype=float)
    df["selection_weight"] = 1.0 / np.sqrt(rank)
    # exploration lane favors high-quality but uncertain candidates
    df["explore_weight"] = 0.65 * (df["selection_weight"] / df["selection_weight"].max()) + 0.35 * (df["model_uncertainty"] / max(1e-12, df["model_uncertainty"].max()))
    return df

def finalize_top_candidates(df, train_seed_sets, n_out=NUM_APTAMERS):
    if df.empty:
        return df
    work = df.sort_values(["score", "position_coverage", "model_mean"], ascending=False).reset_index(drop=True)
    chosen = []
    chosen_sets = []
    for _, row in work.iterrows():
        if len(chosen) >= n_out:
            break
        ks = kmer_set(row["sequence"], DIVERSITY_K)
        if any(jaccard(ks, prev) > DIVERSITY_MAX_JACCARD for prev in chosen_sets):
            continue
        if train_seed_sets:
            max_j_train = max((jaccard(ks, prev) for prev in train_seed_sets), default=0.0)
            if max_j_train >= 0.98:
                continue
        chosen.append(row.to_dict())
        chosen_sets.append(ks)
    out = pd.DataFrame(chosen)
    if len(out) < n_out:
        extra = work[~work["sequence"].isin(out["sequence"].tolist())].head(n_out - len(out))
        out = pd.concat([out, extra], ignore_index=True)
    out = out.head(n_out).copy()
    out["rank"] = np.arange(1, len(out) + 1)
    return out

def insilico_selex(protein_sequence, model_bundle, train_seed_sequences=None):
    train_seed_sequences = train_seed_sequences or []
    train_seed_sets = [kmer_set(s, DIVERSITY_K) for s in train_seed_sequences]
    pool = initial_pool(train_seed_sequences)
    round_rows = []
    by_round = []
    operator_rows = []

    for r in range(1, SELEX_ROUNDS + 1):
        round_mut_rate = max(SELEX_MIN_MUT_RATE, SELEX_MUT_RATE * (0.82 ** (r - 1)))
        print(f"[SELEX] Round {r}/{SELEX_ROUNDS}: scoring {len(pool)} candidates | mut_rate={round_mut_rate:.4f}")
        df = rescore_pool(pool, protein_sequence, model_bundle, train_seed_sets)

        keep_n = max(50, int(len(df) * SELEX_KEEP_FRAC))
        elite_n = max(5, int(len(df) * SELEX_ELITE_FRAC))
        explore_n = max(10, int(len(df) * SELEX_EXPLORATION_FRAC))

        top = df.head(keep_n).copy()
        elites = top.head(elite_n).copy()
        top_div = diversify_sorted(top, keep_n=keep_n, max_jacc=DIVERSITY_MAX_JACCARD)
        explore_df = df.sort_values(["explore_weight", "score"], ascending=False).head(explore_n).copy()
        top_div["round"] = r
        by_round.append(top_div)

        round_rows.append({
            "round": r,
            "pool_size": len(pool),
            "keep_n_raw": keep_n,
            "elite_n": elite_n,
            "explore_n": explore_n,
            "kept_after_diversity": len(top_div),
            "best_score": float(df.iloc[0]["score"]),
            "mean_top_score": float(top["score"].mean()),
            "mean_top_model_mean": float(top["model_mean"].mean()),
            "mean_top_uncertainty": float(top["model_uncertainty"].mean()),
            "mean_top_coverage": float(top["position_coverage"].mean()),
            "mean_top_gc": float(top["gc_content"].mean()),
            "mean_top_mfe": float(np.nanmean(top["mfe"])),
        })

        parents_df = pd.concat([elites, top_div, explore_df], ignore_index=True).drop_duplicates(subset=["sequence"]).reset_index(drop=True)
        if parents_df.empty:
            raise RuntimeError("SELEX stopped: no parents survived scoring/diversity filters.")

        parent_sequences = parents_df["sequence"].tolist()
        parent_dict = {row["sequence"]: row for _, row in parents_df.iterrows()}

        children = []
        # Elitism: carry the best forward unchanged.
        children.extend(elites["sequence"].tolist())

        n_cross = max(1, int(round(SELEX_OFFSPRING_PER_PARENT * SELEX_CROSSOVER_FRAC)))
        n_mut = max(1, SELEX_OFFSPRING_PER_PARENT - n_cross)

        exploit_parents = select_parents_weighted(parents_df.sort_values("selection_weight", ascending=False), len(parent_sequences))
        explore_parents = select_parents_weighted(parents_df.sort_values("explore_weight", ascending=False), len(parent_sequences))

        # Exploitation lane
        for p in exploit_parents:
            for _ in range(n_cross):
                mate = str(rng.choice(parent_sequences))
                c = crossover(p, mate)
                c = mutate_with_indels(c, base_rate=round_mut_rate, indel_rate=SELEX_INDEL_RATE)
                children.append(c)
                operator_rows.append({"round": r, "operator": "exploit_crossover"})
            for _ in range(n_mut):
                c = mutate_with_indels(p, base_rate=round_mut_rate, indel_rate=SELEX_INDEL_RATE)
                children.append(c)
                operator_rows.append({"round": r, "operator": "exploit_mutation"})

        # Exploration lane: use more uncertain parents and stronger mutation.
        for p in explore_parents[: max(10, len(explore_parents)//2)]:
            c = mutate_with_indels(p, base_rate=min(0.18, round_mut_rate * 1.6), indel_rate=min(0.16, SELEX_INDEL_RATE * 1.5))
            children.append(c)
            operator_rows.append({"round": r, "operator": "explore_mutation"})

        # Random injection each round prevents premature convergence.
        inject_n = max(10, int(SELEX_POOL_SIZE * SELEX_RANDOM_INJECTION_FRAC))
        base_probs = estimate_base_probs(train_seed_sequences)
        for _ in range(inject_n):
            L = int(rng.integers(SELEX_LEN_MIN, SELEX_LEN_MAX + 1))
            c = random_seq_biased(L, base_probs) if rng.random() < 0.70 else random_seq(L)
            children.append(c)
            operator_rows.append({"round": r, "operator": "random_injection"})

        next_pool = []
        for s in children:
            s = str(s).upper()
            if len(s) < SELEX_LEN_MIN:
                continue
            if len(s) > SELEX_LEN_MAX:
                s = s[:SELEX_LEN_MAX]
            gc = (s.count("G") + s.count("C")) / max(1, len(s))
            if 0.24 <= gc <= 0.82 and set(s) <= set("ACGT"):
                next_pool.append(s)
        # Add a few parents back if needed.
        next_pool.extend(parent_sequences)
        rng.shuffle(next_pool)
        # Preserve uniqueness but maintain pool size.
        next_pool = list(dict.fromkeys(next_pool))
        if len(next_pool) < SELEX_POOL_SIZE:
            base_probs = estimate_base_probs(train_seed_sequences)
            while len(next_pool) < SELEX_POOL_SIZE:
                L = int(rng.integers(SELEX_LEN_MIN, SELEX_LEN_MAX + 1))
                next_pool.append(random_seq_biased(L, base_probs))
        pool = next_pool[:SELEX_POOL_SIZE]

    round_df = pd.DataFrame(round_rows)
    round_df.to_csv(os.path.join(OUTPUT_DIR, "selex_round_summary.csv"), index=False)
    if operator_rows:
        pd.DataFrame(operator_rows).groupby(["round", "operator"], as_index=False).size().rename(columns={"size": "count"}).to_csv(
            os.path.join(OUTPUT_DIR, "selex_operator_summary.csv"), index=False
        )

    all_top = pd.concat(by_round, ignore_index=True) if by_round else pd.DataFrame()
    all_top.to_csv(os.path.join(OUTPUT_DIR, "selex_top_candidates_by_round.csv"), index=False)

    plt.figure()
    plt.plot(round_df["round"], round_df["best_score"], marker="o")
    plt.title("SELEX convergence: best candidate score")
    plt.xlabel("Round")
    plt.ylabel("Best score")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "selex_convergence_bestscore.tiff"))
    plt.close()

    plt.figure()
    plt.plot(round_df["round"], round_df["mean_top_score"], marker="o")
    plt.title("SELEX convergence: mean top score")
    plt.xlabel("Round")
    plt.ylabel("Mean top score")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "selex_convergence_meantop.tiff"))
    plt.close()

    plt.figure()
    plt.plot(round_df["round"], round_df["mean_top_uncertainty"], marker="o")
    plt.title("SELEX convergence: mean top uncertainty")
    plt.xlabel("Round")
    plt.ylabel("Mean uncertainty")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "selex_convergence_uncertainty.tiff"))
    plt.close()

    final = rescore_pool(pool, protein_sequence, model_bundle, train_seed_sets)
    final_div = finalize_top_candidates(final, train_seed_sets, n_out=NUM_APTAMERS)
    final_div.to_csv(os.path.join(OUTPUT_DIR, "predicted_aptamers_top50.csv"), index=False)

    top3 = finalize_top_candidates(final_div.sort_values(["score", "position_coverage", "model_mean"], ascending=False), train_seed_sets, n_out=3)
    top3.to_csv(os.path.join(OUTPUT_DIR, "selex_top3_final.csv"), index=False)
    with open(os.path.join(OUTPUT_DIR, "selex_top3_report.txt"), "w") as f:
        f.write("Top-3 aptamer candidates (v8 ruthless SELEX)\n")
        f.write("==========================================\n")
        f.write("Ranking emphasizes calibrated ensemble interaction score, positional coverage, structural plausibility, low uncertainty, and diversity.\n")
        f.write("Near-duplicates of training aptamers are penalized rather than blindly recycled.\n\n")
        f.write(top3.to_string(index=False))

    plt.figure(figsize=(11, 8))
    top10 = final_div.head(10).sort_values("score", ascending=True)
    plt.barh(top10["sequence"], top10["score"])
    plt.title("Top 10 predicted aptamer candidates")
    plt.xlabel("Composite specificity score")
    plt.ylabel("Aptamer sequence")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "predicted_aptamers_top10.tiff"))
    plt.close()

    return final_div, top3

# ------------------------------------------------------------
# Result bundling
# ------------------------------------------------------------

def zip_results():
    zpath = os.path.join(OUTPUT_DIR, "aptamer_results.zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for fn in os.listdir(OUTPUT_DIR):
            if fn.endswith((".csv", ".tiff", ".txt", ".pkl")):
                z.write(os.path.join(OUTPUT_DIR, fn), arcname=fn)
    return zpath

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the aptamer pipeline from PDB complexes and predict aptamers for a target protein."
    )
    parser.add_argument("--pdb_dir", default=None, help="Folder containing .pdb/.ent files. Defaults to ./PDB_files or ./PDBS if present.")
    parser.add_argument("--out_dir", default=None, help="Output folder. Defaults to ./results")
    parser.add_argument("--protein_seq", default=None, help="Target protein sequence in one-letter amino-acid code.")
    parser.add_argument("--no_rnafold", action="store_true", help="Disable RNAfold and use the internal secondary-structure proxy.")
    parser.add_argument("--auto_install_viennarna", action="store_true", help="Try to install ViennaRNA automatically on Linux via apt-get.")
    return parser.parse_args()


def resolve_default_pdb_dir():
    candidates = [
        os.path.join(os.getcwd(), "PDB_files"),
        os.path.join(os.getcwd(), "PDBS"),
    ]
    for cand in candidates:
        if os.path.isdir(cand):
            return cand
    return candidates[0]


def main():
    global PDB_DIR, OUTPUT_DIR, USE_RNAFOLD, AUTO_INSTALL_VIENNARNA

    args = parse_args()
    PDB_DIR = os.path.abspath(args.pdb_dir) if args.pdb_dir else resolve_default_pdb_dir()
    OUTPUT_DIR = os.path.abspath(args.out_dir) if args.out_dir else os.path.join(os.getcwd(), "results")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    USE_RNAFOLD = not args.no_rnafold
    AUTO_INSTALL_VIENNARNA = bool(args.auto_install_viennarna) and (os.name != "nt")

    ensure_rnafold_available()

    pdb_files = discover_pdb_files()
    print(f"[INFO] Found {len(pdb_files)} PDB files in {PDB_DIR}")
    if not pdb_files:
        raise RuntimeError(f"No PDB files found under {PDB_DIR}")

    interaction_csv = os.path.join(OUTPUT_DIR, "interaction_data.csv")

    print("[STEP] Extracting sequences, interactions, and secondary structures ...")
    processed = generate_interaction_dataset_from_pdbs(pdb_files, interaction_csv)
    if processed == 0:
        raise RuntimeError("No complexes were processed successfully. Check skip_log.csv")

    print("[STEP] Building specificity matrices for nt(1-3) x aa(1-3) ...")
    build_specificity_outputs(interaction_csv)

    print("[STEP] Training leakage-safe interaction model ...")
    train_out = train_eval_save(interaction_csv)

    print("[STEP] Evaluating held-out sequence retrieval specificity ...")
    holdout_ids = train_out["hold_df"]["pdb_id"].astype(str).unique().tolist()
    evaluate_sequence_retrieval(train_out["bundle"], holdout_ids)

    if args.protein_seq:
        protein_sequence = args.protein_seq.strip().upper()
    else:
        protein_sequence = input(
            "Enter the target protein sequence or motif/domain sequence (AA letters only): "
        ).strip().upper()
    if (not protein_sequence) or (not all(a in VALID_AA_SET for a in protein_sequence)):
        raise ValueError("Invalid protein sequence. Use only standard amino-acid letters: ACDEFGHIKLMNPQRSTVWY")

    print("[STEP] Running in-silico SELEX / candidate generation ...")
    train_seed_sequences = load_training_seed_sequences(train_out["train_pdb_ids"])
    predicted_df, top3 = insilico_selex(protein_sequence, train_out["bundle"], train_seed_sequences=train_seed_sequences)

    zpath = zip_results()

    print("\n" + "=" * 88)
    print("APTAMER PIPELINE COMPLETED (v8)")
    print("=" * 88)
    print(f"Processed complexes: {processed}")
    print(f"Interaction data: {interaction_csv}")
    print(f"Performance summary: {os.path.join(OUTPUT_DIR, 'performance_summary.csv')}")
    print(f"Held-out report: {os.path.join(OUTPUT_DIR, 'heldout_report.txt')}")
    print(f"Sequence retrieval: {os.path.join(OUTPUT_DIR, 'heldout_sequence_retrieval_summary.csv')}")
    print(f"Top-50 aptamers: {os.path.join(OUTPUT_DIR, 'predicted_aptamers_top50.csv')}")
    print(f"Top-3 report: {os.path.join(OUTPUT_DIR, 'selex_top3_report.txt')}")
    print(f"Zip bundle: {zpath}")
    print("=" * 88)

if __name__ == "__main__":
    main()
