"""Age-comparable label crosswalk between adult and pediatric ECG label schemes.

Only rhythm/conduction labels that are *defined the same way across age* enter
the primary transfer analysis; age-specific pediatric labels (congenital, etc.)
are held out as pediatric-only targets. Each harmonised label lists the source
codes that map onto it in PTB-XL (SCP), PhysioNet/CinC (SNOMED) and ZZU-pECG
(statement keywords / ICD-10 prefixes).
"""
from __future__ import annotations

# Shared, age-comparable labels.
SHARED_LABELS = [
    "SR",       # sinus rhythm
    "STach",    # sinus tachycardia
    "SBrad",    # sinus bradycardia
    "AF",       # atrial fibrillation
    "AFL",      # atrial flutter
    "1AVB",     # first-degree AV block
    "RBBB",     # right bundle branch block (complete/incomplete)
    "LBBB",     # left bundle branch block / IVCD
    "PVC",      # ventricular premature complex
    "PAC",      # atrial premature complex
]

# Pediatric-only targets (held out; not used for adult->pediatric transfer).
PEDIATRIC_ONLY = ["CHD", "myocarditis", "cardiomyopathy", "kawasaki", "WPW_pediatric"]

# adult PTB-XL SCP codes -> harmonised label
PTBXL_SCP = {
    "SR": "SR", "STACH": "STach", "SBRAD": "SBrad", "AFIB": "AF", "AFLT": "AFL",
    "1AVB": "1AVB", "CRBBB": "RBBB", "IRBBB": "RBBB", "CLBBB": "LBBB", "ILBBB": "LBBB",
    "IVCD": "LBBB", "PVC": "PVC", "PAC": "PAC",
}

# PhysioNet/CinC SNOMED-CT -> harmonised label (subset)
SNOMED = {
    "426783006": "SR", "427084000": "STach", "426177001": "SBrad",
    "164889003": "AF", "164890007": "AFL", "270492004": "1AVB",
    "59118001": "RBBB", "713427006": "RBBB", "733534002": "LBBB",
    "164909002": "LBBB", "17338001": "PVC", "284470004": "PAC",
}

# ZZU-pECG statement keywords -> harmonised label (case-insensitive substring)
ZZU_KEYWORDS = {
    "sinus rhythm": "SR", "sinus tachycardia": "STach", "sinus bradycardia": "SBrad",
    "atrial fibrillation": "AF", "atrial flutter": "AFL",
    "first degree": "1AVB", "first-degree": "1AVB",
    "right bundle": "RBBB", "left bundle": "LBBB", "intraventricular conduction": "LBBB",
    "ventricular premature": "PVC", "premature ventricular": "PVC",
    "atrial premature": "PAC", "premature atrial": "PAC",
}


def map_ptbxl(scp_code: str) -> str | None:
    return PTBXL_SCP.get(str(scp_code).upper())


def map_snomed(code: str) -> str | None:
    return SNOMED.get(str(code).strip())


def map_zzu_statement(text: str) -> list[str]:
    """Map a free-text pediatric statement to any harmonised labels it mentions."""
    if not isinstance(text, str):
        return []
    t = text.lower()
    found = []
    for kw, lab in ZZU_KEYWORDS.items():
        if kw in t and lab not in found:
            found.append(lab)
    return found


def is_shared(label: str) -> bool:
    return label in SHARED_LABELS
