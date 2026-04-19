import re
import pandas as pd
import numpy as np
import joblib
import json
import os
import spacy
import warnings
from itertools import combinations
warnings.filterwarnings('ignore')

MODELS    = os.path.join('models')
PROCESSED = os.path.join('data', 'processed')

MIN_PRR_ADR   = 2.0
MIN_PRR_DDI   = 3.0
MIN_FREQ_KNOWN = 3

# ═══════════════════════════════════════════════════════════════
# STEP 1 — DRUG SYNONYM MAP
# ═══════════════════════════════════════════════════════════════
DRUG_SYNONYMS = {
    'paracetamol': 'acetaminophen', 'calpol': 'acetaminophen',
    'dolo': 'acetaminophen', 'crocin': 'acetaminophen',
    'tylenol': 'acetaminophen', 'disprin': 'aspirin',
    'ecosprin': 'aspirin', 'loprin': 'aspirin', 'colsprin': 'aspirin',
    'brufen': 'ibuprofen', 'combiflam': 'ibuprofen',
    'advil': 'ibuprofen', 'nurofen': 'ibuprofen',
    'adrenaline': 'epinephrine', 'lignocaine': 'lidocaine',
    'frusemide': 'furosemide', 'salbutamol': 'albuterol',
    'pethidine': 'meperidine', 'glibenclamide': 'glyburide',
    'noradrenaline': 'norepinephrine', 'amoxycillin': 'amoxicillin',
    'augmentin': 'amoxicillin', 'zithromax': 'azithromycin',
    'azee': 'azithromycin', 'taxim': 'cefotaxime',
    'monocef': 'ceftriaxone', 'pan': 'pantoprazole',
    'pantocid': 'pantoprazole', 'rantac': 'ranitidine',
    'stamlo': 'amlodipine', 'amloz': 'amlodipine',
    'lipitor': 'atorvastatin', 'storvas': 'atorvastatin',
    'glucophage': 'metformin', 'glycomet': 'metformin',
    'januvia': 'sitagliptin', 'telma': 'telmisartan',
    'olsar': 'olmesartan', 'betaloc': 'metoprolol',
    'lasilactone': 'furosemide', 'lasix': 'furosemide',
    'aldactone': 'spironolactone', 'lanoxin': 'digoxin',
    'coumadin': 'warfarin', 'plavix': 'clopidogrel',
    'ziaha': 'clopidogrel', 'zyloric': 'allopurinol',
    'lopressor': 'metoprolol', 'cordarone': 'amiodarone',
    'nexium': 'esomeprazole', 'omez': 'omeprazole',
    'prilosec': 'omeprazole', 'zantac': 'ranitidine',
    'ventolin': 'albuterol', 'asthalin': 'albuterol',
    'deriphyllin': 'theophylline',
    'monotrate': 'isosorbide mononitrate',
    'sorbitrate': 'isosorbide dinitrate',
    'gluconorm': 'repaglinide', 'amaryl': 'glimepiride',
    'zoryl': 'glimepiride', 'insulatard': 'insulin',
    'huminsulin': 'insulin', 'mixtard': 'insulin',
    'volini': 'diclofenac', 'voveran': 'diclofenac',
    'clofen': 'diclofenac', 'zerodol': 'aceclofenac',
    'ultracet': 'tramadol', 'nucoxia': 'etoricoxib',
    'osteofos': 'alendronate', 'shelcal': 'calcium carbonate',
    'calcirol': 'cholecalciferol',
}

def normalize_drug_name(name: str) -> str:
    n = name.lower().strip()
    return DRUG_SYNONYMS.get(n, n)

_SUFFIX_RE = re.compile(
    r',?\s*(usp|bp|ep|ip|hcl|hydrochloride|sodium|potassium|calcium|' +
    r'magnesium|sulfate|sulphate|phosphate|acetate|citrate|tartrate|' +
    r'maleate|fumarate|succinate|chloride|bromide|nitrate|carbonate|' +
    r'mesylate|besylate|gluconate|lactate|monohydrate|dihydrate|' +
    r'anhydrous|sterile|injection|oral|tablet|capsule|solution|' +
    r'suspension|cream|gel|extended.release|er|sr|xr|la|cr)\b.*$',
    re.IGNORECASE
)

def normalize_offsides_name(name: str) -> str:
    """Strip pharma suffixes then apply synonym map."""
    n = name.lower().strip()
    n = _SUFFIX_RE.sub('', n).strip().rstrip(',').strip()
    return DRUG_SYNONYMS.get(n, n)

# ═══════════════════════════════════════════════════════════════
# STEP 2 — CURATED FREQUENCIES (last-resort fallback)
# ═══════════════════════════════════════════════════════════════
CURATED_FREQUENCIES = {
    'acetaminophen': [
        ('nausea', 0.05), ('vomiting', 0.03), ('abdominal pain', 0.02),
        ('hepatotoxicity', 0.01), ('hepatic necrosis', 0.005),
        ('liver injury', 0.005), ('rash', 0.005), ('pruritus', 0.003),
        ('urticaria', 0.002), ('anaphylactic shock', 0.001),
        ('agranulocytosis', 0.0001), ('thrombocytopenia', 0.0001),
        ('haemolytic anaemia', 0.0001),
        ('stevens-johnson syndrome', 0.0001),
    ],
    'ibuprofen': [
        ('nausea', 0.05), ('dyspepsia', 0.05), ('abdominal pain', 0.04),
        ('diarrhoea', 0.03), ('vomiting', 0.03), ('gastric ulcer', 0.02),
        ('gastrointestinal haemorrhage', 0.01), ('headache', 0.02),
        ('dizziness', 0.01), ('oedema peripheral', 0.01),
        ('rash', 0.005), ('renal impairment', 0.005),
        ('urticaria', 0.002), ('anaphylactic shock', 0.001),
        ('bronchospasm', 0.001),
    ],
    'aspirin': [
        ('nausea', 0.05), ('dyspepsia', 0.05), ('abdominal pain', 0.04),
        ('gastrointestinal haemorrhage', 0.03), ('gastric ulcer', 0.02),
        ('tinnitus', 0.03), ('bronchospasm', 0.01),
        ('rash', 0.005), ('thrombocytopenia', 0.001),
        ('angioedema', 0.001),
    ],
    'metformin': [
        ('nausea', 0.10), ('diarrhoea', 0.10), ('vomiting', 0.06),
        ('abdominal pain', 0.05), ('decreased appetite', 0.04),
        ('dysgeusia', 0.03), ('lactic acidosis', 0.00003),
        ('vitamin b12 deficiency', 0.01),
    ],
    'warfarin': [
        ('haemorrhage', 0.10), ('ecchymosis', 0.05),
        ('epistaxis', 0.03), ('haematuria', 0.02),
        ('gastrointestinal haemorrhage', 0.02),
        ('intracranial haemorrhage', 0.01),
        ('skin necrosis', 0.005), ('alopecia', 0.005),
        ('nausea', 0.03), ('rash', 0.002),
    ],
    'amoxicillin': [
        ('diarrhoea', 0.10), ('nausea', 0.05), ('rash', 0.05),
        ('urticaria', 0.03), ('vomiting', 0.03),
        ('anaphylactic shock', 0.001),
        ('stevens-johnson syndrome', 0.0001),
        ('pseudomembranous colitis', 0.01), ('candidiasis', 0.03),
    ],
    'atorvastatin': [
        ('myalgia', 0.05), ('diarrhoea', 0.03), ('nausea', 0.03),
        ('headache', 0.02), ('rhabdomyolysis', 0.001),
        ('hepatotoxicity', 0.005), ('insomnia', 0.01),
        ('arthralgia', 0.02),
    ],
    'omeprazole': [
        ('headache', 0.05), ('diarrhoea', 0.04), ('nausea', 0.04),
        ('abdominal pain', 0.03), ('vomiting', 0.02),
        ('flatulence', 0.03), ('hypomagnesaemia', 0.01),
        ('vitamin b12 deficiency', 0.01),
        ('clostridium difficile infection', 0.005),
    ],
    'amlodipine': [
        ('oedema peripheral', 0.10), ('headache', 0.07),
        ('fatigue', 0.04), ('nausea', 0.03), ('flushing', 0.03),
        ('dizziness', 0.03), ('palpitations', 0.02),
        ('abdominal pain', 0.02), ('rash', 0.01),
    ],
    'metoprolol': [
        ('fatigue', 0.10), ('bradycardia', 0.05), ('dizziness', 0.05),
        ('headache', 0.04), ('dyspnoea', 0.03), ('nausea', 0.03),
        ('cold extremities', 0.02), ('hypotension', 0.02),
        ('bronchospasm', 0.01),
    ],
    'clopidogrel': [
        ('haemorrhage', 0.08), ('ecchymosis', 0.05),
        ('nausea', 0.03), ('diarrhoea', 0.04),
        ('abdominal pain', 0.03), ('epistaxis', 0.03),
        ('rash', 0.02), ('neutropenia', 0.001),
    ],
    # ── Antihistamines (H1 blockers) ──────────────────────────────────────
    # Source: FDA prescribing labels; Simons & Simons (2011) Lancet
    'cetirizine': [
        ('somnolence',          0.14),  # #1 ADR — dose-dependent sedation
        ('headache',            0.07),
        ('dry mouth',           0.05),
        ('fatigue',             0.03),
        ('dizziness',           0.02),
        ('nausea',              0.02),
        ('pharyngitis',         0.02),
        ('abdominal pain',      0.01),
        ('urticaria',           0.005),  # paradoxical (rare)
        ('anaphylactic shock',  0.001),
    ],
    'loratadine': [
        ('headache',            0.06),
        ('somnolence',          0.03),
        ('dry mouth',           0.03),
        ('fatigue',             0.02),
        ('nausea',              0.02),
        ('dizziness',           0.01),
        ('anaphylactic shock',  0.001),
    ],
    'fexofenadine': [
        ('headache',            0.10),
        ('nausea',              0.05),
        ('dizziness',           0.02),
        ('fatigue',             0.02),
        ('somnolence',          0.01),
        ('back pain',           0.02),
        ('anaphylactic shock',  0.001),
    ],
    'diphenhydramine': [
        ('somnolence',          0.40),
        ('dry mouth',           0.15),
        ('dizziness',           0.10),
        ('urinary retention',   0.05),
        ('constipation',        0.05),
        ('confusion',           0.03),
        ('tachycardia',         0.02),
        ('blurred vision',      0.02),
    ],
    'chlorphenamine': [
        ('somnolence',          0.25),
        ('dry mouth',           0.10),
        ('dizziness',           0.05),
        ('urinary retention',   0.03),
        ('constipation',        0.03),
        ('tachycardia',         0.02),
    ],
    # ── Common antibiotics ────────────────────────────────────────────────
    'azithromycin': [
        ('nausea',              0.07),
        ('diarrhoea',           0.05),
        ('abdominal pain',      0.03),
        ('vomiting',            0.03),
        ('qt prolongation',     0.01),
        ('hepatotoxicity',      0.005),
    ],
    'ciprofloxacin': [
        ('nausea',              0.05),
        ('diarrhoea',           0.04),
        ('headache',            0.03),
        ('dizziness',           0.02),
        ('qt prolongation',     0.02),
        ('tendinopathy',        0.01),
        ('peripheral neuropathy', 0.003),
        ('photosensitivity',    0.01),
    ],
    # ── Antidiabetics ─────────────────────────────────────────────────────
    'sitagliptin': [
        ('nasopharyngitis',     0.07),
        ('headache',            0.03),
        ('nausea',              0.02),
        ('hypoglycaemia',       0.01),
        ('pancreatitis',        0.001),
    ],
    'glimepiride': [
        ('hypoglycaemia',       0.15),
        ('nausea',              0.03),
        ('dizziness',           0.02),
        ('headache',            0.02),
        ('weight increased',    0.02),
    ],
    # ── Antihypertensives ─────────────────────────────────────────────────
    'lisinopril': [
        ('cough',               0.10),
        ('dizziness',           0.06),
        ('hypotension',         0.05),
        ('headache',            0.05),
        ('hyperkalaemia',       0.04),
        ('renal impairment',    0.02),
        ('angioedema',          0.003),
    ],
    'telmisartan': [
        ('dizziness',           0.04),
        ('headache',            0.03),
        ('diarrhoea',           0.03),
        ('hyperkalaemia',       0.02),
        ('back pain',           0.02),
    ],
    'furosemide': [
        ('hypokalaemia', 0.10), ('hyponatraemia', 0.05),
        ('dehydration', 0.05), ('hypotension', 0.05),
        ('dizziness', 0.04), ('nausea', 0.03),
        ('hyperuricaemia', 0.05), ('ototoxicity', 0.01),
        ('muscle cramps', 0.04),
    ],
}

# Corrects inflated FAERS reporting rates to real-world clinical incidence.
FREQ_OVERRIDES = {
    ('metformin', 'lactic acidosis'): 0.00003,
    ('metformin', 'metabolic acidosis'): 0.001,
    ('warfarin', 'intracranial haemorrhage'): 0.01,
    ('amiodarone', 'pulmonary toxicity'): 0.02,
    ('amiodarone', 'thyroid disorder'): 0.05,
    ('clozapine', 'agranulocytosis'): 0.008,
    ('lithium', 'nephrogenic diabetes insipidus'): 0.03,
}

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — NOISE BLACKLISTS (11 semantic categories)
# ══════════════════════════════════════════════════════════════════════════════

# ── Category 1: Self-harm / Misuse / Overdose ─────────────────────────────
_NOISE_SELF_HARM = {
    'completed suicide', 'suicide attempt', 'suicidal ideation',
    'suicidal behaviour', 'self-harm', 'self-injurious behaviour',
    'intentional overdose', 'intentional self-injury',
    'intentional product misuse', 'poisoning deliberate',
    'drug abuse', 'drug dependence', 'drug misuse',
    'substance abuse', 'overdose', 'accidental overdose',
    'product misuse', 'poisoning',
}

# ── Category 2: Administration errors / Product quality ───────────────────
_NOISE_ADMIN_ERROR = {
    'drug toxicity', 'drug ineffective', 'medication error',
    'therapeutic product effect incomplete',
    'wrong drug administered', 'off label use',
    'wrong technique in drug usage process',
    'accidental exposure to product',
    'product use in unapproved indication',
    'administration site reaction', 'injection site reaction',
    'injection site pain', 'injection site swelling',
    'injection site erythema', 'injection site haematoma',
    'injection site bruising', 'injection site inflammation',
    'infusion related reaction', 'infusion site reaction',
    'infusion site pain', 'product quality issue',
    'device malfunction', 'device occlusion',
    'thrombosis in device',
    'no therapeutic response', 'treatment failure',
    'therapeutic response unexpected', 'toxicity to various agents',
    'drug level increased', 'drug level decreased', 'drug interaction',
    'drug dispensing error', 'prescription error',
}

# ── Category 3: Social / Environmental confounders ────────────────────────
_NOISE_SOCIAL = {
    'emotional distress', 'economic problem', 'social problem',
    'homelessness', 'lack of food', 'family stress',
    'work related problem', 'educational problem',
    'social exclusion', 'victim of crime', 'physical assault',
}

# ── Category 4: Physical injuries / Accidents ─────────────────────────────
_NOISE_INJURY = {
    'injury', 'multiple injuries', 'road traffic accident',
    'accidental injury', 'contusion', 'laceration',
    'fall', 'fracture', 'sprain', 'dislocation', 'wound',
}

# ── Category 5: Medical procedures / Hospitalisation context ──────────────
_NOISE_PROCEDURE = {
    'vasodilation procedure', 'blood transfusion',
    'hospitalisation', 'intensive care', 'mechanical ventilation',
    'surgical procedure', 'catheterisation', 'intubation',
    'cardiopulmonary resuscitation', 'endoscopy', 'atelectasis',
}

# ── Category 6: Generic deterioration / Death ─────────────────────────────
_NOISE_DETERIORATION = {
    'general physical health deterioration', 'condition aggravated',
    'disease progression', 'condition worsened',
    'multiple organ dysfunction syndrome', 'death', 'sudden death',
}

# ── Category 7: Background comorbidities (confounding by indication) ──────
_NOISE_COMORBIDITY = {
    # ── Cardiovascular structural comorbidities ────────────────────────
    'deep vein thrombosis', 'pulmonary embolism',
    'acute myocardial infarction', 'myocardial infarction',
    'cerebrovascular accident', 'stroke',
    'cardiac failure', 'cardiac failure congestive',
    'atrial fibrillation', 'atrial flutter',
    'arrhythmia', 'coronary artery disease',
    'coronary artery occlusion',
    'angina pectoris',
    'peripheral arterial occlusive disease',
    # Cardiac structural lesions — patients on anticoagulants HAVE valve disease
    'mitral valve incompetence', 'mitral valve disease',
    'tricuspid valve incompetence', 'tricuspid valve disease',
    'aortic valve incompetence', 'aortic stenosis',
    'valvular heart disease', 'cardio-respiratory arrest',
    'cardiovascular disorder', 'cardiomegaly', 'cardiac hypertrophy',
    'ventricular tachycardia', 'ventricular fibrillation',
    'cardiac arrest', 'pulmonary hypertension',

    # ── Metabolic / Endocrine background ──────────────────────────────
    'diabetes mellitus', 'type 2 diabetes mellitus',
    'diabetes mellitus inadequate control',
    'blood glucose uncontrolled', 'hyperglycaemia',
    'diabetic neuropathy', 'obesity', 'insulin resistance',
    'hyperlipidaemia', 'hypercholesterolaemia',

    # ── Organ disease background ───────────────────────────────────────
    'chronic kidney disease', 'renal failure chronic',
    'renal impairment', 'liver disease',
    'cirrhosis', 'hepatic failure',
    'sepsis', 'respiratory failure',
    'chronic obstructive pulmonary disease', 'pleural effusion',
    'dyspnoea exertional',
    'gastrooesophageal reflux disease',
    'gastro-oesophageal reflux disease', 'reflux oesophagitis',
    'oesophagitis',
    'osteoarthritis', 'osteoporosis', 'arthritis',
    'gallbladder disorder', 'cholecystitis chronic', 'cholelithiasis',
    'biliary colic',
    'thrombosis', 'thromboembolic event',
    'pancreatitis', 'pancreatitis acute', 'pancreatitis chronic',

    # ── Obstetric / Reproductive context ──────────────────────────────
    'breast feeding', 'pregnancy', 'labour',
    'exposure during pregnancy', 'nephrogenic systemic fibrosis',

    # ── Urological background ──────────────────────────────────────────
    'urinary tract infection', 'urinary tract disorder',
    'urethral disorder', 'urinary incontinence',
    'urinary hesitation', 'pollakiuria',

    # ── Diagnosis-as-ADR confounders ──────────────────────────────────
    'epilepsy', 'sinusitis', 'pneumonia',
    'upper respiratory tract infection', 'bronchitis',
    'transient ischaemic attack',

    # Neurological — diabetic neuropathy predates the drug pair
    'neuropathy peripheral',
    'peripheral neuropathy',

    # GI — gastritis is the indication for PPIs, not caused by drug combinations
    'gastritis',

    # Respiratory — allergy/asthma patients take antihistamines;
    # AERD/aspirin-sensitive asthma handled per-drug via curated entries
    'wheezing',
    'asthma',
    'reactive airway disease',
}

# ── Category 8: Lab / Monitoring values (not clinical events) ─────────────
_NOISE_LAB_VALUES = {
    'blood glucose increased', 'blood glucose decreased',
    'glycosylated haemoglobin increased',
    'international normalised ratio increased',
    'international normalised ratio decreased',
    'international normalised ratio abnormal',
    'international normalised ratio fluctuation',
    'coagulation test abnormal', 'prothrombin level abnormal',
    'prothrombin time prolonged', 'prothrombin time shortened',
    'blood creatinine increased', 'blood urea increased',
    'alanine aminotransferase increased',
    'aspartate aminotransferase increased',
    'gamma-glutamyltransferase increased',
    'alkaline phosphatase increased',
    'blood bilirubin increased', 'bilirubin conjugated increased',
    'white blood cell count decreased', 'white blood cell count increased',
    'platelet count decreased', 'platelet count increased',
    'haemoglobin decreased', 'haematocrit decreased',
    'heart rate increased', 'heart rate decreased',
    'blood pressure increased', 'blood pressure decreased',
    'electrocardiogram abnormal', 'electrocardiogram qt prolonged',
    'lipase increased', 'amylase increased',
    'troponin increased', 'troponin t increased',
    'c-reactive protein increased',
    'urine output decreased', 'oxygen saturation decreased',
    'blood cholesterol increased', 'blood cholesterol decreased',
    'triglycerides increased',
    'low density lipoprotein increased',
    'high density lipoprotein decreased',
}

# ── Category 9: Non-specific / vague symptoms ─────────────────────────────
_NOISE_VAGUE = {
    # 'fatigue' REMOVED — real ADR for antihistamines, beta-blockers, chemo
    # 'anxiety' REMOVED — real ADR for SSRIs, stimulants, corticosteroids
    'pain', 'malaise', 'pyrexia',
    'weight increased', 'weight decreased',
    'feeling abnormal', 'feeling cold', 'feeling hot',
    'general symptom', 'discomfort', 'influenza like illness',
    'abdominal distension', 
}

# ── Category 10: Surveillance / Database metadata labels ──────────────────
_NOISE_SURVEILLANCE = {
    'drug exposure during pregnancy',
    'drug administered to patient of inappropriate age',
    'drug administered to patient of inappropriate sex',
    'inappropriate schedule of drug administration',
    'report duplicate', 'lack of efficacy',
}

# ── Category 11: Teratogenicity confounders (DDI-ONLY) ────────────────────
_NOISE_TERATOGENICITY_DDI_ONLY = {
    'hypospadias', 'heart disease congenital',
    'cleft palate', 'cleft lip', 'spina bifida',
    'neural tube defect', 'ventricular septal defect',
    'atrial septal defect', 'fetal malformation',
    'foetal malformation', 'congenital malformation',
    'congenital anomaly', 'neonatal disorder', 'neonatal death',
    'premature baby', 'low birth weight baby', 'small for dates baby',
    'complex partial seizures',
}

# ══════════════════════════════════════════════════════════════════════════════
# COMBINED BLACKLISTS
# ══════════════════════════════════════════════════════════════════════════════
ADR_NOISE_BLACKLIST = (
    _NOISE_SELF_HARM
    | _NOISE_ADMIN_ERROR
    | _NOISE_SOCIAL
    | _NOISE_INJURY
    | _NOISE_PROCEDURE
    | _NOISE_DETERIORATION
    | _NOISE_COMORBIDITY
    | _NOISE_LAB_VALUES
    | _NOISE_VAGUE
    | _NOISE_SURVEILLANCE
)

# ── Category 12: Immune / Hypersensitivity confounders (DDI-ONLY) ─────────
# SJS/TEN are individual drug hypersensitivity reactions — not DDI mechanisms.
# In TWOSIDES they appear as confounders (both drugs have hypersensitivity
# reports in the same patients). Kept in ADR results for individual drugs.
_NOISE_DDI_IMMUNE_CONFOUNDERS = {
    'stevens-johnson syndrome',
    'toxic epidermal necrolysis',
    'drug reaction with eosinophilia and systemic symptoms',
    'drug hypersensitivity',
    'hypersensitivity',
    'anaphylaxis',
    'anaphylactic reaction',
    'angioedema',
}

DDI_NOISE_BLACKLIST = (
    _NOISE_SELF_HARM
    | _NOISE_ADMIN_ERROR
    | _NOISE_SOCIAL
    | _NOISE_INJURY
    | _NOISE_PROCEDURE
    | _NOISE_DETERIORATION
    | _NOISE_COMORBIDITY
    | _NOISE_LAB_VALUES
    | _NOISE_VAGUE
    | _NOISE_SURVEILLANCE
    | _NOISE_TERATOGENICITY_DDI_ONLY
    | _NOISE_DDI_IMMUNE_CONFOUNDERS   # SJS/TEN/anaphylaxis — individual ADRs, not DDI signals
)

def filter_adr_noise(se_list: list, drug_name: str = '') -> list:
    """Remove global noise AND drug-specific indication confounders."""
    drug_specific = _INDICATION_CONFOUNDERS.get(drug_name.lower().strip(), set())
    return [
        se for se in se_list
        if se['name'].lower().strip() not in ADR_NOISE_BLACKLIST
        and se['name'].lower().strip() not in drug_specific
    ]

def filter_ddi_noise(interactions: list) -> list:
    """Remove blacklisted noise entries from DDI interaction list."""
    return [ix for ix in interactions if ix.lower().strip() not in DDI_NOISE_BLACKLIST]

# ═══════════════════════════════════════════════════════════════
# STEP 3b — DRUG-SPECIFIC INDICATION CONFOUNDERS
# These are terms that are NOT global noise but ARE confounders
# for specific drugs because the drug is prescribed FOR them.
# ═══════════════════════════════════════════════════════════════
_INDICATION_CONFOUNDERS = {
    # Antihistamines — treat urticaria/allergy
    'cetirizine':     {'urticaria', 'hypersensitivity', 'allergic rhinitis', 'pruritus'},
    'loratadine':     {'urticaria', 'hypersensitivity', 'allergic rhinitis', 'pruritus'},
    'fexofenadine':   {'urticaria', 'hypersensitivity', 'allergic rhinitis'},
    'diphenhydramine':{'urticaria', 'hypersensitivity', 'pruritus'},
    'chlorphenamine': {'urticaria', 'hypersensitivity', 'pruritus'},
    # Benzodiazepines — treat anxiety/panic
    'alprazolam':  {'panic attack', 'panic disorder', 'nervousness', 'anxiety disorder'},
    'diazepam':    {'panic attack', 'anxiety disorder', 'nervousness'},
    'clonazepam':  {'panic attack', 'anxiety disorder', 'nervousness'},
    'lorazepam':   {'anxiety disorder', 'nervousness'},
    # Antiepileptics — prescribed FOR epilepsy
    'phenytoin':      {'epilepsy', 'seizure disorder'},
    'carbamazepine':  {'epilepsy', 'seizure disorder'},
    'valproate':      {'epilepsy', 'seizure disorder'},
    'levetiracetam':  {'epilepsy', 'seizure disorder'},
    # Antibiotics — prescribed FOR infections
    'amoxicillin':    {'sinusitis', 'urinary tract infection', 'pneumonia', 'otitis media'},
    'ciprofloxacin':  {'urinary tract infection', 'sinusitis'},
    'azithromycin':   {'sinusitis', 'pneumonia', 'bronchitis'},
    # Antacids / PPIs — prescribed FOR GI complaints
    'omeprazole':   {'gastritis', 'dyspepsia', 'gastrooesophageal reflux disease'},
    'pantoprazole': {'gastritis', 'dyspepsia', 'gastrooesophageal reflux disease'},
    'esomeprazole': {'gastritis', 'dyspepsia', 'gastrooesophageal reflux disease'},
    # Antiplatelets — prescribed BECAUSE patient already has CAD/stroke risk
    'clopidogrel': {
        'coronary artery occlusion', 'coronary artery disease',
        'ischaemic stroke', 'transient ischaemic attack',
        'peripheral arterial occlusive disease',
    },
    'aspirin': {
        'coronary artery occlusion', 'coronary artery disease',
        'ischaemic stroke', 'transient ischaemic attack',
    },
    'prasugrel':  {'coronary artery disease', 'ischaemic stroke'},
    'ticagrelor': {'coronary artery disease', 'ischaemic stroke'},
    # Anticoagulants — prescribed FOR AF / VTE
    'warfarin': {
        'atrial fibrillation', 'atrial flutter',
        'deep vein thrombosis', 'pulmonary embolism',
        'mitral valve disease', 'tricuspid valve disease',
    },
    'apixaban':    {'atrial fibrillation', 'deep vein thrombosis', 'pulmonary embolism'},
    'rivaroxaban': {'atrial fibrillation', 'deep vein thrombosis', 'pulmonary embolism'},
    'dabigatran':  {'atrial fibrillation', 'deep vein thrombosis', 'pulmonary embolism'},
}

# ═══════════════════════════════════════════════════════════════
# STEP 3c — PAIR-SPECIFIC DDI WHITELISTS
# Some terms are normally indication confounders but ARE genuine
# DDI signals for specific drug pairs.
# ═══════════════════════════════════════════════════════════════

# PPI class drugs (CYP2C19 inhibitors)
_PPI_CLASS = {
    'omeprazole', 'esomeprazole', 'pantoprazole',
    'lansoprazole', 'rabeprazole', 'dexlansoprazole',
}
# Antiplatelet drugs activated via CYP2C19
_ANTIPLATELET_CLASS = {'clopidogrel', 'prasugrel', 'ticagrelor', 'ticlopidine'}

# Terms whitelisted for PPI + antiplatelet pairs ONLY.
# FDA issued a safety communication: omeprazole inhibits CYP2C19
# → reduces clopidogrel activation → more MI/angina events.
_PPI_ANTIPLATELET_WHITELIST = {
    'acute myocardial infarction',
    'myocardial infarction',
    'angina unstable',
    'stent thrombosis',
    'ischaemic stroke',
}

# ═══════════════════════════════════════════════════════════════
# STEP 4 — LOAD MODELS & DATA FILES
# ═══════════════════════════════════════════════════════════════
print("Loading models...")

nlp           = spacy.load('en_ner_bc5cdr_md')
adr_model     = joblib.load(os.path.join(MODELS, 'adr_model.pkl'))
adr_mlb       = joblib.load(os.path.join(MODELS, 'adr_mlb.pkl'))
adr_drug_enc  = joblib.load(os.path.join(MODELS, 'adr_drug_encoder.pkl'))
ddi_model     = joblib.load(os.path.join(MODELS, 'ddi_model.pkl'))
ddi_drug_enc  = joblib.load(os.path.join(MODELS, 'ddi_drug_encoder.pkl'))
ddi_label_enc = joblib.load(os.path.join(MODELS, 'ddi_label_encoder.pkl'))

with open(os.path.join(MODELS, 'ddi_label_map.json')) as f:
    ddi_label_map = json.load(f)

drug_names = pd.read_csv(os.path.join(PROCESSED, 'drug_names_clean.csv'))
vocab      = pd.read_csv(os.path.join(PROCESSED, 'drugbank_vocab_clean.csv'))
lipinski   = pd.read_csv(os.path.join(PROCESSED, 'lipinski_clean.csv'))
adr_data   = pd.read_csv(os.path.join(PROCESSED, 'adr_training_data.csv'))

vocab['drug_name']      = vocab['drug_name'].str.lower().str.strip()
drug_names['drug_name'] = drug_names['drug_name'].str.lower().str.strip()
adr_data['drug_name']   = adr_data['drug_name'].str.lower().str.strip()
adr_data['side_effect'] = adr_data['side_effect'].str.lower().str.strip()
adr_data['frequency']   = pd.to_numeric(adr_data['frequency'], errors='coerce').fillna(0.0)

if 'stitch_id' in drug_names.columns:
    drug_names.rename(columns={'stitch_id': 'stitch_flat_id'}, inplace=True)

meddra_freq_raw = pd.read_csv(os.path.join(PROCESSED, 'meddra_freq_clean.csv'))
meddra_freq_raw['side_effect_name'] = meddra_freq_raw['side_effect_name'].str.lower().str.strip()
meddra_freq_raw = meddra_freq_raw.merge(drug_names, on='stitch_flat_id', how='left')
meddra_freq_raw.dropna(subset=['drug_name'], inplace=True)
meddra_freq_raw['freq_lower'] = pd.to_numeric(meddra_freq_raw['freq_lower'], errors='coerce').fillna(0.0)

# ═══════════════════════════════════════════════════════════════
# STEP 5 — LOAD OFFSIDES
# ═══════════════════════════════════════════════════════════════
OFFSIDES_AVAILABLE = False
offsides_index = {}

_offsides_path = os.path.join(PROCESSED, 'offsides_clean.csv')
if os.path.exists(_offsides_path):
    try:
        print("Loading OFFSIDES PRR data...")
        for _chunk in pd.read_csv(_offsides_path, chunksize=200_000,
                                   usecols=['drug_name', 'side_effect', 'PRR', 'freq']):
            _chunk['drug_name']   = (_chunk['drug_name'].str.lower().str.strip()
                                      .apply(normalize_offsides_name))
            _chunk['side_effect'] = _chunk['side_effect'].str.lower().str.strip()
            _chunk['PRR']  = pd.to_numeric(_chunk['PRR'],  errors='coerce').fillna(0.0)
            _chunk['freq'] = pd.to_numeric(_chunk['freq'], errors='coerce').fillna(0.0)

            _strong = _chunk[_chunk['PRR'] >= MIN_PRR_ADR]
            for _, _row in _strong.iterrows():
                _d = _row['drug_name']
                _e = _row['side_effect']
                _p = float(_row['PRR'])
                _f = float(_row['freq'])
                if _e in ADR_NOISE_BLACKLIST:
                    continue
                if _d not in offsides_index:
                    offsides_index[_d] = {}
                if (_e not in offsides_index[_d] or offsides_index[_d][_e]['prr'] < _p):
                    offsides_index[_d][_e] = {'prr': _p, 'freq': _f}

        OFFSIDES_AVAILABLE = True
        print(f"✅ OFFSIDES loaded — {len(offsides_index):,} drugs indexed (PRR >= {MIN_PRR_ADR})")
    except Exception as _e:
        print(f"⚠️ OFFSIDES load failed: {_e}")
else:
    print("ℹ️ offsides_clean.csv not found — using MedDRA + curated fallback")

print("✅ All models loaded!\n")

# ═══════════════════════════════════════════════════════════════
# STEP 6 — HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════
def get_frequency_category(freq: float) -> str:
    if freq >= 0.10:  return "Very Common (>=10%)"
    elif freq >= 0.01: return "Common (1-10%)"
    elif freq >= 0.001: return "Uncommon (0.1-1%)"
    elif freq > 0:    return "Rare (<0.1%)"
    else:             return "Unknown"

def extract_drugs_from_text(text: str) -> list:
    doc   = nlp(text)
    drugs = [ent.text.lower().strip() for ent in doc.ents if ent.label_ == 'CHEMICAL']
    return list(set(drugs))

def get_drug_features(drug_name: str) -> dict:
    drug_name = normalize_drug_name(drug_name)
    match = vocab[vocab['drug_name'] == drug_name]
    if match.empty:
        match = vocab[vocab['synonyms'].str.lower().str.contains(drug_name, na=False, regex=False)]
    if match.empty:
        return None
    drugbank_id = match.iloc[0]['drugbank_id']
    lip = lipinski[lipinski['drugbank_id'] == drugbank_id]
    if lip.empty:
        return {'drugbank_id': drugbank_id, 'mol_weight': 0.0,
                'h_acceptors': 0.0, 'h_donors': 0.0, 'logp': 0.0}
    row = lip.iloc[0]
    return {'drugbank_id': drugbank_id, 'mol_weight': row['mol_weight'],
            'h_acceptors': row['h_acceptors'], 'h_donors': row['h_donors'],
            'logp': row['logp']}

def _get_curated(drug_name: str, min_freq: float, top_n: int) -> list:
    entries = CURATED_FREQUENCIES.get(drug_name, [])
    if not entries:
        return []
    filtered = [(se, f) for se, f in entries
                if f >= min_freq and se.lower().strip() not in ADR_NOISE_BLACKLIST]
    filtered.sort(key=lambda x: x[1], reverse=True)
    return [{'name': se, 'freq': f, 'category': get_frequency_category(f), 'source': 'curated'}
            for se, f in filtered[:top_n]]

def _get_from_offsides(drug_name: str, min_freq: float, top_n: int) -> list:
    if not OFFSIDES_AVAILABLE or drug_name not in offsides_index:
        return []
    entries = offsides_index[drug_name]
    results = []
    for effect, data in entries.items():
        if effect in ADR_NOISE_BLACKLIST:
            continue
        freq = data['freq']
        if min_freq > 0 and freq > 0 and freq < min_freq:
            continue
        freq = FREQ_OVERRIDES.get((drug_name, effect), freq)
        results.append({'name': effect, 'freq': freq, 'prr': data['prr'],
                         'category': get_frequency_category(freq), 'source': 'offsides'})
    freq_known   = sorted([r for r in results if r['freq'] > 0], key=lambda x: x['freq'], reverse=True)
    freq_unknown = sorted([r for r in results if r['freq'] == 0], key=lambda x: x['prr'],  reverse=True)
    combined = freq_known + freq_unknown
    if len(combined) < MIN_FREQ_KNOWN:
        curated = _get_curated(drug_name, min_freq, top_n)
        offsides_names = {r['name'] for r in combined}
        extra = [c for c in curated if c['name'] not in offsides_names]
        combined = combined + extra
    return combined[:top_n]

# ═══════════════════════════════════════════════════════════════
# STEP 7 — CORE: predict_adr
# ═══════════════════════════════════════════════════════════════
def predict_adr(drug_name: str, top_n: int = 10, min_freq: float = 0.0) -> dict:
    drug_name = normalize_drug_name(drug_name)

    indication_noise = _INDICATION_CONFOUNDERS.get(drug_name, set())

    def _is_noise(se_name: str) -> bool:
        n = se_name.lower().strip()
        return n in ADR_NOISE_BLACKLIST or n in indication_noise

    # ── Layer 1: OFFSIDES PRR ──────────────────────────────────────────
    offsides_results = _get_from_offsides(drug_name, min_freq, top_n)
    if offsides_results:
        offsides_results = [se for se in offsides_results if not _is_noise(se['name'])]
        if offsides_results:
            return {
                'drug': drug_name, 'method': 'offsides_prr',
                'predicted_side_effects': offsides_results, 'confidence': 'high',
            }

    # ── Layer 2: MedDRA frequency from SIDER ──────────────────────────
    known = adr_data[adr_data['drug_name'] == drug_name][['side_effect', 'frequency']].drop_duplicates().copy()

    if not known.empty:
        freq_lookup = meddra_freq_raw[meddra_freq_raw['drug_name'] == drug_name][
            ['side_effect_name', 'freq_lower']].drop_duplicates().copy()
        freq_lookup.columns = ['side_effect', 'freq_detail']
        freq_lookup['freq_detail'] = pd.to_numeric(freq_lookup['freq_detail'], errors='coerce')

        known = known.merge(freq_lookup, on='side_effect', how='left')
        known['freq_detail'] = pd.to_numeric(known['freq_detail'], errors='coerce')
        known['frequency']   = known['freq_detail'].combine_first(known['frequency'])
        known.drop(columns=['freq_detail'], inplace=True)
        known['frequency'] = pd.to_numeric(known['frequency'], errors='coerce').fillna(0.0)
        known = known[~known['side_effect'].apply(lambda x: _is_noise(str(x)))].copy()

        freq_known   = known[known['frequency'] > 0].sort_values('frequency', ascending=False)
        freq_unknown = known[known['frequency'] == 0]

        if min_freq > 0:
            fk = freq_known[freq_known['frequency'] >= min_freq]
            if not fk.empty:
                freq_known = fk
            else:
                curated = _get_curated(drug_name, min_freq, top_n)
                if curated:
                    curated = [se for se in curated if not _is_noise(se['name'])]
                    if curated:
                        return {'drug': drug_name, 'method': 'curated_fallback',
                                'predicted_side_effects': curated, 'confidence': 'high'}
                return {'drug': drug_name, 'method': 'database_lookup',
                        'predicted_side_effects': [], 'confidence': 'high',
                        'message': f'No side effects with frequency >= {min_freq * 100:.1f}% found for {drug_name}.'}

        if len(freq_known) >= MIN_FREQ_KNOWN:
            combined = freq_known.head(top_n)
        elif len(freq_known) > 0:
            pad      = top_n - len(freq_known)
            combined = pd.concat([freq_known, freq_unknown.head(pad)])
        else:
            curated = _get_curated(drug_name, min_freq, top_n)
            if curated:
                curated = [se for se in curated if not _is_noise(se['name'])]
                if curated:
                    return {'drug': drug_name, 'method': 'curated_fallback',
                            'predicted_side_effects': curated, 'confidence': 'high'}
            combined = freq_unknown.head(top_n)

        se_list = [
            {'name': row['side_effect'], 'freq': float(row['frequency']),
             'category': get_frequency_category(float(row['frequency'])), 'source': 'sider'}
            for _, row in combined.iterrows()
        ]
        return {'drug': drug_name, 'method': 'database_lookup',
                'predicted_side_effects': se_list, 'confidence': 'high'}

    # ── Layer 3: Curated fallback ──────────────────────────────────────
    curated = _get_curated(drug_name, min_freq, top_n)
    if curated:
        curated = [se for se in curated if not _is_noise(se['name'])]
        if curated:
            return {'drug': drug_name, 'method': 'curated_fallback',
                    'predicted_side_effects': curated, 'confidence': 'high'}

    # ── Layer 4: ML model fallback ─────────────────────────────────────
    if drug_name not in adr_drug_enc.classes_:
        return {'drug': drug_name, 'method': 'not_found',
                'predicted_side_effects': [], 'confidence': 'none',
                'message': f'Drug "{drug_name}" not found in database.'}

    drug_enc = adr_drug_enc.transform([drug_name])[0]
    features = get_drug_features(drug_name)
    mol_w = features['mol_weight']  if features else 0.0
    h_acc = features['h_acceptors'] if features else 0.0
    h_don = features['h_donors']    if features else 0.0
    logp  = features['logp']        if features else 0.0

    X      = np.array([[drug_enc, mol_w, h_acc, h_don, logp]])
    y_pred = adr_model.predict(X)
    labels = adr_mlb.inverse_transform(y_pred)[0]

    se_list = [
        {'name': se, 'freq': 0.0, 'category': 'Unknown', 'source': 'ml_model'}
        for se in sorted(labels)
        if not _is_noise(se)
    ][:top_n]

    return {'drug': drug_name, 'method': 'model_prediction',
            'predicted_side_effects': se_list, 'confidence': 'medium'}

# ═══════════════════════════════════════════════════════════════
# STEP 8 — CORE: predict_ddi
# ═══════════════════════════════════════════════════════════════

# Drug classes where Hyperkalaemia IS a real DDI signal.
_ACE_INHIBITORS_AND_KSP = {
    # ACE inhibitors
    'lisinopril', 'enalapril', 'ramipril', 'captopril',
    'perindopril', 'fosinopril', 'quinapril', 'benazepril',
    'trandolapril', 'moexipril',
    # ARBs
    'losartan', 'valsartan', 'irbesartan', 'candesartan',
    'olmesartan', 'telmisartan', 'azilsartan',
    # Potassium-sparing diuretics
    'spironolactone', 'eplerenone', 'amiloride', 'triamterene',
    # Direct renin inhibitors
    'aliskiren',
}

def predict_ddi(drug_1: str, drug_2: str, top_n: int = 5) -> dict:
    drug_1 = normalize_drug_name(drug_1)
    drug_2 = normalize_drug_name(drug_2)
    if drug_1 == drug_2:
        return {'error': 'Both drugs are the same.'}

    # ── Pair-aware noise logic ─────────────────────────────────────────
    has_ace = (drug_1 in _ACE_INHIBITORS_AND_KSP or drug_2 in _ACE_INHIBITORS_AND_KSP)

    is_ppi_antiplatelet = (
        (drug_1 in _PPI_CLASS and drug_2 in _ANTIPLATELET_CLASS) or
        (drug_1 in _ANTIPLATELET_CLASS and drug_2 in _PPI_CLASS)
    )

    def _is_ddi_noise(effect: str) -> bool:
        n = effect.lower().strip()
        # PPI + antiplatelet whitelist overrides global blacklist
        # (e.g. Clopidogrel + Omeprazole → Acute MI is a REAL DDI via CYP2C19)
        if is_ppi_antiplatelet and n in _PPI_ANTIPLATELET_WHITELIST:
            return False
        if n in DDI_NOISE_BLACKLIST:
            return True
        # Hyperkalaemia: suppress unless ACE/ARB/KSP is in the pair
        if n == 'hyperkalaemia' and not has_ace:
            return True
        return False

    # ── Layer 1: TWOSIDES PRR database lookup ──────────────────────────
    twosides_path = os.path.join(PROCESSED, 'twosides_clean.csv')
    raw_hits = []
    for chunk in pd.read_csv(twosides_path, chunksize=100_000,
                              usecols=['drug_1', 'drug_2', 'interaction_effect', 'PRR']):
        chunk['drug_1'] = chunk['drug_1'].str.lower().str.strip()
        chunk['drug_2'] = chunk['drug_2'].str.lower().str.strip()
        match = chunk[
            ((chunk['drug_1'] == drug_1) & (chunk['drug_2'] == drug_2)) |
            ((chunk['drug_1'] == drug_2) & (chunk['drug_2'] == drug_1))
        ]
        if not match.empty:
            strong = match[pd.to_numeric(match['PRR'], errors='coerce').fillna(0) >= MIN_PRR_DDI]
            for _, row in strong.iterrows():
                raw_hits.append((float(row['PRR']), str(row['interaction_effect'])))
        if len(raw_hits) >= top_n * 5:
            break

    if raw_hits:
        raw_hits.sort(key=lambda x: x[0], reverse=True)
        seen, ordered = set(), []
        for prr, effect in raw_hits:
            key = effect.lower().strip()
            if key not in seen:
                seen.add(key)
                ordered.append(key)

        cleaned = [ix for ix in ordered if not _is_ddi_noise(ix)]

        if cleaned:
            return {
                'drug_1': drug_1, 'drug_2': drug_2,
                'method': 'database_lookup',
                'interactions': [ix.title() for ix in cleaned[:top_n]],
                'confidence': 'high',
            }

        return {
            'drug_1': drug_1, 'drug_2': drug_2,
            'method': 'database_lookup', 'interactions': [], 'confidence': 'high',
            'message': 'No clinically significant interactions detected.',
        }

    # ── Layer 2: ML model fallback ─────────────────────────────────────
    unknown = [d for d in [drug_1, drug_2] if d not in ddi_drug_enc.classes_]
    if unknown:
        return {'drug_1': drug_1, 'drug_2': drug_2, 'method': 'not_found',
                'interactions': [], 'confidence': 'none',
                'message': f'Drug(s) not found: {unknown}'}

    d1_enc = ddi_drug_enc.transform([drug_1])[0]
    d2_enc = ddi_drug_enc.transform([drug_2])[0]
    f1 = get_drug_features(drug_1) or {}
    f2 = get_drug_features(drug_2) or {}
    X  = np.array([[
        d1_enc, d2_enc,
        f1.get('mol_weight', 0), f1.get('h_acceptors', 0),
        f1.get('h_donors', 0),   f1.get('logp', 0),
        f2.get('mol_weight', 0), f2.get('h_acceptors', 0),
        f2.get('h_donors', 0),   f2.get('logp', 0),
        0.0, 0.0,
    ]])
    proba       = ddi_model.predict_proba(X)
    top_indices = np.argsort(proba[0])[::-1][:top_n * 2]
    interactions = [
        ddi_label_map[str(i)]
        for i in top_indices
        if not _is_ddi_noise(ddi_label_map[str(i)])
    ][:top_n]

    return {'drug_1': drug_1, 'drug_2': drug_2,
            'method': 'model_prediction', 'interactions': interactions,
            'confidence': 'medium'}

# ═══════════════════════════════════════════════════════════════
# STEP 9 — FULL PRESCRIPTION PIPELINE
# ═══════════════════════════════════════════════════════════════
def analyze_prescription(text: str, top_n: int = 10, min_freq: float = 0.0) -> dict:
    raw_drugs = extract_drugs_from_text(text)
    drugs = list(set([normalize_drug_name(d) for d in raw_drugs]))
    if not drugs:
        return {'input_text': text, 'drugs_found': [],
                'message': 'No drug names detected in the text.'}
    results = {'input_text': text, 'drugs_found': drugs,
               'adr_predictions': {}, 'ddi_predictions': []}
    for drug in drugs:
        results['adr_predictions'][drug] = predict_adr(drug, top_n=top_n, min_freq=min_freq)
    for d1, d2 in combinations(drugs, 2):
        results['ddi_predictions'].append(predict_ddi(d1, d2, top_n=top_n))
    return results

# ═══════════════════════════════════════════════════════════════
# STEP 10 — SELF-TEST
# ═══════════════════════════════════════════════════════════════
def format_freq_pct(freq: float) -> str:
    if freq <= 0:
        return "Unknown "
    pct = freq * 100
    if pct < 0.001:  return f"< 0.001%"
    elif pct < 0.01: return f"{pct:.4f}%"
    elif pct < 0.1:  return f"{pct:.3f}%"
    else:            return f"{pct:.2f}% "

if __name__ == '__main__':
    sep = "=" * 60

    print(f"\nOFFSIDES active  : {OFFSIDES_AVAILABLE}")
    print(f"ADR blacklist    : {len(ADR_NOISE_BLACKLIST)} entries")
    print(f"DDI blacklist    : {len(DDI_NOISE_BLACKLIST)} entries")
    print(f"MIN_PRR_ADR      : {MIN_PRR_ADR}")
    print(f"MIN_PRR_DDI      : {MIN_PRR_DDI}")

    for drug in ['paracetamol', 'ibuprofen', 'warfarin', 'metformin',
                  'aspirin', 'clopidogrel', 'omeprazole']:
        print(f"\n{sep}\nTEST: {drug.upper()}\n{sep}")
        r = predict_adr(drug, top_n=8)
        print(f"Method: {r['method']} | Confidence: {r['confidence']}")
        for se in r['predicted_side_effects']:
            pct = format_freq_pct(se['freq'])
            prr = f" PRR={se['prr']:.1f}" if se.get('prr', 0) > 0 else ""
            cat = se['category'][:18]
            print(f"  [{cat:<18}] {pct:>9}{prr}  {se['name']}")

    ddi_pairs = [
        ('paracetamol',  'ibuprofen'),
        ('warfarin',     'aspirin'),
        ('aspirin',      'clopidogrel'),
        ('clopidogrel',  'omeprazole'),
        ('warfarin',     'clopidogrel'),
        ('warfarin',     'omeprazole'),
        ('aspirin',      'omeprazole'),
        ('metformin',    'ibuprofen'),
        ('acetaminophen','cetirizine'),
    ]
    for d1, d2 in ddi_pairs:
        print(f"\n{sep}\nDDI: {d1.title()} + {d2.title()}\n{sep}")
        r = predict_ddi(d1, d2, top_n=5)
        print(f"Method: {r['method']} | Confidence: {r['confidence']}")
        interactions = r.get('interactions', [])
        if interactions:
            for ix in interactions:
                print(f"  ⚠️  {ix}")
        else:
            print(f"  ✅  {r.get('message', 'No significant interactions found')}")
