# Open Biomedical Dataset Catalog

A working catalog of datasets usable for computational neuroscience / biomedical-engineering research, with access level and which projects in this repo use them. Access levels:

- **Open** — download with no registration
- **Reg** — free account / click-through terms
- **DUA** — signed data-use agreement (usually free, days–weeks)
- **Cred** — credentialed access (e.g. PhysioNet: CITI training + DUA)
- **App** — formal application / review (e.g. NDA, LONI, dbGaP, UK Biobank fee)

> Never commit raw data to any repository. Each project has a `data/README.md` with exact acquisition steps.

---

## 1. Neuroimaging (MRI / PET)

| Dataset | Content | Size | Access | URL | Used in |
|---|---|---|---|---|---|
| OpenNeuro | 1000+ BIDS datasets (MRI, EEG, MEG, iEEG, PET) | PBs | Open | https://openneuro.org | mriqc-audit, fmri-multiverse, lifespan-norm, tms-target |
| MRIQC WebAPI | Crowdsourced image-quality metrics | 100k+ scans | Open | https://mriqc.nimh.nih.gov | mriqc-audit |
| HCP Young Adult S1200 | 1200 adults 22–37, T1/T2/dMRI/rs- & task-fMRI, MEG subset, twins | ~90 TB | Reg (+App for restricted) | https://db.humanconnectome.org | brain-age, imgtx-nulls, cpm-fair, conn-h2, tms-target |
| HCP Lifespan (Aging, Development) | 5–100+ years, same protocol family | ~10 TB | App (NDA) | https://nda.nih.gov/ccf | brain-age, lifespan-norm |
| HCP Early Psychosis / Disease connectomes | Clinical HCP-style datasets | var. | App (NDA) | https://www.humanconnectome.org/disease-studies | — |
| OASIS-1/2/3/4 | Cross-sectional, longitudinal aging & dementia, MRI+PET+CSF+clinical | 1000s of sessions | DUA (free) | https://www.oasis-brains.org | brain-age, amyloid-mri, atrophy-subtypes, lifespan-norm |
| ADNI | Alzheimer's MRI/PET/CSF/genetics longitudinal | 2000+ subj | App (LONI) | https://adni.loni.usc.edu | amyloid-mri, atrophy-subtypes |
| A4 / LEARN | Preclinical AD screening (amyloid PET) | 4000+ | App (LONI) | https://ida.loni.usc.edu | amyloid-mri |
| AIBL | Australian aging cohort | ~1100 | App (LONI) | https://aibl.csiro.au | — |
| PPMI | Parkinson's Progression Markers Initiative (MRI, DAT-SPECT, biospecimens) | 1000s | App | https://www.ppmi-info.org | — |
| ABCD | 11k children, longitudinal multimodal | 100s TB | App (NDA) | https://abcdstudy.org | cpm-fair |
| UK Biobank | 100k imaging, 500k phenotype/genotype | PBs | App (fee) | https://www.ukbiobank.ac.uk | — |
| BIG40 / UKB imaging GWAS summary stats | GWAS on ~4000 imaging phenotypes | GBs | Open | https://open.win.ox.ac.uk/ukbiobank/big40/ | gwas-ct |
| IXI | 600 healthy adults, T1/T2/PD/MRA/DTI | 20 GB | Open | https://brain-development.org/ixi-dataset/ | lifespan-norm |
| ABIDE I/II | Autism rs-fMRI, 2000+ | ~1 TB | Open | http://fcon_1000.projects.nitrc.org/indi/abide/ | cpm-fair |
| ADHD-200 | ADHD rs-fMRI | ~ | Open | http://fcon_1000.projects.nitrc.org/indi/adhd200/ | — |
| CoRR | Test-retest rs-fMRI | 1600 subj | Open | http://fcon_1000.projects.nitrc.org/indi/CoRR/ | fmri-multiverse |
| NKI-Rockland | Lifespan community sample | 1000+ | Reg | http://fcon_1000.projects.nitrc.org/indi/enhanced/ | lifespan-norm |
| Cam-CAN | 700 adults 18–88, MRI+MEG | ~ | DUA | https://cam-can.mrc-cbu.cam.ac.uk | lifespan-norm |
| MPI-LEMON | Young/old, MRI+EEG+physio | 227 | Open | https://fcon_1000.projects.nitrc.org/indi/retro/MPI_LEMON.html | — |
| AOMIC (PIOP1/PIOP2/ID1000) | 1000+ subj task+rest fMRI | ~ | Open (OpenNeuro) | https://nilab-uva.github.io/AOMIC.github.io/ | fmri-multiverse |
| NARPS (ds001734) | 70-team multiverse dataset | ~ | Open | https://openneuro.org/datasets/ds001734 | fmri-multiverse |
| BrainChart reference models | Lifespan normative curves (Bethlehem 2022) | small | Open | https://github.com/brainchart/Lifespan | lifespan-norm |
| PCNtoolkit pretrained normative models | Cortical thickness/subcortical models | small | Open | https://github.com/predictive-clinical-neuroscience | lifespan-norm |
| SchizConnect / COBRE / MCIC | Schizophrenia MRI | 100s | Reg | http://schizconnect.org | — |
| PING / PNC | Pediatric imaging + genetics | 1000s | App (dbGaP) | https://www.nitrc.org/projects/pnc | — |
| MIDA / SimNIBS example heads | Head models for E-field simulation | small | Open | https://itis.swiss/virtual-population/regional-human-models/mida-model/ | morph-stim, tms-target |
| TCIA (LIDC-IDRI, NSCLC, etc.) | Cancer imaging archive, 100+ collections | 100s TB | Open (some Reg) | https://www.cancerimagingarchive.net | lidc-uq |
| LUNA16 | Lung nodule CT challenge (from LIDC) | 100 GB | Reg | https://luna16.grand-challenge.org | lidc-uq |
| Duke Lung Cancer Screening | 2000+ screening CTs, 2024 | ~ | Open | https://zenodo.org/ (search "Duke Lung") | lidc-uq |
| NLST | National Lung Screening Trial CTs | 200k CT | App (CDAS) | https://cdas.cancer.gov/nlst/ | lidc-uq |
| fastMRI | Raw k-space knee/brain | TBs | Reg | https://fastmri.med.nyu.edu | — |
| BraTS | Brain tumor segmentation | ~ | Reg | https://www.synapse.org/brats | — |
| Medical Segmentation Decathlon | 10 organ segmentation tasks | ~ | Open | http://medicaldecathlon.com | — |
| MOTUM / Cerebral small-vessel disease sets | | | | | — |

## 2. Electrophysiology (EEG / MEG / iEEG / spikes)

| Dataset | Content | Size | Access | URL | Used in |
|---|---|---|---|---|---|
| CHB-MIT Scalp EEG | 23 pediatric epilepsy cases, 198 seizures | 40 GB | Open (PhysioNet) | https://physionet.org/content/chbmit/ | xseizure |
| Siena Scalp EEG | 14 adults, 47 seizures, EEG+ECG | 20 GB | Open (PhysioNet) | https://physionet.org/content/siena-scalp-eeg/ | xseizure |
| TUH EEG Corpus (TUSZ, TUAB, TUAR, TUEV) | 25k+ sessions, largest clinical EEG | TBs | Reg | https://isip.piconepress.com/projects/nedc/html/tuh_eeg/ | xseizure |
| Helsinki neonatal EEG | 79 neonates, 3-expert annotations | ~ | Open (Zenodo) | https://zenodo.org/record/2547147 | xseizure |
| Bonn EEG | Classic small intracranial set | small | Open | https://www.upf.edu/web/ntsa/downloads | — |
| Sleep-EDF Expanded | 197 PSG nights | 8 GB | Open (PhysioNet) | https://physionet.org/content/sleep-edfx/ | spindle-age |
| NSRR (SHHS, MESA, MrOS, CFS, CCSHS, WSC, ABC…) | 40k+ PSG with outcomes | TBs | DUA (free) | https://sleepdata.org | spindle-age |
| Haaglanden Medisch Centrum PSG | 154 PSG | ~ | Open (PhysioNet) | https://physionet.org/content/hmc-sleep-staging/ | spindle-age |
| MASS | Montreal Archive of Sleep Studies | ~ | DUA | http://ceams-carsm.ca/mass/ | spindle-age |
| DREAMS spindle DB | Spindle annotations | small | Open | https://zenodo.org/record/2650142 | spindle-age |
| PhysioNet Motor Imagery (EEGMMIDB) | 109 subj BCI | 3 GB | Open | https://physionet.org/content/eegmmidb/ | — |
| MOABB datasets | 30+ BCI datasets, unified API | ~ | Open | https://moabb.neurotechx.com | — |
| OpenNeuro EEG/MEG/iEEG | 300+ BIDS-EEG datasets | ~ | Open | https://openneuro.org | spindle-age, tms-target |
| MNE sample / MEG datasets | Test datasets | ~ | Open | https://mne.tools/stable/api/datasets.html | — |
| Cam-CAN MEG | Lifespan MEG | ~ | DUA | see above | — |
| HCP MEG | 95 subjects rest+task MEG | ~ | Reg | see above | — |
| DANDI Archive | 300+ NWB datasets (Neuropixels, calcium imaging, patch-clamp, human iEEG) | 100s TB | Open | https://dandiarchive.org | npx-drift, ibl-bwm |
| Allen Brain Observatory – Visual Coding Neuropixels | 58 mice, 100k+ units | ~ | Open (allensdk / DANDI 000021) | https://allensdk.readthedocs.io | npx-drift |
| Allen Visual Behavior Neuropixels / 2P | Behavior-linked recordings | ~ | Open (DANDI 000022 etc.) | https://portal.brain-map.org | npx-drift |
| IBL Brain-wide Map | 600+ insertions, 100+ regions | ~ | Open (ONE api / DANDI 000409) | https://int-brain-lab.github.io/iblenv/ | ibl-bwm |
| Steinmetz 2019 | 8-probe Neuropixels task | ~ | Open (figshare) | https://doi.org/10.6084/m9.figshare.9598406 | ibl-bwm |
| Neural Latents Benchmark | Standardized NWB neural decoding | ~ | Open (DANDI) | https://neurallatents.github.io | ibl-bwm |
| Allen Cell Types Database | Patch-clamp + morphology + transcriptomics, mouse & human | ~ | Open (allensdk) | https://celltypes.brain-map.org | morph2ephys, morph-stim |
| Allen Patch-seq (mouse VISp, human MTG) | 4000+ Patch-seq | ~ | Open (DANDI 000020, 000023, 000035) | https://dandiarchive.org/dandiset/000020 | morph2ephys |
| Brain/MINDS marmoset | Tracer & MRI | ~ | Open | https://dataportal.brainminds.jp | meso-vs-axon |
| Human iEEG: RAM (UPenn), Brain TreeBank, OpenNeuro iEEG-BIDS | Memory & language iEEG | ~ | Reg/Open | https://memory.psych.upenn.edu/RAM | — |
| Neuropixels NHP / Pesaran etc. | Primate recordings | ~ | Open (DANDI) | — | — |

## 3. Cellular / molecular / atlas

| Dataset | Content | Size | Access | URL | Used in |
|---|---|---|---|---|---|
| NeuroMorpho.org | 250k+ SWC reconstructions, 900+ labs | ~ | Open (REST API) | https://neuromorpho.org | nm-scaling, morph2ephys, meso-vs-axon, morph-stim |
| Allen Mouse Brain Connectivity Atlas | 2000+ tracer experiments, CCFv3 | ~ | Open (allensdk) | https://connectivity.brain-map.org | meso-vs-axon |
| Allen Mouse/Human Brain Atlas (ISH, microarray) | Gene expression maps | ~ | Open (allensdk / abagen) | https://brain-map.org | imgtx-nulls, conn-h2 |
| Allen Brain Cell Atlas (ABC Atlas) | Whole mouse brain scRNA-seq + MERFISH; human | TBs | Open (S3) | https://portal.brain-map.org/atlases-and-data/bkp/abc-atlas | gwas-ct |
| MouseLight (Janelia) | 1000+ full axonal reconstructions | ~ | Open | https://ml-neuronbrowser.janelia.org | meso-vs-axon |
| SEU-ALLEN full morphologies | 1700+ complete neurons (Peng 2021) | ~ | Open | https://braintell.org | meso-vs-axon |
| BICCN / NeMO / BIL | Cell census data | PBs | Open | https://biccn.org | gwas-ct |
| CELLxGENE / Siletti 2023 | Human brain 3M nuclei | ~ | Open | https://cellxgene.cziscience.com | gwas-ct |
| Human Protein Atlas | Tissue/brain expression | ~ | Open | https://www.proteinatlas.org | — |
| MICrONS | 1 mm³ cortex EM connectome | TBs | Open | https://www.microns-explorer.org | — |
| FlyWire / hemibrain | Drosophila connectome | ~ | Open | https://flywire.ai | — |
| GTEx | Multi-tissue expression + eQTL | ~ | Open (summary) / App (raw) | https://gtexportal.org | gwas-ct |
| GWAS Catalog / PGC / OpenGWAS | Summary statistics | ~ | Open | https://www.ebi.ac.uk/gwas/ | gwas-ct |
| PsychENCODE | Brain multi-omics | ~ | App | https://psychencode.synapse.org | — |
| Blue Brain Cell Atlas / SSCx models | Biophysical models | ~ | Open | https://bbp.epfl.ch/nmc-portal/ | morph-stim |
| ModelDB | 1800+ computational models | ~ | Open | https://modeldb.science | morph-stim |

## 4. Clinical / EHR / ICU / signals

| Dataset | Content | Size | Access | URL | Used in |
|---|---|---|---|---|---|
| MIMIC-IV (hosp, icu, ed, note, ecg, cxr, echo, waveform) | 300k+ patients, BIDMC 2008–2022 | 100s GB | Cred (PhysioNet) | https://physionet.org/content/mimiciv/ | icu-transport, cuffless-bp, faers-ehr, ed-triage, vent-rl, ecg-xgen |
| MIMIC-III (+ waveform matched subset) | 2001–2012 ICU | ~ | Cred | https://physionet.org/content/mimic3wdb-matched/ | cuffless-bp |
| eICU-CRD | 200k ICU stays, 200 US hospitals | ~ | Cred | https://physionet.org/content/eicu-crd/ | icu-transport, vent-rl |
| HiRID | Bern ICU high-resolution | ~ | Cred | https://physionet.org/content/hirid/ | icu-transport, vent-rl |
| AmsterdamUMCdb | Dutch ICU 23k admissions | ~ | DUA | https://amsterdammedicaldatascience.nl | icu-transport |
| SICdb | Salzburg ICU | ~ | Cred | https://physionet.org/content/sicdb/ | icu-transport |
| INSPIRE (Korea) | Perioperative | ~ | Cred | https://physionet.org/content/inspire/ | — |
| VitalDB | 6000 surgical cases, waveforms | ~ | Open | https://vitaldb.net | cuffless-bp |
| PulseDB | Curated MIMIC+VitalDB for BP | ~ | Open | https://github.com/pulselabteam/PulseDB | cuffless-bp |
| PTB-XL (+ PTB-XL+) | 21k 12-lead ECGs | 3 GB | Open | https://physionet.org/content/ptb-xl/ | ecg-xgen |
| Chapman-Shaoxing / Ningbo | 45k ECGs | ~ | Open | https://physionet.org/content/ecg-arrhythmia/ | ecg-xgen |
| PhysioNet Challenge 2020/2021 sets (CPSC, Georgia, INCART, PTB) | 88k ECGs | ~ | Open | https://physionet.org/content/challenge-2021/ | ecg-xgen |
| CODE-15% | 345k Brazilian ECGs | ~ | Open (Zenodo) | https://zenodo.org/record/4916206 | ecg-xgen |
| MIMIC-IV-ECG | 800k 12-lead ECGs | ~ | Cred | https://physionet.org/content/mimic-iv-ecg/ | ecg-xgen, faers-ehr |
| MIT-BIH, LTAF, AFDB, etc. | Classic arrhythmia | small | Open | https://physionet.org | — |
| PhysioNet Challenge 2019 (sepsis) | 40k ICU stays hourly | ~ | Open | https://physionet.org/content/challenge-2019/ | icu-transport |
| EchoNet-Dynamic / -Pediatric / -LVH | 10k+ echo videos | ~ | DUA (free) | https://echonet.github.io/dynamic/ | echo-uq |
| CAMUS | 500 echo w/ segmentation | ~ | Open | https://www.creatis.insa-lyon.fr/Challenge/camus/ | echo-uq |
| TMED-2 | Echo view/AS labels | ~ | Reg | https://tmed.cs.tufts.edu | echo-uq |
| MIMIC-CXR / CheXpert / NIH ChestX-ray14 | Chest X-rays | 100s GB | Cred / Reg / Open | — | — |
| PPG-DaLiA / WESAD / BIDMC PPG | Wearable signals | small | Open | https://archive.ics.uci.edu | — |
| PhysioNet Wearable (e.g., "Pulse Transit Time PPG") | | | Open | | cuffless-bp |
| All of Us | 400k+ EHR+genomics+wearables | | App (Researcher Workbench) | https://allofus.nih.gov | — |
| N3C | COVID EHR | | App | https://covid.cd2h.org | — |
| SEER | Cancer registry | | Reg | https://seer.cancer.gov | — |
| NHANES | Survey + labs + accelerometry | | Open | https://www.cdc.gov/nchs/nhanes | faers-bias |
| MEPS | Prescribed medicines denominators | | Open | https://meps.ahrq.gov | faers-bias |
| Medicare Part D PUF | Prescriber-level drug volumes | | Open | https://data.cms.gov | faers-bias |
| Synthea | Synthetic EHR | | Open | https://synthetichealth.github.io/synthea/ | — |

## 5. Regulatory / pharmacovigilance / drugs

| Dataset | Content | Size | Access | URL | Used in |
|---|---|---|---|---|---|
| openFDA drug/event (FAERS) | 20M+ adverse-event reports | ~ | Open (API + bulk) | https://open.fda.gov/apis/drug/event/ | faers-bias, unlabeled-ae, faers-ehr |
| openFDA drug/label (SPL) | All US drug labels | ~ | Open | https://open.fda.gov/apis/drug/label/ | unlabeled-ae |
| openFDA device/event (MAUDE), recall, enforcement, 510k, pma, udi, classification | Device regulatory | ~ | Open | https://open.fda.gov/apis/device/ | device-recall |
| FDA AI-enabled medical devices list | ~1000 devices | small | Open | https://www.fda.gov/medical-devices/software-medical-device-samd/artificial-intelligence-enabled-medical-devices | device-recall |
| FDA potential signals (quarterly FAERS) | Ground truth signals | small | Open | https://www.fda.gov/drugs/questions-and-answers-fdas-adverse-event-reporting-system-faers/ | unlabeled-ae |
| VigiAccess (WHO) | Global ICSR counts | ~ | Open (web) | https://www.vigiaccess.org | faers-bias |
| EudraVigilance (public) | EU ICSRs | ~ | Open (web) | https://www.adrreports.eu | faers-bias |
| DrugBank / RxNorm / ATC / MedDRA (licensed) | Ontologies | ~ | Open / Reg / License | — | all pharma |
| SIDER / OFFSIDES / TWOSIDES | Side-effect resources | ~ | Open | http://sideeffects.embl.de | unlabeled-ae |
| ClinicalTrials.gov (AACT) | Trials + results | ~ | Open | https://aact.ctti-clinicaltrials.org | — |
| DailyMed | Label history | ~ | Open | https://dailymed.nlm.nih.gov | unlabeled-ae |

## 6. Other useful

| Dataset | Content | Access | URL |
|---|---|---|---|
| PhysioNet (all) | 800+ databases | Open/Cred | https://physionet.org/about/database/ |
| Zenodo / figshare / OSF / Dryad | Long-tail datasets | Open | — |
| NITRC | Neuroimaging tools & data | Reg | https://www.nitrc.org |
| EBRAINS | Human Brain Project data | Reg | https://ebrains.eu |
| CONP / Canadian Open Neuroscience | | Open | https://portal.conp.ca |
| BossDB | Volumetric EM/X-ray | Open | https://bossdb.org |
| OpenOrganelle | Cellular EM | Open | https://openorganelle.janelia.org |
| Kaggle / grand-challenge.org | Challenge datasets | Reg | — |
| HuggingFace Datasets (biomedical) | Mirrors + benchmarks | Open | https://huggingface.co/datasets |
| ELIXIR/ EGA | Genomic (controlled) | App | https://ega-archive.org |
| Protein Data Bank / AlphaFold DB | Structures | Open | https://www.rcsb.org |
| ChEMBL / PubChem / BindingDB | Chemistry | Open | — |
| Human Cell Atlas | scRNA-seq | Open | https://data.humancellatlas.org |
| METABRIC / TCGA / GDC | Cancer genomics | Open / App | https://portal.gdc.cancer.gov |
| PubMed / PMC OA / OpenAlex | Literature (for gap-finding) | Open | https://openalex.org |
