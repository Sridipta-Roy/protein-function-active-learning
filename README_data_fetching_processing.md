# Data Fetching, Processing, and Labeling

This README documents the data fetching, preprocessing, and labeling workflow used in `001_data_collection.ipynb`. It covers collecting UniProt protein records, cleaning/filtering the dataset, assigning broad function-class labels, and preparing input files for downstream Galaxy workflows.

## Purpose

The goal of this notebook is to build a clean, labeled protein dataset from UniProtKB that can be used for supervised protein function classification, active learning experiments, and Galaxy-based downstream analysis.

The current workflow:

1. Fetches reviewed human protein records from UniProtKB.
2. Handles UniProt pagination using the `next` URL from the response header.
3. Saves raw protein records in TSV and CSV formats.
4. Standardizes column names for easier processing.
5. Filters proteins using sequence quality and annotation criteria.
6. Assigns a single broad function-class label to each protein.
7. Exports Galaxy-ready protein ID and FASTA files.
8. Creates smaller annotation-based protein ID subsets for focused Galaxy analysis.

## Input Data Source

Data is fetched directly from the UniProt REST API.

Current query:

```text
reviewed:true AND organism_id:9606
```

This retrieves reviewed Swiss-Prot human protein records.

Current fetch configuration:

```python
MAX_PROTEINS = 10000
batch_size = 500
sleep_time = 0.3
```

The notebook uses TSV format from UniProt because it is easier to parse into a tabular pandas DataFrame than nested JSON when working with thousands of protein records.

## Project Structure

The notebook automatically creates the required folders.

```text
project_root/
│
├── notebooks/
│   └── 001_data_collection.ipynb
│
├── data/
│   ├── raw/
│   │   ├── uniprot_proteins.tsv
│   │   └── uniprot_proteins.csv
│   │
│   └── processed/
│       ├── uniprot_proteins_clean.csv
│       ├── uniprot_proteins_filtered.csv
│       └── labeled_dataset.csv
│
└── galaxy_inputs/
    ├── protein_ids.txt
    ├── protein_sequences.fasta
    │
    └── subsets/
        ├── enzyme_ids.txt
        ├── transporter_ids.txt
        ├── receptor_ids.txt
        ├── dna_rna_binding_ids.txt
        └── background_all_ids.txt
```

## Dependencies

Install the required Python packages before running the notebook.

```bash
pip install requests tqdm pandas
```

The notebook uses:

```python
import re
import time
import requests
import numpy as np
import pandas as pd
from io import StringIO
from pathlib import Path
from tqdm.notebook import tqdm
from requests.adapters import HTTPAdapter, Retry
```

## Data Fetching Workflow

The notebook defines a `UniProtTSVFetcher` class to fetch protein records from UniProtKB.

Main responsibilities of the fetcher:

- Calls the UniProt REST API in TSV format.
- Requests records in batches.
- Extracts the `next` page URL from the API response.
- Continues fetching until either:
  - `MAX_PROTEINS` is reached, or
  - UniProt returns no next page.
- Uses retry logic for temporary server errors.
- Combines all fetched batches into one pandas DataFrame.
- Checks whether UniProt accession IDs are unique.

The pagination logic is important because UniProt does not return all records in one response. The next page is provided through the response `Link` header.

## Raw Output Files

After fetching, the notebook saves the raw UniProt data in both TSV and CSV formats.

```text
data/raw/uniprot_proteins.tsv
data/raw/uniprot_proteins.csv
```

The TSV file is the preferred raw format because it preserves the UniProt tabular response directly. The CSV file is saved as a convenience for tools that expect CSV input.

## Dataset Inspection

The notebook prints basic dataset statistics, including:

- Total number of proteins downloaded.
- Average and median protein sequence length.
- Number of proteins with function annotations.
- Number of proteins with Gene Ontology terms.
- Number of proteins with EC numbers.
- List of columns returned by UniProt.

It also prints one sample protein record to verify that the downloaded fields look correct.

## Column Standardization

UniProt column names are renamed into simpler project-friendly names.

| Original UniProt Column | Standardized Column |
|---|---|
| `Entry` | `accession` |
| `Entry Name` | `entry_name` |
| `Protein names` | `protein_name` |
| `Gene Names` | `gene_names` |
| `Organism` | `organism` |
| `Length` | `seq_length` |
| `Sequence` | `sequence` |
| `Function [CC]` | `function` |
| `Gene Ontology IDs` | `go_ids` |
| `Gene Ontology (GO)` | `go_terms` |
| `EC number` | `ec_numbers` |
| `Keywords` | `keywords` |

The cleaned standardized file is saved as:

```text
data/processed/uniprot_proteins_clean.csv
```

## Filtering Workflow

The notebook creates a filtered dataset from the cleaned UniProt file.

Filtering steps:

1. Keep records with required fields:
   - `accession`
   - `sequence`

2. Convert protein sequences to uppercase.

3. Remove sequences containing invalid or ambiguous amino acid characters.

   Allowed amino acids:

   ```text
   A C D E F G H I K L M N P Q R S T V W Y
   ```

4. Keep proteins with sequence length between **50 and 1000 amino acids**.

   The upper bound is set to 1000 so that all sequences remain within the
   ESM-2 model input limit (1022 residues). This avoids the need for
   truncation or sliding-window handling at the embedding stage.

5. Remove likely fragments based on protein names containing `fragment`.

6. Keep only proteins that have at least one useful annotation:
   - GO terms
   - Keywords
   - EC numbers

The filtered output is saved as:

```text
data/processed/uniprot_proteins_filtered.csv
```

## Data Filtering Summary

This notebook applies dataset quality filters that are relevant for UniProt-based protein function analysis:

- Remove missing accessions or sequences.
- Remove ambiguous amino acid sequences.
- Remove very short or very long proteins (kept range: 50–1000 aa).
- Remove fragment records.
- Keep proteins with useful functional annotations.

## Labeling Workflow

The notebook assigns a single broad function-class label to each filtered protein. This `function_class` column is the supervised target used by all downstream modeling and active-learning notebooks.

### Classes

Each protein is assigned exactly one of:

```text
enzyme
transporter
receptor
dna_rna_binding
structural
other
```

### Priority cascade

A protein can match several functional descriptions at once (for example, a
membrane-bound kinase matches both enzyme and transporter rules). To produce a
single-label target, overlaps are resolved with an explicit priority order.
The first rule that matches wins:

```text
1. enzyme            (EC number, catalytic / hydrolase / transferase / kinase terms)
2. dna_rna_binding   (DNA/RNA/nucleotide-binding keywords, GO nucleic-acid terms)
3. receptor          (receptor keywords, GO receptor-activity terms)
4. transporter       (transmembrane/transport keywords, GO transport terms)
5. structural        (cytoskeleton, collagen, keratin, ECM, structural terms)
6. other             (no rule matched)
```

The priority order is a deliberate choice and is reported here because it
determines how overlapping proteins are labeled. Enzyme is placed first because
the presence of an EC number is the strongest single signal.

### Label source caveat

Labels are derived from UniProt keywords, GO terms, and EC numbers. The
classifier therefore partly learns to reproduce UniProt's own annotation rules
from sequence, rather than discovering function from scratch. This is
acceptable for the project's scope but should be stated explicitly.

### Class distribution

The notebook prints the class distribution after labeling. For the current
configuration (reviewed human proteins, 50–1000 aa) the distribution is
approximately:

| Class | Count | Share |
|---|---|---|
| enzyme | 2687 | 31.5% |
| dna_rna_binding | 1723 | 20.2% |
| other | 1304 | 15.3% |
| transporter | 1300 | 15.3% |
| receptor | 758 | 8.9% |
| structural | 748 | 8.8% |
| **Total** | **8520** | **100%** |

The imbalance is mild (largest-to-smallest ratio ≈ 3.6:1). Stratified splits
and macro-F1 as the primary metric are sufficient; aggressive resampling is not
required.

### Labeled output

The labeled dataset is saved as the single source of truth for downstream work:

```text
data/processed/labeled_dataset.csv
```

This file contains all filtered columns plus the added `function_class` column.

## Galaxy Input Files

The notebook exports two main files for Galaxy.

### Protein ID File

```text
galaxy_inputs/protein_ids.txt
```

This file contains one UniProt accession per line.

Example:

```text
P31946
P62258
Q04917
```

This file can be used for Galaxy tools or workflows that require UniProt accessions, ID mapping, GO enrichment, pathway analysis, or Reactome-style input.

### FASTA File

```text
galaxy_inputs/protein_sequences.fasta
```

This file contains protein sequences in FASTA format.

Example:

```text
>P31946 14-3-3 protein beta/alpha
MTMDKSELVQKAKLAEQAERYDDMAAAMKAVTEQGHELSNEERNLLSVAYKNVVGARRSS...
```

This file can be used for sequence-based Galaxy tools, such as annotation or similarity search workflows.

## Annotation-Based Subsets

Instead of uploading only one large protein ID file to Galaxy, the notebook also creates smaller focused subsets. These are useful for testing workflows, reducing runtime, and comparing different functional protein groups.

These subsets use the same keyword/GO/EC matching logic as the labeling rules,
but they are independent of the final single-label `function_class` column: a
protein may appear in more than one subset, and each subset is capped at 300
accessions for manageable Galaxy runtime. The subsets are for Galaxy enrichment
input, not for training.

Subset files are saved under:

```text
galaxy_inputs/subsets/
```

### Enzymes

```text
galaxy_inputs/subsets/enzyme_ids.txt
```

Selected using:

- EC numbers.
- Enzyme-related keywords.
- GO terms related to catalytic, hydrolase, transferase, or kinase activity.

### Transporters

```text
galaxy_inputs/subsets/transporter_ids.txt
```

Selected using:

- Transport-related keywords.
- GO transporter activity terms.
- Protein names containing transporter, channel, solute carrier, or pump.

### Receptors

```text
galaxy_inputs/subsets/receptor_ids.txt
```

Selected using:

- Receptor keywords.
- GO receptor activity terms.
- Protein names containing receptor.

### DNA/RNA-Binding Proteins

```text
galaxy_inputs/subsets/dna_rna_binding_ids.txt
```

Selected using:

- DNA-binding, RNA-binding, or nucleotide-binding keywords.
- GO nucleic acid binding terms.
- Protein names related to transcription factors or zinc finger proteins.

### Background File

```text
galaxy_inputs/subsets/background_all_ids.txt
```

This file contains all filtered protein accessions and can be used as the background/reference set for enrichment-style analysis.

## Galaxy Testing Strategy

For initial testing, upload the smaller subset files:

```text
enzyme_ids.txt
transporter_ids.txt
receptor_ids.txt
dna_rna_binding_ids.txt
```

Use:

```text
background_all_ids.txt
```

as the background set when a Galaxy tool asks for a reference or background protein list.

After confirming that the workflow runs correctly, use the full files:

```text
protein_ids.txt
protein_sequences.fasta
```

## Expected Outputs

At the end of the notebook, the following outputs should exist:

| File | Purpose |
|---|---|
| `data/raw/uniprot_proteins.tsv` | Raw UniProt TSV response |
| `data/raw/uniprot_proteins.csv` | Raw UniProt data saved as CSV |
| `data/processed/uniprot_human_reviewed_clean.csv` | Standardized cleaned dataset |
| `data/processed/uniprot_proteins_filtered.csv` | Filtered dataset (50–1000 aa) |
| `data/processed/labeled_dataset.csv` | Labeled dataset with `function_class` (source of truth) |
| `galaxy_inputs/protein_ids.txt` | Full filtered UniProt accession list |
| `galaxy_inputs/protein_sequences.fasta` | Full filtered protein FASTA file |
| `galaxy_inputs/subsets/enzyme_ids.txt` | Enzyme subset |
| `galaxy_inputs/subsets/transporter_ids.txt` | Transporter subset |
| `galaxy_inputs/subsets/receptor_ids.txt` | Receptor subset |
| `galaxy_inputs/subsets/dna_rna_binding_ids.txt` | DNA/RNA-binding subset |
| `galaxy_inputs/subsets/background_all_ids.txt` | Background set for enrichment analysis |

## How to Run

From the project root, open and run:

```text
notebooks/001_data_collection.ipynb
```

Run the notebook from top to bottom.

Recommended order:

1. Set up project folders.
2. Install/import dependencies.
3. Fetch UniProt data.
4. Save raw TSV/CSV files.
5. Inspect dataset statistics.
6. Standardize column names.
7. Apply filtering rules (50–1000 aa).
8. Assign `function_class` labels and save the labeled dataset.
9. Export Galaxy input files.
10. Create annotation-based subsets.

## Notes

- Increase `MAX_PROTEINS` if a larger dataset is needed.
- Keep `batch_size` moderate to avoid very large API responses.
- Keep a small `sleep_time` between requests to avoid overloading the API.
- The 50–1000 aa cutoff keeps all sequences within the ESM-2 input limit; if you raise it, add truncation or sliding-window handling before computing embeddings.
- The labeling priority order directly affects how overlapping proteins are classified. If EDA shows a class being cannibalized, revisit the order and document the change.
- The filtered/labeled dataset is intended for downstream modeling and annotation analysis.
