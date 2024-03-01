import rdkit.Chem as Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem


def calc_metrics(
    smiles_true: list = ["CCO", "CCN", "CCO", "CCC"],
    smiles_predicted: list = ["CCO", "CCN", "CCC", "CCF"],
):
    # Esempio di utilizzo:
    generated_molecules = smiles_predicted
    training_set_molecules = smiles_true

    validity_percentage = sum(
        calculate_validity(mol) for mol in generated_molecules
    ) / len(generated_molecules)
    uniqueness_percentage = calculate_uniqueness(generated_molecules)
    # novelty_percentage = calculate_novelty(generated_molecules, training_set_molecules)

    print(f"Validità: {validity_percentage:.2%}")
    print(f"Unicità: {uniqueness_percentage:.2%}")
    # print(f"Novità: {novelty_percentage:.2%}")


def calculate_validity(molecule):
    try:
        mol = Chem.MolFromSmiles(molecule)
    except:
        return False

    flag = Chem.SanitizeMol(mol, catchErrors=True)
    # Let's be strict. If sanitization fails, return None
    if flag != Chem.SanitizeFlags.SANITIZE_NONE:
        return None

    else:
        return mol is not None


def calculate_uniqueness(molecules):
    unique_molecules = set()
    for mol in molecules:
        unique_molecules.add(mol)
    return len(unique_molecules) / len(molecules)


def calculate_novelty(molecole_predette, molecola_reale):
    # Converte la molecola reale in un fingerprint di Morgan
    molecola_reale = Chem.MolFromSmiles(molecola_reale)
    fp_reale = AllChem.GetMorganFingerprintAsBitVect(molecola_reale, 2)

    novelty_scores = []

    for smiles in molecole_predette:
        # Converte ogni molecola predetta in un fingerprint di Morgan
        molecola_predetta = Chem.MolFromSmiles(smiles)
        fp_predetto = AllChem.GetMorganFingerprintAsBitVect(molecola_predetta, 2)

        # Calcola la similarità di Tanimoto tra la molecola reale e la molecola predetta
        similitudine = DataStructs.TanimotoSimilarity(fp_reale, fp_predetto)

        # La novelty è 1 - similitudine
        novelty = 1 - similitudine
        novelty_scores.append(novelty)

    return novelty_scores
