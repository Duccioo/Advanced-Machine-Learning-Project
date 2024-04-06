import rdkit.Chem as Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem


def calc_metrics(
    smiles_true: list = ["CCO", "CCN", "CCO", "CCC"],
    smiles_predicted: list = ["CCO", "CCN", "CCC", "CCF"],
):
    validity_smiles = [calculate_validity(mol) for mol in smiles_predicted]
    print("Molecole valide e uniche:")
    print(list(set(validity_smiles)))
    validity_percentage = sum(1.0 for elemento in validity_smiles if elemento is not False) / len(
        smiles_predicted
    )
    uniqueness_percentage = calculate_uniqueness(smiles_predicted, smiles_predicted)
    novelty_percentage = calculate_novelty_2(smiles_predicted, smiles_true)

    # print(validity_percentage, uniqueness_percentage, novelty_percentage)

    return validity_percentage, uniqueness_percentage, novelty_percentage


def calculate_validity(smiles):
    if smiles == "" or smiles == None:
        return False

    m = Chem.MolFromSmiles(smiles, sanitize=False)
    if m is None:
        # print("Invalid SMILES: %s" % smiles)
        return False
    else:
        try:
            m = Chem.SanitizeMol(m)
            return smiles
        except:
            # print("Invalid SMILES: %s" % smiles)
            return False


def calculate_uniqueness(mol_new, mol_old):
    unique_molecules = set()
    for mol in mol_new:
        # print(mol)
        unique_molecules.add(mol)

    # print(list(unique_molecules))

    return len(unique_molecules) / len(mol_old)


def calculate_novelty_2(molecole_predette, molecola_reale):
    val_pred_mol = [calculate_validity(mol) for mol in molecole_predette]

    return calculate_uniqueness(val_pred_mol, molecola_reale)


def calculate_novelty(molecole_predette, molecola_reale):
    # Converte la molecola reale in un fingerprint di Morgan
    molecola_reale = Chem.MolFromSmiles(molecola_reale)
    # fp_reale = AllChem.GetMorganFingerprintAsBitVect(molecola_reale, 2)

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
