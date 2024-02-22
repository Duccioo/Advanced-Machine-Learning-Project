import rdkit.Chem as Chem


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
    novelty_percentage = calculate_novelty(generated_molecules, training_set_molecules)

    print(f"Validità: {validity_percentage:.2%}")
    print(f"Unicità: {uniqueness_percentage:.2%}")
    print(f"Novità: {novelty_percentage:.2%}")


def calculate_validity(molecule):
    try:
        mol = Chem.MolFromSmiles(molecule)
        return mol is not None
    except:
        return False


def calculate_uniqueness(molecules):
    unique_molecules = set()
    for mol in molecules:
        unique_molecules.add(mol)
    return len(unique_molecules) / len(molecules)


def calculate_novelty(generated_molecules, training_set_molecules):
    novel_molecules = set(generated_molecules) - set(training_set_molecules)
    return len(novel_molecules) / len(generated_molecules)
