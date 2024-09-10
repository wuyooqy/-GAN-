import requests
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

# 检测是否可以进行可视化


# 网络优化与处理
def requests_get_with_retry(url, retries=3, timeout=5):
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=timeout)
            if response.status_code == 200:
                return response
            else:
                print(f"Attempt {attempt+1}: Received response code {response.status_code}")
        except requests.RequestException as e:
            print(f"Attempt {attempt+1}: Error {e}")
    return None

def can_generate_2d(smiles):
    api_url = "http://hulab.rxnfinder.org/smi2img/"
    request_url = f"{api_url}{smiles}"
    response = requests_get_with_retry(request_url)
    return response is not None

def can_generate_3d(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        mol_with_h = Chem.AddHs(mol)
        success = AllChem.EmbedMolecule(mol_with_h)
        return success >= 0
    except Exception as e:
        print(f"Error generating 3D for {smiles}: {e}")
        return False

smiles_data = pd.read_csv("epoch_data/SMILES.smi", header=None, names=['smiles'])
valid_smiles = []

for index, row in smiles_data.iterrows():
    smiles = row['smiles']
    if can_generate_2d(smiles) and can_generate_3d(smiles):
        valid_smiles.append(smiles)
    else:
        print(f"Cannot generate 2D or 3D for SMILES '{smiles}' at index {index}")

# 保存能够成功生成2D和3D可视化的SMILES字符串
valid_smiles_df = pd.DataFrame(valid_smiles, columns=['smiles'])
valid_smiles_file_path = "epoch_data/Valid_SMILES.smi"
valid_smiles_df.to_csv(valid_smiles_file_path, index=False, header=False)
print(f"Valid SMILES saved to {valid_smiles_file_path}")

