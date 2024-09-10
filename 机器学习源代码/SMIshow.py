import requests
import pandas as pd
from io import BytesIO
from PIL import Image
from rdkit import Chem
from rdkit.Chem import AllChem
from pathlib import Path
import logging
import os

# 2D可视化，3D结构文件生成

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

api_url = "http://hulab.rxnfinder.org/smi2img/"
smiles_data = pd.read_csv("epoch_data/Valid_SMILES.smi", header=None, names=['smiles'])

images_2D = Path("2D_images")
images_3D = Path("3D_images")

# 确保输出目录存在
images_2D.mkdir(parents=True, exist_ok=True)
images_3D.mkdir(parents=True, exist_ok=True)

def requests_get_with_retry(url, retries=3, timeout=5):
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=timeout)
            if response.status_code == 200:
                return response
        except requests.RequestException as e:
            logging.warning(f"Attempt {attempt+1}: Error {e}")
    return None

def generate_2d_image(smiles, index):
    response = requests_get_with_retry(f"{api_url}{smiles}")
    if response:
        image = Image.open(BytesIO(response.content))
        file_path = images_2D / f"image_{index}.png"
        image.save(file_path)
        logging.info(f"Saved 2D: {file_path}")
    else:
        logging.error(f"Error 2D: Failed to retrieve image for SMILES '{smiles}'")

def generate_3d_structure(smiles, index):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        mol_with_h = Chem.AddHs(mol)
        if AllChem.EmbedMolecule(mol_with_h) >= 0:
            try:
                AllChem.UFFOptimizeMolecule(mol_with_h)
            except:
                try:
                    AllChem.MMFFOptimizeMolecule(mol_with_h)
                except:
                    print(f"Warning: Could not optimize molecule for SMILES '{smiles}' using UFF or MMFF94.")
            pdb_file_path = images_3D / f"molecule_{index}.pdb"
            Chem.MolToPDBFile(mol_with_h, str(pdb_file_path))
            logging.info(f"Saved 3D structure to PDB: {pdb_file_path}")
        else:
            logging.error(f"Error 3D: Could not generate 3D structure for SMILES '{smiles}'")
    else:
        logging.error(f"Error 3D: Could not parse SMILES '{smiles}'")


for index, row in smiles_data.iterrows():
    smiles = row['smiles']
    generate_2d_image(smiles, index)
    generate_3d_structure(smiles, index)
