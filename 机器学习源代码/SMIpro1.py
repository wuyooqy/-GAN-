import os
from rdkit import Chem

# 生成分子的文件合并与规范化

smi_folder = 'epoch_data'
# 指定新文件的名称和路径
merged_file_path = os.path.join(smi_folder, 'SMILES.smi')

smi_files = [f for f in os.listdir(smi_folder) if f.endswith('.smi')]

count = 0
max_count = 1000

# 打开一个新文件用于写入合并后的内容
with open(merged_file_path, 'w') as merged_file:
    for smi_file in smi_files:
        if count >= max_count:
            break  # 如果已达到1000个，提前结束循环
        file_path = os.path.join(smi_folder, smi_file)
        with open(file_path, 'r') as file:
            for line in file:
                smiles = line.strip()
                # 将原始SMILES字符串转换为分子对象
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    # 从分子对象生成规范化的SMILES字符串
                    canonical_smiles = Chem.MolToSmiles(mol)
                    if count < max_count - 1:
                        merged_file.write(canonical_smiles + '\n')
                    else:
                        merged_file.write(canonical_smiles)
                    count += 1
                    if count >= max_count:
                        break

print('Merged .smi files into:', merged_file_path)
