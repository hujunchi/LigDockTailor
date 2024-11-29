import argparse
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors
from rdkit.Chem import AllChem, FilterCatalog
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from joblib import load
import zipfile
import os

def convert_sdf_to_mol(sdf_file):
    suppl = Chem.SDMolSupplier(sdf_file)
    mols = [mol for mol in suppl if mol is not None]
    return mols

def calculate_sp3_nitrogen(mol):
    sp3_count = 0
    total_nitrogen_count = 0
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 7:
            total_nitrogen_count += 1
            hybridization = atom.GetHybridization()
            if hybridization == Chem.HybridizationType.SP3:
                sp3_count += 1
    return sp3_count

def apply_structure_filter(mol_supplier, name_list, apply_filter=True):
    # 创建 PAINS 过滤器对象
    writer1 = Chem.SDWriter('./filter.sdf')
    filtered_names = []
    for i, mol in enumerate(mol_supplier):
        if mol is None:
            continue

        if apply_filter:
            # 使用分子量小于500和不是 PAINS 的条件进行筛选
            mol_weight = Descriptors.MolWt(mol)
            if mol_weight < 500:
                try:
                    # 构象优化
                    ff_mol = Chem.AddHs(mol)  # 添加氢原子
                    AllChem.EmbedMolecule(ff_mol, useRandomCoords=True)  # 随机生成初始构象
                    AllChem.MMFFOptimizeMolecule(ff_mol)  # 进行构象优化
                    writer1.write(ff_mol)
                    filtered_names.append(name_list[i])
                except:
                    continue
        else:
            writer1.write(mol)
            filtered_names.append(name_list[i])

    return filtered_names

def calculate_properties_predict(mols, names, output_csv_file,output_zip_file):
    
    data = []
    
    for i, mol in enumerate(mols):
        mol = Chem.AddHs(mol)
        properties = {
            'Molecule_Name': names[i],
            'MW': Descriptors.MolWt(mol),
            'HkA': rdMolDescriptors.CalcHallKierAlpha(mol),
            'FCsp3': Lipinski.FractionCSP3(mol),
            'NNsp3': calculate_sp3_nitrogen(mol),
            'TPSA': Descriptors.TPSA(mol),
            'XLogP': Descriptors.MolLogP(mol),
            'HBD': Lipinski.NumHDonors(mol),
            'HBA': Lipinski.NumHAcceptors(mol),
            'RB': Lipinski.NumRotatableBonds(mol),
            'AromR': rdMolDescriptors.CalcNumAromaticRings(mol),
            'AlipR': rdMolDescriptors.CalcNumAliphaticRings(mol),
            'SatuR': rdMolDescriptors.CalcNumSaturatedRings(mol)
        }
        
        ecfp = AllChem.GetMorganFingerprintAsBitVect(mol,radius=3, nBits=1024,useFeatures=False).ToBitString()
        for ct in range(1024):
            properties[f'Ext{ct+1}'] = ecfp[ct]  # 将指纹分别命名为"Ext1"到"Ext1024"
        
        data.append(properties)

    df = pd.DataFrame(data)

    # Predict by best model
    op_row1 = df.iloc[:,1:13].values
    scaler = MinMaxScaler()
    scaled_data1 = scaler.fit_transform(op_row1)
    df.iloc[:,1:13] = scaled_data1
    selected_rows =df.iloc[:, 1:].values
    X = np.array(selected_rows)
    loaded_model = load("/path/LigDockTailor_model.joblib")
    predictions_result = loaded_model.predict(X)
    y_PRED_prob = loaded_model.predict_proba(X)
    name_ls = ['Glide_SP', 'Glide_XP', 'MOE', 'GOLD', 'FlexX', 'Autodock_Vina', 'Autodock_Vinardo', 'LeDock', 'rDock']
    res = []
    for t in range(len(X)):
        cal = []
        for i in range(len(y_PRED_prob)):
            cal.append(y_PRED_prob[i][t][1])
        soft_rate = max(cal)
        res.append(name_ls[cal.index(soft_rate)])

    df.insert(1, 'Software', res)
    for n in name_ls:
        # 查找所有包含n的索引
        indices = [i for i, software in enumerate(res) if n in software]
        # 提取包含n的分子
        molecules = [filtered_mols[i] for i in indices if filtered_mols[i] is not None]

        # 保存提取的分子为新的SDF文件
        output_sdf_file = './' + n + '.sdf'
        writer = Chem.SDWriter(output_sdf_file)

        for molecule in molecules:
            writer.write(molecule)
    writer.close()
    # 将所有生成的 SDF 文件压缩到一个 ZIP 文件
    with zipfile.ZipFile(output_zip_file, 'w') as zip_file:
        for sdf_file in os.listdir('.'):
            if sdf_file.endswith('.sdf'):
                zip_file.write(sdf_file, os.path.basename(sdf_file))
                os.remove(sdf_file)
    df[['Molecule_Name', 'Software']].to_csv(output_csv_file, index=False)
      
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process molecular data.')
    parser.add_argument('-im', '--input_mol', type=str, help='Input molecule file')
    parser.add_argument('-in', '--input_name', type=str, help='Input molecule name file')
    parser.add_argument('-oc', '--output_csv_file', type=str, help='Output CSV file')
    parser.add_argument('-oz', '--output_zip_file', type=str, help='Output ZIP file')
    parser.add_argument('-on', '--output_name_file', type=str, help='Output molecule name file')

    args = parser.parse_args()

    mols = convert_sdf_to_mol(args.input_mol)
    names = []
    with open(args.input_name, 'r') as file:
        lines = file.readlines()
        names = [name.strip() for name in lines]


    filtered_names = apply_structure_filter(mols, names, apply_filter=True)
    filtered_mols = Chem.SDMolSupplier('./filter.sdf')
    calculate_properties_predict(filtered_mols, filtered_names, args.output_csv_file, args.output_zip_file)

    if args.structure_filter:
        with open(args.output_name_file, 'w') as file:
            file.writelines('\n'.join(filtered_names))