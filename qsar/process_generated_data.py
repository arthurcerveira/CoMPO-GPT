import pandas as pd
import numpy as np
from rdkit.Chem import MACCSkeys
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import rdMolDescriptors
from pathlib import Path


def maccs(mol):

    try:
        maccs_fp = MACCSkeys.GenMACCSKeys(mol)
        return maccs_fp
    except:
        return None

def rdMolDes(mol):
    MDlist=[]
    try:
        MDlist.append(rdMolDescriptors.CalcTPSA(mol))
        MDlist.append(rdMolDescriptors.CalcFractionCSP3(mol))
        MDlist.append(rdMolDescriptors.CalcNumAliphaticCarbocycles(mol))
        MDlist.append(rdMolDescriptors.CalcNumAliphaticHeterocycles(mol))
        MDlist.append(rdMolDescriptors.CalcNumAliphaticRings(mol))
        MDlist.append(rdMolDescriptors.CalcNumAmideBonds(mol))
        MDlist.append(rdMolDescriptors.CalcNumAromaticCarbocycles(mol))
        MDlist.append(rdMolDescriptors.CalcNumAromaticHeterocycles(mol))
        MDlist.append(rdMolDescriptors.CalcNumAromaticRings(mol))
        MDlist.append(rdMolDescriptors.CalcNumHBA(mol))
        MDlist.append(rdMolDescriptors.CalcNumHBD(mol))
        MDlist.append(rdMolDescriptors.CalcNumLipinskiHBA(mol))
        MDlist.append(rdMolDescriptors.CalcNumLipinskiHBD(mol))
        MDlist.append(rdMolDescriptors.CalcNumHeteroatoms(mol))
        MDlist.append(rdMolDescriptors.CalcNumRings(mol))
        MDlist.append(rdMolDescriptors.CalcNumRotatableBonds(mol))
        MDlist.append(rdMolDescriptors.CalcNumSaturatedCarbocycles(mol))
        MDlist.append(rdMolDescriptors.CalcNumSaturatedHeterocycles(mol))
        MDlist.append(rdMolDescriptors.CalcNumSaturatedRings(mol))
        MDlist.append(rdMolDescriptors.CalcHallKierAlpha(mol))
        MDlist.append(rdMolDescriptors.CalcKappa1(mol))
        MDlist.append(rdMolDescriptors.CalcKappa2(mol))
        MDlist.append(rdMolDescriptors.CalcKappa3(mol))
        MDlist.append(rdMolDescriptors.CalcChi0n(mol))
        MDlist.append(rdMolDescriptors.CalcChi0v(mol))
        MDlist.append(rdMolDescriptors.CalcChi1n(mol))
        MDlist.append(rdMolDescriptors.CalcChi1v(mol))
        MDlist.append(rdMolDescriptors.CalcChi2n(mol))
        MDlist.append(rdMolDescriptors.CalcChi2v(mol))
        MDlist.append(rdMolDescriptors.CalcChi3n(mol))
        MDlist.append(rdMolDescriptors.CalcChi3v(mol))
        MDlist.append(rdMolDescriptors.CalcChi4n(mol))
        MDlist.append(rdMolDescriptors.CalcChi4v(mol))
        MDlist.append(rdMolDescriptors.CalcAsphericity(mol))
        MDlist.append(rdMolDescriptors.CalcEccentricity(mol))
        MDlist.append(rdMolDescriptors.CalcInertialShapeFactor(mol))
        MDlist.append(rdMolDescriptors.CalcExactMolWt(mol))
        MDlist.append(rdMolDescriptors.CalcPBF(
            mol))  # Returns the PBF (plane of best fit) descriptor (http://dx.doi.org/10.1021/ci300293f)
        MDlist.append(rdMolDescriptors.CalcPMI1(mol))
        MDlist.append(rdMolDescriptors.CalcPMI2(mol))
        MDlist.append(rdMolDescriptors.CalcPMI3(mol))
        MDlist.append(rdMolDescriptors.CalcRadiusOfGyration(mol))
        MDlist.append(rdMolDescriptors.CalcSpherocityIndex(mol))
        # MDlist.append(rdMolDescriptors.CalcNumBridgeheadAtoms(mol))
        # MDlist.append(rdMolDescriptors.CalcNumAtomStereoCenters(mol))
        # MDlist.append(rdMolDescriptors.CalcNumHeterocycles(mol))
        # MDlist.append(rdMolDescriptors.CalcNumSpiroAtoms(mol))
        # MDlist.append(rdMolDescriptors.CalcNumUnspecifiedAtomStereoCenters(mol))
        MDlist.append(rdMolDescriptors.CalcLabuteASA(mol))
        MDlist.append(rdMolDescriptors.CalcNPR1(mol))
        MDlist.append(rdMolDescriptors.CalcNPR2(mol))
        # for d in rdMolDescriptors.CalcGETAWAY(mol): #197 descr (http://www.vcclab.org/lab/indexhlp/getades.html)
        #    MDlist.append(d)
        for d in rdMolDescriptors.PEOE_VSA_(mol):  # 14 descr
            MDlist.append(d)
        for d in rdMolDescriptors.SMR_VSA_(mol):  # 10 descr
            MDlist.append(d)
        for d in rdMolDescriptors.SlogP_VSA_(mol):  # 12 descr
            MDlist.append(d)
        for d in rdMolDescriptors.MQNs_(mol):  # 42 descr
            MDlist.append(d)
        for d in rdMolDescriptors.CalcCrippenDescriptors(mol):  # 2 descr
            MDlist.append(d)
        for d in rdMolDescriptors.CalcAUTOCORR2D(mol):  # 192 descr
            MDlist.append(d)
        return MDlist
    except:
        return None


def get_fp(name, sdf_path='./sdf'):
    """ Function to get fingerprint from a list of SMILES"""
    fingerprints = []
    smiles = []
    sdFile = Chem.SDMolSupplier(f'{sdf_path}/{name}.sdf')
    inchikey =[]
    for mol in sdFile:

        try:

            inchi = Chem.MolToInchiKey(mol)


            if inchi not in inchikey:
                inchikey.append(inchi)

                fprint = Chem.GetMorganFingerprintAsBitVect(mol, 3, nBits = 2048,useFeatures=True)

                # fingerprints.append()

                rd = rdMolDes(mol)
                if rd is None:
                    continue
                ma = maccs(mol)
                if ma is None:
                    continue
                feature = [x for x in fprint.ToBitString()] + rd  + [x for x in ma.ToBitString()]
                fingerprints.append(feature)
                smi = Chem.MolToSmiles(mol)
                smiles.append(smi)

        except:
            print('error')


    return fingerprints, smiles


def clean(name, sdf_path='./sdf', csv_path='../data'):
    mergedSDF_OUT = Chem.SDWriter(f'{sdf_path}/{name}.sdf')
    df = pd.read_csv(f'{csv_path}/{name}.csv')
    for index, row in df.iterrows():
        try:
            mol = Chem.MolFromSmiles(row['SMILES'])
            mergedSDF_OUT.write(mol)
        except:
            print('error')
    mergedSDF_OUT.close()


# names =[
#     'EGFR_RNN',
#     'EGFR_Transformer',
#     'HTR1A_RNN',
#     'HTR1A_Transformer',
#     'S1PR1_RNN',
#     'S1PR1_Transformer',
# ]

# for name in names:
#     clean(name)
#     fingers, activities, smiles1 = get_fp(name)
#     res = pd.DataFrame(fingers, columns=list(range(2533))).astype('float')
#     df = pd.DataFrame({'SMILES':smiles1})
#     df.to_csv('smiles/{}.smi'.format(name),index=False)
#     fp_array = res.to_numpy()
#     label = np.array(activities)
#     np.save('npy/{}_X.npy'.format(name),fp_array)

# Compare no-target to single-target conditional generation
gen_mols_folder = Path('../generated_molecules').resolve()
sdf_folder = gen_mols_folder / 'sdf'

targets = ['Unconditional', 'EGFR', 'HTR1A', 'S1PR1']

for target in targets:
    clean(target, sdf_path=sdf_folder, csv_path=gen_mols_folder)

    fingers, smiles1 = get_fp(target, sdf_path=sdf_folder)
    res = pd.DataFrame(fingers, columns=list(range(2533))).astype('float')
    df = pd.DataFrame({'SMILES':smiles1})
    df.to_csv(gen_mols_folder / f'smiles/{target}.smi', index=False)
    fp_array = res.to_numpy()
    np.save(gen_mols_folder / f'npy/{target}_X.npy', fp_array)
