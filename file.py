from chembl_webresource_client.new_client import new_client

activity = new_client.activity
# Query IC50 values for a given ChEMBL target
res = activity.filter(target_chembl_id='CHEMBLXXXX', standard_type='IC50')
for entry in res:
    print(entry['molecule_chembl_id'], entry['standard_value'])  # Compound ID, IC50


from pubchempy import get_compounds
compound = get_compounds('CCO', 'smiles')[0]  # Ethanol SMILES → DrugBank ID
print(compound.drugbank_id)