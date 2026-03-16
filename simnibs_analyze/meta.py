# analyse de simulation différente ayant produit des features.csv différents
# au contraire _analysis.py se fait sur un seul .csv (oupsi là il y a une incohérence...)
# Il va falloir correctement définir dans quel cas il y a un seul csv ou dans quel cas il y e na plusieurs


# Liste les fichiers à comparer:
# - all_features.csv
# - le fichier d'analyze de groupe features moyennées etc - lequivalent de fsaverage ?
# - les fichiers niifti créés ? seulement si c'est pour faire des visualisations sinon il faut passer par all_features


... est-ce que ce que je veux faire c'est pas juste une diff de dataframe ??!!  

def load_csv():
    type: all_features ou autre
    process_param : les valeurs du config.yaml qui a aboutit au all_features


def _row_filter():
    # étant donné que c'est chargé comme ?
    # fonction pour selectionner les lignes d'interet par conditions de simulations:)


def _col_filter():
    #selection des colonnes à comparer 


def cluste

def compare():
    #est-cequon veut forcement comparé deux a deux... on va dire que oui et apres on peut faire toutes les comparaisons de maniere systematiques si on veut...
    csv_1 = load ...
    csv_2 = load ...
    

def run():
    
    
# -------- les analyses meta sur les segmentations: 
# j'aurais envie de dire on s'en fou un peu parceque c'est des analyses qui doivent etre fait en amont... pas apres les simulations et le traitement des simulations...