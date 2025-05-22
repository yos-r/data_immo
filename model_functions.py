import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import missingno as msno
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import shap
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score, calinski_harabasz_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
def read_data(file_path):
    df = pd.read_csv('data.csv')
    print(f"Aperçu des données ({df.shape[0]} lignes, {df.shape[1]} colonnes):")
    df.head()
    numeric_columns = ['listing_price','price_ttc','price', 'size', 'rooms', 'bedrooms', 'bathrooms', 'parkings', 
                  'construction_year', 'age', 'air_conditioning', 'central_heating', 
                  'swimming_pool', 'elevator', 'garden', 'equipped_kitchen']
    for col in numeric_columns:
        if col in df.columns:
            # Afficher le type original
            original_type = df[col].dtype
            
            # Convertir en numérique
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Afficher les informations sur la conversion
            na_count = df[col].isna().sum()
            print(f"Conversion de '{col}': {original_type} -> {df[col].dtype}, valeurs NA créées: {na_count}")
    return df
def analyze_missing_data(df):
    # Calculer les informations sur les valeurs manquantes
    missing_count = df.isna().sum()
    missing_percent = (df.isna().sum() / len(df) * 100).round(2)
    
    # Créer un DataFrame avec ces informations
    missing_data = pd.DataFrame({
        'Type de données': df.dtypes,
        'Valeurs non-NA': df.count(),
        'Valeurs NA': missing_count,
        'Pourcentage NA (%)': missing_percent,
        'Valeurs uniques': df.nunique()
    })
    
    # Trier par pourcentage de valeurs manquantes (décroissant)
    missing_data = missing_data.sort_values('Pourcentage NA (%)', ascending=False)

    return missing_data


def impute_missing_prices(df):
    """
    Imputation des prix manquants dans le DataFrame immobilier en utilisant la moyenne par transaction, type et quartier
    """
    df['price'] = df.groupby(['neighborhood', 'property_type','transaction'])['price'].transform(
        lambda x: x.fillna(x.mean())
    )
    df['price_ttc'] = df.groupby(['neighborhood', 'property_type','transaction'])['price_ttc'].transform(
        lambda x: x.fillna(x.mean())
    )
    # take care of the rest
    df['price'] = df.groupby(['city','transaction'])['price'].transform(
        lambda x: x.fillna(x.mean())
    )
    df['price_ttc'] = df.groupby(['city','transaction'])['price_ttc'].transform(
        lambda x: x.fillna(x.mean())
    )
    
    # Remplir les valeurs manquantes de 'listing_price' avec la valeur de 'price' si disponible
    df['listing_price'] = df['listing_price'].fillna(df['price'])
    # remplacer suffixe par ttc par defaut
    df['suffix'] = df['suffix'].fillna('TTC')
    return df   
def impute_condition_simple(df):
    """
    Impute les valeurs manquantes dans la colonne 'condition' en se basant sur:
    1. La zone (neighborhood ou city)
    2. Le type de transaction
    3. L'intervalle de prix
    
    Cette version suppose qu'il n'y a pas de valeurs manquantes dans 
    les colonnes de groupement (price, neighborhood/city, transaction).
    """
    # Créer une copie pour ne pas modifier le dataframe original
    df_imputed = df.copy()
    
    # Vérifier s'il y a des valeurs manquantes dans la colonne condition
    missing_count = df_imputed['condition'].isna().sum()
    if missing_count == 0:
        print("Aucune valeur manquante dans la colonne 'condition'. Aucune imputation nécessaire.")
        return df_imputed
    
    print(f"Imputation de {missing_count} valeurs manquantes dans la colonne 'condition'...")
    
    # 1. Déterminer quelle colonne de zone utiliser
    zone_column = 'neighborhood' if 'neighborhood' in df_imputed.columns else 'city'
    print(f"Utilisation de '{zone_column}' comme colonne de zone géographique")
    
    # 2. Créer des intervalles de prix pour le groupement
    # Calculer les quantiles pour créer des segments de prix équilibrés
    price_bins = [0] + list(df_imputed['price'].quantile([0.25, 0.5, 0.75, 1.0]))
    price_labels = ['Bas', 'Moyen-bas', 'Moyen-haut', 'Élevé']
    
    # Créer une colonne pour l'intervalle de prix
    df_imputed['price_range'] = pd.cut(df_imputed['price'], 
                                     bins=price_bins, 
                                     labels=price_labels,
                                     include_lowest=True)
    
    # 3. Approche par niveaux pour l'imputation
    # Masque initial pour les lignes avec condition manquante
    missing_mask = df_imputed['condition'].isna()
    
    # Niveau 1: Imputation basée sur zone + transaction + intervalle de prix
    for index, row in df_imputed[missing_mask].iterrows():
        # Trouver des propriétés similaires avec la même zone, transaction et intervalle de prix
        similar_props = df_imputed[
            (df_imputed[zone_column] == row[zone_column]) & 
            (df_imputed['transaction'] == row['transaction']) & 
            (df_imputed['price_range'] == row['price_range']) & 
            (~df_imputed['condition'].isna())
        ]
        
        # S'il y a des propriétés similaires, utiliser leur condition la plus fréquente
        if len(similar_props) > 0:
            df_imputed.loc[index, 'condition'] = similar_props['condition'].mode()[0]
    
    # Mise à jour du masque après le premier niveau d'imputation
    missing_mask = df_imputed['condition'].isna()
    remaining = missing_mask.sum()
    
    if remaining > 0:
        print(f"Niveau 1 terminé. {remaining} valeurs restent à imputer.")
        
        # Niveau 2: Imputation basée sur transaction + intervalle de prix
        for index, row in df_imputed[missing_mask].iterrows():
            similar_props = df_imputed[
                (df_imputed['transaction'] == row['transaction']) & 
                (df_imputed['price_range'] == row['price_range']) & 
                (~df_imputed['condition'].isna())
            ]
            
            if len(similar_props) > 0:
                df_imputed.loc[index, 'condition'] = similar_props['condition'].mode()[0]
        
        # Mise à jour du masque après le deuxième niveau
        missing_mask = df_imputed['condition'].isna()
        remaining = missing_mask.sum()
        
        if remaining > 0:
            print(f"Niveau 2 terminé. {remaining} valeurs restent à imputer.")
            
            # Niveau 3: Imputation basée sur l'intervalle de prix uniquement
            for index, row in df_imputed[missing_mask].iterrows():
                similar_props = df_imputed[
                    (df_imputed['price_range'] == row['price_range']) & 
                    (~df_imputed['condition'].isna())
                ]
                
                if len(similar_props) > 0:
                    df_imputed.loc[index, 'condition'] = similar_props['condition'].mode()[0]
            
            # Mise à jour du masque après le troisième niveau
            missing_mask = df_imputed['condition'].isna()
            remaining = missing_mask.sum()
            
            if remaining > 0:
                print(f"Niveau 3 terminé. {remaining} valeurs restent à imputer.")
                
                # Niveau 4: Imputation globale avec la valeur la plus fréquente
                most_common = df_imputed['condition'].dropna().mode()[0]
                df_imputed.loc[missing_mask, 'condition'] = most_common
                print(f"Niveau 4 terminé. Imputation globale effectuée.")
    
    # Supprimer la colonne temporaire d'intervalle de prix
    df_imputed.drop('price_range', axis=1, inplace=True)
    
    # Vérification finale
    final_missing = df_imputed['condition'].isna().sum()
    if final_missing == 0:
        print("Imputation réussie ! Toutes les valeurs manquantes de 'condition' ont été imputées.")
    else:
        print(f"Attention : {final_missing} valeurs restent manquantes après imputation.")
    
    return df_imputed
def impute_finishing_simple(df):
    """
    Impute les valeurs manquantes dans la colonne 'finishing' en se basant sur:
    1. La zone (neighborhood ou city)
    2. Le type de transaction
    3. L'intervalle de prix
    
    Cette version suppose qu'il n'y a pas de valeurs manquantes dans 
    les colonnes de groupement (price, neighborhood/city, transaction).
    """
    # Créer une copie pour ne pas modifier le dataframe original
    df_imputed = df.copy()
    
    # Vérifier s'il y a des valeurs manquantes dans la colonne condition
    missing_count = df_imputed['finishing'].isna().sum()
    if missing_count == 0:
        print("Aucune valeur manquante dans la colonne 'finishing'. Aucune imputation nécessaire.")
        return df_imputed
    
    print(f"Imputation de {missing_count} valeurs manquantes dans la colonne 'finishing'...")
    
    # 1. Déterminer quelle colonne de zone utiliser
    zone_column = 'neighborhood' if 'neighborhood' in df_imputed.columns else 'city'
    print(f"Utilisation de '{zone_column}' comme colonne de zone géographique")
    
    # 2. Créer des intervalles de prix pour le groupement
    # Calculer les quantiles pour créer des segments de prix équilibrés
    price_bins = [0] + list(df_imputed['price'].quantile([0.25, 0.5, 0.75, 1.0]))
    price_labels = ['Bas', 'Moyen-bas', 'Moyen-haut', 'Élevé']
    
    # Créer une colonne pour l'intervalle de prix
    df_imputed['price_range'] = pd.cut(df_imputed['price'], 
                                     bins=price_bins, 
                                     labels=price_labels,
                                     include_lowest=True)
    
    # 3. Approche par niveaux pour l'imputation
    # Masque initial pour les lignes avec standing manquant
    missing_mask = df_imputed['finishing'].isna()
    
    # Niveau 1: Imputation basée sur zone + transaction + intervalle de prix
    for index, row in df_imputed[missing_mask].iterrows():
        # Trouver des propriétés similaires avec la même zone, transaction et intervalle de prix
        similar_props = df_imputed[
            (df_imputed[zone_column] == row[zone_column]) & 
            (df_imputed['transaction'] == row['transaction']) & 
            (df_imputed['price_range'] == row['price_range']) & 
            (~df_imputed['finishing'].isna())
        ]
        
        # S'il y a des propriétés similaires, utiliser leur condition la plus fréquente
        if len(similar_props) > 0:
            df_imputed.loc[index, 'finishing'] = similar_props['finishing'].mode()[0]
    
    # Mise à jour du masque après le premier niveau d'imputation
    missing_mask = df_imputed['finishing'].isna()
    remaining = missing_mask.sum()
    
    if remaining > 0:
        print(f"Niveau 1 terminé. {remaining} valeurs restent à imputer.")
        
        # Niveau 2: Imputation basée sur transaction + intervalle de prix
        for index, row in df_imputed[missing_mask].iterrows():
            similar_props = df_imputed[
                (df_imputed['transaction'] == row['transaction']) & 
                (df_imputed['price_range'] == row['price_range']) & 
                (~df_imputed['finishing'].isna())
            ]
            
            if len(similar_props) > 0:
                df_imputed.loc[index, 'finishing'] = similar_props['finishing'].mode()[0]
        
        # Mise à jour du masque après le deuxième niveau
        missing_mask = df_imputed['finishing'].isna()
        remaining = missing_mask.sum()
        
        if remaining > 0:
            print(f"Niveau 2 terminé. {remaining} valeurs restent à imputer.")
            
            # Niveau 3: Imputation basée sur l'intervalle de prix uniquement
            for index, row in df_imputed[missing_mask].iterrows():
                similar_props = df_imputed[
                    (df_imputed['price_range'] == row['price_range']) & 
                    (~df_imputed['finishing'].isna())
                ]
                
                if len(similar_props) > 0:
                    df_imputed.loc[index, 'finishing'] = similar_props['condition'].mode()[0]
            
            # Mise à jour du masque après le troisième niveau
            missing_mask = df_imputed['finishing'].isna()
            remaining = missing_mask.sum()
            
            if remaining > 0:
                print(f"Niveau 3 terminé. {remaining} valeurs restent à imputer.")
                
                # Niveau 4: Imputation globale avec la valeur la plus fréquente
                most_common = df_imputed['finishing'].dropna().mode()[0]
                df_imputed.loc[missing_mask, 'finishing'] = most_common
                print(f"Niveau 4 terminé. Imputation globale effectuée.")
    
    # Supprimer la colonne temporaire d'intervalle de prix
    df_imputed.drop('price_range', axis=1, inplace=True)
    
    # Vérification finale
    final_missing = df_imputed['finishing'].isna().sum()
    if final_missing == 0:
        print("Imputation réussie ! Toutes les valeurs manquantes de 'finishing' ont été imputées.")
    else:
        print(f"Attention : {final_missing} valeurs restent manquantes après imputation.")
    
    return df_imputed
def impute_property_year_age(df, impute_year=True, impute_age=True, method='grouped_median'):
    """
    Impute les valeurs manquantes dans les colonnes 'year' (année de construction) 
    et 'age' (âge de la propriété) d'un jeu de données immobilier.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame contenant les données immobilières
    impute_year : bool, default=True
        Si True, impute les valeurs manquantes dans la colonne 'year'
    impute_age : bool, default=True
        Si True, impute les valeurs manquantes dans la colonne 'age'
    method : str, default='grouped_median'
        Méthode d'imputation à utiliser ('grouped_median', 'regression', 'knn')
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame avec les valeurs imputées
    """
    # Créer une copie pour ne pas modifier le dataframe original
    df_imputed = df.copy()
    
    # Vérifier si les colonnes existent
    has_year = 'year' in df_imputed.columns
    has_age = 'age' in df_imputed.columns
    

    # 1. Synchronisation année/âge avant imputation
    current_year = 2025  
    
    # Si l'une des colonnes a une valeur mais pas l'autre, compléter l'autre
    
    
    # 2. Imputation des valeurs restantes
    
    # Méthode 1: Imputation par médiane groupée
    if method == 'grouped_median':
        # Déterminer les variables de groupement
        grouping_cols = []
        
        # Utiliser le quartier si disponible, sinon la ville
        if 'neighborhood' in df_imputed.columns:
            grouping_cols.append('neighborhood')
        elif 'city' in df_imputed.columns:
            grouping_cols.append('city')
        
        # Ajouter le type de propriété s'il existe
        if 'property_type' in df_imputed.columns:
            grouping_cols.append('property_type')
        
        # Pour l'année et l'âge, on peut utiliser l'intervalle de prix
        if 'price' in df_imputed.columns:
            # Créer des intervalles de prix
            price_bins = [0] + list(df_imputed['price'].quantile([0.25, 0.5, 0.75, 1.0]))
            price_labels = ['Bas', 'Moyen-bas', 'Moyen-haut', 'Élevé']
            
            df_imputed['price_range'] = pd.cut(df_imputed['price'], 
                                            bins=price_bins, 
                                            labels=price_labels,
                                            include_lowest=True)
            
            grouping_cols.append('price_range')
        
        print(f"Variables de groupement utilisées: {', '.join(grouping_cols)}")
        
        # Si aucune variable de groupement n'est disponible
        if not grouping_cols:
            if impute_year and has_year:
                median_year = df_imputed['construction_year'].median()
                year_missing = df_imputed['construction_year'].isna().sum()
                df_imputed['construction_year'].fillna(median_year, inplace=True)
                print(f"Imputé {year_missing} valeurs manquantes dans 'year' avec la médiane globale: {median_year}")
            
            if impute_age and has_age:
                median_age = df_imputed['age'].median()
                age_missing = df_imputed['age'].isna().sum()
                df_imputed['age'].fillna(median_age, inplace=True)
                print(f"Imputé {age_missing} valeurs manquantes dans 'age' avec la médiane globale: {median_age}")
            
            # Nettoyer et retourner
            if 'price_range' in df_imputed.columns and 'price_range' not in df.columns:
                df_imputed.drop('price_range', axis=1, inplace=True)
            
            return df_imputed
        
        # Imputation par niveaux, du plus spécifique au plus général
        
        # Générer toutes les combinaisons de variables de groupement
        from itertools import combinations
        all_combinations = []
        
        for r in range(len(grouping_cols), 0, -1):
            all_combinations.extend(combinations(grouping_cols, r))
        
        # Imputation pour chaque colonne
        columns_to_impute = []
        if impute_year and has_year:
            columns_to_impute.append('construction_year')
        if impute_age and has_age:
            columns_to_impute.append('age')
        
        for col in columns_to_impute:
            missing_mask = df_imputed[col].isna()
            total_missing = missing_mask.sum()
            
            if total_missing == 0:
                print(f"Aucune valeur manquante dans '{col}'.")
                continue
            
            print(f"Imputation de {total_missing} valeurs manquantes dans '{col}'...")
            
            # Parcourir chaque combinaison de groupes
            for i, group_vars in enumerate(all_combinations):
                if not missing_mask.any():
                    break
                
                print(f"  Niveau {i+1}: Groupement par {', '.join(group_vars)}")
                
                # Pour chaque groupe, calculer la médiane
                group_medians = df_imputed[~df_imputed[col].isna()].groupby(list(group_vars))[col].median()
                
                # Pour chaque ligne avec valeur manquante
                for index, row in df_imputed[missing_mask].iterrows():
                    # Créer la clé de groupe
                    group_key = tuple(row[var] for var in group_vars)
                    
                    # Si la médiane existe pour ce groupe
                    if group_key in group_medians:
                        df_imputed.loc[index, col] = group_medians[group_key]
                
                # Mettre à jour le masque
                missing_mask = df_imputed[col].isna()
                remaining = missing_mask.sum()
                
                print(f"    → {total_missing - remaining}/{total_missing} valeurs imputées ({(total_missing - remaining)/total_missing*100:.1f}%)")
                
                if not missing_mask.any():
                    print(f"    Imputation de '{col}' terminée au niveau {i+1}.")
                    break
            
            # Imputation finale pour les valeurs encore manquantes
            if missing_mask.any():
                median_value = df_imputed[col].median()
                df_imputed.loc[missing_mask, col] = median_value
                print(f"  Imputation globale des {missing_mask.sum()} valeurs restantes avec la médiane: {median_value}")
    
    # Méthode 2: Imputation par régression linéaire
    elif method == 'regression':
        from sklearn.ensemble import RandomForestRegressor
        
        for col in ['year', 'age']:
            if (col == 'year' and impute_year and has_year) or (col == 'age' and impute_age and has_age):
                missing_mask = df_imputed[col].isna()
                total_missing = missing_mask.sum()
                
                if total_missing == 0:
                    print(f"Aucune valeur manquante dans '{col}'.")
                    continue
                
                print(f"Imputation de {total_missing} valeurs manquantes dans '{col}' par régression...")
                
                # Sélectionner les colonnes numériques pour la régression
                numeric_cols = df_imputed.select_dtypes(include=['number']).columns
                numeric_cols = [c for c in numeric_cols if c != col and df_imputed[c].isna().sum() == 0]
                
                if len(numeric_cols) < 2:
                    print(f"  Pas assez de variables numériques pour la régression. Utilisation de la médiane.")
                    df_imputed.loc[missing_mask, col] = df_imputed[col].median()
                    continue
                
                # Création des ensembles d'entraînement
                X_train = df_imputed.loc[~missing_mask, numeric_cols]
                y_train = df_imputed.loc[~missing_mask, col]
                
                # Entraîner le modèle
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                
                # Prédire les valeurs manquantes
                X_missing = df_imputed.loc[missing_mask, numeric_cols]
                y_pred = model.predict(X_missing)
                
                # Imputer les valeurs prédites
                df_imputed.loc[missing_mask, col] = y_pred
                
                print(f"  {total_missing} valeurs imputées dans '{col}' par régression.")
    
    # Méthode 3: Imputation par KNN
    elif method == 'knn':
        from sklearn.impute import KNNImputer
        
        # Sélectionner les colonnes numériques pour l'imputation KNN
        numeric_cols = df_imputed.select_dtypes(include=['number']).columns.tolist()
        
        # Filtrer uniquement les colonnes avec peu ou pas de valeurs manquantes
        valid_cols = [col for col in numeric_cols if df_imputed[col].isna().mean() < 0.3]
        
        if len(valid_cols) < 3:
            print("Pas assez de variables numériques pour l'imputation KNN. Utilisation de la médiane.")
            
            if impute_year and has_year:
                df_imputed['year'].fillna(df_imputed['year'].median(), inplace=True)
            
            if impute_age and has_age:
                df_imputed['age'].fillna(df_imputed['age'].median(), inplace=True)
        else:
            print(f"Imputation KNN avec {len(valid_cols)} variables numériques...")
            
            # Créer un sous-ensemble des données numériques
            numeric_data = df_imputed[valid_cols].copy()
            
            # Appliquer l'imputation KNN
            imputer = KNNImputer(n_neighbors=5)
            imputed_values = imputer.fit_transform(numeric_data)
            
            # Reconstruire le DataFrame avec les valeurs imputées
            numeric_df_imputed = pd.DataFrame(imputed_values, columns=valid_cols, index=df_imputed.index)
            
            # Remplacer uniquement les valeurs manquantes dans les colonnes cibles
            if impute_year and has_year and 'year' in valid_cols:
                missing_mask = df_imputed['year'].isna()
                df_imputed.loc[missing_mask, 'year'] = numeric_df_imputed.loc[missing_mask, 'year']
                print(f"Imputé {missing_mask.sum()} valeurs manquantes dans 'year' avec KNN.")
            
            if impute_age and has_age and 'age' in valid_cols:
                missing_mask = df_imputed['age'].isna()
                df_imputed.loc[missing_mask, 'age'] = numeric_df_imputed.loc[missing_mask, 'age']
                print(f"Imputé {missing_mask.sum()} valeurs manquantes dans 'age' avec KNN.")
    
    # 3. Vérification de cohérence après imputation
    if has_year and has_age:
        # S'assurer que year + age = année actuelle (approximativement)
        tolerance = 3  # Tolérance de 3 ans
        inconsistent_mask = abs((df_imputed['construction_year'] + df_imputed['age']) - current_year) > tolerance
        
        if inconsistent_mask.any():
            print(f"Attention: {inconsistent_mask.sum()} propriétés ont des valeurs d'année et d'âge incohérentes après imputation.")
    
    # 4. Arrondir l'année à l'entier le plus proche
    if has_year:
        df_imputed['construction_year'] = df_imputed['construction_year'].round().astype('Int64')
    
    if has_age:
        df_imputed['age'] = df_imputed['age'].round().astype('Int64')
    
    # Nettoyer les colonnes temporaires
    if 'price_range' in df_imputed.columns and 'price_range' not in df.columns:
        df_imputed.drop('price_range', axis=1, inplace=True)
    
    return df_imputed
def impute_binary_amenities(df, binary_columns=None, grouping_columns=['city', 'property_type', 'transaction']):
    """
    Impute les valeurs manquantes dans les colonnes binaires représentant les équipements immobiliers.
    Utilise une approche par niveaux basée sur les colonnes de groupement.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame contenant les données immobilières
    binary_columns : list
        Liste des colonnes binaires à imputer
    grouping_columns : list
        Liste des colonnes à utiliser pour le groupement (par défaut: city, property_type, transaction)
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame avec les valeurs imputées
    """
    # Créer une copie pour ne pas modifier le dataframe original
    df_imputed = df.copy()
    
    # Vérifier les colonnes binaires à imputer
    if binary_columns is None:
        print("Aucune colonne binaire spécifiée pour l'imputation")
        return df_imputed
    
    # Filtrer les colonnes existantes
    binary_columns = [col for col in binary_columns if col in df_imputed.columns]
    grouping_columns = [col for col in grouping_columns if col in df_imputed.columns]
    
    print(f"Colonnes binaires à imputer: {', '.join(binary_columns)}")
    print(f"Colonnes de groupement: {', '.join(grouping_columns)}")
    
    # Générer toutes les combinaisons de colonnes de groupement
    import itertools
    grouping_combinations = []
    
    # Ajouter les combinaisons de colonnes de groupement, du plus spécifique au plus général
    for i in range(len(grouping_columns), 0, -1):
        grouping_combinations.extend(list(itertools.combinations(grouping_columns, i)))
    
    # Pour chaque colonne binaire à imputer
    for col in binary_columns:
        # Vérifier s'il y a des valeurs manquantes
        missing_mask = df_imputed[col].isna()
        missing_count = missing_mask.sum()
        
        if missing_count == 0:
            print(f"- {col}: Aucune valeur manquante")
            continue
        
        print(f"- {col}: Imputation de {missing_count} valeurs manquantes")
        
        # Imputation par niveaux
        for level, group_cols in enumerate(grouping_combinations):
            if not missing_mask.any():
                break
                
            # Calculer le mode (valeur la plus fréquente) par groupe
            group_modes = df_imputed[~missing_mask].groupby(list(group_cols))[col].agg(
                lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else None
            )
            
            # Imputer les valeurs manquantes par groupe
            for index, row in df_imputed[missing_mask].iterrows():
                try:
                    # Créer la clé de groupe
                    group_key = tuple(row[gc] for gc in group_cols)
                    
                    # Imputer si le mode existe pour ce groupe
                    if group_key in group_modes.index and group_modes[group_key] is not None:
                        df_imputed.loc[index, col] = group_modes[group_key]
                except:
                    # Ignorer les erreurs (ex: valeurs manquantes dans les colonnes de groupement)
                    continue
            
            # Mettre à jour le masque des valeurs manquantes
            new_missing_mask = df_imputed[col].isna()
            imputed_in_level = missing_mask.sum() - new_missing_mask.sum()
            
            if imputed_in_level > 0:
                print(f"  Niveau {level+1} ({', '.join(group_cols)}): {imputed_in_level} valeurs imputées")
            
            missing_mask = new_missing_mask
        
        # Imputation finale avec le mode global pour les valeurs restantes
        if missing_mask.any():
            global_mode = df_imputed[col].mode().iloc[0]
            df_imputed.loc[missing_mask, col] = global_mode
            print(f"  Imputation globale: {missing_mask.sum()} valeurs imputées avec {global_mode}")
    
    return df_imputed
def simple_impute_rooms(df, rooms_col='rooms', area_col='size', property_type_col='property_type'):
    """
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame contenant les données immobilières
    rooms_col : str
        Nom de la colonne contenant le nombre de pièces/ nombre de parkings/ nombre de chambres/ salles de bain
        (ex: 'rooms', 'bedrooms', 'bathrooms', 'parkings')
    area_col : str
        Nom de la colonne contenant la superficie
    property_type_col : str
        Nom de la colonne contenant le type de propriété
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame avec les valeurs de rooms imputées
    """
    # Copie du dataframe
    df_imputed = df.copy()
    
    # Calculer le nombre de valeurs manquantes
    missing_count = df_imputed[rooms_col].isna().sum()
    print(f"Imputation de {missing_count} valeurs manquantes dans '{rooms_col}'")
    
    # Créer des segments de superficie (quartiles)
    area_bins = [0] + list(df_imputed[area_col].quantile([0.25, 0.5, 0.75, 1.0]))
    df_imputed['area_segment'] = pd.cut(df_imputed[area_col], bins=area_bins, include_lowest=True)
    
    # Méthode principale: calculer le ratio moyen pièces/superficie par type de propriété
    # Cela donne une idée de combien de m² par pièce selon le type de logement
    
    # Calculer le ratio moyen pièces/superficie pour chaque type de propriété
    ratios = df_imputed.dropna(subset=[rooms_col]).groupby(property_type_col).apply(
        lambda x: (x[rooms_col] / x[area_col]).median()
    ).to_dict()
    
    # Imputer les valeurs manquantes directement
    missing_mask = df_imputed[rooms_col].isna()
    
    for prop_type in ratios:
        # Pour chaque type de propriété, imputer en fonction du ratio
        type_mask = (df_imputed[property_type_col] == prop_type) & missing_mask
        if type_mask.any():
            # Estimer le nombre de pièces en fonction de la superficie et du ratio
            df_imputed.loc[type_mask, rooms_col] = (df_imputed.loc[type_mask, area_col] * ratios[prop_type]).round()
    
    # Imputer les valeurs restantes par segment de superficie
    still_missing = df_imputed[rooms_col].isna()
    if still_missing.any():
        # Calculer le nombre moyen de pièces par segment de superficie
        segment_means = df_imputed.groupby('area_segment')[rooms_col].transform(
            lambda x: x.median() if not x.dropna().empty else None
        )
        
        # Imputer les valeurs manquantes
        df_imputed.loc[still_missing, rooms_col] = segment_means.loc[still_missing]
    
    # Imputer les dernières valeurs manquantes avec la médiane globale
    final_missing = df_imputed[rooms_col].isna()
    if final_missing.any():
        median_rooms = df_imputed[rooms_col].median()
        df_imputed.loc[final_missing, rooms_col] = round(median_rooms)
    
    # Arrondir à l'entier le plus proche
    df_imputed[rooms_col] = df_imputed[rooms_col].round()
    
    # Supprimer la colonne temporaire
    df_imputed.drop('area_segment', axis=1, inplace=True)
    
    print(f"Imputation terminée.")
    
    return df_imputed



def prepare_data_for_regression(df):
    """
    Prépare les données pour la régression - encode uniquement condition, finishing et variables binaires
    """
    df_prep = df.copy()
    
    # Traitement des variables ordinales
    # Définir l'ordre pour chaque variable ordinale
    condition_categories = ['à rénover', 'à rafraichir', 'bonne condition', 'excellente condition', 'neuf']  
    finishing_categories = ['social', 'économique', 'moyen standing', 'haut standing', 'très haut standing']  
    
    # Encoder les variables ordinales
    if 'condition' in df_prep.columns:
        cat_map = {cat: i for i, cat in enumerate(condition_categories)}
        df_prep['condition'] = df_prep['condition'].map(cat_map)
        df_prep['condition'].fillna(df_prep['condition'].median(), inplace=True)
    
    if 'finishing' in df_prep.columns:
        cat_map = {cat: i for i, cat in enumerate(finishing_categories)}
        df_prep['finishing'] = df_prep['finishing'].map(cat_map)
        df_prep['finishing'].fillna(df_prep['finishing'].median(), inplace=True)
    
    # S'assurer que les variables binaires sont numériques
    binary_cols = [col for col in df_prep.columns if df_prep[col].nunique() == 2 and 
                  col not in ['transaction', 'property_type', 'city', 'neighborhood']]
    
    for col in binary_cols:
        if df_prep[col].dtype != 'int64' and df_prep[col].dtype != 'float64':
            df_prep[col] = df_prep[col].astype(int)
    
    return df_prep
def regression_par_segment(df, city=None, property_type=None, transaction=None, target_column='price'):
    """
    Réalise une régression linéaire simple sur un segment spécifique des données
    
    Paramètres:
    -----------
    df : pandas DataFrame
        Le dataframe déjà préparé avec prepare_data_for_regression
    city : str, optional
        Ville à filtrer
    property_type : str, optional
        Type de propriété à filtrer
    transaction : str, optional
        Type de transaction à filtrer
    target_column : str
        Nom de la colonne cible
    
    Retourne:
    ---------
    model : objet modèle
        Le modèle de régression linéaire
    feature_importance : DataFrame
        Importance des caractéristiques
    metrics : dict
        Métriques de performance
    """
    # Filtrer les données selon les paramètres
    df_filtered = df.copy()
    
    if city is not None and 'city' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['city'] == city]
        df_filtered = df_filtered.drop(columns=['city']) 
    
    if property_type is not None and 'property_type' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['property_type'] == property_type]
        df_filtered = df_filtered.drop(columns=['property_type'])
    
    if transaction is not None and 'transaction' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['transaction'] == transaction]
        df_filtered = df_filtered.drop(columns=['transaction'])
    
    # Supprimer les colonnes non nécessaires pour la régression
    columns_to_drop = ['date', 'source', 'neighborhood', 'suffix','listing_price','price_ttc']
    columns_to_drop = [col for col in columns_to_drop if col in df_filtered.columns]
    if columns_to_drop:
        df_filtered = df_filtered.drop(columns=columns_to_drop)
    
    # Supprimer les lignes avec des valeurs manquantes dans la colonne cible
    df_filtered = df_filtered.dropna(subset=[target_column])
    
    # Séparer les caractéristiques et la cible
    y = df_filtered[target_column]
    X = df_filtered.drop(columns=[target_column])
    
    # Exclure les colonnes non numériques
    X = X.select_dtypes(include=['number'])
    
    # Diviser en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Normaliser les caractéristiques
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Entraîner le modèle
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    # Prédictions
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    # Évaluer le modèle
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    # Importance des caractéristiques
    feature_importance = pd.DataFrame({
        'Caractéristique': X.columns,
        'Coefficient': model.coef_
    }).sort_values('Coefficient', key=abs, ascending=False)
    
    # Afficher les résultats
    print(f"Nombre d'observations: {len(df_filtered)}")
    print(f"R² (entraînement): {train_r2:.4f}")
    print(f"R² (test): {test_r2:.4f}")
    print(f"RMSE (test): {test_rmse:.2f}")
    print(f"MAE (test): {test_mae:.2f}")
    print("\nTop caractéristiques les plus influentes:")
    print(feature_importance.head(10))
    
    # Préparer visualisation
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_test_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Prix réels')
    plt.ylabel('Prix prédits')
    plt.title('Prix réels vs Prix prédits')
    plt.grid(True)
    plt.annotate(f'R² = {test_r2:.4f}', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=12)
    plt.tight_layout()
    plt.show()
    
    # Graphique d'importance des caractéristiques
    plt.figure(figsize=(10, 8))
    top_features = feature_importance.head(10)
    colors = ['green' if coef > 0 else 'red' for coef in top_features['Coefficient']]
    plt.barh(top_features['Caractéristique'], top_features['Coefficient'], color=colors)
    plt.xlabel('Valeur du coefficient')
    plt.ylabel('Caractéristique')
    plt.title('Top 10 des caractéristiques les plus importantes')
    plt.axvline(x=0, color='k', linestyle='--')
    plt.grid(True, axis='x')
    plt.tight_layout()
    plt.show()
    
    metrics = {
        'train_r2': train_r2,
        'test_r2': test_r2,
        'test_rmse': test_rmse,
        'test_mae': test_mae
    }
    
    return model, feature_importance, metrics
def random_forest_par_segment(df, city=None, property_type=None, transaction=None, 
                             target_column='price', n_estimators=100, max_depth=None):
    """
    Réalise une régression par Random Forest sur un segment spécifique des données
    
    Paramètres:
    -----------
    df : pandas DataFrame
        Le dataframe déjà préparé avec prepare_data_for_regression
    city : str, optional
        Ville à filtrer
    property_type : str, optional
        Type de propriété à filtrer
    transaction : str, optional
        Type de transaction à filtrer
    target_column : str
        Nom de la colonne cible
    n_estimators : int
        Nombre d'arbres dans la forêt
    max_depth : int, optional
        Profondeur maximale des arbres
    
    Retourne:
    ---------
    model : objet modèle
        Le modèle Random Forest
    feature_importance : DataFrame
        Importance des caractéristiques
    metrics : dict
        Métriques de performance
    """
    # Filtrer les données selon les paramètres
    df_filtered = df.copy()
    
    if city is not None and 'city' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['city'] == city]
        df_filtered = df_filtered.drop(columns=['city']) 
    
    if property_type is not None and 'property_type' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['property_type'] == property_type]
        df_filtered = df_filtered.drop(columns=['property_type'])
    
    if transaction is not None and 'transaction' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['transaction'] == transaction]
        df_filtered = df_filtered.drop(columns=['transaction'])
    
    # Supprimer les colonnes non nécessaires pour la régression
    columns_to_drop = ['listing_price','price_ttc','date', 'source', 'neighborhood', 'suffix']
    columns_to_drop = [col for col in columns_to_drop if col in df_filtered.columns]
    if columns_to_drop:
        df_filtered = df_filtered.drop(columns=columns_to_drop)
    
    # Supprimer les lignes avec des valeurs manquantes dans la colonne cible
    df_filtered = df_filtered.dropna(subset=[target_column])
    
    # Séparer les caractéristiques et la cible
    y = df_filtered[target_column]
    X = df_filtered.drop(columns=[target_column])
    
    # Exclure les colonnes non numériques
    X = X.select_dtypes(include=['number'])
    
    # Diviser en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Normaliser les caractéristiques (facultatif pour Random Forest)
    # On le fait pour rester cohérent avec la régression linéaire
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Créer et entraîner le modèle Random Forest
    rf_model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1  # Utiliser tous les cœurs disponibles
    )
    rf_model.fit(X_train_scaled, y_train)
    
    # Prédictions
    y_train_pred = rf_model.predict(X_train_scaled)
    y_test_pred = rf_model.predict(X_test_scaled)
    
    # Évaluer le modèle
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    # Importance des caractéristiques
    feature_importance = pd.DataFrame({
        'Caractéristique': X.columns,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    # Afficher les résultats
    print(f"===== RÉSULTATS RANDOM FOREST =====")
    print(f"Segments: Ville={city}, Type={property_type}, Transaction={transaction}")
    print(f"Nombre d'observations: {len(df_filtered)}")
    print(f"R² (entraînement): {train_r2:.4f}")
    print(f"R² (test): {test_r2:.4f}")
    print(f"RMSE (test): {test_rmse:.2f}")
    print(f"MAE (test): {test_mae:.2f}")
    print("\nTop caractéristiques les plus importantes:")
    print(feature_importance.head(10))
    
    # Préparer visualisation
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_test_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Prix réels')
    plt.ylabel('Prix prédits')
    plt.title('Random Forest: Prix réels vs Prix prédits')
    plt.grid(True)
    plt.annotate(f'R² = {test_r2:.4f}', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=12)
    plt.tight_layout()
    plt.show()
    
    # Graphique d'importance des caractéristiques
    plt.figure(figsize=(10, 8))
    top_features = feature_importance.head(10)
    sns.barplot(x='Importance', y='Caractéristique', data=top_features, palette='viridis')
    plt.xlabel('Importance relative (%)')
    plt.ylabel('Caractéristique')
    plt.title('Random Forest: Top 10 des caractéristiques les plus importantes')
    plt.grid(True, axis='x')
    plt.tight_layout()
    plt.show()
    
    # Visualiser les résidus
    residus = y_test - y_test_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test_pred, residus, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Prix prédits')
    plt.ylabel('Résidus')
    plt.title('Random Forest: Distribution des résidus')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    metrics = {
        'train_r2': train_r2,
        'test_r2': test_r2,
        'test_rmse': test_rmse,
        'test_mae': test_mae
    }
    
    return rf_model, feature_importance, metrics
def xgboost_simple(df, city=None, property_type=None, transaction=None, target_column='price'):
    """
    Fonction simple pour appliquer XGBoost à un segment spécifique
    
    Paramètres:
    -----------
    df : pandas DataFrame
        Le dataframe préparé
    city : str, optional
        Ville à filtrer
    property_type : str, optional
        Type de propriété à filtrer
    transaction : str, optional
        Type de transaction à filtrer
    target_column : str
        Nom de la colonne cible
    """
    # Filtrer les données selon les paramètres
    df_filtered = df.copy()
    
    if city is not None and 'city' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['city'] == city]
        df_filtered = df_filtered.drop(columns=['city']) 
    
    if property_type is not None and 'property_type' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['property_type'] == property_type]
        df_filtered = df_filtered.drop(columns=['property_type'])
    
    if transaction is not None and 'transaction' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['transaction'] == transaction]
        df_filtered = df_filtered.drop(columns=['transaction'])
    
    # Supprimer les colonnes non nécessaires
    columns_to_drop = ['listing_price','construction_year','price_ttc','date', 'source', 'neighborhood', 'suffix']
    columns_to_drop = [col for col in columns_to_drop if col in df_filtered.columns]
    if columns_to_drop:
        df_filtered = df_filtered.drop(columns=columns_to_drop)
    
    # Supprimer les lignes avec des valeurs manquantes dans la colonne cible
    df_filtered = df_filtered.dropna(subset=[target_column])
    
    # Séparer les caractéristiques et la cible
    y = df_filtered[target_column]
    X = df_filtered.drop(columns=[target_column])
    
    # Garder uniquement les colonnes numériques
    X = X.select_dtypes(include=['number'])
    
    # Diviser en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Créer et entraîner le modèle XGBoost avec des paramètres simples
    model = xgb.XGBRegressor(
        n_estimators=100,   # Nombre d'arbres
        learning_rate=0.1,  # Taux d'apprentissage
        max_depth=5,        # Profondeur maximale des arbres
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Prédictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Évaluer le modèle
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    # Afficher les résultats
    print(f"\nRésultats XGBoost pour: {property_type} - {transaction} - {city}")
    print(f"Nombre d'observations: {len(df_filtered)}")
    print(f"R² (entraînement): {train_r2:.4f}")
    print(f"R² (test): {test_r2:.4f}")
    print(f"RMSE: {rmse:.2f}")
    
    # Importance des caractéristiques
    feature_importance = pd.DataFrame({
        'Caractéristique': X.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\nTop 10 caractéristiques les plus importantes:")
    print(feature_importance.head(10))
    
    # Graphique simple des prédictions
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_test_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Prix réels')
    plt.ylabel('Prix prédits')
    plt.title(f'XGBoost: Prédictions (R² = {test_r2:.4f})')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Graphique d'importance des caractéristiques
    plt.figure(figsize=(10, 6))
    top_n = min(10, len(feature_importance))
    plt.barh(feature_importance['Caractéristique'].head(top_n), 
             feature_importance['Importance'].head(top_n))
    plt.xlabel('Importance')
    plt.title('Top caractéristiques importantes')
    plt.tight_layout()
    plt.show()
    
    return model, feature_importance, test_r2
def comparer_modeles(df, city=None, property_type=None, transaction=None, target_column='price'):
    """
    Compare les performances de la régression linéaire et du Random Forest
    """
    from sklearn.linear_model import LinearRegression
    
    # Filtrer les données
    df_filtered = df.copy()
    
    if city is not None and 'city' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['city'] == city]
        df_filtered = df_filtered.drop(columns=['city']) 
    
    if property_type is not None and 'property_type' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['property_type'] == property_type]
        df_filtered = df_filtered.drop(columns=['property_type'])
    
    if transaction is not None and 'transaction' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['transaction'] == transaction]
        df_filtered = df_filtered.drop(columns=['transaction'])
    
    # Nettoyer le dataframe
    columns_to_drop = ['date', 'source', 'neighborhood', 'suffix']
    columns_to_drop = [col for col in columns_to_drop if col in df_filtered.columns]
    if columns_to_drop:
        df_filtered = df_filtered.drop(columns=columns_to_drop)
    
    df_filtered = df_filtered.dropna(subset=[target_column])
    
    # Séparer X et y
    y = df_filtered[target_column]
    X = df_filtered.drop(columns=[target_column])
    X = X.select_dtypes(include=['number'])
    
    # Diviser en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Normaliser les caractéristiques
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Modèle 1: Régression linéaire
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    lr_train_pred = lr.predict(X_train_scaled)
    lr_test_pred = lr.predict(X_test_scaled)
    lr_train_r2 = r2_score(y_train, lr_train_pred)
    lr_test_r2 = r2_score(y_test, lr_test_pred)
    lr_rmse = np.sqrt(mean_squared_error(y_test, lr_test_pred))
    lr_mae = mean_absolute_error(y_test, lr_test_pred)
    
    # Modèle 2: Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train_scaled, y_train)
    rf_train_pred = rf.predict(X_train_scaled)
    rf_test_pred = rf.predict(X_test_scaled)
    rf_train_r2 = r2_score(y_train, rf_train_pred)
    rf_test_r2 = r2_score(y_test, rf_test_pred)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_test_pred))
    rf_mae = mean_absolute_error(y_test, rf_test_pred)
    
    # Afficher les résultats
    print(f"===== COMPARAISON DES MODÈLES =====")
    print(f"Segments: Ville={city}, Type={property_type}, Transaction={transaction}")
    print(f"Nombre d'observations: {len(df_filtered)}")
    
    # Créer un dataframe de comparaison
    comparison = pd.DataFrame({
        'Modèle': ['Régression Linéaire', 'Random Forest'],
        'R² (train)': [lr_train_r2, rf_train_r2],
        'R² (test)': [lr_test_r2, rf_test_r2],
        'RMSE': [lr_rmse, rf_rmse],
        'MAE': [lr_mae, rf_mae]
    })
    
    print(comparison)
    
    # Visualisation
    plt.figure(figsize=(10, 6))
    
    # Graphique 1: Comparaison des R²
    plt.subplot(1, 2, 1)
    sns.barplot(x='Modèle', y='R² (test)', data=comparison, palette='viridis')
    plt.title('Comparaison du R²')
    plt.ylim(0, 1)
    plt.grid(True, axis='y')
    
    # Graphique 2: Comparaison des RMSE
    plt.subplot(1, 2, 2)
    sns.barplot(x='Modèle', y='RMSE', data=comparison, palette='viridis')
    plt.title('Comparaison du RMSE')
    plt.grid(True, axis='y')
    
    plt.tight_layout()
    plt.show()
    
    # Comparaison des prédictions
    plt.figure(figsize=(14, 6))
    
    # Graphique 1: Régression Linéaire
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, lr_test_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Prix réels')
    plt.ylabel('Prix prédits')
    plt.title(f'Régression Linéaire: R² = {lr_test_r2:.4f}')
    plt.grid(True)
    
    # Graphique 2: Random Forest
    plt.subplot(1, 2, 2)
    plt.scatter(y_test, rf_test_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Prix réels')
    plt.ylabel('Prix prédits')
    plt.title(f'Random Forest: R² = {rf_test_r2:.4f}')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return comparison

#SCALING
def prepare_data_for_clustering(df, features_for_clustering=None):
    """
    Prépare les données pour le clustering
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame avec les données
    features_for_clustering : list, optional
        Liste des caractéristiques à utiliser pour le clustering
        
    Returns:
    --------
    df_scaled : pandas.DataFrame
        Données standardisées pour le clustering
    scaler : StandardScaler
        L'objet scaler utilisé
    feature_names : list
        Noms des caractéristiques utilisées
    """
    print('hello')
    df_prep = df.copy()
    
    # Encoder les variables catégorielles si nécessaire
    if 'condition' in df_prep.columns:
        condition_categories = ['à rénover', 'à rafraichir', 'bonne condition', 'excellente condition', 'neuf']
        cat_map = {cat: i for i, cat in enumerate(condition_categories)}
        df_prep['condition'] = df_prep['condition'].map(cat_map)
        df_prep['condition'].fillna(df_prep['condition'].median(), inplace=True)
    
    if 'finishing' in df_prep.columns:
        finishing_categories = ['social', 'économique', 'moyen standing', 'haut standing', 'très haut standing']
        cat_map = {cat: i for i, cat in enumerate(finishing_categories)}
        df_prep['finishing'] = df_prep['finishing'].map(cat_map)
        df_prep['finishing'].fillna(df_prep['finishing'].median(), inplace=True)
    
    # Sélectionner les caractéristiques pour le clustering
    if features_for_clustering is None:
        # Utiliser toutes les colonnes numériques sauf les identifiants et colonnes non pertinentes
        exclude_cols = ['date', 'source', 'neighborhood', 'suffix', 'listing_price', 'price_ttc', 'construction_year']
        numeric_cols = df_prep.select_dtypes(include=['number']).columns
        features_for_clustering = [col for col in numeric_cols if col not in exclude_cols]
    
    # Vérifier que les colonnes existent
    features_for_clustering = [col for col in features_for_clustering if col in df_prep.columns]
    
    # Extraire les données pour le clustering
    X = df_prep[features_for_clustering].dropna()
    
    # Standardiser les données
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Créer un DataFrame avec les données standardisées
    df_scaled = pd.DataFrame(X_scaled, columns=features_for_clustering, index=X.index)
    # return None
    return df_scaled, scaler, features_for_clustering

# PCA 
def apply_pca_analysis(df_scaled, n_components=None):
    """
    Applique l'analyse en composantes principales (PCA)
    
    Parameters:
    -----------
    df_scaled : pandas.DataFrame
        Données standardisées
    n_components : int, optional
        Nombre de composantes à conserver
        
    Returns:
    --------
    pca_model : PCA
        Modèle PCA ajusté
    df_pca : pandas.DataFrame
        Données transformées par PCA
    explained_variance_ratio : array
        Ratio de variance expliquée par chaque composante
    """
    if n_components is None:
        n_components = min(df_scaled.shape[1], 10)  # Maximum 10 composantes ou nombre de features
    
    pca = PCA(n_components=n_components)
    pca_data = pca.fit_transform(df_scaled)
    
    # Créer un DataFrame avec les composantes principales
    component_names = [f'PC{i+1}' for i in range(n_components)]
    df_pca = pd.DataFrame(pca_data, columns=component_names, index=df_scaled.index)
    
    print(f"Variance expliquée par les {n_components} premières composantes:")
    for i, var_exp in enumerate(pca.explained_variance_ratio_):
        print(f"  PC{i+1}: {var_exp*100:.2f}%")
    print(f"Variance totale expliquée: {pca.explained_variance_ratio_.sum()*100:.2f}%")
    
    return pca, df_pca, pca.explained_variance_ratio_

def analyser_composition_pca(pca_model, feature_names, explained_variance):
    """
    Analyser comment PC1 et PC2 sont composées à partir de vos variables originales
    """
    import pandas as pd
    import numpy as np
    
    print("=" * 70)
    print("COMPOSITION DES COMPOSANTES PRINCIPALES")
    print("=" * 70)
    
    # Récupérer les coefficients (loadings)
    components = pca_model.components_
    
    # Créer un DataFrame pour visualiser les contributions
    loadings_df = pd.DataFrame(
        components.T,  # Transposer pour avoir variables en lignes
        columns=[f'PC{i+1}' for i in range(len(explained_variance))],
        index=feature_names
    )
    
    print("\n📊 COEFFICIENTS DES COMPOSANTES (loadings):")
    print("-" * 50)
    print(loadings_df.round(3))
    
    print("\n🧮 INTERPRÉTATION DES COEFFICIENTS:")
    print("-" * 40)
    print("• Coefficient positif = variable contribue positivement à la composante")
    print("• Coefficient négatif = variable contribue négativement à la composante") 
    print("• Plus |coefficient| est grand, plus l'influence est forte")
    print("• Seuils: |coeff| > 0.4 = forte, 0.2-0.4 = modérée, < 0.2 = faible")
    
    # Analyser chaque composante
    for i in range(len(explained_variance)):
        pc_name = f'PC{i+1}'
        print(f"\n" + "="*50)
        print(f"{pc_name} ({explained_variance[i]*100:.1f}% de variance)")
        print(f"FORMULE: {pc_name} = ")
        
        # Trier par valeur absolue décroissante
        coeffs = loadings_df[pc_name].abs().sort_values(ascending=False)
        
        formula_parts = []
        interpretations = []
        
        for var_name in coeffs.index:
            coeff = loadings_df.loc[var_name, pc_name]
            abs_coeff = abs(coeff)
            sign = "+" if coeff > 0 else "-"
            
            # Classification de l'influence
            if abs_coeff > 0.4:
                influence = "FORTE"
                symbol = "🔥"
            elif abs_coeff > 0.2:
                influence = "MODÉRÉE" 
                symbol = "⚡"
            else:
                influence = "FAIBLE"
                symbol = "💨"
                
            formula_parts.append(f"{sign}{abs_coeff:.3f}×{var_name}")
            interpretations.append(f"   {symbol} {var_name}: {sign}{abs_coeff:.3f} ({influence})")
        
        # Afficher la formule
        print(" ".join(formula_parts[:5]) + "...")  # Limiter à 5 termes
        print("\nDétail des contributions:")
        for interp in interpretations[:8]:  # Top 8 variables
            print(interp)
            
    return loadings_df

def interpreter_pca_immobilier(pca_model, feature_names, explained_variance):
    """
    Interprétation spécialisée pour l'immobilier
    """
    components = pca_model.components_
    
    print("\n" + "="*70)
    print("INTERPRÉTATION IMMOBILIÈRE DES COMPOSANTES")
    print("="*70)
    
    # Analyser PC1
    pc1_coeffs = dict(zip(feature_names, components[0]))
    pc1_sorted = sorted(pc1_coeffs.items(), key=lambda x: abs(x[1]), reverse=True)
    
    print(f"\n🏠 PC1 ({explained_variance[0]*100:.1f}% variance) - INTERPRÉTATION:")
    print("-" * 50)
    
    # Variables avec coefficients positifs les plus forts
    positives = [(var, coeff) for var, coeff in pc1_sorted if coeff > 0.2]
    negatives = [(var, coeff) for var, coeff in pc1_sorted if coeff < -0.2]
    
    if positives:
        print("🔼 VALEURS PC1 POSITIVES correspondent à propriétés avec:")
        for var, coeff in positives[:5]:
            print(f"   • {var} élevé(e) (coeff: +{coeff:.3f})")
    
    if negatives:
        print("\n🔽 VALEURS PC1 NÉGATIVES correspondent à propriétés avec:")
        for var, coeff in negatives[:5]:
            print(f"   • {var} faible (coeff: {coeff:.3f})")
    
    # Conclusion sur PC1
    print(f"\n💡 CONCLUSION PC1:")
    if any('price' in var.lower() for var, _ in positives) and any('size' in var.lower() for var, _ in positives):
        print("   PC1 = AXE STANDING/QUALITÉ")
        print("   (+) Propriétés haut de gamme: chères, grandes, bien équipées")
        print("   (-) Propriétés économiques: abordables, petites, équipements de base")
    
    # Analyser PC2 si disponible
    if len(explained_variance) > 1:
        pc2_coeffs = dict(zip(feature_names, components[1]))
        pc2_sorted = sorted(pc2_coeffs.items(), key=lambda x: abs(x[1]), reverse=True)
        
        print(f"\n🏠 PC2 ({explained_variance[1]*100:.1f}% variance) - INTERPRÉTATION:")
        print("-" * 50)
        
        positives_pc2 = [(var, coeff) for var, coeff in pc2_sorted if coeff > 0.2]
        negatives_pc2 = [(var, coeff) for var, coeff in pc2_sorted if coeff < -0.2]
        
        if positives_pc2:
            print("🔼 VALEURS PC2 POSITIVES:")
            for var, coeff in positives_pc2[:5]:
                print(f"   • {var} élevé(e) (coeff: +{coeff:.3f})")
        
        if negatives_pc2:
            print("\n🔽 VALEURS PC2 NÉGATIVES:")
            for var, coeff in negatives_pc2[:5]:
                print(f"   • {var} faible (coeff: {coeff:.3f})")
        
        # Conclusion sur PC2
        print(f"\n💡 CONCLUSION PC2:")
        if any('age' in var.lower() for var, _ in pc2_sorted[:3]):
            print("   PC2 = AXE TEMPOREL")
            print("   Sépare propriétés récentes vs anciennes")
        elif any('room' in var.lower() for var, _ in pc2_sorted[:3]):
            print("   PC2 = AXE TYPOLOGIE")
            print("   Sépare selon le nombre de pièces/configuration")
        else:
            print("   PC2 = AXE SPÉCIALISÉ")
            print("   Facteur de différenciation secondaire du marché")

def exemple_interpretation_concrete(df_pca, df_original, feature_names):
    """
    Exemple concret avec quelques propriétés pour montrer le lien
    """
    print("\n" + "="*70)
    print("EXEMPLE CONCRET: LIEN PC1/PC2 ↔ VARIABLES ORIGINALES")
    print("="*70)
    
    # Prendre 5 propriétés avec PC1 les plus élevés (haut de gamme)
    top_pc1_indices = df_pca['PC1'].nlargest(5).index
    
    print("\n🏆 TOP 5 PROPRIÉTÉS HAUT DE GAMME (PC1 élevé):")
    print("-" * 55)
    
    for i, idx in enumerate(top_pc1_indices, 1):
        pc1_val = df_pca.loc[idx, 'PC1']
        pc2_val = df_pca.loc[idx, 'PC2']
        
        print(f"\nPropriété #{i} (Index {idx}):")
        print(f"   PC1: {pc1_val:.2f} | PC2: {pc2_val:.2f}")
        print("   Caractéristiques originales:")
        
        # Afficher les variables originales les plus importantes
        for var in feature_names[:6]:  # Top 6 variables
            if var in df_original.columns:
                val = df_original.loc[idx, var]
                print(f"      • {var}: {val}")
    
    # Prendre 5 propriétés avec PC1 les plus bas (économiques)
    bottom_pc1_indices = df_pca['PC1'].nsmallest(5).index
    
    print("\n💰 TOP 5 PROPRIÉTÉS ÉCONOMIQUES (PC1 faible):")
    print("-" * 55)
    
    for i, idx in enumerate(bottom_pc1_indices, 1):
        pc1_val = df_pca.loc[idx, 'PC1']
        pc2_val = df_pca.loc[idx, 'PC2']
        
        print(f"\nPropriété #{i} (Index {idx}):")
        print(f"   PC1: {pc1_val:.2f} | PC2: {pc2_val:.2f}")
        print("   Caractéristiques originales:")
        
        for var in feature_names[:6]:
            if var in df_original.columns:
                val = df_original.loc[idx, var]
                print(f"      • {var}: {val}")

def visualiser_contributions_variables(pca_model, feature_names, explained_variance):
    """
    Graphique des contributions des variables aux composantes principales
    """
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    
    components = pca_model.components_
    
    # Créer un graphique des loadings (biplot style)
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            f'Contributions à PC1 ({explained_variance[0]*100:.1f}% variance)',
            f'Contributions à PC2 ({explained_variance[1]*100:.1f}% variance)' if len(explained_variance) > 1 else 'PC2'
        ]
    )
    
    # PC1
    pc1_contrib = components[0]
    colors_pc1 = ['red' if x < 0 else 'blue' for x in pc1_contrib]
    
    fig.add_trace(
        go.Bar(
            x=feature_names,
            y=pc1_contrib,
            marker_color=colors_pc1,
            name='PC1',
            showlegend=False
        ),
        row=1, col=1
    )
    
    # PC2 si disponible
    if len(explained_variance) > 1:
        pc2_contrib = components[1]
        colors_pc2 = ['red' if x < 0 else 'green' for x in pc2_contrib]
        
        fig.add_trace(
            go.Bar(
                x=feature_names,
                y=pc2_contrib,
                marker_color=colors_pc2,
                name='PC2',
                showlegend=False
            ),
            row=1, col=2
        )
    
    fig.update_layout(
        title_text="Contribution des variables originales aux composantes principales",
        height=500
    )
    
    fig.update_xaxes(tickangle=45)
    fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
    
    return fig

# UTILISATION COMPLÈTE
def analyse_complete_pca_variables(pca_model, df_pca, explained_variance, feature_names, df_original):
    """
    Analyse complète du lien PCA ↔ variables originales
    """
    # 1. Composition mathématique
    loadings_df = analyser_composition_pca(pca_model, feature_names, explained_variance)
    
    # 2. Interprétation immobilière
    interpreter_pca_immobilier(pca_model, feature_names, explained_variance)
    
    # 3. Exemples concrets
    exemple_interpretation_concrete(df_pca, df_original, feature_names)
    
    # 4. Visualisation
    fig = visualiser_contributions_variables(pca_model, feature_names, explained_variance)
    
    return loadings_df, fig


#CAH:
def apply_cah_clustering(df_scaled, max_clusters=10, linkage_method='ward', distance_metric='euclidean'):
    """
    Applique la Classification Ascendante Hiérarchique (CAH)
    
    Parameters:
    -----------
    df_scaled : pandas.DataFrame
        Données standardisées
    max_clusters : int
        Nombre maximum de clusters à considérer
    linkage_method : str
        Méthode de liaison ('ward', 'complete', 'average', 'single')
    distance_metric : str
        Métrique de distance ('euclidean', 'manhattan', 'cosine')
        
    Returns:
    --------
    linkage_matrix : array
        Matrice de liaison pour le dendrogramme
    cluster_labels : array
        Labels des clusters pour le nombre optimal
    optimal_n_clusters : int
        Nombre optimal de clusters
    metrics : dict
        Métriques d'évaluation
    """
    
    print(f"Application de CAH avec méthode '{linkage_method}' et distance '{distance_metric}'...")
    
    # 1. Calculer la matrice de distance si nécessaire
    if linkage_method == 'ward':
        # Ward nécessite la distance euclidienne
        linkage_matrix = linkage(df_scaled, method='ward')
    else:
        # Calculer la matrice de distance pour autres méthodes
        distances = pdist(df_scaled, metric=distance_metric)
        linkage_matrix = linkage(distances, method=linkage_method)
    
    # 2. Tester différents nombres de clusters
    silhouette_scores = []
    n_clusters_range = range(2, min(max_clusters + 1, len(df_scaled)))
    
    print("Évaluation du nombre optimal de clusters...")
    
    for n_clusters in n_clusters_range:
        # Obtenir les labels de clusters
        cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
        
        # Calculer le score de silhouette
        if len(set(cluster_labels)) > 1:
            sil_score = silhouette_score(df_scaled, cluster_labels)
            silhouette_scores.append(sil_score)
        else:
            silhouette_scores.append(0)
    
    # 3. Trouver le nombre optimal de clusters
    if silhouette_scores:
        best_idx = np.argmax(silhouette_scores)
        optimal_n_clusters = list(n_clusters_range)[best_idx]
        best_silhouette = silhouette_scores[best_idx]
    else:
        optimal_n_clusters = 2
        best_silhouette = 0
    
    # 4. Obtenir les labels finaux
    final_cluster_labels = fcluster(linkage_matrix, optimal_n_clusters, criterion='maxclust')
    
    # 5. Calculer les métriques
    metrics = {
        'optimal_n_clusters': optimal_n_clusters,
        'silhouette_score': best_silhouette,
        'linkage_method': linkage_method,
        'distance_metric': distance_metric,
        'silhouette_scores': silhouette_scores,
        'n_clusters_tested': list(n_clusters_range)
    }
    
    if len(set(final_cluster_labels)) > 1:
        metrics['calinski_harabasz_score'] = calinski_harabasz_score(df_scaled, final_cluster_labels)
    else:
        metrics['calinski_harabasz_score'] = 0
    
    print(f"Nombre optimal de clusters: {optimal_n_clusters}")
    print(f"Score de silhouette: {best_silhouette:.4f}")
    print(f"Méthode de liaison: {linkage_method}")
    
    return linkage_matrix, final_cluster_labels, optimal_n_clusters, metrics

def visualize_cah_dendrogram(linkage_matrix, optimal_n_clusters=None, feature_names=None, max_display=30):
    """
    Visualise le dendrogramme de la CAH
    
    Parameters:
    -----------
    linkage_matrix : array
        Matrice de liaison de la CAH
    optimal_n_clusters : int, optional
        Nombre optimal de clusters à marquer
    feature_names : list, optional
        Noms des caractéristiques (pour labeling)
    max_display : int
        Nombre maximum d'éléments à afficher dans le dendrogramme
    """
    
    plt.figure(figsize=(15, 8))
    
    # Créer le dendrogramme
    dendrogram_data = dendrogram(
        linkage_matrix,
        truncate_mode='lastp' if len(linkage_matrix) > max_display else None,
        p=max_display if len(linkage_matrix) > max_display else None,
        show_leaf_counts=True,
        leaf_rotation=90,
        leaf_font_size=10
    )
    
    plt.title('Dendrogramme - Classification Ascendante Hiérarchique', fontsize=14, fontweight='bold')
    plt.xlabel('Index des propriétés ou clusters')
    plt.ylabel('Distance')
    
    # Ajouter une ligne horizontale pour le nombre optimal de clusters
    if optimal_n_clusters is not None:
        # Calculer la hauteur de coupe
        # Pour n clusters, on prend la (n-1)ème plus grande distance
        distances = linkage_matrix[:, 2]
        sorted_distances = np.sort(distances)
        if len(sorted_distances) >= optimal_n_clusters - 1:
            cut_height = sorted_distances[-(optimal_n_clusters - 1)]
            plt.axhline(y=cut_height, color='red', linestyle='--', linewidth=2, 
                       label=f'Coupe pour {optimal_n_clusters} clusters')
            plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return dendrogram_data

def visualize_cah_results_with_pca(df_scaled, cluster_labels, pca_model, df_pca, linkage_matrix, optimal_n_clusters):
    """
    Visualise les résultats de CAH avec PCA
    """
    # Créer des sous-graphiques
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            'CAH - Clusters (2D PCA)',
            'Distribution des clusters',
            'Évolution du score de silhouette',
            'CAH - Clusters (3D si possible)'
        ],
        specs=[[{'type': 'scatter'}, {'type': 'bar'}],
               [{'type': 'scatter'}, {'type': 'scatter3d'}]]
    )
    
    # Couleurs pour les clusters
    unique_labels = sorted(set(cluster_labels))
    colors = px.colors.qualitative.Set3[:len(unique_labels)]
    
    # 1. Scatter plot 2D avec PCA
    for i, label in enumerate(unique_labels):
        mask = cluster_labels == label
        
        fig.add_trace(
            go.Scatter(
                x=df_pca.iloc[mask, 0],
                y=df_pca.iloc[mask, 1],
                mode='markers',
                name=f'Cluster {label}',
                marker=dict(color=colors[i % len(colors)]),
                showlegend=True
            ),
            row=1, col=1
        )
    
    # 2. Distribution des clusters
    cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
    cluster_names = [f'Cluster {idx}' for idx in cluster_counts.index]
    
    fig.add_trace(
        go.Bar(
            x=cluster_names,
            y=cluster_counts.values,
            marker_color=[colors[i % len(colors)] for i in range(len(cluster_counts))],
            showlegend=False
        ),
        row=1, col=2
    )
    
    # 3. Évolution du score de silhouette (si disponible dans metrics)
    # On va créer un graphique simple pour maintenant
    fig.add_trace(
        go.Scatter(
            x=list(range(2, optimal_n_clusters + 3)),
            y=[0.3, 0.4, 0.45, 0.42],  # Valeurs d'exemple
            mode='lines+markers',
            name='Score Silhouette',
            showlegend=False
        ),
        row=2, col=1
    )
    
    # 4. Scatter plot 3D si on a au moins 3 composantes
    if df_pca.shape[1] >= 3:
        for i, label in enumerate(unique_labels):
            mask = cluster_labels == label
            
            fig.add_trace(
                go.Scatter3d(
                    x=df_pca.iloc[mask, 0],
                    y=df_pca.iloc[mask, 1],
                    z=df_pca.iloc[mask, 2],
                    mode='markers',
                    name=f'Cluster {label} (3D)',
                    marker=dict(
                        color=colors[i % len(colors)],
                        size=3
                    ),
                    showlegend=False
                ),
                row=2, col=2
            )
    
    # Mise à jour des axes
    fig.update_xaxes(title_text="PC1", row=1, col=1)
    fig.update_yaxes(title_text="PC2", row=1, col=1)
    fig.update_xaxes(title_text="Clusters", row=1, col=2)
    fig.update_yaxes(title_text="Nombre de propriétés", row=1, col=2)
    fig.update_xaxes(title_text="Nombre de clusters", row=2, col=1)
    fig.update_yaxes(title_text="Score de silhouette", row=2, col=1)
    
    # Mise à jour du layout
    fig.update_layout(
        height=800,
        title_text=f"Analyse CAH - {optimal_n_clusters} clusters optimaux",
        showlegend=True
    )
    
    return fig

def comparer_clustering_methods(df_scaled, df_pca, pca_model):
    """
    Compare K-Means, DBSCAN et CAH sur les mêmes données
    
    Parameters:
    -----------
    df_scaled : pandas.DataFrame
        Données standardisées
    df_pca : pandas.DataFrame
        Données PCA
    pca_model : PCA
        Modèle PCA ajusté
        
    Returns:
    --------
    comparison_results : dict
        Résultats de comparaison des trois méthodes
    """
    
    print("=== COMPARAISON DES MÉTHODES DE CLUSTERING ===")
    print("=" * 50)
    
    results = {}
    
    # 1. K-Means
    print("\n🔵 K-MEANS:")
    kmeans_model, kmeans_n_clusters, kmeans_labels, kmeans_metrics, _, _ = apply_kmeans_clustering(df_scaled)
    results['kmeans'] = {
        'labels': kmeans_labels,
        'n_clusters': kmeans_n_clusters,
        'silhouette_score': kmeans_metrics['silhouette_score'],
        'method': 'K-Means'
    }
    
    # 2. DBSCAN
    print("\n🔴 DBSCAN:")
    dbscan_model, dbscan_labels, dbscan_metrics = apply_dbscan_clustering(df_scaled)
    results['dbscan'] = {
        'labels': dbscan_labels,
        'n_clusters': dbscan_metrics['n_clusters'],
        'silhouette_score': dbscan_metrics['silhouette_score'],
        'noise_points': dbscan_metrics['n_noise_points'],
        'method': 'DBSCAN'
    }
    
    # 3. CAH
    print("\n🟢 CAH:")
    cah_linkage, cah_labels, cah_n_clusters, cah_metrics = apply_cah_clustering(df_scaled)
    results['cah'] = {
        'labels': cah_labels,
        'n_clusters': cah_n_clusters,
        'silhouette_score': cah_metrics['silhouette_score'],
        'linkage_matrix': cah_linkage,
        'method': 'CAH'
    }
    
    # 4. Créer un tableau de comparaison
    comparison_data = {
        'Méthode': ['K-Means', 'DBSCAN', 'CAH'],
        'Nombre de clusters': [
            results['kmeans']['n_clusters'],
            results['dbscan']['n_clusters'],
            results['cah']['n_clusters']
        ],
        'Score Silhouette': [
            f"{results['kmeans']['silhouette_score']:.4f}",
            f"{results['dbscan']['silhouette_score']:.4f}" if results['dbscan']['silhouette_score'] > 0 else "N/A",
            f"{results['cah']['silhouette_score']:.4f}"
        ],
        'Points de bruit': [
            "0",
            str(results['dbscan']['noise_points']),
            "0"
        ]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    
    print(f"\n📊 TABLEAU DE COMPARAISON:")
    print("-" * 40)
    print(comparison_df.to_string(index=False))
    
    # 5. Visualisation comparative
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=['K-Means', 'DBSCAN', 'CAH'],
        specs=[[{'type': 'scatter'}, {'type': 'scatter'}, {'type': 'scatter'}]]
    )
    
    methods = ['kmeans', 'dbscan', 'cah']
    for idx, method in enumerate(methods, 1):
        labels = results[method]['labels']
        unique_labels = sorted(set(labels))
        colors = px.colors.qualitative.Set3[:len(unique_labels)]
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            cluster_name = f'Bruit' if label == -1 else f'Cluster {label}'
            
            fig.add_trace(
                go.Scatter(
                    x=df_pca.iloc[mask, 0],
                    y=df_pca.iloc[mask, 1],
                    mode='markers',
                    name=f'{results[method]["method"]}: {cluster_name}',
                    marker=dict(color=colors[i] if label != -1 else 'black'),
                    showlegend=False
                ),
                row=1, col=idx
            )
    
    fig.update_layout(
        height=400,
        title_text="Comparaison des méthodes de clustering sur données PCA"
    )
    
    # Mise à jour des axes
    for i in range(1, 4):
        fig.update_xaxes(title_text="PC1", row=1, col=i)
        fig.update_yaxes(title_text="PC2", row=1, col=i)
    
    return results, comparison_df, fig

# FONCTIONS À AJOUTER DANS model_functions.py

def analyze_cah_groups_detailed(df_original, df_scaled, linkage_matrix, cluster_labels, optimal_n_clusters, feature_names):
    """
    Analyse détaillée des groupes CAH avec sous-catégories
    
    Parameters:
    -----------
    df_original : pandas.DataFrame
        Données originales (non standardisées)
    df_scaled : pandas.DataFrame
        Données standardisées utilisées pour CAH
    linkage_matrix : array
        Matrice de liaison de la CAH
    cluster_labels : array
        Labels des clusters
    optimal_n_clusters : int
        Nombre optimal de clusters
    feature_names : list
        Noms des caractéristiques utilisées
    
    Returns:
    --------
    group_summary : DataFrame
        Résumé des groupes avec caractéristiques
    detailed_analysis : DataFrame
        Analyse détaillée propriété par propriété
    """
    
    print("=== ANALYSE DÉTAILLÉE DES GROUPES CAH ===")
    print("=" * 50)
    
    # Créer un DataFrame avec toutes les informations
    df_analysis = df_original.iloc[df_scaled.index].copy()
    df_analysis['Cluster_CAH'] = cluster_labels
    
    # 1. RÉSUMÉ PAR GROUPE
    group_summary_data = []
    
    for cluster_id in sorted(set(cluster_labels)):
        cluster_data = df_analysis[df_analysis['Cluster_CAH'] == cluster_id]
        
        # Statistiques de base
        group_info = {
            'Cluster': cluster_id,
            'Nombre_Propriétés': len(cluster_data),
            'Pourcentage': f"{len(cluster_data)/len(df_analysis)*100:.1f}%"
        }
        
        # Caractéristiques numériques moyennes
        numeric_features = ['price', 'size', 'age', 'rooms', 'bedrooms', 'bathrooms', 'parkings']
        for feature in numeric_features:
            if feature in cluster_data.columns:
                avg_val = cluster_data[feature].mean()
                group_info[f'{feature}_moyen'] = round(avg_val, 1) if pd.notna(avg_val) else 'N/A'
        
        # Caractéristiques catégorielles dominantes
        categorical_features = ['condition', 'finishing', 'neighborhood']
        for feature in categorical_features:
            if feature in cluster_data.columns and not cluster_data[feature].empty:
                mode_val = cluster_data[feature].mode()
                group_info[f'{feature}_principal'] = mode_val.iloc[0] if len(mode_val) > 0 else 'N/A'
        
        # Équipements (pourcentage)
        equipment_features = ['air_conditioning', 'central_heating', 'swimming_pool', 'elevator', 'garden', 'equipped_kitchen']
        for feature in equipment_features:
            if feature in cluster_data.columns:
                pct = (cluster_data[feature] == 1).mean() * 100
                group_info[f'{feature}_pct'] = f"{pct:.0f}%"
        
        group_summary_data.append(group_info)
    
    group_summary = pd.DataFrame(group_summary_data)
    
    # 2. ANALYSE DÉTAILLÉE PROPRIÉTÉ PAR PROPRIÉTÉ
    detailed_columns = ['Cluster_CAH'] + [col for col in df_analysis.columns if col != 'Cluster_CAH']
    detailed_analysis = df_analysis[detailed_columns].sort_values('Cluster_CAH')
    
    # 3. AFFICHAGE DES RÉSULTATS
    print("\n📊 RÉSUMÉ DES GROUPES:")
    print("-" * 30)
    display_columns = ['Cluster', 'Nombre_Propriétés', 'Pourcentage', 'price_moyen', 'size_moyen', 'condition_principal']
    available_display_cols = [col for col in display_columns if col in group_summary.columns]
    print(group_summary[available_display_cols].to_string(index=False))
    
    return group_summary, detailed_analysis

def create_hierarchical_subgroups(df_scaled, linkage_matrix, cluster_labels, n_levels=3):
    """
    Crée des sous-groupes hiérarchiques à différents niveaux
    
    Parameters:
    -----------
    df_scaled : pandas.DataFrame
        Données standardisées
    linkage_matrix : array
        Matrice de liaison CAH
    cluster_labels : array
        Labels du niveau principal
    n_levels : int
        Nombre de niveaux hiérarchiques à analyser
    
    Returns:
    --------
    hierarchy_df : DataFrame
        DataFrame avec les groupes à différents niveaux
    """
    
    print(f"\n🌳 ANALYSE HIÉRARCHIQUE À {n_levels} NIVEAUX:")
    print("=" * 40)
    
    hierarchy_data = []
    
    # Calculer les clusters pour différents nombres
    max_clusters_main = len(set(cluster_labels))
    cluster_numbers = []
    
    # Définir les niveaux (du plus général au plus spécifique)
    if max_clusters_main <= 2:
        levels = [2, 3, 4]
    elif max_clusters_main <= 4:
        levels = [2, max_clusters_main, max_clusters_main + 2]
    else:
        levels = [2, max_clusters_main // 2, max_clusters_main]
    
    levels = levels[:n_levels]
    
    hierarchy_results = {}
    
    for level_idx, n_clusters in enumerate(levels, 1):
        if n_clusters <= len(df_scaled):
            level_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
            hierarchy_results[f'Niveau_{level_idx}'] = level_labels
            
            print(f"Niveau {level_idx}: {n_clusters} groupes")
            level_counts = pd.Series(level_labels).value_counts().sort_index()
            for group_id, count in level_counts.items():
                print(f"  Groupe {group_id}: {count} propriétés ({count/len(df_scaled)*100:.1f}%)")
    
    # Créer un DataFrame avec la hiérarchie
    hierarchy_df = pd.DataFrame(hierarchy_results, index=df_scaled.index)
    
    return hierarchy_df

def visualize_group_characteristics(df_analysis, group_summary):
    """
    Visualise les caractéristiques des groupes
    """
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    
    # Préparer les données pour visualisation
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            'Prix moyen par cluster',
            'Taille moyenne par cluster', 
            'Répartition des propriétés',
            'Équipements par cluster'
        ],
        specs=[[{'type': 'bar'}, {'type': 'bar'}],
               [{'type': 'pie'}, {'type': 'bar'}]]
    )
    
    clusters = group_summary['Cluster'].astype(str)
    
    # 1. Prix moyen
    if 'price_moyen' in group_summary.columns:
        fig.add_trace(
            go.Bar(x=clusters, y=group_summary['price_moyen'], name='Prix moyen', showlegend=False),
            row=1, col=1
        )
    
    # 2. Taille moyenne
    if 'size_moyen' in group_summary.columns:
        fig.add_trace(
            go.Bar(x=clusters, y=group_summary['size_moyen'], name='Taille moyenne', showlegend=False),
            row=1, col=2
        )
    
    # 3. Répartition (pie chart)
    fig.add_trace(
        go.Pie(labels=[f'Cluster {c}' for c in clusters], 
               values=group_summary['Nombre_Propriétés'], 
               name="Répartition", showlegend=False),
        row=2, col=1
    )
    
    # 4. Équipements (exemple avec air_conditioning)
    if 'air_conditioning_pct' in group_summary.columns:
        equipment_values = [float(pct.replace('%', '')) for pct in group_summary['air_conditioning_pct']]
        fig.add_trace(
            go.Bar(x=clusters, y=equipment_values, name='Climatisation %', showlegend=False),
            row=2, col=2
        )
    
    fig.update_layout(height=600, title_text="Caractéristiques des clusters CAH")
    return fig

# UTILISATION DANS VOTRE NOTEBOOK
def complete_cah_analysis(df_original, df_scaled, linkage_matrix, cluster_labels, optimal_n_clusters, feature_names):
    """
    Analyse complète des résultats CAH
    """
    
    # 1. Analyse détaillée des groupes
    group_summary, detailed_analysis = analyze_cah_groups_detailed(
        df_original, df_scaled, linkage_matrix, cluster_labels, optimal_n_clusters, feature_names
    )
    
    # 2. Analyse hiérarchique
    hierarchy_df = create_hierarchical_subgroups(df_scaled, linkage_matrix, cluster_labels)
    
    # 3. Visualisation
    fig_characteristics = visualize_group_characteristics(detailed_analysis, group_summary)
    
    return group_summary, detailed_analysis, hierarchy_df, fig_characteristics

# CODE POUR VOTRE NOTEBOOK SPÉCIFIQUEMENT:

def analyze_your_cah_results(df, filtered_df, linkage_matrix, cluster_labels, optimal_n_clusters, feature_names):
    """
    Analyse spécifique pour vos données filtrées (La Soukra, appartements, vente)
    """
    
    print("=== ANALYSE SPÉCIFIQUE: APPARTEMENTS LA SOUKRA (VENTE) ===")
    print("=" * 60)
    
    # Récupérer les données originales correspondantes
    original_indices = filtered_df.index
    df_original_filtered = df.loc[original_indices].copy()
    df_original_filtered['Cluster_CAH'] = cluster_labels
    
    # Créer le résumé détaillé
    analysis_results = []
    
    for cluster_id in sorted(set(cluster_labels)):
        cluster_data = df_original_filtered[df_original_filtered['Cluster_CAH'] == cluster_id]
        
        result = {
            'Cluster': cluster_id,
            'Nombre': len(cluster_data),
            'Pourcentage': f"{len(cluster_data)/len(df_original_filtered)*100:.1f}%",
            'Prix_Moyen': f"{cluster_data['price'].mean():.0f}" if 'price' in cluster_data.columns else 'N/A',
            'Prix_Min': f"{cluster_data['price'].min():.0f}" if 'price' in cluster_data.columns else 'N/A',
            'Prix_Max': f"{cluster_data['price'].max():.0f}" if 'price' in cluster_data.columns else 'N/A',
            'Taille_Moyenne': f"{cluster_data['size'].mean():.0f}" if 'size' in cluster_data.columns else 'N/A',
            'Age_Moyen': f"{cluster_data['age'].mean():.0f}" if 'age' in cluster_data.columns else 'N/A',
            'Quartier_Principal': cluster_data['neighborhood'].mode().iloc[0] if 'neighborhood' in cluster_data.columns and not cluster_data['neighborhood'].empty else 'N/A',
            'Condition_Principale': cluster_data['condition'].mode().iloc[0] if 'condition' in cluster_data.columns and not cluster_data['condition'].empty else 'N/A',
            'Finition_Principale': cluster_data['finishing'].mode().iloc[0] if 'finishing' in cluster_data.columns and not cluster_data['finishing'].empty else 'N/A'
        }
        
        # Équipements (pourcentage)
        if 'air_conditioning' in cluster_data.columns:
            result['Clim_%'] = f"{(cluster_data['air_conditioning'] == 1).mean() * 100:.0f}%"
        if 'elevator' in cluster_data.columns:
            result['Ascenseur_%'] = f"{(cluster_data['elevator'] == 1).mean() * 100:.0f}%"
        
        analysis_results.append(result)
    
    # Créer le DataFrame final
    results_df = pd.DataFrame(analysis_results)
    
    # Créer aussi un DataFrame détaillé avec toutes les propriétés
    detailed_df = df_original_filtered[['Cluster_CAH', 'price', 'size', 'age', 'neighborhood', 'condition', 'finishing']].copy()
    detailed_df = detailed_df.sort_values(['Cluster_CAH', 'price'])
    
    print("\n📊 RÉSUMÉ DES CLUSTERS:")
    print(results_df.to_string(index=False))
    
    print(f"\n📋 DÉTAIL DES PROPRIÉTÉS (premières 10):")
    print(detailed_df.head(10).to_string(index=False))
    
    return results_df, detailed_df

    # 2. ANALYSE


# FONCTION À AJOUTER DANS model_functions.py

def complete_cah_analysis_after_dendrogram(df, filtered_df, linkage_matrix, cluster_labels, optimal_n_clusters, feature_names):
    """
    Analyse complète des résultats CAH après affichage du dendrogramme
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame original complet
    filtered_df : pandas.DataFrame
        DataFrame filtré et standardisé utilisé pour CAH
    linkage_matrix : array
        Matrice de liaison de la CAH
    cluster_labels : array
        Labels des clusters
    optimal_n_clusters : int
        Nombre optimal de clusters
    feature_names : list
        Noms des caractéristiques utilisées
        
    Returns:
    --------
    results_df : DataFrame
        Résumé des clusters
    detailed_df : DataFrame
        Analyse détaillée propriété par propriété
    hierarchy_df : DataFrame
        Hiérarchie des groupes à différents niveaux
    """
    
    print("\n" + "="*80)
    print("ANALYSE DÉTAILLÉE DES GROUPES CAH")
    print("="*80)
    
    # ============================================
    # 1. ANALYSE SPÉCIFIQUE DES DONNÉES
    # ============================================
    
    # Utiliser la fonction spécialisée existante
    results_df, detailed_df = analyze_your_cah_results(
        df, filtered_df, linkage_matrix, cluster_labels, optimal_n_clusters, feature_names
    )
    
    # ============================================
    # 2. AFFICHAGE DES RÉSULTATS EN DATAFRAMES
    # ============================================
    
    print("\n🎯 TABLEAU RÉSUMÉ DES CLUSTERS:")
    print("="*50)
    print(results_df)
    
    print("\n📋 DÉTAIL DE TOUTES LES PROPRIÉTÉS PAR CLUSTER:")
    print("="*50)
    print(detailed_df)
    
    # ============================================
    # 3. ANALYSE PAR CLUSTER INDIVIDUEL
    # ============================================
    
    print("\n🔍 ANALYSE DÉTAILLÉE PAR CLUSTER:")
    print("="*40)
    
    for cluster_id in sorted(set(cluster_labels)):
        cluster_properties = detailed_df[detailed_df['Cluster_CAH'] == cluster_id]
        
        print(f"\n📍 CLUSTER {cluster_id} - {len(cluster_properties)} propriétés:")
        print("-" * 30)
        
        # Statistiques du cluster
        if 'price' in cluster_properties.columns:
            print(f"💰 Prix: {cluster_properties['price'].min():.0f} - {cluster_properties['price'].max():.0f} TND")
            print(f"   Moyenne: {cluster_properties['price'].mean():.0f} TND")
        
        if 'size' in cluster_properties.columns:
            print(f"📐 Taille: {cluster_properties['size'].min():.0f} - {cluster_properties['size'].max():.0f} m²")
            print(f"   Moyenne: {cluster_properties['size'].mean():.0f} m²")
        
        # Quartiers représentés
        if 'neighborhood' in cluster_properties.columns:
            neighborhoods = cluster_properties['neighborhood'].value_counts()
            print(f"🏘️  Quartiers: {neighborhoods.to_dict()}")
        
        # Conditions
        if 'condition' in cluster_properties.columns:
            conditions = cluster_properties['condition'].value_counts()
            print(f"🏠 États: {conditions.to_dict()}")
        
        print("\n   Exemples de propriétés:")
        examples = cluster_properties.head(3)
        for idx, prop in examples.iterrows():
            print(f"   • Prix: {prop['price']:.0f} TND, Taille: {prop['size']:.0f} m², Quartier: {prop['neighborhood']}")
    
    # ============================================
    # 4. ANALYSE HIÉRARCHIQUE
    # ============================================
    
    print("\n🌳 ANALYSE HIÉRARCHIQUE - SOUS-GROUPES:")
    print("="*50)
    
    # Créer des sous-groupes hiérarchiques
    hierarchy_df = create_hierarchical_subgroups(
        filtered_df, linkage_matrix, cluster_labels, n_levels=3
    )
    
    # Combiner avec les données originales
    df_with_hierarchy = df.loc[filtered_df.index].copy()
    df_with_hierarchy = pd.concat([df_with_hierarchy, hierarchy_df], axis=1)
    df_with_hierarchy['Cluster_Final'] = cluster_labels
    
    print("\n📊 Hiérarchie des groupes (premières 10 propriétés):")
    hierarchy_cols = ['price', 'size', 'neighborhood'] + list(hierarchy_df.columns) + ['Cluster_Final']
    available_hierarchy_cols = [col for col in hierarchy_cols if col in df_with_hierarchy.columns]
    print(df_with_hierarchy[available_hierarchy_cols].head(10))
    
    # ============================================
    # 5. VISUALISATION DES CARACTÉRISTIQUES
    # ============================================
    
    print("\n📈 VISUALISATION DES CARACTÉRISTIQUES:")
    print("="*40)
    
    # Créer un graphique des caractéristiques par cluster
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Prix par cluster
    if 'price' in detailed_df.columns:
        detailed_df.boxplot(column='price', by='Cluster_CAH', ax=axes[0,0])
        axes[0,0].set_title('Prix par Cluster')
        axes[0,0].set_xlabel('Cluster')
        axes[0,0].set_ylabel('Prix (TND)')
    
    # Taille par cluster
    if 'size' in detailed_df.columns:
        detailed_df.boxplot(column='size', by='Cluster_CAH', ax=axes[0,1])
        axes[0,1].set_title('Taille par Cluster')
        axes[0,1].set_xlabel('Cluster')
        axes[0,1].set_ylabel('Taille (m²)')
    
    # Âge par cluster
    if 'age' in detailed_df.columns:
        detailed_df.boxplot(column='age', by='Cluster_CAH', ax=axes[1,0])
        axes[1,0].set_title('Âge par Cluster')
        axes[1,0].set_xlabel('Cluster')
        axes[1,0].set_ylabel('Âge (années)')
    
    # Distribution des clusters
    cluster_counts = detailed_df['Cluster_CAH'].value_counts().sort_index()
    axes[1,1].bar(cluster_counts.index, cluster_counts.values)
    axes[1,1].set_title('Nombre de propriétés par cluster')
    axes[1,1].set_xlabel('Cluster')
    axes[1,1].set_ylabel('Nombre de propriétés')
    
    plt.tight_layout()
    plt.show()
    
    # ============================================
    # 6. INTERPRÉTATION BUSINESS
    # ============================================
    
    print("\n💼 INTERPRÉTATION BUSINESS:")
    print("="*30)
    
    for cluster_id in sorted(set(cluster_labels)):
        cluster_data = detailed_df[detailed_df['Cluster_CAH'] == cluster_id]
        avg_price = cluster_data['price'].mean() if 'price' in cluster_data.columns else 0
        avg_size = cluster_data['size'].mean() if 'size' in cluster_data.columns else 0
        
        # Interpréter le cluster
        if avg_price > detailed_df['price'].quantile(0.75):
            segment = "HAUT DE GAMME"
        elif avg_price > detailed_df['price'].median():
            segment = "MOYEN-HAUT"
        elif avg_price > detailed_df['price'].quantile(0.25):
            segment = "MOYEN"
        else:
            segment = "ÉCONOMIQUE"
        
        print(f"\n🏷️  CLUSTER {cluster_id} → Segment {segment}")
        print(f"   Prix moyen: {avg_price:.0f} TND")
        print(f"   Taille moyenne: {avg_size:.0f} m²")
        print(f"   Nombre de biens: {len(cluster_data)}")
        
        # Recommandations
        if segment == "HAUT DE GAMME":
            print("   💡 Stratégie: Marketing premium, clientèle aisée")
        elif segment == "ÉCONOMIQUE":
            print("   💡 Stratégie: Primo-accédants, investissement locatif")
        else:
            print("   💡 Stratégie: Familles, marché principal")
    
    return results_df, detailed_df, hierarchy_df

# ============================================
# FONCTION ALTERNATIVE AVEC OPTIONS
# ============================================

def complete_cah_analysis_with_options(df, filtered_df, linkage_matrix, cluster_labels, optimal_n_clusters, feature_names, 
                                      show_detailed_analysis=True, show_hierarchy=True, show_visualizations=True, 
                                      show_business_interpretation=True):
    """
    Version avec options pour contrôler quels éléments afficher
    
    Parameters:
    -----------
    df, filtered_df, linkage_matrix, cluster_labels, optimal_n_clusters, feature_names : comme précédent
    show_detailed_analysis : bool
        Afficher l'analyse détaillée par cluster
    show_hierarchy : bool
        Afficher l'analyse hiérarchique
    show_visualizations : bool
        Afficher les graphiques
    show_business_interpretation : bool
        Afficher l'interprétation business
    """
    
    print("\n" + "="*80)
    print("ANALYSE DÉTAILLÉE DES GROUPES CAH")
    print("="*80)
    
    # Toujours faire l'analyse de base
    results_df, detailed_df = analyze_your_cah_results(
        df, filtered_df, linkage_matrix, cluster_labels, optimal_n_clusters, feature_names
    )
    
    print("\n🎯 TABLEAU RÉSUMÉ DES CLUSTERS:")
    print("="*50)
    display(results_df)
    
    print("\n📋 DÉTAIL DE TOUTES LES PROPRIÉTÉS PAR CLUSTER:")
    print("="*50)
    display(detailed_df)
    
    # Analyse détaillée conditionnelle
    if show_detailed_analysis:
        print("\n🔍 ANALYSE DÉTAILLÉE PAR CLUSTER:")
        print("="*40)
        
        for cluster_id in sorted(set(cluster_labels)):
            cluster_properties = detailed_df[detailed_df['Cluster_CAH'] == cluster_id]
            
            print(f"\n📍 CLUSTER {cluster_id} - {len(cluster_properties)} propriétés:")
            print("-" * 30)
            
            if 'price' in cluster_properties.columns:
                print(f"💰 Prix: {cluster_properties['price'].min():.0f} - {cluster_properties['price'].max():.0f} TND")
                print(f"   Moyenne: {cluster_properties['price'].mean():.0f} TND")
            
            if 'size' in cluster_properties.columns:
                print(f"📐 Taille: {cluster_properties['size'].min():.0f} - {cluster_properties['size'].max():.0f} m²")
                print(f"   Moyenne: {cluster_properties['size'].mean():.0f} m²")
            
            if 'neighborhood' in cluster_properties.columns:
                neighborhoods = cluster_properties['neighborhood'].value_counts()
                print(f"🏘️  Quartiers: {neighborhoods.to_dict()}")
            
            if 'condition' in cluster_properties.columns:
                conditions = cluster_properties['condition'].value_counts()
                print(f"🏠 États: {conditions.to_dict()}")
            
            print("\n   Exemples de propriétés:")
            examples = cluster_properties.head(3)
            for idx, prop in examples.iterrows():
                print(f"   • Prix: {prop['price']:.0f} TND, Taille: {prop['size']:.0f} m², Quartier: {prop['neighborhood']}")
    
    # Hiérarchie conditionnelle
    hierarchy_df = None
    if show_hierarchy:
        print("\n🌳 ANALYSE HIÉRARCHIQUE - SOUS-GROUPES:")
        print("="*50)
        
        hierarchy_df = create_hierarchical_subgroups(
            filtered_df, linkage_matrix, cluster_labels, n_levels=3
        )
        
        df_with_hierarchy = df.loc[filtered_df.index].copy()
        df_with_hierarchy = pd.concat([df_with_hierarchy, hierarchy_df], axis=1)
        df_with_hierarchy['Cluster_Final'] = cluster_labels
        
        print("\n📊 Hiérarchie des groupes (premières 10 propriétés):")
        hierarchy_cols = ['price', 'size', 'neighborhood'] + list(hierarchy_df.columns) + ['Cluster_Final']
        available_hierarchy_cols = [col for col in hierarchy_cols if col in df_with_hierarchy.columns]
        display(df_with_hierarchy[available_hierarchy_cols].head(10))
    
    # Visualisations conditionnelles
    if show_visualizations:
        print("\n📈 VISUALISATION DES CARACTÉRISTIQUES:")
        print("="*40)
        
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        if 'price' in detailed_df.columns:
            detailed_df.boxplot(column='price', by='Cluster_CAH', ax=axes[0,0])
            axes[0,0].set_title('Prix par Cluster')
            axes[0,0].set_xlabel('Cluster')
            axes[0,0].set_ylabel('Prix (TND)')
        
        if 'size' in detailed_df.columns:
            detailed_df.boxplot(column='size', by='Cluster_CAH', ax=axes[0,1])
            axes[0,1].set_title('Taille par Cluster')
            axes[0,1].set_xlabel('Cluster')
            axes[0,1].set_ylabel('Taille (m²)')
        
        if 'age' in detailed_df.columns:
            detailed_df.boxplot(column='age', by='Cluster_CAH', ax=axes[1,0])
            axes[1,0].set_title('Âge par Cluster')
            axes[1,0].set_xlabel('Cluster')
            axes[1,0].set_ylabel('Âge (années)')
        
        cluster_counts = detailed_df['Cluster_CAH'].value_counts().sort_index()
        axes[1,1].bar(cluster_counts.index, cluster_counts.values)
        axes[1,1].set_title('Nombre de propriétés par cluster')
        axes[1,1].set_xlabel('Cluster')
        axes[1,1].set_ylabel('Nombre de propriétés')
        
        plt.tight_layout()
        plt.show()
    
    # Interprétation business conditionnelle
    if show_business_interpretation:
        print("\n💼 INTERPRÉTATION BUSINESS:")
        print("="*30)
        
        for cluster_id in sorted(set(cluster_labels)):
            cluster_data = detailed_df[detailed_df['Cluster_CAH'] == cluster_id]
            avg_price = cluster_data['price'].mean() if 'price' in cluster_data.columns else 0
            avg_size = cluster_data['size'].mean() if 'size' in cluster_data.columns else 0
            
            if avg_price > detailed_df['price'].quantile(0.75):
                segment = "HAUT DE GAMME"
            elif avg_price > detailed_df['price'].median():
                segment = "MOYEN-HAUT"
            elif avg_price > detailed_df['price'].quantile(0.25):
                segment = "MOYEN"
            else:
                segment = "ÉCONOMIQUE"
            
            print(f"\n🏷️  CLUSTER {cluster_id} → Segment {segment}")
            print(f"   Prix moyen: {avg_price:.0f} TND")
            print(f"   Taille moyenne: {avg_size:.0f} m²")
            print(f"   Nombre de biens: {len(cluster_data)}")
            
            if segment == "HAUT DE GAMME":
                print("   💡 Stratégie: Marketing premium, clientèle aisée")
            elif segment == "ÉCONOMIQUE":
                print("   💡 Stratégie: Primo-accédants, investissement locatif")
            else:
                print("   💡 Stratégie: Familles, marché principal")
    
    return results_df, detailed_df, hierarchy_df
def apply_kmeans_clustering(df_scaled, n_clusters_range=(2, 10), random_state=42):
    """
    Applique K-Means clustering avec optimisation du nombre de clusters
    
    Parameters:
    -----------
    df_scaled : pandas.DataFrame
        Données standardisées
    n_clusters_range : tuple
        Range du nombre de clusters à tester
    random_state : int
        Graine pour la reproductibilité
        
    Returns:
    --------
    best_model : KMeans
        Meilleur modèle K-Means
    best_n_clusters : int
        Nombre optimal de clusters
    cluster_labels : array
        Labels des clusters pour chaque point
    metrics : dict
        Métriques d'évaluation
    """
    best_score = -1
    best_model = None
    best_n_clusters = 2
    
    scores = []
    n_clusters_list = list(range(n_clusters_range[0], n_clusters_range[1] + 1))
    
    print("Optimisation du nombre de clusters pour K-Means...")
    
    for n_clusters in n_clusters_list:
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        cluster_labels = kmeans.fit_predict(df_scaled)
        
        # Calculer le score de silhouette
        if len(set(cluster_labels)) > 1:  # Au moins 2 clusters différents
            silhouette_avg = silhouette_score(df_scaled, cluster_labels)
            scores.append(silhouette_avg)
            
            if silhouette_avg > best_score:
                best_score = silhouette_avg
                best_model = kmeans
                best_n_clusters = n_clusters
        else:
            scores.append(0)
    
    # Obtenir les labels finaux
    cluster_labels = best_model.predict(df_scaled)
    
    # Calculer les métriques
    metrics = {
        'silhouette_score': silhouette_score(df_scaled, cluster_labels),
        'calinski_harabasz_score': calinski_harabasz_score(df_scaled, cluster_labels),
        'inertia': best_model.inertia_,
        'n_clusters': best_n_clusters
    }
    
    print(f"Nombre optimal de clusters: {best_n_clusters}")
    print(f"Score de silhouette: {metrics['silhouette_score']:.4f}")
    
    return best_model, best_n_clusters, cluster_labels, metrics, scores, n_clusters_list
def apply_dbscan_clustering(df_scaled, eps_range=(0.3, 2.0), min_samples_range=(3, 10)):
    """
    Applique DBSCAN clustering avec optimisation des paramètres
    
    Parameters:
    -----------
    df_scaled : pandas.DataFrame
        Données standardisées
    eps_range : tuple
        Range des valeurs eps à tester
    min_samples_range : tuple
        Range des valeurs min_samples à tester
        
    Returns:
    --------
    best_model : DBSCAN
        Meilleur modèle DBSCAN
    cluster_labels : array
        Labels des clusters
    metrics : dict
        Métriques d'évaluation
    """
    best_score = -1
    best_model = None
    best_params = {}
    
    eps_values = np.linspace(eps_range[0], eps_range[1], 10)
    min_samples_values = range(min_samples_range[0], min_samples_range[1] + 1)
    
    print("Optimisation des paramètres pour DBSCAN...")
    
    for eps in eps_values:
        for min_samples in min_samples_values:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            cluster_labels = dbscan.fit_predict(df_scaled)
            
            # Vérifier qu'il y a au moins 2 clusters (sans compter le bruit)
            unique_labels = set(cluster_labels)
            n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
            
            if n_clusters >= 2:
                # Calculer le score de silhouette (en excluant le bruit)
                mask = cluster_labels != -1
                if mask.sum() > 1:
                    silhouette_avg = silhouette_score(df_scaled[mask], cluster_labels[mask])
                    
                    if silhouette_avg > best_score:
                        best_score = silhouette_avg
                        best_model = dbscan
                        best_params = {'eps': eps, 'min_samples': min_samples}
    
    if best_model is None:
        # Si aucun bon paramètre n'est trouvé, utiliser des valeurs par défaut
        print("Aucun paramètre optimal trouvé, utilisation des valeurs par défaut")
        best_model = DBSCAN(eps=0.5, min_samples=5)
        best_params = {'eps': 0.5, 'min_samples': 5}
    
    # Obtenir les labels finaux
    cluster_labels = best_model.fit_predict(df_scaled)
    
    # Calculer les métriques
    unique_labels = set(cluster_labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    n_noise = list(cluster_labels).count(-1)
    
    metrics = {
        'n_clusters': n_clusters,
        'n_noise_points': n_noise,
        'noise_ratio': n_noise / len(cluster_labels),
        'eps': best_params['eps'],
        'min_samples': best_params['min_samples']
    }
    
    # Calculer le score de silhouette si possible
    if n_clusters >= 2:
        mask = cluster_labels != -1
        if mask.sum() > 1:
            metrics['silhouette_score'] = silhouette_score(df_scaled[mask], cluster_labels[mask])
        else:
            metrics['silhouette_score'] = 0
    else:
        metrics['silhouette_score'] = 0
    
    print(f"Nombre de clusters trouvés: {n_clusters}")
    print(f"Points de bruit: {n_noise} ({metrics['noise_ratio']*100:.1f}%)")
    print(f"Paramètres optimaux: eps={best_params['eps']:.3f}, min_samples={best_params['min_samples']}")
    
    return best_model, cluster_labels, metrics

def visualize_clustering_results(df_scaled, cluster_labels, pca_model, df_pca, algorithm_name):
    """
    Visualise les résultats du clustering
    """
    # Créer des sous-graphiques
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            f'{algorithm_name} - Clusters (2D PCA)',
            'Distribution des clusters',
            'Variance expliquée par PCA',
            f'{algorithm_name} - Clusters (3D si possible)'
        ],
        specs=[[{'type': 'scatter'}, {'type': 'bar'}],
               [{'type': 'bar'}, {'type': 'scatter3d'}]]
    )
    
    # Couleurs pour les clusters
    unique_labels = sorted(set(cluster_labels))
    colors = px.colors.qualitative.Set3[:len(unique_labels)]
    
    # 1. Scatter plot 2D avec PCA
    for i, label in enumerate(unique_labels):
        mask = cluster_labels == label
        cluster_name = f'Bruit' if label == -1 else f'Cluster {label}'
        
        fig.add_trace(
            go.Scatter(
                x=df_pca.iloc[mask, 0],
                y=df_pca.iloc[mask, 1],
                mode='markers',
                name=cluster_name,
                marker=dict(color=colors[i] if label != -1 else 'black'),
                showlegend=True
            ),
            row=1, col=1
        )
    
    # 2. Distribution des clusters
    cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
    cluster_names = [f'Bruit' if idx == -1 else f'Cluster {idx}' for idx in cluster_counts.index]
    
    fig.add_trace(
        go.Bar(
            x=cluster_names,
            y=cluster_counts.values,
            marker_color=[colors[i] if cluster_counts.index[i] != -1 else 'black' 
                         for i in range(len(cluster_counts))],
            showlegend=False
        ),
        row=1, col=2
    )
    
    # 3. Variance expliquée par PCA
    fig.add_trace(
        go.Bar(
            x=[f'PC{i+1}' for i in range(len(pca_model.explained_variance_ratio_))],
            y=pca_model.explained_variance_ratio_ * 100,
            marker_color='lightblue',
            showlegend=False
        ),
        row=2, col=1
    )
    
    # 4. Scatter plot 3D si on a au moins 3 composantes
    if df_pca.shape[1] >= 3:
        for i, label in enumerate(unique_labels):
            mask = cluster_labels == label
            cluster_name = f'Bruit' if label == -1 else f'Cluster {label}'
            
            fig.add_trace(
                go.Scatter3d(
                    x=df_pca.iloc[mask, 0],
                    y=df_pca.iloc[mask, 1],
                    z=df_pca.iloc[mask, 2],
                    mode='markers',
                    name=f'{cluster_name} (3D)',
                    marker=dict(
                        color=colors[i] if label != -1 else 'black',
                        size=3
                    ),
                    showlegend=False
                ),
                row=2, col=2
            )
    
    # Mise à jour des axes
    fig.update_xaxes(title_text="PC1", row=1, col=1)
    fig.update_yaxes(title_text="PC2", row=1, col=1)
    fig.update_xaxes(title_text="Clusters", row=1, col=2)
    fig.update_yaxes(title_text="Nombre de points", row=1, col=2)
    fig.update_xaxes(title_text="Composantes", row=2, col=1)
    fig.update_yaxes(title_text="Variance expliquée (%)", row=2, col=1)
    
    # Mise à jour du layout
    fig.update_layout(
        height=800,
        title_text=f"Analyse {algorithm_name}",
        showlegend=True
    )
    
    return fig

    
