import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# from IPython.display import display
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

# apprentissage supervisé
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
