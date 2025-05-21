import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Importer les fonctions depuis le fichier model_functions.py
from model_functions import (
    # Fonctions d'apprentissage supervisé
    prepare_data_for_regression,
    regression_par_segment,
    random_forest_par_segment,
    xgboost_simple,
    comparer_modeles,
    
    # Fonctions d'imputation
    impute_missing_prices,
    impute_condition_simple,
    impute_finishing_simple,
    impute_property_year_age,
    impute_binary_amenities,
    simple_impute_rooms,
    analyze_missing_data
)

st.set_page_config(
    page_title="Analyse Immobilière ML",
    page_icon="🏠",
    layout="wide"
)

st.title("Analyse du Marché Immobilier Tunisien avec ML")

# Fonction pour nettoyer et préparer les données
def preprocess_data(df):
    # Liste des colonnes numériques à convertir
    numeric_columns = ['price', 'size', 'rooms', 'bedrooms', 'bathrooms', 'parkings', 
                      'construction_year', 'age', 'air_conditioning', 'central_heating', 
                      'swimming_pool', 'elevator', 'garden', 'equipped_kitchen']
    
    # Convertir chaque colonne en numérique
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Convertir les dates
    if 'listing_date' in df.columns:
        try:
            df['listing_date'] = pd.to_datetime(df['listing_date'], errors='coerce')
        except:
            pass
    
    # Remplacer les valeurs potentiellement problématiques
    df = df.replace(['\\N', 'N/A', 'NA', ''], np.nan)
    
    # Standardiser la casse pour les colonnes catégorielles
    categorical_columns = ['property_type', 'transaction', 'city', 'state', 'neighborhood', 'finishing', 'condition']
    for col in categorical_columns:
        if col in df.columns and df[col].dtype == 'object':
            # Convertir tout en minuscules pour standardiser
            df[col] = df[col].str.lower()
    
    return df

# Fonction pour générer les visualisations basiques
def basic_visualizations(df):
    # Visualisation par ville
    if 'city' in df.columns:
        st.subheader("Nombre de propriétés par ville")
        city_counts = df['city'].value_counts().reset_index()
        city_counts.columns = ['ville', 'nombre']
        city_counts['ville'] = city_counts['ville'].str.title()
        
        fig = px.bar(city_counts, x='ville', y='nombre', 
                   title="Nombre de propriétés par ville")
        st.plotly_chart(fig, use_container_width=True)
    
    # Prix moyen par type de propriété
    if 'property_type' in df.columns and 'price' in df.columns:
        valid_price_df = df.dropna(subset=['price'])
        if not valid_price_df.empty:
            st.subheader("Prix moyen par type de propriété")
            price_by_type = valid_price_df.groupby('property_type')['price'].mean().reset_index()
            price_by_type.columns = ['type', 'prix_moyen']
            price_by_type['type'] = price_by_type['type'].str.capitalize()
            
            fig = px.bar(price_by_type, x='type', y='prix_moyen', 
                       title="Prix moyen par type de propriété",
                       labels={'prix_moyen': 'Prix moyen (TND)', 'type': 'Type de bien'})
            st.plotly_chart(fig, use_container_width=True)
    
    # Relation taille vs prix
    if 'size' in df.columns and 'price' in df.columns:
        valid_data = df.dropna(subset=['size', 'price'])
        if len(valid_data) > 5:
            st.subheader("Relation entre taille et prix")
            
            if 'property_type' in valid_data.columns:
                plot_data = valid_data.copy()
                plot_data['property_type_display'] = plot_data['property_type'].str.capitalize()
                
                fig = px.scatter(plot_data, x='size', y='price', 
                               color='property_type_display',
                               title="Relation entre taille et prix",
                               labels={'size': 'Surface (m²)', 'price': 'Prix (TND)', 
                                     'property_type_display': 'Type de bien'})
            else:
                fig = px.scatter(valid_data, x='size', y='price', 
                               title="Relation entre taille et prix",
                               labels={'size': 'Surface (m²)', 'price': 'Prix (TND)'})
            
            st.plotly_chart(fig, use_container_width=True)

# Nouvelles visualisations
def advanced_visualizations(df):
    if df is None or df.empty:
        st.warning("Aucune donnée disponible pour les visualisations avancées.")
        return
        
    col1, col2 = st.columns(2)
    
    # Distribution des conditions
    with col1:
        if 'condition' in df.columns and not df['condition'].isna().all():
            st.subheader("Distribution des états de propriété")
            condition_counts = df['condition'].value_counts().reset_index()
            condition_counts.columns = ['état', 'nombre']
            condition_counts['état'] = condition_counts['état'].str.capitalize()
            
            fig = px.pie(condition_counts, values='nombre', names='état', 
                      title="Distribution des états de propriété",
                      color_discrete_sequence=px.colors.qualitative.Pastel)
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
    
    # Distribution des finitions
    with col2:
        if 'finishing' in df.columns and not df['finishing'].isna().all():
            st.subheader("Niveau de finition des propriétés")
            finishing_counts = df['finishing'].value_counts().reset_index()
            finishing_counts.columns = ['finition', 'nombre']
            finishing_counts['finition'] = finishing_counts['finition'].str.capitalize()
            
            fig = px.pie(finishing_counts, values='nombre', names='finition', 
                      title="Niveau de finition des propriétés",
                      color_discrete_sequence=px.colors.qualitative.Bold)
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
    
    # Visualisation des transactions par quartier
    if 'transaction' in df.columns and 'neighborhood' in df.columns and not df['neighborhood'].isna().all():
        st.subheader("Types de transaction par quartier")
        try:
            transaction_by_neighborhood = pd.crosstab(df['neighborhood'], df['transaction'])
            transaction_by_neighborhood = transaction_by_neighborhood.reset_index()
            transaction_by_neighborhood_melted = pd.melt(
                transaction_by_neighborhood, 
                id_vars=['neighborhood'], 
                var_name='transaction', 
                value_name='count'
            )
            transaction_by_neighborhood_melted['neighborhood'] = transaction_by_neighborhood_melted['neighborhood'].str.title()
            transaction_by_neighborhood_melted['transaction'] = transaction_by_neighborhood_melted['transaction'].str.capitalize()
            
            # Limiter aux 15 quartiers les plus fréquents pour lisibilité
            top_neighborhoods = df['neighborhood'].value_counts().nlargest(15).index
            filtered_data = transaction_by_neighborhood_melted[
                transaction_by_neighborhood_melted['neighborhood'].str.lower().isin(top_neighborhoods)
            ]
            
            if not filtered_data.empty:
                fig = px.bar(filtered_data, x='neighborhood', y='count', color='transaction',
                           title="Types de transaction par quartier (top 15)",
                           labels={'count': 'Nombre', 'neighborhood': 'Quartier', 'transaction': 'Type de transaction'},
                           barmode='stack')
                fig.update_layout(xaxis={'categoryorder': 'total descending'})
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Impossible de générer la visualisation des transactions par quartier: {e}")
    
    # Matrice de corrélation
    numeric_df = df.select_dtypes(include=['number'])
    if numeric_df.shape[1] > 2:
        st.subheader("Matrice de corrélation")
        
        # Sélectionner seulement les colonnes numériques et supprimer les colonnes avec trop de NA
        cols_to_keep = numeric_df.columns[numeric_df.isnull().mean() < 0.5]
        if len(cols_to_keep) >= 2:  # Besoin d'au moins 2 colonnes pour une corrélation
            try:
                corr_df = numeric_df[cols_to_keep].corr()
                
                fig = px.imshow(corr_df, 
                               text_auto=True, 
                               aspect="auto",
                               color_continuous_scale='RdBu_r',
                               title="Corrélation entre les variables")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Impossible de générer la matrice de corrélation: {e}")
    
    # Distribution des prix par type de propriété
    if 'price' in df.columns and 'property_type' in df.columns:
        valid_data = df.dropna(subset=['price', 'property_type'])
        if len(valid_data) > 5:
            st.subheader("Distribution des prix par type de propriété")
            
            fig = px.box(valid_data, x='property_type', y='price',
                       labels={'property_type': 'Type de propriété', 'price': 'Prix (TND)'},
                       title="Distribution des prix par type de propriété",
                       category_orders={"property_type": sorted(valid_data['property_type'].unique())})
            
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)

# Nouvelle section pour les imputations de données
def imputation_section(df):
    st.header("Imputation des données manquantes")
    
    if df is None or df.empty:
        st.error("Aucune donnée disponible pour l'imputation.")
        return df
    
    # Création de l'interface d'imputation
    st.write("Cette section vous permet de compléter les valeurs manquantes dans votre jeu de données.")
    
    try:
        # Analyser les données manquantes
        missing_data_df = analyze_missing_data(df)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("État des valeurs manquantes")
            st.dataframe(missing_data_df)
        
        with col2:
            # Graphique des valeurs manquantes
            missing_cols = missing_data_df[missing_data_df['Valeurs NA'] > 0]
            if not missing_cols.empty:
                fig = px.bar(
                    missing_cols.reset_index(), 
                    x='index', 
                    y='Pourcentage NA (%)',
                    title="Pourcentage de valeurs manquantes par colonne",
                    labels={'index': 'Colonne', 'Pourcentage NA (%)': '% manquant'}
                )
                fig.update_layout(xaxis={'categoryorder': 'total descending'})
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("👍 Aucune valeur manquante dans vos données!")
        
        # Sélection des colonnes à imputer
        st.subheader("Choisir les colonnes à imputer")
        
        # Organisation par catégories
        price_cols = st.multiselect(
            "Colonnes de prix",
            [col for col in df.columns if col in ['price', 'price_ttc', 'listing_price']],
            [col for col in df.columns if col in ['price', 'price_ttc', 'listing_price'] and df[col].isna().sum() > 0]
        )
        
        condition_finishing_cols = st.multiselect(
            "Qualité et finition",
            [col for col in df.columns if col in ['condition', 'finishing']],
            [col for col in df.columns if col in ['condition', 'finishing'] and df[col].isna().sum() > 0]
        )
        
        age_year_cols = st.multiselect(
            "Âge et année de construction",
            [col for col in df.columns if col in ['age', 'construction_year']],
            [col for col in df.columns if col in ['age', 'construction_year'] and df[col].isna().sum() > 0]
        )
        
        room_cols = st.multiselect(
            "Pièces, chambres, salles de bain, etc.",
            [col for col in df.columns if col in ['rooms', 'bedrooms', 'bathrooms', 'parkings']],
            [col for col in df.columns if col in ['rooms', 'bedrooms', 'bathrooms', 'parkings'] and df[col].isna().sum() > 0]
        )
        
        binary_cols = st.multiselect(
            "Équipements (variables binaires)",
            [col for col in df.columns if df[col].nunique() <= 2 and col not in ['transaction', 'city', 'property_type', 'neighborhood']],
            [col for col in df.columns if df[col].nunique() <= 2 and df[col].isna().sum() > 0 and col not in ['transaction', 'city', 'property_type', 'neighborhood']]
        )
        
        # Bouton pour lancer l'imputation
        impute_button = st.button("Imputer les valeurs manquantes", type="primary")
        
        if impute_button:
            if df is None or df.empty:
                st.error("Aucune donnée disponible pour l'imputation.")
                return df
                
            # Créer une copie pour l'imputation
            try:
                df_imputed = df.copy()
                progress_placeholder = st.empty()
                
                with st.spinner("Imputation en cours..."):
                    # Imputation progressive avec barre de progression
                    progress_bar = st.progress(0)
                    
                    # 1. Imputation des prix
                    # 
                    
                    # 2. Imputation de la condition
                    if 'condition' in condition_finishing_cols:
                        progress_placeholder.write("Imputation de la condition...")
                        try:
                            df_imputed = impute_condition_simple(df_imputed)
                            progress_bar.progress(40)
                            time.sleep(0.5)
                        except Exception as e:
                            st.error(f"Erreur lors de l'imputation de la condition: {e}")
                    
                    # 3. Imputation de la finition
                    if 'finishing' in condition_finishing_cols:
                        progress_placeholder.write("Imputation du niveau de finition...")
                        try:
                            df_imputed = impute_finishing_simple(df_imputed)
                            progress_bar.progress(60)
                            time.sleep(0.5)
                        except Exception as e:
                            st.error(f"Erreur lors de l'imputation de la finition: {e}")
                    
                    # 4. Imputation de l'âge et année de construction
                    if age_year_cols:
                        progress_placeholder.write("Imputation de l'âge et année de construction...")
                        try:
                            df_imputed = impute_property_year_age(
                                df_imputed, 
                                impute_year='construction_year' in age_year_cols, 
                                impute_age='age' in age_year_cols
                            )
                            df_imputed['construction_year']=2025-df_imputed['age']
                            progress_bar.progress(75)
                            time.sleep(0.5)
                        except Exception as e:
                            st.error(f"Erreur lors de l'imputation de l'âge/année: {e}")
                    
                    # 5. Imputation des caractéristiques binaires
                    if binary_cols:
                        progress_placeholder.write("Imputation des équipements...")
                        try:
                            df_imputed = impute_binary_amenities(df_imputed, binary_columns=binary_cols)
                            progress_bar.progress(85)
                            time.sleep(0.5)
                        except Exception as e:
                            st.error(f"Erreur lors de l'imputation des équipements: {e}")
                    
                    # 6. Imputation des pièces et caractéristiques
                    for room_col in room_cols:
                        progress_placeholder.write(f"Imputation de {room_col}...")
                        try:
                            df_imputed = simple_impute_rooms(df_imputed, rooms_col=room_col)
                            time.sleep(0.3)
                        except Exception as e:
                            st.error(f"Erreur lors de l'imputation de {room_col}: {e}")
                    # imputation des prix
                    if price_cols:
                        progress_placeholder.write("Imputation des prix...")
                        try:
                            # df_imputed = impute_missing_prices(df_imputed)
                            df_imputed['price'] = df_imputed.groupby(['neighborhood', 'property_type','transaction'])['price'].transform(lambda x: x.fillna(x.mean()))
                            df_imputed['price_ttc'] = df_imputed.groupby(['neighborhood', 'property_type','transaction'])['price_ttc'].transform(lambda x: x.fillna(x.mean()))
                            df_imputed['price'] = df.groupby(['city','transaction'])['price'].transform(lambda x: x.fillna(x.mean()))
                            df_imputed['price_ttc'] = df.groupby(['city','transaction'])['price_ttc'].transform(lambda x: x.fillna(x.mean()))
                            df_imputed = df_imputed[df_imputed['price'].notnull()]
    
                            df_imputed['suffix'] = df_imputed['suffix'].fillna('TTC')
                            
                            df_imputed['listing_price'] = df_imputed['listing_price'].fillna(df_imputed['price'])
                            
                            progress_bar.progress(100)
                            time.sleep(0.5)  
                        except Exception as e:
                            st.error(f"Erreur lors de l'imputation des prix: {e}")
                    progress_bar.progress(100)
                    progress_placeholder.empty()
                
                # Comparer l'avant/après
                st.subheader("Résultats de l'imputation")
                
                col1, col2 = st.columns(2)
                
                # Analyse des valeurs manquantes avant
                with col1:
                    df.drop(columns=['amenities'], inplace=True)
                    st.write("Avant imputation")
                    missing_before = analyze_missing_data(df)
                    st.dataframe(missing_before)
                    
                    # Pourcentage global des données manquantes avant
                    total_elements = df.shape[0] * df.shape[1]
                    total_missing = df.isna().sum().sum()
                    pct_missing_before = (total_missing / total_elements) * 100
                    
                    st.metric(
                        "Pourcentage global de données manquantes",
                        f"{pct_missing_before:.2f}%"
                    )
                
                # Analyse des valeurs manquantes après
                with col2:
                    st.write("Après imputation")
                    df_imputed.drop(columns=['amenities'], inplace=True)
                    missing_after = analyze_missing_data(df_imputed)
                    st.dataframe(missing_after)
                    
                    # Pourcentage global des données manquantes après
                    total_missing_after = df_imputed.isna().sum().sum()
                    pct_missing_after = (total_missing_after / total_elements) * 100
                    
                    st.metric(
                        "Pourcentage global de données manquantes",
                        f"{pct_missing_after:.2f}%",
                        f"-{pct_missing_before - pct_missing_after:.2f}%"
                    )
                
                # Visualisation de l'impact de l'imputation
                st.subheader("Visualisation de l'impact de l'imputation")
                
                # Sélectionnez une colonne pour visualiser l'impact de l'imputation
                all_imputed_cols = price_cols + condition_finishing_cols + age_year_cols + room_cols + binary_cols
                
                if True:
                    vis_col = st.selectbox(
                        "Sélectionner une colonne pour visualiser l'impact de l'imputation",
                        all_imputed_cols
                    )
                    
                    if vis_col in df_imputed.columns:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"Distribution de {vis_col} avant imputation")
                            
                            if pd.api.types.is_numeric_dtype(df[vis_col]):
                                # Histogramme pour les données numériques
                                fig = px.histogram(
                                    df.dropna(subset=[vis_col]), 
                                    x=vis_col,
                                    title=f"Distribution de {vis_col} avant imputation",
                                    nbins=30,
                                    opacity=0.7
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                # Barres pour les données catégorielles
                                value_counts = df[vis_col].value_counts().reset_index()
                                value_counts.columns = ['valeur', 'nombre']
                                
                                fig = px.bar(
                                    value_counts,
                                    x='valeur',
                                    y='nombre',
                                    title=f"Distribution de {vis_col} avant imputation"
                                )
                                st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            st.write(f"Distribution de {vis_col} après imputation")
                            
                            if pd.api.types.is_numeric_dtype(df_imputed[vis_col]):
                                # Histogramme pour les données numériques
                                fig = px.histogram(
                                    df_imputed, 
                                    x=vis_col,
                                    title=f"Distribution de {vis_col} après imputation",
                                    nbins=30,
                                    opacity=0.7
                                )
                                # Ajouter une ligne pour marquer les valeurs imputées
                                orig_values = df[~df[vis_col].isna()][vis_col]
                                fig.add_traces(
                                    px.histogram(
                                        orig_values, 
                                        x=orig_values,
                                        nbins=30,
                                        opacity=0.7
                                    ).data
                                )
                                fig.data[0].marker.color = 'blue'  # Toutes les valeurs
                                fig.data[1].marker.color = 'lightgreen'  # Valeurs originales
                                fig.data[0].name = 'Toutes les valeurs (incluant imputées)'
                                fig.data[1].name = 'Valeurs originales uniquement'
                                fig.update_layout(barmode='overlay', legend=dict(orientation='h'))
                                
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                # Barres pour les données catégorielles
                                value_counts = df_imputed[vis_col].value_counts().reset_index()
                                value_counts.columns = ['valeur', 'nombre']
                                
                                fig = px.bar(
                                    value_counts,
                                    x='valeur',
                                    y='nombre',
                                    title=f"Distribution de {vis_col} après imputation"
                                )
                                st.plotly_chart(fig, use_container_width=True)
                
                # Option pour continuer avec les données imputées
                if st.button("Utiliser les données imputées pour la suite de l'analyse"):
                    st.session_state['df_imputed'] = df_imputed
                    st.success("✅ Les données imputées sont maintenant utilisées pour l'analyse!")
                    st.experimental_rerun()  # Réexécuter l'application pour utiliser les données imputées
                    
                return df_imputed  # Retourner les données imputées
                
            except Exception as e:
                st.error(f"Erreur lors de l'imputation: {e}")
                st.info("Conseil: Vérifiez les données et essayez à nouveau.")
                return df
    except Exception as e:
        st.error(f"Erreur lors de l'analyse des données manquantes: {e}")
        return df
        
    return df

# Nouvelle section pour l'apprentissage supervisé
def supervised_learning_section(df, filtered_df):
    st.header("Modèles d'Apprentissage Supervisé")
    
    if df is None or filtered_df is None or df.empty or filtered_df.empty:
        st.error("Aucune donnée disponible pour l'apprentissage supervisé.")
        return
    
    # Préparer les données pour la régression
    with st.spinner("Préparation des données pour l'apprentissage..."):
        try:
            df_prep = prepare_data_for_regression(filtered_df)
            st.success("Données préparées pour l'apprentissage supervisé !")
        except Exception as e:
            st.error(f"Erreur lors de la préparation des données : {e}")
            return
    
    # Paramètres pour les modèles
    st.subheader("Paramètres du modèle")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if 'city' in filtered_df.columns:
            city_options = ["Toutes"] + sorted(filtered_df['city'].dropna().unique().tolist())
            selected_city = st.selectbox("Ville pour le modèle", city_options)
            selected_city = None if selected_city == "Toutes" else selected_city
        else:
            selected_city = None
            st.write("Information sur la ville non disponible")
    
    with col2:
        if 'property_type' in filtered_df.columns:
            property_options = ["Tous"] + sorted(filtered_df['property_type'].dropna().unique().tolist())
            selected_property = st.selectbox("Type de propriété pour le modèle", property_options)
            selected_property = None if selected_property == "Tous" else selected_property
        else:
            selected_property = None
            st.write("Information sur le type de propriété non disponible")
    
    with col3:
        if 'transaction' in filtered_df.columns:
            transaction_options = ["Toutes"] + sorted(filtered_df['transaction'].dropna().unique().tolist())
            selected_transaction = st.selectbox("Type de transaction pour le modèle", transaction_options)
            selected_transaction = None if selected_transaction == "Toutes" else selected_transaction
        else:
            selected_transaction = None
            st.write("Information sur le type de transaction non disponible")
    
    # Sélection du modèle
    model_type = st.selectbox(
        "Sélectionner le modèle",
        ["Comparaison de modèles", "Régression Linéaire", "Random Forest", "XGBoost"]
    )
    
    # Bouton pour lancer l'entraînement
    if st.button("Entraîner le modèle"):
        with st.spinner("Entraînement du modèle en cours..."):
            # Vérifier qu'il y a assez de données
            if len(df_prep) < 10:
                st.error("Pas assez de données pour l'entraînement du modèle. Veuillez élargir les critères de sélection.")
            else:
                try:
                    # Créer un conteneur pour les résultats
                    results_container = st.container()
                    
                    with results_container:
                        st.subheader(f"Résultats pour {model_type}")
                        
                        # Exécuter le modèle sélectionné
                        if model_type == "Comparaison de modèles":
                            comparison = comparer_modeles(
                                df_prep, 
                                city=selected_city, 
                                property_type=selected_property, 
                                transaction=selected_transaction
                            )
                            st.dataframe(comparison)
                            
                            try:
                                # Convertir les figures Matplotlib en Plotly pour Streamlit
                                st.pyplot(plt.gcf())  # Récupère la figure actuelle (courante)
                            except Exception as e:
                                st.warning(f"Impossible d'afficher le graphique: {e}")
                        
                        elif model_type == "Régression Linéaire":
                            model, importance, metrics = regression_par_segment(
                                df_prep, 
                                city=selected_city, 
                                property_type=selected_property, 
                                transaction=selected_transaction
                            )
                            
                            # Afficher les métriques
                            st.write(f"R² (test): {metrics['test_r2']:.4f}")
                            st.write(f"RMSE (test): {metrics['test_rmse']:.2f}")
                            st.write(f"MAE (test): {metrics['test_mae']:.2f}")
                            
                            # Afficher l'importance des caractéristiques
                            st.subheader("Importance des caractéristiques")
                            
                            # Créer un graphique Plotly pour l'importance des caractéristiques
                            top_features = importance.head(10)
                            fig = px.bar(
                                top_features,
                                x='Coefficient',
                                y='Caractéristique',
                                orientation='h',
                                title="Top 10 des caractéristiques les plus importantes",
                                color='Coefficient',
                                color_continuous_scale=px.colors.diverging.RdBu,
                                color_continuous_midpoint=0
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            try:
                                # Convertir les figures Matplotlib en Plotly pour Streamlit
                                st.pyplot(plt.gcf())
                            except Exception as e:
                                st.warning(f"Impossible d'afficher le graphique: {e}")
                        
                        elif model_type == "Random Forest":
                            model, importance, metrics = random_forest_par_segment(
                                df_prep, 
                                city=selected_city, 
                                property_type=selected_property, 
                                transaction=selected_transaction
                            )
                            
                            # Afficher les métriques
                            st.write(f"R² (test): {metrics['test_r2']:.4f}")
                            st.write(f"RMSE (test): {metrics['test_rmse']:.2f}")
                            st.write(f"MAE (test): {metrics['test_mae']:.2f}")
                            
                            # Afficher l'importance des caractéristiques
                            st.subheader("Importance des caractéristiques")
                            
                            # Créer un graphique Plotly pour l'importance des caractéristiques
                            top_features = importance.head(10)
                            fig = px.bar(
                                top_features,
                                x='Importance',
                                y='Caractéristique',
                                orientation='h',
                                title="Top 10 des caractéristiques les plus importantes",
                                color='Importance',
                                color_continuous_scale='Viridis'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            try:
                                # Convertir les figures Matplotlib en Plotly pour Streamlit
                                st.pyplot(plt.gcf())
                            except Exception as e:
                                st.warning(f"Impossible d'afficher le graphique: {e}")
                        
                        elif model_type == "XGBoost":
                            model, importance, test_r2 = xgboost_simple(
                                df_prep, 
                                city=selected_city, 
                                property_type=selected_property, 
                                transaction=selected_transaction
                            )
                            
                            # Afficher les métriques
                            st.write(f"R² (test): {test_r2:.4f}")
                            
                            # Afficher l'importance des caractéristiques
                            st.subheader("Importance des caractéristiques")
                            
                            # Créer un graphique Plotly pour l'importance des caractéristiques
                            top_features = importance.head(10)
                            fig = px.bar(
                                top_features,
                                x='Importance',
                                y='Caractéristique',
                                orientation='h',
                                title="Top 10 des caractéristiques les plus importantes",
                                color='Importance',
                                color_continuous_scale='Viridis'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            try:
                                # Convertir les figures Matplotlib en Plotly pour Streamlit
                                st.pyplot(plt.gcf())
                            except Exception as e:
                                st.warning(f"Impossible d'afficher le graphique: {e}")
                            
                except Exception as e:
                    st.error(f"Une erreur s'est produite lors de l'entraînement du modèle: {e}")

# Application principale
# Initialiser les variables de session
if 'df_imputed' not in st.session_state:
    st.session_state['df_imputed'] = None

# Téléchargement du fichier
uploaded_file = st.file_uploader("Télécharger un fichier CSV", type=['csv'])

if uploaded_file is not None:
    try:
        # Essayer de charger avec différents séparateurs
        df = None
        try:
            df = pd.read_csv(uploaded_file, sep=',')
        except:
            try:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, sep=';')
            except:
                try:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, sep='\t')
                except Exception as e:
                    st.error(f"Impossible de lire le fichier CSV: {e}")
        
        if df is None or df.empty:
            st.error("Le fichier est vide ou n'a pas pu être lu correctement.")
        else:
            # Prétraiter les données
            df = preprocess_data(df)
            
            # Utiliser les données imputées si disponibles
            if st.session_state['df_imputed'] is not None:
                df = st.session_state['df_imputed']
                st.success("Utilisation des données imputées!")
            else:
                st.success("Fichier chargé avec succès!")
            
            # Filtres dans la barre latérale
            st.sidebar.title("Filtres")
            
            if 'transaction' in df.columns:
                unique_transactions = sorted(df['transaction'].dropna().unique().tolist())
                display_transactions = ["Tous"] + [t.capitalize() for t in unique_transactions if isinstance(t, str)]
                transaction_map = {t.capitalize(): t for t in unique_transactions if isinstance(t, str)}
                transaction_map["Tous"] = "Tous"
                
                selected_display_transaction = st.sidebar.selectbox("Type de Transaction", display_transactions)
                selected_transaction = transaction_map[selected_display_transaction]
            else:
                selected_transaction = "Tous"
            
            if 'property_type' in df.columns:
                unique_property_types = sorted(df['property_type'].dropna().unique().tolist())
                display_property_types = ["Tous"] + [p.capitalize() for p in unique_property_types if isinstance(p, str)]
                property_type_map = {p.capitalize(): p for p in unique_property_types if isinstance(p, str)}
                property_type_map["Tous"] = "Tous"
                
                selected_display_property = st.sidebar.selectbox("Type de Bien", display_property_types)
                selected_property = property_type_map[selected_display_property]
            else:
                selected_property = "Tous"
            
            if 'city' in df.columns:
                unique_cities = sorted(df['city'].dropna().unique().tolist())
                display_cities = ["Toutes"] + [c.title() for c in unique_cities if isinstance(c, str)]
                city_map = {c.title(): c for c in unique_cities if isinstance(c, str)}
                city_map["Toutes"] = "Toutes"
                
                selected_display_city = st.sidebar.selectbox("Ville", display_cities)
                selected_city = city_map[selected_display_city]
            else:
                selected_city = "Toutes"
            
            # Appliquer les filtres
            filtered_df = df.copy()
            if selected_transaction != "Tous" and 'transaction' in df.columns:
                filtered_df = filtered_df[filtered_df['transaction'] == selected_transaction]
            if selected_property != "Tous" and 'property_type' in df.columns:
                filtered_df = filtered_df[filtered_df['property_type'] == selected_property]
            if selected_city != "Toutes" and 'city' in df.columns:
                filtered_df = filtered_df[filtered_df['city'] == selected_city]
            
            # Créer des onglets pour organiser l'interface
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["Aperçu des données", "Statistiques de base", "Imputation", "Visualisations avancées", "Apprentissage supervisé"])
            
            with tab1:
                st.subheader("Aperçu des données")
                st.dataframe(filtered_df.head())
                
                st.subheader("Informations sur les colonnes")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("Types de données:")
                    st.write(pd.DataFrame({'Type': filtered_df.dtypes}))
                
                with col2:
                    st.write("Valeurs manquantes:")
                    missing_data = pd.DataFrame({
                        'Manquantes': filtered_df.isna().sum(), 
                        'Pourcentage': (filtered_df.isna().sum() / len(filtered_df) * 100).round(2)
                    })
                    st.write(missing_data)
            
            with tab2:
                # Statistiques de base
                st.subheader("Statistiques descriptives")
                
                # Sélectionner uniquement les colonnes numériques pour describe()
                numeric_df = filtered_df.select_dtypes(include=['number'])
                if not numeric_df.empty:
                    st.dataframe(numeric_df.describe())
                else:
                    st.warning("Aucune colonne numérique disponible pour l'analyse statistique.")
                
                # Métriques clés
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Nombre de propriétés", len(filtered_df))
                
                with col2:
                    if 'price' in filtered_df.columns:
                        avg_price = filtered_df['price'].dropna().mean()
                        if pd.notna(avg_price):
                            st.metric("Prix moyen", f"{avg_price:,.0f} TND")
                        else:
                            st.metric("Prix moyen", "N/A")
                    else:
                        st.metric("Prix moyen", "Colonne manquante")
                
                with col3:
                    if 'size' in filtered_df.columns:
                        avg_size = filtered_df['size'].dropna().mean()
                        if pd.notna(avg_size):
                            st.metric("Surface moyenne", f"{avg_size:.1f} m²")
                        else:
                            st.metric("Surface moyenne", "N/A")
                    else:
                        st.metric("Surface moyenne", "Colonne manquante")
                
                # Visualisations de base
                basic_visualizations(filtered_df)
            
            with tab3:
                # Section d'imputation
                updated_df = imputation_section(df)
                if updated_df is not None:
                    df = updated_df  # Mettre à jour le dataframe avec les données imputées
            
            with tab4:
                # Visualisations avancées
                advanced_visualizations(filtered_df)
            
            with tab5:
                # Section d'apprentissage supervisé
                supervised_learning_section(df, filtered_df)
        
    except Exception as e:
        st.error(f"Une erreur s'est produite lors du traitement du fichier: {e}")
        st.info("Conseil: Vérifiez le format de votre fichier CSV et assurez-vous que les colonnes sont correctement nommées.")