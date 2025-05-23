import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Importer les fonctions depuis le fichier model_functions.py
from model_functions import *

# Configuration de la page
st.set_page_config(
    page_title="Analyse Immobilière ",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour améliorer l'apparence
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #2563EB;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #3B82F6;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
    }
    .highlight {
        background-color: #EFF6FF;
        padding: 20px;
        border-radius: 5px;
        border-left: 5px solid #3B82F6;
        margin-bottom: 20px;
    }
    .info-box {
        background-color: #DBEAFE;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 15px;
    }
    .stat-card {
        background-color: #F8FAFC;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 15px;
        text-align: center;
    }
    .footer {
        margin-top: 50px;
        padding-top: 20px;
        border-top: 1px solid #E5E7EB;
        text-align: center;
        font-size: 0.9rem;
        color: #6B7280;
    }
</style>
""", unsafe_allow_html=True)

# Header de l'application
st.title("🏠 Analyse du Marché Immobilier Tunisien")

# Introduction et contexte
with st.expander("📌 À propos de cette application", expanded=True):
    st.markdown("""
    <div class="highlight">
    <h4>Contexte</h4>
    <p>Cette application vous permet d'analyser en profondeur le marché immobilier tunisien à travers des données collectées depuis des sites de franchises immobilières  (Century 21, REMAX, Tecnocasa et Newkey). Que vous soyez un investisseur, un agent immobilier, ou simplement à la recherche d'un bien, cet outil vous fournit des insights précieux sur les tendances du marché.</p>
    
    <h4>Fonctionnalités</h4>
    <ul>
        <li><strong>Analyse exploratoire</strong> : Visualisez les distributions des prix, surfaces, et autres caractéristiques des biens</li>
        <li><strong>Traitement des données</strong> : Nettoyez et imputez les valeurs manquantes pour une analyse plus précise</li>
        <li><strong>Modélisation prédictive</strong> : Utilisez des algorithmes d'apprentissage automatique pour prédire les prix et identifier les facteurs déterminants</li>
        <li><strong>Segmentation géographique</strong> : Analysez les spécificités du marché par ville et quartier</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
uploaded_file = st.file_uploader("Télécharger un fichier CSV", type=['csv'])


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
                    st.rerun()  # Réexécuter l'application pour utiliser les données imputées
                    
                return df_imputed  # Retourner les données imputées
                
            except Exception as e:
                st.error(f"Erreur lors de l'imputation: {e}")
                st.info("Conseil: Vérifiez les données et essayez à nouveau.")
                return df
    except Exception as e:
        st.error(f"Erreur lors de l'analyse des données manquantes: {e}")
        return df
        
    return df

# REMPLACER VOTRE FONCTION add_price_prediction_section PAR CELLE-CI

def simple_price_calculator(model, feature_names, df_regression, model_type="Régression Linéaire"):
    """
    Calculateur de prix simple corrigé pour éviter l'erreur de dimensionnalité
    """
    
    st.markdown("---")
    st.subheader(f"🔮 Calculateur de Prix - {model_type}")
    
    # Obtenir un échantillon des données d'entraînement pour la structure
    sample_row = df_regression.iloc[0:1].copy()  # Prendre la première ligne comme template
    
    # Statistiques pour valeurs par défaut
    stats = df_regression.describe()
    
    # Interface simple en 2 colonnes
    col1, col2 = st.columns(2)
    
    # Dictionnaire pour stocker les inputs utilisateur
    user_inputs = {}
    
    with col1:
        st.markdown("#### 📐 **Caractéristiques Principales**")
        
        # Surface
        if 'size' in feature_names:
            size_default = int(stats.loc['mean', 'size']) if 'size' in stats.columns else 100
            user_inputs['size'] = st.number_input("Surface (m²)", value=size_default, min_value=20, max_value=1000, step=5, key=f"size_{model_type}")
        
        # Pièces
        if 'rooms' in feature_names:
            rooms_default = int(stats.loc['mean', 'rooms']) if 'rooms' in stats.columns else 3
            user_inputs['rooms'] = st.number_input("Pièces", value=rooms_default, min_value=1, max_value=15, step=1, key=f"rooms_{model_type}")
        
        # Chambres
        if 'bedrooms' in feature_names:
            bedrooms_default = int(stats.loc['mean', 'bedrooms']) if 'bedrooms' in stats.columns else 2
            user_inputs['bedrooms'] = st.number_input("Chambres", value=bedrooms_default, min_value=0, max_value=10, step=1, key=f"bedrooms_{model_type}")
        
        # Salles de bain
        if 'bathrooms' in feature_names:
            bathrooms_default = int(stats.loc['mean', 'bathrooms']) if 'bathrooms' in stats.columns else 1
            user_inputs['bathrooms'] = st.number_input("Salles de bain", value=bathrooms_default, min_value=1, max_value=5, step=1, key=f"bathrooms_{model_type}")
        
        # Parkings
        if 'parkings' in feature_names:
            parkings_default = int(stats.loc['mean', 'parkings']) if 'parkings' in stats.columns else 1
            user_inputs['parkings'] = st.number_input("Parkings", value=parkings_default, min_value=0, max_value=5, step=1, key=f"parkings_{model_type}")
    
    with col2:
        st.markdown("#### ⭐ **Qualité & Équipements**")
        
        # Âge
        if 'age' in feature_names:
            age_default = int(stats.loc['mean', 'age']) if 'age' in stats.columns else 10
            user_inputs['age'] = st.number_input("Âge (années)", value=age_default, min_value=0, max_value=100, step=1, key=f"age_{model_type}")
        
        # État
        if 'condition' in feature_names:
            user_inputs['condition'] = st.selectbox(
                "État",
                options=[0, 1, 2, 3, 4],
                format_func=lambda x: ["À rénover", "À rafraîchir", "Bonne", "Excellente", "Neuf"][x],
                index=2,
                key=f"condition_{model_type}"
            )
        
        # Standing
        if 'finishing' in feature_names:
            user_inputs['finishing'] = st.selectbox(
                "Standing",
                options=[0, 1, 2, 3, 4],
                format_func=lambda x: ["Social", "Économique", "Moyen", "Haut", "Très haut"][x],
                index=2,
                key=f"finishing_{model_type}"
            )
        
        # Équipements
        if 'elevator' in feature_names:
            user_inputs['elevator'] = 1 if st.checkbox("🏢 Ascenseur", key=f"elevator_{model_type}") else 0
        
        if 'air_conditioning' in feature_names:
            user_inputs['air_conditioning'] = 1 if st.checkbox("❄️ Climatisation", key=f"ac_{model_type}") else 0
        
        if 'central_heating' in feature_names:
            user_inputs['central_heating'] = 1 if st.checkbox("🔥 Chauffage", value=True, key=f"heating_{model_type}") else 0
        
        if 'swimming_pool' in feature_names:
            user_inputs['swimming_pool'] = 1 if st.checkbox("🏊 Piscine", key=f"pool_{model_type}") else 0
        
        if 'garden' in feature_names:
            user_inputs['garden'] = 1 if st.checkbox("🌳 Jardin", key=f"garden_{model_type}") else 0
        
        if 'equipped_kitchen' in feature_names:
            user_inputs['equipped_kitchen'] = 1 if st.checkbox("👨‍🍳 Cuisine équipée", value=True, key=f"kitchen_{model_type}") else 0
    
    
    
    # CALCUL EN TEMPS RÉEL
    try:
        # Créer une copie de la ligne sample pour la prédiction
        prediction_row = sample_row.copy()
        
        # Mettre à jour avec les valeurs de l'utilisateur
        for feature, value in user_inputs.items():
            if feature in prediction_row.columns:
                prediction_row[feature] = value
        
        # Supprimer la colonne 'price' si elle existe (c'est la variable cible)
        if 'price' in prediction_row.columns:
            prediction_row = prediction_row.drop('price', axis=1)
        
        # S'assurer que toutes les features attendues sont présentes
        missing_features = set(feature_names) - set(prediction_row.columns)
        if missing_features:
            st.warning(f"⚠️ Features manquantes: {missing_features}")
            # Ajouter les features manquantes avec des valeurs par défaut
            for feature in missing_features:
                prediction_row[feature] = 0
        
        # Réorganiser les colonnes dans l'ordre attendu par le modèle
        prediction_row = prediction_row[feature_names]
        
        # Prédiction
        predicted_price = model.predict(prediction_row)[0]
        
        # AFFICHAGE DU PRIX
        st.markdown("### 💰 **Prix Estimé**")
        
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
            padding: 1.5rem;
            border-radius: 10px;
            text-align: center;
            color: white;
            margin: 1rem 0;
        ">
            <h1 style="margin: 0; font-size: 2.5rem; color: white;">
                {predicted_price:,.0f} TND
            </h1>
        </div>
        """, unsafe_allow_html=True)
        
        # Métriques rapides
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'size' in user_inputs and user_inputs['size'] > 0:
                price_per_sqm = predicted_price / user_inputs['size']
                st.metric("Prix/m²", f"{price_per_sqm:,.0f} TND")
        
        with col2:
            if 'price' in df_regression.columns:
                market_avg = df_regression['price'].mean()
                diff_pct = ((predicted_price - market_avg) / market_avg) * 100
                st.metric("vs Marché", f"{diff_pct:+.1f}%")
        
        with col3:
            lower = predicted_price * 0.9
            upper = predicted_price * 1.1
            st.metric("Fourchette", f"{lower:,.0f} - {upper:,.0f}")
        
        # Debug info (optionnel)
        with st.expander("🔧 Info Debug", expanded=False):
            st.write(f"Features utilisées: {len(feature_names)}")
            st.write(f"Valeurs utilisateur: {len(user_inputs)}")
            st.write(f"Shape finale: {prediction_row.shape}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Inputs utilisateur:**")
                for k, v in user_inputs.items():
                    st.write(f"• {k}: {v}")
            
            with col2:
                st.write("**Features modèle:**")
                for i, feature in enumerate(feature_names[:10]):  # Afficher les 10 premières
                    st.write(f"• {feature}")
                if len(feature_names) > 10:
                    st.write(f"... et {len(feature_names)-10} autres")
    
    except Exception as e:
        st.error(f"❌ Erreur dans le calcul: {str(e)}")
        
        # Debug détaillé
        st.write("**Debug détaillé:**")
        st.write(f"• Modèle attend: {len(feature_names)} features")
        st.write(f"• Features: {feature_names}")
        st.write(f"• User inputs: {len(user_inputs)} valeurs")
        
        if 'prediction_row' in locals():
            st.write(f"• Prediction row shape: {prediction_row.shape}")
            st.write(f"• Prediction row columns: {list(prediction_row.columns)}")


def supervised_learning_section(df, filtered_df):
    st.header("🤖 Apprentissage Supervisé - Prédiction des Prix")
    
    if df is None or filtered_df is None or df.empty or filtered_df.empty:
        st.error("❌ Aucune donnée disponible pour l'apprentissage supervisé.")
        return
    
    st.markdown("""
    <div class="info-box">
    L'apprentissage supervisé permet de prédire les prix immobiliers en analysant les relations entre 
    les caractéristiques des propriétés et leurs prix. Trois algorithmes sont disponibles : 
    Régression Linéaire, Random Forest et XGBoost.
    </div>
    """, unsafe_allow_html=True)
    
    # ============================================
    # SECTION 1: VÉRIFICATION ET PRÉPARATION DES DONNÉES
    # ============================================
    
   
    # ============================================
    # SECTION 2: FILTRES POUR LA MODÉLISATION
    # ============================================
    
    st.subheader("🔧 Configuration du Modèle")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Filtre par ville
        if 'city' in df.columns:
            city_options = ["Toutes"] + sorted(df['city'].dropna().unique().tolist())
            selected_city = st.selectbox("Ville pour le modèle", city_options, key="regression_city")
            selected_city = None if selected_city == "Toutes" else selected_city
        else:
            selected_city = None
            st.info("Information sur la ville non disponible")
    
    with col2:
        # Filtre par type de propriété
        if 'property_type' in df.columns:
            property_options = ["Tous"] + sorted(df['property_type'].dropna().unique().tolist())
            selected_property = st.selectbox("Type de propriété pour le modèle", property_options, key="regression_property")
            selected_property = None if selected_property == "Tous" else selected_property
        else:
            selected_property = None
            st.info("Information sur le type de propriété non disponible")
    
    with col3:
        # Filtre par type de transaction
        if 'transaction' in df.columns:
            transaction_options = ["Toutes"] + sorted(df['transaction'].dropna().unique().tolist())
            selected_transaction = st.selectbox("Type de transaction pour le modèle", transaction_options, key="regression_transaction")
            selected_transaction = None if selected_transaction == "Toutes" else selected_transaction
        else:
            selected_transaction = None
            st.info("Information sur le type de transaction non disponible")
    
    # ============================================
    # SECTION 3: SÉLECTION DE L'ALGORITHME
    # ============================================
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        algorithm = st.selectbox(
            "Sélectionner l'algorithme",
            [
                "Régression Linéaire", 
                "Random Forest Classification Prix",  # ← Nouvelle option
                "XGBoost Classification Prix"  # ← Nouvelle option
            ],
            help="Choisissez l'algorithme d'apprentissage supervisé à utiliser"
        )
    
    with col2:
        # Options avancées
        with st.expander("⚙️ Options avancées"):
            test_size = st.slider("Taille ensemble test (%)", 10, 40, 20) / 100
            random_state = st.number_input("Graine aléatoire", value=42, min_value=0)
            
            # Paramètres spécifiques aux modèles
            if algorithm in ["Random Forest", "Comparaison des 3 modèles"]:
                n_estimators = st.slider("Nombre d'arbres (Random Forest)", 50, 500, 100)
                max_depth_rf = st.slider("Profondeur max (Random Forest)", 3, 20, 10)
            
            if algorithm in ["XGBoost Classification Prix", "Comparaison des 3 modèles"]:
                optimize_params = st.checkbox("Optimiser les hyperparamètres", value=False, 
                                                    help="Recherche automatique des meilleurs paramètres",
                                                    key="xgb_class_optimize")
               
                threshold_low = st.slider("Seuil sous-estimation", 0.5, 0.9, 0.75, 0.05,
                                                key="xgb_class_threshold_low")
                threshold_high = st.slider("Seuil surestimation", 1.1, 1.5, 1.25, 0.05,
                                                key="xgb_class_threshold_high")
            if algorithm in ["Random Forest Classification Prix"]:
                pass
            
    
    
    # ============================================
    # SECTION 4: PRÉPARATION SÉCURISÉE DES DONNÉES
    # ============================================
    
    def prepare_data_safely(df, selected_city, selected_property, selected_transaction):
        """Préparation sécurisée des données avec gestion d'erreurs"""
        try:
            # Copier les données
            df_work = df.copy()
            
            # Appliquer les filtres
            filters_applied = []
            if selected_city is not None:
                df_work = df_work[df_work['city'] == selected_city]
                filters_applied.append(f"Ville: {selected_city}")
            if selected_property is not None:
                df_work = df_work[df_work['property_type'] == selected_property]
                filters_applied.append(f"Type: {selected_property}")
            if selected_transaction is not None:
                df_work = df_work[df_work['transaction'] == selected_transaction]
                filters_applied.append(f"Transaction: {selected_transaction}")
            
            st.info(f"🔍 Filtres appliqués: {', '.join(filters_applied) if filters_applied else 'Aucun'}")
            
            # Vérifier qu'on a assez de données
            if len(df_work) < 10:
                st.error(f"❌ Pas assez de données après filtrage ({len(df_work)} observations). Minimum requis: 10")
                return None, None
            
            # Supprimer les lignes avec prix manquant
            df_work = df_work.dropna(subset=['price'])
            
            if len(df_work) < 10:
                st.error(f"❌ Pas assez de données avec prix valides ({len(df_work)} observations). Minimum requis: 10")
                return None, None
            
            # Préparer les données pour la régression
            df_regression = prepare_data_for_regression(df_work)
            
            # Vérifier les valeurs manquantes dans les caractéristiques
            numeric_cols = df_regression.select_dtypes(include=['number']).columns
            features_cols = [col for col in numeric_cols if col != 'price']
            
            # Afficher les statistiques de préparation
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Observations après filtrage", len(df_work))
            with col2:
                st.metric("Observations avec prix valides", len(df_regression))
            with col3:
                st.metric("Caractéristiques disponibles", len(features_cols))
            
            # Traiter les valeurs manquantes dans les caractéristiques
            missing_in_features = df_regression[features_cols].isna().sum()
            if missing_in_features.sum() > 0:
                # st.warning("⚠️ Valeurs manquantes détectées dans les caractéristiques. Imputation en cours...")
                
                # Imputation simple
                from sklearn.impute import SimpleImputer
                imputer = SimpleImputer(strategy='median')
                df_regression[features_cols] = imputer.fit_transform(df_regression[features_cols])
                
                # st.success("✅ Imputation des valeurs manquantes terminée.")
            
            return df_regression, filters_applied
            
        except Exception as e:
            st.error(f"❌ Erreur lors de la préparation des données: {e}")
            return None, None
    
    # ============================================
    # SECTION 5: EXÉCUTION DES MODÈLES
    # ============================================
    
    if st.button("🚀 Entraîner le Modèle", type="primary"):
        with st.spinner("🔄 Préparation des données..."):
            df_regression, filters_applied = prepare_data_safely(
                df, selected_city, selected_property, selected_transaction
            )
        
        if df_regression is None:
            st.stop()
        
        try:
            st.success(f"✅ Données préparées: {len(df_regression)} observations prêtes pour l'entraînement")
            
            # ============================================
            # EXÉCUTION SELON L'ALGORITHME SÉLECTIONNÉ
            # ============================================
            
            if algorithm == "Régression Linéaire":
                st.subheader("📈 Résultats - Régression Linéaire")
                
                with st.spinner("🔄 Entraînement de la régression linéaire..."):
                    try:
                        model, importance, metrics = regression_par_segment(
                            df_regression,
                            city=selected_city,
                            property_type=selected_property,
                            transaction=selected_transaction,
                            target_column='price'
                        )
                        
                        # Afficher les métriques
                        display_regression_metrics(metrics, "Régression Linéaire")
                        
                        # NOUVEAU: Affichage détaillé des coefficients
                        display_linear_regression_coefficients(importance, model if hasattr(model, 'intercept_') else None)
                        
                        # Graphique d'importance des caractéristiques
                        display_feature_importance(importance, "Régression Linéaire", "Coefficient")
                        
                        # Capturer et afficher les graphiques matplotlib
                        st.pyplot(plt.gcf())
                        plt.close()
                        
                        # Enlever la variable cible ET les variables catégorielles non désirées
                        
                    except Exception as e:
                        st.error(f"❌ Erreur lors de la régression linéaire: {e}")
                            
            elif algorithm == "Random Forest Classification Prix":
                st.subheader("🌲 Classification Random Forest - Estimation des Prix")
                
                # Options spécifiques à la classification Random Forest
                with st.expander("⚙️ Options de Classification Random Forest", expanded=True):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        optimize_params_rf = st.checkbox("Optimiser les hyperparamètres", value=False, 
                                                        help="Recherche automatique des meilleurs paramètres",
                                                        key="rf_class_optimize")
                        test_size_rf = st.slider("Taille ensemble test (%)", 10, 40, 20, 
                                                key="rf_class_test_size") / 100
                    
                    with col2:
                        n_estimators_rf = st.slider("Nombre d'arbres", 50, 500, 200,
                                                key="rf_class_n_estimators")
                        max_depth_rf = st.slider("Profondeur max", 3, 20, 10,
                                                key="rf_class_max_depth")
                        threshold_low_rf = st.slider("Seuil sous-estimation", 0.5, 0.9, 0.75, 0.05,
                                                    key="rf_class_threshold_low")
                        threshold_high_rf = st.slider("Seuil surestimation", 1.1, 1.5, 1.25, 0.05,
                                                    key="rf_class_threshold_high")
                
                with st.spinner("🔄 Création des catégories de prix..."):
                    try:
                        # 1. Créer les catégories de prix
                        st.info("📊 Étape 1: Création des catégories de prix basées sur le marché local")
                        
                        # Modifier la fonction create_price_category pour utiliser les seuils personnalisés
                        def create_price_category_custom(df, grouping_columns=['city', 'property_type', 'transaction'], 
                                                    low_threshold=threshold_low_rf, high_threshold=threshold_high_rf):
                            df_category = df.copy()
                            df_category['price_per_sqm'] = df_category['price'] / df_category['size']
                            
                            # Filtrer les valeurs aberrantes
                            price_per_sqm_median = df_category['price_per_sqm'].median()
                            price_per_sqm_std = df_category['price_per_sqm'].std()
                            lower_bound = price_per_sqm_median - 3 * price_per_sqm_std
                            upper_bound = price_per_sqm_median + 3 * price_per_sqm_std
                            valid_mask = (df_category['price_per_sqm'] >= lower_bound) & (df_category['price_per_sqm'] <= upper_bound)
                            df_category = df_category[valid_mask]
                            
                            # Calculer moyennes par marché local
                            market_avg_price_per_sqm = df_category.groupby(grouping_columns)['price_per_sqm'].mean().reset_index()
                            market_avg_price_per_sqm.columns = list(grouping_columns) + ['market_avg_price_per_sqm']
                            df_category = df_category.merge(market_avg_price_per_sqm, on=grouping_columns, how='left')
                            df_category['price_ratio'] = df_category['price_per_sqm'] / df_category['market_avg_price_per_sqm']
                            
                            # Catégorisation binaire avec seuils personnalisés
                            def categorize_price(ratio):
                                if pd.isna(ratio):
                                    return 1
                                elif ratio < low_threshold or ratio > high_threshold:
                                    return 0  # Mal estimé
                                else:
                                    return 1  # Bien estimé
                            
                            df_category['price_category'] = df_category['price_ratio'].apply(categorize_price)
                            category_labels = {0: 'Mal estimé', 1: 'Bien estimé'}
                            df_category['price_category_label'] = df_category['price_category'].map(category_labels)
                            
                            return df_category
                        
                        df_with_categories = create_price_category_custom(df_regression)
                        
                        # Afficher les statistiques des catégories
                        category_stats = df_with_categories['price_category'].value_counts().sort_index()
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Observations avec catégories", len(df_with_categories))
                        with col2:
                            mal_estime = category_stats.get(0, 0)
                            st.metric("Mal estimé", f"{mal_estime} ({mal_estime/len(df_with_categories)*100:.1f}%)")
                        with col3:
                            bien_estime = category_stats.get(1, 0)
                            st.metric("Bien estimé", f"{bien_estime} ({bien_estime/len(df_with_categories)*100:.1f}%)")
                        
                        st.success("✅ Catégories de prix créées avec succès")
                        
                        # 2. Classification Random Forest
                        st.info("🌲 Étape 2: Classification avec Random Forest")
                        
                        model, results, feature_importance = random_forest_price_classification(
                            df_with_categories,
                            city=selected_city,
                            property_type=selected_property,
                            transaction=selected_transaction,
                            test_size=test_size_rf,
                            optimize_params=optimize_params_rf,
                            n_estimators=n_estimators_rf,
                            max_depth=max_depth_rf
                        )
                        
                        # 3. Afficher les résultats
                        st.subheader("📊 Résultats de la Classification Random Forest")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("🎯 Précision (Test)", f"{results['test_accuracy']:.3f}")
                        with col2:
                            st.metric("📈 Précision (Train)", f"{results['train_accuracy']:.3f}")
                        with col3:
                            n_test = len(results['y_test'])
                            st.metric("🔢 Échantillon test", n_test)
                        with col4:
                            overfitting = results['train_accuracy'] - results['test_accuracy']
                            if overfitting > 0.1:
                                st.metric("⚠️ Surapprentissage", f"+{overfitting:.3f}", delta_color="off")
                            else:
                                st.metric("✅ Généralisation", f"{overfitting:.3f}")
                        
                        # 4. Matrice de confusion
                        st.subheader("🔍 Matrice de Confusion")
                        
                        cm = results['confusion_matrix']
                        class_names = results['class_names']
                        
                        # Créer une heatmap avec plotly
                        fig_cm = px.imshow(
                            cm,
                            x=class_names,
                            y=class_names,
                            color_continuous_scale='Greens',  # Vert pour Random Forest
                            text_auto=True,
                            title="Matrice de Confusion - Random Forest",
                            labels=dict(x="Prédiction", y="Réalité")
                        )
                        fig_cm.update_layout(width=400, height=400)
                        
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            st.plotly_chart(fig_cm, use_container_width=True)
                        
                        with col2:
                            st.write("**Interprétation de la matrice :**")
                            
                            # Calculs détaillés
                            tn, fp, fn, tp = cm.ravel()
                            precision_0 = tn / (tn + fn) if (tn + fn) > 0 else 0
                            precision_1 = tp / (tp + fp) if (tp + fp) > 0 else 0
                            recall_0 = tn / (tn + fp) if (tn + fp) > 0 else 0
                            recall_1 = tp / (tp + fn) if (tp + fn) > 0 else 0
                            
                            st.write(f"• **Vrais Positifs (Bien classé comme Bien):** {tp}")
                            st.write(f"• **Vrais Négatifs (Mal classé comme Mal):** {tn}")
                            st.write(f"• **Faux Positifs (Mal classé comme Bien):** {fp}")
                            st.write(f"• **Faux Négatifs (Bien classé comme Mal):** {fn}")
                            
                            st.write("**Métriques par classe :**")
                            st.write(f"• **Précision 'Mal estimé':** {precision_0:.3f}")
                            st.write(f"• **Précision 'Bien estimé':** {precision_1:.3f}")
                            st.write(f"• **Rappel 'Mal estimé':** {recall_0:.3f}")
                            st.write(f"• **Rappel 'Bien estimé':** {recall_1:.3f}")
                        
                        # 5. Importance des caractéristiques
                        st.subheader("📈 Importance des Caractéristiques - Random Forest")
                        
                        # Graphique d'importance
                        top_features = feature_importance.head(10)
                        fig_importance = px.bar(
                            top_features,
                            x='Importance',
                            y='Caractéristique',
                            orientation='h',
                            title="Top 10 des caractéristiques les plus importantes (Random Forest)",
                            color='Importance',
                            color_continuous_scale='Greens'
                        )
                        fig_importance.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
                        st.plotly_chart(fig_importance, use_container_width=True)
                        
                        # Tableau détaillé
                        with st.expander("📋 Tableau complet des caractéristiques"):
                            st.dataframe(feature_importance, use_container_width=True)
                        
                        # 6. Rapport de classification détaillé
                        st.subheader("📋 Rapport de Classification Détaillé")
                        
                        # Convertir le rapport en DataFrame pour un meilleur affichage
                        report_dict = results['classification_report']
                        report_df = pd.DataFrame(report_dict).transpose()
                        
                        # Formater les valeurs numériques
                        for col in ['precision', 'recall', 'f1-score']:
                            if col in report_df.columns:
                                report_df[col] = report_df[col].apply(lambda x: f"{x:.3f}" if isinstance(x, (int, float)) else x)
                        
                        st.dataframe(report_df, use_container_width=True)
                        
                        # 7. Analyse des erreurs spécifique à Random Forest
                        with st.spinner("Analyse des erreurs en cours..."):
                            try:
                                # Utiliser la version sécurisée
                                error_data, error_types = analyze_misclassified_properties_rf_safe(
                                    df_with_categories, results, model, feature_importance
                                )
                                
                                if len(error_data) > 0:
                                    st.subheader("❌ Analyse des Erreurs - Random Forest")
                                    
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        st.metric("Total d'erreurs", len(error_data))
                                        st.metric("Taux d'erreur", f"{len(error_data)/len(results['y_test'])*100:.1f}%")
                                    
                                    with col2:
                                        # Types d'erreurs
                                        if len(error_types) > 0:
                                            for _, row in error_types.iterrows():
                                                st.write(f"• {row['actual_label']} → {row['predicted_label']}: {row['count']} cas")
                                    
                                    # Exemples d'erreurs avec probabilités Random Forest
                                    st.write("**Exemples de propriétés mal classifiées (avec probabilités) :**")
                                    
                                    if len(error_data) > 0:
                                        # Afficher le DataFrame des erreurs
                                        display_cols = ['actual_label', 'predicted_label', 'prob_mal_estime', 'prob_bien_estime', 'confiance']
                                        st.dataframe(error_data[display_cols].round(3), use_container_width=True)
                                        
                                        # Analyse de confiance
                                        st.write("**Analyse de confiance des erreurs :**")
                                        low_confidence = (error_data['confiance'] < 0.6).sum()
                                        high_confidence = (error_data['confiance'] > 0.8).sum()
                                        
                                        col1, col2, col3 = st.columns(3)
                                        with col1:
                                            st.metric("Erreurs peu confiantes (<60%)", low_confidence)
                                        with col2:
                                            st.metric("Erreurs très confiantes (>80%)", high_confidence)
                                        with col3:
                                            avg_confidence = error_data['confiance'].mean()
                                            st.metric("Confiance moyenne", f"{avg_confidence:.3f}")
                                        
                                        # Graphique de distribution des confiances
                                        fig_conf = px.histogram(
                                            error_data, 
                                            x='confiance', 
                                            nbins=10,
                                            title="Distribution des niveaux de confiance des erreurs",
                                            labels={'confiance': 'Niveau de confiance', 'count': 'Nombre d\'erreurs'}
                                        )
                                        st.plotly_chart(fig_conf, use_container_width=True)
                                else:
                                    st.success("🎉 Aucune erreur de classification ! Modèle parfait.")
                                    
                            except Exception as e:
                                st.warning(f"⚠️ Impossible d'analyser les erreurs en détail: {e}")
                                st.info("Le modèle fonctionne correctement, mais l'analyse détaillée des erreurs n'est pas disponible.")
                                
                                # Afficher quand même les métriques de base
                                total_errors = (results['y_test'] != results['y_test_pred']).sum()
                                error_rate = total_errors / len(results['y_test']) * 100
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("Total d'erreurs (estimation)", total_errors)
                                with col2:
                                    st.metric("Taux d'erreur (estimation)", f"{error_rate:.1f}%")
                        # 8. Spécificités Random Forest
                        st.subheader("🌲 Spécificités Random Forest")
                        
                        # Information sur les arbres
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Nombre d'arbres", model.n_estimators)
                        with col2:
                            st.metric("Profondeur max", model.max_depth if model.max_depth else "Illimitée")
                        with col3:
                            # Calcul de la diversité des arbres (estimation)
                            oob_score = getattr(model, 'oob_score_', None)
                            if oob_score:
                                st.metric("Score OOB", f"{oob_score:.3f}")
                            else:
                                st.metric("Features par arbre", f"{model.max_features}")
                        
                        # 9. Recommandations spécifiques Random Forest
                        st.subheader("💡 Recommandations - Random Forest")
                        
                        accuracy = results['test_accuracy']
                        if accuracy > 0.9:
                            st.success("🌟 **Excellent modèle Random Forest** : Très haute précision, excellent pour la production")
                            st.info("🎯 Random Forest excelle avec cette complexité de données")
                        elif accuracy > 0.8:
                            st.success("✅ **Bon modèle Random Forest** : Précision satisfaisante, robuste aux outliers")
                            st.info("🌲 La nature d'ensemble de Random Forest apporte de la stabilité")
                        elif accuracy > 0.7:
                            st.warning("⚠️ **Modèle Random Forest acceptable** : Pourrait bénéficier de plus d'arbres ou de données")
                        elif accuracy > 0.6:
                            st.warning("🔄 **Random Forest à améliorer** : Augmenter n_estimators ou optimiser les paramètres")
                        else:
                            st.error("❌ **Random Forest insuffisant** : Revoir la sélection des features ou les seuils")
                        
                        # Conseils d'amélioration spécifiques Random Forest
                        st.write("**Conseils d'amélioration Random Forest :**")
                        st.write("• Augmenter le nombre d'arbres (n_estimators) pour plus de stabilité")
                        st.write("• Ajuster max_depth pour contrôler le surapprentissage")
                        st.write("• Utiliser max_features='sqrt' pour plus de diversité entre arbres")
                        st.write("• Considérer min_samples_split et min_samples_leaf pour la régularisation")
                        st.write("• Random Forest est naturellement robuste aux valeurs aberrantes")
                        
                        # Comparaison avec XGBoost
                        st.info("🔄 **Comparaison avec XGBoost** : Random Forest est généralement plus stable et moins sensible aux hyperparamètres, tandis que XGBoost peut atteindre une précision légèrement supérieure avec un bon tuning.")
                        
                    except Exception as e:
                        st.error(f"❌ Erreur lors de la classification Random Forest: {e}")
                        st.info("💡 Vérifiez que vos données contiennent les colonnes nécessaires (price, size, city, property_type, transaction)")
                        
                        # Debug info pour Random Forest
                        if 'df_regression' in locals():
                            st.write("**Colonnes disponibles dans df_regression:**")
                            st.write(list(df_regression.columns))
                        
                        # Suggestions spécifiques
                        st.write("**Suggestions de débogage Random Forest :**")
                        st.write("• Vérifiez que le dataset contient assez d'observations (>100 recommandé)")
                        st.write("• Assurez-vous que les seuils ne créent pas de classes déséquilibrées")
                        st.write("• Random Forest nécessite des variables numériques bien encodées")
            elif algorithm == "XGBoost Classification Prix":
                st.subheader("🎯 Classification XGBoost - Estimation des Prix")
                
               
                with st.spinner("🔄 Création des catégories de prix..."):
                    try:
                        # 1. Créer les catégories de prix
                        st.info("📊 Étape 1: Création des catégories de prix basées sur le marché local")
                        
                        # Modifier la fonction create_price_category pour utiliser les seuils personnalisés
                        def create_price_category_custom(df, grouping_columns=['city', 'property_type', 'transaction'], 
                                                    low_threshold=threshold_low, high_threshold=threshold_high):
                            df_category = df.copy()
                            df_category['price_per_sqm'] = df_category['price'] / df_category['size']
                            
                            # Filtrer les valeurs aberrantes
                            price_per_sqm_median = df_category['price_per_sqm'].median()
                            price_per_sqm_std = df_category['price_per_sqm'].std()
                            lower_bound = price_per_sqm_median - 3 * price_per_sqm_std
                            upper_bound = price_per_sqm_median + 3 * price_per_sqm_std
                            valid_mask = (df_category['price_per_sqm'] >= lower_bound) & (df_category['price_per_sqm'] <= upper_bound)
                            df_category = df_category[valid_mask]
                            
                            # Calculer moyennes par marché local
                            market_avg_price_per_sqm = df_category.groupby(grouping_columns)['price_per_sqm'].mean().reset_index()
                            market_avg_price_per_sqm.columns = list(grouping_columns) + ['market_avg_price_per_sqm']
                            df_category = df_category.merge(market_avg_price_per_sqm, on=grouping_columns, how='left')
                            df_category['price_ratio'] = df_category['price_per_sqm'] / df_category['market_avg_price_per_sqm']
                            
                            # Catégorisation binaire avec seuils personnalisés
                            def categorize_price(ratio):
                                if pd.isna(ratio):
                                    return 1
                                elif ratio < low_threshold or ratio > high_threshold:
                                    return 0  # Mal estimé
                                else:
                                    return 1  # Bien estimé
                            
                            df_category['price_category'] = df_category['price_ratio'].apply(categorize_price)
                            category_labels = {0: 'Mal estimé', 1: 'Bien estimé'}
                            df_category['price_category_label'] = df_category['price_category'].map(category_labels)
                            
                            return df_category
                        
                        df_with_categories = create_price_category_custom(df_regression)
                        
                        # Afficher les statistiques des catégories
                        category_stats = df_with_categories['price_category'].value_counts().sort_index()
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Observations avec catégories", len(df_with_categories))
                        with col2:
                            mal_estime = category_stats.get(0, 0)
                            st.metric("Mal estimé", f"{mal_estime} ({mal_estime/len(df_with_categories)*100:.1f}%)")
                        with col3:
                            bien_estime = category_stats.get(1, 0)
                            st.metric("Bien estimé", f"{bien_estime} ({bien_estime/len(df_with_categories)*100:.1f}%)")
                        
                        st.success("✅ Catégories de prix créées avec succès")
                        
                        # 2. Classification XGBoost
                        st.info("🤖 Étape 2: Classification avec XGBoost")
                        
                        model, results, feature_importance = xgboost_price_classification(
                            df_with_categories,
                            city=selected_city,
                            property_type=selected_property,
                            transaction=selected_transaction,
                            test_size=test_size,
                            optimize_params=optimize_params
                        )
                        
                        # 3. Afficher les résultats
                        st.subheader("📊 Résultats de la Classification")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("🎯 Précision (Test)", f"{results['test_accuracy']:.3f}")
                        with col2:
                            st.metric("📈 Précision (Train)", f"{results['train_accuracy']:.3f}")
                        with col3:
                            n_test = len(results['y_test'])
                            st.metric("🔢 Échantillon test", n_test)
                        with col4:
                            overfitting = results['train_accuracy'] - results['test_accuracy']
                            if overfitting > 0.1:
                                st.metric("⚠️ Surapprentissage", f"+{overfitting:.3f}", delta_color="off")
                            else:
                                st.metric("✅ Généralisation", f"{overfitting:.3f}")
                        
                        # 4. Matrice de confusion
                        st.subheader("🔍 Matrice de Confusion")
                        
                        cm = results['confusion_matrix']
                        class_names = results['class_names']
                        
                        # Créer une heatmap avec plotly
                        fig_cm = px.imshow(
                            cm,
                            x=class_names,
                            y=class_names,
                            color_continuous_scale='Blues',
                            text_auto=True,
                            title="Matrice de Confusion",
                            labels=dict(x="Prédiction", y="Réalité")
                        )
                        fig_cm.update_layout(width=400, height=400)
                        
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            st.plotly_chart(fig_cm, use_container_width=True)
                        
                        with col2:
                            st.write("**Interprétation de la matrice :**")
                            
                            # Calculs détaillés
                            tn, fp, fn, tp = cm.ravel()
                            precision_0 = tn / (tn + fn) if (tn + fn) > 0 else 0
                            precision_1 = tp / (tp + fp) if (tp + fp) > 0 else 0
                            recall_0 = tn / (tn + fp) if (tn + fp) > 0 else 0
                            recall_1 = tp / (tp + fn) if (tp + fn) > 0 else 0
                            
                            st.write(f"• **Vrais Positifs (Bien classé comme Bien):** {tp}")
                            st.write(f"• **Vrais Négatifs (Mal classé comme Mal):** {tn}")
                            st.write(f"• **Faux Positifs (Mal classé comme Bien):** {fp}")
                            st.write(f"• **Faux Négatifs (Bien classé comme Mal):** {fn}")
                            
                            st.write("**Métriques par classe :**")
                            st.write(f"• **Précision 'Mal estimé':** {precision_0:.3f}")
                            st.write(f"• **Précision 'Bien estimé':** {precision_1:.3f}")
                            st.write(f"• **Rappel 'Mal estimé':** {recall_0:.3f}")
                            st.write(f"• **Rappel 'Bien estimé':** {recall_1:.3f}")
                        
                        # 5. Importance des caractéristiques
                        st.subheader("📈 Importance des Caractéristiques")
                        
                        # Graphique d'importance
                        top_features = feature_importance.head(10)
                        fig_importance = px.bar(
                            top_features,
                            x='Importance',
                            y='Caractéristique',
                            orientation='h',
                            title="Top 10 des caractéristiques les plus importantes",
                            color='Importance',
                            color_continuous_scale='Viridis'
                        )
                        fig_importance.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
                        st.plotly_chart(fig_importance, use_container_width=True)
                        
                        # Tableau détaillé
                        with st.expander("📋 Tableau complet des caractéristiques"):
                            st.dataframe(feature_importance, use_container_width=True)
                        
                        # 6. Rapport de classification détaillé
                        st.subheader("📋 Rapport de Classification Détaillé")
                        
                        # Convertir le rapport en DataFrame pour un meilleur affichage
                        report_dict = results['classification_report']
                        report_df = pd.DataFrame(report_dict).transpose()
                        
                        # Formater les valeurs numériques
                        for col in ['precision', 'recall', 'f1-score']:
                            if col in report_df.columns:
                                report_df[col] = report_df[col].apply(lambda x: f"{x:.3f}" if isinstance(x, (int, float)) else x)
                        
                        st.dataframe(report_df, use_container_width=True)
                        
                        # 7. Analyse des erreurs
                        # if st.button("🔍 Analyser les Erreurs de Classification", key="xgb_class_analyze_errors"):
                        with st.spinner("Analyse des erreurs en cours..."):
                            error_data, error_types = analyze_misclassified_properties(
                                df_with_categories, results, model, feature_importance
                            )
                            
                            if len(error_data) > 0:
                                st.subheader("❌ Analyse des Erreurs")
                                
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.metric("Total d'erreurs", len(error_data))
                                    st.metric("Taux d'erreur", f"{len(error_data)/len(results['y_test'])*100:.1f}%")
                                
                                with col2:
                                    # Types d'erreurs
                                    for _, row in error_types.iterrows():
                                        st.write(f"• {row['actual_label']} → {row['predicted_label']}: {row['count']} cas")
                                
                                # Exemples d'erreurs
                                st.write("**Exemples de propriétés mal classifiées :**")
                                examples = error_data.head(5)[['price', 'size', 'price_ratio', 'market_avg_price_per_sqm', 
                                                            'actual_label', 'predicted_label']]
                                st.dataframe(examples, use_container_width=True)
                            else:
                                st.success("🎉 Aucune erreur de classification ! Modèle parfait.")
                        
                        # 8. Recommandations
                        st.subheader("💡 Recommandations")
                        
                        accuracy = results['test_accuracy']
                        if accuracy > 0.9:
                            st.success("🌟 **Excellent modèle** : Très haute précision, déployable en production")
                        elif accuracy > 0.8:
                            st.success("✅ **Bon modèle** : Précision satisfaisante, peut être utilisé avec confiance")
                        elif accuracy > 0.7:
                            st.warning("⚠️ **Modèle acceptable** : Précision correcte, mais des améliorations sont possibles")
                        elif accuracy > 0.6:
                            st.warning("🔄 **Modèle à améliorer** : Précision faible, revoir les données ou les paramètres")
                        else:
                            st.error("❌ **Modèle insuffisant** : Précision très faible, revoir complètement l'approche")
                        
                        # Conseils d'amélioration
                        st.write("**Conseils d'amélioration :**")
                        st.write("• Ajuster les seuils de catégorisation selon votre connaissance du marché")
                        st.write("• Ajouter plus de données pour les segments sous-représentés")
                        st.write("• Considérer des variables supplémentaires (localisation précise, équipements...)")
                        st.write("• Essayer l'optimisation des hyperparamètres si non activée")
                        
                    except Exception as e:
                        st.error(f"❌ Erreur lors de la classification: {e}")
                        st.info("💡 Vérifiez que vos données contiennent les colonnes nécessaires (price, size, city, property_type, transaction)")
            # else:  # Comparaison des 3 modèles
            #     st.subheader("🔄 Comparaison des 3 Modèles")
                
            #     results = {}
            #     errors = {}
                
            #     # Régression Linéaire
            #     with st.spinner("🔄 Test Régression Linéaire..."):
            #         try:
            #             model_lr, importance_lr, metrics_lr = regression_par_segment(
            #                 df_regression, selected_city, selected_property, selected_transaction
            #             )
            #             results['Régression Linéaire'] = {
            #                 'model': model_lr,
            #                 'importance': importance_lr,
            #                 'metrics': metrics_lr,
            #                 'r2': metrics_lr['test_r2'],
            #                 'rmse': metrics_lr['test_rmse'],
            #                 'mae': metrics_lr['test_mae']
            #             }
            #             plt.close()  # Fermer les graphiques matplotlib
            #         except Exception as e:
            #             errors['Régression Linéaire'] = str(e)
                
            #     # Random Forest
            #     with st.spinner("🔄 Test Random Forest..."):
            #         try:
            #             model_rf, importance_rf, metrics_rf = random_forest_par_segment(
            #                 df_regression, selected_city, selected_property, selected_transaction,
            #                 n_estimators=n_estimators if 'n_estimators' in locals() else 100,
            #                 max_depth=max_depth_rf if 'max_depth_rf' in locals() else None
            #             )
            #             results['Random Forest'] = {
            #                 'model': model_rf,
            #                 'importance': importance_rf,
            #                 'metrics': metrics_rf,
            #                 'r2': metrics_rf['test_r2'],
            #                 'rmse': metrics_rf['test_rmse'],
            #                 'mae': metrics_rf['test_mae']
            #             }
            #             plt.close()  # Fermer les graphiques matplotlib
            #         except Exception as e:
            #             errors['Random Forest'] = str(e)
                
            #     # XGBoost
            #     with st.spinner("🔄 Test XGBoost..."):
            #         try:
            #             model_xgb, importance_xgb, r2_xgb = xgboost_simple(
            #                 df_regression, selected_city, selected_property, selected_transaction
            #             )
            #             results['XGBoost'] = {
            #                 'model': model_xgb,
            #                 'importance': importance_xgb,
            #                 'r2': r2_xgb,
            #                 'rmse': 'N/A',  # XGBoost simple ne retourne que R2
            #                 'mae': 'N/A'
            #             }
            #             plt.close()  # Fermer les graphiques matplotlib
            #         except Exception as e:
            #             errors['XGBoost'] = str(e)
                
            #     # Afficher les erreurs s'il y en a
            #     if errors:
            #         st.warning("⚠️ Certains modèles ont échoué:")
            #         for model_name, error in errors.items():
            #             st.error(f"❌ {model_name}: {error}")
                
            #     # Afficher la comparaison si on a au moins un résultat
            #     if results:
            #         display_model_comparison(results)
            #     else:
            #         st.error("❌ Aucun modèle n'a pu être entraîné avec succès.")
        
        except Exception as e:
            st.error(f"❌ Erreur générale lors de l'entraînement: {e}")
            st.info("💡 Vérifiez la qualité de vos données et réessayez avec des filtres différents.")

def display_regression_metrics(metrics, model_name):
    """Afficher les métriques de régression de manière claire"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 R² Test", f"{metrics['test_r2']:.4f}")
    with col2:
        st.metric("📈 R² Train", f"{metrics['train_r2']:.4f}")
    with col3:
        st.metric("📏 RMSE", f"{metrics['test_rmse']:.0f}")
    with col4:
        st.metric("📐 MAE", f"{metrics['test_mae']:.0f}")
    
    # Évaluation qualitative
    performance = get_performance_label(metrics['test_r2'])
    overfitting = check_overfitting(metrics['train_r2'], metrics['test_r2'])
    
    col1, col2 = st.columns(2)
    with col1:
        if metrics['test_r2'] > 0.7:
            st.success(f"✅ {performance}")
        elif metrics['test_r2'] > 0.5:
            st.info(f"👍 {performance}")
        elif metrics['test_r2'] > 0.3:
            st.warning(f"⚠️ {performance}")
        else:
            st.error(f"❌ {performance}")
    
    with col2:
        if overfitting:
            st.warning("⚠️ Surapprentissage détecté")
        else:
            st.success("✅ Pas de surapprentissage")

def display_feature_importance(importance_df, model_name, value_col):
    """Afficher l'importance des caractéristiques avec Plotly"""
    st.subheader(f"📊 Importance des Caractéristiques - {model_name}")
    
    # Prendre les 10 plus importantes
    top_features = importance_df.head(10).copy()
    
    # Créer le graphique Plotly
    if value_col == "Coefficient":
        # Pour la régression linéaire, utiliser une échelle de couleur divergente
        fig = px.bar(
            top_features,
            x=value_col,
            y='Caractéristique',
            orientation='h',
            title=f"Top 10 des caractéristiques - {model_name}",
            color=value_col,
            color_continuous_scale='RdBu_r',
            color_continuous_midpoint=0
        )
    else:
        # Pour les autres modèles, utiliser une échelle normale
        fig = px.bar(
            top_features,
            x=value_col,
            y='Caractéristique',
            orientation='h',
            title=f"Top 10 des caractéristiques - {model_name}",
            color=value_col,
            color_continuous_scale='Viridis'
        )
    
    fig.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True)
    
    # Tableau détaillé
    with st.expander("📋 Détail de toutes les caractéristiques"):
        st.dataframe(importance_df, use_container_width=True)

def display_model_comparison(results):
    """Afficher la comparaison des modèles"""
    st.subheader("🏆 Comparaison des Modèles")
    
    # Créer le tableau de comparaison
    comparison_data = []
    for model_name, result in results.items():
        comparison_data.append({
            'Modèle': model_name,
            'R² Score': f"{result['r2']:.4f}",
            'RMSE': f"{result['rmse']:.0f}" if result['rmse'] != 'N/A' else 'N/A',
            'MAE': f"{result['mae']:.0f}" if result['mae'] != 'N/A' else 'N/A',
            'Performance': get_performance_label(result['r2'])
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True)
    
    # Identifier le meilleur modèle
    best_model = max(results.items(), key=lambda x: x[1]['r2'])
    st.success(f"🏆 **Meilleur modèle:** {best_model[0]} (R² = {best_model[1]['r2']:.4f})")
    
    # Graphique de comparaison
    r2_scores = [result['r2'] for result in results.values()]
    model_names = list(results.keys())
    
    fig = px.bar(
        x=model_names,
        y=r2_scores,
        title="Comparaison des scores R²",
        labels={'x': 'Modèle', 'y': 'Score R²'},
        color=r2_scores,
        color_continuous_scale='Viridis'
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Recommandations
    st.subheader("💡 Recommandations")
    
    best_r2 = best_model[1]['r2']
    if best_r2 > 0.8:
        st.success("✅ Excellente performance ! Le modèle est très fiable pour la prédiction.")
    elif best_r2 > 0.6:
        st.info("👍 Bonne performance. Le modèle peut être utilisé avec confiance.")
    elif best_r2 > 0.4:
        st.warning("⚠️ Performance modérée. Considérez l'ajout de plus de données ou de caractéristiques.")
    else:
        st.error("❌ Performance faible. Revoyez la sélection des caractéristiques ou la qualité des données.")

def display_linear_regression_coefficients(importance_df, model=None):
    """Afficher les coefficients de la régression linéaire de manière détaillée"""
    st.subheader("🔢 Coefficients de la Régression Linéaire")
    
    # Trier par valeur absolue décroissante pour voir les plus importants
    coeffs_sorted = importance_df.copy()
    coeffs_sorted['Coefficient_Abs'] = coeffs_sorted['Coefficient'].abs()
    coeffs_sorted = coeffs_sorted.sort_values('Coefficient_Abs', ascending=False)
    
    # Affichage en colonnes
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.write("### 📊 Équation de Régression")
        
        # Construire l'équation
        equation_parts = []
        for idx, row in coeffs_sorted.head(10).iterrows():  # Top 10 pour lisibilité
            coeff = row['Coefficient']
            feature = row['Caractéristique']
            
            if coeff > 0:
                sign = "+" if len(equation_parts) > 0 else ""
                equation_parts.append(f"{sign} {coeff:.3f} × {feature}")
            else:
                equation_parts.append(f"- {abs(coeff):.3f} × {feature}")
        
        # Afficher l'équation
        if equation_parts:
            equation = "**Prix** = " + " ".join(equation_parts[:5])  # Limiter à 5 termes
            if len(equation_parts) > 5:
                equation += " + ..."
            
            if model and hasattr(model, 'intercept_'):
                equation += f" + {model.intercept_:.2f}"
            
            st.markdown(equation)
        
        # Tableau détaillé des coefficients
        st.write("### 📋 Tableau Détaillé des Coefficients")
        
        # Créer un DataFrame enrichi pour l'affichage
        display_coeffs = coeffs_sorted.copy()
        display_coeffs['Impact'] = display_coeffs['Coefficient'].apply(get_coefficient_impact)
        display_coeffs['Coefficient_Formaté'] = display_coeffs['Coefficient'].apply(lambda x: f"{x:+.4f}")
        display_coeffs['Interprétation'] = display_coeffs.apply(get_coefficient_interpretation, axis=1)
        
        # Afficher le tableau
        st.dataframe(
            display_coeffs[['Caractéristique', 'Coefficient_Formaté', 'Impact', 'Interprétation']],
            use_container_width=True,
            hide_index=True
        )
    
    with col2:
        st.write("### 🎯 Analyse des Impacts")
        
        # Statistiques sur les coefficients
        positive_coeffs = coeffs_sorted[coeffs_sorted['Coefficient'] > 0]
        negative_coeffs = coeffs_sorted[coeffs_sorted['Coefficient'] < 0]
        
        st.metric("📈 Variables positives", len(positive_coeffs))
        st.metric("📉 Variables négatives", len(negative_coeffs))
        
        if len(positive_coeffs) > 0:
            max_positive = positive_coeffs.iloc[0]
            st.success(f"🔝 Plus fort impact positif:\n**{max_positive['Caractéristique']}**\n(+{max_positive['Coefficient']:.3f})")
        
        if len(negative_coeffs) > 0:
            max_negative = negative_coeffs.iloc[0]
            st.error(f"🔻 Plus fort impact négatif:\n**{max_negative['Caractéristique']}**\n({max_negative['Coefficient']:.3f})")
        
        # Guide d'interprétation
        st.write("### 💡 Guide d'Interprétation")
        st.info("""
        **Coefficient positif (+)** : 
        Augmente le prix
        
        **Coefficient négatif (-)** : 
        Diminue le prix
        
        **Valeur absolue** : 
        Force de l'impact
        """)
    
    # Graphique des coefficients avec interpretation
    st.write("### 📊 Visualisation des Coefficients")
    
    # Créer un graphique avec des couleurs selon l'impact
    top_coeffs = coeffs_sorted.head(15)  # Top 15 pour la visualisation
    
    colors = ['green' if x > 0 else 'red' for x in top_coeffs['Coefficient']]
    
    fig = px.bar(
        top_coeffs,
        x='Coefficient',
        y='Caractéristique',
        orientation='h',
        title="Impact des Caractéristiques sur le Prix (Top 15)",
        color='Coefficient',
        color_continuous_scale='RdBu_r',
        color_continuous_midpoint=0,
        hover_data={'Coefficient': ':.4f'}
    )
    
    # Ajouter une ligne verticale à zéro
    fig.add_vline(x=0, line_dash="dash", line_color="black", line_width=1)
    
    # Annotations pour clarification
    fig.add_annotation(
        x=top_coeffs['Coefficient'].max() * 0.7,
        y=len(top_coeffs) - 1,
        text="Augmente le prix",
        showarrow=False,
        font=dict(color="green", size=12)
    )
    
    if top_coeffs['Coefficient'].min() < 0:
        fig.add_annotation(
            x=top_coeffs['Coefficient'].min() * 0.7,
            y=len(top_coeffs) - 1,
            text="Diminue le prix",
            showarrow=False,
            font=dict(color="red", size=12)
        )
    
    fig.update_layout(
        height=600,
        yaxis={'categoryorder': 'total ascending'},
        xaxis_title="Coefficient de Régression",
        yaxis_title="Caractéristiques"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Analyse contextuelle pour l'immobilier
    st.write("### 🏠 Analyse Contextuelle Immobilier")
    
    # Identifier les variables immobilières classiques et leurs impacts
    real_estate_analysis = analyze_real_estate_coefficients(coeffs_sorted)
    
    if real_estate_analysis:
        for category, analysis in real_estate_analysis.items():
            with st.expander(f"📊 {category}", expanded=False):
                st.write(analysis)

def get_coefficient_impact(coeff):
    """Déterminer l'impact d'un coefficient"""
    abs_coeff = abs(coeff)
    if abs_coeff > 1000:
        return "🔥 Très Fort"
    elif abs_coeff > 500:
        return "⚡ Fort"
    elif abs_coeff > 100:
        return "📈 Modéré"
    elif abs_coeff > 10:
        return "💨 Faible"
    else:
        return "🔸 Très Faible"

def get_coefficient_interpretation(row):
    """Interpréter un coefficient dans le contexte immobilier"""
    coeff = row['Coefficient']
    feature = row['Caractéristique'].lower()
    
    if coeff > 0:
        direction = "augmente"
        emoji = "📈"
    else:
        direction = "diminue"
        emoji = "📉"
    
    # Interprétations spécifiques à l'immobilier
    if 'size' in feature or 'surface' in feature:
        return f"{emoji} Chaque m² supplémentaire {direction} le prix de {abs(coeff):.0f} TND"
    elif 'room' in feature or 'piece' in feature:
        return f"{emoji} Chaque pièce supplémentaire {direction} le prix de {abs(coeff):.0f} TND"
    elif 'age' in feature or 'ancien' in feature:
        return f"{emoji} Chaque année d'ancienneté {direction} le prix de {abs(coeff):.0f} TND"
    elif 'bathroom' in feature or 'salle' in feature:
        return f"{emoji} Chaque salle de bain supplémentaire {direction} le prix de {abs(coeff):.0f} TND"
    elif 'parking' in feature:
        return f"{emoji} Chaque place de parking {direction} le prix de {abs(coeff):.0f} TND"
    elif 'elevator' in feature or 'ascenseur' in feature:
        return f"{emoji} La présence d'un ascenseur {direction} le prix de {abs(coeff):.0f} TND"
    elif 'condition' in feature or 'etat' in feature:
        return f"{emoji} L'amélioration de l'état {direction} le prix de {abs(coeff):.0f} TND"
    elif 'finishing' in feature or 'finition' in feature:
        return f"{emoji} L'amélioration du standing {direction} le prix de {abs(coeff):.0f} TND"
    else:
        return f"{emoji} Cette caractéristique {direction} le prix de {abs(coeff):.0f} TND"

def analyze_real_estate_coefficients(coeffs_df):
    """Analyser les coefficients dans le contexte immobilier"""
    analysis = {}
    
    # Identifier les différentes catégories
    size_vars = coeffs_df[coeffs_df['Caractéristique'].str.contains('size|surface', case=False, na=False)]
    room_vars = coeffs_df[coeffs_df['Caractéristique'].str.contains('room|piece|bedroom|bathroom', case=False, na=False)]
    age_vars = coeffs_df[coeffs_df['Caractéristique'].str.contains('age|year|ancien', case=False, na=False)]
    amenity_vars = coeffs_df[coeffs_df['Caractéristique'].str.contains('elevator|parking|garden|pool|kitchen', case=False, na=False)]
    quality_vars = coeffs_df[coeffs_df['Caractéristique'].str.contains('condition|finishing|standing', case=False, na=False)]
    
    # Analyse de la superficie
    if not size_vars.empty:
        size_coeff = size_vars.iloc[0]['Coefficient']
        if size_coeff > 0:
            analysis['📐 Impact de la Superficie'] = f"""
            ✅ **Coefficient positif**: {size_coeff:.2f} TND/m²
            
            📊 **Interprétation**: Chaque mètre carré supplémentaire augmente le prix de {size_coeff:.0f} TND.
            
            💡 **Insight Business**: La superficie est un facteur valorisant, ce qui est normal sur le marché immobilier.
            """
        else:
            analysis['📐 Impact de la Superficie'] = f"""
            ⚠️ **Coefficient négatif**: {size_coeff:.2f} TND/m²
            
            🤔 **Attention**: Résultat contre-intuitif qui peut indiquer un problème dans les données ou une corrélation avec d'autres variables.
            """
    
    # Analyse des pièces
    if not room_vars.empty:
        room_analysis = "🏠 **Impact du Nombre de Pièces**:\n\n"
        for _, room_var in room_vars.iterrows():
            coeff = room_var['Coefficient']
            feature = room_var['Caractéristique']
            if coeff > 0:
                room_analysis += f"✅ {feature}: +{coeff:.0f} TND par pièce supplémentaire\n"
            else:
                room_analysis += f"⚠️ {feature}: {coeff:.0f} TND (impact négatif)\n"
        
        analysis['🏠 Configuration des Pièces'] = room_analysis
    
    # Analyse de l'âge
    if not age_vars.empty:
        age_coeff = age_vars.iloc[0]['Coefficient']
        if age_coeff < 0:
            analysis['⏰ Impact de l\'Âge'] = f"""
            ✅ **Dépréciation normale**: {age_coeff:.2f} TND/an
            
            📊 **Interprétation**: Chaque année d'ancienneté réduit le prix de {abs(age_coeff):.0f} TND.
            
            💡 **Insight**: Dépréciation annuelle de {abs(age_coeff):.0f} TND, soit {abs(age_coeff)*10:.0f} TND sur 10 ans.
            """
        else:
            analysis['⏰ Impact de l\'Âge'] = f"""
            🤔 **Coefficient positif**: {age_coeff:.2f} TND/an
            
            ⚠️ **Attention**: Résultat contre-intuitif. Possible effet "vintage" ou corrélation avec la localisation.
            """
    
    # Analyse des équipements
    if not amenity_vars.empty:
        amenity_analysis = "⚡ **Impact des Équipements**:\n\n"
        for _, amenity in amenity_vars.iterrows():
            coeff = amenity['Coefficient']
            feature = amenity['Caractéristique']
            if coeff > 0:
                amenity_analysis += f"✅ {feature}: +{coeff:.0f} TND\n"
            else:
                amenity_analysis += f"❌ {feature}: {coeff:.0f} TND\n"
        
        analysis['⚡ Équipements et Commodités'] = amenity_analysis
    
    # Analyse de la qualité
    if not quality_vars.empty:
        quality_analysis = "✨ **Impact de la Qualité**:\n\n"
        for _, quality in quality_vars.iterrows():
            coeff = quality['Coefficient']
            feature = quality['Caractéristique']
            quality_analysis += f"• {feature}: {coeff:+.0f} TND par niveau de qualité\n"
        
        analysis['✨ Qualité et Finitions'] = quality_analysis
    
    return analysis

def check_overfitting(train_r2, test_r2):
    """Vérifier s'il y a du surapprentissage"""
    return (train_r2 - test_r2) > 0.1

# ============================================
# SECTION D'AIDE POUR L'INTERPRÉTATION
# ============================================

def add_supervised_learning_help():
    """Section d'aide pour l'apprentissage supervisé"""
    with st.expander("💡 Guide d'Interprétation - Apprentissage Supervisé", expanded=False):
        st.markdown("""
        ## 📈 Métriques de Performance
        
        ### **R² (Coefficient de Détermination)**
        - **0.8-1.0** : Excellent modèle, prédictions très fiables
        - **0.6-0.8** : Bon modèle, prédictions fiables
        - **0.4-0.6** : Modèle modéré, prédictions acceptables
        - **0.2-0.4** : Modèle faible, prédictions peu fiables
        - **< 0.2** : Modèle très faible, à revoir complètement
        
        ### **RMSE (Root Mean Square Error)**
        - Erreur moyenne en TND
        - Plus faible = mieux
        - À comparer au prix moyen des biens
        
        ### **MAE (Mean Absolute Error)**
        - Erreur absolue moyenne en TND
        - Plus faible = mieux
        - Plus robuste aux valeurs aberrantes que RMSE
        
        ---
        
        ## 🤖 Algorithmes
        
        ### **📈 Régression Linéaire**
        **Avantages :**
        - Simple et interprétable
        - Rapide à entraîner
        - Coefficients indiquent l'impact de chaque variable
        
        **Inconvénients :**
        - Suppose des relations linéaires
        - Sensible aux valeurs aberrantes
        - Peut sous-performer sur des données complexes
        
        ### **🌲 Random Forest**
        **Avantages :**
        - Gère les relations non-linéaires
        - Robuste aux valeurs aberrantes
        - Fournit l'importance des variables
        - Évite souvent le surapprentissage
        
        **Inconvénients :**
        - Moins interprétable
        - Plus lent à entraîner
        - Peut sur-ajuster avec peu de données
        
        ### **⚡ XGBoost**
        **Avantages :**
        - Très haute performance
        - Gère bien les données manquantes
        - Optimisations avancées
        
        **Inconvénients :**
        - Complexe à paramétrer
        - Risque de surapprentissage
        - Moins interprétable
        
        ---
        
        ## 🚨 Signaux d'Alerte
        
        - **Surapprentissage** : R² train >> R² test (différence > 0.1)
        - **Sous-apprentissage** : R² train et test très faibles
        - **Données insuffisantes** : < 50 observations
        - **Caractéristiques peu informatives** : Toutes les importances similaires
        
        ---
        
        ## 💡 Conseils d'Amélioration
        
        1. **Plus de données** : Augmenter la taille de l'échantillon
        2. **Ingénierie des caractéristiques** : Créer de nouvelles variables
        3. **Nettoyage des données** : Éliminer les outliers
        4. **Segmentation** : Entraîner des modèles par segment (ville, type)
        5. **Validation croisée** : Tester sur plusieurs échantillons
        """)
def get_performance_label(r2_score):
    """Obtenir un label de performance basé sur le score R²"""
    if r2_score > 0.8:
        return "Excellent"
    elif r2_score > 0.6:
        return "Bon"
    elif r2_score > 0.4:
        return "Modéré"
    elif r2_score > 0.2:
        return "Faible"
    else:
        return "Très faible"

def check_overfitting(train_r2, test_r2):
    """Vérifier s'il y a du surapprentissage"""
    return (train_r2 - test_r2) > 0.1


# Version complète avec aide
def supervised_learning_section_complete(df, filtered_df):
    """Version complète avec section d'aide"""
    supervised_learning_section(df, filtered_df)
    add_supervised_learning_help()
# Version complète avec aide


    
def unsupervised_learning_section(df, filtered_df):
    st.header("🤖 Apprentissage Non Supervisé - Clustering")
    
    if df is None or filtered_df is None or df.empty or filtered_df.empty:
        st.error("Aucune donnée disponible pour l'apprentissage non supervisé.")
        return
    
    st.markdown("""
    <div class="info-box">
    L'apprentissage non supervisé permet de découvrir des structures cachées dans les données sans avoir de variable cible.
    Nous utiliserons trois algorithmes de clustering : K-Means, DBSCAN et CAH (Classification Ascendante Hiérarchique).
    </div>
    """, unsafe_allow_html=True)
    
    # ============================================
    # SECTION 1: FILTRES ET SÉLECTION DES DONNÉES
    # ============================================
    
    st.subheader("🔍 Filtres et Sélection des Données")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Filtre par ville
        if 'city' in filtered_df.columns:
            city_options = ["Toutes"] + sorted(filtered_df['city'].dropna().unique().tolist())
            selected_city = st.selectbox("Ville", city_options, key="clustering_city")
            selected_city = None if selected_city == "Toutes" else selected_city
        else:
            selected_city = None
            st.info("Information sur la ville non disponible")
    
    with col2:
        # Filtre par type de propriété
        if 'property_type' in filtered_df.columns:
            property_options = ["Tous"] + sorted(filtered_df['property_type'].dropna().unique().tolist())
            selected_property = st.selectbox("Type de propriété", property_options, key="clustering_property")
            selected_property = None if selected_property == "Tous" else selected_property
        else:
            selected_property = None
            st.info("Information sur le type de propriété non disponible")
    
    with col3:
        # Filtre par type de transaction
        if 'transaction' in filtered_df.columns:
            transaction_options = ["Toutes"] + sorted(filtered_df['transaction'].dropna().unique().tolist())
            selected_transaction = st.selectbox("Type de transaction", transaction_options, key="clustering_transaction")
            selected_transaction = None if selected_transaction == "Toutes" else selected_transaction
        else:
            selected_transaction = None
            st.info("Information sur le type de transaction non disponible")
    
    # Appliquer les filtres
    df_for_clustering = filtered_df.copy()
    filter_applied = False
    
    if selected_city is not None:
        df_for_clustering = df_for_clustering[df_for_clustering['city'] == selected_city]
        filter_applied = True
    if selected_property is not None:
        df_for_clustering = df_for_clustering[df_for_clustering['property_type'] == selected_property]
        filter_applied = True
    if selected_transaction is not None:
        df_for_clustering = df_for_clustering[df_for_clustering['transaction'] == selected_transaction]
        filter_applied = True
    
    # Afficher les informations sur le filtrage
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Observations avant filtrage", len(filtered_df))
    with col2:
        st.metric("Observations après filtrage", len(df_for_clustering))
    with col3:
        reduction_pct = ((len(filtered_df) - len(df_for_clustering)) / len(filtered_df)) * 100 if len(filtered_df) > 0 else 0
        st.metric("Réduction", f"-{reduction_pct:.1f}%")
    
    # Vérifier qu'on a assez de données
    if len(df_for_clustering) < 10:
        st.warning("⚠️ Pas assez de données pour l'analyse de clustering (minimum 10 observations). Veuillez élargir les filtres.")
        return
    
    # ============================================
    # SECTION 2: SÉLECTION DES CARACTÉRISTIQUES
    # ============================================
    
    st.subheader("📊 Sélection des Caractéristiques")
    
    # Obtenir les colonnes numériques disponibles
    numeric_cols = df_for_clustering.select_dtypes(include=['number']).columns.tolist()
    exclude_cols = ['date', 'source', 'neighborhood', 'suffix', 'listing_price', 'price_ttc', 'construction_year']
    available_features = [col for col in numeric_cols if col not in exclude_cols]
    
    if not available_features:
        st.error("❌ Aucune caractéristique numérique disponible pour le clustering.")
        return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_features = st.multiselect(
            "Sélectionner les caractéristiques pour le clustering",
            available_features,
            default=available_features[:6] if len(available_features) >= 6 else available_features,
            help="Choisissez les caractéristiques qui seront utilisées pour identifier les groupes de propriétés similaires"
        )
    
    with col2:
        st.write("**Caractéristiques disponibles:**")
        st.write(f"• Total: {len(available_features)}")
        st.write(f"• Sélectionnées: {len(selected_features)}")
        if selected_features:
            st.write("**Liste sélectionnée:**")
            for feature in selected_features:
                st.write(f"- {feature}")
    
    if not selected_features:
        st.warning("⚠️ Veuillez sélectionner au moins une caractéristique pour continuer.")
        return
    
    # ============================================
    # SECTION 3: PARAMÈTRES DES ALGORITHMES
    # ============================================
    
    st.subheader("⚙️ Configuration des Algorithmes")
    
    col1, col2 = st.columns(2)
    
    with col1:
        algorithm = st.selectbox(
            "Algorithme de clustering",
            ["K-Means", "DBSCAN", "CAH (Classification Ascendante Hiérarchique)", "Comparaison des 3 méthodes"],
            help="Choisissez l'algorithme de clustering à utiliser"
        )
    
    with col2:
        n_components_pca = st.slider(
            "Nombre de composantes PCA pour visualisation",
            min_value=2,
            max_value=min(len(selected_features), 10),
            value=min(3, len(selected_features)),
            help="Nombre de composantes principales à conserver pour la visualisation"
        )
    
    # Paramètres spécifiques selon l'algorithme
    st.write("**Paramètres spécifiques:**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Paramètres K-Means
        if algorithm in ["K-Means", "Comparaison des 3 méthodes"]:
            st.write("🔵 **K-Means**")
            max_clusters = min(10, len(df_for_clustering) // 5)
            n_clusters_range = st.slider(
                "Nombre de clusters à tester",
                min_value=2,
                max_value=max_clusters,
                value=(2, min(8, max_clusters)),
                key="kmeans_range"
            )
    
    with col2:
        # Paramètres DBSCAN
        if algorithm in ["DBSCAN", "Comparaison des 3 méthodes"]:
            st.write("🔴 **DBSCAN**")
            eps_range = st.slider(
                "Range eps",
                min_value=0.1,
                max_value=5.0,
                value=(0.3, 2.0),
                step=0.1,
                key="dbscan_eps"
            )
            min_samples_range = st.slider(
                "Range min_samples",
                min_value=2,
                max_value=20,
                value=(3, 10),
                key="dbscan_samples"
            )
    
    with col3:
        # Paramètres CAH
        if algorithm in ["CAH (Classification Ascendante Hiérarchique)", "Comparaison des 3 méthodes"]:
            st.write("🟢 **CAH**")
            linkage_method = st.selectbox(
                "Méthode de liaison",
                ["ward", "complete", "average", "single"],
                index=0,
                key="cah_linkage"
            )
            max_clusters_cah = st.slider(
                "Nombre max de clusters CAH",
                min_value=2,
                max_value=min(15, len(df_for_clustering) // 3),
                value=min(10, len(df_for_clustering) // 3),
                key="cah_max"
            )
    
    # ============================================
    # SECTION 4: LANCEMENT DE L'ANALYSE
    # ============================================
    
    if st.button("🚀 Lancer l'Analyse de Clustering", type="primary"):
        with st.spinner("🔄 Préparation des données..."):
            try:
                # Préparer les données pour le clustering
                df_scaled, scaler, feature_names = prepare_data_for_clustering(
                    df_for_clustering, 
                    features_for_clustering=selected_features
                )
                
                if len(df_scaled) < 10:
                    st.error("❌ Pas assez de données valides après nettoyage. Vérifiez vos données.")
                    return
                
                st.success(f"✅ Données préparées: {len(df_scaled)} observations avec {len(feature_names)} caractéristiques")
                
                # Afficher les caractéristiques utilisées
                st.info(f"**Caractéristiques utilisées:** {', '.join(feature_names)}")
                
                # Appliquer PCA pour la visualisation
                with st.spinner("🔄 Application de l'analyse PCA..."):
                    pca_model, df_pca, explained_variance = apply_pca_analysis(df_scaled, n_components_pca)
                
                st.write(f"**Variance expliquée par PCA:** {explained_variance.sum()*100:.1f}% (composantes 1-{n_components_pca})")
                
                # ============================================
                # EXÉCUTION DES ALGORITHMES
                # ============================================
                
                if algorithm == "K-Means":
                    st.subheader("🔵 Résultats K-Means")
                    
                    with st.spinner("🔄 Exécution de K-Means..."):
                        kmeans_model, best_n_clusters, cluster_labels, metrics, scores, n_clusters_list = apply_kmeans_clustering(
                            df_scaled, 
                            n_clusters_range=n_clusters_range,
                            random_state=42
                        )
                    
                    # Afficher les métriques
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("🎯 Clusters optimaux", best_n_clusters)
                    with col2:
                        st.metric("📊 Score Silhouette", f"{metrics['silhouette_score']:.4f}")
                    with col3:
                        st.metric("📈 Score Calinski-Harabasz", f"{metrics['calinski_harabasz_score']:.0f}")
                    with col4:
                        st.metric("⚡ Inertie", f"{metrics['inertia']:.0f}")
                    
                    # Graphique d'optimisation
                    fig_optimization = px.line(
                        x=n_clusters_list, 
                        y=scores,
                        title="Optimisation du nombre de clusters - Score de Silhouette",
                        labels={'x': 'Nombre de clusters', 'y': 'Score de Silhouette'},
                        markers=True
                    )
                    fig_optimization.add_vline(
                        x=best_n_clusters, 
                        line_dash="dash", 
                        line_color="red", 
                        annotation_text=f"Optimal: {best_n_clusters}"
                    )
                    fig_optimization.update_layout(hovermode="x unified")
                    st.plotly_chart(fig_optimization, use_container_width=True)
                    
                    # Visualisation avec matplotlib
                    st.subheader("📊 Visualisations K-Means")
                    try:
                        # Utiliser le code corrigé avec matplotlib
                        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                        
                        # 1. Clusters en 2D PCA
                        colors = plt.cm.Set3(np.linspace(0, 1, len(set(cluster_labels))))
                        for i, label in enumerate(sorted(set(cluster_labels))):
                            mask = cluster_labels == label
                            axes[0,0].scatter(df_pca.iloc[mask, 0], df_pca.iloc[mask, 1], 
                                             c=[colors[i]], label=f'Cluster {label}', alpha=0.7)
                        axes[0,0].set_xlabel('PC1')
                        axes[0,0].set_ylabel('PC2')
                        axes[0,0].set_title('K-Means - Clusters (2D PCA)')
                        axes[0,0].legend()
                        axes[0,0].grid(True, alpha=0.3)
                        
                        # 2. Distribution des clusters
                        cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
                        bars = axes[0,1].bar(cluster_counts.index, cluster_counts.values, color=colors[:len(cluster_counts)])
                        axes[0,1].set_xlabel('Cluster')
                        axes[0,1].set_ylabel('Nombre de propriétés')
                        axes[0,1].set_title('Distribution des clusters K-Means')
                        axes[0,1].grid(True, alpha=0.3)
                        
                        # Ajouter les valeurs sur les barres
                        for bar, count in zip(bars, cluster_counts.values):
                            axes[0,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                                           str(count), ha='center', va='bottom')
                        
                        # 3. Évolution du score de silhouette
                        axes[1,0].plot(n_clusters_list, scores, 'o-', linewidth=2, markersize=8)
                        axes[1,0].axvline(x=best_n_clusters, color='red', linestyle='--', 
                                          label=f'Optimal: {best_n_clusters} clusters')
                        axes[1,0].set_xlabel('Nombre de clusters')
                        axes[1,0].set_ylabel('Score de silhouette')
                        axes[1,0].set_title('Optimisation du nombre de clusters')
                        axes[1,0].legend()
                        axes[1,0].grid(True, alpha=0.3)
                        
                        # 4. Variance expliquée par PCA
                        pc_names = [f'PC{i+1}' for i in range(len(explained_variance))]
                        axes[1,1].bar(pc_names, explained_variance * 100, color='lightblue', edgecolor='black')
                        axes[1,1].set_xlabel('Composantes principales')
                        axes[1,1].set_ylabel('Variance expliquée (%)')
                        axes[1,1].set_title('Variance expliquée par PCA')
                        axes[1,1].grid(True, alpha=0.3)
                        
                        # Ajouter les pourcentages sur les barres
                        for i, var in enumerate(explained_variance):
                            axes[1,1].text(i, var * 100 + 1, f'{var*100:.1f}%', ha='center', va='bottom')
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Analyser les clusters
                        analyze_kmeans_clusters(df_for_clustering, df_scaled, cluster_labels, feature_names)
                        
                    except Exception as e:
                        st.error(f"Erreur lors de la visualisation: {e}")
                
                elif algorithm == "DBSCAN":
                    st.subheader("🔴 Résultats DBSCAN")
                    
                    with st.spinner("🔄 Exécution de DBSCAN..."):
                        dbscan_model, cluster_labels, metrics = apply_dbscan_clustering(
                            df_scaled,
                            eps_range=eps_range,
                            min_samples_range=min_samples_range
                        )
                    
                    # Afficher les métriques
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("🎯 Clusters trouvés", metrics['n_clusters'])
                    with col2:
                        st.metric("🔍 Points de bruit", metrics['n_noise_points'])
                    with col3:
                        st.metric("📊 Ratio de bruit", f"{metrics['noise_ratio']*100:.1f}%")
                    with col4:
                        if metrics['silhouette_score'] > 0:
                            st.metric("📈 Score Silhouette", f"{metrics['silhouette_score']:.4f}")
                        else:
                            st.metric("📈 Score Silhouette", "N/A")
                    
                    # Afficher les paramètres optimaux
                    st.info(f"**Paramètres optimaux:** eps={metrics['eps']:.3f}, min_samples={metrics['min_samples']}")
                    
                    # Visualisation DBSCAN
                    st.subheader("📊 Visualisations DBSCAN")
                    try:
                        visualize_dbscan_results(df_pca, cluster_labels, explained_variance)
                        
                        # Analyser les clusters DBSCAN
                        analyze_dbscan_clusters(df_for_clustering, df_scaled, cluster_labels, feature_names, metrics)
                        
                    except Exception as e:
                        st.error(f"Erreur lors de la visualisation DBSCAN: {e}")
                
                elif algorithm == "CAH (Classification Ascendante Hiérarchique)":
                    st.subheader("🟢 Résultats CAH")
                    
                    with st.spinner("🔄 Exécution de CAH..."):
                        linkage_matrix, cluster_labels, optimal_n_clusters, cah_metrics = apply_cah_clustering(
                            df_scaled,
                            max_clusters=max_clusters_cah,
                            linkage_method=linkage_method
                        )
                    
                    # Afficher les métriques
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("🎯 Clusters optimaux", optimal_n_clusters)
                    with col2:
                        st.metric("📊 Score Silhouette", f"{cah_metrics['silhouette_score']:.4f}")
                    with col3:
                        st.metric("🔗 Méthode de liaison", linkage_method)
                    
                    # Dendrogramme
                    st.subheader("🌳 Dendrogramme")
                    try:
                        dendrogram_data = visualize_cah_dendrogram(
                            linkage_matrix,
                            optimal_n_clusters=optimal_n_clusters,
                            max_display=min(50, len(df_scaled))
                        )
                        st.pyplot(plt.gcf())
                        
                        # Analyse complète CAH
                        results_df, detailed_df, hierarchy_df = complete_cah_analysis_after_dendrogram(
                            df, df_scaled, linkage_matrix, cluster_labels, optimal_n_clusters, feature_names
                        )
                        
                        # Afficher les résultats sous forme de tableaux
                        st.subheader("📋 Analyse des Clusters CAH")
                        st.dataframe(results_df, use_container_width=True)
                        
                        with st.expander("🔍 Détail de toutes les propriétés par cluster"):
                            st.dataframe(detailed_df, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Erreur lors de l'analyse CAH: {e}")
                
                else:  # Comparaison des 3 méthodes
                    st.subheader("🔄 Comparaison des 3 Méthodes")
                    
                    results = {}
                    
                    # K-Means
                    with st.spinner("🔄 Exécution de K-Means..."):
                        kmeans_model, kmeans_n_clusters, kmeans_labels, kmeans_metrics, _, _ = apply_kmeans_clustering(
                            df_scaled, n_clusters_range=n_clusters_range, random_state=42
                        )
                        results['K-Means'] = {
                            'labels': kmeans_labels,
                            'n_clusters': kmeans_n_clusters,
                            'silhouette_score': kmeans_metrics['silhouette_score'],
                            'method': 'K-Means'
                        }
                    
                    # DBSCAN
                    with st.spinner("🔄 Exécution de DBSCAN..."):
                        dbscan_model, dbscan_labels, dbscan_metrics = apply_dbscan_clustering(
                            df_scaled, eps_range=eps_range, min_samples_range=min_samples_range
                        )
                        results['DBSCAN'] = {
                            'labels': dbscan_labels,
                            'n_clusters': dbscan_metrics['n_clusters'],
                            'silhouette_score': dbscan_metrics['silhouette_score'],
                            'noise_points': dbscan_metrics['n_noise_points'],
                            'method': 'DBSCAN'
                        }
                    
                    # CAH
                    with st.spinner("🔄 Exécution de CAH..."):
                        linkage_matrix, cah_labels, cah_n_clusters, cah_metrics = apply_cah_clustering(
                            df_scaled, max_clusters=max_clusters_cah, linkage_method=linkage_method
                        )
                        results['CAH'] = {
                            'labels': cah_labels,
                            'n_clusters': cah_n_clusters,
                            'silhouette_score': cah_metrics['silhouette_score'],
                            'method': 'CAH'
                        }
                    
                    # Tableau de comparaison
                    comparison_data = {
                        'Méthode': ['K-Means', 'DBSCAN', 'CAH'],
                        'Nombre de clusters': [
                            results['K-Means']['n_clusters'],
                            results['DBSCAN']['n_clusters'],
                            results['CAH']['n_clusters']
                        ],
                        'Score Silhouette': [
                            f"{results['K-Means']['silhouette_score']:.4f}",
                            f"{results['DBSCAN']['silhouette_score']:.4f}" if results['DBSCAN']['silhouette_score'] > 0 else "N/A",
                            f"{results['CAH']['silhouette_score']:.4f}"
                        ],
                        'Points de bruit': [
                            "0",
                            str(results['DBSCAN']['noise_points']),
                            "0"
                        ]
                    }
                    
                    comparison_df = pd.DataFrame(comparison_data)
                    st.dataframe(comparison_df, use_container_width=True)
                    
                    # Visualisations comparatives
                    st.subheader("📊 Visualisations Comparatives")
                    
                    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
                    
                    methods = ['K-Means', 'DBSCAN', 'CAH']
                    labels_list = [kmeans_labels, dbscan_labels, cah_labels]
                    
                    for idx, (method, labels) in enumerate(zip(methods, labels_list)):
                        unique_labels = sorted(set(labels))
                        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
                        
                        for i, label in enumerate(unique_labels):
                            mask = labels == label
                            if label == -1:
                                axes[idx].scatter(df_pca.iloc[mask, 0], df_pca.iloc[mask, 1], 
                                                 c='black', label='Bruit', alpha=0.7, marker='x')
                            else:
                                axes[idx].scatter(df_pca.iloc[mask, 0], df_pca.iloc[mask, 1], 
                                                 c=[colors[i]], label=f'Cluster {label}', alpha=0.7)
                        
                        axes[idx].set_xlabel('PC1')
                        axes[idx].set_ylabel('PC2')
                        axes[idx].set_title(f'{method}')
                        axes[idx].legend()
                        axes[idx].grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Recommandations
                    st.subheader("💡 Recommandations")
                    
                    best_method = max(results.items(), key=lambda x: x[1]['silhouette_score'])
                    st.success(f"🏆 **Meilleure méthode:** {best_method[0]} (Score Silhouette: {best_method[1]['silhouette_score']:.4f})")
                    
                    # Recommandations contextuelles
                    if results['DBSCAN']['noise_points'] > len(df_scaled) * 0.3:
                        st.warning("⚠️ DBSCAN détecte beaucoup de bruit (>30%). Considérez K-Means ou CAH.")
                    
                    if results['K-Means']['silhouette_score'] > 0.5:
                        st.info("👍 K-Means montre une bonne séparation des clusters.")
                    
                    if results['CAH']['silhouette_score'] > results['K-Means']['silhouette_score']:
                        st.info("🌳 CAH pourrait être plus approprié pour vos données hiérarchiques.")
            
            except Exception as e:
                st.error(f"❌ Une erreur s'est produite lors de l'analyse: {e}")
                st.info("💡 Vérifiez que vos données sont bien formatées et qu'il y a suffisamment d'observations.")

# ============================================
# FONCTIONS AUXILIAIRES POUR LES VISUALISATIONS
# ============================================

def analyze_kmeans_clusters(df_original, df_scaled, cluster_labels, feature_names):
    """Analyse détaillée des clusters K-Means"""
    st.subheader("🔍 Analyse Détaillée des Clusters K-Means")
    
    # Créer un DataFrame avec les résultats
    # Ensure indices match between df_original and df_scaled
    analysis_df = df_original.reset_index(drop=True).iloc[:len(cluster_labels)].copy()
    analysis_df['Cluster'] = cluster_labels
    
    for cluster_id in sorted(set(cluster_labels)):
        with st.expander(f"📊 Cluster {cluster_id} - Analyse", expanded=False):
            cluster_data = analysis_df[analysis_df['Cluster'] == cluster_id]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Nombre de propriétés", len(cluster_data))
                
                if 'price' in cluster_data.columns:
                    st.metric("Prix moyen", f"{cluster_data['price'].mean():.0f} TND")
                    st.metric("Prix médian", f"{cluster_data['price'].median():.0f} TND")
                
                if 'size' in cluster_data.columns:
                    st.metric("Taille moyenne", f"{cluster_data['size'].mean():.0f} m²")
            
            with col2:
                if 'neighborhood' in cluster_data.columns:
                    neighborhoods = cluster_data['neighborhood'].value_counts().head(3)
                    st.write("**Top 3 quartiers:**")
                    for neighborhood, count in neighborhoods.items():
                        st.write(f"• {neighborhood}: {count} propriétés")
                
                if 'condition' in cluster_data.columns:
                    conditions = cluster_data['condition'].value_counts().head(3)
                    st.write("**États principaux:**")
                    for condition, count in conditions.items():
                        st.write(f"• {condition}: {count} propriétés")

def visualize_dbscan_results(df_pca, cluster_labels, explained_variance):
    """Visualisation spécifique pour DBSCAN"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Clusters DBSCAN en 2D PCA
    unique_labels = sorted(set(cluster_labels))
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = cluster_labels == label
        if label == -1:
            axes[0,0].scatter(df_pca.iloc[mask, 0], df_pca.iloc[mask, 1], 
                             c='black', label='Bruit', alpha=0.7, marker='x')
        else:
            axes[0,0].scatter(df_pca.iloc[mask, 0], df_pca.iloc[mask, 1], 
                             c=[colors[i]], label=f'Cluster {label}', alpha=0.7)
    
    axes[0,0].set_xlabel('PC1')
    axes[0,0].set_ylabel('PC2')
    axes[0,0].set_title('DBSCAN - Clusters (2D PCA)')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Distribution des clusters DBSCAN
    cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
    cluster_names = ['Bruit' if idx == -1 else f'Cluster {idx}' for idx in cluster_counts.index]
    bar_colors = ['black' if idx == -1 else colors[i] for i, idx in enumerate(cluster_counts.index)]
    
    bars = axes[0,1].bar(range(len(cluster_counts)), cluster_counts.values, color=bar_colors)
    axes[0,1].set_xticks(range(len(cluster_counts)))
    axes[0,1].set_xticklabels(cluster_names, rotation=45)
    axes[0,1].set_ylabel('Nombre de propriétés')
    axes[0,1].set_title('Distribution des clusters DBSCAN')
    axes[0,1].grid(True, alpha=0.3)
    
    # Ajouter les valeurs sur les barres
    for bar, count in zip(bars, cluster_counts.values):
        axes[0,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                       str(count), ha='center', va='bottom')
    
    # 3. Variance expliquée par PCA
    pc_names = [f'PC{i+1}' for i in range(len(explained_variance))]
    axes[1,0].bar(pc_names, explained_variance * 100, color='lightblue', edgecolor='black')
    axes[1,0].set_xlabel('Composantes principales')
    axes[1,0].set_ylabel('Variance expliquée (%)')
    axes[1,0].set_title('Variance expliquée par PCA')
    axes[1,0].grid(True, alpha=0.3)
    
    # Ajouter les pourcentages sur les barres
    for i, var in enumerate(explained_variance):
        axes[1,0].text(i, var * 100 + 1, f'{var*100:.1f}%', ha='center', va='bottom')
    
    # 4. Histogramme des distances aux points centraux (si applicable)
    axes[1,1].hist(cluster_labels, bins=len(unique_labels), color='lightcoral', edgecolor='black', alpha=0.7)
    axes[1,1].set_xlabel('Cluster ID')
    axes[1,1].set_ylabel('Fréquence')
    axes[1,1].set_title('Distribution des assignations de clusters')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)

def analyze_dbscan_clusters(df_original, df_scaled, cluster_labels, feature_names, metrics):
    """Analyse détaillée des clusters DBSCAN"""
    st.subheader("🔍 Analyse Détaillée des Clusters DBSCAN")
    
    # Évaluation de la qualité
    noise_ratio = metrics['noise_ratio']
    n_clusters = metrics['n_clusters']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if noise_ratio < 0.1:
            st.success("✅ Excellent ratio de bruit (<10%)")
        elif noise_ratio < 0.2:
            st.info("👍 Bon ratio de bruit (<20%)")
        elif noise_ratio < 0.4:
            st.warning("⚠️ Ratio de bruit acceptable (<40%)")
        else:
            st.error("❌ Ratio de bruit problématique (>40%)")
    
    with col2:
        if n_clusters >= 2 and n_clusters <= 6:
            st.success(f"✅ Nombre optimal de clusters: {n_clusters}")
        elif n_clusters == 1:
            st.warning("⚠️ Un seul cluster détecté")
        elif n_clusters == 0:
            st.error("❌ Aucun cluster détecté")
        else:
            st.info(f"📊 {n_clusters} clusters détectés")
    
    with col3:
        if metrics['silhouette_score'] > 0.5:
            st.success(f"✅ Excellente cohésion: {metrics['silhouette_score']:.3f}")
        elif metrics['silhouette_score'] > 0.3:
            st.info(f"👍 Bonne cohésion: {metrics['silhouette_score']:.3f}")
        elif metrics['silhouette_score'] > 0:
            st.warning(f"⚠️ Cohésion faible: {metrics['silhouette_score']:.3f}")
        else:
            st.error("❌ Impossible de calculer la cohésion")
    
    # Analyser chaque cluster + bruit
    analysis_df = df_original.reset_index(drop=True).iloc[:len(cluster_labels)].copy()
    analysis_df['Cluster'] = cluster_labels
    
    unique_labels = sorted(set(cluster_labels))
    
    for label in unique_labels:
        cluster_data = analysis_df[analysis_df['Cluster'] == label]
        
        if label == -1:
            with st.expander(f"🔍 Points de Bruit - {len(cluster_data)} propriétés", expanded=False):
                st.warning("⚠️ Ces propriétés ont des caractéristiques uniques/atypiques")
                
                if 'price' in cluster_data.columns:
                    st.write(f"**Prix:** {cluster_data['price'].min():.0f} - {cluster_data['price'].max():.0f} TND")
                    prix_median_general = analysis_df[analysis_df['Cluster'] != -1]['price'].median()
                    bonnes_affaires = cluster_data[cluster_data['price'] < prix_median_general]
                    if len(bonnes_affaires) > 0:
                        st.info(f"💰 {len(bonnes_affaires)} propriétés sous le prix médian (bonnes affaires potentielles)")
                
                if len(cluster_data) <= 10:
                    st.dataframe(cluster_data[['price', 'size', 'neighborhood'] if all(col in cluster_data.columns for col in ['price', 'size', 'neighborhood']) else cluster_data.columns[:5]])
        else:
            with st.expander(f"📊 Cluster {label} - {len(cluster_data)} propriétés", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    if 'price' in cluster_data.columns:
                        st.metric("Prix moyen", f"{cluster_data['price'].mean():.0f} TND")
                        st.metric("Écart-type prix", f"{cluster_data['price'].std():.0f} TND")
                    
                    if 'size' in cluster_data.columns:
                        st.metric("Taille moyenne", f"{cluster_data['size'].mean():.0f} m²")
                
                with col2:
                    if 'neighborhood' in cluster_data.columns:
                        neighborhoods = cluster_data['neighborhood'].value_counts().head(3)
                        st.write("**Quartiers principaux:**")
                        for neighborhood, count in neighborhoods.items():
                            pct = (count / len(cluster_data)) * 100
                            st.write(f"• {neighborhood}: {count} ({pct:.0f}%)")
                    
                    if 'condition' in cluster_data.columns:
                        conditions = cluster_data['condition'].value_counts().head(2)
                        st.write("**États principaux:**")
                        for condition, count in conditions.items():
                            pct = (count / len(cluster_data)) * 100
                            st.write(f"• {condition}: {count} ({pct:.0f}%)")

def create_cluster_profile_radar(df_with_clusters, selected_features, cluster_labels):
    """Créer un graphique radar pour profiler les clusters"""
    unique_clusters = [c for c in sorted(set(cluster_labels)) if c != -1]  # Exclure le bruit
    
    if len(unique_clusters) <= 1 or len(selected_features) <= 2:
        return None
    
    # Calculer les moyennes par cluster
    cluster_means = df_with_clusters.groupby('Cluster')[selected_features].mean()
    
    # Normaliser les valeurs (0-1) pour le radar chart
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    cluster_means_normalized = pd.DataFrame(
        scaler.fit_transform(cluster_means),
        columns=cluster_means.columns,
        index=cluster_means.index
    )
    
    # Créer le graphique radar avec Plotly
    fig = go.Figure()
    
    colors = px.colors.qualitative.Set3
    
    for i, (cluster_id, row) in enumerate(cluster_means_normalized.iterrows()):
        if cluster_id != -1:  # Exclure le bruit
            fig.add_trace(go.Scatterpolar(
                r=row.values.tolist() + [row.values[0]],  # Fermer le polygone
                theta=row.index.tolist() + [row.index[0]],
                fill='toself',
                name=f'Cluster {cluster_id}',
                marker_color=colors[i % len(colors)],
                opacity=0.6
            ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickmode='linear',
                tick0=0,
                dtick=0.2
            )
        ),
        showlegend=True,
        title="Profil des clusters (valeurs normalisées 0-1)",
        height=500
    )
    
    return fig

# ============================================
# SECTION D'AIDE ET INTERPRÉTATION
# ============================================

def add_clustering_help_section():
    """Section d'aide pour l'interprétation des résultats"""
    with st.expander("💡 Guide d'Interprétation des Résultats de Clustering", expanded=False):
        st.markdown("""
        ## 🔵 K-Means
        **Principe :** Divise les données en k clusters en minimisant la variance intra-cluster.
        
        **Métriques importantes :**
        - **Score de Silhouette** (0 à 1) : Mesure la qualité du clustering. Plus proche de 1 = meilleur.
        - **Inertie** : Somme des distances au carré des points à leur centroïde. Plus faible = mieux.
        - **Score Calinski-Harabasz** : Ratio de dispersion entre/dans les clusters. Plus élevé = mieux.
        
        **Avantages :** Rapide, stable, bon pour clusters sphériques.
        **Inconvénients :** Nécessite de spécifier k, sensible aux outliers.
        
        ---
        
        ## 🔴 DBSCAN
        **Principe :** Identifie des clusters de densité et marque les points isolés comme bruit.
        
        **Paramètres clés :**
        - **eps** : Distance maximale entre deux points pour qu'ils soient voisins.
        - **min_samples** : Nombre minimum de points pour former un cluster.
        
        **Métriques importantes :**
        - **Points de bruit** : Points qui ne peuvent être assignés à aucun cluster.
        - **Ratio de bruit** : Pourcentage de points de bruit. <20% = bon, >40% = problématique.
        
        **Avantages :** Détecte automatiquement le nombre de clusters, robuste aux outliers, clusters de forme arbitraire.
        **Inconvénients :** Sensible aux paramètres, difficile avec densités variables.
        
        ---
        
        ## 🟢 CAH (Classification Ascendante Hiérarchique)
        **Principe :** Construit une hiérarchie de clusters en fusionnant progressivement les plus proches.
        
        **Méthodes de liaison :**
        - **Ward** : Minimise la variance intra-cluster (recommandé).
        - **Complete** : Distance maximale entre clusters.
        - **Average** : Distance moyenne entre clusters.
        - **Single** : Distance minimale entre clusters.
        
        **Avantages :** Dendrogramme informatif, pas besoin de spécifier k à l'avance, déterministe.
        **Inconvénients :** Coûteux en calcul, sensible au bruit.
        
        ---
        
        ## 📊 Interprétation Business
        
        ### Types de Segments Immobiliers Typiques :
        
        **🏠 Segment Économique**
        - Prix bas, tailles modestes
        - Cible : Étudiants, jeunes professionnels
        - Stratégie : Accessibilité, localisation transport
        
        **🏡 Segment Familial**
        - Prix moyen, surfaces généreuses
        - Cible : Familles, primo-accédants
        - Stratégie : Rapport qualité/prix, commodités
        
        **💎 Segment Premium**
        - Prix élevé, équipements haut de gamme
        - Cible : Cadres, investisseurs
        - Stratégie : Luxe, services, localisations privilégiées
        
        **🔍 Points Atypiques (Bruit DBSCAN)**
        - Propriétés uniques ou mal valorisées
        - Opportunités d'investissement potentielles
        - À investiguer individuellement
        
        ---
        
        ## 🎯 Conseils d'Analyse
        
        1. **Validation Métier** : Les clusters doivent avoir du sens business.
        2. **Taille des Clusters** : Éviter les clusters trop petits (<5% des données).
        3. **Stabilité** : Tester plusieurs algorithmes pour confirmer.
        4. **Actionabilité** : Chaque cluster doit permettre des actions marketing distinctes.
        5. **Équilibrage** : Préférer des clusters de tailles relativement équilibrées.
        
        ### Signaux d'Alerte :
        - Score Silhouette < 0.2 : Clustering peu fiable
        - Trop de bruit DBSCAN (>50%) : Revoir les paramètres
        - Un seul gros cluster : Données trop homogènes ou paramètres inadéquats
        """)

# Ajouter la section d'aide à la fin de la fonction principale
def unsupervised_learning_section_complete(df, filtered_df):
    """Version complète avec section d'aide"""
    # Appeler la fonction principale
    unsupervised_learning_section(df, filtered_df)
    
    # Ajouter la section d'aide
    add_clustering_help_section()

# ============================================
# FONCTIONS UTILITAIRES SUPPLÉMENTAIRES
# ============================================

def export_clustering_results(df_with_clusters, algorithm_name, cluster_labels):
    """Permettre l'export des résultats de clustering"""
    st.subheader(f"💾 Export des Résultats {algorithm_name}")
    
    if st.button(f"Télécharger les résultats {algorithm_name}"):
        # Préparer les données pour export
        export_df = df_with_clusters.copy()
        export_df[f'Cluster_{algorithm_name}'] = cluster_labels
        
        # Convertir en CSV
        csv = export_df.to_csv(index=False)
        
        st.download_button(
            label=f"📁 Télécharger CSV avec clusters {algorithm_name}",
            data=csv,
            file_name=f"clustering_results_{algorithm_name}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )
        
        st.success("✅ Fichier prêt pour téléchargement!")

def display_cluster_statistics_summary(results_dict):
    """Afficher un résumé statistique des différents algorithmes"""
    st.subheader("📈 Résumé Statistique Comparatif")
    
    summary_data = []
    for method, result in results_dict.items():
        summary_data.append({
            'Algorithme': method,
            'Nombre de Clusters': result['n_clusters'],
            'Score Silhouette': f"{result['silhouette_score']:.4f}",
            'Points de Bruit': result.get('noise_points', 0),
            'Recommandation': get_algorithm_recommendation(result)
        })
    
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True)

def get_algorithm_recommendation(result):
    """Générer une recommandation basée sur les résultats"""
    silhouette = result['silhouette_score']
    n_clusters = result['n_clusters']
    noise_points = result.get('noise_points', 0)
    
    if silhouette > 0.6:
        return "🌟 Excellent"
    elif silhouette > 0.4:
        return "👍 Bon"
    elif silhouette > 0.2:
        return "⚠️ Acceptable"
    elif n_clusters == 0:
        return "❌ Échec"
    else:
        return "🔄 À revoir"
# Application principale
# Initialiser les variables de session
if 'df_imputed' not in st.session_state:
    st.session_state['df_imputed'] = None

# Téléchargement du fichier

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
            tab1, tab2, tab3, tab4, tab5,tab6 = st.tabs(["Aperçu des données", "Statistiques de base","Visualisations avancées", "Imputation" , "Apprentissage supervisé","Apprentissage non supervisé"])
            
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
                        avg_price_sale = filtered_df.loc[filtered_df['transaction'] == 'sale', 'price'].dropna().mean()
                        if pd.notna(avg_price_sale):
                            st.metric("Prix moyen de vente", f"{avg_price_sale:,.0f} TND")
                        else:
                            st.metric("Prix moyen", "N/A")
                        avg_price_rent = filtered_df.loc[filtered_df['transaction'] == 'rent', 'price'].dropna().mean()
                        if pd.notna(avg_price_rent):
                            st.metric("Prix moyen de location", f"{avg_price_rent:,.0f} TND")
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
                # Visualisations avancées
                advanced_visualizations(filtered_df)
                
            with tab4:
                # Section d'imputation
                df = imputation_section(df)
            
            with tab5:
                # Section d'apprentissage supervisé
                supervised_learning_section_complete(df, filtered_df)
                
            with tab6:
                
                unsupervised_learning_section(df, filtered_df)
        
    except Exception as e:
        st.error(f"Une erreur s'est produite lors du traitement du fichier: {e}")
        st.info("Conseil: Vérifiez le format de votre fichier CSV et assurez-vous que les colonnes sont correctement nommées.")