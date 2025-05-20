import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.title("Analyse du Marché Immobilier Tunisien")

# Option pour uploader un fichier
uploaded_file = st.file_uploader("Télécharger un fichier CSV", type=['csv'])

if uploaded_file is not None:
    # Charger les données depuis le fichier uploadé
    df = pd.read_csv(uploaded_file)
    
    # Prétraiter les données numériques
    # Remplacer les valeurs non numériques par NaN dans la colonne 'price'
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df['size'] = pd.to_numeric(df['size'], errors='coerce')
    df['rooms'] = pd.to_numeric(df['rooms'], errors='coerce')
    df['bedrooms'] = pd.to_numeric(df['bedrooms'], errors='coerce')
    df['bathrooms'] = pd.to_numeric(df['bathrooms'], errors='coerce')
    df['parkings'] = pd.to_numeric(df['parkings'], errors='coerce')
    
    st.success("Fichier chargé avec succès!")
    
    # Filtres pour les données
    st.sidebar.title("Filtres")
    
    transaction_types = ["Tous"] + sorted(df['transaction'].unique().tolist())
    selected_transaction = st.sidebar.selectbox("Type de Transaction", transaction_types)
    
    property_types = ["Tous"] + sorted(df['property_type'].unique().tolist())
    selected_property = st.sidebar.selectbox("Type de Bien", property_types)
    
    # Appliquer les filtres
    filtered_df = df.copy()
    if selected_transaction != "Tous":
        filtered_df = filtered_df[filtered_df['transaction'] == selected_transaction]
    if selected_property != "Tous":
        filtered_df = filtered_df[filtered_df['property_type'] == selected_property]
    
    # Afficher les données brutes
    st.subheader("Aperçu des données")
    st.dataframe(filtered_df.head())
    
    # Statistiques de base
    st.subheader("Statistiques descriptives")
    st.write(filtered_df.describe())
    
    # Métriques clés
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Nombre de propriétés", len(filtered_df))
    with col2:
        # Seulement sur les valeurs non-NaN
        avg_price = filtered_df['price'].dropna().mean()
        st.metric("Prix moyen", f"{avg_price:,.0f} TND" if not pd.isna(avg_price) else "N/A")
    with col3:
        avg_size = filtered_df['size'].dropna().mean()
        st.metric("Surface moyenne", f"{avg_size:.1f} m²" if not pd.isna(avg_size) else "N/A")
    
    # Visualisations
    st.subheader("Nombre de propriétés par ville")
    city_counts = filtered_df['city'].value_counts().reset_index()
    city_counts.columns = ['ville', 'nombre']
    fig = px.bar(city_counts, x='ville', y='nombre', title="Nombre de propriétés par ville")
    st.plotly_chart(fig, use_container_width=True)
    
    # Vérifier si la colonne prix contient des données valides avant de créer le graphique
    if not filtered_df['price'].isna().all():
        st.subheader("Prix moyen par type de propriété")
        price_by_type = filtered_df.groupby('property_type')['price'].mean().reset_index()
        price_by_type.columns = ['type', 'prix_moyen']
        fig = px.bar(price_by_type, x='type', y='prix_moyen', title="Prix moyen par type de propriété")
        st.plotly_chart(fig, use_container_width=True)
        
        # Ajouter un graphique scatter plot pour taille vs prix
        if not filtered_df['size'].isna().all():
            st.subheader("Relation entre taille et prix")
            scatter_df = filtered_df.dropna(subset=['size', 'price'])
            fig = px.scatter(scatter_df, x='size', y='price', color='property_type',
                           title="Relation entre taille et prix",
                           labels={'size': 'Surface (m²)', 'price': 'Prix (TND)'})
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Impossible d'afficher les graphiques liés aux prix - données manquantes ou invalides.")
    
else:
    st.info("Veuillez télécharger un fichier CSV pour commencer l'analyse.")