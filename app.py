import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(
    page_title="Analyse Immobilière",
    page_icon="🏠",
    layout="wide"
)

st.title("Analyse du Marché Immobilier Tunisien")

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
        if col in df.columns:
            # Convertir tout en minuscules pour standardiser
            df[col] = df[col].str.lower() if df[col].dtype == 'object' else df[col]
    
    return df

# Option pour uploader un fichier
uploaded_file = st.file_uploader("Télécharger un fichier CSV", type=['csv'])

if uploaded_file is not None:
    try:
        # Essayer de charger avec différents séparateurs
        try:
            df = pd.read_csv(uploaded_file, sep=',')
        except:
            try:
                # Réinitialiser le curseur du fichier avant de le relire
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, sep=';')
            except:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, sep='\t')
        
        # Prétraiter les données
        df = preprocess_data(df)
        
        st.success("Fichier chargé avec succès!")
        
        # Créer des onglets pour organiser l'interface
        tab1, tab2, tab3 = st.tabs(["Aperçu des données", "Analyse de base", "Visualisations"])
        
        with tab1:
            st.subheader("Aperçu des données")
            st.dataframe(df.head())
            
            st.subheader("Informations sur les colonnes")
            buffer = st.empty()
            with buffer.container():
                col1, col2 = st.columns(2)
                with col1:
                    st.write("Types de données:")
                    st.write(pd.DataFrame({'Type': df.dtypes}))
                with col2:
                    st.write("Valeurs manquantes:")
                    st.write(pd.DataFrame({'Manquantes': df.isna().sum(), 
                                          'Pourcentage': df.isna().sum() / len(df) * 100}))
        
        with tab2:
            # Filtres pour les données - INSENSIBLES À LA CASSE
            st.sidebar.title("Filtres")
            
            if 'transaction' in df.columns:
                # Obtenir les valeurs uniques et les trier
                unique_transactions = sorted(df['transaction'].dropna().unique().tolist())
                # Créer un affichage plus propre pour l'interface utilisateur
                display_transactions = ["Tous"] + [t.capitalize() for t in unique_transactions]
                # Créer un dictionnaire de mapping pour retrouver les valeurs d'origine
                transaction_map = {t.capitalize(): t for t in unique_transactions}
                transaction_map["Tous"] = "Tous"
                
                selected_display_transaction = st.sidebar.selectbox("Type de Transaction", display_transactions)
                selected_transaction = transaction_map[selected_display_transaction]
            else:
                selected_transaction = "Tous"
            
            if 'property_type' in df.columns:
                unique_property_types = sorted(df['property_type'].dropna().unique().tolist())
                display_property_types = ["Tous"] + [p.capitalize() for p in unique_property_types]
                property_type_map = {p.capitalize(): p for p in unique_property_types}
                property_type_map["Tous"] = "Tous"
                
                selected_display_property = st.sidebar.selectbox("Type de Bien", display_property_types)
                selected_property = property_type_map[selected_display_property]
            else:
                selected_property = "Tous"
            
            if 'city' in df.columns:
                unique_cities = sorted(df['city'].dropna().unique().tolist())
                display_cities = ["Toutes"] + [c.title() for c in unique_cities]
                city_map = {c.title(): c for c in unique_cities}
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
            
            # Statistiques de base
            st.subheader("Statistiques descriptives")
            
            # Sélectionner uniquement les colonnes numériques pour describe()
            numeric_df = filtered_df.select_dtypes(include=['number'])
            if not numeric_df.empty:
                st.write(numeric_df.describe())
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
        
        with tab3:
            st.subheader("Visualisations")
            
            # Visualisation par ville - avec titres correctement formatés
            if 'city' in filtered_df.columns:
                st.subheader("Nombre de propriétés par ville")
                city_counts = filtered_df['city'].value_counts().reset_index()
                city_counts.columns = ['ville', 'nombre']
                # Capitaliser les noms de ville pour l'affichage
                city_counts['ville'] = city_counts['ville'].str.title()
                
                fig = px.bar(city_counts, x='ville', y='nombre', 
                           title="Nombre de propriétés par ville")
                st.plotly_chart(fig, use_container_width=True)
            
            # Prix moyen par type de propriété - avec titres correctement formatés
            if 'property_type' in filtered_df.columns and 'price' in filtered_df.columns:
                valid_price_df = filtered_df.dropna(subset=['price'])
                if not valid_price_df.empty:
                    st.subheader("Prix moyen par type de propriété")
                    price_by_type = valid_price_df.groupby('property_type')['price'].mean().reset_index()
                    price_by_type.columns = ['type', 'prix_moyen']
                    # Capitaliser les types de bien pour l'affichage
                    price_by_type['type'] = price_by_type['type'].str.capitalize()
                    
                    fig = px.bar(price_by_type, x='type', y='prix_moyen', 
                               title="Prix moyen par type de propriété",
                               labels={'prix_moyen': 'Prix moyen (TND)', 'type': 'Type de bien'})
                    st.plotly_chart(fig, use_container_width=True)
            
            # Relation taille vs prix
            if 'size' in filtered_df.columns and 'price' in filtered_df.columns:
                valid_data = filtered_df.dropna(subset=['size', 'price'])
                if len(valid_data) > 5:  # Au moins 5 points pour un scatter plot
                    st.subheader("Relation entre taille et prix")
                    
                    # Si property_type existe, l'utiliser pour colorer les points
                    if 'property_type' in valid_data.columns:
                        # Créer une copie pour l'affichage avec property_type capitalisé
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
            
            # Carte thermique des commodités (si la colonne amenities existe)
            if 'amenities' in filtered_df.columns:
                st.subheader("Analyse des commodités")
                
                # Extraire et compter les commodités
                amenities_list = []
                for amenities in filtered_df['amenities'].dropna():
                    if '+' in amenities:
                        amenities_list.extend([a.strip().lower() for a in amenities.split('+')])
                    elif ',' in amenities:
                        amenities_list.extend([a.strip().lower() for a in amenities.split(',')])
                    else:
                        amenities_list.append(amenities.strip().lower())
                
                if amenities_list:
                    amenities_counts = pd.Series(amenities_list).value_counts().reset_index()
                    amenities_counts.columns = ['commodité', 'nombre']
                    # Capitaliser pour l'affichage
                    amenities_counts['commodité'] = amenities_counts['commodité'].str.capitalize()
                    # Prendre les 15 plus fréquentes
                    top_amenities = amenities_counts.head(15)
                    
                    fig = px.bar(top_amenities, x='commodité', y='nombre', 
                               title="Top 15 des commodités les plus fréquentes",
                               labels={'nombre': 'Fréquence', 'commodité': 'Commodité'})
                    fig.update_layout(xaxis={'categoryorder': 'total descending'})
                    st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        st.error(f"Une erreur s'est produite lors du traitement du fichier: {e}")
        st.info("Conseil: Vérifiez le format de votre fichier CSV et assurez-vous que les colonnes sont correctement nommées.")

else:
    st.info("Veuillez télécharger un fichier CSV pour commencer l'analyse.")
    
    # Afficher un exemple de format attendu
    st.subheader("Format de données attendu")
    example_data = """
    source,neighborhood,city,state,transaction,property_type,listing_date,price,suffix,size,rooms
    century 21,La Soukra,La Soukra,Ariana,rent,Appartement,2025-05-08,1500,TTC,110,3
    century 21,Hammam Sousse,Hammam Sousse,Sousse,rent,Appartement,2025-05-08,900,TTC,70,2
    """
    st.code(example_data, language="csv")
    
# # Ajout d'une section pour les insights
# if uploaded_file is not None and 'filtered_df' in locals():
#     st.header("Insights du marché immobilier")
    
#     # Calculer des insights intéressants basés sur les données
#     insights = []
    
#     # Insight 1: Type de bien le plus cher en moyenne
#     if 'property_type' in filtered_df.columns and 'price' in filtered_df.columns:
#         valid_price_df = filtered_df.dropna(subset=['price'])
#         if not valid_price_df.empty:
#             price_by_type = valid_price_df.groupby('property_type')['price'].mean()
#             if not price_by_type.empty:
#                 most_expensive_type = price_by_type.idxmax()
#                 highest_avg_price = price_by_type.max()
#                 insights.append(f"Le type de bien le plus cher en moyenne est '{most_expensive_type.capitalize()}' avec un prix moyen de {highest_avg_price:,.0f} TND.")
    
#     # Insight 2: Quartier le plus représenté
#     if 'neighborhood' in filtered_df.columns:
#         neighborhood_counts = filtered_df['neighborhood'].value_counts()
#         if not neighborhood_counts.empty:
#             top_neighborhood = neighborhood_counts.index[0]
#             top_count = neighborhood_counts.iloc[0]
#             insights.append(f"Le quartier le plus représenté est '{top_neighborhood.title()}' avec {top_count} propriétés.")
    
#     # Insight 3: Rapport qualité-prix (prix/m²)
#     if 'price' in filtered_df.columns and 'size' in filtered_df.columns:
#         filtered_df['price_per_sqm'] = filtered_df['price'] / filtered_df['size']
#         valid_data = filtered_df.dropna(subset=['price_per_sqm'])
        
#         if not valid_data.empty and 'city' in valid_data.columns:
#             price_per_sqm_by_city = valid_data.groupby('city')['price_per_sqm'].mean().sort_values()
            
#             if not price_per_sqm_by_city.empty:
#                 best_value_city = price_per_sqm_by_city.index[0]
#                 best_value_price = price_per_sqm_by_city.iloc[0]
#                 insights.append(f"La ville offrant le meilleur rapport qualité-prix est '{best_value_city.title()}' avec un prix moyen de {best_value_price:,.0f} TND/m².")
    
#     # Afficher les insights
#     if insights:
#         st.subheader("Points clés")
#         for i, insight in enumerate(insights, 1):
#             st.markdown(f"**{i}.** {insight}")
#     else:
#         st.info("Pas assez de données pour générer des insights pertinents.")
    
# Pied de page
st.markdown("---")
st.markdown("© 2025 - Mini-Projet d'Analyse du Marché Immobilier Tunisien")