import streamlit as st
import pandas as pd
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from unidecode import unidecode

# Utilisation du cache de streamlit pour accélérer le processus de chargement

@st.cache_data(show_spinner=False)
def data():
    df = pd.read_pickle(r"C:\Users\emman\Desktop\Projet2-0305\BDD_ML\df_ml.pkl")
    return df


@st.cache_resource(show_spinner=False)
def train(X):
    modelNN = NearestNeighbors(metric = "cosine", n_jobs=-1)
    modelNN.fit(X)
    return modelNN

@st.cache_data(show_spinner=False)
def df_to_show():
    to_show = pd.read_pickle(r"C:\Users\emman\Desktop\Projet2-0305\BDD_ML\to_show.pkl")
    return to_show


X = data()
to_show = df_to_show()
modelNN = train(X)

image = Image.open(r'C:\Users\emman\Desktop\Projet2-0305\Marque\logo2.png')
st.image(image)
st.title('')
st.title('')
st.title('"Te Creuse plus la tête !"')
st.write("Le système de recommandation de film qui lutte contre l'indécision cinématographique.")
st.markdown('_"Te Creuse plus la tête", te fais des recommandations depuis sa bibliothèque de plus de 22 000 films !_')
st.title('')
movie_to_transform = st.text_input("Quel est votre film préféré ?")
film = unidecode(movie_to_transform.lower())    # Permet de rendre le film insensible à la casse 
st.markdown("_Exemples : Titanic, Avatar, Forrest Gump..._")

# Faire une condition afin que le système ne charge pas si aucun film n'est saisi. Cela nous permet de gagner du temps de chargement également. 
if len(film) > 0:
    try :               # Try/except afin d'éviter les erreurs à l'affichage 
        film_reference = X.loc[film].to_frame().T
        neigh_dist, neigh_index = modelNN.kneighbors(
        film_reference,
        n_neighbors = 6)
        recommande = neigh_index[0][1:]
        films_reco = X.iloc[recommande]
        liste_film_reco = films_reco.index.to_list()   # On enregistre le nom des films recommandés dans une liste
        movie_to_show = to_show.loc[liste_film_reco].reset_index(drop=True)  # La liste nous permet d'afficher les films recommandés dans le DF avec les informations complètes
        st.title('')
        st.header("Nos recommandations : ")
        st.write(movie_to_show)
        email = "moncinema@creuse.fr"
        text_with_email = f"Envoyez vos recommandations à [{email}](mailto:{email})"
        st.markdown(text_with_email, unsafe_allow_html=True)
    except :
        st.write('Pas de film trouvé')
