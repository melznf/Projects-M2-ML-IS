"""PROJET MACHINE LEARNING
Mélissa Zennaf - Master 2 Mécen
Janvier 2021
"""

"""Ce module ainsi que toutes les fonctions qu'il contient est destiné à être utilisé au cours du projet de machine learning en Master 2 Mécen.
Ce projet contient trois phases :
- phase de web scraping
- phase de preprocessing
- phase d'apprentissage

La problématique de ce projet est la suivante : 
En tant que future jeune diplômée, je serai prochainement amenée à trouver du travail et ce sera certainement dans un autre départemet que l'Indre-et-Loire. Ma problématique est de savoir dans quel département français je suis susceptible de percevoir la rémunération la plus élevée. Existe-t-il des disparités de salaire entre les départements français ? Quels sont les déterminents et aboutissants de ces disparités ? Dans quel département s'installer pour maximiser sa rémunération ?

Ce module contiendra toutes les fonctions utilisées au cours de l'élaboration de la réponse à cette problématique. On commence par importer les librairies nécéssaires à la réalisation de chaque phase de ce projet.
"""




"""Librairies pour toutes les phases
"""
import pandas as pd
import numpy as np
from string import digits
import matplotlib.pyplot as plt


"""Librairies pour le web scraping
"""
from requests import get
from bs4 import BeautifulSoup as BS
import re

"""Librairies pour le preprocessing
"""
import sklearn
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import PolynomialFeatures

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif


"""Librairies pour le machine learning
"""
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, Lasso, LassoLars

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import validation_curve
from sklearn.model_selection import learning_curve

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.pipeline import Pipeline

from sklearn.metrics import max_error
from sklearn.metrics import r2_score
from sklearn.metrics import max_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error




"""Déclaration des listes pour le web scraping
"""

liste_ligne_nom=[]
liste_ligne_rmoyen=[]
liste_ligne_code1=[]
liste_ligne_nom6=[]
liste_ligne_decla=[]
liste_ligne_code2=[]
liste_ligne_nom7=[]
liste_ligne_taux_commun=[]
liste_ligne_code3=[]
liste_ligne_nom8=[]
liste_ligne_isf=[]
liste_ligne_code4=[]
liste_ligne_nom9=[]
liste_ligne_impot=[]
liste_ligne_code5=[]
liste_ligne_nom10=[]
liste_ligne_taxe_fonc=[]
liste_ligne_code6=[]
liste_ligne_nom11=[]
liste_ligne_taxe_om=[]
liste_ligne_code7=[]
liste_ligne_nom12=[]
liste_ligne_prlv_ent=[]
liste_ligne_code8=[]
liste_ligne_nom2=[]
liste_ligne_pib_2015=[]
liste_ligne_pib_2005=[]
liste_ligne_pib_2000=[]
liste_ligne_code=[]
liste_ligne_nom3=[]
liste_ligne_pop_1931=[]
liste_ligne_pop_1999=[]
liste_ligne_pop_2008=[]
liste_ligne_pop_2012=[]
liste_ligne_pop_2014=[]
liste_ligne_pop_2016=[]
liste_ligne_pop_2017=[]
liste_ligne_pop_2018=[]
liste_ligne_superficie=[]
liste_ligne_densite=[]
liste_ligne_nom4=[]
liste_ligne_code_2=[]
liste_ligne_nb_commune=[]
liste_ligne_pop_municipale=[]
liste_ligne_hab_m=[]
liste_ligne_taxe_fonc_bati=[]




""" Fonctions pour le web scraping
"""

def adresse(ad):
    """ Cette fonction prend en entrée un lien URL et renvoit le code html de la page. 
    Exemple :   adresse("https://fr.wikipedia.org/wiki/Rien")
    renvoit le code source de la page web correspondant à cet URL.
    """
    page_brute=get(ad)
    texte = page_brute.text
    texte = texte.replace("\n", " ")
    texte = texte.replace("\t", " ")
    motif = re.compile("<tr>\s*<td><a(.*?)</tr>")
    soupe = BS(texte, "lxml")
    return soupe




def get_pages(token, nb):
    """ Cette fonction prend en entrée un URL et un nombre entier.
    So but est de créer automatiquement des URL pour les cas où il y a plusieurs pages 
    sur lesquelles on souhaite faire du web scraping. Elle retourne une liste d'URL. 
    Exemple :   token = 'http://www.journaldunet.com/economie/impots/classement/departements/impot-revenu?page='
                pages = get_pages(token, 3)
                pages
                    ['http://www.journaldunet.com/economie/impots/classement/departements/impot-revenu?page=2',
                    'http://www.journaldunet.com/economie/impots/classement/departements/impot-revenu?page=3']
    """
    pages = []
    for i in range(2, nb+1):
        j = token + str(i)
        pages.append(j)
    return pages




def find_jdn(soupe):
    """ Cette fonction prend en entrée le code source d'une page web.
    Son but est de trouver les tableaux dans le code source d'une page web du site Journal du net. 
    Une fois cela fait, elle récupère le corps du tableau et enfin les lignes de celui-ci. Elle retourne
    toutes les lignes du tableau trouvé. 
    Exemple :  find_jdn(soupe) avec soupe l'output de la fonction adresse(ad) où ad est un URL. 
    """
    balise_table=soupe.find_all(name="table", attrs={"class" : ["odTable"]})
    table=balise_table[0]
    corps=table.find_next(name="tbody")
    lignes=corps.find_all(name="tr")
    ligne1,*_ = lignes
    return lignes




def find_wiki(soupe):
    """ Cette fonction prend en entrée le code source d'une page web.
    Son but est de trouver les tableaux dans le code source d'une page web du site Wikipédia. 
    Une fois cela fait, elle récupère le corps du tableau et enfin les lignes de celui-ci. Elle retourne
    toutes les lignes du tableau trouvé. Elle est utilisée pour la page 
    'https://fr.wikipedia.org/wiki/Liste_des_d%C3%A9partements_fran%C3%A7ais_class%C3%A9s_par_population_et_superficie'
    et la page 
    'https://fr.wikipedia.org/wiki/Nombre_de_communes_par_d%C3%A9partement_en_France_au_1er_janvier_2014'.
    Exemple :  find_jdn(soupe) avec soupe l'output de la fonction adresse(ad) où ad est un URL. 
    """
    balise_table=soupe.find_all(name="table", attrs={"class" : ["wikitable", "sortable"]})
    table1, table2=balise_table    
    corps =table2.find_next(name="tbody")
    lignes=corps.find_all(name="tr")[1:]
    return lignes




def find_wiki_2(soupe):
    """ Cette fonction prend en entrée le code source d'une page web.
    Son but est de trouver les tableaux dans le code source d'une page web du site Wikipédia. 
    Une fois cela fait, elle récupère le corps du tableau et enfin les lignes de celui-ci. Elle retourne
    toutes les lignes du tableau trouvé. 
    Elle est utilisée pour la page   
'https://fr.wikipedia.org/wiki/Liste_des_d%C3%A9partements_fran%C3%A7ais_class%C3%A9s_par_produit_int%C3%A9rieur_brut_par_habitant'
    Exemple :  find_jdn(soupe) avec soupe l'output de la fonction adresse(ad) où ad est un URL. 
    """
    balise_table2=soupe.find_all(name="table", attrs={"class" : ["wikitable", "sortable", "alternance"]})
    table2=balise_table2[0]
    corps =table2.find_next(name="tbody")
    lignes=corps.find_all(name="tr", attrs={"align" : ["right"]})
    return lignes




def gestion_ligne5(lignes):
    """ Cette fonction prend en entrée les lignes d'un tableau. 
    Son but est d'allonger chaque liste qui correspond à une colonne du tableau à chaque boucle. Elle nécessite la 
    déclaration des listes réalisées à la section du même nom. Elle retourne une liste de listes dans laquelle chaque liste 
    correspond à une colonne du tableau. 
    Exemple : gestion_ligne5(lignes) avec lignes=list(find_jdn(adresse(ad)) où ad est un URL.
    Cette fonction s'applique à l'URL 
    'http://www.journaldunet.com/business/salaire/classement/departements/salaires' uniquement.
    """
    for i in range(len(lignes)):
        colonnes = lignes[i].find_all(name="td")
        rang, nom, salaire = colonnes
        s=nom.text.replace('(','').replace(')', '')
        remove_digits = str.maketrans('', '', digits)
        res = s.translate(remove_digits)
        liste_ligne_nom.append(
        res.replace(' ', '').replace('', ' '))
        liste_ligne_code1.append(int(re.findall('\d+', nom.text)[0]))
        liste_ligne_rmoyen.append(
        salaire.text.rstrip().replace("\xa0", "").replace(" ", "").replace("€nets/mois", ""))
    return[liste_ligne_nom, liste_ligne_rmoyen, liste_ligne_code1]
    
        

        
def gestion_ligne9(lignes):
    """ Cette fonction prend en entrée les lignes d'un tableau. 
    Son but est d'allonger chaque liste qui correspond à une colonne du tableau à chaque boucle. Elle nécessite la 
    déclaration des listes réalisées à la section du même nom. Elle retourne une liste de listes dans laquelle chaque liste 
    correspond à une colonne du tableau. 
    Cette fonction s'applique à l'URL 
    'http://www.journaldunet.com/economie/impots/classement/departements/revenu-fiscal' et
    'http://www.journaldunet.com/economie/impots/classement/departements/revenu-fiscal?page=2'.
    Exemple :   lignes4=liste_lignes_jdn("http://www.journaldunet.com/economie/impots/classement/departements/revenu-fiscal")
                lignes42=liste_lignes_jdn("http://www.journaldunet.com/economie/impots/classement/departements/revenu-fiscal?page=2")
                gestion_ligne9(lignes4)
                res1=gestion_ligne9(lignes42).
    """
    for i in range(len(lignes)):
        colonnes = lignes[i].find_all(name="td")
        rang, nom, decla = colonnes
        s=nom.text.replace('(','').replace(')', '')
        remove_digits = str.maketrans('', '', digits)
        res = s.translate(remove_digits)
        liste_ligne_nom6.append(
        res.replace(' ', '').replace('', ' '))
        liste_ligne_code2.append(int(re.findall('\d+', nom.text)[0]))
        liste_ligne_decla.append(
        decla.text.rstrip().replace("\xa0", "").replace(" ", "").replace("€", ""))
    return[liste_ligne_nom6,liste_ligne_decla, liste_ligne_code2]
        
      
        
        
def gestion_ligne12(lignes):
    """ Cette fonction prend en entrée les lignes d'un tableau. 
    Son but est d'allonger chaque liste qui correspond à une colonne du tableau à chaque boucle. Elle nécessite la 
    déclaration des listes réalisées à la section du même nom. Elle retourne une liste de listes dans laquelle chaque liste 
    correspond à une colonne du tableau. 
    Cette fonction s'applique à l'URL 
    'http://www.journaldunet.com/economie/impots/classement/departements/impot-revenu' et
    'http://www.journaldunet.com/economie/impots/classement/departements/impot-revenu?page=2' et
    'http://www.journaldunet.com/economie/impots/classement/departements/impot-revenu?page=3'.
    Exemple :   lignes8=list(find_jdn(adresse("http://www.journaldunet.com/economie/impots/classement/departements/impot-revenu")))
                gestion_ligne12(lignes8)
                token = 'http://www.journaldunet.com/economie/impots/classement/departements/impot-revenu?page='
                pages = get_pages(token,3)
                for i in range(len(pages)) :
                    lignes82=list(find_jdn(adresse(pages[i])))
                    res8=gestion_ligne12(lignes82)
    """
    for i in range(len(lignes)):
        colonnes = lignes[i].find_all(name="td")
        rang, nom, impot = colonnes
        s=nom.text.replace('(','').replace(')', '')
        remove_digits = str.maketrans('', '', digits)
        res = s.translate(remove_digits)
        liste_ligne_nom9.append(
        res.replace(' ', '').replace('', ' '))
        liste_ligne_code5.append(int(re.findall('\d+', nom.text)[0]))
        liste_ligne_impot.append(
        impot.text.rstrip().replace("\xa0", "").replace(" ", "").replace("€", ""))
    return[liste_ligne_nom9,
        liste_ligne_impot,
        liste_ligne_code5]
        
    
        
        
def gestion_ligne14(lignes):
    """ Cette fonction prend en entrée les lignes d'un tableau. 
    Son but est d'allonger chaque liste qui correspond à une colonne du tableau à chaque boucle. Elle nécessite la 
    déclaration des listes réalisées à la section du même nom. Elle retourne une liste de listes dans laquelle chaque liste 
    correspond à une colonne du tableau. 
    Cette fonction s'applique à l'URL 
    'http://www.journaldunet.com/economie/impots/classement/departements/taxe-fonciere-bati' et
    'http://www.journaldunet.com/economie/impots/classement/departements/taxe-fonciere-bati?page=2'.
    Exemple :   lignes10=list(find_jdn(adresse("http://www.journaldunet.com/economie/impots/classement/departements/taxe-fonciere-bati")))
                lignes102=list(find_jdn(adresse("http://www.journaldunet.com/economie/impots/classement/departements/taxe-fonciere-bati?page=2")))
                gestion_ligne14(lignes10)
                res6=gestion_ligne14(lignes102)
    """
    for i in range(len(lignes)):
        colonnes = lignes[i].find_all(name="td")
        rang, nom, taxe_om = colonnes
        s=nom.text.replace('(','').replace(')', '')
        remove_digits = str.maketrans('', '', digits)
        res = s.translate(remove_digits)
        liste_ligne_nom11.append(
        res.replace(' ', '').replace('', ' '))
        liste_ligne_code7.append(int(re.findall('\d+', nom.text)[0]))
        liste_ligne_taxe_om.append(
        taxe_om.text.rstrip().replace("\xa0", "").replace(" ", "").replace("%", ""))
    return[liste_ligne_nom11,
        liste_ligne_taxe_om,
        liste_ligne_code7]
    
    


def gestion_ligne6(lignes):
    """ Cette fonction prend en entrée les lignes d'un tableau. 
    Son but est d'allonger chaque liste qui correspond à une colonne du tableau à chaque boucle. Elle nécessite la 
    déclaration des listes réalisées à la section du même nom. Elle retourne une liste de listes dans laquelle chaque liste 
    correspond à une colonne du tableau. 
    Cette fonction s'applique à l'URL 
    'https://fr.wikipedia.org/wiki/Liste_des_d%C3%A9partements_fran%C3%A7ais
    _class%C3%A9s_par_produit_int%C3%A9rieur_brut_par_habitant'.
    Exemple :   lignes3=list(find_wiki(adresse("https://fr.wikipedia.org/wiki/Liste_des_d%C3%A9partements_fran%C3%A7ais
    _class%C3%A9s_par_produit_int%C3%A9rieur_brut_par_habitant")))
                res3=gestion_ligne6(lignes3)
    """
    for i in range(len(lignes)):
        colonnes = lignes[i].find_all(name="td")
        rang1, rang2, rang3, nom, pib_2015, pib_2005, pib_2000, progression_pib, progression_rang =colonnes
        liste_ligne_nom2.append(nom.findNext(name="a").text.replace('', ' ')),
        liste_ligne_pib_2015.append(pib_2015.text.replace(' ', '')),
        liste_ligne_pib_2005.append(pib_2005.text.replace(' ', '')),
        liste_ligne_pib_2000.append(pib_2000.text.replace(' ', ''))
    return[liste_ligne_nom2,
        liste_ligne_pib_2015,
        liste_ligne_pib_2005,
        liste_ligne_pib_2000]
        
        
        

def gestion_ligne7(lignes):
    """ Cette fonction prend en entrée les lignes d'un tableau. 
    Son but est d'allonger chaque liste qui correspond à une colonne du tableau à chaque boucle. Elle nécessite la 
    déclaration des listes réalisées à la section du même nom. Elle retourne une liste de listes dans laquelle chaque liste 
    correspond à une colonne du tableau. 
    Cette fonction s'applique à l'URL 
    'https://fr.wikipedia.org/wiki/Liste_des_d%C3%A9partements_fran%C3%A7ais
_class%C3%A9s_par_population_et_superficie'. 
    Exemple :   lignes4=list(find_wiki_2(adresse("https://fr.wikipedia.org/wiki/Liste_des_d%C3%A9partements_fran%C3%A7ais
_class%C3%A9s_par_population_et_superficie")))
                del lignes4[3]
                res4=gestion_ligne7(lignes4)
    Cette fonction permer de gérer le tableau de la page web alors que celui-ci présente des données manquantes et des cellules 
    fusionnées. 
    """
    for i in range(len(lignes)):
        if i == 11 or i==54 :
            colonnes1 = lignes[i].find_all(name="td")
            rang, code, nom, rien, pop_2012, pop_2014, pop_2016, pop_2017, pop_2018, superficie, densite = colonnes1
            liste_ligne_code.append(code.text),
            liste_ligne_nom3.append(nom.find_next(name="a").text.replace('', ' ')),
            liste_ligne_pop_2012.append(pop_2012.text.rstrip().replace("\xa0", "")),
            liste_ligne_pop_2014.append(pop_2014.text.rstrip().replace("\xa0", "")),
            liste_ligne_pop_2016.append(pop_2016.text.rstrip().replace("\xa0", "")),
            liste_ligne_pop_2017.append(pop_2017.text.rstrip().replace("\xa0", "")),
            liste_ligne_pop_2018.append(pop_2017.text.rstrip().replace("\xa0", "")),
            liste_ligne_superficie.append(superficie.text.rstrip().replace("\xa0", "")),
            liste_ligne_densite.append(densite.text.rstrip().replace("\xa0",""))
        if i == 82 :
            colonnes2 = lignes[i].find_all(name="td")
            rang, code, nom, pop_1931, pop_1999, pop_2008, pop_2012, rien, pop_2017, pop_2018, superficie, densite = colonnes2
            liste_ligne_code.append(code.text),
            liste_ligne_nom3.append(nom.find_next(name="a").text.replace('', ' ')),
            liste_ligne_pop_2012.append(pop_2012.text.rstrip().replace("\xa0", "")),
            liste_ligne_pop_2014.append(' '),
            liste_ligne_pop_2016.append(' '),
            liste_ligne_pop_2017.append(pop_2017.text.rstrip().replace("\xa0", "")),
            liste_ligne_pop_2018.append(pop_2017.text.rstrip().replace("\xa0", "")),
            liste_ligne_superficie.append(superficie.text.rstrip().replace("\xa0", "")),
            liste_ligne_densite.append(densite.text.rstrip().replace("\xa0",""))
    
        elif i !=11 and i!=54 and i!=82:
            colonnes3 = lignes[i].find_all(name="td")
            rang, code, nom, pop_1931, pop_1999, pop_2008, pop_2012, pop_2014, pop_2016, pop_2017,pop_2018, superficie, densite = colonnes3
            liste_ligne_code.append(code.text),
            liste_ligne_nom3.append(nom.find_next(name="a").text.replace('', ' ')),
            liste_ligne_pop_2012.append(pop_2012.text.rstrip().replace("\xa0", "")),
            liste_ligne_pop_2014.append(pop_2014.text.rstrip().replace("\xa0", "")),
            liste_ligne_pop_2016.append(pop_2016.text.rstrip().replace("\xa0", "")),
            liste_ligne_pop_2017.append(pop_2017.text.rstrip().replace("\xa0", "")),
            liste_ligne_pop_2018.append(pop_2017.text.rstrip().replace("\xa0", "")),
            liste_ligne_superficie.append(superficie.text.rstrip().replace("\xa0", "")),
            liste_ligne_densite.append(densite.text.rstrip().replace("\xa0",""))
    return[liste_ligne_code,
        liste_ligne_nom3,
        liste_ligne_pop_2012,
        liste_ligne_pop_2014,
        liste_ligne_pop_2016,
        liste_ligne_pop_2017,
        liste_ligne_pop_2018,
        liste_ligne_superficie,
        liste_ligne_densite]
        
    
    
      
def gestion_ligne8(lignes):
    """ Cette fonction prend en entrée les lignes d'un tableau. 
    Son but est d'allonger chaque liste qui correspond à une colonne du tableau à chaque boucle. Elle nécessite la 
    déclaration des listes réalisées à la section du même nom. Elle retourne une liste de listes dans laquelle chaque liste 
    correspond à une colonne du tableau. 
    Cette fonction s'applique à l'URL 
    'https://fr.wikipedia.org/wiki/Nombre_de_communes_par_d%C3%A9partement_en_France_au_1er_janvier_2014'
    Exemple :   lignes5=list(find_wiki_2(adresse("https://fr.wikipedia.org/wiki/Nombre_de_communes_par_d%C3%A9partement_en_France
    _au_1er_janvier_2014")))
                res5=gestion_ligne8(lignes5)
    """
    for i in range(len(lignes)):
        colonnes = lignes[i].find_all(name="td")
        rang, code, nom, nb_commune, pop_municipale, hab_m = colonnes
        liste_ligne_code_2.append(code.text),
        liste_ligne_nom4.append(nom.find_next(name="a").text.replace('', ' ')),
        liste_ligne_nb_commune.append(nb_commune.text.rstrip().replace("\xa0", "")),
        liste_ligne_pop_municipale.append(pop_municipale.text.rstrip().replace("\xa0", "")),
        liste_ligne_hab_m.append(hab_m.text.rstrip().replace("\xa0", ""))  
    return[liste_ligne_nom4,
        liste_ligne_code_2,
        liste_ligne_nb_commune,
        liste_ligne_pop_municipale,
        liste_ligne_hab_m]
        
    
    
    
def construit_df(res1, res2, res3, res4, res5, res6, res8):
    """Cette fonction prend en entrée des listes des listes de listes obtenues grâce
    aux fonctions précédentes de la section web scraping.
    Elle permet de fusionner plusieurs sources sous la forme d'un data frame pandas.
    Exemple :   construit_df(res1, res2, res3, res4, res5)
    renvoit :   
    """
    columns1=['nom','code', 'nb_commune', 'pop_municipale', 'hab_m']
    columns4=['code', 'nom', 'pop_2012', 'pop_2014', 'pop_2016', 'pop_2017', 'pop_2018', 'superficie', 'densite']
    columns3=['nom','pib_2015', 'pib_2005', 'pib_2000']
    columns5=['nom', 'décla', 'code']
    columns2=['nom','rmoyen', 'code']
    columns7=['nom', 'isf', 'code']
    columns9=[ 'nom', 'taxe_fonc_bati', 'code']
    columns8=['nom', 'impot', 'code']
    columns10=['nom', 'taxe_om', 'code']
    dt1=np.matrix(res5).T
    dt2=np.matrix(res4).T
    dt3=np.matrix(res3).T
    dt4=np.matrix(res2).T
    dt5=np.matrix(res1).T
    dt6=np.matrix(res6).T
    dt7=np.matrix(res8).T
    df1=pd.DataFrame(dt1, columns=columns1)
    df2=pd.DataFrame(dt2, columns=columns4)
    df3=pd.DataFrame(dt3, columns=columns3)
    df4=pd.DataFrame(dt4, columns=columns5)
    df5=pd.DataFrame(dt5, columns=columns2)
    df6=pd.DataFrame(dt6, columns=columns10)
    df7=pd.DataFrame(dt7, columns=columns8)
    df0=df2.merge(df1, on=['code', 'nom'])
    df=df0.merge(df3, on=['nom'])
    df=df.merge(df4, on=['nom'])
    df=df.merge(df5, on=['nom'])
    df=df.merge(df6, on=['nom', 'code'])
    df=df.merge(df7, on=['nom', 'code'])
    return df
    
    
    

""" Fonctions pour le preprocessing
"""

def convert_float(df):
    """ Cette fonction prend comme argument un data frame pandas.
    Cette fonction permet de transformer des chiffres à virgules en float dans les colonnes d'un data frame pandas.
    On commence par remplacer les virgules par des points puis on convertit en float64.
    Exemple :   convert_float(df)
    convertit les colonnes densite, suerficie et taxe_om.
    """
    df['densite'] = [x.replace(',', '.') for x in df['densite']]
    df['densite'] = df['densite'].astype(float)
    df['superficie'] = [x.replace(',', '.') for x in df['superficie']]
    df['superficie'] = df['superficie'].astype(float)
    df['taxe_om'] = [x.replace(',', '.') for x in df['taxe_om']]
    df['taxe_om'] = df['taxe_om'].astype(float)
    return df




def drop_inutile(df):
    """Cette fonction prend comme argument un data frame pandas.
    Elle permet de se débarasser des variables inutiles ou redondants.
    Exemple :   drop_inutile(df)
    renvoit le data frame sans les colonnes pib_2015, code_y, code_x et code.
    """
    df.drop(['pib_2015'], axis='columns', inplace=True)
    df.drop(['code_y'], axis='columns', inplace=True)
    df.drop(['code_x'], axis='columns', inplace=True)
    df.drop(['code'], axis='columns', inplace=True)
    return df




def select_features_kbest(df):
    """Cette fonction prend en argument un data frame pandas.
    Elle permet de sélectionner les sept variables qui ont le plus d'importance selon le score f_classif.
    Exemple :   select_features_kbest(df)
    renvoit les scores obtenus par chaque variable selon le sélecteur, les variables selectionnées et 
    renvoit une graphique matplotlib.pyplot représentant l'importance des variables."""
    X=['pop_2018', 'superficie' , 'densite', 'nb_commune', 'pop_municipale', 'hab_m', 'pib_2005',  'pib_2000', 'décla',  'taxe_om', 'impot']
    y=['rmoyen']
    u=df[X]
    selector = SelectKBest(f_classif, k=7)
    selector.fit(df[X], np.ravel(df[y]))
    scores = -np.log10(selector.pvalues_)
    plt.bar(range(len(X)), scores,color='green')
    plt.xticks(range(len(X)), X, rotation='vertical')
    plt.show()
    return scores, selector



    
def data_frame_final(selector, df):
    """Cette fonction prend en argument un selecteur du type SelectKBest() et un data frame pandas. 
    Il permet de retrouver les indices des variables sélectionnées par le sélecteur et de recréer un data frame pandas.
    Exemple :   selector = SelectKBest(f_classif, k=7)
                selector.fit(df[X], np.ravel(df[y]))
                data_frame_final(selector, df)
    Renvoit un data frame pandas avec seulement les variables sélectionnées par le le sélecteur.
    """
    X=['pop_2018', 'superficie' , 'densite', 'nb_commune', 'pop_municipale', 'hab_m', 'pib_2005',  'pib_2000', 'décla',  'taxe_om', 'impot']
    y=['rmoyen']
    u=df[X]
    f = selector.get_support(1)
    df_new = u[u.columns[f]]
    v=df['rmoyen']
    w=df['nom']
    df_new['revenu_moyen']=v
    df_new['departement']=w
    df_new=df_new.set_index('departement')
    return df_new




def robuste_std(df):
    """Cette fonction prend en entrée un data frame pandas. 
    Elle permet de transformer les variables 'pop_2018', 'superficie' , 'densite', 'nb_commune', 'pop_municipale', 'hab_m', 'pib_2005',  'pib_2000', 'décla',  'taxe_om', 'impot', 'rmoyen' selon le transformateur RobustScaler. 
    Exemple :   robuste_std(df)
    renvoit le data frame avec les variables précédentes transformées.
    """
    df[['pop_2018', 'superficie' , 'densite', 'nb_commune', 'pop_municipale', 'hab_m', 'pib_2005',  'pib_2000', 'décla',  'taxe_om', 'impot', 'rmoyen']] = RobustScaler().fit_transform(df[['pop_2018', 'superficie' , 'densite', 'nb_commune', 'pop_municipale', 'hab_m', 'pib_2005',  'pib_2000', 'décla',  'taxe_om', 'impot', 'rmoyen']])
    return pd.DataFrame(df)




def melange(df):
    """ Cette fonction prend en entrée un data frame pandas. 
    Elle permet de mélanger les lignes du data frame de manière aléatoire. 
    Exemple :   melange(df)
    renvoit le data frame mélangé de manière aléatoire. 
    """
    df_shuffled=sklearn.utils.shuffle(df, random_state=4)
    return df_shuffled




def outliers(df):
    """ Cette fonction prend en entrée un data frame pandas.
    Elle pemet de supprimer la ligne du data frame pour laquelle la variable hab_m est maximale.
    Exemple :   outliers(df)
    renvoit le data frame sans la ligne contenant la valeur maximale de la colonne hab_m.
    """
    indexNames1 = df[ df['hab_m'] == df['hab_m'].max() ].index
    df.drop(indexNames1 , inplace=True)
    return df




def multiplot(df):
    """ Cette fonction prend en entrée un data frame pandas. 
    Elle permet de réaliser neuf scatterplots représentant les variables du data frame en fonction de la variable rmoyen.
    """
    plt.subplot(3, 3, 1)
    plt.scatter(df['pop_2018'],df['rmoyen'], label='pop2018',color='green')
    plt.legend()
    plt.subplot(3, 3, 2)
    plt.scatter(df['superficie'],df['rmoyen'], label='superficie',color='green')
    plt.legend()
    plt.subplot(3, 3, 3)
    plt.scatter(df['nb_commune'],df['rmoyen'], label='nb_commune',color='green')
    plt.legend()
    plt.subplot(3, 3, 4)
    plt.scatter(df['densite'],df['rmoyen'], label='densite',color='green')
    plt.legend()
    plt.subplot(3, 3, 5)
    plt.scatter(df['pop_municipale'],df['rmoyen'], label='pop_municipale',color='green')
    plt.legend()
    plt.subplot(3, 3, 6)
    plt.scatter(df['hab_m'],df['rmoyen'], label='hab_m',color='green')
    plt.legend()
    plt.subplot(3, 3, 7)
    plt.scatter(df['pib_2005'],df['rmoyen'], label='pib_2005',color='green')
    plt.legend()
    plt.subplot(3, 3, 8)
    plt.scatter(df['décla'],df['rmoyen'], label='decla',color='green')
    plt.legend()
    plt.subplot(3, 3, 9)
    plt.scatter(df['impot'],df['rmoyen'], label='impot',color='green')
    plt.legend()
    
    
    
    
def multiplot2(df):
    """ Cette fonction prend en entrée un data frame pandas. 
    Elle permet de réaliser sept scatterplots représentant les variables du data frame en fonction de la variable revenu_moyen.
    """
    plt.subplot(2, 4, 1)
    plt.scatter(df['pop_2018'],df['revenu_moyen'], label='pop2018',color='green')
    plt.legend()
    plt.subplot(2, 4, 2)
    plt.scatter(df['pop_municipale'],df['revenu_moyen'], label='pop_municipale',color='green')
    plt.legend()
    plt.subplot(2, 4, 3)
    plt.scatter(df['densite'],df['revenu_moyen'], label='densite',color='green')
    plt.legend()
    plt.subplot(2, 4, 4)
    plt.scatter(df['hab_m'],df['revenu_moyen'], label='hab_m',color='green')
    plt.legend()
    plt.subplot(2, 4, 5)
    plt.scatter(df['pib_2000'],df['revenu_moyen'], label='pib_2000',color='green')
    plt.legend()
    plt.subplot(2, 4, 6)
    plt.scatter(df['décla'],df['revenu_moyen'], label='decla',color='green')
    plt.legend()
    plt.subplot(2, 4, 7)
    plt.scatter(df['impot'],df['revenu_moyen'], label='impot',color='green')
    plt.legend()
    
  


""" Fonctions utilisées dans la phase de machine learning
""" 
    
def analyse(modele, data):
    """ Cette fonction prend en argument le nom d'un modèle entre guillemet et un data frame du type pandas.
    Elle permet de renvoyer différentes métriques pour chaque modèle et de fournir la meilleure combinaison 
    de paramètres parmis celles proposées dans le dictionnaire propre au modèle choisi.
    Les modèles diponibles pour cette fonction sont : Lin_reg (régression linéaire), Ridge (régression Ridge), 
    Lasso (régression Lasso), LassoLars (régression Lasso LARS), GradBoost (boosting du gradient), RandF
    (forêt aléatoire), Bagging.
    Exemple :   analyse('Lin_reg', data)
    renvoit     NMSE : -0.4961859977838066
                MaxErr : -2.104349955890469
                R2 : 0.5417844488555927
                NMAE : -0.24805443260748067
                Somme des résidus : [1.98729921e-14]
                Moyenne des résidus : 3.090120751873352e-16
    On peut aussi stocker le modèle, les meilleurs paramères du modèle le meilleur score du modèle, 
    la liste des résidus et le score sur les données d'entraînement.
    """
    X=['pop_2018' , 'densite', 'pop_municipale', 'hab_m', 'pib_2000', 'décla','impot']
    y=['revenu_moyen']
    X_train, X_test, y_train, y_test = train_test_split(data[X], data[y], test_size=0.33, random_state=42)
    if (modele=='Ridge')==True:
        param_grid2={'alpha':[0.01, 0.05, 0.1, 0.5, 1],
              'fit_intercept': ['True',  'False'],
              'normalize': ['True',  'False'],
              'copy_X': ['True',  'False'],
               'max_iter':[500, 1000, 1500],
               'solver':['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']}
        clf2 = GridSearchCV(Ridge(), param_grid2, cv=5, scoring='neg_mean_absolute_error')
        clf2.fit(X_train, np.ravel(y_train))
        NMSE2=cross_val_score(clf2, X_train, np.ravel(y_train), cv=2, scoring='neg_mean_squared_error')
        print(f"NMSE : {np.mean(NMSE2)}")
        MaxErr2=cross_val_score(clf2, X_train, np.ravel(y_train), cv=2, scoring='max_error')
        print(f"MaxErr : {np.mean(MaxErr2)}")
        Rsq2=cross_val_score(clf2, X_train, np.ravel(y_train), cv=2, scoring='r2')
        print(f"R2 : {np.mean(Rsq2)}")
        print(f"NMAE : {clf2.score(X_train, y_train)}")
        pred2=clf2.predict(X_train)
        array_test2=np.array(y_train)
        resid2=[]
        for i in range(len(pred2)):
            residuals2=array_test2[i]-pred2[i]
            resid2.append(np.asarray(residuals2))
        print(f"Somme des résidus : {sum(resid2)}")
        print(f"Moyenne des résidus : {np.mean(resid2)}")
        return  clf2, clf2.best_params_, clf2.best_score_, resid2, clf2.score(X_train, y_train)
    
    elif (modele=='Lin_reg')==True:
        param_grid={'scaler__interaction_only':['False','True'],
                    'scaler__degree':[1, 2],
           'lin_reg__normalize':['False','True'],
           'scaler__include_bias':['False','True'],
           'lin_reg__fit_intercept':['False', 'True']}
        clf = GridSearchCV(Pipeline([('scaler', PolynomialFeatures()), ('lin_reg', LinearRegression())]),param_grid, cv=5,scoring='neg_mean_absolute_error')
        clf.fit(X_train, np.ravel(y_train))
        NMSE=cross_val_score(clf, X_train, np.ravel(y_train), cv=2, scoring='neg_mean_squared_error')
        print(f"NMSE : {np.mean(NMSE)}")
        MaxErr=cross_val_score(clf, X_train, np.ravel(y_train), cv=2, scoring='max_error')
        print(f"MaxErr : {np.mean(MaxErr)}")
        Rsq=cross_val_score(clf, X_train, np.ravel(y_train), cv=2, scoring='r2')
        print(f"R2 : {np.mean(Rsq)}")
        print(f"NMAE : {clf.score(X_train, y_train)}")
        pred=clf.predict(X_train)
        array_test=np.array(y_train)
        resid=[]
        for i in range(len(pred)):
            residuals=array_test[i]-pred[i]
            resid.append(np.asarray(residuals))
        print(f"Somme des résidus : {sum(resid)}")
        print(f"Moyenne des résidus : {np.mean(resid)}")
        return  clf, clf.best_params_, clf.best_score_, resid, clf.score(X_train, y_train)
    
    elif (modele=='Lasso')==True:
        param_grid3={'fit_intercept':['False','True'],
           'alpha':[0.01, 0.05, 0.1, 0.5, 1],
           'normalize':['False','True'],
           'positive':['False', 'True'],
            'random_state':[0],
            'selection':['cyclic', 'random'],
            'copy_X':['False','True'],
            'warm_start':['False','True']}
        clf3 = GridSearchCV(Lasso(), param_grid3, cv=5,scoring='neg_mean_absolute_error')
        clf3.fit(X_train, np.ravel(y_train))
        NMSE3=cross_val_score(clf3, X_train, np.ravel(y_train), cv=2, scoring='neg_mean_squared_error')
        print(f"NMSE : {np.mean(NMSE3)}")
        MaxErr3=cross_val_score(clf3, X_train, np.ravel(y_train), cv=2, scoring='max_error')
        print(f"MaxErr : {np.mean(MaxErr3)}")
        Rsq3=cross_val_score(clf3, X_train, np.ravel(y_train), cv=2, scoring='r2')
        print(f"R2 : {np.mean(Rsq3)}")
        print(f"NMAE : {clf3.score(X_train, y_train)}")
        pred3=clf3.predict(X_train)
        array_test3=np.array(y_train)
        resid3=[]
        for i in range(len(pred3)):
            residuals3=array_test3[i]-pred3[i]
            resid3.append(np.asarray(residuals3))
        print(f"Somme des résidus : {sum(resid3)}")
        print(f"Moyenne des résidus : {np.mean(resid3)}")
        return  clf3, clf3.best_params_, clf3.best_score_, resid3, clf3.score(X_train, y_train)
    
    elif (modele=='LassoLars')==True:
        param_grid4={'copy_X':['False','True'],
           'fit_intercept':['False','True'],
           'fit_path':['False','True'],
           'random_state':[0],
           'alpha':[0.01, 0.05, 0.1, 0.5, 1],
           'jitter':[0.01, 0.05, 0.1, 0.5, 1],
           'max_iter':[1000, 2000, 3000],
           'normalize':['True', 'False']}
        clf4 = GridSearchCV(LassoLars(), param_grid4, cv=5,scoring='neg_mean_absolute_error')
        clf4.fit(X_train, np.ravel(y_train))
        NMSE4=cross_val_score(clf4, X_train, np.ravel(y_train), cv=2, scoring='neg_mean_squared_error')
        print(f"NMSE : {np.mean(NMSE4)}")
        MaxErr4=cross_val_score(clf4, X_train, np.ravel(y_train), cv=2, scoring='max_error')
        print(f"MaxErr : {np.mean(MaxErr4)}")
        Rsq4=cross_val_score(clf4, X_train, np.ravel(y_train), cv=2, scoring='r2')
        print(f"R2 : {np.mean(Rsq4)}")
        print(f"NMAE : {clf4.score(X_train, y_train)}")
        pred4=clf4.predict(X_train)
        array_test4=np.array(y_train)
        resid4=[]
        for i in range(len(pred4)):
            residuals4=array_test4[i]-pred4[i]
            resid4.append(np.asarray(residuals4))
        print(f"Somme des résidus : {sum(resid4)}")
        print(f"Moyenne des résidus : {np.mean(resid4)}")
        return clf4, clf4.best_params_, clf4.best_score_, resid4, clf4.score(X_train, y_train)

    elif (modele=='GradBoost')==True:
        param_grid5={'n_estimators': [5, 8, 10, 25, 50],
          'max_depth': [2, 3, 4],
          'min_samples_split': [2, 3, 4],
          'learning_rate': [0.01, 0.05, 0.1, 0.5, 1]}
        clf5 = GridSearchCV(GradientBoostingRegressor(), param_grid5, cv=5,scoring='neg_mean_absolute_error')
        clf5.fit(X_train, np.ravel(y_train))
        NMSE5=cross_val_score(clf5, X_train, np.ravel(y_train), cv=2, scoring='neg_mean_squared_error')
        print(f"NMSE : {np.mean(NMSE5)}")
        MaxErr5=cross_val_score(clf5, X_train, np.ravel(y_train), cv=2, scoring='max_error')
        print(f"MaxErr : {np.mean(MaxErr5)}")
        Rsq5=cross_val_score(clf5, X_train, np.ravel(y_train), cv=2, scoring='r2')
        print(f"R2 : {np.mean(Rsq5)}")
        print(f"NMAE : {clf5.score(X_train, y_train)}")
        pred5=clf5.predict(X_train)
        array_test5=np.array(y_train)
        resid5=[]
        for i in range(len(pred5)):
            residuals5=array_test5[i]-pred5[i]
            resid5.append(np.asarray(residuals5))
        print(f"Somme des résidus : {sum(resid5)}")
        print(f"Moyenne des résidus : {np.mean(resid5)}")
        return  clf5, clf5.best_params_, clf5.best_score_, resid5, clf5.score(X_train, y_train)
    
    elif (modele=='RandF')==True:
        param_grid6={'n_estimators': [5, 8, 10, 25, 50],
          'max_depth': [2, 3],
          #'criterion':['mse', 'mae'],
          #'min_samples_leaf': [2, 3, 4],
          'max_features':[2, 5],
          #'max_leaf_nodes': [2, 3, 4],
          'min_samples_split': [2, 3, 4],
          'bootstrap': ['True', 'False'],
          'max_samples':[2, 3, 4],
          'warm_start':['False','True']}
        clf6 = GridSearchCV(RandomForestRegressor(), param_grid6, cv=3,scoring='neg_mean_absolute_error')
        clf6.fit(X_train, np.ravel(y_train))
        print(f"NMAE : {clf6.score(X_train, y_train)}")
        pred6=clf6.predict(X_train)
        array_test6=np.array(y_train)
        NMSE6=mean_squared_error(y_train, pred6)
        print(f"NMSE : {NMSE6}")
        Rsq6=r2_score(y_train, pred6)
        print(f"R2 : {Rsq6}")
        MaxErr6=max_error(y_train, pred6)
        print(f"MaxError : {MaxErr6}")
        resid6=[]
        for i in range(len(pred6)):
            residuals6=array_test6[i]-pred6[i]
            resid6.append(np.asarray(residuals6))
        print(f"Somme des résidus : {sum(resid6)}")
        print(f"Moyenne des résidus : {np.mean(resid6)}")
        return  clf6, clf6.best_params_, clf6.best_score_, resid6, clf6.score(X_train, y_train)
    
    elif (modele=='AdaBoost')==True:
        param_grid7={'n_estimators': [5, 8, 10, 25, 50],
          'base_estimator':[DecisionTreeRegressor(), LinearRegression()],
          'learning_rate': [0.01, 0.05, 0.1, 0.5, 1],
          'loss':['linear', 'square', 'exponential']}
        clf7 = GridSearchCV(AdaBoostRegressor(), param_grid7, cv=5,scoring='neg_mean_absolute_error')
        clf7.fit(X_train, np.ravel(y_train))
        NMSE7=cross_val_score(clf7, X_train, np.ravel(y_train), cv=2, scoring='neg_mean_squared_error')
        print(f"NMSE : {np.mean(NMSE7)}")
        MaxErr7=cross_val_score(clf7, X_train, np.ravel(y_train), cv=2, scoring='max_error')
        print(f"MaxErr : {np.mean(MaxErr7)}")
        Rsq7=cross_val_score(clf7, X_train, np.ravel(y_train), cv=2, scoring='r2')
        print(f"R2 : {np.mean(Rsq7)}")
        print(f"NMAE : {clf7.score(X_train, y_train)}")
        pred7=clf7.predict(X_train)
        array_test7=np.array(y_train)
        resid7=[]
        for i in range(len(pred7)):
            residuals7=array_test7[i]-pred7[i]
            resid7.append(np.asarray(residuals7))
        print(f"Somme des résidus : {sum(resid7)}")
        print(f"Moyenne des résidus : {np.mean(resid7)}")
        return  clf7, clf7.best_params_, clf7.best_score_, resid7, clf7.score(X_train, y_train)
    
    elif (modele=='Bagging')==True:
        param_grid8={'warm_start': ['False', 'True'],
          'base_estimator':[DecisionTreeRegressor(), LinearRegression()],
          'n_estimators':[5, 8, 10, 25, 50],
        'bootstrap':['True', 'False'],
          'max_features': [0.5, 1],
          'n_jobs':[1, 2, 3],
          'random_state':[0]}
        clf8 = GridSearchCV(BaggingRegressor(), param_grid8, cv=3,scoring='neg_mean_absolute_error')
        clf8.fit(X_train, np.ravel(y_train))
        pred8=clf8.predict(X_train)
        array_test8=np.array(y_train)
        NMSE8=mean_squared_error(y_train, pred8)
        print(f"NMSE : {NMSE8}")
        Rsq8=r2_score(y_train, pred8)
        print(f"R2 : {Rsq8}")
        MaxErr8=max_error(y_train, pred8)
        print(f"MaxError : {MaxErr8}")
        print(f"NMAE : {clf8.score(X_train, y_train)}")
        resid8=[]
        for i in range(len(pred8)):
            residuals8=array_test8[i]-pred8[i]
            resid8.append(np.asarray(residuals8))
        print(f"Somme des résidus : {sum(resid8)}")
        print(f"Moyenne des résidus : {np.mean(resid8)}")
        return  clf8, clf8.best_params_, clf8.best_score_, resid8, clf8.score(X_train, y_train)
    
    else:
        print('Veuillez renseigner un modèle parmi ceux disponibles : Lin_reg, Ridge, Lasso, LassoLars, GradBoost, RandF, Bagging')
        
        
        
        
def verif_overfitting(modele, mod, data):
    """ La fonction prend en paramètres le nom d'un modèle, un modèle, un data frame pandas.
    Cette fonction permet de stocker les scores sur les échantillons d'entraînement et de validation.
    Elle génère également une plot des scores sur les deux échantillons. On peut grapiquement repérer la présence
    de surapprentissage.
    Exemple :   train_scores_ridge, valid_scores_ridge = verif_overfitting('Ridge', mod_ridge.best_estimator_, data)
    avec mod_ridge le modèle obtenu suite à l'éxecution de la fonction GridSearchCV.
    Elle renvoit un graphique avec les scores sur les échantillons de validation et d'entraînement. Les modèles disponibles
    sont : Ridge, Lasso et Bagging.
    """
    X=['pop_2018' , 'densite', 'pop_municipale', 'hab_m', 'pib_2000', 'décla','impot']
    y=['revenu_moyen']
    X_train, X_test, y_train, y_test = train_test_split(data[X], data[y], test_size=0.33, random_state=42)
    if (modele=='Lasso')==True:
        train_scores_lasso, valid_scores_lasso = validation_curve(mod, X_train, y_train, scoring='neg_mean_absolute_error',param_name='alpha',param_range=[0.01, 0.05, 0.1, 0.5, 1], cv=5)
        plt.plot(range(len(train_scores_lasso)),np.mean(train_scores_lasso,axis=1), color='green', label='train')
        plt.plot(range(len(valid_scores_lasso)),np.mean(valid_scores_lasso,axis=1),  color='yellow', label='validation')
        plt.legend()
        plt.show()
        return train_scores_lasso, valid_scores_lasso
    elif (modele=='Bagging')==True:
        train_scores_bagging, valid_scores_bagging = validation_curve(mod, X_train, np.ravel(y_train), param_name='n_estimators', param_range=[10, 25, 50, 70, 100],scoring='neg_mean_absolute_error', cv=5)
        plt.plot(range(len(train_scores_bagging)),np.mean(train_scores_bagging, axis=1), color='green', label='train')
        plt.plot(range(len(valid_scores_bagging)),np.mean(valid_scores_bagging, axis=1), color='yellow', label='validation')
        plt.legend()
        plt.show()
        return train_scores_bagging, valid_scores_bagging
    elif (modele=='Ridge')==True:
        train_scores_ridge, valid_scores_ridge = validation_curve(mod, X_train, np.ravel(y_train),param_name='alpha',param_range=[0.01, 0.05, 0.1, 0.5, 1], scoring='neg_mean_absolute_error', cv=5)
        plt.plot(range(len(train_scores_ridge)),np.mean(train_scores_ridge, axis=1), color='green', label='train')
        plt.plot(range(len(valid_scores_ridge)),np.mean(valid_scores_ridge, axis=1), color='yellow', label='validation')
        plt.legend()
        plt.show()
        return train_scores_ridge, valid_scores_ridge
    else :
        print("Veuillez renseigner un modèle parmis ceux disponibles : Lasso, Bagging, Ridge")
        
        
        
        
def verif_apprentissage(modele, mod, data):
    """ La fonction prend en paramètres le nom d'un modèle, un modèle, un data frame pandas.
    Cette fonction permet de stocker les scores sur les échantillons d'entraînement et de validation 
    en fonction de la taille de l'échantillon et la taille de l'échantillon. On peut alors déterminer 
    graphiquement si le modèle pourrait avoir un meilleur score avec plus de données.
    Elle génère également une plot des scores sur les deux échantillons. 
    Exemple :   train_sizes_bagging2, train_scores_bagging2, valid_scores_bagging2=verif_apprentissage('Bagging', mod_Bagging.best_estimator_, data)
    avec mod_Bagging le modèle obtenu suite à l'éxecution de la fonction GridSearchCV.
    Elle renvoit un graphique avec les scores sur les échantillons de validation et d'entraînement.
    Les modèles disponibles sont : Ridge, Lasso et Bagging.
    """
    X=['pop_2018' , 'densite', 'pop_municipale', 'hab_m', 'pib_2000', 'décla','impot']
    y=['revenu_moyen']
    X_train, X_test, y_train, y_test = train_test_split(data[X], data[y], test_size=0.33, random_state=42)
    if (modele=='Lasso')==True:
        train_sizes_lasso2, train_scores_lasso2, valid_scores_lasso2 = learning_curve(mod, X_train, np.ravel(y_train), train_sizes=np.linspace(0.2, 1.0, 10), cv=5, scoring='neg_mean_absolute_error')
        plt.plot(train_sizes_lasso2,np.mean(train_scores_lasso2,axis=1), color='green', label='train')
        plt.plot(train_sizes_lasso2,np.mean(valid_scores_lasso2,axis=1),  color='yellow', label='validation')
        plt.legend()
        plt.show()
        return train_sizes_lasso2, train_scores_lasso2, valid_scores_lasso2
    elif (modele=='Bagging')==True:
        train_sizes_bagging2, train_scores_bagging2, valid_scores_bagging2 = learning_curve(mod, X_train, np.ravel(y_train), train_sizes=np.linspace(0.2, 1.0, 10), cv=5, scoring='neg_mean_absolute_error')
        plt.plot(train_sizes_bagging2,np.mean(train_scores_bagging2,axis=1), color='green', label='train')
        plt.plot(train_sizes_bagging2,np.mean(valid_scores_bagging2,axis=1),  color='yellow', label='validation')
        plt.legend()
        plt.show()
        return train_sizes_bagging2, train_scores_bagging2, valid_scores_bagging2
    elif (modele=='Ridge')==True:
        train_sizes_ridge2, train_scores_ridge2, valid_scores_ridge2 = learning_curve(mod, X_train, np.ravel(y_train), train_sizes=np.linspace(0.2, 1.0, 10), cv=5, scoring='neg_mean_absolute_error')
        plt.plot(train_sizes_ridge2,np.mean(train_scores_ridge2,axis=1), color='green', label='train')
        plt.plot(train_sizes_ridge2,np.mean(valid_scores_ridge2,axis=1),  color='yellow', label='validation')
        plt.legend()
        plt.show()
        return train_sizes_ridge2, train_scores_ridge2, valid_scores_ridge2
    else :
        print("Veuillez renseigner un modèle parmis ceux disponibles : Lasso, Bagging, Ridge")
        
        
        

def metriques(y_test, revenu_moyen_prédit):
    """Cette fonction prend en argument la liste des variables à prédire et la liste des prédictions.
    Elle permet d'afficher différentes métriques d'un modèle sur les données test. 
    Exemple :   metriques(y_test, revenu_moyen_prédit)
    renvoit :   METRIQUES SUR LES DONNEES DE TEST
                MSE : 0.28136
                R2 : 0.53707
                MaxError : 2.2672
                MAE : 0.31425
    """
    print("METRIQUES SUR LES DONNEES DE TEST")
    MSE=mean_squared_error(y_test, revenu_moyen_prédit)
    print(f"MSE : {np.round(MSE,5)}")
    Rsq=r2_score(y_test, revenu_moyen_prédit)
    print(f"R2 : {np.round(Rsq, 5)}")
    MaxErr=max_error(y_test, revenu_moyen_prédit)
    print(f"MaxError : {np.round(MaxErr, 5)}")
    MaxErr=mean_absolute_error(y_test, revenu_moyen_prédit)
    print(f"MAE : {np.round(MaxErr,5)}")
    
    
    
    
def valeurs_predites(array_test, revenu_moyen_prédit):
    """Cette fonction prend en argument la liste des variables à prédire et la liste des prédictions.
    Elle permet d'afficher les valeurs réelles et les valeurs prédites sur les données test.
    Exemple :   valeurs_predites(y_test, revenu_moyen_prédit)
    renvoit :   REVENU PREDIT/REVENU REEL 
                prédit : -0.10068 ; réel : [-0.0088]
                prédit : -0.43798 ; réel : [-0.20242]
                prédit : 0.98986 ; réel : [1.05638]
                prédit : -0.11493 ; réel : [-0.22442]
                prédit : 0.02665 ; réel : [0.32728]
                prédit : 0.38624 ; réel : [0.23927]
    """
    print("REVENU PREDIT/REVENU REEL ")
    for i in range(2, len(revenu_moyen_prédit)):
        print(f"prédit : {np.round(revenu_moyen_prédit[i],5)} ; réel : {np.round(array_test[i],5)}")
        
        
        

def coefficients(X_train, mod_ridge):
    """Cette fonction prend en entrée la matrice des features et un nom de modèle.
    Elle perlet d'afficher le nom des variables et le coefficient qui leur est associé suite à une estimation.
    Exemple :   coefficients(X_train, mod_ridge)
    renvoit :   COEFFICIENTS DE CHAQUE FEATURE
                pop_2018 : 0.06547948628122444
                densite : -0.01151521883879092
                pop_municipale : 0.031544182895183605
                hab_m : -0.02398495797775591
                pib_2000 : 0.17017749485619024
                décla : 0.25052378766378336
                impot : 0.5085668102669565
    """
    print("COEFFICIENTS DE CHAQUE FEATURE")
    for i in range(len(X_train.columns)):
        print(X_train.columns[i],":", mod_ridge.best_estimator_.coef_[i])