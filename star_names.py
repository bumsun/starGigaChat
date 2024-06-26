#@title
# Acamar    13847   Groombridge 1830    57939
# Achernar  7588    Hadar   68702
# Acrux 60718   Hamal   9884
# Adhara    33579   Izar    72105
# Agena 68702   Kapteyn's star  24186
# Albireo   95947   Kaus Australis  90185
# Alcor 65477   Kocab   72607
# Alcyone   17702   Kruger 60   110893
# Aldebaran 21421   Luyten's star   36208
# Alderamin 105199  Markab  113963
# Algenib   1067    Megrez  59774
# Algieba   50583   Menkar  14135
# Algol 14576   Merak   53910
# Alhena    31681   Mintaka 25930
# Alioth    62956   Mira    10826
# Alkaid    67301   Mirach  5447
# Almaak    9640    Mirphak 15863
# Alnair    109268  Mizar   65378
# Alnath    25428   Nihal   25606
# Alnilam   26311   Nunki   92855
# Alnitak   26727   Phad    58001
# Alphard   46390   Pleione 17851
# Alphekka  76267   Polaris 11767
# Alpheratz 677 Pollux  37826
# Alshain   98036   Procyon 37279
# Altair    97649   Proxima 70890
# Ankaa 2081    Rasalgethi  84345
# Antares   80763   Rasalhague  86032
# Arcturus  69673   Red Rectangle   30089
# Arneb 25985   Regulus 49669
# Babcock's star    112247  Rigel   24436
# Barnard's star    87937   Rigil Kent  71683
# Bellatrix 25336   Sadalmelik  109074
# Betelgeuse    27989   Saiph   27366
# Campbell's star   96295   Scheat  113881
# Canopus   30438   Shaula  85927
# Capella   24608   Shedir  3179
# Caph  746 Sheliak 92420
# Castor    36850   Sirius  32349
# Cor Caroli    63125   Spica   65474
# Cyg X-1   98298   Tarazed 97278
# Deneb 102098  Thuban  68756
# Denebola  57632   Unukalhai   77070
# Diphda    3419    Van Maanen 2    3829
# Dubhe 54061   Vega    91262
# Enif  107315  Vindemiatrix    63608
# Etamin    87833   Zaurak  18543
# Fomalhaut 113368  3C 273  60936
import pandas as pd

str_names = """Acamar 13847 
Groombridge_1830 57939
Achernar 7588   
Hadar 68702
Acrux 60718 
Hamal 9884
Adhara 33579    
Izar 72105
Agena 68702 
Kapteyn's_star 24186
Albireo 95947   
Kaus_Australis 90185
Alcor 65477 
Kocab 72607
Alcyone 17702   
Kruger_60 110893
Aldebaran 21421 
Luyten's_star 36208
Alderamin 105199    
Markab 113963
Algenib 1067    
Megrez 59774
Algieba 50583   
Menkar 14135
Algol 14576 
Merak 53910
Alhena 31681    
Mintaka 25930
Alioth 62956    
Mira 10826
Alkaid 67301    
Mirach 5447
Almaak 9640 
Mirphak 15863
Alnair 109268   
Mizar 65378
Alnath 25428    
Nihal 25606
Alnilam 26311   
Nunki 92855
Alnitak 26727   
Phad 58001
Alphard 46390   
Pleione 17851
Alphekka 76267  
Polaris 11767
Alpheratz 677   
Pollux 37826
Alshain 98036   
Procyon 37279
Altair 97649    
Proxima 70890
Ankaa 2081  
Rasalgethi 84345
Antares 80763   
Rasalhague 86032
Arcturus 69673  
Red_Rectangle 30089
Arneb 25985 
Regulus 49669
Babcock's_star 112247   
Rigel 24436
Barnard's_star 87937    
Rigil_Kent 71683
Bellatrix 25336 
Sadalmelik 109074
Betelgeuse 27989    
Saiph 27366
Campbell's_star 96295   
Scheat 113881
Canopus 30438   
Shaula 85927
Capella 24608   
Shedir 3179
Caph 746    
Sheliak 92420
Castor 36850    
Sirius 32349
Cor_Caroli 63125    
Spica 65474
Cyg_X-1 98298   
Tarazed 97278
Deneb 102098    
Thuban 68756
Denebola 57632  
Unukalhai 77070
Diphda 3419 
Van_Maanen_2 3829
Dubhe 54061 
Vega 91262
Enif 107315 
Vindemiatrix 63608
Etamin 87833    
Zaurak 18543
Fomalhaut 113368    
3C_273 60936"""

def get_star_names():
    global str_names


    arr_names = str_names.replace("\t","").split("\n")
    dict_names = {}
    for item in arr_names:
        arr = item.split(" ")
        dict_names[int(arr[1])] = arr[0]
    
    df_names = pd.DataFrame.from_dict(dict_names, orient='index')
    df_names.columns = ['name']
    return df_names
