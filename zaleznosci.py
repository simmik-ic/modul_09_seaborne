import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt

#PRZYGOTOWANIE DANYCH
#wczytaj dane z pliku
df = pd.read_csv('HRDataset.csv')

#usuń brakujące elementy
df.drop(['LastPerformanceReview_Date','DaysLateLast30'],axis=1,inplace=True)
df.dropna(thresh=3,inplace=True)

#przekonwertuj dane w odpowiednich kolumnach na format dt
df['DOB'] = pd.to_datetime(df['DOB'],format='%m/%d/%y')
df['DOB'] = df['DOB'].apply(lambda d: d if d.year < 2025 else d.replace(year=d.year - 100))
df['DateofTermination'] = pd.to_datetime(df['DateofTermination'],format='%m/%d/%y')
df['DateofHire'] = pd.to_datetime(df['DateofHire'],format='%m/%d/%Y')

#funkcja do obliczania stażu pracy
def count_seniority(row):
    if pd.isnull(row['DateofTermination']):
        end_date = dt.datetime(2019,9,27)
    else:
        end_date = row['DateofTermination']

    return (end_date - row['DateofHire'])/np.timedelta64(1,'D') / 365.25
#dołącz kolumnę stażu pracy
df['Seniority'] = df.apply(lambda row: count_seniority(row),axis=1)

#funkcja do obliczania wieku pracownika
def count_age(row):
    return (dt.datetime(2019,9,27) - row['DOB'])/np.timedelta64(1,'D') / 365.25
#dołącz kolumnę wieku pracownika
df['Age'] = df.apply(lambda row: count_age(row),axis=1)

#-----------------------------------------------------------------------------------------------------
#1. Manager vs. ocena wydajności
#przygotuj tabelę przestawną, która zlicza oceny dla każdego menedżera
manager_pt = df.pivot_table(index=df['ManagerName'], columns=df['PerformanceScore'], 
                            values='EmpID', aggfunc='count').fillna(0).sort_index()
#ustandaryzuj wyniki dla każdego menedżera
manager_pt = manager_pt.div(manager_pt.sum(axis=1), axis=0).round(3)

plt.figure(figsize=(10,6))
sns.heatmap(data=manager_pt)
plt.tight_layout() 

#ustaw etykiety osi legendy w %
cbar = plt.gca().collections[0].colorbar
ticks = cbar.get_ticks()
cbar.set_ticks(ticks)
cbar.set_ticklabels([f'{tick*100:.0f}%' for tick in ticks])


#2. Żródła vs. staż pracy
order = df['RecruitmentSource'].value_counts().index                #ustal porządek wyświetlania kategorii

plt.figure(figsize=(10,8))
sns.boxplot(x='RecruitmentSource',y='Seniority',data=df,order=order)
plt.xticks(rotation=90)
plt.tight_layout() 


#3. Stan cywilny vs. zadowolenie
order = ['Single','Married','Separated','Divorced','Widowed']
plt.figure(figsize=(10,6))
sns.barplot(x='MaritalDesc',y='EmpSatisfaction',data=df, order=order)
#plt.xticks(rotation=90)
plt.tight_layout() 


#4. Struktura wieku
step = 5
bin_min = np.floor(df['Age'].min()/step)*step -step
bin_max = np.ceil(df['Age'].max()/step)*step +(2*step)
bins = np.arange(bin_min, bin_max, step)

actually_working = df

plt.figure(figsize=(10,6))
sns.histplot(data=df[df['DateofTermination'].isna()], x='Age', bins=bins)
plt.xticks(bins)


#5. Wiek vs. specjalne projekty
projects_pt = df.pivot_table(index=df['SpecialProjectsCount'], columns=df['Age'].astype(int), 
                             values='EmpID', aggfunc='count').fillna(0)

plt.figure(figsize=(10,6))
sns.heatmap(data=projects_pt)
plt.gca().invert_yaxis()


#-----------------------------------------------------------------------------------------------------
analysis = """
1. 
Pracownicy podlegający pod niektórych menedżerów (Alex Sweetwater, Brannon Miller, Eric Dougall, Janet Kong, Jennifer Zamora) 
częściej posiadają ocenę 'Exceeds'. Natomiast ci podlegający pod menedżera Debra Houlihan mają więcej ocen 'Needs Improvement'.
2.
Pomijając źródła 'Other', aby zapewnić jak najdłuższe zatrudnienie należy skupić się na źródłach pozyskania pracowników takich jak: 
'Billboards', 'Website Banner Ads', 'On-campus Recruiting', 'Information Sessions' oraz 'MBTA Ads'.
3.
Najmniej zadowoleni są pracowanicy 'Married', zaś najbardziej 'Widowed'.
4.
Najwięcej pracowników ma od 30 do 45 lat. W starszych rocznikach liczba zatrudnionych spada wraz z wiekiem, z wyjątkiem grupy 65+.
W firmie nie pracują osoby w wieku poniżej 27 lat, co wydaje się trochę dziwne. Biorąc po uwagę, że ostatnie daty zatrudnienia
przypadają na 2017 rok (a zakończenia pracy na 2016), być może aktualność bazy danych jest wcześniejsza niż 27 września 2019.
5.
Większą liczbę specjalnych projektów realizują młodzi pracownicy.
"""

print(analysis)
plt.show()
plt.close('all')