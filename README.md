
# What's the big picture? 
* Some problems are so difficult that no one researcher, research group, research institute, or multi-national company can make meaningful progress. It takes a world wide effort and collaborations between industry and academia. Drug discovery is one such area. Advances in data science (aka artificial intelligence, machine learning) are being applied to data sets from high-throughput experimental techniques and historical databases of biomedical literature, publicly available to the world community.

* The process of small molecule drug development involves the gradual reduction of tens of thousands of small molecules to a drug candidate that eventually is given to patients in clinical trials. This is a long (decades, often the whole career of a researcher), costly process and engages all corners of our interconnected economy (scientists, physicians, doctors, entrepreneurs, investors, pharmaceutical companies, government officials). These real world constraints pressure research questions to shy away from too much risk and leave many diseases untreated. But computational methods that have become popularized within the past decade can help make data driven decisions earlier in the decision making process, so that drugs can be developed better, faster, and cheaper. At this workshop you will get hands on experience solving the types of problems that keep our researchers up at night.

# The plan
* Input data: a precomputed and relatively clean data set of ~1000 drugs-like molecules by ~100 chemical features
* Goal: Your job is to categorize drug-like molecules into a smaller diverse and representative set. This is a real-world unsupervised multi-class classification problem encountered in a biotech startup. There is underlying structure in this data set and we have solved it one way and are curious to see how you solve it. 
* Hints: you will be given clues about the structure of the data at the event, but for now it's top secret! We have prepared Python code snippets (pandas, numpy, scikit-learn) for a solution using k-means clustering to move you along toward the goal within the time constraints of the event.
* This jupyter notebook is here to help facilitate the workshop


```python
from IPython.display import Image
Image("Screen Shot 2016-10-27 at 3.29.58 PM.png")
```




![png](output_2_0.png)



# Technical remarks
* If you don't have pandas, numpy, scikit-learn, matplotlib, etc installed then do so with
> pip install pandas, numpy, scikit-learn, matplotlib
* You can check which libraries you have installed with
> pip freeze

## Import data


```python
import pandas as pd
inputfile = 'chemicalDataForStudents20161027-110104.csv'
df = pd.read_csv(inputfile, sep=',')

```


```python
# take a peak at the data
print df.shape
print df.tail(3)

```

    (1650, 111)
           LabuteASA  MaxAbsEStateIndex  MaxAbsPartialCharge  MaxEStateIndex  \
    1514  164.703793          14.880596             0.496768       14.880596   
    161   148.584776           5.798308             0.493601        5.798308   
    1220  138.515751          12.549085             0.347020       12.549085   
    
          MaxPartialCharge  MinAbsEStateIndex  MinAbsPartialCharge  \
    1514          0.350866           0.002799             0.350866   
    161           0.215753           0.686138             0.215753   
    1220          0.244674           0.081001             0.244674   
    
          MinEStateIndex  MinPartialCharge  MolLogP         ...          \
    1514       -0.875036         -0.496768  3.48928         ...           
    161         0.686138         -0.493601  4.33182         ...           
    1220       -0.668981         -0.347020  1.25950         ...           
    
          fr_term_acetylene  fr_tetrazole  fr_thiazole  fr_thiophene  \
    1514                  0             0            0             0   
    161                   0             0            0             0   
    1220                  0             0            0             0   
    
          fr_unbrch_alkane  fr_urea  \
    1514                 0        0   
    161                  5        0   
    1220                 0        0   
    
                                                     smiles  \
    1514  COc1ccc(cc1)c2ccc(c(c2)F)N\3C(=O)CS/C3=C(/C#N)...   
    161                Cc1cc(on1)CCCCCCCOc2ccc(cc2)C3=NCCO3   
    1220  CCCC[C@H](CN(C=O)O)C(=O)N[C@H](C(=O)N(C)C)C(C)...   
    
                                     codeName  y_pred  clusterSize_y_pred  
    1514        JamesWatt-JeanBaptisteLamarck    1390                   1  
    161         DanielBernoulli-CharlesDarwin    1391                   1  
    1220  Empedocles-CharlesAugustindeCoulomb    1392                   1  
    
    [3 rows x 111 columns]



```python
# the chemicals can be represented by a string.
print df.head().smiles

# each compound has a codeName
# the codeNames are how we will can refer to them after the analysis (rather than by row number or smiles)
print df.head().codeName 
```

    525           Cc1cc(nc(c1)N)COC[C@H](CN)OCc2cc(cc(n2)N)C
    526         Cc1cc(nc(c1)N)COCC[C@@H](CN)OCc2cc(cc(n2)N)C
    527    Cc1cc(nc(c1)N)COC[C@@H]([C@H](C)OCc2cc(cc(n2)N...
    528          Cc1cc(nc(c1)N)COC[C@@H](CN)OCc2cc(cc(n2)N)C
    415    CC(C)(C)NC(=O)[C@@H](c1ccccc1)NC(=O)N(C)Cc2ccc...
    Name: smiles, dtype: object
    525              JamesClerkMaxwell-ErnstMayr
    526                      BillNye-FrankHornby
    527            CharlesLyell-ErwinSchrodinger
    528                Empedocles-GustavKirchoff
    415    CharlesAugustindeCoulomb-FrancisCrick
    Name: codeName, dtype: object


# Cleaning the data
* Real data is messy. Data sanitization involves 
    * removing features or samples that didn't compute for all samples  
    * removing outliers that you suspect are artefacts or that will wildnly bias the predictions that come from the data
    * The data provided has been filtered a bit, bit be warned that this is an important part of the process and can take a long time

# Normalizing the data
* The features need to be treated equally. Just because units change from grams to kilograms does not mean there is a 100x difference
* There are various ways to standardize data. You may have read about standard scores (Z-statistic). In the end each feature should be centred around the same value and have the same max and min.
* The way this is done should preserve the variation in each feature. So remember your numerical methods computer science class and beware of subtracting errors and the like.


```python
# just get features from data, remove labels
df_un = df.drop(['codeName', 'smiles'], 1)

# normalize
import numpy as np
df_norm = (df_un - df_un.mean()) / (df_un.max() - df_un.min())
X = np.array(df_norm)

# X is basically scaled to be between 1 and zero in way that is robust to real word data
# you can uncomment this to check
# print 'mean', np.mean(X,0)
# print 'max', np.max(X,0)
# print 'min', np.min(X,0) 
```

# K-means clustering
* Read these links
    * https://en.wikipedia.org/wiki/K-means_clustering
    * http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
* K-means clustering comes up with labels for unlabelled data. It takes the data and a parameter (we call it k here) that fixes the number of clusters
* Try out different values of k using the code below
* The key line of code below is
> y_pred = KMeans(n_clusters=k, random_state=random_state).fit_predict(X)
* It takes the normalized data and asigns cluster labels to it, such that there are k unique clusters.
* Properties of k
    * k is integer, since clusters are countable
    * k is at least 1. This would be one big cluster
    * k is at most teh number of samples (the rows of X). This would treat every sample as its own cluster (a singleton) 


```python
# cluster by kmeans
from sklearn.cluster import KMeans
import random
random.seed(0)
k = int(random.uniform(1, len(X))) # set k without any prior knowledge... any number between 1 and the number of samples
print 'k', k
random_state = 0
y_pred = KMeans(n_clusters=k, random_state=random_state).fit_predict(X)
df['y_pred'] = y_pred # plot and analyze unnormalized data with labels
```

    k 1393


* Now that the clustering is done we can look at the sizes of the clsuters. The function 
>np.histogram
* outputs two arrays, [the number of clusters of a given size], [the size of the clusters]


```python
# look at cluster size
print np.histogram(df.groupby('y_pred').size(), bins = np.append(np.unique(df.groupby( ["y_pred"] ).size()), np.max(df.groupby( ["y_pred"] ).size())+1))
# add in cluster size to df
df = pd.merge(df, pd.DataFrame({'clusterSize_y_pred' : df.groupby( ["y_pred"] ).size()}).reset_index(), on='y_pred') 
print df.tail()
```

    (array([1161,  208,   23,    1]), array([1, 2, 3, 4, 5]))
           LabuteASA  MaxAbsEStateIndex  MaxAbsPartialCharge  MaxEStateIndex  \
    1645  147.806545          14.525346             0.378511       14.525346   
    1646  149.812648          12.574862             0.477880       12.574862   
    1647  181.229439          14.001653             0.460949       14.001653   
    1648  203.683877          11.743598             0.438042       11.743598   
    1649  124.973421          11.101663             0.507823       11.101663   
    
          MaxPartialCharge  MinAbsEStateIndex  MinAbsPartialCharge  \
    1645          0.154401           0.031220             0.154401   
    1646          0.330899           0.050841             0.330899   
    1647          0.258894           0.164579             0.258894   
    1648          0.233112           0.053390             0.233112   
    1649          0.230804           0.042424             0.230804   
    
          MinEStateIndex  MinPartialCharge  MolLogP         ...          \
    1645       -0.910887         -0.378511  2.73290         ...           
    1646       -1.397395         -0.477880  1.01617         ...           
    1647       -0.569145         -0.460949  1.87380         ...           
    1648       -0.053390         -0.438042  4.59410         ...           
    1649       -1.291383         -0.507823  1.83460         ...           
    
          fr_term_acetylene  fr_tetrazole  fr_thiazole  fr_thiophene  \
    1645                  0             0            0             0   
    1646                  0             0            0             0   
    1647                  0             0            0             0   
    1648                  0             0            0             0   
    1649                  0             0            0             0   
    
          fr_unbrch_alkane  fr_urea  \
    1645                 0        0   
    1646                 0        0   
    1647                 0        0   
    1648                 0        0   
    1649                 0        0   
    
                                                     smiles  \
    1645  Cn1cc(cn1)[C@H]2C[C@H]3CSC(=N[C@]3(CO2)c4ccc(c...   
    1646  [H]/N=C/1\NC(=O)[C@]2(S1)C=C(C[C@H]([C@@H]2NC(...   
    1647  c1cc(oc1)c2nc3nc(nc(n3n2)N)NCCN4CCN(CC4)c5ccc(...   
    1648  CCC(=O)Nc1cccc(c1)Oc2c3cc[nH]c3nc(n2)Nc4ccc(cc...   
    1649     c1cc2c(cc1O)OC[C@]3([C@@H]2Oc4c3cc5c(c4)OCO5)O   
    
                               codeName  y_pred  clusterSize_y_pred  
    1645  MichaelFaraday-GalileoGalilei    1317                   1  
    1646          CarlBosch-RobertHooke     404                   1  
    1647      FrancisGalton-Anaximander     402                   1  
    1648  BenjaminThompson-KonradLorenz    1332                   1  
    1649    RobertKoch-AndreMarieAmpere     268                   1  
    
    [5 rows x 111 columns]



```python
# get top clusters
topClusters=df[['y_pred', 'clusterSize_y_pred']].drop_duplicates().sort_values(by='clusterSize_y_pred', ascending=[0]).head()
print topClusters

```

          y_pred  clusterSize_y_pred
    525      142                   4
    1103     199                   3
    341       57                   3
    36      1233                   3
    546      218                   3


# Sanity check... 2d chemical structures
* do the compunds in the same clusters look the same?
* use this webtool to check https://cactus.nci.nih.gov/gifcreator/

# Submit you classes to Cyclica
* Since we know the real clusters by another method we can compare your to ours
* Output your final list of y_pred classes with the codeNames and smiles and we can go back and check if they are the same as our classes
* The code below outputs a csv file. Details of how to submit will be given at the workshop



```python
# sort data
df = df.sort_values(by=['clusterSize_y_pred', 'y_pred'], ascending=[0,1])

# output data
import time
timestr = time.strftime("%Y%m%d-%H%M%S")
initials='gw'
output = 'predictedClasses' + initials + timestr +'.csv'
df.to_csv(output, sep=',', index=False)
df.head(50)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LabuteASA</th>
      <th>MaxAbsEStateIndex</th>
      <th>MaxAbsPartialCharge</th>
      <th>MaxEStateIndex</th>
      <th>MaxPartialCharge</th>
      <th>MinAbsEStateIndex</th>
      <th>MinAbsPartialCharge</th>
      <th>MinEStateIndex</th>
      <th>MinPartialCharge</th>
      <th>MolLogP</th>
      <th>...</th>
      <th>fr_term_acetylene</th>
      <th>fr_tetrazole</th>
      <th>fr_thiazole</th>
      <th>fr_thiophene</th>
      <th>fr_unbrch_alkane</th>
      <th>fr_urea</th>
      <th>smiles</th>
      <th>codeName</th>
      <th>y_pred</th>
      <th>clusterSize_y_pred</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>525</th>
      <td>141.729151</td>
      <td>5.759574</td>
      <td>0.383683</td>
      <td>5.759574</td>
      <td>0.123477</td>
      <td>0.226210</td>
      <td>0.123477</td>
      <td>-0.226210</td>
      <td>-0.383683</td>
      <td>1.31854</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Cc1cc(nc(c1)N)COC[C@H](CN)OCc2cc(cc(n2)N)C</td>
      <td>JamesClerkMaxwell-ErnstMayr</td>
      <td>142</td>
      <td>4</td>
    </tr>
    <tr>
      <th>526</th>
      <td>148.094093</td>
      <td>5.819164</td>
      <td>0.383683</td>
      <td>5.819164</td>
      <td>0.123477</td>
      <td>0.098079</td>
      <td>0.123477</td>
      <td>-0.098079</td>
      <td>-0.383683</td>
      <td>1.70864</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>Cc1cc(nc(c1)N)COCC[C@@H](CN)OCc2cc(cc(n2)N)C</td>
      <td>BillNye-FrankHornby</td>
      <td>142</td>
      <td>4</td>
    </tr>
    <tr>
      <th>527</th>
      <td>148.094093</td>
      <td>6.124918</td>
      <td>0.383683</td>
      <td>6.124918</td>
      <td>0.123477</td>
      <td>0.181412</td>
      <td>0.123477</td>
      <td>-0.259896</td>
      <td>-0.383683</td>
      <td>1.70704</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Cc1cc(nc(c1)N)COC[C@@H]([C@H](C)OCc2cc(cc(n2)N...</td>
      <td>CharlesLyell-ErwinSchrodinger</td>
      <td>142</td>
      <td>4</td>
    </tr>
    <tr>
      <th>528</th>
      <td>141.729151</td>
      <td>5.759574</td>
      <td>0.383683</td>
      <td>5.759574</td>
      <td>0.123477</td>
      <td>0.226210</td>
      <td>0.123477</td>
      <td>-0.226210</td>
      <td>-0.383683</td>
      <td>1.31854</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Cc1cc(nc(c1)N)COC[C@@H](CN)OCc2cc(cc(n2)N)C</td>
      <td>Empedocles-GustavKirchoff</td>
      <td>142</td>
      <td>4</td>
    </tr>
    <tr>
      <th>415</th>
      <td>185.862478</td>
      <td>12.947534</td>
      <td>0.477530</td>
      <td>12.947534</td>
      <td>0.339488</td>
      <td>0.007434</td>
      <td>0.339488</td>
      <td>-1.177893</td>
      <td>-0.477530</td>
      <td>2.91090</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>CC(C)(C)NC(=O)[C@@H](c1ccccc1)NC(=O)N(C)Cc2ccc...</td>
      <td>CharlesAugustindeCoulomb-FrancisCrick</td>
      <td>6</td>
      <td>3</td>
    </tr>
    <tr>
      <th>416</th>
      <td>185.862478</td>
      <td>12.911895</td>
      <td>0.477530</td>
      <td>12.911895</td>
      <td>0.339488</td>
      <td>0.003046</td>
      <td>0.339488</td>
      <td>-1.172572</td>
      <td>-0.477530</td>
      <td>2.91250</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>CCCCNC(=O)[C@H](c1ccccc1)NC(=O)N(C)Cc2ccc3c(c2...</td>
      <td>WolfgangErnstPauli-Lucretius</td>
      <td>6</td>
      <td>3</td>
    </tr>
    <tr>
      <th>417</th>
      <td>201.824746</td>
      <td>13.056169</td>
      <td>0.477530</td>
      <td>13.056169</td>
      <td>0.339488</td>
      <td>0.016616</td>
      <td>0.339488</td>
      <td>-1.181843</td>
      <td>-0.477530</td>
      <td>3.31260</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>CN(Cc1ccc2c(c1C(=O)O)OCO2)C(=O)N[C@@H](c3ccccc...</td>
      <td>ErwinSchrodinger-EvangelistaTorricelli</td>
      <td>6</td>
      <td>3</td>
    </tr>
    <tr>
      <th>151</th>
      <td>112.519202</td>
      <td>12.044467</td>
      <td>0.504068</td>
      <td>12.044467</td>
      <td>0.200850</td>
      <td>0.003845</td>
      <td>0.200850</td>
      <td>-0.738647</td>
      <td>-0.504068</td>
      <td>2.57680</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>c1ccc(cc1)C2=CC(=O)c3c(cc(c(c3O)O)O)O2</td>
      <td>LinusPauling-IsaacNewton</td>
      <td>36</td>
      <td>3</td>
    </tr>
    <tr>
      <th>152</th>
      <td>123.997689</td>
      <td>12.233506</td>
      <td>0.507966</td>
      <td>12.233506</td>
      <td>0.203372</td>
      <td>0.033681</td>
      <td>0.203372</td>
      <td>-0.472106</td>
      <td>-0.507966</td>
      <td>2.58540</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>COc1c(cc2c(c1O)C(=O)C=C(O2)c3ccc(cc3)O)O</td>
      <td>CarlFriedrichGauss-BillNye</td>
      <td>36</td>
      <td>3</td>
    </tr>
    <tr>
      <th>153</th>
      <td>112.519202</td>
      <td>12.350655</td>
      <td>0.507822</td>
      <td>12.350655</td>
      <td>0.199995</td>
      <td>0.009887</td>
      <td>0.199995</td>
      <td>-0.312312</td>
      <td>-0.507822</td>
      <td>2.57680</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>c1cc(c(cc1C2=COc3cc(ccc3C2=O)O)O)O</td>
      <td>SigmundFreud-BenjaminFranklin</td>
      <td>36</td>
      <td>3</td>
    </tr>
    <tr>
      <th>220</th>
      <td>130.436067</td>
      <td>14.155215</td>
      <td>0.378494</td>
      <td>14.155215</td>
      <td>0.154894</td>
      <td>0.034118</td>
      <td>0.154894</td>
      <td>-0.853547</td>
      <td>-0.378494</td>
      <td>3.16290</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>C[C@]1(C[C@H](SC(=N1)N)c2cncnc2)c3ccc(cc3F)F</td>
      <td>LeonardodaVinci-JamesWatson</td>
      <td>44</td>
      <td>3</td>
    </tr>
    <tr>
      <th>221</th>
      <td>130.436067</td>
      <td>14.314858</td>
      <td>0.378512</td>
      <td>14.314858</td>
      <td>0.154285</td>
      <td>0.254836</td>
      <td>0.154285</td>
      <td>-0.794143</td>
      <td>-0.378512</td>
      <td>3.08860</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>C[C@]1(CCSC(=N1)N)c2cc(c(cc2F)F)c3cncnc3</td>
      <td>JeanBaptisteLamarck-ThomasKuhn</td>
      <td>44</td>
      <td>3</td>
    </tr>
    <tr>
      <th>222</th>
      <td>136.691989</td>
      <td>14.247607</td>
      <td>0.378494</td>
      <td>14.247607</td>
      <td>0.154895</td>
      <td>0.047960</td>
      <td>0.154895</td>
      <td>-0.867130</td>
      <td>-0.378494</td>
      <td>3.97774</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Cc1c(c(on1)C)[C@@H]2C[C@@](N=C(S2)N)(C)c3ccc(c...</td>
      <td>CarolusLinnaeus-FranzBoas</td>
      <td>44</td>
      <td>3</td>
    </tr>
    <tr>
      <th>341</th>
      <td>194.858937</td>
      <td>12.370274</td>
      <td>0.488253</td>
      <td>12.370274</td>
      <td>0.488253</td>
      <td>0.250590</td>
      <td>0.423170</td>
      <td>-1.478263</td>
      <td>-0.423170</td>
      <td>2.46130</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>B(c1ccccc1CN2CCN(CC2)C3=NC(=O)/C(=C/c4ccc(c(c4...</td>
      <td>Lucretius-Avicenna</td>
      <td>57</td>
      <td>3</td>
    </tr>
    <tr>
      <th>342</th>
      <td>194.858937</td>
      <td>12.356106</td>
      <td>0.487918</td>
      <td>12.356106</td>
      <td>0.487918</td>
      <td>0.236423</td>
      <td>0.423177</td>
      <td>-1.443879</td>
      <td>-0.423177</td>
      <td>2.46130</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>B(c1ccc(cc1)CN2CCN(CC2)C3=NC(=O)/C(=C/c4ccc(c(...</td>
      <td>LouisdeBroglie-HenryMoseley</td>
      <td>57</td>
      <td>3</td>
    </tr>
    <tr>
      <th>343</th>
      <td>194.858937</td>
      <td>12.362383</td>
      <td>0.487928</td>
      <td>12.362383</td>
      <td>0.487928</td>
      <td>0.242705</td>
      <td>0.423177</td>
      <td>-1.460291</td>
      <td>-0.423177</td>
      <td>2.46130</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>B(c1cccc(c1)CN2CCN(CC2)C3=NC(=O)/C(=C/c4ccc(c(...</td>
      <td>FranzBoas-HermannvonHelmholtz</td>
      <td>57</td>
      <td>3</td>
    </tr>
    <tr>
      <th>105</th>
      <td>148.269255</td>
      <td>12.380685</td>
      <td>0.312156</td>
      <td>12.380685</td>
      <td>0.236417</td>
      <td>0.064117</td>
      <td>0.236417</td>
      <td>-3.479281</td>
      <td>-0.312156</td>
      <td>3.31770</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>CCCN1c2ccc(cc2CCC1=O)NS(=O)(=O)Cc3ccccc3</td>
      <td>MaxPlanck-JackHorner</td>
      <td>62</td>
      <td>3</td>
    </tr>
    <tr>
      <th>106</th>
      <td>154.634197</td>
      <td>12.465613</td>
      <td>0.312156</td>
      <td>12.465613</td>
      <td>0.236417</td>
      <td>0.064672</td>
      <td>0.236417</td>
      <td>-3.492131</td>
      <td>-0.312156</td>
      <td>3.62612</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>CCCN1c2ccc(cc2CCC1=O)NS(=O)(=O)Cc3ccc(cc3)C</td>
      <td>IsaacNewton-HeinrichHertz</td>
      <td>62</td>
      <td>3</td>
    </tr>
    <tr>
      <th>107</th>
      <td>141.904313</td>
      <td>12.363117</td>
      <td>0.315211</td>
      <td>12.363117</td>
      <td>0.236417</td>
      <td>0.066713</td>
      <td>0.236417</td>
      <td>-3.481908</td>
      <td>-0.315211</td>
      <td>2.84592</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Cc1ccc(cc1)CS(=O)(=O)Nc2ccc3c(c2)CCC(=O)N3C</td>
      <td>Lucretius-LouisPasteur</td>
      <td>62</td>
      <td>3</td>
    </tr>
    <tr>
      <th>693</th>
      <td>217.369820</td>
      <td>13.986122</td>
      <td>0.443692</td>
      <td>13.986122</td>
      <td>0.407311</td>
      <td>0.053205</td>
      <td>0.407311</td>
      <td>-4.004868</td>
      <td>-0.443692</td>
      <td>3.13200</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>CC(C)[C@H]1Cc2cc(ccc2S(=O)(=O)N(C1)C[C@H]([C@H...</td>
      <td>FrancisCrick-AlbertEinstein</td>
      <td>114</td>
      <td>3</td>
    </tr>
    <tr>
      <th>694</th>
      <td>223.734762</td>
      <td>14.081573</td>
      <td>0.443692</td>
      <td>14.081573</td>
      <td>0.407311</td>
      <td>0.047279</td>
      <td>0.407311</td>
      <td>-4.031627</td>
      <td>-0.443692</td>
      <td>3.52210</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>CC(C)(C)[C@@H]1Cc2cc(ccc2S(=O)(=O)N(C1)C[C@H](...</td>
      <td>WilliamHarvey-AlessandroVolta</td>
      <td>114</td>
      <td>3</td>
    </tr>
    <tr>
      <th>695</th>
      <td>223.734762</td>
      <td>14.081573</td>
      <td>0.443692</td>
      <td>14.081573</td>
      <td>0.407311</td>
      <td>0.047279</td>
      <td>0.407311</td>
      <td>-4.031627</td>
      <td>-0.443692</td>
      <td>3.52210</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>CC(C)(C)[C@H]1Cc2cc(ccc2S(=O)(=O)N(C1)C[C@H]([...</td>
      <td>MarieCurie-AlexanderVonHumboldt</td>
      <td>114</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1000</th>
      <td>161.901036</td>
      <td>7.541884</td>
      <td>0.485185</td>
      <td>7.541884</td>
      <td>0.150640</td>
      <td>0.007064</td>
      <td>0.150640</td>
      <td>-0.263058</td>
      <td>-0.485185</td>
      <td>1.24734</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>c1cc(cc(c1)O[C@H]2CO[C@H]3[C@@H]2OC[C@H]3Oc4cc...</td>
      <td>JamesWatt-JamesWatson</td>
      <td>148</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1001</th>
      <td>161.901036</td>
      <td>7.544344</td>
      <td>0.485160</td>
      <td>7.544344</td>
      <td>0.150640</td>
      <td>0.009842</td>
      <td>0.150640</td>
      <td>-0.273739</td>
      <td>-0.485160</td>
      <td>1.24734</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>c1cc(cc(c1)O[C@@H]2CO[C@H]3[C@@H]2OC[C@@H]3Oc4...</td>
      <td>FriedrichAugustKekule-MichaelFaraday</td>
      <td>148</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1002</th>
      <td>161.901036</td>
      <td>7.439745</td>
      <td>0.485185</td>
      <td>7.439745</td>
      <td>0.150639</td>
      <td>0.019691</td>
      <td>0.150639</td>
      <td>-0.234382</td>
      <td>-0.485185</td>
      <td>1.24734</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>c1cc(ccc1C(=N)N)O[C@H]2CO[C@H]3[C@@H]2OC[C@H]3...</td>
      <td>GalileoGalilei-HenryMoseley</td>
      <td>148</td>
      <td>3</td>
    </tr>
    <tr>
      <th>11</th>
      <td>190.631761</td>
      <td>12.020077</td>
      <td>0.312609</td>
      <td>12.020077</td>
      <td>0.258254</td>
      <td>0.175468</td>
      <td>0.258254</td>
      <td>-0.175468</td>
      <td>-0.312609</td>
      <td>3.65440</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>c1cc(ccc1CC2CCN(CC2)CCc3cnn(c3)c4c5c(ccn4)C(=O...</td>
      <td>WernerHeisenberg-BenjaminFranklin</td>
      <td>171</td>
      <td>3</td>
    </tr>
    <tr>
      <th>12</th>
      <td>184.266819</td>
      <td>12.009169</td>
      <td>0.312609</td>
      <td>12.009169</td>
      <td>0.258254</td>
      <td>0.177748</td>
      <td>0.258254</td>
      <td>-0.177748</td>
      <td>-0.312609</td>
      <td>3.57930</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>c1cc(ccc1C2CCN(CC2)CCc3cnn(c3)c4c5c(ccn4)C(=O)...</td>
      <td>ArthurEddington-PeterDebye</td>
      <td>171</td>
      <td>3</td>
    </tr>
    <tr>
      <th>13</th>
      <td>194.570085</td>
      <td>12.020277</td>
      <td>0.312609</td>
      <td>12.020277</td>
      <td>0.258254</td>
      <td>0.185669</td>
      <td>0.258254</td>
      <td>-0.185669</td>
      <td>-0.312609</td>
      <td>4.23270</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>c1cnc(c2c1C(=O)NC=N2)n3cc(cn3)CCN4CCC(CC4)c5cc...</td>
      <td>JagadishChandraBose-AlbertEinstein</td>
      <td>171</td>
      <td>3</td>
    </tr>
    <tr>
      <th>217</th>
      <td>138.697590</td>
      <td>9.485220</td>
      <td>0.507966</td>
      <td>9.485220</td>
      <td>0.115120</td>
      <td>0.191250</td>
      <td>0.115120</td>
      <td>0.191250</td>
      <td>-0.507966</td>
      <td>4.92030</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>c1ccc(c(c1)N=C(c2ccc(cc2)O)c3ccc(cc3)O)Cl</td>
      <td>CharlesAugustindeCoulomb-Anaximander</td>
      <td>180</td>
      <td>3</td>
    </tr>
    <tr>
      <th>218</th>
      <td>134.759266</td>
      <td>9.506331</td>
      <td>0.507966</td>
      <td>9.506331</td>
      <td>0.115120</td>
      <td>0.217313</td>
      <td>0.115120</td>
      <td>0.217313</td>
      <td>-0.507966</td>
      <td>4.57532</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Cc1ccccc1N=C(c2ccc(cc2)O)c3ccc(cc3)O</td>
      <td>HenryMoseley-AmedeoAvogadro</td>
      <td>180</td>
      <td>3</td>
    </tr>
    <tr>
      <th>219</th>
      <td>128.394324</td>
      <td>9.462216</td>
      <td>0.507966</td>
      <td>9.462216</td>
      <td>0.115120</td>
      <td>0.216220</td>
      <td>0.115120</td>
      <td>0.216220</td>
      <td>-0.507966</td>
      <td>4.26690</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>c1ccc(cc1)N=C(c2ccc(cc2)O)c3ccc(cc3)O</td>
      <td>Euclid-Avicenna</td>
      <td>180</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1103</th>
      <td>146.057666</td>
      <td>10.279459</td>
      <td>0.385467</td>
      <td>10.279459</td>
      <td>0.138992</td>
      <td>0.341144</td>
      <td>0.138992</td>
      <td>-0.613237</td>
      <td>-0.385467</td>
      <td>4.00098</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>C[C@H](c1nc2cnc3c(c2n1C4CCC(CC4)CCC#N)cc[nH]3)O</td>
      <td>Avicenna-JamesWatt</td>
      <td>199</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1104</th>
      <td>133.327782</td>
      <td>10.193187</td>
      <td>0.385467</td>
      <td>10.193187</td>
      <td>0.138992</td>
      <td>0.157389</td>
      <td>0.138992</td>
      <td>-0.634183</td>
      <td>-0.385467</td>
      <td>3.22078</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>C[C@H](c1nc2cnc3c(c2n1C4CCC(CC4)C#N)cc[nH]3)O</td>
      <td>ArthurEddington-LinusPauling</td>
      <td>199</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1105</th>
      <td>139.692724</td>
      <td>10.240533</td>
      <td>0.385467</td>
      <td>10.240533</td>
      <td>0.138992</td>
      <td>0.314854</td>
      <td>0.138992</td>
      <td>-0.622015</td>
      <td>-0.385467</td>
      <td>3.61088</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>C[C@H](c1nc2cnc3c(c2n1C4CCC(CC4)CC#N)cc[nH]3)O</td>
      <td>MarianoArtigas-RichardFeynman</td>
      <td>199</td>
      <td>3</td>
    </tr>
    <tr>
      <th>546</th>
      <td>148.927912</td>
      <td>12.326558</td>
      <td>0.393567</td>
      <td>12.326558</td>
      <td>0.236881</td>
      <td>0.205146</td>
      <td>0.236881</td>
      <td>-1.125546</td>
      <td>-0.393567</td>
      <td>-1.72870</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>CCCC(C(=O)N[C@@H]1[C@@H]([C@H](O[C@H]1n2cnc3c2...</td>
      <td>JamesClerkMaxwell-GottfriedLeibniz</td>
      <td>218</td>
      <td>3</td>
    </tr>
    <tr>
      <th>547</th>
      <td>148.927912</td>
      <td>12.378086</td>
      <td>0.393567</td>
      <td>12.378086</td>
      <td>0.237148</td>
      <td>0.080400</td>
      <td>0.237148</td>
      <td>-1.134309</td>
      <td>-0.393567</td>
      <td>-1.87280</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>CC(C)[C@@H](C(=O)N[C@@H]1[C@@H]([C@H](O[C@H]1n...</td>
      <td>AlbertEinstein-GottfriedLeibniz</td>
      <td>218</td>
      <td>3</td>
    </tr>
    <tr>
      <th>548</th>
      <td>155.292854</td>
      <td>12.536974</td>
      <td>0.393567</td>
      <td>12.536974</td>
      <td>0.237158</td>
      <td>0.029474</td>
      <td>0.237158</td>
      <td>-1.134521</td>
      <td>-0.393567</td>
      <td>-1.48270</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>CC[C@H](C)[C@@H](C(=O)N[C@@H]1[C@@H]([C@H](O[C...</td>
      <td>Anaximander-HenryCavendish</td>
      <td>218</td>
      <td>3</td>
    </tr>
    <tr>
      <th>188</th>
      <td>158.025073</td>
      <td>11.988379</td>
      <td>0.496768</td>
      <td>11.988379</td>
      <td>0.407501</td>
      <td>0.019053</td>
      <td>0.407501</td>
      <td>-1.083643</td>
      <td>-0.496768</td>
      <td>2.51200</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>CC(C)NC(=O)O[C@@H]1CC[C@@](c2c1nnn2Cc3ccc(cc3)...</td>
      <td>Avicenna-JohnDalton</td>
      <td>219</td>
      <td>3</td>
    </tr>
    <tr>
      <th>189</th>
      <td>158.025073</td>
      <td>11.988379</td>
      <td>0.496768</td>
      <td>11.988379</td>
      <td>0.407501</td>
      <td>0.019053</td>
      <td>0.407501</td>
      <td>-1.083643</td>
      <td>-0.496768</td>
      <td>2.51200</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>CC(C)NC(=O)O[C@@H]1CC[C@](c2c1nnn2Cc3ccc(cc3)O...</td>
      <td>LouisdeBroglie-FrancisGalton</td>
      <td>219</td>
      <td>3</td>
    </tr>
    <tr>
      <th>190</th>
      <td>158.025073</td>
      <td>11.988379</td>
      <td>0.496768</td>
      <td>11.988379</td>
      <td>0.407501</td>
      <td>0.019053</td>
      <td>0.407501</td>
      <td>-1.083643</td>
      <td>-0.496768</td>
      <td>2.51200</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>CC(C)NC(=O)O[C@H]1CC[C@@](c2c1nnn2Cc3ccc(cc3)O...</td>
      <td>AlexanderFleming-CarlSagan</td>
      <td>219</td>
      <td>3</td>
    </tr>
    <tr>
      <th>831</th>
      <td>137.462123</td>
      <td>13.727927</td>
      <td>0.352008</td>
      <td>13.727927</td>
      <td>0.323307</td>
      <td>0.144989</td>
      <td>0.323307</td>
      <td>-0.291978</td>
      <td>-0.352008</td>
      <td>2.80530</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>C[C@@H](CC(=O)NCc1ccc2c(c1)NC(=O)N2)c3ccccc3F</td>
      <td>LouisPasteur-ThomasKuhn</td>
      <td>233</td>
      <td>3</td>
    </tr>
    <tr>
      <th>832</th>
      <td>136.772520</td>
      <td>13.606261</td>
      <td>0.348248</td>
      <td>13.606261</td>
      <td>0.323307</td>
      <td>0.268656</td>
      <td>0.323307</td>
      <td>-0.368969</td>
      <td>-0.348248</td>
      <td>2.71500</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>C/C(=C\c1ccccc1F)/C(=O)NCc2ccc3c(c2)NC(=O)N3</td>
      <td>WernerHeisenberg-RobertHooke</td>
      <td>233</td>
      <td>3</td>
    </tr>
    <tr>
      <th>833</th>
      <td>143.827065</td>
      <td>13.804605</td>
      <td>0.349565</td>
      <td>13.804605</td>
      <td>0.323307</td>
      <td>0.147304</td>
      <td>0.323307</td>
      <td>-0.293785</td>
      <td>-0.349565</td>
      <td>3.36630</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>C[C@@H](CC(=O)N[C@H](C)c1ccc2c(c1)NC(=O)N2)c3c...</td>
      <td>AageBohr-EmilFischer</td>
      <td>233</td>
      <td>3</td>
    </tr>
    <tr>
      <th>143</th>
      <td>135.810779</td>
      <td>12.106708</td>
      <td>0.477639</td>
      <td>12.106708</td>
      <td>0.346775</td>
      <td>0.073073</td>
      <td>0.346775</td>
      <td>-3.898899</td>
      <td>-0.477639</td>
      <td>1.66550</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>c1cc(ccc1CCNS(=O)(=O)c2ccsc2C(=O)O)C(=O)O</td>
      <td>JaneGoodall-AlexanderVonHumboldt</td>
      <td>258</td>
      <td>3</td>
    </tr>
    <tr>
      <th>144</th>
      <td>142.175721</td>
      <td>12.116470</td>
      <td>0.477639</td>
      <td>12.116470</td>
      <td>0.346775</td>
      <td>0.150715</td>
      <td>0.346775</td>
      <td>-3.863136</td>
      <td>-0.477639</td>
      <td>2.05560</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>c1cc(ccc1CCCNS(=O)(=O)c2ccsc2C(=O)O)C(=O)O</td>
      <td>EvangelistaTorricelli-ClaudiusPtolemy</td>
      <td>258</td>
      <td>3</td>
    </tr>
    <tr>
      <th>145</th>
      <td>129.445837</td>
      <td>12.105157</td>
      <td>0.477639</td>
      <td>12.105157</td>
      <td>0.346775</td>
      <td>0.072284</td>
      <td>0.346775</td>
      <td>-3.954005</td>
      <td>-0.477639</td>
      <td>1.62300</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>c1cc(ccc1CNS(=O)(=O)c2ccsc2C(=O)O)C(=O)O</td>
      <td>AlessandroVolta-JohannesKepler</td>
      <td>258</td>
      <td>3</td>
    </tr>
    <tr>
      <th>173</th>
      <td>145.727128</td>
      <td>11.780729</td>
      <td>0.550172</td>
      <td>11.780729</td>
      <td>0.311615</td>
      <td>0.003531</td>
      <td>0.311615</td>
      <td>-1.085590</td>
      <td>-0.550172</td>
      <td>0.87400</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>c1c(cc(c(c1[N+](=O)[O-])O)I)CC(=O)NCCCCCC(=O)[O-]</td>
      <td>JohnvonNeumann-ReneDescartes</td>
      <td>276</td>
      <td>3</td>
    </tr>
    <tr>
      <th>174</th>
      <td>126.465290</td>
      <td>11.681878</td>
      <td>0.502092</td>
      <td>11.681878</td>
      <td>0.310480</td>
      <td>0.017613</td>
      <td>0.310480</td>
      <td>-0.835222</td>
      <td>-0.502092</td>
      <td>1.60410</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>c1cc(c(cc1CC(=O)NCCCCCC(=O)O)[N+](=O)[O-])O</td>
      <td>CarlSagan-WillardGibbs</td>
      <td>276</td>
      <td>3</td>
    </tr>
    <tr>
      <th>175</th>
      <td>126.465290</td>
      <td>11.671878</td>
      <td>0.550172</td>
      <td>11.671878</td>
      <td>0.310480</td>
      <td>0.005082</td>
      <td>0.310480</td>
      <td>-1.085222</td>
      <td>-0.550172</td>
      <td>0.26940</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>c1cc(c(cc1CC(=O)NCCCCCC(=O)[O-])[N+](=O)[O-])O</td>
      <td>FlorenceNightingale-ErnstHaeckel</td>
      <td>276</td>
      <td>3</td>
    </tr>
    <tr>
      <th>210</th>
      <td>169.244099</td>
      <td>12.730152</td>
      <td>0.496758</td>
      <td>12.730152</td>
      <td>0.259824</td>
      <td>0.043983</td>
      <td>0.259824</td>
      <td>-0.381021</td>
      <td>-0.496758</td>
      <td>3.42917</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>[H]/N=C(\Cc1cccc(c1)OC)/NC(=O)c2ccc(cc2OC3CCNC...</td>
      <td>FrancisGalton-FrancescoRedi</td>
      <td>318</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
<p>50 rows × 111 columns</p>
</div>



# More ideas
* in case you've gotten this far you can explore
    * plotting the distribution of features and comaring between y_pred classes
    * dimensionality rediction with principle components analysis http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
    * features selection: features that describe the most variation between classes
    



```python

```
