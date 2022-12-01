# Bloc_6

E-mail: mathieu1290@gmail.com

Video:

Dossier sur google drive qui contient les bases de données:

Description du projet: ce projet consiste à développer une application qui permet d'identifier le niveau d'agressivité verbale de certaines conversations. Cette 
application pourrait éventuellement servir dans des établissements scolaires pour détecter les lieux où ont lieu des actes de violence à travers des insultes.
Nous avons entrainé trois modèles (regression logistique, xgboost et réseau de neuronnes) à l'aide de trois bases de données contenant en tout 

Dossier Notebooks: dossier qui contient les notebooks utilisés pour faire le pre-processing, l'apprentissage supervisé (regression logistique, xgboost),
l'apprentissage profond avec un réseau de neuronnes.

Dossier Local_Deploy_With_Microphone: dossier qui contient tous les fichiers pour développer en local une application qui enregistre des communications par un micro 
dans un intervale de temps défini à l'avance et donne une note d'agressivité en utilisant trois modèles: regression logistique, xgboost et réseau de neuronnes.

Fichier APPinProduction.py: code de l'application qui a été mise en production sur streamlit, l'application permet de uploader un fichier wav puis donne une note
d'agressivité à chaque phrase détectée. 





