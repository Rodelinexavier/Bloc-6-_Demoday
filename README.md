# Bloc_6

E-mail: mathieu1290@gmail.com

Video:

Lien vers l'application: https://mmatthieu1290-hate-speech-detection-from-appinproduction-dupsz5.streamlit.app/

Dossier sur google drive qui contient les bases de données (les fichiers sont trop imposants pour être déposés sur GitHub: 
https://drive.google.com/drive/folders/1hsEim7Hmhzk_r9qbO9smL0goVBNMXC5g

Description du projet: ce projet consiste à développer une application qui permet d'identifier le niveau d'agressivité verbale de certaines conversations. Cette 
application pourrait éventuellement servir dans des établissements scolaires pour détecter les lieux où ont lieu des actes de violence à travers des insultes.
Nous avons entrainé trois modèles (regression logistique, xgboost et réseau de neuronnes) à l'aide de trois bases de données contenant en tout 

Dossier Notebooks: dossier qui contient les notebooks utilisés pour faire le pre-processing, l'apprentissage supervisé (regression logistique, xgboost),
l'apprentissage profond avec un réseau de neuronnes.

Dossier Local_Deploy_With_Microphone: dossier qui contient tous les fichiers pour développer en local une application qui enregistre des communications par un micro 
dans un intervale de temps défini à l'avance et donne une note d'agressivité en utilisant trois modèles: regression logistique, xgboost et réseau de neuronnes.

Dossier APP_in_production_streamlit: dossier qui contient tous les fichiers pour mettre en production sur streamlit une application qui lit un fichier wav contenant 
des conversations, le divise en phrase et donne à chaque phrase une note d'agressivité en utilisant trois modèles: regression logistique, xgboost et réseau de 
neuronnes.




