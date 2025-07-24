# Readme Keltische Münzbildinterpolation
- Google Drive Link der Modelle:
  - best_model.pth für die Interpolation
  - final_model.pth für das Clustering
  - **beide Modelle sind nur für die Münzvorderseiten geeignet**

 - Starten des Streamlit Dashboards: 
   - der Pfad des Modells muss in der **main.py** angegeben werden, wenn nicht beide Dateien im selben Ordner liegen
   - **streamlit run main.py** startet das Dashbord

- Verwenden des Dashboards:
  1. Zwei Münzbilder hochladen
  2. Den Stempel des linken Bildes passend zum rechten ausrichten
  3. Zwischen den zwei Bildern mittels Interpolationsfakter-Slider interpolieren
  - Optional:
    - Einzelbilder herunterladen
    - Interpolationsreihe mit beliebiger Schrittzahl erstellen und herunterladen
