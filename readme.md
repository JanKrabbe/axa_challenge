# Axa Challenge

### Inhaltsverzeichnis

- [Aufgabenstellung](#aufgabenstellung)
- [1. Brainstorming](#1-brainstorming)
- [2. Datenanalyse](#2-datenanalyse)
- [3. Modellerstellung](#3-modellerstellung)
- [4. Implementierung in Beispielanwendung](#4-implementierung-in-beispielanwendung)
- [Projektinstallation](#projektinstallation)


### Aufgabenstellung

Daten eines New Yorker Fahrradverleihs (Citibike) wertstiftend nutzen. Dabei z. B. 
Kooperationsmöglichkeiten mit Versicherung skizzieren und öffentliche Verkehrsunfalldaten als 
zusätzliche Datenquelle verwenden.


### 1. Brainstorming

In einem ersten Schritt wurde ein allgemeines Brainstorming durchgeführt, welches in 
`brainstorming.md` dokumentiert ist. Das hauptsächliche daraus entstandene und in der Datei 
genauer beschriebene **Szenario**, das in der Challenge verfolgt wird, ist: 

**Jedem Kunden soll zu Beginn einer Fahrt eine Zusatzversicherung (umfasst eigene Gesundheit und
Haftpflicht) angeboten werden. Der Preis dieser Versicherung wird individuell zu Beginn der 
aktuellen Fahrt festgelegt.**

### 2. Datenanalyse

Zunächst wurden die Daten geladen und betrachtet, damit auch die nötigen Vorverarbeitungsschritte 
identifiziert werden konnten. Diese Betrachtung ist in `data_inspection.ipynb` dokumentiert und 
war die Grundlage für die Entwicklung der Datensatzklassen im `datasets/` Ordner.

Danach wurde die explorative Datenanalyse aus `explorative_data_analysis.ipynb` durchgeführt, die 
hauptsächlich für Citibike interessante Erkenntnisse liefern könnte und teilweise Ansatzpunkte 
liefert, wenn einheitliche Versicherungspreise für alle Citibike Nutzer kalkuliert werden sollen. 
In `danger_analysis.ipynb` wird die Gefährlichkeit einzelner Stadtgebiete untersucht, wobei unter 
anderem die NYPD Crash Daten mit den Citibike Daten verknüpft werden.

### 3. Modellerstellung

Eine Möglichkeit wäre es, ein **Modell** zu erstellen, das **basierend auf der Startposition und 
-zeit sowie dem Fahrrad- und Nutzertyp einen Gefahrenscore vorhersagt, der zur Berechnung des 
Versicherungspreises eingesetzt werden könnte**. Ein Ansatz zur Berechnung des Groundtruth 
Gefahrenscores wird in `approach_routecalculation.ipynb` beschrieben, konnte aber im Rahmen der 
Challenge nicht weiter verfolgt werden. 

In `crash_model_fitting.ipynb` wurden mehrere einfache Scikit-learn Modelle gefittet, welche aus 
einem gegebenen Paar aus Ort und Tageszeit eine Unfallhäufigkeit vorhersagen. Dazu wurden eine 
einfache Hyperparametersuche und eine kurze Evaluierung durchgeführt. Das beste dieser Modelle wurde
gespeichert, damit es in einer Beispielanwendung verwendet werden kann. 

### 4. Implementierung in Beispielanwendung

Um die Verwendung eines gefitteten Modells zu demonstrieren, wird das Modell aus 3. in 
`insurance_calculation.ipynb` zur Bestimmung beispielhafter Versicherungspreise für einzelne Fahrten 
angewendet.

## Projektinstallation

Die Challenge wurde mit Python 3.13.2 bearbeitet. 

Über die `requirements.txt` Datei können die benötigten Pakete installiert werden: 

```
pip install -r requirements.txt
```

Mit 

```
python -m pytest
```

kann geprüft werden, ob alle Klassen wie geplant ausgeführt werden können.

Die Daten können unter folgenden Links bezogen werden, wobei von den Citibike Daten der erste 
Ausschnitt des Dezembers 2023 verwendet wurde: 

- https://s3.amazonaws.com/tripdata/index.html
- https://data.cityofnewyork.us/Public-Safety/Motor-Vehicle-Collisions-Crashes/h9gi-nx95/about_data