# Brainstorming AXA Challenge

**Ziel**: Citibike Daten wertstiftend nutzen, Kooperationsmöglichkeiten mit Versicherung skizzieren

**Verfügbare Daten**: 

- Von Citibike bereitgestellte Fahrten
- NYPD Crash Daten
- Es könnten noch viele zusätzliche Daten (wie die monatlichen Reports von Citibike, Wetterdaten oder 
Kriminalitätsstatistiken) verwendet werden, würde aber vermutlich den Rahmen der Challenge sprengen

### 1. Ideensammlung vor Betrachtung der Daten

**Versicherungskontext**:

- Fahrräder gegen Diebstahl und Vandalismus versichern
- Versicherung gegen Umsatzausfälle bei extremen Wetterlagen (z. B. lange Hitze oder Kälte bei der 
deutlich weniger Fahrrad gefahren wird als üblich)
- Fahrräder gegen Verschleiß versichern (ggf. gar nicht so relevant im Versicherungskontext)
- Versicherung für durch Fahrräder entstehende Umweltschäden (Bsp. E-Roller, die in Flüsse geworfen 
werden)

- Anbieten von Versicherung für jeden Kunden bei Unfällen (besonders in den USA relevant)   
    - Versicherung der Gesundheit
    - Haftpflichtversicherung
    - Bei Nutzung klassischer Fahrräder sollten andere Versicherungsbeträge verlangt werden als bei 
    E-Bikes (E-Bikes sind deutlich wertvoller und das Fahrverhalten unterscheidet sich auch)
    - Erwarteter Schadensbetrag ändert sich je nach Ort, Zeit, Wetter etc. Die Versicherung kann 
    entweder so kalkuliert sein, dass alle den gleichen Betrag zahlen oder jeder Kunde bekommt je
    nach Nutzung ein individuelles Angebot

**Allgemein wertstiftender Kontext**:

- Optimierung der Fahrradbereitstellung
    - Umverteilung, wenn nicht bei allen Stationen gleich viele Fahrten enden wie beginnen
    - Anzahl bereitgestellter Fahrzeuge bzw. Stationen in verschiedenen Gebieten
    - Vorhersagen über die Zeit treffen, damit Planung für die Zukunft möglich ist
    - Verschleiß berücksichtigen: Reparatur bzw. Neukauf planen, Qualität und Sicherheit 
    gewährleisten, Reparatururteile in angemessener Stückzahl bereithalten, Schwachstellen 
    identifizieren und Fahrräder verbessern
    - Wartung (inkl. Pumpen etc.) planen
    - Einfluss auf Stadtplanung nehmen (Fahrradfahrersicherheit etc.)
    - Bei bestimmten Stellen die Nutzer auf Gefahren hinweisen
    - Ggf. Optimierung von Rewardprogrammen
    - Marketingstrategie verbessern


### 2. Anforderungen an Challenge Lösung 

Die folgenden Dinge sollten aus der für die Challenge entwickelten Lösung erkennbar sein bzw. darin
umgesetzt werden:

- Betrachtung/ Verknüpfung beider Datenquellen
- Problemlösungsfähigkeiten
- Darstellung meiner Herangehensweise an ein Data Science Projekt
- Darstellen meiner Programmier- bzw. Modellierungsfähigkeiten 

### 3. Plan nach Betrachtung der Daten und Anforderungen

Bei Betrachtung der Daten fällt auf, dass für die meisten der oben genannten Ideen Informationen in 
den Daten fehlen. Aus den Daten an sich lassen sich keine direkten Unfälle mit Citibike in Verbindung 
bringen (bei nur einem Eintrag in der Crash Datenbank wurde Citibike erwähnt).

Es wurde sich auf das folgende Szenario festgelegt, da sich mit diesem die Anforderungen erfüllen 
lassen und es für den Versicherungskontext relevant ist (auch wenn fraglich ist, ob eine Versicherung
ein solches Angebot tatsächlich anbieten möchte).

**Szenario**

**Jedem Kunden soll zu Beginn einer Fahrt eine Zusatzversicherung (umfasst eigene Gesundheit und
Haftpflicht) angeboten werden. Der Preis dieser Versicherung wird individuell zu Beginn der 
aktuellen Fahrt festgelegt.**

Es werden keine (monatlichen) Tarife für einzelne Kunden angeboten, da aus den Daten keine 
kundenspezifischen Informationen hervorgehen. 

Für eine Preisberechnung liegen also ausschließlich die *Fahrtinformationen*
- Art des gemieteten Fahrrads (Klassisch/ E-Bike)
- Typ des Kunden (Casual/ Member)
- Startort der Fahrt
- Startzeit der Fahrt

vor.

Es ergibt sich folgender Spielraum für die Datenanalyse:

- Visualisierung von Fahrradverkehr, Unfällen und Unfällen in Abhängigkeit des Verkehrs
- Vorhersagen von Unfällen je nach Ort und Tageszeit
- Vorhersagen von Fahrdauern und Entfernungen aus den oben aufgelisteten *Fahrtinformationen*
- Verknüpfung: ggf. direktes Vorhersagen eines Gefahrenscores aus den oben aufgelistet 
*Fahrtinformationen*


Mit Berücksichtigung der Anforderungen und der zeitlichen Beschränkung der Challenge müssen 
vermutlich nicht alle Citibike Daten verwendet werden. Auch anhand eines Teils lässt sich die 
Vorgehensweise demonstrieren und die Datenmenge könnte auch im Nachhinein noch erhöht werden.