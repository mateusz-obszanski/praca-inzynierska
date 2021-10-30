# Koncepcja

## Multi-agent
Multiple agent tabu search - z supervisorem - planowanie wyboru punktów do wykluczenia/dodania do listy do obsłużenia przez jednego salesmana. Punkty mogą być przypisane różnym salesmanom.

## Problem
Multiple TSP - n salesmanów do m punktów - algorytm genetyczny. Każdy scheduled agent z tabu search mógłby użyć genetycznego dla swojej puli punktów. Tabu search odpowiadałby wyłącznie za podział punktów.

Dla symulacji każda droga ma wylosowane wcześniej odchylenie standardowe czasu przebycia/kosztu. Możliwe rozszerzenie o randomizację procesów w wierzchołkach.

# Problemy
### Problem 1
tsp odwiedza tylko raz każdy wierzchołek. Tabu search mógłby działać iteracyjnie - zamiast pojedyńczego podziału, symulowany jest multipodział - salesmani, którzy nie przekroczyli zadanego czasu działania mają ponownie przydzielane wierzchołki, aż do momentu gdy wszyscy przekroczą zadany czas.
Na koniec iteracji zwracana jest lista wolnych salesmanów i dopóki nie jest pusta, algorytm wykonuje się ponownie. Rodzi to problem 2.
Ewentualnie przy podziale niektóre wierzchołki mogą być dublowane i połączenia między kopiami usuwane. Wtedy nie trzeba modelu iteracyjnego i rozwiązany jest problem 3.

### Problem 2
 - czy dzielić te same listy tabu pomiędzy iteracjami? chyba nie. Jeśli tak, to może przetransformowane/częściowe/tylko krótko-długo-etc. terminowa?

### Problem 3
 - salesmani wracają do punktu wyjścia.

### Problem 4
 - chromosomy kodują decyzje, różna ilość decyzji dla każdego salesmana - różna długość chromosomów do crossoveru. Jak je dopasować?
 Rozwiązanie - stała długość chromosomu, -1 jako filler. Wtedy występuje problem 6

### Problem 5
 - w crossoverze do wyboru rezultatu porównywać dystans/czas czy funkcję kosztu? Jeśli drugie, porównywać koszt dla symulacji wykonywanych do którego momentu, by porównanie miało znaczenie? Można brać tylko te wierzcuołki pod uwagę dla kosztu, które są przypisane do salesmana

### Problem 6
Postać chromosomu:
\[1, 5, 7, 3, 8, 80, 2, 7, 4\ \| 3, 7]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^   ******
sekwencja wierzchołków          punkty podziału wierzchołków między salesmanów

Dla fillerów (np. -1) - jak podzielić chromosom? Co w przypadku wstawienia wierzchołka lub usunięcia?
Rozwiązanie - odpowiednio zredukować (odjąć) od części drugiej.

### Problem 7
asymetria (wiatr na drogach), niespełnienie warunku trójkąta dla kosztów

### Problem 8
Redundancja obliczeń.
Rozwiązanie - cache, ale jak? Jeśli zmiana następuje w jednym miejscu, od pewnego punktu w czasie symulacji
cachowanie nie ma sensu, trzeba by ten punkt wyznaczać. Przy pierwszej symulacji rozwiązania można dodawać etykiety, np. czas rozpoczęcia.
Rozwiązanie 2 - tabu search dla mutacji i crossoverów, generalnie dla losowań


### Problem 9
Determinizm symulacji
Rozwiązanie - zapamiętywanie ziarna w configu. Osobne generatory liczb dla symulacji (z resetowaniem ziarna, choć tutaj może nie ma losowości) i dla agentów.


## Uwagi
Dla pierwszej iteracji, problem może być single lub multiple depot tsp (startują z tego samego/różnego miejsca), potem tylko multiple depot
Prawdopodobnie potrzebne **multidepot multicriteria tsp** (conflicting criteria - np. czas realizacji i koszt/zysk w wierzchołkach)
Dla multikryterium, użyć punktu/obszaru Pareto - różne wagi na różne kryteria
Zamiast dystansu, czas przebycia (jest zmieniany w różnych chwilach)