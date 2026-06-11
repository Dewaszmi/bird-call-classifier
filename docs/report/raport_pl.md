# Raport techniczny projektu klasyfikatora odgłosów ptaków

## Wprowadzenie, zakres i motywacja

Zakresem projektu było wytrenowanie sieci neuronowej w stylu VGG do poprawnej identyfikacji wybranych gatunków ptaków polskiej fauny na podstawie nagrań odgłosów. Inspiracją były wczesne implementacje VGG (ok. 2014 r.) oraz ich zastosowania w klasyfikacji danych, zwłaszcza obrazów.

Z tego powodu jedną z charakterystycznych cech naszej implementacji jest sposób formatowania danych przed przekazaniem ich do sieci — model trenuje bezpośrednio na obrazach, a nie na surowym audio, konkretnie na mel-spektrogramach. Jest to zatem w fundamentalnym sensie model klasyfikacji obrazów, a nie audio.

Wytrenowany model umożliwia inferencję na niestandardowych nagraniach dostarczonych przez użytkownika; udało nam się z powodzeniem zidentyfikować żywe nagranie ptaka z pobliskiego lasu.

## Pozyskiwanie danych

Pierwszym krokiem potoku jest pozyskanie surowych danych treningowych w postaci krótkich (do 30 sekund) klipów audio pobieranych ze strony [xeno-canto](https://xeno-canto.org/). Górny limit czasu był określany podczas wywołań API, aby uniknąć pobierania zbyt długich nagrań, które spowalniały proces downloadu.

### Obawy dotyczące różnic w jakości nagrań

Zbiór danych obejmuje 18 gatunków po 200 nagrań każdy, co okazało się wystarczające do osiągnięcia przyzwoitych wyników treningu. Jedną z obaw związanych z użytecznością modelu trenowanego na danych z xeno-canto były możliwe różnice w jakości sprzętu nagrywającego. Ponieważ społeczność xeno-canto składa się z wielu zaangażowanych obserwatorów przyrody i ornitologów, rozważaliśmy, czy typowe uploady nie pochodzą ze sprzętu wysokiej jakości (być może z wbudowaną redukcją szumów). Trenowanie wyłącznie na tak czystych danych mogłoby dać system nieskuteczny wobec głośnych nagrań z telefonu — dokładnie takich, jakich używaliśmy my.

Z tego powodu toczyły się debaty, czy podczas preprocessingu nie należałoby dodawać sztucznego szumu lub artefaktów. Na szczęście w końcowym scenariuszu nie okazało się to poważnym problemem, prawdopodobnie dlatego, że wiele nagrań na xeno-canto pochodzi od amatorów ze sprzętem zbliżonym do możliwości zespołu.

## Preprocessing danych

### Konwersja do mel-spektrogramów

Po pobraniu surowe dane są konwertowane do dwuwymiarowych tablic reprezentujących mel-spektrogramy — powszechny sposób reprezentacji audio stosowany w klasyfikacji. Spektrogramy to obrazy wizualizujące zmianę w czasie na osi X oraz poszczególne docelowe częstotliwości na osi Y, przy czym poszczególne piksele wskazują moc danej częstotliwości w danym przedziale czasowym. Tablice są zapisywane jako macierze numpy w formacie .npy.

Domyślnie po konwersji macierze mają różne wymiary, ponieważ macierze reprezentujące krótsze klipy naturalnie mają mniej ramek czasowych (np. 30-sekundowy klip może dać tablicę 128×128, a 15-sekundowy — 128×64). Zmienna długość tablic nie stanowi problemu dla samego kodu treningowego, ponieważ implementacja dopuszcza różne wymiary macierzy. Jednak batching wymaga, aby macierze ładowane na GPU miały zgodne wymiary. Z tego powodu niedopasowane długości skutecznie wymuszają rozmiar batcha równy jeden, wyłączając główną zaletę równoległości GPU. Prowadziłoby to do treningu w stylu CPU, co jest bardzo nieefektywne.

### Różne podejścia do problemu batchingu

Efektywny trening wymagał znalezienia sposobu na zapewnienie jednakowego rozmiaru tablic w pojedynczym batchu. Oryginalny opis projektu sugerował ustawienie wszystkich spektrogramów na rozdzielczość 128×128, jednak ostatecznie zastosowano inne podejście.

#### Problem ze stretchowaniem czasowym

Pierwszą ideą było użycie biblioteki librosa w Pythonie do rozciągnięcia krótszych nagrań wzdłuż osi czasu do docelowej długości 30 sekund. Klipy o różnej długości byłyby jednak rozciągane o różne współczynniki, co wprowadzałoby niespójną rozdzielczość (kompresję) między próbkami.

Na przykład przy `hop_length` równym 512 klip o rzeczywistej długości 30 sekund zostałby pierwotnie przekształcony na ~1292 kosze czasowe mel, podczas gdy klip 5-sekundowy — tylko na ~215. Skutkowałoby to niespójną rozdzielczością na osi czasu: kolumna pierwszej macierzy obejmowałaby sześć razy mniejszy odcinek rzeczywistego czasu niż ta sama kolumna drugiej macierzy, co mocno utrudniałoby trening.

#### Padding ciszą

Dlatego zdecydowaliśmy się na dopełnianie krótszych klipów ciszą po prawej stronie (od końca nagrania) do pożądanej długości. Rozwiązało to problem niedopasowanych wymiarów, jednak nierozważne paddingowanie doprowadziło do kolejnego problemu. Ponieważ padding zawsze występował po prawej stronie, znaczna część przetwarzanych klipów zawierała wartościowe dane tylko we wczesnych fragmentach (blisko początku, czyli w pierwszych kolumnach macierzy), a późniejsze ramki czasowe były interpretowane jako cisza. Model uczył się wtedy błędnego wzorca, że istotne zdarzenia występują głównie na początku klipów, przez co zdarzenia bliżej końca mogły być ignorowane. Ta implementacja preprocessingu została nazwana strategią paddingu „naive”.

Opracowano alternatywną strategię o nazwie „length-bucketing”. Ponieważ osobne batche mogą różnić się rozmiarem macierzy, a istotne jest jedynie to, by macierze w pojedynczym batchu miały zgodne wymiary, klipy w jednym batchu muszą być dopełniane tylko względem siebie nawzajem. Na przykład gdy najdłuższy klip ma 20 sekund, a najkrótszy 16 sekund, maksymalna ilość dopełnionej ciszy to tylko 4 sekundy — znacznie mniej wpływająca na trening niż średni padding w strategii „naive”. Strategia ta grupuje (bucketuje) spektrogramy o podobnej szerokości, aby ładować je do jednego batcha, minimalizując wymagany padding.

Kolejną strategią do przetestowania była „masked-gap”, łącząca podejście „naive” z maskowaniem sztucznych regionów ciszy w warstwie GAP, przed warstwami w pełni połączonymi sieci. Teoretycznie pozwalałoby to niemal całkowicie zneutralizować problem paddingu przy zachowaniu równoległości GPU, jednak z powodu problemów z dostępnością GPU zespół nie zdążył tego dokładnie zbadać.

#### Padding naive vs length-bucketing

Jak szacowano, padding oparty na length-bucketingu dawał znacząco lepsze wyniki treningu kosztem dłuższego czasu treningu — z konieczności grupowania klipów o podobnej długości. Dlatego to właśnie to podejście jest domyślnie używane w skrypcie treningowym.

## Architektura sieci

Rdzeń architektury to zmodyfikowana wersja klasycznej implementacji VGG opisanej w [artykule Simonyan & Zisserman z 2014 r.](https://arxiv.org/abs/1409.1556), z nowoczesnymi modyfikacjami zapewniającymi stabilniejszy i szybszy trening.

#### Bloki VGG

Część konwolucyjna składa się z 4 bloków VGG, z których każdy zbudowany jest z dwóch nałożonych konwolucji 3×3, każdej z własną warstwą ReLU, po których następują standardowe warstwy max poolingu. Użycie dwóch filtrów 3×3 daje pole recepcyjne 5×5, ale przy mniejszej liczbie parametrów i dodatkowej nieliniowości dzięki dodatkowej warstwie aktywacji.
Między dwiema warstwami konwolucyjnymi dodano BatchNorm do normalizacji każdego kanału cech (reskalowanie na podstawie obliczonej średniej i wariancji), co utrzymuje aktywacje w stabilnym zakresie. Poprawia to szybkość treningu poprzez ograniczenie ryzyka zanikających / eksplodujących parametrów, a także wprowadza łagodną regularyzację. Tę modyfikację zaproponowali [Ioffe & Szegedy w 2015 r.](https://arxiv.org/abs/1502.03167).
Ustawienie hiperparametru stride na wartość 2 powoduje, że każdy blok redukuje wejście o połowę (dzieląc wysokość i szerokość na pół). W zamian liczba kanałów rośnie, więc późniejsze warstwy mają mniejsze wymiary przestrzenne, ale coraz więcej kanałów.

[Rozdziały 22–25 (CNN)](https://bnaskrecki.faculty.wmi.amu.edu.pl/nnets/_build/html)

#### Warstwy niekonwolucyjne

Po 4 blokach VGG zaimplementowano warstwę Global Average Pooling (GAP), uśredniającą wartości ze wszystkich pozostałych pozycji przestrzennych. Oryginalne VGG Simonyan / Zissermana używało zamiast tego wielu warstw w pełni połączonych, obejmujących znacznie więcej parametrów (domyślnie po 4096 neuronów każda). Sprawdzało się to dobrze przy większych i bardziej złożonych danych ImageNet, ale groziłoby znaczącym overfittingiem przy mniejszych mel-spektrogramach. Ponadto ze względu na implementację spektrogramów o zmiennej długości warstwy FC wymagałyby dodatkowych obejść, aby akceptować wejścia o różnych rozmiarach, podczas gdy warstwa GAP sprowadza wszystko do wektora o wymiarach \[B, 256, 1, 1\] (batch, kanał, wysokość, szerokość).

Po warstwie GAP implementacja niewiele różni się od oryginału: wymagana warstwa flatten, następnie dropout do regularyzacji, a na końcu pojedyncza warstwa w pełni połączona zwracająca surowe logity dla każdej klasy, które następnie trafiają do funkcji softmax i są zwracane jako szacowane prawdopodobieństwa.

#### Przykład

| Etap | Kształt wyjścia | Opis |
|------|-----------------|------|
| Wejście | (B, 1, 128, 128) | Obraz log-mel |
| Blok 1 | (B, 32, 64, 64) | Blok VGG 1 |
| Blok 2 | (B, 64, 32, 32) | Blok VGG 2 |
| Blok 3 | (B, 128, 16, 16) | Blok VGG 3 |
| Blok 4 | (B, 256, 8, 8) | Blok VGG 4 |
| AdaptiveAvgPool2d | (B, 256, 1, 1) | Warstwa GAP |
| Flatten | (B, 256) | Wektor cech |
| Dropout | (B, 256) | Regularyzacja |
| Linear (Fully Connected) | (B, num_classes) | Zwraca surowe logity |

*Przykładowa tabela pokazująca zmieniające się kształty wyjścia dla każdej warstwy w BirdVGG przy wejściu 128×128 (macierz reprezentująca 30-sekundowy klip audio) [kształt macierzy w formacie (batch_size, channels, height, width)]*

## Szczegóły treningu i wyniki

#### Specyfikacja hiperparametrów

Potok treningowy domyślnie używa 75 epok, choć 50 również wydaje się wystarczających do osiągnięcia przyzwoitych wyników. Początkowy learning rate wynosi 1e-4 i liniowo maleje do 1e-5 (jednej dziesiątej wartości początkowej) na końcu treningu. Tak duża wartość learning rate jest akceptowalna ze względu na stosunkowo niską złożoność problemu. Trening stosuje też weight decay na poziomie 1e-4 dla regularyzacji.

Optymalizatorem jest AdamW — de facto domyślny optymalizator we współczesnym deep learningu. Dane są dzielone w proporcjach 70/15/15 na zbiory treningowy, walidacyjny i testowy. [Rozdział 27](https://bnaskrecki.faculty.wmi.amu.edu.pl/nnets/_build/html/part8_optimization/ch27_optimizers.html)

#### Wyniki treningu

Spośród dwóch zaimplementowanych strategii batchingu opisanych wcześniej „length-bucketing” konsekwentnie przewyższał batching „naive”. Trening z length-bucketingiem trwał jednak znacznie dłużej, przy ciekawym zjawisku, że dodawanie kolejnych klas czasem skracało czas treningu mimo większej ilości danych do przetworzenia. Wynika to z tego, że więcej danych daje gęstszą dystrybucję długości klipów, co powoduje, że średni batch wymaga mniejszego paddingu i ma mniejszą szerokość, co zmniejsza rozmiar wejścia.

Trening przeprowadzono na pojedynczej karcie Nvidia GTX 1070; strategia naive zajmowała około 7 minut, a length-bucketing około 54 minut dla 75 epok.

Główną metryką dyplomową jest F-score, obliczany jako średnia harmoniczna precyzji i czułości, powszechnie stosowana w problemach klasyfikacji.

![Wykres F1](pictures/f1_chart.png)

*Macro F1 w trakcie treningu dla obu strategii przez 75 epok.*

![Wykresy loss](pictures/loss_charts.png)

*Straty treningowe i walidacyjne w trakcie treningu dla obu strategii przez 75 epok.*

## Inferencja

Uruchomienie inferencji przez skrypt predykcji zwraca 3 klasy o najwyższym prawdopodobieństwie wraz z ich procentami. Aby uwzględnić scenariusze niskiej pewności, model zwraca też znormalizowaną entropię Shannona oraz margines między prawdopodobieństwami pierwszej i drugiej klasy. Zaimplementowano obsługę niepewności: poprawna identyfikacja wymaga, aby prawdopodobieństwo klasy najwyższej wynosiło co najmniej 30%, a margines co najmniej 15%.

Ze względu na stosunkowo dużą liczbę klas i naturalny szum w nagraniach audio rzadko udaje się osiągnąć przewidywane prawdopodobieństwo wyższe niż około 60%. Obliczona entropia często ma wartość około 0,5, co może myląco sugerować niską pewność, nawet przy bardzo wysokim marginesie między pierwszą a drugą najbardziej prawdopodobną klasą. Dlatego zalecamy wykorzystywanie wartości marginesu zamiast klasycznej entropii do oceny pewności predykcji.

## Uwagi końcowe

Udało nam się wytrenować model klasyfikacji audio rozpoznający polską awifaunę na podstawie nagrań odgłosów. Możliwe są usprawnienia szybkości treningu; prawdopodobnie implementacja maskowania dopełnionej ciszy oraz dodatkowa augmentacja danych mogłyby poprawić osiągnięte wyniki. Mimo to jesteśmy zadowoleni z efektów.

*Przetłumaczono przy użyciu AI, oryginalny raport w pliku report_en.md*
