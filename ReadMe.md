# SACI2022:  Konferencia cikkhez írt programom

A cikk az overleaf-ben érhető el. Röviden arrról szól, hogy van egy nagyon hosszú idősorunk.<br>
Ebből az idősorból képezzük egy előre csatolt neurális hálónak (Feed Foreward Neural Network - FFNN)<br>
a bemeneteit, ahol a bemenetek a ![equation](http://latex.codecogs.com/gif.latex?t_{-1}%2c%20t_{-2}%2c%2e%2e%2e%2c%20t_{-29}%2ct_{-30}) időpontban mért értékek.<br>
A neurális hálót nem tanítjuk.<br>
A kimenetén keletkező érték alapján azonban egy ágens döntéseket hoz.<br>
Az ágens döntései nyomán előálló értéket akarjuk maximalizálni.<br>




Mivel ez az érték csak és kizárólag a neurális háló kimenetén előálló értéktől függ, ezért olyan súlyokat,<br>
vagy azokat a súlyokat keressük, amelyek nyomán a neurális háló kimenetén előálló értékek alapján<br>
az ágens döntései nyomán előálló érték a legnagyobb lesz.


# Hogyan kell használni a programot

A program három külön álló részből áll.
- driver (driver.py)
- worker (worker.py)
- Driver.ipynb (ezen keresztül vezéreljük a drivert)

# Telepítés

A program megköveteli, hogy az ELKH Cloudun létre legyenek hozva a Virtuális Gépek.
A virtuális gépeket a cloud init mechanizmuson keredsztül egy a `cloud_init_driver.txt` és a `cloud_init_worker.txt` fájlok segítségével érdemes létrehozni.
Ennek hatására a virtuális gépekre felkerülnek a szükséges fájlok, függőségek, és elindul rajta az adott szolgáltatás.

A driver-t vezérlő `Driver.ipynb` futtatásához a driveren futó **Jupyter Notebook** alkalmazásra is szükség van.
A telepítés során ez is felkerül a Driver Virtuális Gépre.
Ha mégsem indulna el akkor lehet manuálisan is indítani az adott szolgáltatást.

**Fontos** megjegyezni, hogy a Driver gép távoli eléréséhez szükséges hozzárendelni a **Floating IP** címet. Egyéb esetben nem fogjuk tudni kívűlről elérni a szolgáltatást.

## Kiegészítő lépés a telepítéshez

Ha szeretnénk jelszóval védeni a Jupyter Notebook alkalmazást akkor szükséges kiadni az alábbi parancsokat a Driver gépen:

```
source notebook/bin/activate
jupyter notebook password
```

# Manuális indítás

A `NAPI_INDITAS.txt` fájlban leírtam, hogy milyen parancsokat kell kiadni az egyes gépeken, hogy a programot megfelelően használni tudjuk.


# Hogyan használjuk a programot

Szinte mindent a Driver.ipynb Jupyter Notebookon keresztül lehet vezérelni és beállítani.
Ez alól egy kivétel van, hogy melyik adat fájlt olvassa be a rendszer. Ez fixen be van égetve a programba és nem is volt cél a kivezetése.

1. Végezzük el a telepítést.
2. Ha ellenőriztük, hogy fut a Jupyter Notebook szolgálatatás akkor lépjünk be a Driver gépre az erőforrásunkhoz rendelt **Floating IP** címen keresztül.
3. Nyissuk meg a Driver.ipynb fájlt.
4. Futtassuk le a programot és kész.

## Milyen paramétereket lehet beállítani a programban
- nRowsRead (int) maximum 5.6 millió (ToDo: erre irni egy lekezelést, ennél több sort ugyanis nem tartalmaz az adatunk)
- generation (int) ennyi generáicón keresztül fog futni a program
- factor (float) (2) ennyivel fogja leosztani a standard normális eloszlásból származó értékeket (kisebb érték nagyobb mutációt eredményez 0.2 azt jelenti, hogy a súlyokhoz adható véletlen szám (-5, +5) között fog mozogni. Ez már egy nagyon radikális mutáció. Eddig [1,2,10] értékeket használtan (erős, közepes, gyenge) mutáció imitálására.

## Hogyan állíthatom be a populáció számát
Röviden sehogy. Bővebben: Az van, hogy workers_addresses lista elemszámával egyenlő a populáció száma.

Más szavakkal a populáció szám nem választható, hanem rá van kötve a driver programban elhelyezett cimeket tartalmazó lista hosszával.

## Egy fontos észrevétel a beolvasott adatok méretével kapcsolatban (ToDo)
Jelenleg nem jártam alaposan utána, hogy mi az okat annak a jelenségnek, hogy 490 ezer sornál többet olvasok be akkor a Worker meghal, kinyirja magát a Flask.

# Mit csinál a program a háttérben.

A program egy **server - workers** koncepciót használ. A gépek közötti adatcsere **REST API** segítségével történik.

A Driver program és Worker program is Flask modult használ.

A Driver soha nem használja az adatokat. Azokat csak a Workerek használják.<br>
A Worekerek ezeket az adatokat a felhőben való iniciálizációjuk során letöltik.<br>
Egész pontosan klonozzák ezt a gitrepositoriumot és ez tartalmazza az adat fájlt is.<br>

A Workerek nem állítanak elő semmilyen Machine Learning modelt. Ezt a Driver programtól kapják meg.<br>
Ez minimális töbletet adatot jelent ahhoz képest, mintha csak a súlymátrixot küldeném át.<br>
Viszont lényegesen leegyszerűsítette az implementációt.<br>
A Worker programnak ezáltal nem kell implementálnia a NeuralNet osztályt, ezáltal lényegesen kevesebb hibalehetőség adódik, kevesebb konfiugurációs beállítást kell kezelnünk és a Worker program kódja is könnyebben átlátható lett.

## Hogyan zajlik az adatcsere

A Driver tart egy listát a Worker gépek lokális ip címével. Ez alapján tudja, hogy kikkel kell felvennie a kapcsolatot.
Az inicializáció során küldi el a Workereknek a saját elérhetőségét is.

A Driver program állítja elő a neurális hálók variánsait.

Ezeket direkt (!) szekvenicálisan átküldi a Workerekenek az **http://<woerker_ip_address:8080>/Uploader** cimére ahol a Flask fogadja a fájlt, de-serializálja a modelt, és meghívja az `evaluate(mlp)` függvényt a kapott modellel.<br>
Az `evaluate()` egy külön szálon indul el, ezáltal az **Uploader** REST API **200** OK választkodót ad vissza Drivernek.

A Driver miután mindegyik Workernek elküldte a modelt belekerül egy `while` ciklusba, amely addig igaz amíg az összes Workertől vissza nem kapjuk a kiszámított értéket.

Eközben a Workereken az `evaluate()` a `Trader` osztály egy példányát fehasználva kiszámolja az adott modelhez tartozó **fitness_score** értéket.<br>
Amint végez ezzel a számítással vissza küldi az eredményt a Driver ép **http://<driver_ip_address:8080>/Receiver** cimére.<br>

A Driver fenn tart egy számlálót arról, hogy hány Workertől kapta meg a választ.<br>
Amint meg van az összes válasz megszakítja a `while` ciklust és a program tovább lép a következő generáció kiszámítására.

Ennek első lépése, hogy kiválasztja a legjobban teljesítő modelt, a hozzá tartozó érték alapján.

Ennek a modelnek az alapján létrehozza az új generációt.

Az új generáció egyes elemeit ismét egyenként elküldi egy-egy Workernek.

Ezen a ponton a folyamat ismétli önmagát, amig el nem érjük az előre megadott generáció számát.

### A Driver kivezetett API-jai:
- [GET] testpoint(value=<str>)<br>
  `# resp = requests.get('http://192.168.0.xxx:8080/testpoint?value=123456789')`
- [GET] calltestpoint()<br>
  `# resp = requests.get('http://192.168.0.xxx:8080/calltestpoint)`<br>
  segítségével le tudom tesztelni, hogy egy adott Worker megfelelően üzemel-e.
- [GET] setup(generation=<str>, factor=<str>, dummy=<str>)<br>
  `# resp = requests.get('http://192.168.0.xxx:8080/setup?generation=5000&factor=2&dummy=ABC)`<br>
- [GET] initialize(_nRowsRead=<int>)<br>
  `# resp = requests.get('http://192.168.0.xxx:8080/initialize?_nRowsRead=98765)`<br>
- [GET] setupworkers()<br>
  `# resp = requests.get('http://192.168.0.xxx:8080/setupworkers)`<br>
- [GET] initializeworkers()<br>
  `# resp = requests.get('http://192.168.0.xxx:8080/initializeworkers)`<br>
- [GET] testworkerscalc()<br>
  `# resp = requests.get('http://192.168.0.xxx:8080/testworkerscalc)`<br>
  Ugyan azt a csomagot küldi el az összes workernek.<br>
  Azt vizsgálom, hogy ugyan azt az eredményt adja-e vissza az összes worker.<br>
  Ez is egy assert, ha nem ugyan az az eredmény akkor valahol nem stimmel valami és megállítom a programot.
- [GET] evolution2()<br>
  `# resp = requests.get('http://192.168.0.xxx:8080/evolution2)`<br>
  Gyakorlatilag ezen keresztül indítom el az Evolúciós keresést.

### A Woerker kivezetett  API-jai:
- [GET] setup (driver_ip=<str>, worker_id=<str>, nRowsRead=<str>, window=<str>, threshold=<str>)<br>
  `# resp = requests.get('http://192.168.0.xxx:8080/setup?worker_id=1&nRowsRead=2998&window=20&threshold=-1000')`
- [GET] initialize()
- [POST] uploader({'file': file}: <dict>)
- [GET] testpoint(value=<str>)<br>
  `# resp = requests.get('http://192.168.0.xxx:8080/testpoint?value=123456789')`
- [GET] update()<br>
   `# resp = requests.get('http://192.168.0.xxx:8080/update')`
  Leállítja a Flask service-t, letölti a GitHub repozitoriumból az legfrissebb verziót <'git pull'> parancson keresztül, újra indítja a Flask servicet.

# Known issues, bugs
  1. A Wokerek megrogynak ha a beolvasott sorok száma `nRowsRead` meghaladja a 800.000 sort.
  2. Amikor a Driveren keresztül inicializálom a Workereket akkor az szekvenciálisan kerül végrehajtásra.
  3. A `driver.py` a várakozás alatt a `while` ciklusba bele tettem egy timeoutot, hogy mindenképpen zárja a ciklust, ha meghalad egy előre beállított értéket. Jelenleg ez bele van égetve a programba és szűk keresztmentszet lehet, ha a Worker ténylegesen tovább számol.
  
# Saját magam figyelmébe ajánlom.
Nem érdemes bíbelődni az API tanulmányozásával. A Driver.ipynb Jupyter Notebook fájlt érdmes megnyitni és lépésenként végig menni rajta. Eléggé kézenfekvő a működése és sok megjegyzéssel tele tüzdelt fájl.
  
Magát a programot is elláttam a megjegyzéseimmel. Ez a mértékadó.
  
A Workereken soha nem kell paraméterezni a Neurális Hálót, mert a Driver a komplett modelt átküldi.<br>
A Worker oldalon csak ileszti a modelt az adatokra.
  
# ToDo
Csinálni egy olyan tesztet, hogy az egész párhuzamos helyett szekvenciálisan fut le.<br>
Ezt könnyen megtehetem, hiszen csak a workereknél az evaluation() metodust kell az uploader API-ban kivezetnem a külön szálról és akkor az API hívás bevárja a választ, ezáltal szekvenciálisan fut le.

# Further Work
Ha majd egyszer sok időm lesz akkor még a következő dolgokat lehet érdemes kipróbálni, a teljesség igénye nélkül:
- A Trader osztály statisztikáit is visszakértni a Jupyter Notebookba elemzés céljából
- A Trader osztályt tovább lehet fejelszteni: MaxDD, Corr(Equity)
- Mutáció vagyis a Randomer osztály továbbfejlesztése: (Mutation selection rate, Crossover, PointSplit vs. Coub inheritance)
- Az se lenne rossz ha minden egyet fitness score értéket visszaadná a Jupyternek elemzés céljából

