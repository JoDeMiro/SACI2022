# SACI2022:  Konferencia cikkhez írt programom

A cikk az overleaf-ben érhető el. Röviden arrról szól, hogy van egy nagyon hosszú idősorunk.<br>
Ebből az idősorból képezzük egy előre csatolt neurális hálónak (Feed Foreward Neural Network - FFNN)<br>
a bemeneteit, ahol a bemenetek a **t-1, t-2, ..., t-30** időpontban mért értékek.<br>
A neurális hálót nem tanítjuk.<br>
A kimenetén keletkező érték alapján azonban egy ágens döntéseket hoz.<br>
Az ágens döntései nyomán előálló értéket akarjuk maximalizálni.<br>

![equation](http://latex.codecogs.com/gif.latex?t-1%3D%5Ctext%20%7B%20sensor%20reading%20%7D)

![equation](http://latex.codecogs.com/gif.latex?O_t%3D%5Ctext%20%7B%20Onset%20event%20at%20time%20bin%20%7D%20t)


- <img src="https://latex.codecogs.com/gif.latex?P(s | O_t )=\text { Probability of a sensor reading value when sleep onset is observed at a time bin } t " />

![equation](http://latex.codecogs.com/gif.latex?P%28s%20%7C%20O_t%20%29%3D%5Ctext%20%7B%20Probability%20of%20a%20sensor%20reading%20value%20when%20sleep%20onset%20is%20observed%20at%20a%20time%20bin%20%7D%20t)

![equation](http://latex.codecogs.com/gif.latex?P%28s%20%7C%20O_t%20%29%3D%5Ctext%20%7B%20Probability%20of%20%20onset%20is%20observed%20at%20a%20time%20bin%20%7D%20t)

![equation](http://latex.codecogs.com/gif.latex?P%28s%20%7C%20O_t%20%29%3D%5Ctext%20%7B%20Probability%20%7D%20t%7B-1%7D)

![equation](http://latex.codecogs.com/gif.latex?t_{-1}%20t_{-2})


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
2. Ha ellenőriztük, hogy fut a Jupyter Notebook szolgálatatás akkor lépjünk be a Driver gépre az erőforrásunkhoz rendelt **Floatin IP** címen keresztül.
3. Nyissuk meg a Driver.ipynb fájlt.
4. Futtassuk le a programot és kész.

## Milyen paramétereket lehet beállítani a programban
- nRowsRead (int) maximum 5.6 millió (ToDo: erre irni egy lekezelést, ennél több sort ugyanis nem tartalmaz az adatunk)
- generation (int) ennyi generáicón keresztül fog futni a program
- factor (float) (2) ennyivel fogja leosztani a standard normális eloszlásból származó értékeket (kisebb érték nagyobb mutációt eredményez 0.2 azt jelenti, hogy a súlyokhoz adható véletlen szám (-5, +5) között fog mozogni. Ez már egy nagyon radikális mutáció. Eddig [1,2,10] értékeket használtan (erős, közepes, gyenge) mutáció imitálására.

## Egy fontos észrevétel a beolvasott adatok méretével kapcsolatban (ToDo)
Jelenleg nem jártam alaposan utána, hogy mi az okat annak a jelenségnek, hogy 490 ezer sornál többet olvasok be akkor a Worker meghal, kinyirja magát a Flask.

# Mit csinál a program a háttérben.

A program egy **server - workers** koncepciót használ. A gépek közötti adatcsere **REST API** segítségével történik.

A Driver program és Worker program is Flask modult használ.

A Driver soha nem használja az adatokat. Azokat csak a Workerek használják. A Worekerek ezeket az adatokat a felhőben való iniciálizációjuk során letöltik.
Egész pontosan klonozzák ezt a gitrepositoriumot és ez tartalmazza az adat fájlt is.

A Workerek nem állítanak elő semmilyen Machine Learning modelt. Ezt a Driver programtól kapják meg.
Ez minimális töbletet adatot jelent ahhoz képest, mintha csak a súlymátrixot küldeném át.
Viszont lényegesen leegyszerűsítette az implementációt.
A Worker programnak ezáltal nem kell implementálnia a NeuralNet osztályt, ezáltal lényegesen kevesebb hibalehetőség adódik, kevesebb konfiugurációs beállítást kell kezelnünk és a Worker program kódja is könnyebben átlátható lett.

## Hogyan zajlik az adatcsere

A Driver tart egy listát a Worker gépek lokális ip címével. Ez alapján tudja, hogy kikkel kell felvennie a kapcsolatot.
Az inicializáció során küldi el a Workereknek a saját elérhetőségét is.

A Driver program állítja elő a neurális hálók variánsait.

Ezeket (!) szekvenicálisan átküldi a Workerekenek az **http://<woerker_ip_address:8080>/Uploader** cimére ahol a Flask fogadja a fájlt, de-serializálja a modelt, és meghívja az evaluate(mlp) függvényt a kapott modellel.

A Driver kivezetett REST API-jai:
- aba
- baba
- kaba
- 



