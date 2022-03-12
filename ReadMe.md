# SACI2022:  Konferencia cikkhez írt programom

A cikk az overleaf-ben érhető el. Röviden arrról szól, hogy van egy nagyon hosszú idősorunk.<br>
Ebből az idősorból képezzük egy előre csatolt neurális hálónak (Feed Foreward Neural Network - FFNN)<br>
a bemeneteit, ahol a bemenetek a t-1, t-2, ..., t-30 időpontban mért értékek.<br>
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
A virtuális gépeket a cloud init mechanizmuson keredsztül egy a cloud_init_driver.txt és a worker_init.txt fájlok segítségével érdemes létrehozni.
Ennek hatására a virtuális gépekre felkerülnek a szükséges fájlok, függőségek, és elindul rajta az adott szolgáltatás.

A driver-t vezérlő Driver.ipynb futtatásához a driveren futó Jupyter Notebook alkalmazásra is szükség van.
A telepítés során ez is felkerül a Driver Virtuális Gépre.
Ha mégsem indulna el akkor lehet manuálisan is indítani az adott szolgáltatást.

**Fontos** megjegyezni, hogy a Driver gép távoli eléréséhez szükséges hozzárendelni a *Floatin IP* címet. Egyéb esetben nem fogjuk tudni kívűlről elérni a szolgáltatást.

## Kiegészítő lépés a telepítéshez

Ha szeretnénk jelszóval védeni a Jupyter Notebook alkalmazást akkor szükséges kiadni az alábbi parancsokat a Driver gépen:

```
source notebook/bin/activate
jupyter notebook password
```

# Manuális indítás

A NAPI_INDITAS.txt fájlban leírtam, hogy milyen parancsokat kell kiadni az egyes gépeken, hogy a programot megfelelően használni tudjuk.


# Hogyan működik a program

Szinte mindent a Driver.ipynb Jupyter Notebookon keresztül lehet vezérelni és beállítani.
Ez alól egy kivétel van, hogy melyik adat fájlt olvassa be a rendszer. Ez fixen be van égetve a programba és nem is volt cél a kivezetése.

1. Végezzük el a telepítést.
2. Ha ellenőriztük, hogy fut a Jupyter Notebook szolgálatatás akkor lépjünk be a Driver gépre az erőforrásunkhoz rendelt *Floatin IP* címen keresztül.
3. 



