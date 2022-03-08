ReadMe.md

A Driver példányosításánál a cloud_init_driver.txt fájlt kell használni.

-----

Ezzel a scripttel kell létrehozni a virtuális gépet

Rá kell rakni még a Floating IP címet.

Ha beléptem, akkor már minden telepítve van, de le kell jelszavazni a notebookot.


A Driverre felteszek egy Jupyter Notebookot is, onnan fogom vezérelni.

Passwordözzük le
```
jupyter notebook password
```


Csinálok neki egy indítót `run-jupyter.sh`
```
mkdir driver
```

Lerántom a Github repóból a korábbi munkámat. (nem akarom clonozni az egész repot)<br>
Privát Repo ugyhogy ez csak nekem és csak **tokennel** fog menni ami mindíg változik (JoDeMiro)

```
cd driver
wget https://raw.githubusercontent.com/JoDeMiro/SACI22/main/SACI22_019.ipynb?token=token -O SACI_019.ipynb
cd ..
```

```
touch run-jupyter.sh

echo "#!/bin/bash" >> ~/run-jupyter.sh
echo "source /home/ubuntu/notebook/bin/activate" >> ~/run-jupyter.sh
echo "cd driver" >> ~/run-jupyter.sh
echo "jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --config=~/.jupyter/jupyter_notebook_config.py" >> ~/run-jupyter.sh

chmod u+x run-jupyter.sh
```




