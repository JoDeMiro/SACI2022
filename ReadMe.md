ReadMe.md

sudo apt -y install openssh-client

wget --no-cache https://raw.githubusercontent.com/JoDeMiro/SACI2022/main/requirements.txt

pip install -r requirements.txt

-----
Innen vettem az öteltet

https://github.com/occopus/docs/blob/devel/tutorials/tensorflow-keras-jupyter/nodes/cloud_init_jupyter_server.yaml
-----

Ezzel a scripttel kell létrehozni a virtuális gépet

#cloud-config

packages:
  - git
  - openssh-client

runcmd:
- echo "-------> JoDeMiro starts."
- wget --no-cache https://raw.githubusercontent.com/JoDeMiro/SACI2022/main/requirements.txt
- pip install -r requirements.txt
- echo "-------> JoDeMiro finished."


A Driverre felteszek egy Jupyter Notebookot is, onnan fogom vezérelni.

```
sudo su
apt update && sudo apt -y upgrade
apt-get autoremove
pip install --upgrade pip
pip install virtualenv
exit
virtualenv notebook
cd notebook
source bin/activate
pip install jupyter
```

Passwordözzük le
```
jupyter notebook password
```


Csinálok neki egy indítót `run-jupyter.sh`
```
mkdir driver
```

Lerántom a Github repóból a korábbi munkámat. (nem akarom clonozni az egész repot)<br>
Privát Repo ugyhogy ez csak nekem és csak tokennel fog menni ami mindíg változik (JoDeMiro)

```
cd driver
wget https://raw.githubusercontent.com/JoDeMiro/SACI22/main/SACI22_019.ipynb?token=mokuskerek_token -O SACI_019.ipynb
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




