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
