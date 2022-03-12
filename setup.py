import setuptools  

with open("ReadMe.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(  
    name="elkhevolver",  
    version="0.1.1",  
    author="JoDeMiro",
    install_requires=['numpy>=1.14.2','gym>=0.10.4', 'redis>=2.10.6', 'timeout-decorator>=0.4.0', 'matplotlib>=3.0.3'],
    author_email="jr.istvan.pintye@gmail.com",  
    description="PyPI: A Python 3 Library for Distributed the Genetic Algorithm for a specific test.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JoDeMiro/SACI22022",
    packages=setuptools.find_packages(),
    python_requires=">=3.8",
    classifiers=[
          'Intended Audience :: Science/Research',
          'Operating System :: OS Independent',
          'Programming Language :: Python :: 3.8',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
          ]
    )