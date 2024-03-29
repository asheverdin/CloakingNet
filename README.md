# CloakingNet

This repo contains the parts of original implementation of:

Arsen Sheverdin, Francesco Monticone, and Constantinos Valagiannopoulos - ["Photonic Inverse Design with Neural Networks: The Case of Invisibility in the Visible"](https://journals.aps.org/prapplied/abstract/10.1103/PhysRevApplied.14.024054)

<p float="center">
<img width="300" src="media/schema.png" />
</p>
<!-- ![alt text](https://github.com/arsen-sheverdin/CloakingNet/blob/master/media/schema.png "Logo Title Text 1") -->

## How to run the code

### Dependencies 

- [Conda with Python](https://www.anaconda.com)  
- Install the environment `cloaking_net`

  ```bash
  bash install_dependencies.sh
  ```
- To activate the environment, run:
  ```bash
  source activate cloaking_net
  ```
- In order to donwload the data, run:
  ```bash
  bash download_dataset.sh
  ```

### Usage


- To train and reproduce the results of sphere dimensions' generation, run:
  ```bash
  python main.py
  ```
- View all avaiable command-line arguments by running:

  ``` bash
  python main.py --help
  ```    

## Cite

Please cite our paper if you found this code useful and use it in your own work:
```
@article{PhysRevApplied.14.024054,
  title = {Photonic Inverse Design with Neural Networks: The Case of Invisibility in the Visible},
  author = {Sheverdin, Arsen and Monticone, Francesco and Valagiannopoulos, Constantinos},
  journal = {Phys. Rev. Applied},
  volume = {14},
  issue = {2},
  pages = {024054},
  numpages = {10},
  year = {2020},
  month = {Aug},
  publisher = {American Physical Society},
  doi = {10.1103/PhysRevApplied.14.024054},
  url = {https://link.aps.org/doi/10.1103/PhysRevApplied.14.024054}
}
```




