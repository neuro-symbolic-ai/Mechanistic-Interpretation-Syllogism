# Mechanistic Interpretation Syllogism

![](imgs/pipeline.png)

This Jupyter Notebook contains the code to reproduce the results presented to the paper ["A Mechanistic Interpretation of Syllogistic Reasoning in Auto-Regressive Language Models"](https://arxiv.org/abs/2408.08590).

### Project Setup
```
conda create -n mechsyllogism python=3.9 -y
conda activate mechsyllogism
bash ./scripts/install_dependencies.sh

```

### System Dependencies
+ python >= 3.9.18
+ pytorch >= 2.2.0
+ plotly >= 5.19.0


### How to Use?
- Needed Installation
```
!pip install transformer_lens
!pip install circuitsvis
```
- Locate the `dataset` folder and `dataset_generator.py`, `helper_functions.py` files are in the same directory as the `main.ipynb` notebook file. 
- Run the cells in order to replicate our experiments in the `main.ipynb`.


## TODO

<details>
<summary>TODO list</summary>

[   ] System dependencies

[   ] Readme file

</details>
