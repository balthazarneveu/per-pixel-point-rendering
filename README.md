# Per-pixel point rendering
- Program: [MVA Master's degree](https://www.master-mva.com/) class on [3D point clouds](https://www.caor.minesparis.psl.eu/presentation/cours-npm3d/). ENS Paris-Saclay.
- Author of this code: [Balthazar Neveu](https://www.linkedin.com/in/balthazarneveu/)
- Study of [ADOP: Approximate Differentiable One-Pixel Point Rendering](https://arxiv.org/pdf/2110.06635.pdf)


### Setup
Local install of `pixr`
```bash
pip install -e .
```


#### Code structure
- rendering:
  - [per pixel splatting](src/pixr/rendering/splatting.py)
  - [differentiability check](src/pixr/rendering/differentiate_forward_project.py)

- synthesis:
  - [world definition with triangle primitives](src/pixr/synthesis/world_simulation.py)
  - [view synthesis of a scene rasterizer]
  - interactive visualization
  - 