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
  

- synthesis / rasterizer:
  - world definition [with triangle primitives](src/pixr/synthesis/world_simulation.py) or [meshes](src/pixr/synthesis/world_from_mesh.py)
  - [view synthesis of a scene rasterizer](src/pixr/rasterizer/rasterizer_sequential.py)

- studies: 
  - [interactive visualization](studies/interactive_projections.py)
  - [rasterizer check](studies/interactive_rasterizer.py)
  - [differentiability check of splatting](studies/differentiate_forward_project.py) . :warning: so far splatting is not differentiable with regard to camera.