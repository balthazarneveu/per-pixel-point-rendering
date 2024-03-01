import torch
from pixr.rasterizer.rasterizer_sequential import shade_screen_space_sequential
from pixr.rasterizer.rasterizer_parallel import shade_screen_space_parallel


def shade_screen_space(
        cc_triangles: torch.Tensor,
        colors: torch.Tensor,
        depths: torch.Tensor,
        width: int, height: int,
        show_depth: bool = False,
        no_grad: bool = True,
        debug: bool = False,
        for_loop: bool = True,
        limit: int = -1
) -> torch.Tensor:
    if for_loop:
        return shade_screen_space_sequential(
            cc_triangles, colors, depths, width, height,
            show_depth=show_depth, no_grad=no_grad, debug=debug, limit=limit
        )
    else:
        # WARNING: This is not working properly!
        return shade_screen_space_parallel(
            cc_triangles, colors, depths, width, height,
            show_depth=show_depth, no_grad=no_grad, debug=debug, limit=limit
        )
