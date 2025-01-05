"""Example for solving pose graph optimization problems loaded from `.g2o` files.

For a summary of options:

    python pose_graph_g2o.py --help

"""

import pathlib
from typing import Literal

import jax
import jaxls
import matplotlib.pyplot as plt
import tyro

import _g2o_utils


def main(
    g2o_path: pathlib.Path = pathlib.Path(__file__).parent / "data/input_M3500_g2o.g2o",
    linear_solver: Literal[
        "conjugate_gradient", "cholmod", "dense_cholesky"
    ] = "conjugate_gradient",
    termination: jaxls.TerminationConfig = jaxls.TerminationConfig(),
    trust_region: jaxls.TrustRegionConfig = jaxls.TrustRegionConfig(),
    sparse_mode: Literal["blockrow", "coo", "csr"] = "blockrow",
    output_dir: pathlib.Path = pathlib.Path(__file__).parent / "exp/figs",
    verbose: bool = True,
) -> None:

    # Parse g2o file.
    with jaxls.utils.stopwatch("Reading g2o file"):
        g2o: _g2o_utils.G2OData = _g2o_utils.parse_g2o(g2o_path)
        jax.block_until_ready(g2o)

    # Making graph.
    with jaxls.utils.stopwatch("Making graph"):
        graph = jaxls.FactorGraph.make(factors=g2o.factors, variables=g2o.pose_vars)
        jax.block_until_ready(graph)

    with jaxls.utils.stopwatch("Making solver"):
        initial_vals = jaxls.VarValues.make(
            (
                pose_var.with_value(pose)
                for pose_var, pose in zip(g2o.pose_vars, g2o.initial_poses)
            )
        )

    with jaxls.utils.stopwatch("Running solve"):
        solution_vals = graph.solve(
            initial_vals,
            trust_region=None,
            linear_solver=linear_solver,
            termination=termination,
            sparse_mode=sparse_mode,
            verbose=verbose,
        )

    # Plot
    f = plt.figure()
    if isinstance(g2o.pose_vars[0], jaxls.SE2Var):
        plt.plot(
            *(initial_vals.get_stacked_value(jaxls.SE2Var).translation().T),
            c="r",
            label="Initial",
        )
        plt.plot(
            *(solution_vals.get_stacked_value(jaxls.SE2Var).translation().T),
            # Equivalent:
            # *(onp.array([solution_poses.get_value(v).translation() for v in pose_variables]).T),
            c="b",
            label="Optimized",
        )

    # Visualize 3D poses
    elif isinstance(g2o.pose_vars[0], jaxls.SE3Var):
        ax = plt.axes(projection="3d")
        ax.set_box_aspect((1.0, 1.0, 1.0))  # type: ignore
        ax.plot3D(  # type: ignore
            *(initial_vals.get_stacked_value(jaxls.SE3Var).translation().T),
            c="r",
            label="Initial",
        )
        ax.plot3D(  # type: ignore
            *(solution_vals.get_stacked_value(jaxls.SE3Var).translation().T),
            c="b",
            label="Optimized",
        )

    else:
        assert False

    id = f"{g2o_path.stem}_{linear_solver}_{sparse_mode}_term_config_{termination.max_iterations}_{termination.cost_tolerance}_{termination.gradient_tolerance}_{termination.parameter_tolerance}_trust_region_{trust_region.lambda_initial}_{trust_region.lambda_factor}_{trust_region.lambda_min}_{trust_region.lambda_max}_{trust_region.step_quality_min}"
    output_dir.mkdir(parents=True, exist_ok=True)
    # plt.title(f"Optimization on {g2o_path.stem} linear_solver={linear_solver} sparse_mode={sparse_mode}")
    plt.legend()
    # plt.draw()
    plt.tight_layout()  # Ensure all elements are visible

    f.savefig(output_dir / f"{id}.pdf", bbox_inches='tight', dpi=300)
    # plt.show()
    plt.close()  # Close the figure to free memory


if __name__ == "__main__":
    tyro.cli(main)
