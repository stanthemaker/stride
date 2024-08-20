from stride import *
import numpy as np
import os
import json
import argparse
import matplotlib.pyplot as plt
import math
import numpy.random as random


"""

To run the script within the Mosaic runtime, we will need to use the command:

    mrun -nw 4 -nth 6 python fwi_noise.py [json config file]

"""

def parse_args():
    parser = argparse.ArgumentParser(description="run fwi")
    parser.add_argument("json_file", type=str, help="Path to the json file.")
    return parser.parse_args()


def ellipt_coordinates(num, radius):  # centered at [0,0]

    angles = np.linspace(0, 2 * np.pi, num, endpoint=False)
    geometry = np.zeros((num, 2))
    for index, angle in zip(range(num), angles):

        geometry[index, 0] = radius[0] * np.cos(angle)
        geometry[index, 1] = radius[1] * np.sin(angle)

    return geometry


async def main(runtime):
    args = parse_args()
    with open(args.json_file, "r") as file:
        config = json.load(file)
    foldername = os.path.basename(args.json_file).split('.')[0]
    dx = config["dx"]
    cfl = config["cfl"]
    num_source = config["num_source"]
    num_receiver = config["num_receiver"]
    num_iters = config["num_iters"]
    max_freqs = config["max_freqs"]  # [0.3e6, 0.4e6]
    batchsize = config["batchsize"]
    map_path = config["map_path"]
    f_centre = config["source_centerfreq"]
    acq_folder = config["acq_folder"]
    acq_name = config["acq_name"]
    noise_type = config["noise_type"]

    if noise_type == "shift":
        lambda_factor = config["lambda_factor"]
    elif noise_type == "gaussian":
        sigma= config["sigma"]
    elif noise_type =="sensitivity":
        scale_range= config["scale_range"]

    dx_scale = dx / 0.3e-3
    cfl_scale = cfl / 0.5
    sos = 1500.0

    output_folder = os.path.join(os.getcwd(), foldername)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # basics setting
    grid_size = int(2100 / dx_scale)
    space = Space(
        shape=(grid_size, grid_size), extra=(100, 100), absorbing=(80, 80), spacing=dx
    )  # wavelenght : 3e-03 meter
    time = Time(
        start=0.0e-7, step=cfl * dx / sos, num=int(10000 / dx_scale / cfl_scale)
    )
    grid = Grid(space, time)
    problem = Problem(
        name="fullbodyring", space=space, time=time, output_folder=output_folder
    )

    # define transducer geometry
    problem.transducers.default()
    geometry_type = "elliptical"
    problem.geometry.default(geometry_type, num_receiver, radius=[0.28, 0.28])
    receivers = problem.geometry.locations

    offset = problem.geometry.num_locations
    coords_offest = grid_size / 2 * dx
    coordinates = ellipt_coordinates(num_source, [0.22, 0.22])
    for index in range(coordinates.shape[0]):
        _coordinates = coordinates[index, :]
        if len(_coordinates) != problem.geometry.space.dim:
            _coordinates = np.pad(_coordinates, ((0, 1),))
            _coordinates[-1] = problem.geometry.space.limit[2] / 2

        problem.geometry.add(
            index + offset,
            problem.geometry._transducers.get(0),
            _coordinates + coords_offest,
        )
    sources = problem.geometry.locations[offset:]

    # ------------------------ Forward problme ------------------------ #
    t = np.arange(0, time.num * time.step, time.step)
    b = 2 / f_centre
    delay = 0.03 * time.num * time.step
    wave = np.exp(-((t - delay) ** 2) / b**2) * np.cos(
        2 * np.pi * f_centre * (t - delay)
    )

    for source in sources:
        problem.acquisitions.add(
            Shot(
                source.id,
                sources=[source],
                receivers=receivers,
                geometry=problem.geometry,
                problem=problem,
            )
        )

    for shot in problem.acquisitions.shots:
        shot.wavelets.data[0, :] = wave

    runtime.logger.info(f"Total number of shots:{len(problem.acquisitions.shots)}")

    # define sos ground true map  /
    map = np.load(map_path)
    pad_height = (grid_size - map.shape[0]) // 2
    pad_width = (grid_size - map.shape[1]) // 2

    data = np.pad(
        map,
        pad_width=((pad_height, pad_height), (pad_width, pad_width)),
        constant_values=1500,
    )

    vp_true = ScalarField(name="vp", grid=problem.grid, data=data)
    problem.medium.add(vp_true)

    pde = IsoAcousticDevito.remote(
        grid=problem.grid,
        len=runtime.num_workers,
        boundary_type="complex_frequency_shift_PML_2",
        kernel="OT4",
    )

    try:
        problem.acquisitions.load(
            path=acq_folder, project_name=acq_name, version=0
        )
        runtime.logger.info("Acquistions loaded")
    except OSError:
        runtime.logger.info("Fail to load Acquistions")

    shot_ids = problem.acquisitions.remaining_shot_ids
    if not len(shot_ids):
        runtime.logger.warning("No need to run forward, observed already exists")
    else:
        runtime.logger.info(f"forward problem started")

        @runtime.async_for(shot_ids)
        async def loop(worker, shot_id):
            runtime.logger.info("Giving shot %d to %s" % (shot_id, worker.uid))
            worker_idx = int(worker.uid.split(":")[1] * 4 + worker.uid.split(":")[2])
            dev = 0 if worker_idx % 2 == 0 else 1

            sub_problem = problem.sub_problem(shot_id)
            wavelets = sub_problem.shot.wavelets
            traces = await pde(
                wavelets,
                vp_true,
                problem=sub_problem,
                runtime=worker,
                dump=True,
                platform="nvidia-acc",
                devito_args={"deviceid": dev},
            ).result()

            runtime.logger.info("Shot %d retrieved" % sub_problem.shot_id)
            shot = problem.acquisitions.get(shot_id)
            shot.observed.data[:] = traces.data
            if np.any(np.isnan(shot.observed.data)) or np.any(
                np.isinf(shot.observed.data)
            ):
                raise ValueError("Nan or inf detected in shot %d" % shot_id)

            shot.append_observed(path=problem.output_folder, project_name=problem.name)
            runtime.logger.info("Retrieved traces for shot %d" % sub_problem.shot_id)

        _ = await loop
        runtime.logger.info(f"forward problem completed")

    # ------------------------ Add noise ------------------------ #
    shot_ids = problem.acquisitions.shot_ids
    shots = [problem.acquisitions.get(id) for id in shot_ids]
    if noise_type =="gaussian":
        for shot in shots:
            for i in range(num_receiver):
                noise = np.random.normal(0, sigma, shot.observed.data[i].shape)
                shot.observed.data[i] = shot.observed.data[i] + noise
        runtime.logger.info(f"Gaussian noise applied")
        
    elif noise_type =="sensitivity":
        low, high =scale_range
        scaling_factors = np.random.uniform(low=low, high=high, size=(num_receiver, 1))
        for shot in shots:
            for i in range(num_receiver):
                shot.observed.data[i] = shot.observed.data[i] * scaling_factors[i]
        runtime.logger.info(f"Scaling noise applied")

    elif noise_type =="shift":
        lambda_min = sos / max_freqs[-1]
        sigma = lambda_min / lambda_factor
        for loc in problem.geometry.locations:
            x , y = loc.coordinates 
            perturb_x = random.normal(0,sigma)
            perturb_y = random.normal(0,sigma)
            loc.coordinates  = np.array([x+perturb_x , y +perturb_y])
        runtime.logger.info(f"Shift noise applied")

    elif noise_type == "none":
        runtime.logger.info(f"No noise is applied")
    else:
        runtime.logger.info("Unexpected noise type:", noise_type)
        exit()
        
    # ------------------------ FWI ------------------------ #
    vp = ScalarField.parameter(name="vp", grid=grid, needs_grad=True)
    vp.fill(1500.0)
    problem.medium.add(vp)

    loss = L2DistanceLoss.remote(len=runtime.num_workers)
    step_size = 10
    process_grad = ProcessGlobalGradient()
    process_model = ProcessModelIteration(min=1400.0, max=1600.0)

    optimiser = GradientDescent(
        vp, step_size=step_size, process_grad=process_grad, process_model=process_model
    )
    await pde.clear_operators()

    optimisation_loop = OptimisationLoop()
    runtime.logger.info(f"FWI started")
    num_blocks = len(max_freqs)

    # Start iterating over each block in the optimisation
    for block, freq in optimisation_loop.blocks(num_blocks, max_freqs):
        await adjoint(
            problem,
            pde,
            loss,
            optimisation_loop,
            optimiser,
            vp,
            num_iters=num_iters,
            select_shots=dict(num=batchsize, randomly=True),
            f_max=freq,
            # dump=True,
        )
    runtime.logger.info(f"FWI completed")


if __name__ == "__main__":
    mosaic.run(main)
