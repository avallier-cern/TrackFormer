from abc import ABC, abstractmethod
from torch.utils.data import DataLoader, Dataset, IterableDataset, get_worker_info
import torch
import math
import os
from tqdm.auto import tqdm
from rich.console import Console
from rich.progress import track

console = Console()
from src.datasets.utils import ParticleGun, Detector, EventGenerator
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import lightning as L
from pathlib import Path
import trackml.dataset
import pandas as pd


##################################################################

#################################
#           TOY TRACK           #
#################################


class ToyTrackDataset(IterableDataset):
    """
    Generates track data on the fly using ToyTrack module.
    See https://github.com/ryanliu30
    """

    def __init__(
        self, hole_inefficiency=0, d0=0.1, noise=0, lambda_=50, pt_dist=[1, 5]
    ):
        super().__init__()
        self.hole_inefficiency = hole_inefficiency
        self.d0 = d0
        self.noise = noise
        self.pt_dist = pt_dist
        self.detector = self._create_detector()
        self.particle_gun = self._create_particle_gun()

    def _create_detector(self):
        return Detector(
            dimension=2, hole_inefficiency=self.hole_inefficiency
        ).add_from_template("barrel", min_radius=0.5, max_radius=3, number_of_layers=10)

    def _create_particle_gun(self):
        return ParticleGun(
            dimension=2,
            num_particles=1,
            pt=self.pt_dist,
            pphi=[-np.pi, np.pi],
            vx=[0, self.d0 * 0.5**0.5, "normal"],
            vy=[0, self.d0 * 0.5**0.5, "normal"],
        )

    def __iter__(self):
        self.event_gen = EventGenerator(self.particle_gun, self.detector, self.noise)
        return self

    def __next__(self):
        # an event
        event = self.event_gen.generate_event()
        x = torch.tensor([event.hits.x, event.hits.y], dtype=torch.float).T.contiguous()
        return (
            x,
            torch.ones(x.shape[0], dtype=bool),
            torch.tensor([event.particles.pt], dtype=torch.float),
        )


class ToytrackDataModule(L.LightningDataModule):
    """ToyTrack Lightning Data Module"""

    def __init__(
        self,
        batch_size: int = 20,
        num_workers: int = 10,
        persistence: bool = False,
        pin_memory: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["_class_path"])
        self.dataset = ToyTrackDataset()
        console.rule("Streaming ToyTrack")

    def train_dataloader(self):
        return self._create_dataloader(self.dataset)

    def val_dataloader(self):
        return self._create_dataloader(self.dataset)

    def test_dataloader(self):
        return self._create_dataloader(self.dataset)

    def _create_dataloader(self, dataset):
        """Helper method to create a DataLoader."""
        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=self.collate_fn,
            persistent_workers=self.hparams.num_workers > 0
            and self.hparams.persistence,
            pin_memory=self.hparams.pin_memory,
        )

    @staticmethod
    def collate_fn(ls):
        """Batch maker"""
        x, mask, pt = zip(*ls)
        return (
            pad_sequence(x, batch_first=True),
            pad_sequence(mask, batch_first=True),
            torch.cat(pt).squeeze(),
        )


##################################################################

######################################
#       BASE Realistic Datasets       #
#######################################


class IterBase(IterableDataset, ABC):
    """Iterable Base class for TrackML and ACTS datasets.

    Attributes:
        folder (Path): directory containing dataset.
    """

    def __init__(self, dataset_dir, folder="train", dataset=None, **kwargs):
        self.path = Path(dataset_dir) / folder
        self.available_events = self._event_range()

        # Add kwargs to the class
        for key, value in kwargs.items():
            setattr(self, key, value)

    def _event_range(self):

        event_numbers = []
        for file in self.path.glob("*"):
            # Keep only files
            if not file.is_file():
                continue
            # Keep only files with the prefix "event"
            if not file.stem.startswith("event"):
                continue
            event_numbers.append(file.stem.split("-")[0])

        if not event_numbers:
            raise FileNotFoundError(
                "Uh-oh! Looks like there data files are missing ..."
            )

        return sorted(list(set(event_numbers)))

    @abstractmethod
    def _preprocessor(self, event: str):
        """preprocessing logic."""
        raise NotImplementedError

    @abstractmethod
    def _load_event(self, eventfiles):
        """loading logic."""
        raise NotImplementedError

    def __iter__(self):
        worker_info = get_worker_info()
        total_events = len(self.available_events)
        if worker_info is None:  # Single-process
            iter_start = 0
            iter_end = total_events
        else:
            # Split workload among workers
            per_worker = int(math.ceil((total_events) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, total_events)

        n_event_split = getattr(self, "n_events_split", None)

        for i in range(iter_start, iter_end):
            if n_event_split is None:
                event_files = self._load_event(self.available_events[i])
            else:
                event_files = self._load_event(self.available_events[i], n_event_split)
            processed_data = self._preprocessor(event_files)
            yield from processed_data


class RootIterBase(IterBase):
    """Iterable Base class for ROOT datasets."""

    def __init__(self, dataset_dir, folder="train", dataset=None, **kwargs):
        self.path = Path(dataset_dir) / folder
            
        # Find all ROOT files in the subdirectories
        print(f"Looking for root files in {self.path}")
        self.root_files = sorted(list(self.path.glob("*/*.root")))

        # Remove files starting with "performance"
        self.root_files = [
            f for f in self.root_files if not f.name.startswith("performance")
        ]
        
        if not self.root_files:
            raise FileNotFoundError(
                f"Uh-oh! Looks like there are no ROOT files in {self.path} ..."
            )
        
        super().__init__(dataset_dir, folder, dataset, **kwargs)
        
        if hasattr(self, "n_events_split"):
            self.available_events = sorted(
                list(
                    set(
                        int(event) // self.n_events_split * self.n_events_split
                        for event in self.available_events
                    )
                )
            )
        else:
            self.available_events = self._event_range()


    def _event_range(self):
        # Find the number of events in the ROOT files
        import uproot

        event_numbers = []
        for root_file in self.root_files:
            if not root_file.name.startswith("tracksummary_ambi"):
                continue
            with uproot.open(root_file) as f:
                keys = list(f.keys())
                if len(keys) == 1 or True:
                    keys = keys[0]

                
                event_number_key = [k for k in f[keys].keys() if "event" in k]
                assert (
                    len(event_number_key) == 1
                ), "Expected exactly one event number key, found: {}".format(
                    event_number_key
                )
                event_number_key = event_number_key[0]
                assert (
                    event_number_key == "event_nr"
                ), "Expected event number key to be 'event_nr', found: {}".format(
                    event_number_key
                )
                if hasattr(self, "n_events_split"):
                    assert self.n_events_split == len(
                        f[keys][event_number_key].array().tolist()
                    ), "Expected number of events to be {}, found: {}".format(
                        self.n_events_split,
                        len(f[keys][event_number_key].array().tolist()),
                    )
                self.n_events_split = len(f[keys][event_number_key].array().tolist())
                print(f"Found {self.n_events_split} events in {root_file}")
                event_numbers.extend(f[keys][event_number_key].array().tolist())

        unique_event_numbers = set(event_numbers)

        if( len(unique_event_numbers) != len(event_numbers) ):
            raise ValueError( "Duplicate event numbers found across ROOT files." )

        return sorted(list(unique_event_numbers))

    def _preprocessor(self, event: str):
        """preprocessing logic."""
        raise NotImplementedError

    def _load_event(self, eventfiles):
        """loading logic."""
        raise NotImplementedError


def convert_tree_to_dataframe(f, keys, branches_to_load=None, verbose=False):
    """
    Build a single DataFrame with rows per (event, sublist, elem)
    from mixed scalar, single-jagged [event][sublist], and double-jagged
    [event][sublist][elem] branches.

    Single-jagged fields are repeated (constant) across all elements of
    their corresponding sublist.
    """
    import awkward as ak
    import pandas as pd
    from collections import OrderedDict

    if branches_to_load is None:
        branches_to_load = f[keys].keys()
    ak_array = f[keys].arrays(library="ak", filter_name=branches_to_load)

    if verbose:
        print("Branch types:")
        for key in ak_array.fields:
            print(f"{key}: {ak.type(ak_array[key])}")

    # ---------- helpers ----------
    def jag_depth(t):
        d = 0
        cur = getattr(t, "content", None)
        while isinstance(cur, ak.types.ListType):
            d += 1
            cur = getattr(cur, "content", None)
            if d >= 3:
                return 3
        return d

    # ---------- partition ----------
    scalars = OrderedDict()
    singles = OrderedDict()  # [event][sublist]
    doubles = OrderedDict()  # [event][sublist][elem]

    for k in ak_array.fields:
        t = ak_array[k].type
        d = jag_depth(t)
        if d == 0:
            scalars[k] = ak_array[k]
        elif d == 1:
            singles[k] = ak_array[k]
        elif d == 2:
            doubles[k] = ak_array[k]
        elif verbose:
            print(f"Skipping {k}: jagged depth >= 3 not supported")

    if not doubles and not singles:
        if not scalars:
            raise RuntimeError("No usable branches found.")
        if verbose:
            print("Only scalar fields found.")
        df = pd.DataFrame({k: ak.to_numpy(v) for k, v in scalars.items()})
        return df

    if not doubles:
        # No double-jagged → keep original single-jagged path (1 row per element)
        ref = next(iter(singles.values()))
        event_idx = ak.local_index(ref, axis=0)
        elem_idx = ak.local_index(ref, axis=1)
        bcast = ak.broadcast_arrays(ref, *scalars.values()) if scalars else [ref]
        b_sc = dict(zip(scalars.keys(), bcast[1:])) if scalars else {}
        zipped = ak.zip(
            {**singles, **b_sc, "event_idx": event_idx, "elem_idx": elem_idx}
        )
        flat = ak.flatten(zipped, axis=1)
        return pd.DataFrame({k: ak.to_numpy(flat[k]) for k in flat.fields})

    # ---------- merged path with double reference ----------
    ref_name, ref = next(iter(doubles.items()))
    if verbose:
        print(f"Using '{ref_name}' as reference double-jagged grid.")

    # Sanity: single-jagged sublist counts must match ref sublist counts per event
    ref_sublists_per_event = ak.num(ref, axis=1)  # shape: [events]
    for k, arr in singles.items():
        counts = ak.num(arr, axis=1)
        if not ak.all(counts == ref_sublists_per_event):
            raise ValueError(
                f"Single-jagged '{k}' has per-event lengths {counts} that do not "
                f"match the number of sublists in reference '{ref_name}' ({ref_sublists_per_event}). "
                "Cannot keep it constant per sublist."
            )

    # Broadcast scalars to the ref grid
    bcast = ak.broadcast_arrays(ref, *scalars.values()) if scalars else [ref]
    b_scalars = dict(zip(scalars.keys(), bcast[1:])) if scalars else {}

    # Broadcast single-jagged to [event][sublist][elem] by matching outer axes and
    # repeating across elem within each sublist
    b_singles = {}
    for k, arr in singles.items():
        try:
            b = ak.broadcast_arrays(ref, arr)[1]
            b_singles[k] = b
        except Exception as e:
            raise ValueError(
                f"Failed to broadcast single-jagged '{k}' to ref grid: {e}"
            )

    # Broadcast other double-jagged fields to the ref grid (drop if incompatible)
    b_doubles = {ref_name: ref}
    for k, arr in doubles.items():
        if k == ref_name:
            continue
        try:
            b = ak.broadcast_arrays(ref, arr)[1]
            b_doubles[k] = b
        except Exception as e:
            print(
                f"Warning: dropping double-jagged '{k}' (not broadcastable to ref): {e}"
            )

    # Indices
    event_idx = ak.local_index(ref, axis=0)  # axis=0
    sublist_idx = ak.local_index(ref, axis=1)  # axis=1
    elem_idx = ak.local_index(ref, axis=2)  # axis=2

    # Zip everything and flatten two levels → rows per (event, sublist, elem)
    record = {
        **b_doubles,
        **b_singles,  # now constant across elements of the sublist
        **b_scalars,
        "event_idx": event_idx,
        "sublist_idx": sublist_idx,
        "elem_idx": elem_idx,
    }
    zipped = ak.zip(record)
    flat1 = ak.flatten(zipped, axis=1)
    flat2 = ak.flatten(flat1, axis=1)

    df = pd.DataFrame({k: ak.to_numpy(flat2[k]) for k in flat2.fields})
    if verbose:
        print(df.head())
        print("Merged DataFrame shape:", df.shape)
        print("Columns:", df.columns.tolist())
    return df


########################################### streamline datasets:


def compute_signed_curvature(pT, q, B):
    """
    Compute the signed curvature of a track in a uniform magnetic field
    using Lorentz force.

    Parameters:
      pT : Transverse momentum (e.g. in GeV)
      q  : Charge of the particle (in units of elementary charge)
      B  : Magnetic field strength (in Tesla)

    Returns:
        signed_kappa: Signed curvature of the track (in m^-1)
    """
    signed_kappa = (q / (pT * 1e9)) * (B * 299_792_458)
    # R = pT * 1e9 / (np.abs(q) * B * 299_792_458)
    return signed_kappa


def compute_circle_parameters(x_v, y_v, phi0, signed_kappa):
    """
    Compute the parameters of
    the circle in the transverse plane.

    Parameters:
      x_v        : x-coordinate of the production vertex
      y_v        : y-coordinate of the production vertex
      phi0       : Initial azimuthal angle of the particle (radians)
      signed_kappa: Signed curvature of the track

    Returns:
        x_c, y_c, R: Center and radius of the circle
    """
    # Compute the radius of the circle
    R = 1.0 / np.abs(signed_kappa)
    # Compute the center of the circle
    x_c = x_v + (1.0 / signed_kappa) * np.sin(phi0)
    y_c = y_v - (1.0 / signed_kappa) * np.cos(phi0)

    return x_c, y_c, R


def compute_perigee(x_c, y_c, R):
    """
    Compute the coordinates of the perigee (point of closest approach)
    in the transverse plane.

    Parameters:
      x_c : x-coordinate of the circle center
      y_c : y-coordinate of the circle center
      R   : Radius of the circle

    Returns:
        x_perigee, y_perigee: Coordinates of the perigee
    """
    # The perigee in the transverse plane is reached when the azimuth of the circle equals that of its center.
    phi_c = np.arctan2(y_c, x_c)
    x_perigee = x_c - R * np.cos(phi_c)
    y_perigee = y_c - R * np.sin(phi_c)

    return x_perigee, y_perigee


def compute_impact_parameters(
    p_x, p_y, p_z, q, B, x_v, y_v, z_v, reference_point=(0, 0, 0)
):
    """
    Compute the transverse impact parameter d0 and the longitudinal impact parameter z0
    for a truth track in a uniform field.

    Parameters:
        p_x, p_y, p_z : Momentum components of the particle (in GeV/c).
        q             : Charge of the particle (in elementary charge units).
        B             : Magnetic field strength (in Tesla).
        x_v, y_v, z_v : Production vertex coordinates (in mm).
        reference_point: Reference point for the perigee calculation (default: origin).

    Returns:
        d0 : Signed transverse impact parameter (in mm).
        z0 : Longitudinal impact parameter (in mm).
        perigee_coords: (x, y, z) coordinates of the perigee (point of closest approach).
    """
    # === Step 0: Express the vertex in the reference frame ===
    x_ref, y_ref, z_ref = reference_point
    x_v = x_v - x_ref
    y_v = y_v - y_ref
    z_v = z_v - z_ref

    # === Step 1: Calculate curvature and radius ===
    pT = np.sqrt(p_x**2 + p_y**2)
    signed_kappa = compute_signed_curvature(pT=pT, q=q, B=B)
    # Convert to mm^-1
    signed_kappa = signed_kappa / 1000

    # === Step 2: Compute the circle center in the transverse plane ===
    phi0 = np.arctan2(p_y, p_x)
    x_c, y_c, R = compute_circle_parameters(
        x_v=x_v, y_v=y_v, phi0=phi0, signed_kappa=signed_kappa
    )

    # === Step 3: Find the perigee (closest approach to the origin) ===
    x_perigee, y_perigee = compute_perigee(x_c=x_c, y_c=y_c, R=R)
    # Check that the perigee is on the circle
    assert np.allclose((x_perigee - x_c) ** 2 + (y_perigee - y_c) ** 2, R**2)

    # === Step 4: Compute the (unsigned) distance from the origin to the perigee ===
    d0_unsigned = np.sqrt(x_perigee**2 + y_perigee**2)

    # === Step 5: Assign a sign to d0 ===
    # Sign is given by the z-component of the cross product of the perigee position and the momentum (angular momentum).
    cross_z = x_perigee * p_y - y_perigee * p_x
    sign = np.sign(cross_z)
    # default to +1 if zero
    # sign = ((sign + 1)/2) - 1
    # sign[sign == 0] = 1

    d0 = sign * d0_unsigned

    # === Step 6: Compute the z-coordinate at the perigee ===
    # Compute the path length from the vertex to the perigee
    # The angle between the vertex and the perigee with respect to the center of the circle
    # is twice the angle between the vertex and the midpoint of the perigee and the center of the circle.
    # Compute the distance between the vertex and the perigee
    d_vertex_perigee = np.sqrt((x_v - x_perigee) ** 2 + (y_v - y_perigee) ** 2)
    # Compute the distance between the vertex and the midpoint
    d_vertex_midpoint = d_vertex_perigee / 2
    # Compute the angle between the vertex and the midpoint
    angle_vertex_midpoint = np.arctan2(d_vertex_midpoint, R)
    # Compute the angle between the vertex and the perigee
    angle_vertex_perigee = 2 * angle_vertex_midpoint

    # Determine the sign of the angle difference
    # = sign of inner product of the vector from the vertex to the perigee and the momentum
    propagation_vector_sign = np.sign((x_perigee - x_v) * p_x + (y_perigee - y_v) * p_y)
    assert (
        propagation_vector_sign.shape == angle_vertex_perigee.shape
    ), f"{propagation_vector_sign.shape} != {angle_vertex_perigee.shape}"
    # Compute the signed angle between the vertex and the perigee
    angle_vertex_perigee = propagation_vector_sign * angle_vertex_perigee

    # Compute the path length between the vertex and the perigee
    s_vertex_perigee = R * angle_vertex_perigee

    # Compute the z-coordinate at the perigee using the linear propagation along z
    z_perigee = z_v + s_vertex_perigee * (p_z / pT)
    z0 = z_perigee

    return d0, z0, (x_perigee, y_perigee, z_perigee)


def apply_z_symmetry(hits):
    """
    Apply z symmetry to the hits.
    The z coordinate is multiplied by the sign of the mean dz of the first 3 hits.
    The dz is the difference between the z coordinate and the z coordinate of the first hit.
    """
    # Compute the dz of the hits
    hits["dz"] = hits["z"] - hits["z"].iloc[0]
    # Get the mean dz of the first 3 hits
    mean_dz = hits["dz"].iloc[:3].mean()
    # Get the sign of the mean dz
    z_sign = np.sign(mean_dz)
    # Multiply the z coordinate by the sign
    hits["z"] = hits["z"] * z_sign
    hits["tz"] = hits["tz"] * z_sign
    hits["dz"] = hits["dz"] * z_sign
    # Multiply track parameters depending on z by the sign
    if "z0" in hits.columns:
        hits["z0"] = hits["z0"] * z_sign
    if "z_perigee" in hits.columns:
        hits["z_perigee"] = hits["z_perigee"] * z_sign
    hits["pz"] = hits["pz"] * z_sign
    hits["peta"] = hits["peta"] * z_sign
    hits["ptheta"] = np.arctan2(hits["pT"], hits["pz"])

    return hits


class TrackMLDataset(IterBase):
    """Iterable class for TrackML"""

    def _load_event(self, event_prefix):
        self.event = event_prefix

        def get_file_path(filename):
            # Check if the compressed version exists; otherwise, fall back to uncompressed
            gz_path = self.path / f"{filename}.csv.gz"
            csv_path = self.path / f"{filename}.csv"
            return gz_path if gz_path.exists() else csv_path

        hits = get_file_path(f"{event_prefix}-hits")
        particles = get_file_path(f"{event_prefix}-particles")
        cells = get_file_path(f"{event_prefix}-cells")
        truth = get_file_path(f"{event_prefix}-truth")
        if getattr(self, "verbose", False):
            print(f"Loading event {event_prefix}")

        # Handle empty csv files
        hits_df = pd.read_csv(hits) if hits.stat().st_size > 0 else pd.DataFrame()
        try:
            cells_df = (
                pd.read_csv(cells) if cells.stat().st_size > 0 else pd.DataFrame()
            )
            # This raises an OSError and EmptyDataError for some non empty files
        except OSError:
            cells_df = pd.DataFrame()
        except pd.errors.EmptyDataError:
            cells_df = pd.DataFrame()
        particles_df = (
            pd.read_csv(particles) if particles.stat().st_size > 0 else pd.DataFrame()
        )
        truth_df = pd.read_csv(truth) if truth.stat().st_size > 0 else pd.DataFrame()
        if not "event_id" in hits_df.columns:
            hits_df["event_id"] = event_prefix
        if not "event_id" in cells_df.columns:
            cells_df["event_id"] = event_prefix
        if not "event_id" in particles_df.columns:
            particles_df["event_id"] = event_prefix
        if not "event_id" in truth_df.columns:
            truth_df["event_id"] = event_prefix

        return (
            hits_df,
            cells_df,
            particles_df,
            truth_df,
        )

    def _preprocessor(self, event_files):
        """Preprocesses data for the specified event.

        Args:
            event_files (tuple): Tuple containing the loaded event data files.
        """
        # Get kwargs input_variables if available
        default_inputs = ["x", "y", "z"]
        input_variables = getattr(self, "input_variables", default_inputs)

        # Get kwargs output_variables if available
        output_variables = getattr(self, "output_variables", ["pT", "pz"])

        groups = self._preprocess_groups(event_files)
        for group in groups:
            inputs = group[input_variables].values
            target = group[output_variables].values[0]

            zxy = torch.tensor(inputs, dtype=torch.float32)
            target_tensor = torch.tensor(target, dtype=torch.float32)

            mask = torch.ones(zxy.shape[0], dtype=torch.bool)
            yield zxy, mask, target_tensor

    def _preprocess_groups(self, event_files):
        # Get kwargs output_variables if available
        self.output_variables = getattr(self, "output_variables", ["pT", "pz"])

        hits, _, particles, truth = event_files
        # Preprocess the particles dataframe
        particles = self._preprocess_particles(particles)

        # Merge the dataframes
        merged_df = pd.merge(truth, particles, on="particle_id")
        merged_df = pd.merge(merged_df, hits, on="hit_id")

        # Get kwargs truth_position if available
        truth_position = getattr(self, "truth_position", True)
        if truth_position:
            # Override reconstructed position by truth position
            merged_df["x"] = merged_df["tx"]
            merged_df["y"] = merged_df["ty"]
            merged_df["z"] = merged_df["tz"]

        # Get kwargs input_variables if available
        default_inputs = ["x", "y", "z"]
        input_variables = getattr(self, "input_variables", default_inputs)

        if (
            any([var not in merged_df.columns for var in input_variables])
            or getattr(self, "sort_by_radius", False)
            or getattr(self, "cut_scattered", False)
        ):
            # Add other coordinate system
            merged_df["tr"] = np.sqrt(merged_df["tx"] ** 2 + merged_df["ty"] ** 2)
            merged_df["tphi"] = np.arctan2(merged_df["ty"], merged_df["tx"])
            merged_df["r"] = np.sqrt(merged_df["x"] ** 2 + merged_df["y"] ** 2)
            merged_df["phi"] = np.arctan2(merged_df["y"], merged_df["x"])

        grouped = merged_df.groupby("particle_id")

        for _, group in grouped:
            # Cut scattered tracks
            if getattr(self, "cut_scattered", False):
                # This is a truth particle cut

                # The track must be ordered by radius
                if not "tr" in group:
                    group["tr"] = np.sqrt(group["tx"] ** 2 + group["ty"] ** 2)
                group_sorted = group.sort_values("tr")
                # Compute the angle between the hits
                if not "tphi" in group:
                    group_sorted["tphi"] = np.arctan2(
                        group_sorted["ty"], group_sorted["tx"]
                    )
                group_sorted["dphi"] = (
                    group_sorted["tphi"] - group_sorted["tphi"].iloc[0]
                )
                # Correct for periodicity
                group_sorted["dphi"] = np.where(
                    group_sorted["dphi"] > np.pi,
                    group_sorted["dphi"] - 2 * np.pi,
                    group_sorted["dphi"],
                )
                group_sorted["dphi"] = np.where(
                    group_sorted["dphi"] < -np.pi,
                    group_sorted["dphi"] + 2 * np.pi,
                    group_sorted["dphi"],
                )
                # Check if the angle is monotonically increasing
                scattered = (
                    (group_sorted["dphi"].shift(-1) - group_sorted["dphi"])
                    * (group_sorted["dphi"].shift(-2) - group_sorted["dphi"].shift(-1))
                    < 0
                ).any()
                if scattered:
                    continue

            # Sort by the hits by radius
            if getattr(self, "sort_by_radius", False):
                if not "r" in group:
                    group["r"] = np.sqrt(group["x"] ** 2 + group["y"] ** 2)
                group = group.sort_values("r")

            # Add custom features
            if "dphi" in input_variables:
                # Remove phi of the first hit
                group["dphi"] = group["phi"] - group["phi"].iloc[0]
                # Correct for periodicity
                group["dphi"] = np.where(
                    group["dphi"] > np.pi, group["dphi"] - 2 * np.pi, group["dphi"]
                )
                group["dphi"] = np.where(
                    group["dphi"] < -np.pi, group["dphi"] + 2 * np.pi, group["dphi"]
                )

            if (
                "pT_circle_estimate" in input_variables
                or "pT_circle_estimate_inv" in input_variables
            ):
                # Estimate pT from the circle fit
                from src.my_model.benchmarks import CircleFit

                cf = CircleFit()
                points = group[["x", "y"]].values
                points = torch.tensor(points, dtype=torch.float32)

                # Make it a batch of 1 2D list of points
                points = points.unsqueeze(0)
                r = cf.fit(points).tolist()
                pt_fit = np.array(r) * 1.0 * 2 * 299_792_458 / 1e9 / 1000
                group["pT_circle_estimate"] = np.full(group.shape[0], pt_fit)
                group["pT_circle_estimate_inv"] = 1 / np.full(group.shape[0], pt_fit)

            # Get kwargs z_symmetry if available
            z_symmetry = getattr(self, "z_symmetry", False)
            if z_symmetry:
                group = apply_z_symmetry(group)

            yield group

    def _preprocess_particles(self, particles):
        # Preprocess the particles dataframe
        particles["pT"] = np.sqrt(particles["px"] ** 2 + particles["py"] ** 2)

        p = np.sqrt(particles["px"] ** 2 + particles["py"] ** 2 + particles["pz"] ** 2)
        particles["peta"] = np.arctanh(particles["pz"] / p)

        if any([var not in particles.columns for var in self.output_variables]):
            # Add other track parameters
            particles["qopT"] = particles["q"] / particles["pT"]
            particles["qpT"] = particles["q"] * particles["pT"]
            particles["phi0"] = np.arctan2(particles["py"], particles["px"])
            if any(
                [
                    var in self.output_variables
                    for var in ["d0", "z0", "x_perigee", "y_perigee", "z_perigee"]
                ]
            ):
                (
                    particles["d0"],
                    particles["z0"],
                    (
                        particles["x_perigee"],
                        particles["y_perigee"],
                        particles["z_perigee"],
                    ),
                ) = compute_impact_parameters(
                    p_x=particles["px"],
                    p_y=particles["py"],
                    p_z=particles["pz"],
                    q=particles["q"],
                    B=2,
                    x_v=particles["vx"],
                    y_v=particles["vy"],
                    z_v=particles["vz"],
                    reference_point=(0, 0, 0),
                )
            particles["ptheta"] = np.arctan2(particles["pT"], particles["pz"])

        # Get kwargs min_hits if available
        min_hits = getattr(self, "min_hits", 5)
        particles = particles[particles["nhits"] >= min_hits]

        # Get kwargs min_pt and max_pt if available
        min_pt = getattr(self, "min_pt", 0)
        max_pt = getattr(self, "max_pt", np.inf)

        particles = particles[(particles["pT"] >= min_pt) & (particles["pT"] <= max_pt)]

        # Get kwargs keep_secondaries if available
        keep_secondaries = getattr(self, "keep_secondaries", True)
        if not keep_secondaries:
            secondary_selection = (np.abs(particles["vx"]) >= 1) | (
                np.abs(particles["vy"]) >= 1
            )
            particles = particles[~secondary_selection]

        # Get kwargs min_abs_eta and max_abs_eta if available
        min_abs_eta = getattr(self, "min_abs_eta", 0)
        max_abs_eta = getattr(self, "max_abs_eta", np.inf)

        particles = particles[
            (np.abs(particles["peta"]) >= min_abs_eta)
            & (np.abs(particles["peta"]) <= max_abs_eta)
        ]

        return particles


def combine_segments(segments: list[int]) -> int:
    """
    Pack segments into a 64-bit integer using big-endian bit-fields,
    then mask to 64 bits.
    """
    # Bit widths from MultiIndex<std::uint64_t, 12, 12, 16, 8, 16>
    BIT_WIDTHS = [12, 12, 16, 8, 16]
    full_mask = (1 << 64) - 1
    val = 0
    for seg, width in zip(segments, BIT_WIDTHS):
        if seg < 0 or seg >= (1 << width):
            raise ValueError(f"Segment {seg} out of {width}-bit range")
        val = (val << width) | seg
    return val & full_mask


def parse_particle_id(s: str) -> int:
    """
    Convert a pipe-separated string 'a|b|c|d|e' into a uint64.
    """
    parts = [int(p) for p in s.strip().split("|") if p]
    return combine_segments(parts)


class ActsDatasetProcessing:
    def _preprocessor(self, event_files):
        """Preprocesses data for the specified event.

        Args:
            event_files (tuple): Tuple containing the loaded event data files.
        """
        # Get kwargs input_variables, ... if available
        default_inputs = ["x", "y", "z"]
        input_variables = getattr(self, "input_variables", default_inputs)
        output_variables = getattr(self, "output_variables", ["pT", "pz"])
        extra_variables = getattr(self, "extra_variables", [])

        groups = self._preprocess_groups(event_files)
        for group in groups:
            inputs = group[input_variables].values
            target = group[output_variables].values[0]
            extra = group[extra_variables].values[0] if extra_variables else []

            zxy = torch.tensor(inputs, dtype=torch.float32)
            target_tensor = torch.tensor(target, dtype=torch.float32)
            extra_tensor = (
                torch.tensor(extra, dtype=torch.float32) #if extra_variables else []
            )

            mask = torch.ones(zxy.shape[0], dtype=torch.bool)
            yield zxy, mask, target_tensor, extra_tensor

    def _preprocess_groups(self, event_files):
        """Preprocesses data for the specified event.

        Args:
            event_files (tuple): Tuple containing the loaded event data files.
        """
        verbose = getattr(self, "verbose", False)

        hits, tracks, particles = event_files

        # Preprocess the particles dataframe
        particles = self._preprocess_particles(particles)

        # Add Hit_ID to the hits dataframe (index)
        # Group by event_id then assign a hit_id unique per event (restarting from 0 for each event)
        hits["hit_id"] = hits.groupby("event_id").cumcount()

        truth_tracks = getattr(self, "truth_tracks", True)
        track_index = "particle_id"
        if not truth_tracks:
            if isinstance(self, ActsDataset):
                # Convert particle_id to 64-bit unsigned int
                tracks["particle_id"] = tracks["particleId"].apply(parse_particle_id)
                del tracks["particleId"]
                # Verify that the particle_id is in the particles dataframe
                if not tracks["particle_id"].isin(particles["particle_id"]).all():
                    raise ValueError(
                        f"Particle id {len(tracks['particle_id'][~tracks['particle_id'].isin(particles['particle_id'])])} not in particles dataframe"
                    )

                track_index = "track_id"
                # Extract the hits from the tracks
                # Convert "[5747,7769,13699,]" to [5747, 7769, 13699]
                # Remove the brackets and split by comma
                tracks["Hits_ID"] = tracks["Hits_ID"].str.strip("[]").str.split(",")
                # Convert to int
                tracks["Hits_ID"] = tracks["Hits_ID"].apply(
                    lambda x: [int(i) for i in x if i]
                )
                # Explode the dataframe
                tracks = tracks.explode("Hits_ID")
                # Keep only Hits_ID, particle_id and track_id
                tracks = tracks[["Hits_ID", "particle_id", "track_id", "event_id"]]
                # Rename Hits_ID to hit_id
                tracks.rename(columns={"Hits_ID": "hit_id"}, inplace=True)

                # Merge with hits dataframe
                hits = pd.merge(
                    hits,
                    tracks,
                    on=["hit_id", "event_id"],
                    validate="one_to_many",
                )
            elif isinstance(self, ActsRootDataset):
                pass
            else:
                raise NotImplementedError(f"Dataset {type(self)} not supported")

        # Get kwargs truth_position if available
        truth_position = getattr(self, "truth_position", True)
        if truth_position:
            # Override reconstructed position by truth position
            hits["x"] = hits["tx"]
            hits["y"] = hits["ty"]
            hits["z"] = hits["tz"]

        # Get kwargs input_variables if available
        default_inputs = ["x", "y", "z"]
        input_variables = getattr(self, "input_variables", default_inputs)

        if (
            any([var not in hits.columns for var in input_variables])
            or getattr(self, "sort_by_radius", False)
            or getattr(self, "cut_scattered", False)
        ):
            # Add other coordinate system
            hits["tr"] = np.sqrt(hits["tx"] ** 2 + hits["ty"] ** 2)
            hits["tphi"] = np.arctan2(hits["ty"], hits["tx"])
            hits["r"] = np.sqrt(hits["x"] ** 2 + hits["y"] ** 2)
            hits["phi"] = np.arctan2(hits["y"], hits["x"])

        if verbose:
            # print the number of particles that does not have corresponding hits
            n_particles_without_hits = len(
                particles[~particles["particle_id"].isin(hits["particle_id"])]
            )
            if n_particles_without_hits > 0:
                console.log(
                    f"[red]Warning: {n_particles_without_hits} particles do not have corresponding hits."
                )
                # Print the particle ids
                console.log(
                    f"[red]Particle ids: {particles[~particles['particle_id'].isin(hits['particle_id'])]['particle_id'].unique()}"
                )

        ref_cols = ["particle_id", "event_id"]
        if "track_id" in hits.columns and "track_id" in particles.columns:
            ref_cols.append("track_id")

        merged_df = pd.merge(
            hits, particles, on=ref_cols, validate="many_to_one"
        )

        # Add a variable counting the number of hits per track index
        merged_df["n_hits_group"] = merged_df.groupby(["event_id", track_index])[
            "hit_id"
        ].transform("count")

        # Add custom features
        if "dphi" in input_variables or "dphi0" in self.output_variables:
            # Sort by ["event_id", track_index, "r"]
            merged_df.sort_values(["event_id", track_index, "r"], inplace=True)

            # Get kwargs phi_offset if available
            phi_offset_type = getattr(self, "phi_offset", "first")
            if phi_offset_type == "first":
                # Get the first hit for each track
                first_hits = (
                    merged_df.groupby(["event_id", track_index]).first().reset_index()
                )

                # Add a column "dphi" by subtracting the phi of the first hit of the track to the phi of the hits of the same track
                merged_df = pd.merge(
                    merged_df,
                    first_hits[["event_id", track_index, "phi"]].rename(
                        columns={"phi": "phi_offset"}
                    ),
                    on=["event_id", track_index],
                    how="left",
                )
            elif isinstance(phi_offset_type, (int, float)):
                merged_df["phi_offset"] = float(phi_offset_type)
            elif phi_offset_type == "min":
                # Get the minimum phi for each track
                min_hits = (
                    merged_df.groupby(["event_id", track_index])["phi"]
                    .min()
                    .reset_index()
                )
                merged_df = pd.merge(
                    merged_df,
                    min_hits.rename(columns={"phi": "phi_offset"}),
                    on=["event_id", track_index],
                    how="left",
                )
            else:
                raise ValueError(
                    f"phi_offset must be 'first', 'min' or a float, got {phi_offset_type}"
                )

        if "dphi" in input_variables:
            merged_df["dphi"] = merged_df["phi"] - merged_df["phi_offset"]
            # Correct for periodicity
            merged_df["dphi"] = np.where(
                merged_df["dphi"] > np.pi,
                merged_df["dphi"] - 2 * np.pi,
                merged_df["dphi"],
            )
            merged_df["dphi"] = np.where(
                merged_df["dphi"] < -np.pi,
                merged_df["dphi"] + 2 * np.pi,
                merged_df["dphi"],
            )
        if "dphi0" in self.output_variables:
            merged_df["dphi0"] = merged_df["phi0"] - merged_df["phi_offset"]
            # Correct for periodicity
            merged_df["dphi0"] = np.where(
                merged_df["dphi0"] > np.pi,
                merged_df["dphi0"] - 2 * np.pi,
                merged_df["dphi0"],
            )
            merged_df["dphi0"] = np.where(
                merged_df["dphi0"] < -np.pi,
                merged_df["dphi0"] + 2 * np.pi,
                merged_df["dphi0"],
            )

        grouped = merged_df.groupby(["event_id", track_index])
        if verbose:
            print(f"Processing event {self.event}")
            print(f"Number of tracks: {len(grouped)}")


        # Get kwargs min_hits if available
        min_hits = getattr(self, "min_hits", 5)
        for group_id, group in grouped:
            if verbose:
                print(f"Processing track {group_id} with {group.shape[0]} hits")
            assert (
                group["n_hits_group"].iloc[0] == group.shape[0]
            ), f"Number of hits {group['n_hits_group'].iloc[0]} does not match the number of hits in the group {group.shape[0]}"
            # Cut tracks with too few hits
            if group.shape[0] < min_hits:
                if verbose:
                    print(
                        f"Skipping track with {group.shape[0]} hits (min_hits={min_hits})"
                    )
                continue

            # Cut scattered tracks
            if getattr(self, "cut_scattered", False):
                # This is a truth particle cut

                # The track must be ordered by radius
                if not "tr" in group:
                    group["tr"] = np.sqrt(group["tx"] ** 2 + group["ty"] ** 2)
                group_sorted = group.sort_values("tr")
                # Compute the angle between the hits
                if not "tphi" in group:
                    group_sorted["tphi"] = np.arctan2(
                        group_sorted["ty"], group_sorted["tx"]
                    )
                group_sorted["dphi"] = (
                    group_sorted["tphi"] - group_sorted["tphi"].iloc[0]
                )
                # Correct for periodicity
                group_sorted["dphi"] = np.where(
                    group_sorted["dphi"] > np.pi,
                    group_sorted["dphi"] - 2 * np.pi,
                    group_sorted["dphi"],
                )
                group_sorted["dphi"] = np.where(
                    group_sorted["dphi"] < -np.pi,
                    group_sorted["dphi"] + 2 * np.pi,
                    group_sorted["dphi"],
                )
                # Check if the angle is monotonically increasing
                scattered = (
                    (group_sorted["dphi"].shift(-1) - group_sorted["dphi"])
                    * (group_sorted["dphi"].shift(-2) - group_sorted["dphi"].shift(-1))
                    < 0
                ).any()
                if scattered:
                    if verbose:
                        # Print the track id and number of hits
                        print(
                            f"Skipping scattered track {group['particle_id'].iloc[0]} with {group.shape[0]} hits"
                        )
                    continue

            # Sort by the hits by radius
            if getattr(self, "sort_by_radius", False):
                if not "r" in group:
                    group["r"] = np.sqrt(group["x"] ** 2 + group["y"] ** 2)
                group = group.sort_values("r")
            elif getattr(self, "sort_by_dz", False):
                # Compute the mean z of the hits
                mean_z = group["z"].mean()
                group["dz"] = (group["z"] - mean_z) * mean_z
                # Sort by absolute distance to the mean z
                # group["dz"] = np.abs(group["dz"])
                group = group.sort_values("dz")
            elif getattr(self, "sort_by_distance", False):
                group["distance"] = np.sqrt(
                    group["x"] ** 2 + group["y"] ** 2 + group["z"] ** 2
                )
                group = group.sort_values("distance")

            if (
                "pT_circle_estimate" in input_variables
                or "pT_circle_estimate_inv" in input_variables
            ):
                # Estimate pT from the circle fit
                from src.my_model.benchmarks import CircleFit

                cf = CircleFit()
                points = group[["x", "y"]].values
                points = torch.tensor(points, dtype=torch.float32)

                # Make it a batch of 1 2D list of points
                points = points.unsqueeze(0)
                r = cf.fit(points).tolist()
                pt_fit = np.array(r) * 1.0 * 2 * 299_792_458 / 1e9 / 1000
                group["pT_circle_estimate"] = np.full(group.shape[0], pt_fit)
                group["pT_circle_estimate_inv"] = 1 / np.full(group.shape[0], pt_fit)

            # Get kwargs z_symmetry if available
            z_symmetry = getattr(self, "z_symmetry", False)
            if z_symmetry:
                group = apply_z_symmetry(group)

            yield group

    def _preprocess_particles(self, particles):
        # Preprocess the particles dataframe
        particles = particles.copy()

        # Get kwargs particle_types if available
        particle_types = getattr(self, "particle_types", None)
        if particle_types is not None:
            # Filter particles based on the specified particle types
            particles = particles[particles["particle_type"].isin(particle_types)]

        # Get kwargs keep_secondaries if available
        keep_secondaries = getattr(self, "keep_secondaries", True)
        if not keep_secondaries:
            secondary_selection = (np.abs(particles["vx"]) >= 1) | (
                np.abs(particles["vy"]) >= 1
            )
            particles = particles[~secondary_selection]

        if not "pT" in particles.columns:
            # Calculate transverse momentum pT
            particles["pT"] = np.sqrt(particles["px"] ** 2 + particles["py"] ** 2)

        # Get kwargs min_pt and max_pt if available
        min_pt = getattr(self, "min_pt", 0)
        max_pt = getattr(self, "max_pt", np.inf)

        particles = particles[(particles["pT"] >= min_pt) & (particles["pT"] <= max_pt)]

        if not "peta" in particles.columns:
            p = np.sqrt(
                particles["px"] ** 2 + particles["py"] ** 2 + particles["pz"] ** 2
            )
            particles["peta"] = np.arctanh(particles["pz"] / p)

        # Get kwargs min_abs_eta and max_abs_eta if available
        min_abs_eta = getattr(self, "min_abs_eta", 0)
        max_abs_eta = getattr(self, "max_abs_eta", np.inf)

        particles = particles[
            (np.abs(particles["peta"]) >= min_abs_eta)
            & (np.abs(particles["peta"]) <= max_abs_eta)
        ]

        # Get kwargs output_variables if available
        output_variables = getattr(self, "output_variables", ["pT", "pz"])

        computed_impact_parameters = getattr(self, "computed_impact_parameters", False)
        if computed_impact_parameters and not any(
            impact_parameter in output_variables
            for impact_parameter in ["d0", "z0", "x_perigee", "y_perigee", "z_perigee"]
        ):
            raise ValueError(
                "computed_impact_parameters is True, but no impact parameters are requested in output_variables."
            )
        if (
            any([var not in particles.columns for var in output_variables])
            or computed_impact_parameters
        ):
            # Add other track parameters
            if not "qopT" in particles.columns:
                particles["qopT"] = particles["q"] / particles["pT"]
            if not "qpT" in particles.columns:
                particles["qpT"] = particles["q"] * particles["pT"]
            if not "phi0" in particles.columns:
                particles["phi0"] = np.arctan2(particles["py"], particles["px"])
            if (
                any(
                    [
                        var in output_variables
                        #for var in ["d0", "z0", "x_perigee", "y_perigee", "z_perigee"]
                        # small hack to avoid computing impact parameters while we have them now in input data
                        for var in ["x_perigee", "y_perigee", "z_perigee"]
                    ]
                )
                or computed_impact_parameters
            ):
                (
                    computed_d0,
                    computed_z0,
                    (
                        particles["x_perigee"],
                        particles["y_perigee"],
                        particles["z_perigee"],
                    ),
                ) = compute_impact_parameters(
                    p_x=particles["px"],
                    p_y=particles["py"],
                    p_z=particles["pz"],
                    q=particles["q"],
                    B=2,
                    x_v=particles["vx"],
                    y_v=particles["vy"],
                    z_v=particles["vz"],
                    reference_point=(0, 0, 0),
                )
                if not "d0" in particles.columns or computed_impact_parameters:
                    particles["d0"] = computed_d0
                if not "z0" in particles.columns or computed_impact_parameters:
                    particles["z0"] = computed_z0
            if not "ptheta" in particles.columns:
                particles["ptheta"] = np.arctan2(particles["pT"], particles["pz"])

        return particles

    def _preprocess_inputs(self, input_hits):
        zxy = input_hits.values
        zxy = torch.tensor(zxy, dtype=torch.float32)
        # Create a mask for the hits
        mask = torch.ones(zxy.shape[0], dtype=torch.bool)
        return zxy, mask


class ActsDataset(ActsDatasetProcessing, IterBase):

    def _load_event(self, event_prefix):
        self.event = event_prefix
        particle_file = getattr(self, "particle_file", "particles_simulated")
        hits_file = getattr(self, "hits_file", "hits")
        track_file = getattr(self, "track_file", "tracks_ambi")

        particles = self.path / f"{event_prefix}-{particle_file}.csv"
        hits = self.path / f"{event_prefix}-{hits_file}.csv"
        tracks = self.path / f"{event_prefix}-{track_file}.csv"
        if getattr(self, "verbose", False):
            print(f"Loading event {event_prefix}")
        hits_df = pd.read_csv(hits)
        tracks_df = pd.read_csv(tracks)
        particles_df = pd.read_csv(particles)
        if not "event_id" in hits_df.columns:
            hits_df["event_id"] = event_prefix
        if not "event_id" in particles_df.columns:
            particles_df["event_id"] = event_prefix
        if not "event_id" in tracks_df.columns:
            tracks_df["event_id"] = event_prefix
        return (
            hits_df,
            tracks_df,
            particles_df,
        )


def extract_barcode(
    df: pd.DataFrame,
    idx_cols: list[str],
    barcode_col: str,
    barcode_index_col: str,
    check_constant: bool = False,
) -> pd.DataFrame:
    """Extracts the barcode from the particle_id column and adds it as separate columns.

    Args:
        df (pd.DataFrame): Dataframe containing a particle_id column.
        idx_cols (list[str]): List of columns to use as index for pivoting.
        barcode_col (str): Name of the column containing the barcode (particle_id).
        barcode_index_col (str): Name of the column containing the barcode index (elem_idx).
        check_constant (bool): Whether to check that all other columns are constant within each event.
    """
    # expected mapping from elem_idx -> output column name
    _COLS = {
        0: "vertex_primary",
        1: "vertex_secondary",
        2: "particle",
        3: "generation",
        4: "sub_particle",
    }

    # Pivot rows -> columns
    wide = (
        df.pivot_table(
            index=idx_cols,
            columns=barcode_index_col,
            values=barcode_col,
            aggfunc="first",
        )
        .rename(columns=_COLS)
        .reset_index()
    )

    # Keep/verify all other columns are constant within each event
    value_cols = [
        c
        for c in df.columns
        if c not in (set(idx_cols) | {barcode_col, barcode_index_col})
    ]

    if value_cols:
        if check_constant:
            # Check that all value_cols are constant within each group
            nunique = df.groupby(idx_cols, dropna=False)[value_cols].nunique(
                dropna=False
            )
            varying = {
                c: nunique.index[nunique[c] > 1].tolist()
                for c in value_cols
                if (nunique[c] > 1).any()
            }
            if varying:
                raise ValueError(
                    "Some columns vary within an event and cannot be kept unambiguously: "
                    + ", ".join(f"{c} (groups: {len(varying[c])})" for c in varying)
                )

        # Take the first row per group (since they are constant within the group)
        meta = (
            df.sort_values(
                idx_cols
                + ([barcode_index_col] if barcode_index_col in df.columns else [])
            )
            .groupby(idx_cols, as_index=False)[value_cols]
            .first()
        )

        # Merge metadata back onto the pivoted table
        wide = wide.merge(meta, on=idx_cols, how="left")

    # Final column order (event keys + particle id columns + the rest)
    particle_cols = list(_COLS.values())
    other_cols = [
        c for c in wide.columns if c not in (set(idx_cols) | set(particle_cols))
    ]
    wide = wide[idx_cols + particle_cols + other_cols]
    assert (
        wide[particle_cols].notna().all().all()
    ), "NaN values found in particle id columns"
    # Make sure that the columns in wide are the same as in df (except for barcode_col, barcode_index_col)
    assert set(wide.columns) == (
        set(df.columns) - {barcode_col, barcode_index_col} | set(_COLS.values())
    ), (
        "Columns in wide do not match columns in df"
        + f" ({wide.columns} vs {df.columns})"
    )
    return wide


class ActsRootDataset(ActsDatasetProcessing, RootIterBase):

    def _load_event(self, event_prefix, n_events_split=100):

        verbose = getattr(self, "verbose", False)
        if verbose:
            print(
                f"Loading event {event_prefix} and split {event_prefix // n_events_split * n_events_split}"
            )

        # Use truth hit position if available, otherwise use reconstructed hit position ("measurements")
        truth_hit_position = getattr(self, "truth_hit_position", False)

        # Use true tracks, otherwise use the reco tracks (trackstates_ambi.root) 
        truth_tracks = getattr(self, "truth_tracks", True)

        if verbose:
            if truth_hit_position:
                print("Using truth hit position (tx, ty, tz)")
            else:
                print("Using reconstructed hit position (rec_gx, rec_gy, rec_gz)")

            if truth_tracks:
                print("Using truth tracks as target (as if the track finding was perfect)")
            else:            
                print("Using reconstructed tracks (from trackstates_ambi file)")

        self.event = event_prefix
        # Alexis: particles_hits_helix is most probably a custom name from jeremy, not prapagated to ACTS main
        # in recetn ACTS version the pregigee true parameters are saved in "particles_simulation" file 
        # (if the option writeHelixParematers is enabled)
        particle_file = getattr(self, "particle_file", "particles_hits_helix")
        hits_file = getattr(self, "hits_file", "hits")
        measurements_file = getattr(self, "measurements_file", "measurements")
        track_hits_file = getattr(self, "track_hits_file", "trackstates_ambi")
        track_params_file = getattr(self, "track_params_file", "tracksummary_ambi")
        
        # Find particle truth file
        particles = self.path.glob(
            f"*_split_start_{event_prefix // n_events_split * n_events_split}_n_{n_events_split}/{particle_file}.root"
        )

        # Find the "hits" file : hits.root if truth_hit_position is True, measurements.root otherwise
        if truth_hit_position:
            hits = self.path.glob(
                f"*_split_start_{event_prefix // n_events_split * n_events_split}_n_{n_events_split}/{hits_file}.root"
            )
            measurements = []  # We will get the hit true position from the hits.root file
        else:
            hits = []  # We will get the hit position from the measurements file
            measurements = self.path.glob(
                f"*_split_start_{event_prefix // n_events_split * n_events_split}_n_{n_events_split}/{measurements_file}.root"
            )

        if not truth_tracks:
            # Use the reco tracks from trackstates_ambi.root and tracksummary_ambi.root files
            track_hits = self.path.glob(
                f"*_split_start_{event_prefix // n_events_split * n_events_split}_n_{n_events_split}/{track_hits_file}.root"
            )
            track_params = self.path.glob(
                f"*_split_start_{event_prefix // n_events_split * n_events_split}_n_{n_events_split}/{track_params_file}.root"
            )
        else:
            # Use the true tracks, do not need reco tracks info
            track_hits = []  
            track_params = []

        # Ensure we have exactly one file for particles and hits
        particles = list(particles)
        hits = list(hits)
        measurements = list(measurements)
        track_hits = list(track_hits)
        track_params = list(track_params)

        assert (
            len(particles) == 1
        ), f"Expected exactly one particles file, got {len(particles)}"

        particles = particles[0]

        if truth_hit_position:
            assert len(hits) == 1, f"Expected exactly one hits file, got {len(hits)}"
            hits = hits[0]
        else:
            assert (
                len(measurements) == 1
            ), f"Expected exactly one measurements file, got {len(measurements)}"
            measurements = measurements[0]

        if not truth_tracks:
            assert (
                len(track_hits) == 1
            ), f"Expected exactly one track hits file, got {len(track_hits)}"
            assert (
                len(track_params) == 1
            ), f"Expected exactly one track params file, got {len(track_params)}"
            track_hits = track_hits[0]
            track_params = track_params[0]
        
        import uproot

        # Load particles
        with uproot.open(particles) as f:
            particles = convert_tree_to_dataframe(f, keys=list(f.keys())[0])

        if any("perigee_" in col for col in particles.columns):
            console.print(
                "[yellow]Warning: 'perigee_' columns found in particles dataframe.",
                style="yellow",
            )
            # Force replacement of "perigee_x" variables to "x" in particles
            # First remove the columns if they exist
            perigee_cols = [
                col.replace("perigee_", "")
                for col in particles.columns
                if col.startswith("perigee_")
                and col.replace("perigee_", "") in particles.columns
            ]
            particles.drop(columns=perigee_cols, inplace=True)
            particles.rename(
                columns={k: k.replace("perigee_", "") for k in particles.columns},
                inplace=True,
            )

        # Load hits
        if truth_tracks:
            tracks = pd.DataFrame()

            if truth_hit_position:
                with uproot.open(hits) as f:
                    hits = convert_tree_to_dataframe(f, keys=list(f.keys())[0])

                hits = extract_barcode(
                    hits,
                    idx_cols=["event_id"] + ["event_idx"],
                    barcode_col="barcode",
                    barcode_index_col="elem_idx",
                )
            else:
                with uproot.open(measurements) as f:
                    branches_to_load = [
                        "event_nr",
                        "particles_vertex_primary",
                        "particles_vertex_secondary",
                        "particles_particle",
                        "particles_generation",
                        "particles_sub_particle",
                        "rec_gx",
                        "rec_gy",
                        "rec_gz",
                        "true_x",
                        "true_y",
                        "true_z",
                        "rec_loc0",
                        "rec_loc1",
                        "rec_time",
                        "var_loc0",
                        "var_loc1",
                        "var_time",
                    ]
                    hits = convert_tree_to_dataframe(
                        f, keys=list(f.keys())[0], branches_to_load=branches_to_load
                    )

                # Rename some columns to match the truth hits
                hits.rename(
                    columns={
                        "rec_gx": "x",
                        "rec_gy": "y",
                        "rec_gz": "z",
                        "true_x": "tx",
                        "true_y": "ty",
                        "true_z": "tz",
                        "particles": "barcodes",
                        "event_nr": "event_id",
                    },
                    inplace=True,
                )


        else:

            # Opening the reconstructed track files (tracksummary_ambi.root)
            # with information on fitted track paremeters and truth-matched true particle
            with uproot.open(track_params) as f:
                ########################################

                branches_to_load = [
                    "event_nr",
                    "track_nr",
                    "majorityParticleId_vertex_primary",
                    "majorityParticleId_vertex_secondary",
                    "majorityParticleId_particle",
                    "majorityParticleId_generation",
                    "majorityParticleId_sub_particle",
                    "t_charge",
                    "t_theta",
                    "t_eta",
                    "t_phi",
                    "t_p",
                    "t_d0",
                    "t_z0",
                ]
                branches_to_load += [
                    "t_vx",
                    "t_vy",
                    "t_vz",
                    "t_px",
                    "t_py",
                    "t_pz",
                    "t_pT",
                    # "t_time",
                    "trackClassification",
                    "hasFittedParams",
                    "nMajorityHits",
                ]

                track_particles = convert_tree_to_dataframe(
                    f, keys=list(f.keys())[0], branches_to_load=branches_to_load
                )

                branches_to_load = [
                    "event_nr",
                    "track_nr",
                    "majorityParticleId_vertex_primary",
                    "majorityParticleId_vertex_secondary",
                    "majorityParticleId_particle",
                    "majorityParticleId_generation",
                    "majorityParticleId_sub_particle",
                    "nMajorityHits",
                    "eLOC0_fit",
                    "eLOC1_fit",
                    "ePHI_fit",
                    "eTHETA_fit",
                    "eQOP_fit",
                    "eT_fit",
                    "hasFittedParams",
                    "nStates",
                    "nMeasurements",
                    "nOutliers",
                    "nHoles",
                    "nSharedHits",
                    "chi2Sum",
                    "NDF",
                ]

                track_params = convert_tree_to_dataframe(
                    f, keys=list(f.keys())[0], branches_to_load=branches_to_load
                )


            with uproot.open(track_hits) as f:
                branches_to_load = [
                    "event_nr",
                    "track_nr",
                    "stateType",
                    "t_x",
                    "t_y",
                    "t_z",
                    "t_dx",
                    "t_dy",
                    "t_dz",
                    "g_x_hit",
                    "g_y_hit",
                    "g_z_hit",
                    # "particle_ids",  # missing in v44.0.0
                ]

                ########################################
                track_hits = convert_tree_to_dataframe(
                    f,
                    keys=list(f.keys())[0],
                    branches_to_load=branches_to_load,
                )

            track_hits.rename(
                columns={
                    "event_nr": "event_id",
                    "track_nr": "track_id",
                    "t_x": "tx",
                    "t_y": "ty",
                    "t_z": "tz",
                    "g_x_hit": "x",
                    "g_y_hit": "y",
                    "g_z_hit": "z",
                },
                inplace=True,
            )
            track_params.rename(
                columns={
                    "event_nr": "event_id",
                    "track_nr": "track_id",
                    "majorityParticleId_vertex_primary": "vertex_primary",
                    "majorityParticleId_vertex_secondary": "vertex_secondary",
                    "majorityParticleId_particle": "particle",
                    "majorityParticleId_generation": "generation",
                    "majorityParticleId_sub_particle": "sub_particle",
                },
                inplace=True,
            )
            track_particles.rename(
                columns={
                    "event_nr": "event_id",
                    "track_nr": "track_id",
                    "majorityParticleId_vertex_primary": "vertex_primary",
                    "majorityParticleId_vertex_secondary": "vertex_secondary",
                    "majorityParticleId_particle": "particle",
                    "majorityParticleId_generation": "generation",
                    "majorityParticleId_sub_particle": "sub_particle",
                },
                inplace=True,
            )

            hits = track_hits
            if all(
                col in track_particles.columns
                for col in [
                    "vertex_primary",
                    "vertex_secondary",
                    "particle",
                    "generation",
                    "sub_particle",
                ]
            ):
                # Combine vertex_primary, vertex_secondary, particle, generation and sub_particle into a single string
                track_particles["particle_id"] = track_particles[
                    [
                        "vertex_primary",
                        "vertex_secondary",
                        "particle",
                        "generation",
                        "sub_particle",
                    ]
                ].apply(lambda row: "_".join(row.values.astype(str)), axis=1)
                track_params["particle_id"] = track_params[
                    [
                        "vertex_primary",
                        "vertex_secondary",
                        "particle",
                        "generation",
                        "sub_particle",
                    ]
                ].apply(lambda row: "_".join(row.values.astype(str)), axis=1)
                particles["particle_id"] = particles[
                    [
                        "vertex_primary",
                        "vertex_secondary",
                        "particle",
                        "generation",
                        "sub_particle",
                    ]
                ].apply(lambda row: "_".join(row.values.astype(str)), axis=1)

                if not "particle_id" in hits:
                    # Use the particle_id of track_particles with matching event_id and track_id
                    hits["particle_id"] = hits.merge(
                        track_particles[["event_id", "track_id", "particle_id"]],
                        on=["event_id", "track_id"],
                        how="left",
                        validate="many_to_one",
                    )["particle_id"]

            particles = pd.merge(
                track_particles,
                particles,
                on=["event_id", "particle_id"],
                # how="inner",
                how="left",
                validate="many_to_one",
            )
            tracks = track_params

        if all(
            col in hits.columns
            for col in [
                "particles_vertex_primary",
                "particles_vertex_secondary",
                "particles_particle",
                "particles_generation",
                "particles_sub_particle",
            ]
        ):
            # Combine vertex_primary, vertex_secondary, particle, generation and sub_particle into a single string
            if not "particle_id" in hits:
                hits["particle_id"] = hits[
                    [
                        "particles_vertex_primary",
                        "particles_vertex_secondary",
                        "particles_particle",
                        "particles_generation",
                        "particles_sub_particle",
                    ]
                ].apply(lambda row: "_".join(row.values.astype(str)), axis=1)
            particles["particle_id"] = particles[
                [
                    "vertex_primary",
                    "vertex_secondary",
                    "particle",
                    "generation",
                    "sub_particle",
                ]
            ].apply(lambda row: "_".join(row.values.astype(str)), axis=1)

        # Renaming columns
        particles.rename(
            columns={
                "pt": "pT",
                "phi": "phi0",
                "eta": "peta",
                "theta": "ptheta",
            },
            inplace=True,
        )

        # Sanity checks
        # assert len(hits["event_id"].unique()) == len(
        #     particles["event_id"].unique()
        # ), f"Mismatch in number of unique event_ids: {len(hits['event_id'].unique())} in hits and {len(particles['event_id'].unique())} in particles"
        assert truth_tracks or len(hits["event_id"].unique()) == len(
            tracks["event_id"].unique()
        ), f"Mismatch in number of unique event_ids: {len(hits['event_id'].unique())} in hits and {len(tracks['event_id'].unique())} in tracks"
        assert (
            len(hits["event_id"].unique()) <= n_events_split
        ), f"Mismatch in number of unique event_ids: {len(hits['event_id'].unique())} in hits and {n_events_split} in split"

        return (
            hits,
            tracks,
            particles,
        )


class DatasetWrapper(Dataset):
    """
    Traditional torch dataloading from saved object.

    Attributes:
        data_file (Path): Path to save/load preprocessed data.
        dataset_dir (Path): Directory containing dataset.
        folder (str): (train, test, val) to load.
    """

    def __init__(
        self,
        dataset_dir,
        folder,
        dataset="tml",
        split_size=1_000_000, # number of tracks per chunk
        dynamic_load=False,
        **kwargs,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.dataset_type = dataset.lower()
        self.folder = folder
        self.load = kwargs.pop("load", True)
        dataset_suffix = kwargs.pop("dataset_suffix", "")
        # Add kwargs input_variables and output_variables if available
        input_variables = kwargs.get("input_variables", ["tx", "ty", "tz"])
        output_variables = kwargs.get("output_variables", ["pT", "pz"])
        variable_suffix = (
            f"_i_{'_'.join(input_variables)}_o_{'_'.join(output_variables)}"
        )
        dataset_suffix = variable_suffix + (
            "_" + dataset_suffix if dataset_suffix else ""
        )
        self.data_file_suffix = dataset_suffix
        self.data_file = (
            self.dataset_dir / f"preprocessed_{self.folder}{self.data_file_suffix}.pt"
        )
        self.split_size = split_size  # Number of samples per chunk
        self.wrapper_workers = kwargs.get("wrapper_workers", int(os.cpu_count()))
        if self.wrapper_workers and self.wrapper_workers > 0:
            torch.multiprocessing.set_sharing_strategy("file_system")
        self.datalist = None
        self.dynamic_load = dynamic_load
        self.current_loaded_chunk = -1

        # Check if dataset is valid
        if self.dataset_type not in ("tml", "acts", "acts_root"):
            raise ValueError(
                f"Invalid dataset type '{dataset}'. Expected 'tml', 'acts' or 'acts_root'."
            )

        # Set the dataset class
        if self.dataset_type == "tml":
            self.ds_class = TrackMLDataset
        elif self.dataset_type == "acts":
            self.ds_class = ActsDataset
        elif self.dataset_type == "acts_root":
            self.ds_class = ActsRootDataset

        # Add kwargs to the class
        self.ds_class_kwargs = kwargs

        self.__setup()

    def __setup(self):
        """Sets up the dataset by loading from preprocessed data if available, or processing and saving it."""
        # Preprocess the data if not already done
        if not self._is_preprocessed():
            self._preprocess_data()
        else:
            console.print(
                f"Preprocessed data already exists for {self.folder} folder.", style="cyan"
            )

        # If load is set to False, skip loading the data
        if not self.load:
            console.print(
                "Skipping loading data as load is set to False.", style="yellow"
            )
            return
        # If dynamic_load is set to True, load the data dynamically (i.e., on demand)
        # This is useful for large datasets that cannot fit into memory
        if self.dynamic_load:
            console.print(
                "Dynamic loading is enabled. Data will be loaded on demand.",
                style="yellow",
            )
            self.datalist = []
            return

        # Load the data from the preprocessed file
        if self.data_file.is_file():
            console.print(f"Loading data from {self.data_file}", style="cyan")
            data = torch.load(self.data_file)
            self.datalist = [data]
        else:
            self.datalist = self._load_split_data()
        self._build_index_map()

    def _is_preprocessed(self):
        """Check if the dataset has been preprocessed."""
        return self.data_file.is_file() or self._is_split_data()

    def _is_split_data(self):
        """Check if the dataset is split into multiple files."""
        final_split_filename = self.data_file.with_name(
            f"preprocessed_{self.folder}{self.data_file_suffix}_final{self.data_file.suffix}"
        )
        return final_split_filename.is_file()

    def _load_split_data(self) -> list[dict]:
        """Loads the split data from multiple files."""
        datalist = []
        i = 0
        while True:
            split_filename = self.data_file.with_name(
                f"preprocessed_{self.folder}{self.data_file_suffix}_chunk_{i}{self.data_file.suffix}"
            )
            if split_filename.is_file():
                console.print(f"Loading split data from {split_filename}", style="cyan")
                split_data = torch.load(split_filename)
                datalist.append(split_data)
                i += 1
            else:
                break
        # Load the final chunk if it exists
        final_filename = self.data_file.with_name(
            f"preprocessed_{self.folder}{self.data_file_suffix}_final{self.data_file.suffix}"
        )
        if final_filename.is_file():
            console.print(f"Loading final data from {final_filename}", style="cyan")
            final_data = torch.load(final_filename)
            datalist.append(final_data)
        else:
            raise FileNotFoundError(
                f"Final file {final_filename} does not exist, there is an error in the preprocessing."
            )
        return datalist

    def _load_full_data(self):
        """Loads the entire dataset from a single file."""
        if self.data_file.is_file():
            console.print(f"Loading data from {self.data_file}", style="cyan")
            return torch.load(self.data_file)

    def _preprocess_data(self):
        """Preprocesses the dataset if not already done."""
        console.print(
            "Processing and saving data...",
            style="cyan",
        )
        already_preprocessed = self.split_size * self._get_next_split_index()
        if already_preprocessed > 0:
            console.print(
                f"Already preprocessed {already_preprocessed} samples. Resuming from there...",
                style="yellow",
            )
        else:
            console.print(
                "No preprocessed data found. Starting from scratch.", style="red"
            )
        ds = self.ds_class(self.dataset_dir, self.folder, **self.ds_class_kwargs)
        ds_loader = DataLoader(ds, num_workers=self.wrapper_workers)
        zxy_list, mask_list, target_list, extra_list = [], [], [], []
        particle_index = -1
        for particle_index, variables in enumerate(ds_loader):
            if particle_index < already_preprocessed:
                continue
            if particle_index % 1000 == 0:
                print(f"Processing particle {particle_index}")

            # Add the current batch of data to the chunk
            zxy, mask, target_tensor, extra_tensor = [var.squeeze() for var in variables]
            zxy_list.append(zxy)
            mask_list.append(mask)
            target_list.append(target_tensor)
            extra_list.append(extra_tensor)

            # If the chunk reaches the split_size, save it and clear the chunk
            if len(zxy_list) >= self.split_size:
                self._save_data(zxy_list, mask_list, target_list, extra_list)
                # Clear the chunk after saving
                zxy_list.clear()
                mask_list.clear()
                target_list.clear()
                extra_list.clear()

        # Save any remaining data after the loop ends
        if zxy_list:
            self._save_data(zxy_list, mask_list, target_list, extra_list, final=True)

        print(f"Processed {particle_index+1} particles")
        if particle_index < 0:
            raise ValueError("No particles were processed. Check your dataset.")

    def _save_data(self, zxy_list, mask_list, target_list, extra_list, final=False):
        """Saves the dataset chunk, splitting it into parts if necessary based on split_size."""
        zxy_tensor = pad_sequence(zxy_list, batch_first=True, padding_value=0.0)
        mask_tensor = pad_sequence(mask_list, batch_first=True, padding_value=0)
        target_tensor = torch.stack(target_list)
        extra_tensor = torch.stack(extra_list)

        lengths = torch.tensor([z.shape[0] for z in zxy_list])
        data_to_save = {
            "zxy": zxy_tensor,
            "mask": mask_tensor,
            "target": target_tensor,
            "lengths": lengths,
            "extra": extra_tensor,
        }
        # Save the chunk to a split file
        if final:
            # If this is the final chunk, save it with a different name
            split_filename = self.data_file.with_name(
                f"preprocessed_{self.folder}{self.data_file_suffix}_final{self.data_file.suffix}"
            )
        else:
            # Increment the split index for the next chunk
            split_filename = self.data_file.with_name(
                f"preprocessed_{self.folder}{self.data_file_suffix}_chunk_{self._get_next_split_index()}{self.data_file.suffix}"
            )
        # Save the chunk data
        torch.save(data_to_save, split_filename)
        print(f"Chunk dataset saved to {split_filename}")

    def _get_next_split_index(self):
        """Get the next index for the split dataset file."""
        # Check how many files already exist
        i = 0
        while (
            self.data_file.with_name(
                f"preprocessed_{self.folder}{self.data_file_suffix}_chunk_{i}{self.data_file.suffix}"
            )
        ).is_file():
            i += 1
        return i

    def _build_index_map(self):
        self.index_map = []
        for chunk_idx, chunk in enumerate(self.datalist):
            n = chunk["zxy"].shape[0]
            self.index_map.extend([(chunk_idx, i) for i in range(n)])

    def __getitem__(self, index):
        """Returns the data at the specified index."""
        if self.dynamic_load:
            # If dynamic loading is enabled, load the data on demand
            # Find the chunk file that contains the index
            chunk_index = index // self.split_size
            chunk_offset = index % self.split_size
            chunk_filename = self.data_file.with_name(
                f"preprocessed_{self.folder}{self.data_file_suffix}_chunk_{chunk_index}{self.data_file.suffix}"
            )
            if not chunk_filename.is_file():
                # Consider the case where the chunk is the final one
                # If the chunk index is not 0 check if the previous chunk is valid
                # The final chunk is then supposed to be the one containing the index
                if chunk_index != 0:
                    previous_chunk_filename = self.data_file.with_name(
                        f"preprocessed_{self.folder}{self.data_file_suffix}_chunk_{chunk_index - 1}{self.data_file.suffix}"
                    )
                    if not previous_chunk_filename.is_file():
                        raise FileNotFoundError
                chunk_filename = self.data_file.with_name(
                    f"preprocessed_{self.folder}{self.data_file_suffix}_final{self.data_file.suffix}"
                )
            chunk = torch.load(chunk_filename)
            zxy = chunk["zxy"][chunk_offset]
            mask = chunk["mask"][chunk_offset]
            target = chunk["target"][chunk_offset]
            length = chunk["lengths"][chunk_offset]
            zxy = zxy[:length]
            mask = mask[:length]
            return zxy, mask, target
        else:
            chunk_idx, sample_idx = self.index_map[index]
            chunk = self.datalist[chunk_idx]
            zxy = chunk["zxy"][sample_idx]
            mask = chunk["mask"][sample_idx]
            target = chunk["target"][sample_idx]
            length = chunk["lengths"][sample_idx]
            zxy = zxy[:length]
            mask = mask[:length]
            return zxy, mask, target

    def __len__(self):
        """Returns the length of the dataset."""
        if self.dynamic_load:
            # If dynamic loading is enabled, calculate the length based on the number of chunks
            # and the split size
            count = 0
            i = 0
            while True:
                chunk_filename = self.data_file.with_name(
                    f"preprocessed_{self.folder}{self.data_file_suffix}_chunk_{i}{self.data_file.suffix}"
                )
                if not chunk_filename.is_file():
                    break
                chunk = torch.load(chunk_filename, map_location="cpu")
                count += chunk["zxy"].shape[0]
                i += 1
            # Add the length of the final chunk if it exists
            final_filename = self.data_file.with_name(
                f"preprocessed_{self.folder}{self.data_file_suffix}_final{self.data_file.suffix}"
            )
            if final_filename.is_file():
                chunk = torch.load(final_filename, map_location="cpu")
                count += chunk["zxy"].shape[0]
            return count
        else:
            return len(self.index_map)


class ShardedChunkDataset(DatasetWrapper, IterableDataset):
    def __init__(self, dataset_dir, folder, split_size=1_000_000, **kwargs):
        super().__init__(dataset_dir, folder, split_size=split_size, **kwargs)
        import glob

        self.chunk_files = sorted(
            glob.glob(
                str(
                    self.dataset_dir
                    / f"preprocessed_{self.folder}{self.data_file_suffix}_chunk_*.pt"
                )
            )
        )
        final_chunk = (
            self.dataset_dir
            / f"preprocessed_{self.folder}{self.data_file_suffix}_final.pt"
        )
        if final_chunk.is_file():
            self.chunk_files.append(str(final_chunk))

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            assigned_chunks = self.chunk_files
        else:
            total_workers = worker_info.num_workers
            worker_id = worker_info.id
            assigned_chunks = self.chunk_files[worker_id::total_workers]

        for chunk_path in assigned_chunks:
            chunk_data = torch.load(chunk_path, map_location="cpu")
            for sample in chunk_data:
                yield sample


class DataModule(L.LightningDataModule):
    """
    Lightning DataModule for managing TrackML or ACTS datasets.
    Args:
        dataset_type (str): Type of dataset ('tml' or 'acts').
        dataset_dir (path): the path to where the dataset file is
    """

    def __init__(
        self,
        dataset_type,
        dataset_dir,
        batch_size=32,
        num_workers=os.cpu_count() - 2,
        use_wrapper=True,
        persistance=False,
        pin_memory=False,
        dynamic_load=False,
        kwargs={},  # kwargs for the dataset class
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["_class_path"])
        dataset = self.hparams.dataset_type.lower()

        # Check if dataset is valid
        if dataset not in ("tml", "acts", "acts_root"):
            raise ValueError(
                f"Invalid dataset_type '{dataset}'. Expected 'tml' or 'acts'."
            )

        # Set the dataset class
        if use_wrapper:
            if not dynamic_load:
                self.dataset_class = DatasetWrapper
            else:
                self.dataset_class = ShardedChunkDataset
        elif dataset == "tml":
            self.dataset_class = TrackMLDataset
        elif dataset == "acts":
            self.dataset_class = ActsDataset
        elif dataset == "acts_root":
            self.dataset_class = ActsRootDataset

        # Add kwargs to the class
        self.dataset_class_kwargs = kwargs

    def setup(self, stage=None):
        """Setup datasets for training, validation, and testing."""
        console.rule(f"{self.hparams.dataset_type.capitalize()} Dataset")

        if stage in ("fit", None):
            self.train_dataset = self._create_dataset("train")
            self.val_dataset = self._create_dataset("val")

        if stage == "validate":
            self.val_dataset = self._create_dataset("train", load=False)
            self.val_dataset = self._create_dataset("val", load=False)

        if stage in ("test", None):
            self.test_dataset = self._create_dataset("test")

    def preprocess_data(self, folders=("train", "val", "test")):
        """Run dataset preprocessing without creating dataloaders or training.

        This instantiates the configured wrapper datasets with ``load=False`` so
        that the preprocessing step writes the cached files and exits before any
        data loading for training starts.

        Args:
            folders (tuple[str, ...]): Dataset folders to preprocess.
        """
        if not self.hparams.use_wrapper:
            raise ValueError(
                "preprocess_data() requires use_wrapper=True so the preprocessed "
                "files can be written by DatasetWrapper."
            )

        console.rule(f"Preprocessing {self.hparams.dataset_type.capitalize()} Dataset")
        for folder in folders:
            console.print(f"Preprocessing folder '{folder}'", style="cyan")
            self._create_dataset(folder, load=False)

        console.print(f"Finished preprocessing {self.hparams.dataset_type.capitalize()} Dataset", style="cyan")

    def train_dataloader(self):
        return self._create_dataloader(self.train_dataset)

    def val_dataloader(self):
        return self._create_dataloader(self.val_dataset)

    def test_dataloader(self):
        return self._create_dataloader(self.test_dataset)

    def _create_dataloader(self, dataset):
        """Helper function to initialize data loaders."""
        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=self.collate_fn,
            persistent_workers=bool(self.hparams.num_workers)
            and self.hparams.persistance,
            pin_memory=self.hparams.pin_memory,
        )

    def _create_dataset(self, folder, load=True):
        """Helper method to create dataset for the given folder"""
        return self.dataset_class(
            dataset_dir=self.hparams.dataset_dir,
            folder=folder,
            dataset=self.hparams.dataset_type,
            **self.dataset_class_kwargs,
            dynamic_load=self.hparams.dynamic_load,
            load=load,
        )

    @staticmethod
    def collate_fn(batch):
        """Generic collate function for padding sequences."""
        inputs, masks, targets = zip(*batch)
        inputs = pad_sequence(inputs, batch_first=True)
        masks = pad_sequence(masks, batch_first=True, padding_value=0)
        return inputs, masks, torch.stack(targets, dim=0)
