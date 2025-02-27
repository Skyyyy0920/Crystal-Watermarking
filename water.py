import os
import time
import torch
import argparse
import chemparse
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from torch_geometric.data import DataLoader, Data
from eval_utils import load_model, lattices_to_params_shape, get_crystals_list
from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.io.cif import CifWriter
from p_tqdm import p_map

chemical_symbols = [
    # 0
    'X',
    # 1
    'H', 'He',
    # 2
    'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    # 3
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
    # 4
    'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
    # 5
    'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
    'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
    # 6
    'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy',
    'Ho', 'Er', 'Tm', 'Yb', 'Lu',
    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi',
    'Po', 'At', 'Rn',
    # 7
    'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk',
    'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
    'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc',
    'Lv', 'Ts', 'Og']


def diffusion(loader, model, step_lr):
    frac_coords = []
    num_atoms = []
    atom_types = []
    lattices = []
    for idx, batch in enumerate(loader):
        if torch.cuda.is_available():
            batch.cuda()

        batch_size = batch.num_graphs
        l_T, x_T = torch.randn([batch_size, 3, 3]).to(model.device), torch.rand([batch.num_nodes, 3]).to(model.device)

        watermark_pattern_l = generate_watermark_pattern(l_T.shape)
        watermark_pattern_x = generate_watermark_pattern(x_T.shape)
        l_T_watered = add_watermark_to_noise(l_T, watermark_pattern_l)
        x_T_watered = add_watermark_to_noise(x_T, watermark_pattern_x)

        outputs, traj = model.sample(batch, l_T_watered, x_T_watered, step_lr)

        frac_coords.append(outputs['frac_coords'].detach().cpu())
        num_atoms.append(outputs['num_atoms'].detach().cpu())
        atom_types.append(outputs['atom_types'].detach().cpu())
        lattices.append(outputs['lattices'].detach().cpu())

    frac_coords = torch.cat(frac_coords, dim=0)
    num_atoms = torch.cat(num_atoms, dim=0)
    atom_types = torch.cat(atom_types, dim=0)
    lattices = torch.cat(lattices, dim=0)
    lengths, angles = lattices_to_params_shape(lattices)

    return frac_coords, atom_types, lattices, lengths, angles, num_atoms


class SampleDataset(Dataset):
    def __init__(self, formula, num_evals):
        super().__init__()
        self.formula = formula
        self.num_evals = num_evals
        self.get_structure()

    def get_structure(self):
        self.composition = chemparse.parse_formula(self.formula)
        chem_list = []
        for elem in self.composition:
            num_int = int(self.composition[elem])
            chem_list.extend([chemical_symbols.index(elem)] * num_int)
        self.chem_list = chem_list

    def __len__(self) -> int:
        return self.num_evals

    def __getitem__(self, index):
        return Data(
            atom_types=torch.LongTensor(self.chem_list),
            num_atoms=len(self.chem_list),
            num_nodes=len(self.chem_list),
        )


def get_pymatgen(crystal_array):
    frac_coords = crystal_array['frac_coords']
    atom_types = crystal_array['atom_types']
    lengths = crystal_array['lengths']
    angles = crystal_array['angles']
    try:
        structure = Structure(
            lattice=Lattice.from_parameters(
                *(lengths.tolist() + angles.tolist())),
            species=atom_types, coords=frac_coords, coords_are_cartesian=False)
        return structure
    except:
        return None


def generate_watermark_pattern(shape):
    """
    Generate a low-frequency mask where the low-frequency region is set to 1 and the rest is set to 0.
    The low-frequency region includes the first row, the first column, and the top-left corner area.

    Args:
        shape (tuple): The shape of the input tensor, e.g., (H, W) for 2D or (C, H, W) for 3D.

    Returns:
        low_freq_mask (torch.Tensor): A mask with the same shape as the input, where the low-frequency region is 1 and the rest is 0.
    """
    low_freq_mask = torch.zeros(shape)

    if len(shape) == 2:
        low_freq_mask[0, :] = 1
        low_freq_mask[:, 0] = 1
    elif len(shape) == 3:
        low_freq_mask[:, 0, :] = 1
        low_freq_mask[:, :, 0] = 1
    else:
        raise ValueError("Only 2D (H, W) or 3D (C, H, W) shapes are supported.")

    return low_freq_mask


def add_watermark_to_noise(noise, watermark_pattern, strength=1.0):
    """
    Add watermark to noise using FFT with adjustable strength.

    Args:
        noise (torch.Tensor): The noise tensor (e.g., l_T or x_T).
        watermark_pattern (np.ndarray): A binary or continuous pattern representing the watermark.
        strength (float): Scaling factor for the watermark.

    Returns:
        torch.Tensor: Noise with watermark embedded.
    """
    # Convert noise to numpy for FFT operations
    noise_np = noise.cpu().numpy()

    # Perform FFT on the noise
    fft_noise = np.fft.fftn(noise_np)

    # Ensure the watermark is applied only to the selected low-frequency region
    watermarked_fft = fft_noise.copy()  # Copy the original FFT result
    watermarked_fft[watermark_pattern == 1] += np.mean(watermarked_fft[watermark_pattern != 1])

    # Perform inverse FFT to return to time domain
    watermarked_noise_np = np.fft.ifftn(watermarked_fft).real

    # Convert back to PyTorch tensor
    watermarked_noise = torch.tensor(watermarked_noise_np, dtype=noise.dtype, device=noise.device)

    return watermarked_noise


def detect_watermark_from_noise(noise, watermark_pattern, threshold=0.8):
    """
    Detect watermark from noise using FFT.

    Args:
        noise (torch.Tensor): The noise tensor extracted from the crystal structure.
        watermark_pattern (np.ndarray): The original watermark pattern.
        threshold (float): Similarity threshold to determine if the watermark exists.

    Returns:
        bool: Whether the watermark is detected.
        float: Similarity score between the extracted pattern and the original watermark.
    """
    # Convert noise to numpy for FFT operations
    noise_np = noise.cpu().numpy()

    # Perform FFT on the noise
    fft_noise = np.fft.fftn(noise_np)

    # Extract low-frequency components
    freq_shape = fft_noise.shape
    low_freq_mask = np.zeros(freq_shape)
    low_freq_mask[:freq_shape[0] // 2, :freq_shape[1] // 2, :freq_shape[2] // 2] = 1
    extracted_pattern = fft_noise * low_freq_mask

    # Compute similarity (e.g., normalized cross-correlation)
    similarity = np.corrcoef(extracted_pattern.flatten(), watermark_pattern.flatten())[0, 1]

    # Determine if watermark is detected
    watermark_detected = similarity >= threshold

    return watermark_detected, similarity


def main(args):
    model_path = Path(args.model_path)
    model, _, cfg = load_model(model_path, load_data=False)

    if torch.cuda.is_available():
        model.to('cuda')

    tar_dir = os.path.join(args.save_path, args.formula)
    os.makedirs(tar_dir, exist_ok=True)

    test_set = SampleDataset(args.formula, args.num_evals)
    test_loader = DataLoader(test_set, batch_size=min(args.batch_size, args.num_evals))

    (frac_coords, atom_types, lattices, lengths, angles, num_atoms) = diffusion(test_loader, model, args.step_lr)

    crystal_list = get_crystals_list(frac_coords, atom_types, lengths, angles, num_atoms)

    structure_list = p_map(get_pymatgen, crystal_list)

    for i, structure in enumerate(structure_list):
        tar_file = os.path.join(tar_dir, f"{args.formula}_{i + 1}.cif")
        if structure is not None:
            writer = CifWriter(structure)
            writer.write_file(tar_file)
        else:
            print(f"{i + 1} Error Structure.")

    # Assume we have a generated crystal structure
    generated_noise = extract_noise_from_crystal(generated_crystal)

    # Detect watermark
    watermark_detected, similarity = detect_watermark_from_noise(generated_noise, watermark_pattern)

    if watermark_detected:
        print(f"Watermark detected with similarity {similarity:.2f}")
    else:
        print("No watermark detected.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',
                        # default=r"G:\Crystal Watermarking\Crystal-Watermarking\cdvae\prop_models\diffcsp\perovskite\epoch=124-step=1499.ckpt",
                        default=r"G:\Crystal Watermarking\Crystal-Watermarking\diffcsp\prop_models\perovskite")
    parser.add_argument('--save_path', default=r"G:\Crystal Watermarking\Crystal-Watermarking\save")
    parser.add_argument('--formula', default="H2O")
    parser.add_argument('--num_evals', default=1, type=int)
    parser.add_argument('--batch_size', default=500, type=int)
    parser.add_argument('--step_lr', default=1e-5, type=float)

    args = parser.parse_args()
    main(args)
