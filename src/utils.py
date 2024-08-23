# -*- coding: utf-8 -*-
"""
@author: avasque1@jh.edu
"""
import os, re, yaml, time, shutil
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp

from PIL import Image
from tqdm import tqdm
from collections import defaultdict
from scipy.ndimage import gaussian_filter1d

import sascorer
from rdkit import Chem
from rdkit.Chem.Descriptors import CalcMolDescriptors
from rdkit.Chem import QED, Descriptors, AllChem, rdMolDescriptors, Draw

import torch
import torch.optim as optim

def read_yaml(file_path):
    '''
    Reads a YAML file and returns its contents as a Python dictionary.
    '''
    try:
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)
        return data
    except Exception as e:
        print(f"Error reading the YAML file: {e}")
        return None

def make_dir(path: str) -> None:
    '''
    Create results directory
    '''
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path)
    print('Created Folder at: ', path)

def count_parameters(model: object) -> int:
    '''
    Counts total network parameters
    '''
    return sum(p.numel() for p in model.parameters())

def count_trainable_parameters(model):
    '''
    Only counts trainable parameters
    '''
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_mol(smiles):
    '''
    Converts SMILES string to a RDKit MOL object
    '''
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return mol

def getMolecularDescriptors(smiles):
    '''
    Extract RDKit molecular descriptors
    '''
    features = CalcMolDescriptors(get_mol(smiles))
    return np.array(list(features.values()))

def tokenize_smiles(smiles: str) -> list:
    '''
    Generate tokens
    '''
    pattern = r"(\[[^\[\]]*\])"
    tokens = re.split(pattern, smiles)
    return [token for token in tokens if token]

def build_vocabulary(smiles_list: list) -> dict:
    '''
    Build reference vocab
    '''
    vocab = defaultdict(int)
    for smiles in smiles_list:
        tokens = tokenize_smiles(smiles)
        for token in tokens:
            if token not in vocab:
                vocab[token] = len(vocab) + 1
    return vocab

def encode_smiles(smiles: str, vocab: dict) -> list:
    '''
    Encodes (tokenizes) smiles 
    '''
    tokens = tokenize_smiles(smiles)
    return [vocab[token] for token in tokens]

def decode_smiles(encoded_smiles: list, vocab: dict) -> str:
    '''
    Decodes (de-tokenizes) smiles
    '''
    inv_vocab = {v: k for k, v in vocab.items()}
    return ''.join([inv_vocab[token] for token in encoded_smiles])

def plot_losses(history: dict, sigma: int, save: bool, savepath: str) -> None:
    '''
    Plots discriminator and generator losses on same axis 
    '''
    fig, axes = plt.subplots(2, 1, sharex=True)

    ##populate upper plot
    axes[0].plot(history['epoch'], gaussian_filter1d(history['g_loss'], sigma=sigma), label='g_loss')
    axes[0].set_ylabel('Generator Loss')

    ##populate lower plot
    axes[1].plot(history['epoch'], gaussian_filter1d(history['d_loss'], sigma=sigma), c='orange', label='d_loss')
    axes[1].set_ylabel('Discriminator Loss')

    ##configure plots
    axes[0].grid()
    axes[1].grid()
    axes[0].legend(loc='best')
    axes[1].legend(loc='best')
    axes[1].set_xlabel('Epochs')
    fig.suptitle('Generator and Discriminator Losses')
    
    if save:
        plt.savefig(savepath + '/train_losses.png')

def generate_smiles(generator: object, vocab: list, num_samples: int, max_length: int, device: object) -> list:
    """
    Generate SMILES strings using the trained generator
    """
    with torch.no_grad():
        generator.eval()  #set to eval mode
        noise = torch.randint(1, max_length, (num_samples, max_length)).to(device) #gen random noise with desired size
        generated_indices = generator(noise).long().detach().cpu().numpy() #make gens
        return [decode_smiles(indices, vocab) for indices in generated_indices]

def process_smiles_in_parallel(smiles_list: list, function_object: object, n_processors: int) -> tuple:
    '''
    Uses python's multiprocess library to generate data
    '''
    with mp.Pool(processes=n_processors) as pool:
        results = pool.map(function_object, smiles_list)
    return results

def get_mol(smiles: str) -> object:
    '''
    Gets RDKit Generated Mol Object for a single smiles string
    '''
    return Chem.MolFromSmiles(smiles)

def calc_check_validity(smiles: str) -> int:
    '''
    Calculates validity of a single smiles string
    '''
    mol = get_mol(smiles)
    if mol:
        return 1
    else:
        return 0

##Run capture. There's a known bug with RDKit that outputs warnings.
def check_smiles_validity(smiles_list: list) -> tuple:
    """
    Check the smiles validity using RDKit (list of smiles)
    """
    valid_smiles, invalid_smiles = [], []
    for smiles in smiles_list:
        mol = get_mol(smiles)
        if mol:
            valid_smiles.append(smiles)
        else:
            invalid_smiles.append(smiles)
    return valid_smiles, invalid_smiles

def canonicalize_smiles(smiles: list) -> object:
    """
    Canonicalize the input SMILES string.
    """
    mol = get_mol(smiles)
    if mol:
        return Chem.MolToSmiles(mol, canonical=True)
    else:
        return None

def is_novel(canonical_smiles: str, known_canonical_smiles: list) -> int:
    '''
    Checks uniqueness of a single smiles canonical representation vs known canonical smiles
    '''
    if canonical_smiles is None:
        return None
    if canonical_smiles in known_canonical_smiles:
        return 0
    else:
        return 1

def estimate_solubility(smiles: str) -> tuple:
    '''
    Estimates aqeous solubility of a single smiles string
    '''
    mol = get_mol(smiles)
    if mol is None:
        return 0
    
    ##calc mol descriptors
    mol_weight = Descriptors.MolWt(mol)
    logP = Descriptors.MolLogP(mol)
    num_h_donors = Descriptors.NumHDonors(mol)
    num_h_acceptors = Descriptors.NumHAcceptors(mol)
    
    ##linear model to calculate solubility
    logS = -0.01 * mol_weight + 0.5 * num_h_donors + 0.5 * num_h_acceptors - logP
    return np.round(logS, 3)

def calculate_qed(smiles: list) -> float:
    """
    Calculate Quantitative Estimation of Drug-likeness score (QED) for a single smiles string
    """
    mol = get_mol(smiles)
    if mol is None:
        return None
        
    qed_score = QED.qed(mol)
    return np.round(qed_score, 3)

def calculate_sa_score(smiles: str) -> float:
    """
    Calculates the synthetic accessibility (SA) score for a given SMILES string.
    Lower scores indicate easier synthesis.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string")
    
    ##compute SA score using the sascorer script
    sa_score = sascorer.calculateScore(mol) / (rdMolDescriptors.CalcNumRings(mol) + 1.0)
    return np.round(sa_score, 3)

def plot_single_mol(smiles: str, size=(400, 400)) -> object:
    '''
    Plot Only a single molecule
    '''
    mol = get_mol(smiles)
    img = Draw.MolToImage(mol, size=(500, 500))
    return img

def plot_smiles_grid(smiles_list: list, save=True, grid_size=(4, 4)) -> None:
    '''
    Plots a grid of molecule images.
    '''
    mols = [get_mol(smile) for smile in smiles_list]
    img = Draw.MolsToGridImage(mols, molsPerRow=grid_size[1], subImgSize=(400, 200), useSVG=False, returnPNG=True)
    return img

def summary_stats(valid_smiles: list, invalid_smiles: list) -> list:
    '''
    Basic Stats
    '''
    tot_gen_cnt = len(valid_smiles) + len(invalid_smiles)
    valid_cnt = len(valid_smiles)
    unique_valid_cnt = len(set(valid_smiles))
    invalid_cnt = len(invalid_smiles)
    unique_invalid_cnt = len(list(set(invalid_smiles)))
    unique_valid_perc = np.round(100*(unique_valid_cnt/tot_gen_cnt), 3)
    unique_invalid_perc = np.round(100*(unique_invalid_cnt/tot_gen_cnt), 3)
    unique_val_ratio = np.round(unique_valid_cnt/unique_invalid_cnt, 2)
    
    print("total generated cnt: ", tot_gen_cnt)
    print("valid cnt: ", valid_cnt)
    print("invalid Count: ", invalid_cnt)
    print("unique valid cnt: ", unique_valid_cnt)
    print("unique invalid cnt: ", unique_invalid_cnt)
    print("percent unique valid: ", unique_valid_perc)
    print("percent unique invalid: ", unique_invalid_perc)
    print("ratio unique_valid/unique_invalid: ", unique_val_ratio)

    stats = [tot_gen_cnt, valid_cnt, unique_valid_cnt, invalid_cnt, unique_invalid_cnt, unique_valid_perc, 
             unique_invalid_perc, unique_val_ratio]
    return stats

def train_gan(generator: object, discriminator: object, g_optimizer: object, d_optimizer: object, criterion: object, g_schedule: object, d_schedule: object, data_loader: object, 
              run_extra_times: int, clip_value: float, n_epochs: int, multi_gpu: bool, device: object, print_loss_every: int) -> tuple:
    '''
    Simple GAN Trainer
    '''
    history = {'epoch': [] ,'g_loss': [], 'd_loss': []}
    for epoch in range(n_epochs):
        t1 = time.time()
        for real_smiles in tqdm(data_loader):
            batch_size = real_smiles.size(0)

            ##according to WGAN, this should help to stabalize training, original number was 5 -times disc updates to every 1 gen updates
            if run_extra_times:
                for i in range(run_extra_times):
                    ##-----train discriminator-----
                    d_optimizer.zero_grad()
        
                    ##generate real and fake labels (real are ones and fake are zeros)
                    real_labels = torch.ones(batch_size, 1).to(device)
                    fake_labels = torch.zeros(batch_size, 1).to(device)
        
                    ##gen real smiles and forward pass discriminator
                    real_smiles = real_smiles.to(device)
                    real_outputs = discriminator(real_smiles)
        
                    ##calc binary loss
                    d_loss_real = criterion(real_outputs, real_labels)
        
                    ##if multi-gpu, then need to use network module, otherwise use network directrly
                    if multi_gpu:
                        noise = torch.randint(1, generator.module.max_length, (batch_size, generator.module.max_length)).to(device)
                    else:
                        noise = torch.randint(1, generator.max_length, (batch_size, generator.max_length)).to(device)
        
                    ##get fake smiles by forward passing generator
                    fake_smiles = generator(noise)
                    fake_smiles = fake_smiles.long()
        
                    ##show fake smiles to discriminator then calculate loss based on what the discriminator thinks is fake
                    fake_outputs = discriminator(fake_smiles)
                    d_loss_fake = criterion(fake_outputs, fake_labels)
                    
                    ##TODO: Add gradient clipping by following WGAN-GC
                    
                    ##sum loss, backprop, and update weights
                    d_loss = d_loss_real + d_loss_fake
                    d_loss.backward()
                    d_optimizer.step()

                    ##Apply weight clipping to discriminator
                    for p in discriminator.parameters():
                        p.data.clamp_(-clip_value, clip_value)

            ##-----Train Generator-----
            g_optimizer.zero_grad()

            ##reused noise vector
            fake_smiles = generator(noise)
            fake_smiles = fake_smiles.long()

            ###show fake smiles 
            fake_outputs = discriminator(fake_smiles)

            ##calculate the generated loss by comparing fake_outputs to real_labels
            ##want these two to have very low loss
            ##back prop and update weights
            g_loss = criterion(fake_outputs, real_labels)
            g_loss.backward()
            g_optimizer.step()

        ##Logging and Displaying Losses
        if (epoch) % print_loss_every == 0:
            t2 = time.time() - t1
            print(f'Epoch [{epoch}/{n_epochs}], D Loss: {d_loss.item():.5f}, G Loss: {g_loss.item():.5f}, Runtime/Epoch: {t2:.5f}')

        ##Step Schedulers
        g_schedule.step()
        d_schedule.step()
        
        ##record Losses
        history['epoch'].append(epoch)
        history['g_loss'].append(g_loss.item())
        history['d_loss'].append(d_loss.item())

    del real_labels, fake_labels, real_smiles, fake_smiles, d_loss, g_loss
    torch.cuda.empty_cache()
    
    return history, generator, discriminator