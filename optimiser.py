import sys
import os
import torch
import numpy as np
import pandas as pd
import argparse
import cv2
from scipy.optimize import minimize
from scipy.spatial import distance

sys.path.append('/home/sam/Documents/qlab/StyleGANEX')

from models.psp import pSp


def load_torch_embeddings(directory):
    embeddings = {}
    for filename in os.listdir(directory):
        if filename.endswith(".pt"):
            file_path = os.path.join(directory, filename)
            embedding = torch.load(file_path)['wplus'].numpy()
            base_filename = filename.split('_')[0]
            embeddings[base_filename] = embedding
    return embeddings

def test_embedding(path):
    embedding = torch.load(path)['wplus'].numpy()
    return embedding
    
def load_scores(directory):
    all_dfs = []

    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            file_path = os.path.join(directory, filename)
            df = pd.read_csv(file_path)
            df['filename'] = df['image'].apply(lambda x: os.path.basename(x).split('.')[0])
            all_dfs.append(df)

    total_df = pd.concat(all_dfs)

    median_scores = total_df.groupby('filename')['label'].median()

    return median_scores.to_dict()

def filter_embeddings_and_scores(embeddings, scores):
    filtered_embeddings = {}
    filtered_scores = {}

    for name, emb in embeddings.items():
        if name in scores:
            filtered_embeddings[name] = emb
            filtered_scores[name] = scores[name]

    return filtered_embeddings, filtered_scores

def load_styleganex_inversion_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    opts = ckpt['opts']
    opts['checkpoint_path'] = ckpt_path
    opts['device'] = device
    opts = argparse.Namespace(**opts)
    pspex = pSp(opts).to(device).eval()
    pspex.latent_avg = pspex.latent_avg.to(device)
    return pspex

def generate_image_from_embedding(pspex, embedding, device):
    if embedding.ndim < 3:
        embedding = embedding.reshape(1, *embedding.shape[-2:])
    embedding_tensor = torch.from_numpy(embedding).to(device).float()
    with torch.no_grad():
        y_hat, _ = pspex.decoder([embedding_tensor], input_is_latent=True, randomize_noise=False)
        y_hat = torch.clamp(y_hat, -1, 1)
        return y_hat

def select_high_scoring_embeddings(embeddings, scores, top_n=10):
    sorted_embeddings = sorted(embeddings.items(), key=lambda x: scores.get(x[0], 0), reverse=True)
    return dict(sorted_embeddings[:top_n])

def interpolate_embeddings(embedding1, embedding2, alpha):
    if embedding1.ndim != embedding2.ndim:
        embedding1 = embedding1.reshape(embedding2.shape)
    return (1 - alpha) * embedding1 + alpha * embedding2

def estimate_attractiveness_score(synthetic_embedding, reference_embeddings, reference_scores, k=3):
    closest_embeddings = sorted(
        reference_embeddings.items(), 
        key=lambda x: distance.euclidean(synthetic_embedding.flatten(), x[1].flatten())
    )[:k]

    total_score = 0
    total_weight = 0
    for name, emb in closest_embeddings:
        emb_distance = distance.euclidean(synthetic_embedding.flatten(), emb.flatten())
        weight = 1 / (emb_distance + 1e-5)
        total_score += reference_scores[name] * weight
        total_weight += weight

    return total_score / total_weight if total_weight > 0 else 0

def optimise_embedding_for_attractiveness(input_embedding, reference_embeddings, reference_scores, max_iterations=500):
    optimization_bounds = [(0, 1)] * np.prod(input_embedding.shape)
    best_score = -float('inf')
    best_embedding = None

    for _, initial_embedding in reference_embeddings.items():
        def objective_function(embedding):
            synthetic_embedding = interpolate_embeddings(input_embedding, embedding.reshape(input_embedding.shape), alpha=0.5)
            return -estimate_attractiveness_score(synthetic_embedding, reference_embeddings, reference_scores)

        result = minimize(objective_function, initial_embedding.flatten(), bounds=optimization_bounds, method='L-BFGS-B', options={'maxiter': max_iterations})
        optimised_embedding = result.x.reshape(input_embedding.shape)
        optimised_score = -objective_function(optimised_embedding.flatten())

        if optimised_score > best_score:
            best_score = optimised_score
            best_embedding = optimised_embedding

    return best_embedding, best_score

def save_image(img, filename):
    tmp = ((img.detach().cpu().numpy().transpose(1, 2, 0) + 1.0) * 127.5).astype(np.uint8)
    cv2.imwrite(filename, cv2.cvtColor(tmp, cv2.COLOR_RGB2BGR))

def main():
    np.random.seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    embeddings = load_torch_embeddings('./inversions')
    scores = load_scores('./scores')

    embeddings, scores = filter_embeddings_and_scores(embeddings, scores)

    pspex = load_styleganex_inversion_model('./models/styleganex_inversion.pt', device)

    reference_embeddings = select_high_scoring_embeddings(embeddings, scores, top_n=10)
    embedding_shape = list(embeddings.values())[0].shape

    input_embedding = np.random.rand(*embedding_shape)
    #input_embedding = test_embedding("inversions/ILip77SbmOE_inversion.pt")

    optimised_embedding, optimised_score = optimise_embedding_for_attractiveness(input_embedding, reference_embeddings, scores)
    generated_image = generate_image_from_embedding(pspex, optimised_embedding, device)

    save_image(generated_image[0].cpu(), 'optimised_attractive_image.jpg')
    print(optimised_score)

if __name__ == "__main__":
   main()
