import sys
import os
import torch
import numpy as np
import random
import pandas as pd
import argparse
import cv2
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor

sys.path.append('/home/sam/Documents/qlab/StyleGANEX')

from models.psp import pSp

def load_torch_embeddings(directory):
    embeddings = {}
    for filename in os.listdir(directory):
        if filename.endswith(".pt"):
            try:
                file_path = os.path.join(directory, filename)
                embedding = torch.load(file_path)['wplus'].numpy()
                base_filename = filename.split('_')[0]
                embeddings[base_filename] = embedding
            except:
                pass
    return embeddings

def load_embedding(file_path):
    return torch.load(file_path)['wplus'].numpy()

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

    inversions_no_scores = []
    scores_no_inversions = []

    for name, emb in embeddings.items():
        if name in scores:
            filtered_embeddings[name] = emb
            filtered_scores[name] = scores[name]
        else:
            inversions_no_scores.append(name)
    
    for score_name in scores.keys():
        if score_name not in embeddings:
            scores_no_inversions.append(score_name)


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

def train_random_forest_model(reference_embeddings, reference_scores):
    X = np.array(list(reference_embeddings.values())).reshape(len(reference_embeddings), -1)
    y = np.array(list(reference_scores.values()))
    model = RandomForestRegressor(n_estimators=50)
    model.fit(X, y)
    return model

def optimise_embedding_for_attractiveness(input_embedding, model, max_iterations=50):
    optimization_bounds = [(-1, 1)] * np.prod(input_embedding.shape)

    def objective_function(embedding):
        synthetic_embedding = embedding.reshape(input_embedding.shape)
        score = model.predict(synthetic_embedding.reshape(1, -1))
        print(f"Current embedding summary: Mean={np.mean(synthetic_embedding)}, Std={np.std(synthetic_embedding)}")
        return -score

    result = minimize(objective_function, input_embedding.flatten(), bounds=optimization_bounds, method='L-BFGS-B', options={'maxiter': max_iterations, 'disp': True})
    optimised_embedding = result.x.reshape(input_embedding.shape)
    optimised_score = -objective_function(optimised_embedding.flatten())

    return optimised_embedding, optimised_score

def save_image(img, filename):
    tmp = ((img.detach().cpu().numpy().transpose(1, 2, 0) + 1.0) * 127.5).astype(np.uint8)
    cv2.imwrite(filename, cv2.cvtColor(tmp, cv2.COLOR_RGB2BGR))

def main():
    np.random.seed(1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    embeddings = load_torch_embeddings('./MEBeauty-database-inversions/')
    scores = load_scores('./scores')

    embeddings, scores = filter_embeddings_and_scores(embeddings, scores)
    pspex = load_styleganex_inversion_model('./models/styleganex_inversion.pt', device)

    print(len(embeddings.keys()))

    print("training")
    random_forest_model = train_random_forest_model(embeddings, scores)
    print("finished")

    unique_shapes = {emb.shape for emb in embeddings.values()}

    if unique_shapes:
        selected_shape = random.choice(list(unique_shapes))
        input_embedding = np.random.uniform(low=-1, high=1, size=selected_shape)
    else:
        raise ValueError("No embeddings found to determine a shape.")
    
    optimised_embedding, optimised_score = optimise_embedding_for_attractiveness(input_embedding, random_forest_model)

    generated_image = generate_image_from_embedding(pspex, optimised_embedding, device)
    save_image(generated_image[0].cpu(), 'optimised_attractive_image.jpg')
    print(optimised_score)

if __name__ == "__main__":
    main()
