import pandas as pd
from sentence_transformers import SentenceTransformer

class DataLoader:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.datasets = {}

    def load_csvs(self, files):
        for name, path in files.items():
            df = pd.read_csv(path)
            # Normalizar columnas
            if "pregunta" not in df.columns or "respuesta" not in df.columns:
                raise ValueError(f"CSV {path} debe tener columnas 'pregunta' y 'respuesta'")
            df = df.dropna(subset=["pregunta", "respuesta"]).reset_index(drop=True)
            df["tema"] = name
            # Precomputar embeddings (tensor)
            df["embedding"] = df["pregunta"].apply(lambda x: self.model.encode(str(x), convert_to_tensor=True))
            self.datasets[name] = df

    def get_all_data(self):
        return pd.concat(self.datasets.values(), ignore_index=True)
