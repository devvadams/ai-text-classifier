import pandas as pd

def load_and_clean_data(path):
    print("Loading dataset...")

    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["label", "text"],
        on_bad_lines="skip"
    )

    # Cleaning
    df = df.dropna(subset=["text"])
    df = df[df["text"].astype(str).str.strip() != ""]
    df = df[df["label"].isin(["ham", "spam"])]

    df["text"] = df["text"].astype(str).str.lower().str.strip()

    df = df.drop_duplicates().reset_index(drop=True)

    print(f"Clean dataset size: {len(df)}")

    return df