def load_clean_csv(path):
    try:
        df = pd.read_csv(path, header=None, encoding="latin1", engine="python")

        if df.shape[1] == 1:
            col = df[0].astype(str).str.strip()
            df = col.str.split(",", expand=True)

            if df.shape[1] == 1:
                df = col.str.split(r"\s+", expand=True)

        if df.shape[1] < 3:
            print(" Bad format:", path)
            return None

        df = df.iloc[:, :3]
        df.columns = ["x", "y", "z"]

        df = df.apply(pd.to_numeric, errors='coerce')
        df = df.dropna()

        if len(df) < WINDOW_SIZE:
            print("❌ Too small:", path)
            return None

        return df

    except Exception as e:
        print(" Error:", path, e)
        return None

