import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PowerTransformer, Normalizer
from src.config.paths import RAW_FILE, PROCESSED_DIR, CLEAN_FILE
from src.utils.logger import get_logger

logger = get_logger()


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(RAW_FILE, low_memory=False)

    # ---------------------------------------------------
    # 1. Drop unnecessary / high-cardinality columns
    # ---------------------------------------------------
    drop_cols = [
        "max_glu_serum",
        "A1Cresult",
        "diag_1",
        "diag_2",
        "diag_3",
        "medical_specialty"
    ]
    data.drop(columns=[c for c in drop_cols if c in data.columns], inplace=True)

    # ---------------------------------------------------
    # 2. Target mapping
    # ---------------------------------------------------
    data["readmitted"] = data["readmitted"].map(
        {"<30": 1, ">30": 0, "NO": 0}
    )
    data.dropna(subset=["readmitted"], inplace=True)

    # ---------------------------------------------------
    # 3. Separate numeric and categorical
    # ---------------------------------------------------
    numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
    numeric_cols.remove("readmitted")

    categorical_cols = data.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    # ---------------------------------------------------
    # 4. Remove Near-Zero Variance
    # ---------------------------------------------------
    if numeric_cols:
        selector = VarianceThreshold(threshold=0.01)
        numeric_data = selector.fit_transform(data[numeric_cols])

        kept_numeric_cols = [
            col for col, keep in zip(numeric_cols, selector.get_support()) if keep
        ]

        numeric_df = pd.DataFrame(
            numeric_data,
            columns=kept_numeric_cols,
            index=data.index
        )
    else:
        numeric_df = pd.DataFrame(index=data.index)
        kept_numeric_cols = []

    # ---------------------------------------------------
    # 5. Remove Highly Correlated (> 0.90)
    # ---------------------------------------------------
    if len(kept_numeric_cols) > 1:
        corr_matrix = numeric_df.corr().abs()
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        to_drop = [
            column for column in upper.columns if any(upper[column] > 0.9)
        ]
        numeric_df.drop(columns=to_drop, inplace=True)

    # ---------------------------------------------------
    # 6. Box-Cox Transformation (positive numeric only)
    # ---------------------------------------------------
    positive_cols = [
        col for col in numeric_df.columns
        if (numeric_df[col] > 0).all()
    ]

    if positive_cols:
        pt = PowerTransformer(method="box-cox")
        numeric_df[positive_cols] = pt.fit_transform(
            numeric_df[positive_cols]
        )

    # ---------------------------------------------------
    # 7. One-Hot Encoding (low-cardinality only)
    # ---------------------------------------------------
    categorical_cols = [
        col for col in categorical_cols
        if data[col].nunique() < 20
    ]

    if categorical_cols:
        encoder = OneHotEncoder(
            drop="first",
            sparse_output=False,
            handle_unknown="ignore"
        )
        encoded = encoder.fit_transform(data[categorical_cols])

        encoded_df = pd.DataFrame(
            encoded,
            columns=encoder.get_feature_names_out(categorical_cols),
            index=data.index
        )
    else:
        encoded_df = pd.DataFrame(index=data.index)

    # ---------------------------------------------------
    # 8. Combine numeric + encoded
    # ---------------------------------------------------
    final_df = pd.concat([numeric_df, encoded_df], axis=1)
    final_df["readmitted"] = data["readmitted"].values

    # ---------------------------------------------------
    # 9. Scaling
    # ---------------------------------------------------
    feature_cols = final_df.columns.drop("readmitted")

    scaler = StandardScaler()
    final_df[feature_cols] = scaler.fit_transform(
        final_df[feature_cols]
    )

    # ---------------------------------------------------
    # 10. Spatial Sign Transformation
    # ---------------------------------------------------
    normalizer = Normalizer(norm="l2")
    final_df[feature_cols] = normalizer.fit_transform(
        final_df[feature_cols]
    )

    final_df.to_csv(CLEAN_FILE, index=False)

    logger.info(f"Saved cleaned data: {CLEAN_FILE}")
    logger.info(f"Final Shape: {final_df.shape}")


if __name__ == "__main__":
    main()