from imblearn.over_sampling import SMOTE


def apply_smote(X, y):

    smote = SMOTE(random_state=42)

    X_resampled, y_resampled = smote.fit_resample(X, y)

    print("SMOTE balanced shape:", X_resampled.shape)

    return X_resampled, y_resampled