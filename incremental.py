import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import numpy as np
import random
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

def incremental_updation(args, Mat_Int, Mat_Sim_DD, Mat_Sim_TT, Name_D, Name_T):
    print("\n===== Incremental Updation Test (One Drug) =====")

    # Load drug embeddings (with string IDs as index)
    emb_D_df = pd.read_csv(args.KGE_drug, sep=' ', header=None, index_col=0)
    emb_D = emb_D_df.values
    drug_ids = list(emb_D_df.index)

    # Load target embeddings
    emb_T_df = pd.read_csv(args.KGE_target, sep=' ', header=None, index_col=0)
    emb_T = emb_T_df.values
    target_ids = list(emb_T_df.index)

    Mat_Int = np.array(Mat_Int)
    Mat_Sim_DD = np.array(Mat_Sim_DD)
    Mat_Sim_TT = np.array(Mat_Sim_TT)

    # Randomly select 1 drug to remove
    removed_index = random.randint(0, len(drug_ids) - 1)
    removed_drug = drug_ids[removed_index]
    print(f"Holding out drug: {removed_drug}")

    # Get corresponding interaction vector
    interaction_vector = Mat_Int[removed_index]

    # Remove the drug from interaction matrix, similarity matrix, and embeddings
    X_train = np.delete(Mat_Int, removed_index, axis=0)
    Y_train = np.delete(emb_D, removed_index, axis=0)
    Sim_DD_new = np.delete(np.delete(Mat_Sim_DD, removed_index, axis=0), removed_index, axis=1)

    # Learn regression model: interaction_vector -> drug embedding
    regressor = make_pipeline(
        StandardScaler(),
        PCA(n_components=min(50, X_train.shape[1])),
        Ridge(alpha=1.0)
    )
    regressor.fit(X_train, Y_train)
    predicted_embedding = regressor.predict(interaction_vector.reshape(1, -1)).flatten()

    # Predict interaction scores with all targets using predicted embedding
    from pred_new_dti import Vector_D_fusion, Vector_T_fusion
    scores, labels = [], []
    for j, target_id in enumerate(Name_T.keys()):
        V_emb_target = emb_T[j]
        num_col = j
        num_row = len(Y_train)  # index of the new drug in updated embeddings

        D_fusion = Vector_D_fusion(
            num_row=num_row,
            num_col=num_col,
            V_emb_drug=predicted_embedding,
            v_emb_target=V_emb_target,
            Mat_Sim_DD=Sim_DD_new,
            Mat_Int=X_train,
            sort_concat=args.concat
        )

        T_fusion = Vector_T_fusion(
            num_row=num_row,
            num_col=num_col,
            V_emb_drug=predicted_embedding,
            v_emb_target=V_emb_target,
            Mat_Sim_TT=Mat_Sim_TT,
            Mat_Int=X_train,
            sort_concat=args.concat
        )

        x = np.concatenate((D_fusion, T_fusion)).reshape(1, -1)

        # Use a simple dummy model (you can replace with saved model later)
        from sklearn.ensemble import RandomForestClassifier
        dummy_model = RandomForestClassifier(n_estimators=10, random_state=42)
        dummy_X = np.random.rand(100, x.shape[1])
        dummy_y = np.random.randint(0, 2, 100)
        dummy_model.fit(dummy_X, dummy_y)

        prob = dummy_model.predict_proba(x)[0][1]
        scores.append(prob)
        labels.append(interaction_vector[j])

    auc = roc_auc_score(labels, scores)
    print(f"AUC for predicted embedding of drug {removed_drug}: {auc:.4f}")
    return auc

