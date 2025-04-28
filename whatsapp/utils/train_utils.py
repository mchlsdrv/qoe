import os
import pathlib
import numpy as np
import pandas as pd
import torch
import torch.utils.data
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm

from configs.params import LR_REDUCTION_FREQ, LR_REDUCTION_FCTR, DROPOUT_START, DROPOUT_P
from regression_utils import calc_errors
from utils.aux_funcs import plot_losses
from utils.data_utils import calc_data_reduction, normalize_columns, get_data, get_input_data


def save_checkpoint(model, optimizer, filename: str or pathlib.Path):
    filename = pathlib.Path(filename)
    print(f'=> Saving checkpoint to \'{filename}\' ...')
    state = dict(
        state_dict=model.state_dict(),
        optimizer=optimizer.state_dict(),
    )
    os.makedirs(filename.parent, exist_ok=True)
    torch.save(state, filename)


def load_checkpoint(model, checkpoint_file: str or pathlib.Path):
    print('=> Loading checkpoint ...')
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint['state_dict'])


def get_train_val_losses(
        model: torch.nn.Module, epochs: int,
        train_data_loader: torch.utils.data.DataLoader, val_data_loader: torch.utils.data.DataLoader,
        loss_function: torch.nn, optimizer: torch.optim,
        lr_reduce_frequency: int, lr_reduce_factor: float = 1.0,
        dropout_epoch_start: int = 20, dropout_epoch_delta: int = 20, p_dropout_init: float = 0.1,
        p_dropout_max: float = 0.5,
        checkpoint_file: pathlib.Path = None,
        checkpoint_save_frequency: int = 10,
        device: torch.device = torch.device('cpu'),
        tokenize: bool = False,
        save_dir: str or pathlib.Path = pathlib.Path('./outputs')
):
    # - Load the checkpoint to continue training
    if isinstance(checkpoint_file, pathlib.Path):
        load_checkpoint(model=model, checkpoint_file=checkpoint_file)

    # - Make sure the save_dir exists and of type pathlib.Path
    assert isinstance(save_dir, str) or isinstance(save_dir,
                                                   pathlib.Path), f'save_dir must be of type str or pathlib.Path, but is of type {type(save_dir)}!'
    os.makedirs(save_dir, exist_ok=True)
    save_dir = pathlib.Path(save_dir)

    # - Initialize the p_drop to 0.0
    p_drop = 0.0

    # - Initialize the train / val loss arrays
    train_losses = np.array([])
    val_losses = np.array([])

    for epch in range(epochs):
        model.train(True)
        if epch > dropout_epoch_start and p_drop < p_dropout_max:
            p_drop = p_dropout_init * ((epch - dropout_epoch_start + dropout_epoch_delta) // dropout_epoch_delta)
        print(f'\nEpoch: {epch}/{epochs} ({100 * epch / epochs:.2f}% done)')
        print(f'\t ** INFO ** p_drop = {p_drop:.4f}')
        btch_train_losses = np.array([])
        btch_pbar = tqdm(train_data_loader)
        for data in btch_pbar:

            input_data, Y = get_input_data(data=data, tokenize=tokenize, device=device)

            results = model(*input_data)
            loss = loss_function(results, Y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            btch_train_losses = np.append(btch_train_losses, loss.item())

        # - Reduce learning rate every
        if epch > 0 and epch % lr_reduce_frequency == 0:
            old_lr = optimizer.param_groups[0]['lr']
            new_lr = old_lr * lr_reduce_factor
            optimizer.param_groups[0]['lr'] = new_lr
            print(f'\t ** INFO ** The learning rate was changed from {old_lr} -> {new_lr}')

        train_losses = np.append(train_losses, btch_train_losses.mean())

        btch_val_losses = np.array([])
        model.eval()
        with torch.no_grad():
            for data in val_data_loader:

                input_data, Y = get_input_data(data=data, tokenize=tokenize, device=device)

                results = model(*input_data)
                loss = loss_function(results, Y)
                btch_val_losses = np.append(btch_val_losses, loss.item())

        val_losses = np.append(val_losses, btch_val_losses.mean())

        plot_losses(train_losses=train_losses, val_losses=val_losses, save_dir=save_dir)

        # - Save the checkpoint
        if epch > 0 and epch % checkpoint_save_frequency == 0:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                filename=save_dir / f'checkpoints/epoch_{epch}_checkpoint.ckpt'
            )

        print(f'''
==========================================
================= STATS ==================
==========================================
    Epoch {epch + 1} loss:
        > train = {train_losses.mean():.4f}
        > validation = {val_losses.mean():.4f}
==========================================
==========================================
        ''')

    return train_losses, val_losses


def train_model(model, epochs, train_data_loader, validation_data_loader, loss_function, optimizer, learning_rate, save_dir, tokenize: bool =False):
    # - Train
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    # - Train the model
    train_losses, val_losses = get_train_val_losses(
        model=model,
        epochs=epochs,
        train_data_loader=train_data_loader,
        val_data_loader=validation_data_loader,
        loss_function=loss_function(),
        optimizer=optimizer(model.parameters(), lr=learning_rate),
        lr_reduce_frequency=LR_REDUCTION_FREQ,
        lr_reduce_factor=LR_REDUCTION_FCTR,
        dropout_epoch_start=DROPOUT_START,
        p_dropout_init=DROPOUT_P,
        tokenize=tokenize,
        device=device,
        save_dir=save_dir
    )

    # - Save the train / val loss metadata
    np.save(save_dir / 'train_losses.npy', train_losses)
    np.save(save_dir / 'val_losses.npy', val_losses)


def test_model(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, tokenize: bool = False, device: torch.device = torch.device('cpu')):
    y_true = np.array([])
    y_pred = np.array([])
    model.eval()
    with torch.no_grad():
        for data in tqdm(data_loader):
            input_data, Y = get_input_data(data=data, tokenize=tokenize, device=device)

            # - Calculate predictions
            preds = model(*input_data)

            # - Append to the output arrays
            y_true = np.append(y_true, Y.cpu().numpy())
            y_pred = np.append(y_pred, preds.cpu().numpy())

    return y_true, y_pred


def run_cv(model, model_params: dict, cv_root_dir: pathlib.Path or str, n_folds: int, features: list, labels: list, save_dir: pathlib.Path or None, nn_params: dict):
    tokenize = True if model_params.get('model_name') == 'bert-base-uncased' else False

    train_data_reductions = np.array([])
    test_data_reductions = np.array([])
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    results = pd.DataFrame(columns=['true', 'predicted', 'error (%)'], dtype=np.float32)
    for fld_idx, fold_dir in enumerate(os.listdir(cv_root_dir)):
        if fold_dir[0] != '.':
            cv_save_dir = save_dir / f'cv{fld_idx + 1}'

            cv_train_dir = cv_save_dir / 'train'
            os.makedirs(cv_train_dir)

            cv_test_dir = cv_save_dir / 'test'
            os.makedirs(cv_test_dir)

            feat_lbls_names = [*features, *labels]

            # - Train data
            train_df = pd.read_csv(cv_root_dir / fold_dir / 'train_data.csv')
            train_df = train_df.loc[:, feat_lbls_names]
            train_data_len_orig = len(train_df)

            # -- Remove the test data rows where the label is 0
            train_df = train_df.loc[train_df.loc[:, *labels] > 0]
            train_data_len_reduced = len(train_df)
            train_rdct_pct = calc_data_reduction(original_size=train_data_len_orig, reduced_size=train_data_len_reduced)
            train_data_reductions = np.append(train_data_reductions, train_rdct_pct)

            train_df, _, _ = normalize_columns(data_df=train_df, columns=[*features])

            # - Test data
            test_df = pd.read_csv(cv_root_dir / fold_dir / 'test_data.csv')
            test_df = test_df.loc[:, feat_lbls_names]
            test_data_len_orig = len(test_df)

            # -- Remove the test data rows where the label is 0
            test_df = test_df.loc[test_df.loc[:, *labels] > 0]
            test_data_len_reduced = len(test_df)
            test_rdct_pct = calc_data_reduction(original_size=test_data_len_orig, reduced_size=test_data_len_reduced)
            test_data_reductions = np.append(test_data_reductions, test_rdct_pct)

            test_df, _, _ = normalize_columns(data_df=test_df, columns=[*features])

            knn_classifier = KNeighborsClassifier(n_neighbors=5)
            res = knn_classifier.fit(train_df.loc[:, features].values, train_df.loc[:, labels].values)

            train_data, val_data, test_data, test_ds = get_data(
                train_df=train_df,
                test_df=test_df,
                features=features,
                labels=labels,
                batch_size=nn_params.get('batch_size'),
                val_prop=nn_params.get('val_prop'),
                tokenize=tokenize
            )

            if isinstance(val_data, torch.utils.data.DataLoader):
                # - For NN-based models

                # - Build the model
                mdl = model(**model_params)

                train_model(
                    model=mdl,
                    epochs=nn_params.get('epochs'),
                    train_data_loader=train_data,
                    validation_data_loader=val_data,
                    loss_function=nn_params.get('loss_function'),
                    optimizer=nn_params.get('optimizer'),
                    learning_rate=nn_params.get('learning_rate'),
                    save_dir=cv_train_dir,
                    tokenize=tokenize
                )

                # - Test the model
                y_true, y_pred = test_model(
                    model=mdl,
                    data_loader=test_data,
                    tokenize=tokenize,
                    device=device
                )
            else:
                # - For ML-based models
                X_train, y_train = train_data[0], train_data[1]
                model.fit(
                    X_train,
                    y_train
                )

                X_test, y_true = test_data[0], test_data[1]

                # - Predict the y_test labels based on X_test features
                y_true, y_pred = model.predict(X_test)

            errs = calc_errors(true=y_true, predicted=y_pred)

            print(f'''
            Mean Test Errors - {fld_idx + 1} CV Fold:
                > Mean true values      : {y_true.mean():.2f}+/-{y_true.std():.3f}
                > Mean predicted values : {y_pred.mean():.2f}+/-{y_pred.std():.3f}
                > Mean errors           : {errs.mean():.2f}+/-{errs.std():.3f}
''')
            results = pd.concat([
                results,
                pd.DataFrame(
                    {
                        'true': y_true.flatten(),
                        'predicted': y_pred.flatten(),
                        'error (%)': errs.flatten()
                    }
                )
            ], ignore_index=True)

            results.to_csv(cv_test_dir / f'{fld_idx}_fold_final_results.csv')

    results.to_csv(cv_root_dir / f'{n_folds}_folds_cv_results.csv')

    mean_error = results.loc[:, "error (%)"].mean()
    std_error = results.loc[:, "error (%)"].std()
    print(f'''
Mean Stats on {n_folds} CV for {labels}:
    Mean Errors (%)
    ---------------
    {mean_error:.2f}+/-{std_error:.3f}

    Mean reduced data (%)
        - Train: {train_data_reductions.mean():.2f}+/-{train_data_reductions.std():.3f}
        - Test: {test_data_reductions.mean():.2f}+/-{test_data_reductions.std():.3f}
    ''')
    return results
