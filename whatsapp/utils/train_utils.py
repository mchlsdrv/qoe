import os
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.utils.data
from tqdm import tqdm

from configs.params import VAL_PROP, DEVICE, LR_REDUCTION_FREQ, LR_REDUCTION_FCTR, DROPOUT_START, DROPOUT_P
from models import TransformerForRegression
from regression_utils import calc_errors
from utils.aux_funcs import run_pca, get_number_of_parameters, unstandardize_results, get_errors, plot_losses
from utils.data_utils import get_train_val_split, QoEDataset, calc_data_reduction, normalize_columns, get_data


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


def train_model(model, epochs, train_data_loader, validation_data_loader, loss_function, optimizer, learning_rate, save_dir, tokenize: bool =False):
    # - Train
    # - Create the train directory
    train_save_dir = save_dir / f'train'
    os.makedirs(train_save_dir, exist_ok=True)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    # - Train the model
    train_losses, val_losses = run_train(
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
    np.save(train_save_dir / 'train_losses.npy', train_losses)
    np.save(train_save_dir / 'val_losses.npy', val_losses)

    # - Plot the train / val losses
    plt.plot(train_losses, label='train')
    plt.plot(val_losses, label='val')
    plt.suptitle('Train / Validation Loss Plot')
    plt.legend()
    plt.savefig(train_save_dir / 'train_val_loss.png')
    plt.close()


def get_input_data(data, tokenize: bool, device: torch.cuda.device):
    if tokenize:
        X, att_msk, Y = data
        X = X.to(device)
        Y = Y.to(device)
        att_msk = att_msk.to(device)

        input_data = [X, att_msk]
    else:
        X, Y = data
        X = X.to(device)
        Y = Y.to(device)

        input_data = [X]

    return input_data, Y


def run_train(
        model: torch.nn.Module, epochs: int,
        train_data_loader: torch.utils.data.DataLoader, val_data_loader: torch.utils.data.DataLoader,
        loss_function: torch.nn, optimizer: torch.optim,
        lr_reduce_frequency: int, lr_reduce_factor: float = 1.0,
        dropout_epoch_start: int = 20, dropout_epoch_delta: int = 20, p_dropout_init: float = 0.1, p_dropout_max: float = 0.5,
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
    assert isinstance(save_dir, str) or isinstance(save_dir, pathlib.Path), f'save_dir must be of type str or pathlib.Path, but is of type {type(save_dir)}!'
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
        print(f'Epoch: {epch}/{epochs} ({100 * epch / epochs:.2f}% done)')
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


def run_test(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, device: torch.device = torch.device('cpu')):
    test_results = pd.DataFrame()
    model.eval()
    with torch.no_grad():
        for (X, Y) in data_loader:
            X = X.to(device)
            Y = Y.to(device)
            btch_preds = model(X)

            for y, pred in zip(Y, btch_preds):
                d = dict()

                # - Add labels
                for i, y_val in enumerate(y):
                    d[f'true_{i}'] = np.float32(y_val.cpu().numpy())

                # - Add preds
                for i, pred_val in enumerate(pred):
                    d[f'pred_{i}'] = np.float32(pred_val.detach().cpu().numpy())

                # - Create the cv_5_folds frame
                btch_results = pd.DataFrame(d, index=pd.Index([0]))

                # - Add the batch cv_5_folds frame to the total results
                test_results = pd.concat([test_results, btch_results])

    test_results = test_results.reset_index(drop=True)

    return test_results


def search_parameters(
        model, test_data_root: pathlib.Path, data_dirs: list, features: list, labels: list, batch_size_numbers: list,
        epoch_numbers: list, layer_numbers: list, unit_numbers: list, loss_functions: list, optimizers: list,
        initial_learning_rates: list, save_dir: pathlib.Path
):
    # - Will hold the final results
    ablation_results = pd.DataFrame()

    # - Will hold the metadata for all experiments
    test_metadata = pd.DataFrame()
    test_dir_names = os.listdir(test_data_root)
    test_dir_names = [test_dir_name for test_dir_name in test_dir_names if test_dir_name in data_dirs]

    # - Total experiments
    cv_folds = len(test_dir_names)
    n_batch_sizes = len(batch_size_numbers)
    n_epoch_numbers = len(epoch_numbers)
    n_layer_numbers = len(layer_numbers)
    n_unit_numbers = len(unit_numbers)
    n_loss_funcs = len(loss_functions)
    n_optimizers = len(optimizers)
    n_init_lrs = len(initial_learning_rates)
    total_exp = cv_folds * n_batch_sizes * n_epoch_numbers * n_layer_numbers * n_unit_numbers * n_loss_funcs * n_optimizers * n_init_lrs
    print(f'''
    =================================================================================================================================================================================
    = ***************************************************************************************************************************************************************************** =
    =================================================================================================================================================================================
            > Total Number of experiments in ablation:
                - CV folds: {cv_folds}
                - Batch sizes: {n_batch_sizes}
                - Epoch numbers: {n_epoch_numbers}
                - Layer numbers: {n_layer_numbers}
                - Unit numbers: {n_unit_numbers}
                - Loss functions: {n_loss_funcs}
                - Optimizers: {n_optimizers}
                - Initial learning rates: {n_init_lrs}
                => N = {cv_folds} x {n_batch_sizes} x {n_epoch_numbers} x {n_layer_numbers} x {n_unit_numbers} x {n_loss_funcs} x {n_optimizers} x {n_init_lrs} = {total_exp}
    =================================================================================================================================================================================
    = ***************************************************************************************************************************************************************************** =
    =================================================================================================================================================================================
            ''')
    exp_idx = 0
    for fldr_idx, test_dir_name in enumerate(test_dir_names):

        # - Get the path to the current cv_5_folds folder
        data_folder = test_data_root / test_dir_name

        print(f'\t - Data folders: {data_folder} - {fldr_idx + 1}/{cv_folds} ({100 * (fldr_idx + 1) / cv_folds:.2f}% done)')

        # - Get the train / train_test cv_5_folds
        train_data_df = pd.read_csv(data_folder / 'train_data.csv')
        _, train_pca = run_pca(dataset_df=train_data_df.loc[:, features])
        test_data_df = pd.read_csv(data_folder / 'test_data.csv')

        for epch_idx, n_epochs in enumerate(epoch_numbers):
            # print('******************************************************************************************')
            # print(f'\t - Epochs: {n_epochs} - {epch_idx + 1}/{len(epoch_numbers)} ({100 * (epch_idx + 1) / len(epoch_numbers):.2f}% done)')
            # print('******************************************************************************************')
            for btch_idx, batch_size in enumerate(batch_size_numbers):
                # print('******************************************************************************************')
                # print(f'\t - Batch size: {batch_size} - {btch_idx + 1}/{len(batch_size_numbers)} ({100 * (btch_idx + 1) / len(batch_size_numbers):.2f}% done)')
                # print('******************************************************************************************')
                for lyr_idx, n_layers in enumerate(layer_numbers):
                    # print('******************************************************************************************')
                    # print(f'\t - Layers number: {n_layers} - {lyr_idx + 1}/{len(layer_numbers)} ({100 * (lyr_idx + 1) / len(layer_numbers):.2f}% done)')
                    # print('******************************************************************************************')
                    for units_idx, n_units in enumerate(unit_numbers):
                        # print('******************************************************************************************')
                        # print(f'\t - Units number: {n_units} - {units_idx + 1}/{len(unit_numbers)} ({100 * (units_idx + 1) / len(unit_numbers):.2f}% done)')
                        # print('******************************************************************************************')
                        for loss_func_idx, loss_func in enumerate(loss_functions):
                            loss_func_name = 'mse'
                            # print('******************************************************************************************')
                            # print(f'\t - Loss function: {loss_func} - {loss_func_idx + 1}/{len(loss_functions)} ({100 * (loss_func_idx + 1) / len(loss_functions):.2f}% done)')
                            # print('******************************************************************************************')
                            for opt_idx, opt in enumerate(optimizers):
                                opt_name = 'adam'
                                if opt == torch.optim.SGD:
                                    opt_name = 'sgd'
                                if opt == torch.optim.Adamax:
                                    opt_name = 'adamax'
                                # print('******************************************************************************************')
                                # print(f'\t - Optimizer: {opt} - {opt_idx + 1}/{len(optimizers)} ({100 * (opt_idx + 1) / len(optimizers):.2f}% done)')
                                # print('******************************************************************************************')
                                for init_lr_idx, init_lr in enumerate(initial_learning_rates):
                                    # print('******************************************************************************************')
                                    # print(f'\t - Initial learning rate: {init_lr} - {init_lr_idx + 1}/{len(initial_learning_rates)} ({100 * (init_lr_idx + 1) / len(initial_learning_rates):.2f}% done)')
                                    # print('******************************************************************************************')

                                    exp_idx += 1

                                    # - Split into train / val

                                    train_data_df.loc[:, labels] /= (train_data_df.loc[:, labels].max() - train_data_df.loc[:, labels].min())
                                    test_data_df.loc[:, labels] /= (test_data_df.loc[:, labels].max() - test_data_df.loc[:, labels].min())
                                    train_df, val_df = get_train_val_split(train_data_df, validation_proportion=VAL_PROP)

                                    # - Dataset
                                    train_ds = QoEDataset(
                                        data_df=train_df,
                                        feature_columns=features,
                                        label_columns=labels,
                                        pca=train_pca
                                    )
                                    val_ds = QoEDataset(
                                        data_df=val_df,
                                        feature_columns=features,
                                        label_columns=labels,
                                        pca=train_pca
                                    )

                                    # - Data Loader
                                    train_dl = torch.utils.data.DataLoader(
                                        train_ds,
                                        batch_size=batch_size,
                                        shuffle=True,
                                        pin_memory=True,
                                        num_workers=1,
                                        drop_last=True
                                    )

                                    val_batch_size = batch_size // 4
                                    val_dl = torch.utils.data.DataLoader(
                                        val_ds,
                                        batch_size=val_batch_size if val_batch_size > 0 else 1,
                                        shuffle=False,
                                        pin_memory=True,
                                        num_workers=1,
                                        drop_last=True
                                    )

                                    # - Build the model
                                    mdl = model(
                                        n_features=len(features),
                                        n_labels=len(labels),
                                        n_layers=n_layers,
                                        n_units=n_units
                                    )
                                    n_train_params, n_non_train_params = get_number_of_parameters(model=mdl)
                                    mdl.to(DEVICE)

                                    # - Train
                                    # - Create the train directory
                                    train_save_dir = save_dir / f'Data={test_dir_name}_Epochs={n_epochs}_Batch={batch_size}_Layers={n_layers}_Units={n_units}_Opt={opt_name}_lr={init_lr}'
                                    os.makedirs(train_save_dir, exist_ok=True)

                                    # - Train the model
                                    train_losses, val_losses = run_train(
                                        model=mdl,
                                        epochs=n_epochs,
                                        train_data_loader=train_dl,
                                        val_data_loader=val_dl,
                                        loss_function=loss_func(),
                                        optimizer=opt(mdl.parameters(), lr=init_lr),
                                        lr_reduce_frequency=LR_REDUCTION_FREQ,
                                        lr_reduce_factor=LR_REDUCTION_FCTR,
                                        dropout_epoch_start=DROPOUT_START,
                                        p_dropout_init=DROPOUT_P,
                                        device=DEVICE,
                                        save_dir=save_dir
                                    )

                                    # - Save the train / val loss metadata
                                    np.save(train_save_dir / 'train_losses.npy', train_losses)
                                    np.save(train_save_dir / 'val_losses.npy', val_losses)

                                    # - Plot the train / val losses
                                    plt.plot(train_losses, label='train')
                                    plt.plot(val_losses, label='val')
                                    plt.suptitle('Train / Validation Loss Plot')
                                    plt.legend()
                                    plt.savefig(train_save_dir / 'train_val_loss.png')
                                    plt.close()

                                    # - Get the train_test dataloader
                                    test_ds = QoEDataset(
                                        data_df=test_data_df,
                                        feature_columns=features,
                                        label_columns=labels,
                                        pca=train_pca
                                    )
                                    test_dl = torch.utils.data.DataLoader(
                                        test_ds,
                                        batch_size=16,
                                        shuffle=False,
                                        pin_memory=True,
                                        num_workers=1,
                                        drop_last=True
                                    )

                                    # - Test the model
                                    test_res = run_test(
                                        model=mdl,
                                        data_loader=test_dl,
                                        device=DEVICE
                                    )
                                    if test_ds.labels_std != 0:
                                        test_res = unstandardize_results(results=test_res, data_set=test_ds, n_columns=len(test_res.columns) // 2)

                                    # - Save the train_test metadata
                                    test_res.to_csv(train_save_dir / f'test_results.csv', index=False)

                                    # - Get the train_test errors
                                    test_errs = get_errors(results=test_res, columns=test_ds.label_columns)

                                    # - Save the train_test metadata
                                    test_errs.to_csv(train_save_dir / f'test_errors.csv', index=False)

                                    test_res = pd.concat([test_res, test_errs], axis=1)

                                    # configuration_test_metadata = pd.concat([test_metadata, test_res], axis=1).reset_index(drop=True)

                                    test_res.loc[:, 'dataset'] = test_dir_name
                                    test_res.loc[:, 'epochs'] = n_epochs
                                    test_res.loc[:, 'batch_size'] = batch_size
                                    test_res.loc[:, 'layers'] = n_layers
                                    test_res.loc[:, 'units'] = n_units
                                    test_res.loc[:, 'loss'] = loss_func_name
                                    test_res.loc[:, 'optimizer'] = opt_name
                                    test_res.loc[:, 'initial_lr'] = init_lr
                                    test_res.loc[:, 'n_params'] = n_train_params + n_non_train_params
                                    test_res.loc[:, 'n_train_params'] = n_train_params
                                    test_res.loc[:, 'n_non_train_params'] = n_non_train_params

                                    test_res.to_csv(train_save_dir / 'test_metadata.csv', index=False)

                                    # - Add the configuration train_test metadata to the global train_test metadata
                                    test_metadata = pd.concat([test_metadata, test_res]).reset_index(drop=True)

                                    # - Save the final metadata and the results
                                    test_metadata.to_csv(save_dir / 'test_metadata_tmp.csv', index=False)

                                    # - Get the results for the current configuration
                                    configuration_results = pd.DataFrame(
                                        dict(
                                            dataset=test_dir_name,
                                            epochs=n_epochs,
                                            batch_size=batch_size,
                                            layers=n_layers,
                                            units=n_units,
                                            loss=loss_func_name,
                                            optimizer=opt_name,
                                            initial_lr=init_lr,
                                            n_params=n_train_params + n_non_train_params,
                                            n_train_params=n_train_params,
                                            n_non_train_params=n_non_train_params
                                        ),
                                        index=pd.Index([0])
                                    )

                                    # -- Add the errors to the configuration results
                                    configuration_results = pd.concat([configuration_results, pd.DataFrame(test_errs.abs().mean()).T], axis=1)

                                    # - Add the results for the current configuration to the final parameter_selection results
                                    ablation_results = pd.concat([ablation_results, configuration_results], axis=0).reset_index(drop=True)

                                    # - Save the final metadata and the results
                                    ablation_results.to_csv(save_dir / 'ablation_final_results_tmp.csv', index=False)

                                    print(f'''
===========================================================
=================== Final Stats ===========================
===========================================================
Configuration:
    > {n_epochs} epochs
    > {batch_size} batch size
    > {n_layers} layers
    > {n_units} units
    > {n_epochs} epochs

Number of Parameters:
    > Trainable: {n_train_params}
    > Non-Trainable: {n_non_train_params}
    => Total: {n_train_params + n_non_train_params}

Mean Errors:
{test_errs.abs().mean()}

Status:
    - Experiment {exp_idx}/{total_exp} ({100 * exp_idx / total_exp:.2f}% done)
===========================================================
                                    ''')

    # - Save the final metadata and the results
    ablation_results.to_csv(save_dir / 'ablation_final_results.csv', index=False)

    # - Save the final metadata and the results
    test_metadata.to_csv(save_dir / 'test_metadata.csv', index=False)


def run_cv(model, model_params: dict, cv_root_dir: pathlib.Path, n_folds: int, features: list, labels: list, save_dir: pathlib.Path or None, nn_params: dict):
    tokenize = True if isinstance(model_params.get('model'), TransformerForRegression) else False

    train_data_reductions = np.array([])
    test_data_reductions = np.array([])
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    results = pd.DataFrame(columns=['true', 'predicted', 'error (%)'], dtype=np.float32)
    for idx, fold_dir in enumerate(os.listdir(cv_root_dir)):
        if fold_dir[0] != '.':
            cv_save_dir = save_dir / f'cv{idx+1}'
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
                # mdl = model(model_name='bert-base-uncased')

                train_model(
                    model=mdl,
                    epochs=nn_params.get('epochs'),
                    train_data_loader=train_data,
                    validation_data_loader=val_data,
                    loss_function=nn_params.get('loss_function'),
                    optimizer=nn_params.get('optimizer'),
                    learning_rate=nn_params.get('learning_rate'),
                    save_dir=cv_save_dir,
                    tokenize=tokenize
                )
                y_test, y_pred = predict(model=mdl, data_loader=test_data, device=device)
            else:
                # - For ML-based models
                X_train, y_train = train_data[0], train_data[1]
                model.fit(
                    X_train,
                    y_train
                )

                X_test, y_test = test_data[0], test_data[1]

                # - Predict the y_test labels based on X_test features
                y_test, y_pred = model.predict(X_test)

            errors = calc_errors(true=y_test, predicted=y_pred)
            results = pd.concat([
                results,
                pd.DataFrame(
                    {
                        'true': y_test.flatten(),
                        'predicted': y_pred.flatten(),
                        'error (%)': errors.flatten()
                    }
                )
            ], ignore_index=True)

    if isinstance(save_dir, pathlib.Path):
        results.to_csv(save_dir / 'final_results.csv')
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


def predict(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, device: torch.device = torch.device('cpu')):
    y_true = np.array([])
    y_pred = np.array([])
    model.eval()
    with torch.no_grad():
        for (X, y) in tqdm(data_loader):
            # - Move the data to device
            X = X.to(device)
            y = y.to(device)

            # - Calculate predictions
            preds = model(X)

            # - Append to the output arrays
            y_true = np.append(y_true, y.cpu().numpy())
            y_pred = np.append(y_pred, preds.cpu().numpy())

    return y_true, y_pred
