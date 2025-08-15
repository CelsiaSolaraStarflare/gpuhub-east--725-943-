import pandas as pd
import numpy as np
import torch
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Import your cross-attention LSTM model
from model import LSTMWithCrossAttention

def convert_to_yyyy_mm(date_str):
    """Converts date strings from various formats to 'YYYY-MM'."""
    for fmt in ('%d-%b', '%Y-%m'):
        try:
            dt = datetime.strptime(str(date_str), fmt)
            if fmt == '%d-%b':
                # Heuristic for '8-Jan' -> 2008
                year = 2000 + dt.day if dt.day < 50 else 1900 + dt.day
                return dt.replace(year=year).strftime('%Y-%m')
            return dt.strftime('%Y-%m')
        except (ValueError, TypeError):
            continue
    return None


def calculate_metrics(y_true, y_pred, feature_names=None):
    """Calculate MSE, MAE, MAPE, and RMSE metrics per feature and overall."""
    if feature_names is None:
        feature_names = ['Value', 'Weight']  # Only use Value and Weight
    
    metrics = {}
    
    # Calculate metrics for each feature separately
    for f_idx, feature in enumerate(feature_names):
        y_true_feature = y_true[:, :, f_idx].flatten()
        y_pred_feature = y_pred[:, :, f_idx].flatten()
        
        # Calculate basic metrics
        mse = mean_squared_error(y_true_feature, y_pred_feature)
        mae = mean_absolute_error(y_true_feature, y_pred_feature)
        rmse = np.sqrt(mse)
        
        # Calculate MAPE with improved handling of zero/near-zero values
        # Only calculate MAPE for non-zero true values
        non_zero_mask = np.abs(y_true_feature) > 1e-6
        if np.any(non_zero_mask):
            mape = np.mean(np.abs((y_true_feature[non_zero_mask] - y_pred_feature[non_zero_mask]) / 
                                 y_true_feature[non_zero_mask])) * 100
        else:
            mape = np.inf  # If all true values are zero
        
        metrics[feature] = {
            'MSE': mse,
            'MAE': mae,
            'MAPE': mape,
            'RMSE': rmse
        }
    
    # Calculate overall metrics (all features combined)
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    overall_mse = mean_squared_error(y_true_flat, y_pred_flat)
    overall_mae = mean_absolute_error(y_true_flat, y_pred_flat)
    overall_rmse = np.sqrt(overall_mse)
    
    # Overall MAPE with better handling
    non_zero_mask = np.abs(y_true_flat) > 1e-6
    if np.any(non_zero_mask):
        overall_mape = np.mean(np.abs((y_true_flat[non_zero_mask] - y_pred_flat[non_zero_mask]) / 
                                     y_true_flat[non_zero_mask])) * 100
    else:
        overall_mape = np.inf
    
    metrics['Overall'] = {
        'MSE': overall_mse,
        'MAE': overall_mae,
        'MAPE': overall_mape,
        'RMSE': overall_rmse
    }
    
    return metrics


def load_and_preprocess(csv_path):
    """Loads, cleans, and preprocesses the commodity data from a CSV file."""
    df = pd.read_csv(csv_path, on_bad_lines='skip')
    df['Time'] = df['Time'].apply(convert_to_yyyy_mm).dropna()
    
    # Clean numeric columns
    for col in ['Containerized Vessel Total Exports Value ($US)', 'Containerized Vessel Total Exports SWT (kg)']:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
    df = df.rename(columns={
        'Containerized Vessel Total Exports Value ($US)': 'Value',
        'Containerized Vessel Total Exports SWT (kg)': 'Weight'
    })
    df[['Value', 'Weight']] = df[['Value', 'Weight']].fillna(0)

    # Filter top 10 commodities by total export value
    top_commodities = df.groupby('Commodity')['Value'].sum().nlargest(10).index
    df = df[df['Commodity'].isin(top_commodities)]
    
    # Aggregate by month
    monthly_df = df.groupby(['Time', 'Commodity']).agg({'Value': 'sum', 'Weight': 'sum'}).reset_index()
    
    monthly_df = monthly_df.sort_values(by='Time')
    
    # Create integer-based time sequence
    monthly_df['Time_dt'] = pd.to_datetime(monthly_df['Time'])
    start_date = monthly_df['Time_dt'].min()
    monthly_df['Time_seq'] = ((monthly_df['Time_dt'].dt.year - start_date.year) * 12 +
                              (monthly_df['Time_dt'].dt.month - start_date.month))
    
    return monthly_df


def dataframe_to_multigroup_data(df, feature_cols=None):
    """Pivots the DataFrame to a 3D NumPy array (time, group, feature)."""
    if feature_cols is None:
        feature_cols = ['Value', 'Weight']  # Only use Value and Weight
    
    pivot_df = df.pivot_table(index='Time_seq', columns='Commodity', values=feature_cols).fillna(0)
    
    # Ensure consistent ordering
    times = sorted(pivot_df.index)
    groups = sorted(df['Commodity'].unique())
    
    # Reorder columns to be (feature, group) then reshape
    pivot_df = pivot_df.reindex(columns=pd.MultiIndex.from_product([feature_cols, groups]))
    
    data = np.zeros((len(times), len(groups), len(feature_cols)))
    for i, feature in enumerate(feature_cols):
        data[:, :, i] = pivot_df[feature].values
        
    return np.array(times), data, groups, feature_cols


def get_scalers(data, num_features):
    """Creates and fits a StandardScaler for each feature."""
    scalers = [StandardScaler() for _ in range(num_features)]
    scaler_stats = {}
    
    # Fit each scaler on its corresponding feature data across all time and groups
    for i in range(num_features):
        feature_data = data[:, :, i].reshape(-1, 1)
        scalers[i].fit(feature_data)
        
        # Save scaler statistics for reference
        scaler_stats[f'feature_{i}'] = {
            'mean': scalers[i].mean_[0],
            'std': scalers[i].scale_[0],
            'min': feature_data.min(),
            'max': feature_data.max()
        }
    
    return scalers, scaler_stats


def save_scalers(scalers, scaler_stats, feature_names, save_path):
    """Save scaler objects and statistics."""
    import pickle
    
    scaler_data = {
        'scalers': scalers,
        'stats': scaler_stats,
        'feature_names': feature_names
    }
    
    with open(save_path, 'wb') as f:
        pickle.dump(scaler_data, f)
    
    print(f"Scalers saved to {save_path}")
    print("\nNormalization Statistics:")
    print("="*50)
    for i, feature in enumerate(feature_names):
        stats = scaler_stats[f'feature_{i}']
        print(f"{feature}:")
        print(f"  Original range: [{stats['min']:.2e}, {stats['max']:.2e}]")
        print(f"  Mean: {stats['mean']:.2e}")
        print(f"  Std:  {stats['std']:.2e}")
        print(f"  After normalization: mean=0, std=1")
    print("="*50)


def scale_data(data, scalers):
    """Scales 3D or 4D data using the provided scalers with high precision."""
    # Ensure we use float64 for maximum precision
    scaled_data = np.copy(data).astype(np.float64)
    original_shape = data.shape

    # Temporarily reshape to (samples, features)
    reshaped_data = scaled_data.reshape(-1, original_shape[-1])
    
    # Apply scaling to each feature column
    for i, scaler in enumerate(scalers):
        feature_col = reshaped_data[:, i].reshape(-1, 1).astype(np.float64)
        reshaped_data[:, i] = scaler.transform(feature_col).flatten()
        
    # Reshape back to the original shape
    return reshaped_data.reshape(original_shape)


def inverse_scale_data(data, scalers):
    """Applies inverse scaling to 3D or 4D data with high precision."""
    # Ensure we use float64 for maximum precision
    inv_scaled_data = np.copy(data).astype(np.float64)
    original_shape = data.shape

    # Temporarily reshape to (samples, features)
    reshaped_data = inv_scaled_data.reshape(-1, original_shape[-1])

    # Apply inverse scaling to each feature column
    for i, scaler in enumerate(scalers):
        feature_col = reshaped_data[:, i].reshape(-1, 1).astype(np.float64)
        reshaped_data[:, i] = scaler.inverse_transform(feature_col).flatten()

    # Reshape back to the original shape
    return reshaped_data.reshape(original_shape)


class MultiGroupSequenceDataset(Dataset):
    """PyTorch Dataset for multi-group time series."""
    def __init__(self, data, input_window, output_window):
        self.data = data
        self.input_window = input_window
        self.output_window = output_window
        self.features, self.targets = self._prepare()

    def _prepare(self):
        X, Y = [], []
        total_len = self.input_window + self.output_window
        for i in range(len(self.data) - total_len + 1):
            window = self.data[i : i + total_len]
            X.append(window[:self.input_window])
            Y.append(window[-self.output_window:])
        
        X = torch.tensor(np.array(X), dtype=torch.float32)
        Y = torch.tensor(np.array(Y), dtype=torch.float32)
        # Flatten target for the model's linear output layer
        Y = Y.view(Y.shape[0], -1)
        return X, Y

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


def train_model(model, loader, criterion, optimizer, device, epochs):
    """Standard model training loop."""
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs} — Train Loss: {avg_loss:.6f}")


def evaluate_model(model, loader, criterion, device):
    """Standard model evaluation loop."""
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            total_loss += criterion(pred, y).item()
    avg_loss = total_loss / len(loader)
    print(f"Test Loss: {avg_loss:.6f}")


def predict_future(model, data, scalers, input_window, output_window, device):
    """Makes a future prediction and returns it along with the ground truth."""
    model.eval()
    
    # Use the last part of the data for validation/prediction
    input_data_unscaled = data[-(input_window + output_window):-output_window]
    true_future_unscaled = data[-output_window:]

    # Scale the input window
    input_data_scaled = scale_data(input_data_unscaled[np.newaxis, ...], scalers)[0]
    
    # Create tensor and add batch dimension
    input_tensor = torch.tensor(input_data_scaled[None, ...], dtype=torch.float32).to(device)
    
    with torch.no_grad():
        pred_scaled = model(input_tensor).cpu().numpy()
        
    # Reshape and inverse-transform the prediction
    num_g, num_f = data.shape[1], data.shape[2]
    pred_reshaped = pred_scaled.reshape(output_window, num_g, num_f)
    pred_unscaled = inverse_scale_data(pred_reshaped, scalers)
    
    return true_future_unscaled, pred_unscaled


def plot_predictions(t, data, true_future, pred, groups, feature_cols, input_window, out_dir):
    """Plots historical data, ground truth, and future predictions."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    
    output_window = len(true_future)
    history_t = t[-(input_window + output_window):-output_window]
    future_t = t[-output_window:]

    for g_idx, group in enumerate(groups):
        for f_idx, feature in enumerate(feature_cols):
            plt.figure(figsize=(12, 6))
            
            # Plot history from the unscaled data
            history_data = data[-(input_window + output_window):-output_window, g_idx, f_idx]
            plt.plot(history_t, history_data, label='History')

            # Plot Ground Truth
            plt.plot(future_t, true_future[:, g_idx, f_idx], 'k-', label='Ground Truth')
            
            # Plot prediction
            plt.plot(future_t, pred[:, g_idx, f_idx], 'r--', label='Predicted Future')
            
            plt.title(f'Forecast for {group} - {feature}')
            plt.xlabel("Time Step (Months)")
            plt.ylabel("Value")
            plt.legend()
            plt.tight_layout()
            
            file_path = Path(out_dir) / f'{group.replace(" ", "_")}_{feature}.png'
            plt.savefig(file_path)
            print(f"Saved plot: {file_path}")
            plt.close()


def main():
    # --- Configuration ---
    csv_path = 'LSTM/csv'
    input_window = 36
    output_window = 12
    batch_size = 32
    epochs = 200
    lr = 0.0005
    plot_dir = 'results/real_data_plots'

    # --- Data Processing ---
    df = load_and_preprocess(csv_path)
    t, data, groups, feature_cols = dataframe_to_multigroup_data(df)
    
    # Ensure high precision for large values
    data = data.astype(np.float64)
    
    num_features = data.shape[2]
    
    print(f"\nData shape: {data.shape}")
    print(f"Data type: {data.dtype} (precision: {np.finfo(data.dtype).precision} digits)")
    print(f"Features: {feature_cols}")
    print(f"Groups (commodities): {len(groups)}")
    
    # Print original data statistics
    print(f"\nOriginal Data Statistics:")
    print("="*50)
    for i, feature in enumerate(feature_cols):
        feature_data = data[:, :, i]
        print(f"{feature}:")
        print(f"  Min:    {feature_data.min():.2e}")
        print(f"  Max:    {feature_data.max():.2e}")
        print(f"  Mean:   {feature_data.mean():.2e}")
        print(f"  Std:    {feature_data.std():.2e}")
        # Check for precision issues
        unique_vals = len(np.unique(feature_data[feature_data > 0]))
        total_vals = np.sum(feature_data > 0)
        print(f"  Unique non-zero values: {unique_vals}/{total_vals} ({100*unique_vals/max(total_vals,1):.1f}%)")
    print("="*50)
    
    scalers, scaler_stats = get_scalers(data, num_features)
    scaled_data = scale_data(data, scalers)
    
    # Save scalers for future use
    save_dir = Path(plot_dir).parent / 'models'
    save_dir.mkdir(parents=True, exist_ok=True)
    scaler_save_path = save_dir / 'scalers_attn.pkl'
    save_scalers(scalers, scaler_stats, feature_cols, scaler_save_path)
    
    # Verify scaling worked correctly
    print(f"\nScaled Data Verification:")
    print("="*50)
    for i, feature in enumerate(feature_cols):
        scaled_feature_data = scaled_data[:, :, i]
        print(f"{feature} (after scaling):")
        print(f"  Mean:   {scaled_feature_data.mean():.10f} (should be ~0)")
        print(f"  Std:    {scaled_feature_data.std():.10f} (should be ~1)")
        print(f"  Range:  [{scaled_feature_data.min():.6f}, {scaled_feature_data.max():.6f}]")
        # Check if we still have variation after scaling
        unique_scaled = len(np.unique(np.round(scaled_feature_data, 6)))
        total_scaled = scaled_feature_data.size
        print(f"  Unique scaled values: {unique_scaled}/{total_scaled} ({100*unique_scaled/total_scaled:.1f}%)")
    print("="*50)

    # --- Dataset and Loaders ---
    # Split data into training and validation sets
    split_idx = len(scaled_data) - output_window
    train_data = scaled_data[:split_idx]
    val_data = scaled_data[split_idx:]  # Keep validation data for evaluation
    
    train_dataset = MultiGroupSequenceDataset(train_data, input_window, output_window)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Create validation dataset for evaluation
    val_dataset = MultiGroupSequenceDataset(scaled_data[:-output_window], input_window, output_window)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # --- Model Initialization ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_groups = data.shape[1]
    model = LSTMWithCrossAttention(
        num_groups=num_groups,
        features_per_group=num_features,
        hidden_size=128,
        output_dim=output_window * num_groups * num_features,
        layers=2,
        dropout=0.2
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    # --- Training ---
    print("Starting model training...")
    train_model(model, train_loader, criterion, optimizer, device, epochs)
    
    # --- Validation Loss ---
    print("\nEvaluating on validation set...")
    evaluate_model(model, val_loader, criterion, device)

    # --- Save the trained model ---
    save_dir = Path(plot_dir).parent / 'models'
    save_dir.mkdir(parents=True, exist_ok=True)
    model_save_path = save_dir / 'lstm_attn.pth'
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    # --- Prediction and Plotting ---
    print("\nGenerating future predictions...")
    true_future, prediction = predict_future(model, data, scalers, input_window, output_window, device)
    
    # --- DETAILED OUTPUT: Show actual values ---
    print(f"\nDETAILED PREDICTION RESULTS:")
    print(f"="*80)
    print(f"Data shapes:")
    print(f"  true_future: {true_future.shape}")
    print(f"  prediction: {prediction.shape}")
    print(f"  Groups (commodities): {groups}")
    print(f"  Features: {feature_cols}")
    
    # Show actual values for first timestep (t+1) across all commodities and features
    print(f"\nACTUAL VALUES vs PREDICTIONS (t+1 only):")
    print(f"="*80)
    
    for g_idx, group in enumerate(groups):
        print(f"\n--- {group} ---")
        for f_idx, feature in enumerate(feature_cols):
            true_val = true_future[0, g_idx, f_idx]
            pred_val = prediction[0, g_idx, f_idx]
            error = abs(true_val - pred_val)
            pct_error = (error / abs(true_val)) * 100 if abs(true_val) > 1e-6 else float('inf')
            
            print(f"  {feature}:")
            print(f"    True:  {true_val:12.2f}")
            print(f"    Pred:  {pred_val:12.2f}")
            print(f"    Error: {error:12.2f} ({pct_error:.1f}%)")
    
    # --- Calculate and Display Metrics for Next Point Only (t+1) ---
    # Extract only the first time step (next point) from the predictions
    true_next_point = true_future[0:1]  # Shape: (1, num_groups, num_features)
    pred_next_point = prediction[0:1]   # Shape: (1, num_groups, num_features)
    
    metrics = calculate_metrics(true_next_point, pred_next_point, feature_cols)
    print(f"\n" + "="*60)
    print("PREDICTION METRICS (Next Point Only - t+1)")
    print("="*60)
    
    # Display metrics for each feature
    for feature in feature_cols:
        print(f"\n{feature}:")
        print(f"  MSE:  {metrics[feature]['MSE']:.2e}")
        print(f"  MAE:  {metrics[feature]['MAE']:.2e}")
        print(f"  MAPE: {metrics[feature]['MAPE']:.2f}%" if metrics[feature]['MAPE'] != np.inf else "  MAPE: N/A (zero values)")
        print(f"  RMSE: {metrics[feature]['RMSE']:.2e}")
    
    # Display overall metrics
    print(f"\nOverall (All Features Combined):")
    print(f"  MSE:  {metrics['Overall']['MSE']:.2e}")
    print(f"  MAE:  {metrics['Overall']['MAE']:.2e}")
    print(f"  MAPE: {metrics['Overall']['MAPE']:.2f}%" if metrics['Overall']['MAPE'] != np.inf else "  MAPE: N/A (zero values)")
    print(f"  RMSE: {metrics['Overall']['RMSE']:.2e}")
    print("="*60)
    
    plot_predictions(t, data, true_future, prediction, groups, feature_cols, input_window, plot_dir)
    print("\nScript finished.")

if __name__ == '__main__':
    main()
