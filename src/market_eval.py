
import pandas_datareader.data as web
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

def plot_lppls_scenarios(
    model,
    predict_linear_params,
    lppl_test,
    tickers,
    scenarios,
    device=None,
    nrows=None,
    ncols=None,
    figsize=(12, 8),
    N_SELECT=252,
    show=True
):
    """
    Plot LPPLS fits for each (scenario, ticker) pair over given date ranges.

    Args:
        model: trained PyTorch model
        predict_linear_params: function
        lppl_test: function
        tickers: dict, e.g. {'NASDAQ-100': '^NDX', ...}
        scenarios: dict, e.g. {'Dot-com Crash': (start_date, end_date), ...}
        device: torch.device or None
        nrows, ncols: subplot grid shape (auto if None)
        figsize: tuple
        N_SELECT: number of samples for prediction
        show: If True, plt.show()

    Returns:
        axs: matplotlib axes array
        result_dict: {(scenario, ticker): dict with prediction details}
    """
    def get_full_period_data(ticker, start, end):
        data = web.DataReader(ticker, 'stooq', start, end)
        data = data.reset_index()[['Date', 'Close']]
        data = data.sort_values('Date')
        price = np.log(data['Close']).values.reshape(-1)
        time = np.array(data['Date'].apply(pd.Timestamp.toordinal).values)
        normalized_time = (time - time.min()) / (time.max() - time.min())
        normalized_price = (price - price.min()) / (price.max() - price.min())
        peak_idx = np.argmax(price)
        return normalized_time, normalized_price, price, data['Date'].values, peak_idx

    def lppls_predict_and_fit(model, device, DataSeries, N_SELECT=252):
        model.eval()
        model = model.to(device)
        with torch.no_grad():
            indices = np.linspace(0, len(DataSeries[1]) - 1, min(N_SELECT, len(DataSeries[1])), dtype=int)
            data_tensor = torch.tensor(DataSeries[1][indices], dtype=torch.float32).to(device)
            data_tensor = data_tensor.unsqueeze(0)
            output = model(data_tensor)
            prediction = output.squeeze(0).cpu().tolist()
            pred_tc, pred_m, pred_omega = prediction
            original_length = len(DataSeries[1])
            rescaled_tc_index = int((pred_tc * (original_length - 1)) / (len(indices) - 1))
            # Predict linear params
            a, b, c1, c2 = predict_linear_params(rescaled_tc_index, pred_m, pred_omega, DataSeries[1])
            # Reconstruct LPPLS fit
            t_full = np.linspace(0, original_length-1, original_length)
            lppls_curve = lppl_test(t_full, rescaled_tc_index, pred_m, pred_omega, a, b, c1, c2)
            lppls_curve_scaled = (lppls_curve - lppls_curve.min()) / (lppls_curve.max() - lppls_curve.min() + 1e-12)
            return lppls_curve_scaled, rescaled_tc_index, pred_m, pred_omega, a, b, c1, c2

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    if nrows is None or ncols is None:
        nrows = len(scenarios)
        ncols = len(tickers)

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharex=False, sharey=False)
    plt.subplots_adjust(hspace=0.3, wspace=0.2)

    # If single row/col, make axs always 2D for consistency
    if nrows == 1 and ncols == 1:
        axs = np.array([[axs]])
    elif nrows == 1:
        axs = axs.reshape(1, -1)
    elif ncols == 1:
        axs = axs.reshape(-1, 1)

    result_dict = {}
    for row, (scenario, (start, end)) in enumerate(scenarios.items()):
        for col, (name, ticker) in enumerate(tickers.items()):
            ax = axs[row, col]
            norm_time, norm_price, raw_price, dates, peak_idx = get_full_period_data(ticker, start, end)
            DataSeries = np.array([norm_time, norm_price])
            result = lppls_predict_and_fit(model, device, DataSeries, N_SELECT=N_SELECT)
            lppls_curve, rescaled_tc_index, pred_m, pred_omega, a, b, c1, c2 = result
            t_full = np.linspace(0, len(DataSeries[1])-1, len(DataSeries[1]))
            ax.plot(t_full, DataSeries[1], label='Log Price', lw=2)
            ax.plot(t_full, lppls_curve, label='LPPLS Fit', linestyle='--', lw=2)
            ax.axvline(x=rescaled_tc_index, color='black', linestyle=':', lw=2, label=f'Pred $t_c$ ({rescaled_tc_index})')
            # Optionally: ax.axvline(x=peak_idx, color='red', linestyle=':', lw=2, label=f'Peak ({peak_idx})')
            start_year = pd.Timestamp(dates[0]).year
            end_year = pd.Timestamp(dates[-1]).year
            ax.set_title(f"{scenario} ({start_year}-{end_year}): {name}")
            ax.set_xlabel("Time Index")
            ax.set_ylabel("Normalized log(Price)")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            # Store results
            result_dict[(scenario, name)] = dict(
                rescaled_tc_index=rescaled_tc_index,
                pred_m=pred_m,
                pred_omega=pred_omega,
                a=a, b=b, c1=c1, c2=c2,
                peak_idx=peak_idx,
                t_full=t_full,
                lppls_curve=lppls_curve,
                norm_price=DataSeries[1]
            )

    plt.tight_layout()
    if show:
        plt.show()
    return axs, result_dict
