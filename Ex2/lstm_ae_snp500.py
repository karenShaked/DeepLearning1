from create_data import download_snp500
from graphs import plot_signal_vs_time

# Download data
snp500 = download_snp500()  # [num_of_companies, num_of_dates]

# 3.3.1  graphs of the daily max value for the stocks AMZN and GOOGL
AMZN = snp500[:, snp500[0] == 'AMZN']
plot_signal_vs_time(AMZN, "Amazon Stock Price vs. Time", time='Date', signal='Stock Price')
GOOGL = snp500[:, snp500[0] == 'GOOGL']
plot_signal_vs_time(GOOGL, "Google Stock Price vs. Time", time='Date', signal='Stock Price')

# 3.3.2

