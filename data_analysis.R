# Load the CSV file using the relative path
path = "/Users/johanbjerkem/Documents/code/oil_forecast"
setwd(path)
data <- read.csv("field_production_data/processed_data.csv")
colnames(data) <- c("field", "date", "production")
data <- subset(data, field == "Greater Ekofisk Area")
data$date <- as.Date(data$date, format = "%Y-%m-%d")

# Make column 'date' date object
ssb_data <- read.csv("ssb_turnover_data.csv", sep=";")

# Select the columns I can understand and want to work with
ssb_data <- ssb_data[, c(1, 2, 4)]

# Rename the columns
colnames(ssb_data) <- c("date", "turnover", "turnover_seasonal")

# Make column 'date' date object
ssb_data$date <- as.Date(paste0(sub("M", "-", ssb_data$date), "-01"), format = "%Y-%m-%d")

# Merge data
merged_data <- merge(data, ssb_data, by = "date", all.x = TRUE)
str(merged_data)

# Load the ggplot2 package if it's not already loaded
if (!require(ggplot2)) {
  install.packages("ggplot2")
  library(ggplot2)
}

# Create a ggplot
ggplot(merged_data, aes(x = date, y = `production`)) +
  geom_line() +
  labs(
    x = "date",
    y = "production",
    title = "Net Oil production Over Time"
  ) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  theme(plot.margin = margin(1, 1, 1, 1, "cm"))

# Will fill the production drops, because 2024 is not a year that should be affected, and I'm
# concerned the drops will affect the prediction too much, b
# Define the drop dates of production
drop_dates <- as.Date(c("2013-06-01", "2016-06-01", "2019-06-01", "2022-06-01"))

# Define the window size (3 months interval)
window_size <- 3

# Iterate through the drop dates
for (date in drop_dates) {
  # Find the index of the row that matches the drop date
  drop_index <- which(merged_data$date == date)
  
  # Check if the index exists
  if (length(drop_index) > 0) {
    # Calculate the start and end indexes for the window
    start_index <- max(1, drop_index - window_size)
    end_index <- min(nrow(merged_data), drop_index + window_size)
    
    # Calculate the mean production within the window
    mean_production <- mean(merged_data$`production`[start_index:end_index], na.rm = TRUE)
    
    # Replace the value at the drop index with the mean
    merged_data$`production`[drop_index] <- mean_production
  }
}

ggplot(merged_data, aes(x = date, y = `production`)) +
  geom_line() +
  labs(
    x = "date",
    y = "production",
    title = "Net Oil production Over Time"
  ) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  theme(plot.margin = margin(1, 1, 1, 1, "cm"))

# Set a random seed for reproducibility
set.seed(123)

# Needed to fix the comma, as it created errors
merged_data$turnover_seasonal <- as.numeric(gsub(",", "", merged_data$turnover_seasonal))

# Fit a linear model with production as the outcome and turnover, turnover_seasonal, and date as predictors
lm.fit <- lm(production ~ turnover + turnover_seasonal + date, data = merged_data)

# Summary of the linear model
summary(lm.fit)
# Production is highly dependent on date, and somewhat turnover_seasonal. As turnover and turnover_seasonal
# are correlated i drop turnover

lm.fit2 <- lm(production ~ turnover_seasonal + date, data = merged_data)

summary(lm.fit2)
# Now all predictors seem highly correlated

# Load the forecast package (if not already loaded)
if (!require(forecast)) {
  install.packages("forecast")
  library(forecast)
}

# Create time series with frequency 36 due to three years between production drops
ts_data <- ts(merged_data$production, frequency = 36)  

# Fit an ARIMA model
arima_model <- auto.arima(ts_data)

# Print the model summary
summary(arima_model)

# Make forecasts
forecast_result <- forecast(arima_model, h = 17)  # Forecasting 5 months ahead

# Plot the forecast
plot(forecast_result)

# Define the starting month (August) and year (2023)
starting_month <- 8
year <- 2023

# Create a sequence of date objects with the 1st of each forecasted month
forecast_dates <- seq(as.Date(paste(year, starting_month, "01", sep = "-")), by = "1 month", length.out = 5)


data <- read.csv("field_production_data/processed_data.csv")
colnames(data) <- c("field", "date", "production")
data$date <- as.Date(data$date, format = "%Y-%m-%d")

data <- merge(data, ssb_data, by = "date", all.x = TRUE)

area_list <- list("ELDFISK", "EKOFISK", "EMBLA", "Greater Ekofisk Area")

# Create an empty list to store data frames for each area
area_forecast_list <- list()

# Loop through each area
for (area in area_list) {
  # Subset the data for the current area
  area_data <- subset(data, field == area)
  
  # Create a time series
  ts_data <- ts(area_data$production, frequency = 36)
  
  # Fit an ARIMA model
  arima_model <- auto.arima(ts_data)
  
  # Make forecasts
  forecast_result <- forecast(arima_model, h = 17)
  
  # Access the forecasted values
  forecast_values <- forecast_result$mean
  
  # Define the starting month and year
  starting_month <- 8
  year <- 2023
  
  # Create a sequence of date objects with the 1st of each forecasted month
  forecast_dates <- seq(
    as.Date(paste(year, starting_month, "01", sep = "-")),
    by = "1 month",
    length.out = 17
  )

  # Create a data frame for the current area's forecasts
  area_forecast_df <- data.frame(
    field = rep(area, 17),
    date = forecast_dates,
    production = forecast_values
  )

  # Append the current area's forecasts to the list
  area_forecast_list[[area]] <- area_forecast_df
}

# Initialize an empty list to store data frames
combined_df_list <- list()

# Loop through each element in area_forecast_list and convert to a data frame
for (area_name in names(area_forecast_list)) {
  area_df <- area_forecast_list[[area_name]]
  
  # Convert the production column from time series to numeric
  area_df$production <- as.numeric(area_df$production)
  
  # Add the data frame to the list
  combined_df_list[[area_name]] <- area_df
}

# Combine the data frames into a single data frame
combined_df <- do.call(rbind, combined_df_list)

# Specify the file name for the output CSV file
output_file <- "field_production_data/analyzed_data.csv"

# Create the full relative path to the output file
output_path <- file.path(path, output_file)

# Write the combined data frame to a CSV file using the relative path
write.csv(combined_df, file = output_path, row.names = FALSE)

# Print a message to confirm the file has been saved
cat("Data has been saved to:", output_path, "\n")






