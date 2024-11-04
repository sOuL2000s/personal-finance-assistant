import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import datetime

# Sample expense data
data = {
    "Date": ["2024-10-01", "2024-10-05", "2024-10-12", "2024-10-15", "2024-10-20"],
    "Category": ["Groceries", "Utilities", "Transport", "Entertainment", "Groceries"],
    "Amount": [50, 80, 20, 40, 60]
}

# Convert to DataFrame and format date
df = pd.DataFrame(data)
df["Date"] = pd.to_datetime(df["Date"])

# Categorize expenses and visualize spending by category
category_summary = df.groupby("Category")["Amount"].sum()
category_summary.plot(kind="bar", title="Spending by Category")
plt.xlabel("Category")
plt.ylabel("Amount Spent ($)")
plt.show()

# Prepare data for forecasting (amounts over time)
df = df.set_index("Date").resample("W")["Amount"].sum().reset_index()

# Use a simple linear regression model to predict future expenses
model = LinearRegression()
df["Week"] = range(len(df))  # Assign weeks for regression
X = df[["Week"]]
y = df["Amount"]
model.fit(X, y)

# Predict next 4 weeks' expenses
future_weeks = pd.DataFrame({"Week": [len(df) + i for i in range(1, 5)]})
predicted_expenses = model.predict(future_weeks)

# Display predictions
for i, expense in enumerate(predicted_expenses, 1):
    print(f"Predicted expense for week {i + len(df)}: ${expense:.2f}")

# Summary report
print("\nBudget Report:")
print(df[["Date", "Amount"]].to_string(index=False))
