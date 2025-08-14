
# Vehicle Registration Data Dashboard

This project is a **Streamlit** dashboard for analyzing vehicle registration data (maker-wise, category-wise, and year-over-year comparisons).

---

## Setup Instructions

1. **Clone the repository**
   ```bash
   git clone 
   cd <repository-folder>
   ```

2. **Install required dependencies**
   ```bash
   pip install pandas streamlit plotly
   ```

3. **Add the data file**
   - Place your data file in the same directory as the script.
   - Expected file name: `reportTable.xlsx`
   - The file should have the following columns (minimum requirement):
     - `Year`
     - `Maker`
     - `Category`
     - `Count` (Number of registrations)

4. **Run the Streamlit app**
   ```bash
   streamlit run app.py
   ```

5. **Access the dashboard**
   - Open your browser and go to: http://localhost:8501

---

## Data Assumptions

- **Year**: Must be numeric and represent the registration year.
- **Maker**: Manufacturer name of the vehicle (e.g., "Maruti Suzuki India Ltd", "Volvo Group India Pvt Ltd").
- **Category**: Type of vehicle.
- **Count**: Integer representing the total registrations for that maker and category in the given year.
- Data cleaning will:
  - Remove rows with missing critical values.
  - Strip extra spaces from text columns.
  - Ensure consistent capitalization.
  - Handle special cases like quotes in maker names.

---

## Features

- **Year-over-Year (YoY) Maker-Wise Analysis**: Compare changes in registrations by selecting a specific maker.
- **Maker vs Category Comparison**: Compare the performance of one maker against different categories for a specific year.
- **Interactive Graphs**: Powered by Plotly for better visualization.

