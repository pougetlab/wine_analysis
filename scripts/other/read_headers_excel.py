import pandas as pd

def list_sorted_headers(excel_file):
    """
    Reads an Excel file and lists all headers from the first row in alphabetical order.

    Parameters:
        excel_file (str): Path to the Excel file.
    """
    # Read the first row only
    df = pd.read_excel(excel_file, nrows=0)

    # Extract headers and sort them
    sorted_headers = sorted(df.columns)

    # Print the sorted headers
    for header in sorted_headers:
        print(header)

# Example usage
excel_path = "/Users/juliaroquette/Work/oenology/data/Press_wines_2022/Tot_Chromato_2022_MERLOT_PRESSE_MIS_EN_FORME_SM.xlsx"  # Replace with your actual file path
# excel_path = "/Users/juliaroquette/Work/oenology/data/Press_wines_2022/Tot_Chromato_2022_CABERNET_SAUV_PRESSE_MIS_EN_FORME_SM.xlsx"  # Replace with your actual file path
list_sorted_headers(excel_path)