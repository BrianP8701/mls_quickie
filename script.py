from ast import literal_eval
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import json
import requests
import os
import time
import concurrent.futures
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
import multiprocessing
from dotenv import load_dotenv
import folium
from sklearn.cluster import DBSCAN
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from time import sleep
import math
from pydantic import BaseModel
import csv

load_dotenv()
google_places_api_key = os.getenv('GOOGLE_PLACES_API_KEY')

class SalePriceInfo(BaseModel):
     mls_no: str
     address: str
     city: str
     lot_sqft: float
     sell_price: float
def analyze_sell_prices(clusters_file: str, csv_file_path: str, output_file: str):
    """
    Analyze sell prices for each cluster and save the results to a JSON file.
    
    :param clusters_file: Path to the JSON file containing clusters.
    :param csv_file_path: Path to the CSV file containing property data.
    :param output_file: Path to the output JSON file for sell price analysis.
    """
    with open(clusters_file, 'r') as file:
        clusters = json.load(file)
    
    df = pd.read_csv(csv_file_path)

    # Clean 'SP' and 'Lot SqFt' columns
    df['SP'] = df['SP'].replace('[\$,]', '', regex=True).astype(float)
    df['Lot SqFt'] = df['Lot SqFt'].replace('[\$,]', '', regex=True).astype(float)
    
    results = []
    
    for cluster in clusters:
        cluster_df = df[df['MLS No'].isin(cluster)]
        
        if 'SP' not in cluster_df.columns:
            raise KeyError("The column 'SP' does not exist in the DataFrame.")
        
        sale_prices = cluster_df['SP'].tolist()
        average_sell_price = np.mean(sale_prices)
        median_sell_price = np.median(sale_prices)
        
        sp_objects = [
            SalePriceInfo(
                mls_no=row['MLS No'],
                address=row['Address'],
                city=row['City'],
                lot_sqft=row['Lot SqFt'],
                sell_price=row['SP']
            ).model_dump() for _, row in cluster_df.sort_values(by='SP').iterrows()
        ]
        
        results.append({
            'average_sell_price': average_sell_price,
            'median_sell_price': median_sell_price,
            'sell_price_list': sp_objects
        })
    
    with open(output_file, 'w') as file:
        json.dump(results, file, indent=4)

def get_city_lat_long(cities, output_file, api_key):
    """
    Get latitude and longitude for a list of cities and save to a JSON file.

    :param cities: List of city names.
    :param output_file: Path to the output JSON file.
    :param api_key: Google Places API key.
    """
    city_lat_long = {}
    total_cities = len(cities)
    base_url = "https://maps.googleapis.com/maps/api/geocode/json"

    # Limit to the first 5 cities
    for index, city in enumerate(cities[:5]):
        try:
            params = {
                'address': city,
                'key': api_key
            }
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            geocode_result = response.json()

            if geocode_result['results']:
                location = geocode_result['results'][0]['geometry']['location']
                city_lat_long[city] = {'latitude': location['lat'], 'longitude': location['lng']}
                print(city, location['lat'], location['lng'])
            else:
                print(f"Could not find location for {city}")
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data for {city}: {e}")
        
        # Print progress
        print(f"Progress: {index + 1}/{total_cities}")
        
        # Add a delay to avoid hitting rate limits
        sleep(1)  # Adjust the delay as needed

    # Save the city latitude and longitude data to a JSON file
    with open(output_file, 'w') as file:
        json.dump(city_lat_long, file, indent=4)

def get_cities_with_latitude_above(file_path: str, x: float) -> list:
    """
    Get a list of cities where the latitude is greater than a specified value.

    :param file_path: Path to the JSON file containing city latitude and longitude data.
    :param x: Latitude threshold.
    :return: List of city names with latitude greater than x.
    """
    with open(file_path, 'r') as file:
        city_lat_long = json.load(file)
    
    cities_above_latitude = [
        city for city, coords in city_lat_long.items() if coords['latitude'] > x
    ]
    
    return cities_above_latitude

def pick_random_rows(file_path: str, n: int = 50, output_file: str = 'random_50_rows.csv'):
    """
    Pick n random rows from the CSV file and save to another CSV file.
    
    :param file_path: Path to the input CSV file.
    :param n: Number of random rows to pick.
    :param output_file: Path to the output CSV file.
    """
    df = pd.read_csv(file_path)
    df = df.sample(n=n)
    df.to_csv(output_file, index=False)

def analyze_column(file_path: str, column_index: int):
    """
    Analyze a specific column in the CSV file.
    
    :param file_path: Path to the input CSV file.
    :param column_index: Index of the column to analyze.
    """
    df = pd.read_csv(file_path)
    column_header = df.columns[column_index]
    unique_values = df[column_header].nunique()
    sample_values = df[column_header].sample(n=50).tolist()
    
    print(f"Column header: {column_header}")
    print(f"Number of unique values: {unique_values}")
    
    value_counts = df[column_header].value_counts()
    for value, count in value_counts.items():
        print(f"{value}: {count}")

def filter_rows_by_value(file_path: str, column_index: int, value: str, output_file: str, include: bool = True):
    """
    Filter rows where the value in the specified column matches the given value.
    
    :param file_path: Path to the input CSV file.
    :param column_index: Index of the column to filter by.
    :param value: Value to filter by.
    :param output_file: Path to the output CSV file.
    :param include: Boolean to include or exclude the rows with the specified value.
    """
    df = pd.read_csv(file_path)
    column_header = df.columns[column_index]
    
    if include:
        df_filtered = df[df[column_header] == value]
    else:
        df_filtered = df[df[column_header] != value]
    
    df_filtered.to_csv(output_file, index=False)

def analyze_mls_groups(file_path: str):
    """
    Analyze the MLS groups from a file containing a list of lists.
    
    :param file_path: Path to the input file.
    """
    with open(file_path, 'r') as file:
        data = file.read()
    
    mls_groups = eval(data)
    average_length = sum(len(sublist) for sublist in mls_groups) / len(mls_groups)
    max_length = max(len(sublist) for sublist in mls_groups)
    min_length = min(len(sublist) for sublist in mls_groups)
    total_sublists = len(mls_groups)
    total_items = sum(len(sublist) for sublist in mls_groups)
    
    print(f"Average length of sublists: {average_length}")
    print(f"Maximum length of sublists: {max_length}")
    print(f"Minimum length of sublists: {min_length}")
    print(f"Total number of sublists: {total_sublists}")
    print(f"Total number of items in all sublists: {total_items}")

def find_duplicates(file_path: str, columns: list, output_file: str):
    """
    Find duplicates based on specified columns and save the duplicate indices to a file.
    
    :param file_path: Path to the input CSV file.
    :param columns: List of column indices to check for duplicates.
    :param output_file: Path to the output file for duplicate indices.
    """
    df = pd.read_csv(file_path)
    hashmap = {}
    duplicate_indices = {}
    duplicate_rows = []

    for index, row in df.iterrows():
        key = tuple(row[col] for col in columns)
        if key in hashmap:
            print(f"Duplicate found at row index: {index}, value: {key}")
            duplicate_rows.append(index)
            if key in duplicate_indices:
                duplicate_indices[key].append(index)
            else:
                duplicate_indices[key] = [hashmap[key], index]
        else:
            hashmap[key] = index

    with open(output_file, 'w') as file:
        for idx in duplicate_rows:
            file.write(f"{idx}\n")

    with open('duplicate_indices_hashmap.txt', 'w') as file:
        for key, indices in duplicate_indices.items():
            file.write(f"{key}: {indices}\n")

    for value, indices in duplicate_indices.items():
        print(f"Value: {value}, Count of duplicates: {len(indices)}")

def check_absolute_duplicates(file_path: str, duplicate_indices_file: str):
    """
    Check for absolute duplicates in the dataset.
    
    :param file_path: Path to the input CSV file.
    :param duplicate_indices_file: Path to the file containing duplicate indices.
    """
    df = pd.read_csv(file_path)
    
    with open(duplicate_indices_file, 'r') as file:
        duplicate_rows = [int(line.strip()) for line in file]

    duplicate_indices = {}
    with open('duplicate_indices_hashmap.txt', 'r') as file:
        for line in file:
            key, indices = line.strip().split(': ')
            key = eval(key)
            indices = eval(indices)
            duplicate_indices[key] = indices

    for value, indices in duplicate_indices.items():
        absolute_duplicates = 0
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                if df.iloc[indices[i]].equals(df.iloc[indices[j]]):
                    absolute_duplicates += 1
        print(f"Value: {value}, (Absolute duplicates: {absolute_duplicates}, Total duplicates: {len(indices)})")

def sample_duplicates(file_path: str, duplicate_indices_file: str, output_file: str, sample_size: int = 50):
    """
    Sample a specified number of duplicate groups and save them to a new CSV file.
    
    :param file_path: Path to the input CSV file.
    :param duplicate_indices_file: Path to the file containing duplicate indices.
    :param output_file: Path to the output CSV file.
    :param sample_size: Number of duplicate groups to sample.
    """
    df = pd.read_csv(file_path)
    
    duplicate_indices = {}
    with open(duplicate_indices_file, 'r') as file:
        for line in file:
            key, indices = line.strip().split(': ')
            key = key.replace('nan', 'None')
            key = literal_eval(key)
            indices = literal_eval(indices)
            duplicate_indices[key] = indices

    sampled_duplicates = []
    for value, indices in list(duplicate_indices.items())[:sample_size]:
        for idx in indices:
            sampled_duplicates.append(df.iloc[idx])

    sampled_duplicates_df = pd.DataFrame(sampled_duplicates)
    sampled_duplicates_df.to_csv(output_file, index=False)

def dedupe_by_closing_date(file_path: str, output_file: str):
    """
    Dedupe the dataset by keeping the row with the most recent closing date for each group of duplicates.
    
    :param file_path: Path to the input CSV file.
    :param output_file: Path to the output CSV file.
    """
    df = pd.read_csv(file_path)
    original_row_count = len(df)

    if 'Closing Date' not in df.columns:
        raise KeyError("The column 'Closing Date' does not exist in the DataFrame.")

    duplicate_indices = {}
    with open('duplicate_indices_hashmap.txt', 'r') as file:
        for line in file:
            key, indices = line.strip().split(': ')
            key = key.replace('nan', 'None')
            key = literal_eval(key)
            indices = literal_eval(indices)
            duplicate_indices[key] = indices

    rows_to_delete = []
    for value, indices in duplicate_indices.items():
        most_recent_index = max(indices, key=lambda x: pd.to_datetime(df.loc[x, 'Closing Date']))
        rows_to_delete.extend([idx for idx in indices if idx != most_recent_index])

    rows_to_delete.sort(reverse=True)
    df.drop(rows_to_delete, inplace=True)

    print(f"Original number of rows: {original_row_count}")
    print(f"Number of rows to delete: {len(rows_to_delete)}")
    print(f"Number of rows after deletion: {len(df)}")

    df.to_csv(output_file, index=False)

def count_rows_before_year(file_path: str, year: int):
    """
    Count the number of rows with a closing date before the specified year.
    
    :param file_path: Path to the input CSV file.
    :param year: The year to filter by.
    """
    df = pd.read_csv(file_path)
    df['Closing Date'] = pd.to_datetime(df['Closing Date'])
    rows_before_year = df[df['Closing Date'].dt.year <= year]
    oldest_date = df['Closing Date'].min()

    print(f"Number of rows before {year}: {len(rows_before_year)}")
    print(f"Oldest closing date: {oldest_date}")

def count_unique_cities(file_path: str):
    """
    Count the number of unique cities in the dataset.
    
    :param file_path: Path to the input CSV file.
    """
    df = pd.read_csv(file_path)
    unique_cities = df['City'].unique()
    print(f"Number of unique cities: {len(unique_cities)}")
    print(f"Unique cities: {unique_cities}")

def get_unique_values(file_path: str, column_name: str) -> dict:
    """
    Get a dictionary of unique values and their counts in a specified column of a CSV file.
    
    :param file_path: Path to the input CSV file.
    :param column_name: Name of the column to extract unique values from.
    :return: Dictionary of unique values and their counts in the specified column.
    """
    df = pd.read_csv(file_path)
    value_counts = df[column_name].value_counts().to_dict()
    return value_counts

def filter_cities_by_row_count(file_path: str, min_rows: int, output_file: str):
    """
    Filter cities with at least a specified number of rows and save to a new CSV file.
    
    :param file_path: Path to the input CSV file.
    :param min_rows: Minimum number of rows for a city to be included.
    :param output_file: Path to the output CSV file.
    """
    df = pd.read_csv(file_path)
    city_counts = df['City'].value_counts()
    filtered_cities = city_counts[city_counts >= min_rows].index
    filtered_df = df[df['City'].isin(filtered_cities)]
    filtered_df.to_csv(output_file, index=False)

def save_city_counts(file_path: str, output_file: str):
    """
    Save the count of rows for each city to a new CSV file.
    
    :param file_path: Path to the input CSV file.
    :param output_file: Path to the output CSV file.
    """
    df = pd.read_csv(file_path)
    city_counts = df['City'].value_counts()
    city_counts.to_csv(output_file, header=True)

def cluster_by_city(json_file_path: str, csv_file_path: str, output_file: str):
    """
    Further cluster existing clusters by city and save the new clusters to a JSON file.
    
    :param json_file_path: Path to the JSON file containing initial clusters.
    :param csv_file_path: Path to the CSV file containing property data.
    :param output_file: Path to the output JSON file containing new clusters.
    """
    # Load initial clusters
    with open(json_file_path, 'r') as file:
        initial_clusters = json.load(file)
    
    # Load CSV data
    df = pd.read_csv(csv_file_path)
    
    if 'City' not in df.columns:
        raise KeyError("The column 'City' does not exist in the DataFrame.")
    
    new_clusters = []
    
    for cluster in initial_clusters:
        cluster_df = df[df['MLS No'].isin(cluster)]
        city_clusters = cluster_df.groupby('City')['MLS No'].apply(list).tolist()
        new_clusters.extend(city_clusters)
    
    with open(output_file, 'w') as file:
        json.dump(new_clusters, file, indent=4)

def print_column_headers(file_path: str, column_indices: list):
    """
    Print the column headers for the specified column indices.
    
    :param file_path: Path to the input CSV file.
    :param column_indices: List of column indices.
    """
    df = pd.read_csv(file_path)
    headers = [df.columns[idx] for idx in column_indices]
    for idx, header in zip(column_indices, headers):
        print(f"Column index {idx}: {header}")

def read_city_clusters(file_path):
    """
    Read city clusters from a JSON file containing a list of lists.
    
    :param file_path: Path to the JSON file.
    :return: List of lists representing city clusters.
    """
    with open(file_path, 'r') as file:
        city_clusters = json.load(file)
    return city_clusters

def load_relevant_columns(file_path, relevant_columns, indices, index_column='MLS No'):
    """
    Load relevant columns from a CSV file for specific indices based on a specific column.
    
    :param file_path: Path to the CSV file.
    :param relevant_columns: List of column names to load.
    :param indices: List of row indices to load.
    :param index_column: Column name to use for filtering indices.
    :return: DataFrame with relevant columns for the specified indices.
    """
    df = pd.read_csv(file_path)
    df_filtered = df[df[index_column].isin(indices)]
    return df_filtered[relevant_columns]

def normalize_data(df):
    """
    Normalize the data using StandardScaler.
    
    :param df: DataFrame to normalize.
    :return: Normalized data as a NumPy array.
    """
    # Convert columns to numeric, coercing errors to NaN
    df = df.apply(pd.to_numeric, errors='coerce')
    
    # Fill NaN values with the mean of the column
    df = df.fillna(df.mean())
    
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(df)
    return normalized_data

def determine_optimal_clusters(data):
    """
    Determine the optimal number of clusters using WCSS and Silhouette Scores.
    
    :param data: Normalized data.
    :return: Tuple of WCSS and Silhouette Scores lists.
    """
    wcss = []
    silhouette_scores = []
    for k in range(2, 11):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(data, kmeans.labels_))
    
    return wcss, silhouette_scores

def apply_kmeans(data, n_clusters):
    """
    Apply K-Means clustering to the data.
    
    :param data: Normalized data.
    :param n_clusters: Number of clusters.
    :return: Cluster labels.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(data)
    return kmeans.labels_

def save_clusters_to_file(clusters, output_file):
    """
    Save clusters to a JSON file.
    
    :param clusters: List of cluster labels.
    :param output_file: Path to the output JSON file.
    """
    with open(output_file, 'w') as file:
        json.dump(clusters, file, indent=4)


def convert_txt_to_json(txt_file_path: str, json_file_path: str):
    """
    Convert a .txt file with a list of lists on each line to a JSON file with a list of lists.
    
    :param txt_file_path: Path to the input .txt file.
    :param json_file_path: Path to the output JSON file.
    """
    with open(txt_file_path, 'r') as txt_file:
        lines = txt_file.readlines()
    
    clusters = [literal_eval(line.strip()) for line in lines]
    
    with open(json_file_path, 'w') as json_file:
        json.dump(clusters, json_file, indent=4)

def get_address_strings(mls_data_path: str) -> dict:
    """
    Get a dictionary of address strings for all MLS Nos in the dataset.
    
    :param mls_data_path: Path to the CSV file containing MLS data.
    :return: Dictionary with MLS No as keys and address strings as values.
    """
    df = pd.read_csv(mls_data_path)
    address_dict = {}
    for _, row in df.iterrows():
        unit = row['Unit'] if pd.notna(row['Unit']) else ''
        address_string = f"{row['Address']}, {unit}, {row['City']}, CA".replace(', ,', ',')
        address_dict[row['MLS No']] = address_string
    return address_dict

def get_lat_long(address: str, api_key: str) -> dict:
    """
    Get the latitude and longitude for a given address using Google Places API.
    
    :param address: The address to search for.
    :param api_key: Your Google Places API key.
    :return: Dictionary with 'lat' and 'lng' keys.
    """
    base_url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {
        'address': address,
        'key': api_key
    }
    response = requests.get(base_url, params=params)
    response_data = response.json()
    
    if response_data['status'] == 'OK':
        location = response_data['results'][0]['geometry']['location']
        return {'lat': location['lat'], 'lng': location['lng']}
    else:
        # Log the error message for better debugging
        print(f"Error fetching data from Google Places API: {response_data['status']}")
        if 'error_message' in response_data:
            print(f"Error message: {response_data['error_message']}")
        raise Exception(f"Error fetching data from Google Places API: {response_data['status']}")


def save_mls_lat_long(mls_file_path: str, mls_data_path: str, api_key: str, output_file: str):
    """
    Save the latitude and longitude for each MLS No in a CSV file.
    
    :param mls_file_path: Path to the JSON file containing MLS No lists.
    :param mls_data_path: Path to the CSV file containing MLS data.
    :param api_key: Your Google Places API key.
    :param output_file: Path to the output CSV file.
    """
    # Load MLS groups from JSON file
    with open(mls_file_path, 'r') as file:
        mls_groups = json.load(file)
    
    # Flatten the list of lists to get all unique MLS numbers
    mls_flat_list = {mls_no for sublist in mls_groups for mls_no in sublist}
    
    # Get address strings for MLS numbers
    address_dict = get_address_strings(mls_data_path)
    
    # Read already processed MLS Nos from the output file
    processed_mls_nos = set()
    if os.path.exists(output_file):
        with open(output_file, 'r') as csv_file:
            next(csv_file)  # Skip header
            for line in csv_file:
                processed_mls_nos.add(line.split(',')[0])
    
    # Remove already processed MLS numbers
    mls_flat_list = mls_flat_list - processed_mls_nos

    total_mls_nos = len(mls_flat_list)
    processed_count = 0
    batch_size = 100
    rate_limit = 3000  # requests per minute

    # Open the output file for appending
    with open(output_file, 'a') as csv_file:
        if not processed_mls_nos:
            csv_file.write('MLS No,Latitude,Longitude\n')  # Write header if file is empty
        
        # Process each MLS number in batches
        mls_list = list(mls_flat_list)
        for i in range(0, len(mls_list), batch_size):
            batch = mls_list[i:i + batch_size]
            with ThreadPoolExecutor(max_workers=batch_size) as executor:
                future_to_mls = {executor.submit(get_lat_long, address_dict.get(mls_no), api_key): mls_no for mls_no in batch if address_dict.get(mls_no)}
                
                for future in as_completed(future_to_mls):
                    mls_no = future_to_mls[future]
                    try:
                        lat_long = future.result()
                        csv_file.write(f"{mls_no},{lat_long['lat']},{lat_long['lng']}\n")
                        processed_count += 1
                        print(f"Processed {processed_count} out of {total_mls_nos} MLS Nos")
                    except Exception as e:
                        print(f"Error processing MLS No {mls_no}: {e}")
            
            # Sleep to respect the rate limit
            time.sleep(60 / (rate_limit / batch_size))

def get_address_strings_2(mls_data_path: str) -> dict:
    """
    Get a dictionary of address strings for all MLS Nos in the dataset.
    
    :param mls_data_path: Path to the CSV file containing MLS data.
    :return: Dictionary with MLS No as keys and address strings as values.
    """
    try:
        df = pd.read_csv(mls_data_path)
        if not isinstance(df, pd.DataFrame):
            raise ValueError("The loaded data is not a DataFrame.")
        
        address_dict = {}
        for _, row in df.iterrows():
            unit = row['Unit'] if pd.notna(row['Unit']) else ''
            address_string = f"{row['Address']}, {unit}, {row['City']}, CA".replace(', ,', ',')
            address_dict[row['MLS No']] = address_string
        return address_dict
    except Exception as e:
        print(f"Error reading MLS data from {mls_data_path}: {e}")
        return {}

def update_mls_lat_long(clusters_file_path: str, mls_data_path: str, api_key: str, lat_long_csv: str):
    """
    Update the latitude and longitude for each MLS No in the first cluster of the clusters JSON file and overwrite in the CSV file.
    
    :param clusters_file_path: Path to the JSON file containing MLS No clusters.
    :param mls_data_path: Path to the CSV file containing MLS data.
    :param api_key: Your Google Places API key.
    :param lat_long_csv: Path to the CSV file containing existing MLS No latitude and longitude data.
    """
    # Load MLS clusters from JSON file
    with open(clusters_file_path, 'r') as file:
        mls_clusters = json.load(file)
    
    # Process only the first cluster
    if not mls_clusters:
        print("No clusters found in the file.")
        return
    
    first_cluster = mls_clusters[0]
    
    # Get address strings for MLS numbers
    address_dict = get_address_strings_2(mls_data_path)
    if not address_dict:
        print("Failed to get address strings. Exiting.")
        return
    
    # Read existing MLS No lat/long data from the CSV file
    lat_long_data = {}
    with open(lat_long_csv, 'r') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            lat_long_data[row['MLS No']] = {'lat': row['Latitude'], 'lng': row['Longitude']}
    
    # Update lat/long data for MLS numbers in the first cluster
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_mls = {executor.submit(get_lat_long, address_dict.get(mls_no), api_key): mls_no for mls_no in first_cluster if address_dict.get(mls_no)}
        
        for future in as_completed(future_to_mls):
            mls_no = future_to_mls[future]
            try:
                lat_long = future.result()
                lat_long_data[mls_no] = {'lat': lat_long['lat'], 'lng': lat_long['lng']}
                print(f"Updated {mls_no} with lat: {lat_long['lat']}, lng: {lat_long['lng']}")
            except Exception as e:
                print(f"Error updating MLS No {mls_no}: {e}")
    
    # Write updated lat/long data back to the CSV file
    with open(lat_long_csv, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=['MLS No', 'Latitude', 'Longitude'])
        writer.writeheader()
        for mls_no, lat_long in lat_long_data.items():
            writer.writerow({'MLS No': mls_no, 'Latitude': lat_long['lat'], 'Longitude': lat_long['lng']})

def dedupe_by_first_column(file_path: str, output_file: str):
    """
    Remove duplicate rows based on the first column and save the deduplicated data to a new CSV file.
    
    :param file_path: Path to the input CSV file.
    :param output_file: Path to the output CSV file.
    """
    df = pd.read_csv(file_path)
    deduped_df = df.drop_duplicates(subset=df.columns[0])
    deduped_df.to_csv(output_file, index=False)

def analyze_cluster_sizes(json_file_path: str):
    """
    Analyze cluster sizes from a JSON file containing a list of clusters.
    
    :param json_file_path: Path to the JSON file.
    """
    with open(json_file_path, 'r') as file:
        clusters = json.load(file)
    
    cluster_sizes = [len(cluster) for cluster in clusters]
    total_clusters = len(cluster_sizes)
    min_size = min(cluster_sizes)
    max_size = max(cluster_sizes)
    median_size = np.median(cluster_sizes)
    average_size = np.mean(cluster_sizes)
    
    print(f"Total number of clusters: {total_clusters}")
    print(f"Minimum cluster size: {min_size}")
    print(f"Maximum cluster size: {max_size}")
    print(f"Median cluster size: {median_size}")
    print(f"Average cluster size: {average_size}")

def cluster_by_city_and_bt(file_path: str, output_file: str):
    """
    Cluster rows by city and then by 'BT' column, saving the clusters to a new JSON file.
    
    :param file_path: Path to the input CSV file.
    :param output_file: Path to the output JSON file containing clusters.
    """
    df = pd.read_csv(file_path)
    
    if 'City' not in df.columns or 'BT' not in df.columns:
        raise KeyError("The columns 'City' and/or 'BT' do not exist in the DataFrame.")
    
    clusters = df.groupby(['City', 'BT'])['MLS No'].apply(list).tolist()
    
    with open(output_file, 'w') as file:
        json.dump(clusters, file, indent=4)

def delete_small_clusters(json_file_path: str, min_size: int, output_file: str):
    """
    Delete clusters smaller than a specified size and save the remaining clusters to a new JSON file.
    
    :param json_file_path: Path to the input JSON file containing clusters.
    :param min_size: Minimum size of clusters to keep.
    :param output_file: Path to the output JSON file for remaining clusters.
    """
    with open(json_file_path, 'r') as file:
        clusters = json.load(file)
    
    filtered_clusters = [cluster for cluster in clusters if len(cluster) >= min_size]
    
    with open(output_file, 'w') as file:
        json.dump(filtered_clusters, file, indent=4)

def count_unique_elements_in_clusters(json_file_path: str) -> int:
    """
    Flatten the clusters into a list and count the number of unique elements.
    
    :param json_file_path: Path to the JSON file containing clusters.
    :return: Number of unique elements in the flattened list.
    """
    with open(json_file_path, 'r') as file:
        clusters = json.load(file)
    
    flattened_list = [item for cluster in clusters for item in cluster]
    unique_elements = set(flattened_list)
    
    print(f"Number of unique elements in clusters: {len(unique_elements)}")
    return len(unique_elements)

def cluster_by_sqft_within_percentage(json_file_path: str, csv_file_path: str, percentage: float, output_file: str):
    """
    Cluster properties within each cluster by 'SqFt' within a specified percentage range.
    
    :param json_file_path: Path to the JSON file containing initial clusters.
    :param csv_file_path: Path to the CSV file containing property data.
    :param percentage: Percentage range to cluster 'SqFt' within.
    :param output_file: Path to the output JSON file for new clusters.
    """
    with open(json_file_path, 'r') as file:
        initial_clusters = json.load(file)
    
    df = pd.read_csv(csv_file_path)
    
    # Remove non-numeric characters and convert 'Lot SqFt' to numeric
    df['Lot SqFt'] = df['Lot SqFt'].replace('[\$,]', '', regex=True).astype(float)
    df.dropna(subset=['Lot SqFt'], inplace=True)
    
    new_clusters = []
    
    for cluster in initial_clusters:
        cluster_df = df[df['MLS No'].isin(cluster)].copy()
        cluster_df.sort_values(by='Lot SqFt', inplace=True)
        
        visited = set()
        while not cluster_df.empty:
            # Randomly select an item
            random_index = cluster_df.sample(n=1).index[0]
            sqft = cluster_df.loc[random_index, 'Lot SqFt']
            
            lower_bound = sqft * (1 - percentage / 100)
            upper_bound = sqft * (1 + percentage / 100)
            
            # Find all items within the percentage range
            similar_sqft_df = cluster_df[(cluster_df['Lot SqFt'] >= lower_bound) & (cluster_df['Lot SqFt'] <= upper_bound)]
            similar_ids = similar_sqft_df['MLS No'].tolist()
            
            # Add to new cluster
            new_clusters.append(similar_ids)
            visited.update(similar_ids)
            
            # Remove clustered items from the DataFrame
            cluster_df = cluster_df.drop(similar_sqft_df.index)
        
        # Ensure all MLS Nos are included in the new clusters
        unclustered_mls_nos = set(cluster) - visited
        for mls_no in unclustered_mls_nos:
            new_clusters.append([mls_no])
    
    with open(output_file, 'w') as file:
        json.dump(new_clusters, file, indent=4)

def calculate_price_spread(json_file_path: str, csv_file_path: str, output_file: str):
    """
    Calculate the price spread (IQR) for each cluster and save the results to a file.
    
    :param json_file_path: Path to the JSON file containing clusters.
    :param csv_file_path: Path to the CSV file containing property data.
    :param output_file: Path to the output file for price spreads.
    """
    with open(json_file_path, 'r') as file:
        clusters = json.load(file)
    
    df = pd.read_csv(csv_file_path)
    
    price_spreads = []
    total_clusters = len(clusters)
    for index, cluster in enumerate(clusters):
        price_spreads.append(calculate_iqr(cluster, df))
        print(f"Processed {index + 1}/{total_clusters} clusters "
              f"({((index + 1) / total_clusters) * 100:.2f}%)")

    with open(output_file, 'w') as file:
        for spread in price_spreads:
            file.write(f"{spread}\n" if spread is not None else "0\n")  # Handle None values

def analyze_price_spreads(price_spreads_file: str):
    """
    Analyze the price spreads and print the 10 max, min, median, 25th and 75th percentile values, median, and average value.
    
    :param price_spreads_file: Path to the file containing price spreads.
    """
    with open(price_spreads_file, 'r') as file:
        price_spreads = [float(line.strip()) for line in file if line.strip()]

    price_spreads.sort()

    max_values = price_spreads[-10:]
    min_values = price_spreads[:10]
    median_value = np.median(price_spreads)
    average_value = np.mean(price_spreads)
    percentile_25 = np.percentile(price_spreads, 25)
    percentile_75 = np.percentile(price_spreads, 75)

    print(f"10 Maximum Values: {max_values}")
    print(f"10 Minimum Values: {min_values}")
    print(f"Median Value: {median_value}")
    print(f"Average Value: {average_value}")
    print(f"25th Percentile: {percentile_25}")
    print(f"75th Percentile: {percentile_75}")

def cluster_by_bed_bath_ratio(json_file_path: str, csv_file_path: str, output_file: str):
    """
    Cluster properties by a combined bedroom and bathroom ratio within a specified percentage range.
    
    :param json_file_path: Path to the JSON file containing initial clusters.
    :param csv_file_path: Path to the CSV file containing property data.
    :param output_file: Path to the output JSON file for new clusters.
    """
    with open(json_file_path, 'r') as file:
        initial_clusters = json.load(file)
    
    df = pd.read_csv(csv_file_path)
    
    new_clusters = []
    
    for cluster in initial_clusters:
        cluster_df = df[df['MLS No'].isin(cluster)]
        cluster_df['BR'] = pd.to_numeric(cluster_df['BR'], errors='coerce')
        cluster_df['Bth'] = pd.to_numeric(cluster_df['Bth'], errors='coerce')
        cluster_df.dropna(subset=['BR', 'Bth'], inplace=True)
        
        # Calculate the combined bedroom and bathroom ratio
        cluster_df['BR_Bth_Ratio'] = cluster_df['BR'] + (cluster_df['Bth'] * 0.5)
        
        # Sort by the combined ratio
        cluster_df.sort_values(by='BR_Bth_Ratio', inplace=True)
        
        visited = set()
        for _, row in cluster_df.iterrows():
            if row['MLS No'] in visited:
                continue
            
            br_bth_value = row['BR_Bth_Ratio']
            lower_bound = br_bth_value * 0.85
            upper_bound = br_bth_value * 1.15
            
            similar_br_bth_df = cluster_df[
                (cluster_df['BR_Bth_Ratio'] >= lower_bound) &
                (cluster_df['BR_Bth_Ratio'] <= upper_bound)
            ]
            
            similar_ids = similar_br_bth_df['MLS No'].tolist()
            if not any(id in visited for id in similar_ids):
                new_clusters.append(similar_ids)
                visited.update(similar_ids)
    
    with open(output_file, 'w') as file:
        json.dump(new_clusters, file, indent=4)

def sort_clusters_by_price_spread(price_spreads_file: str, clusters_file: str, output_file: str):
    """
    Sort clusters by price spread in descending order and save to a new JSON file.
    
    :param price_spreads_file: Path to the file containing price spreads.
    :param clusters_file: Path to the JSON file containing clusters.
    :param output_file: Path to the output JSON file for sorted clusters.
    """
    with open(price_spreads_file, 'r') as file:
        price_spreads = [float(line.strip()) for line in file if line.strip()]

    with open(clusters_file, 'r') as file:
        clusters = json.load(file)

    # Pair clusters with their price spreads
    clusters_with_spreads = list(zip(price_spreads, clusters))
    
    # Sort clusters by price spread in descending order
    clusters_with_spreads.sort(reverse=True, key=lambda x: x[0])
    
    # Extract sorted clusters
    sorted_clusters = [cluster for _, cluster in clusters_with_spreads]
    
    with open(output_file, 'w') as file:
        json.dump(sorted_clusters, file, indent=4)

def create_clusters_with_unique_elements(clusters_file: str, num_unique_elements: int, output_file: str):
    """
    Create a new clusters file with a specified number of unique elements.
    
    :param clusters_file: Path to the JSON file containing clusters.
    :param num_unique_elements: Number of unique elements desired in the new clusters.
    :param output_file: Path to the output JSON file for new clusters.
    """
    with open(clusters_file, 'r') as file:
        clusters = json.load(file)

    unique_elements = set()
    new_clusters = []

    for cluster in clusters:
        for element in cluster:
            unique_elements.add(element)
            if len(unique_elements) > num_unique_elements:
                break
        new_clusters.append(cluster)
        if len(unique_elements) > num_unique_elements:
            break

    with open(output_file, 'w') as file:
        json.dump(new_clusters, file, indent=4)

def flatten_clusters(clusters_file: str, output_file: str):
    """
    Flatten all clusters into a single list and save to a new JSON file.
    
    :param clusters_file: Path to the JSON file containing clusters.
    :param output_file: Path to the output JSON file for the flattened cluster.
    """
    with open(clusters_file, 'r') as file:
        clusters = json.load(file)
    
    # Flatten all clusters into a single list
    flattened_cluster = [item for cluster in clusters for item in cluster]
    
    # Save the flattened cluster as a single list inside a list
    with open(output_file, 'w') as file:
        json.dump([flattened_cluster], file, indent=4)

def geographical_clustering_with_dbscan(mls_lat_long_file: str, eps: float, min_samples: int, output_file: str):
    """
    Perform DBSCAN clustering based on geographical coordinates to minimize overlap.
    
    :param mls_lat_long_file: Path to the CSV file containing MLS No, Latitude, and Longitude.
    :param eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    :param min_samples: The number of samples in a neighborhood for a point to be considered as a core point.
    :param output_file: Path to the output JSON file for clusters.
    """
    df = pd.read_csv(mls_lat_long_file)
    coordinates = df[['Latitude', 'Longitude']].values
    
    # Adjust eps and min_samples to find optimal clustering
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    df['Cluster'] = dbscan.fit_predict(coordinates)
    
    # Filter out noise points (Cluster label -1)
    clusters = df[df['Cluster'] != -1].groupby('Cluster')['MLS No'].apply(list).tolist()
    
    with open(output_file, 'w') as file:
        json.dump(clusters, file, indent=4)

def geographical_clustering_with_kmeans(mls_lat_long_file: str, k: int, output_file: str):
    """
    Perform K-Means clustering based on geographical coordinates.
    
    :param mls_lat_long_file: Path to the CSV file containing MLS No, Latitude, and Longitude.
    :param k: Number of clusters.
    :param output_file: Path to the output JSON file for clusters.
    """
    df = pd.read_csv(mls_lat_long_file)
    coordinates = df[['Latitude', 'Longitude']].values
    
    kmeans = KMeans(n_clusters=k, random_state=42)
    df['Cluster'] = kmeans.fit_predict(coordinates)
    
    clusters = df.groupby('Cluster')['MLS No'].apply(list).tolist()
    
    with open(output_file, 'w') as file:
        json.dump(clusters, file, indent=4)

def visualize_clusters(mls_lat_long_file: str, clusters_file: str):
    """
    Visualize clusters on a map using folium.
    
    :param mls_lat_long_file: Path to the CSV file containing MLS No, Latitude, and Longitude.
    :param clusters_file: Path to the JSON file containing clusters.
    """
    df = pd.read_csv(mls_lat_long_file)
    with open(clusters_file, 'r') as file:
        clusters = json.load(file)
    
    # Create a map centered around the average location
    avg_lat = df['Latitude'].mean()
    avg_long = df['Longitude'].mean()
    m = folium.Map(location=[avg_lat, avg_long], zoom_start=10)
    
    # Add points to the map
    for cluster in clusters:
        for mls_no in cluster:
            row = df[df['MLS No'] == mls_no]
            folium.CircleMarker(
                location=(row['Latitude'].values[0], row['Longitude'].values[0]),
                radius=5,
                color='blue',
                fill=True,
                fill_color='blue'
            ).add_to(m)
    
    # Save the map to an HTML file
    m.save('clusters_map.html')

def remove_mls_below_latitude(clusters_file: str, mls_lat_long_file: str, latitude_threshold: float, output_file: str):
    """
    Remove MLS Nos from clusters where the latitude is below a given threshold.
    
    :param clusters_file: Path to the JSON file containing clusters.
    :param mls_lat_long_file: Path to the CSV file containing MLS No, Latitude, and Longitude.
    :param latitude_threshold: Latitude threshold to filter MLS Nos.
    :param output_file: Path to the output JSON file for filtered clusters.
    """
    # Load clusters
    with open(clusters_file, 'r') as file:
        clusters = json.load(file)
    
    # Load MLS latitude data
    df = pd.read_csv(mls_lat_long_file)
    mls_lat_dict = df.set_index('MLS No')['Latitude'].to_dict()
    
    # Filter clusters
    filtered_clusters = []
    for cluster in clusters:
        filtered_cluster = [mls_no for mls_no in cluster if mls_lat_dict.get(mls_no, float('inf')) >= latitude_threshold]
        if filtered_cluster:
            filtered_clusters.append(filtered_cluster)
    
    # Save filtered clusters
    with open(output_file, 'w') as file:
        json.dump(filtered_clusters, file, indent=4)

def filter_mls_by_latitude(mls_lat_long_file: str, latitude_threshold: float, output_file: str):
    """
    Create a new CSV file with MLS Nos where the latitude is above a given threshold.
    
    :param mls_lat_long_file: Path to the CSV file containing MLS No, Latitude, and Longitude.
    :param latitude_threshold: Latitude threshold to filter MLS Nos.
    :param output_file: Path to the output CSV file for filtered MLS Nos.
    """
    df = pd.read_csv(mls_lat_long_file)
    filtered_df = df[df['Latitude'] >= latitude_threshold]
    filtered_df.to_csv(output_file, index=False)

def visualize_clusters_with_circles(mls_lat_long_file: str, clusters_file: str, visualization_file: str):
     """
     Visualize clusters on a map using folium, with a circle around each cluster.
     
     :param mls_lat_long_file: Path to the CSV file containing MLS No, Latitude, and Longitude.
     :param clusters_file: Path to the JSON file containing clusters.
     """
     df = pd.read_csv(mls_lat_long_file)
     with open(clusters_file, 'r') as file:
         clusters = json.load(file)
     
     # Create a map centered around the average location
     avg_lat = df['Latitude'].mean()
     avg_long = df['Longitude'].mean()
     m = folium.Map(location=[avg_lat, avg_long], zoom_start=10)
     
     # Define colors for clusters
     colors = ['red', 'green', 'blue', 'purple', 'orange', 'darkred', 'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue', 'darkpurple', 'white', 'pink', 'lightblue', 'lightgreen', 'gray', 'black', 'lightgray']
     
     # Add points and circles to the map
     for cluster_index, cluster in enumerate(clusters):
         cluster_color = colors[cluster_index % len(colors)]
         cluster_latitudes = []
         cluster_longitudes = []
         
         for mls_no in cluster:
             row = df[df['MLS No'] == mls_no]
             lat = row['Latitude'].values[0]
             long = row['Longitude'].values[0]
             cluster_latitudes.append(lat)
             cluster_longitudes.append(long)
             
             folium.CircleMarker(
                 location=(lat, long),
                 radius=5,
                 color='blue',
                 fill=True,
                 fill_color='blue'
             ).add_to(m)
         
         # Draw a circle around the cluster
         if cluster_latitudes and cluster_longitudes:
             center_lat = np.mean(cluster_latitudes)
             center_long = np.mean(cluster_longitudes)
             # Calculate the radius based on the spread of the cluster
             lat_spread = max(cluster_latitudes) - min(cluster_latitudes)
             long_spread = max(cluster_longitudes) - min(cluster_longitudes)
             radius = max(lat_spread, long_spread) * 111000 / 2  # Convert degrees to meters
             folium.Circle(
                 location=(center_lat, center_long),
                 radius=radius,
                 color=cluster_color,
                 fill=False
             ).add_to(m)
     
     # Save the map to an HTML file
     m.save(visualization_file)

def visualize_clusters_with_unique_colors(mls_lat_long_file: str, clusters_file: str, visualization_file: str):
    """
    Visualize clusters on a map using folium, with each cluster's dots having a unique color.
    
    :param mls_lat_long_file: Path to the CSV file containing MLS No, Latitude, and Longitude.
    :param clusters_file: Path to the JSON file containing clusters.
    :param visualization_file: Path to the output HTML file for visualization.
    """
    df = pd.read_csv(mls_lat_long_file)
    with open(clusters_file, 'r') as file:
        clusters = json.load(file)
    
    # Create a map centered around the average location
    avg_lat = df['Latitude'].mean()
    avg_long = df['Longitude'].mean()
    m = folium.Map(location=[avg_lat, avg_long], zoom_start=10)
    
    # Define colors for clusters
    colors = ['red', 'green', 'blue', 'purple', 'orange', 'darkred', 'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue', 'darkpurple', 'white', 'pink', 'lightblue', 'lightgreen', 'gray', 'black', 'lightgray']
    
    # Add points to the map with unique colors for each cluster
    for cluster_index, cluster in enumerate(clusters):
        cluster_color = colors[cluster_index % len(colors)]
        
        for mls_no in cluster:
            row = df[df['MLS No'] == mls_no]
            lat = row['Latitude'].values[0]
            long = row['Longitude'].values[0]
            
            folium.CircleMarker(
                location=(lat, long),
                radius=5,
                color=cluster_color,
                fill=True,
                fill_color=cluster_color
            ).add_to(m)
    
    # Save the map to an HTML file
    m.save(visualization_file)

def cluster_by_bt(json_file_path: str, csv_file_path: str, output_file: str):
    """
    Cluster properties by 'BT' column within existing clusters and save the new clusters to a JSON file.
    
    :param json_file_path: Path to the JSON file containing initial clusters.
    :param csv_file_path: Path to the CSV file containing property data.
    :param output_file: Path to the output JSON file for new clusters.
    """
    with open(json_file_path, 'r') as file:
        initial_clusters = json.load(file)
    
    df = pd.read_csv(csv_file_path)
    
    new_clusters = []
    
    for cluster in initial_clusters:
        cluster_df = df[df['MLS No'].isin(cluster)]
        
        if 'BT' not in cluster_df.columns:
            raise KeyError("The column 'BT' does not exist in the DataFrame.")
        
        bt_clusters = cluster_df.groupby('BT')['MLS No'].apply(list).tolist()
        new_clusters.extend(bt_clusters)
    
    with open(output_file, 'w') as file:
        json.dump(new_clusters, file, indent=4)

def extract_lot_sqft_from_clusters(json_file_path: str, csv_file_path: str, output_file: str):
    """
    Extract each MLS No with its 'Lot SqFt' value from the clusters and save to a JSON file.
    
    :param json_file_path: Path to the JSON file containing clusters.
    :param csv_file_path: Path to the CSV file containing property data.
    :param output_file: Path to the output JSON file.
    """
    with open(json_file_path, 'r') as file:
        clusters = json.load(file)
    
    df = pd.read_csv(csv_file_path)
    
    # Check the data types and the first few rows
    print(df.dtypes)
    print(df.head())
    
    # Remove non-numeric characters and convert 'Lot SqFt' to numeric
    df['Lot SqFt'] = df['Lot SqFt'].replace('[\$,]', '', regex=True).astype(float)
    
    # Check how many NaN values are present
    num_nan = df['Lot SqFt'].isna().sum()
    print(f"Number of NaN values in 'Lot SqFt': {num_nan}")
    
    # Create a dictionary for quick lookup
    mls_to_lot_sqft = df.set_index('MLS No')['Lot SqFt'].to_dict()
    
    mls_lot_sqft = {}
    
    for cluster in clusters:
        for mls_no in cluster:
            if mls_no in mls_to_lot_sqft:
                mls_lot_sqft[mls_no] = mls_to_lot_sqft[mls_no]
    
    with open(output_file, 'w') as file:
        json.dump(mls_lot_sqft, file, indent=4)

def get_top_clusters(clusters_file: str, top_n: int, output_file: str):
    """
    Get the top N clusters based on size and save them to a new JSON file.
    
    :param clusters_file: Path to the JSON file containing clusters.
    :param top_n: Number of top clusters to extract.
    :param output_file: Path to the output JSON file for top clusters.
    """
    with open(clusters_file, 'r') as file:
        clusters = json.load(file)

    # Get the top N clusters
    top_clusters = clusters[:top_n]
    
    # Save the top clusters to a new file
    with open(output_file, 'w') as file:
        json.dump(top_clusters, file, indent=4)

def get_value_counts_in_clusters(clusters_file: str, csv_file_path: str, column_name: str) -> dict:
    """
    Get a dictionary of unique values and their counts in a specified column of a CSV file,
    considering only the rows specified by a cluster JSON file.

    :param clusters_file: Path to the JSON file containing clusters.
    :param csv_file_path: Path to the CSV file containing property data.
    :param column_name: Name of the column to extract unique values from.
    :return: Dictionary of unique values and their counts in the specified column.
    """
    # Load clusters
    with open(clusters_file, 'r') as file:
        clusters = json.load(file)
    
    # Load CSV data
    df = pd.read_csv(csv_file_path)
    
    # Check if the column exists
    if column_name not in df.columns:
        raise KeyError(f"The column '{column_name}' does not exist in the DataFrame.")
    
    # Track which MLS Nos have been checked
    checked_mls_nos = set()
    value_counts = {}

    for cluster in clusters:
        for mls_no in cluster:
            if mls_no not in checked_mls_nos:
                checked_mls_nos.add(mls_no)
                # Get the row corresponding to the MLS No
                row = df[df['MLS No'] == mls_no]
                if not row.empty:
                    value = row.iloc[0][column_name]
                    if value in value_counts:
                        value_counts[value] += 1
                    else:
                        value_counts[value] = 1

    return value_counts

def create_initial_cluster_from_list(mls_list: list, output_json_file: str):
    """
    Create an initial cluster JSON file with all MLS Nos in a single list.

    :param mls_list: List of MLS Nos.
    :param output_json_file: Path to the output JSON file.
    """
    # Create a single cluster with all MLS Nos
    initial_cluster = [mls_list]
    
    # Save the cluster to a JSON file
    with open(output_json_file, 'w') as file:
        json.dump(initial_cluster, file, indent=4)

def get_column_values(csv_file_path: str, column_index: int) -> list:
    """
    Get all values from a specified column index in a CSV file.

    :param csv_file_path: Path to the CSV file.
    :param column_index: Index of the column to extract values from.
    :return: List of values from the specified column.
    """
    df = pd.read_csv(csv_file_path)
    if column_index < 0 or column_index >= len(df.columns):
        raise IndexError("Column index is out of range.")
    
    return df.iloc[:, column_index].tolist()

def visualize_top_clusters(mls_lat_long_file: str, top_clusters_file: str, visualization_file: str):
    """
    Visualize the top clusters on a map using folium, with rank numbers.
    
    :param mls_lat_long_file: Path to the CSV file containing MLS No, Latitude, and Longitude.
    :param top_clusters_file: Path to the JSON file containing top clusters.
    :param visualization_file: Path to the output HTML file for visualization.
    """
    df = pd.read_csv(mls_lat_long_file)
    with open(top_clusters_file, 'r') as file:
        top_clusters = json.load(file)
    
    # Create a map centered around the average location
    avg_lat = df['Latitude'].mean()
    avg_long = df['Longitude'].mean()
    m = folium.Map(location=[avg_lat, avg_long], zoom_start=10)
    
    # Define colors for clusters
    colors = ['red', 'green', 'blue', 'purple', 'orange', 'darkred', 'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue', 'darkpurple', 'pink', 'lightblue', 'lightgreen', 'gray']
    
    # Add points and circles to the map
    for cluster_index, cluster in enumerate(top_clusters):
        cluster_color = colors[cluster_index % len(colors)]
        
        for mls_no in cluster:
            row = df[df['MLS No'] == mls_no]
            folium.CircleMarker(
                location=(row['Latitude'].values[0], row['Longitude'].values[0]),
                radius=5,
                color=cluster_color,
                fill=True,
                fill_color=cluster_color
            ).add_to(m)
        
        # Add a label with the rank number
        if cluster:
            first_mls_no = cluster[0]
            first_row = df[df['MLS No'] == first_mls_no]
            text_color = 'black' if cluster_color in ['white', 'lightgray', 'beige', 'lightblue', 'lightgreen'] else 'white'
            folium.Marker(
                location=(first_row['Latitude'].values[0], first_row['Longitude'].values[0]),
                icon=folium.DivIcon(html=f'''
                    <div style="font-size: 12px; color: {text_color}; background-color: {cluster_color}; padding: 4px 8px; border-radius: 3px; width: 60px; text-align: center;">
                        <b>Rank {cluster_index + 1}</b>
                    </div>
                ''')
            ).add_to(m)
    
    # Save the map to an HTML file
    m.save(visualization_file)

def generate_markers_json(mls_lat_long_file: str, top_clusters_file: str, output_json_file: str):
    """
    Generate a JSON file of markers with rank labels for visualization.
    
    :param mls_lat_long_file: Path to the CSV file containing MLS No, Latitude, and Longitude.
    :param top_clusters_file: Path to the JSON file containing top clusters.
    :param output_json_file: Path to the output JSON file for markers.
    """
    df = pd.read_csv(mls_lat_long_file)
    with open(top_clusters_file, 'r') as file:
        top_clusters = json.load(file)
    
    # Define colors for clusters, excluding 'white' and 'lightgray'
    colors = ['yellow', 'red', 'green', 'blue', 'purple', 'orange', 'darkred', 'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue', 'darkpurple', 'pink', 'lightblue', 'lightgreen', 'gray', 'cyan', 'magenta', 'lime', 'gold', 'aqua', 'fuchsia']
    
    # Remove 'white' and 'lightgray' from the list
    colors = [color for color in colors if color not in ['white', 'lightgray']]
    
    markers = []

    for cluster_index, cluster in enumerate(top_clusters):
        cluster_color = colors[cluster_index % len(colors)]
        
        for mls_no in cluster:
            row = df[df['MLS No'] == mls_no]
            markers.append({
                "position": [row['Latitude'].values[0], row['Longitude'].values[0]],
                "options": {
                    "color": cluster_color,
                    "radius": 5,
                    "fill": True,
                    "fillColor": cluster_color
                }
            })
        
        # Add a label with the rank number
        if cluster:
            first_mls_no = cluster[0]
            first_row = df[df['MLS No'] == first_mls_no]
            # Ensure text color is black for yellow
            text_color = 'black' if cluster_color in ['white', 'lightgray', 'beige', 'lightblue', 'lightgreen', 'yellow'] else 'white'
            markers.append({
                "position": [first_row['Latitude'].values[0], first_row['Longitude'].values[0]],
                "options": {
                    "icon": {
                        "html": f'''
                            <div style="font-size: 12px; color: {text_color}; background-color: {cluster_color}; padding: 4px 8px; border-radius: 3px; width: 60px; text-align: center;">
                                <b>Rank {cluster_index + 1}</b>
                            </div>
                        '''
                    }
                }
            })
    
    # Save the markers to a JSON file
    with open(output_json_file, 'w') as file:
        json.dump(markers, file, indent=4)

def determine_eps(mls_lat_long_file: str, min_samples: int):
    df = pd.read_csv(mls_lat_long_file)
    coordinates = df[['Latitude', 'Longitude']].values
    
    # Compute the k-nearest neighbors
    neighbors = NearestNeighbors(n_neighbors=min_samples)
    neighbors_fit = neighbors.fit(coordinates)
    distances, indices = neighbors_fit.kneighbors(coordinates)
    
    # Sort the distances to the k-th nearest neighbor
    distances = np.sort(distances[:, min_samples-1], axis=0)
    
    # Plot the distances
    plt.plot(distances)
    plt.title('K-Distance Graph')
    plt.xlabel('Points sorted by distance')
    plt.ylabel('Distance to {}-th nearest neighbor'.format(min_samples))
    plt.show()
    
def create_initial_cluster(csv_file_path: str, output_json_file: str):
    """
    Create an initial cluster JSON file with all MLS Nos in a single list.
    
    :param csv_file_path: Path to the input CSV file.
    :param output_json_file: Path to the output JSON file.
    """
    df = pd.read_csv(csv_file_path)
    
    if 'MLS No' not in df.columns:
        raise KeyError("The column 'MLS No' does not exist in the DataFrame.")
    
    # Create a single cluster with all MLS Nos
    initial_cluster = [df['MLS No'].tolist()]
    
    # Save the cluster to a JSON file
    with open(output_json_file, 'w') as file:
        json.dump(initial_cluster, file, indent=4)

def filter_clusters_by_column_values(clusters_file: str, csv_file_path: str, column_name: str, values: list, output_file: str):
    """
    Filter clusters by a list of specific column values and save the filtered clusters to a new JSON file.
    
    :param clusters_file: Path to the JSON file containing initial clusters.
    :param csv_file_path: Path to the CSV file containing property data.
    :param column_name: Name of the column to filter by.
    :param values: List of values to filter by.
    :param output_file: Path to the output JSON file for filtered clusters.
    """
    # Load initial clusters
    with open(clusters_file, 'r') as file:
        initial_clusters = json.load(file)
    
    # Load CSV data
    df = pd.read_csv(csv_file_path)
    
    # Check if the column exists
    if column_name not in df.columns:
        raise KeyError(f"The column '{column_name}' does not exist in the DataFrame.")
    
    # Create a dictionary to map MLS No to the column value
    mls_to_value = df.set_index('MLS No')[column_name].to_dict()
    
    # Filter clusters
    filtered_clusters = []
    for cluster in initial_clusters:
        filtered_cluster = [mls_no for mls_no in cluster if mls_to_value.get(mls_no) in values]
        if filtered_cluster:
            filtered_clusters.append(filtered_cluster)
    
    # Save the filtered clusters to a new JSON file
    with open(output_file, 'w') as file:
        json.dump(filtered_clusters, file, indent=4)

def limit_cluster_size(json_file_path: str, max_items: int, output_file: str):
    """
    Limit the size of each cluster to a specified maximum number of items.

    :param json_file_path: Path to the JSON file containing clusters.
    :param max_items: Maximum number of items allowed in each cluster.
    :param output_file: Path to the output JSON file for modified clusters.
    """
    import random

    with open(json_file_path, 'r') as file:
        clusters = json.load(file)

    limited_clusters = []
    for cluster in clusters:
        if len(cluster) > max_items:
            # Randomly sample max_items from the cluster
            limited_cluster = random.sample(cluster, max_items)
        else:
            limited_cluster = cluster
        limited_clusters.append(limited_cluster)

    with open(output_file, 'w') as file:
        json.dump(limited_clusters, file, indent=4)

def refine_large_clusters_with_dbscan(clusters_file: str, csv_file_path: str, max_size: int, eps: float, min_samples: int, output_file: str):
    """
    Refine large clusters by applying DBSCAN to clusters larger than a specified size.
    
    :param clusters_file: Path to the JSON file containing initial clusters.
    :param csv_file_path: Path to the CSV file containing property data with Latitude and Longitude.
    :param max_size: Maximum size of clusters to be refined.
    :param eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    :param min_samples: The number of samples in a neighborhood for a point to be considered as a core point.
    :param output_file: Path to the output JSON file for refined clusters.
    """
    with open(clusters_file, 'r') as file:
        initial_clusters = json.load(file)
    
    df = pd.read_csv(csv_file_path)
    
    refined_clusters = []
    total_clusters = len(initial_clusters)
    refined_count = 0
    
    for i, cluster in enumerate(initial_clusters):
        print(f"Processing cluster {i+1}/{total_clusters} with size {len(cluster)}")
        if len(cluster) > max_size:
            # Extract the subset of the DataFrame corresponding to the current cluster
            cluster_df = df[df['MLS No'].isin(cluster)]
            coordinates = cluster_df[['Latitude', 'Longitude']].values
            
            # Apply DBSCAN to the large cluster
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            cluster_df['SubCluster'] = dbscan.fit_predict(coordinates)
            
            # Filter out noise points (SubCluster label -1)
            sub_clusters = cluster_df[cluster_df['SubCluster'] != -1].groupby('SubCluster')['MLS No'].apply(list).tolist()
            refined_clusters.extend(sub_clusters)
            refined_count += 1
            print(f"Refined cluster {i+1} into {len(sub_clusters)} sub-clusters")
        else:
            # Keep small clusters as they are
            refined_clusters.append(cluster)
    
    print(f"Total clusters refined: {refined_count}")
    
    with open(output_file, 'w') as file:
        json.dump(refined_clusters, file, indent=4)
    print(f"Refined clusters saved to {output_file}")

import math
from sklearn.cluster import KMeans
import pandas as pd
import json

def refine_large_clusters_with_kmeans(clusters_file: str, csv_file_path: str, max_size: int, desired_size: int, output_file: str):
    """
    Refine large clusters by applying K-Means to clusters larger than a specified size.
    
    :param clusters_file: Path to the JSON file containing initial clusters.
    :param csv_file_path: Path to the CSV file containing property data with Latitude and Longitude.
    :param max_size: Maximum size of clusters to be refined.
    :param desired_size: Desired size of each sub-cluster.
    :param output_file: Path to the output JSON file for refined clusters.
    """
    with open(clusters_file, 'r') as file:
        initial_clusters = json.load(file)
    
    df = pd.read_csv(csv_file_path)
    
    refined_clusters = []
    total_clusters = len(initial_clusters)
    refined_count = 0
    total_points_before = sum(len(cluster) for cluster in initial_clusters)
    
    for i, cluster in enumerate(initial_clusters):
        print(f"Processing cluster {i+1}/{total_clusters} with size {len(cluster)}")
        if len(cluster) > max_size:
            # Calculate the number of sub-clusters
            n_clusters = math.ceil(len(cluster) / desired_size)
            print(f"  Calculated {n_clusters} sub-clusters for cluster size {len(cluster)}")
            
            # Extract the subset of the DataFrame corresponding to the current cluster
            cluster_df = df[df['MLS No'].isin(cluster)]
            coordinates = cluster_df[['Latitude', 'Longitude']].values
            
            # Apply K-Means to the large cluster
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_df['SubCluster'] = kmeans.fit_predict(coordinates)
            
            # Group by the new sub-cluster labels
            sub_clusters = cluster_df.groupby('SubCluster')['MLS No'].apply(list).tolist()
            refined_clusters.extend(sub_clusters)
            refined_count += 1
            print(f"Refined cluster {i+1} into {len(sub_clusters)} sub-clusters")
        else:
            # Keep small clusters as they are
            refined_clusters.append(cluster)
    
    total_points_after = sum(len(cluster) for cluster in refined_clusters)
    print(f"Total clusters refined: {refined_count}")
    print(f"Total points before: {total_points_before}, Total points after: {total_points_after}")
    
    with open(output_file, 'w') as file:
        json.dump(refined_clusters, file, indent=4)
    print(f"Refined clusters saved to {output_file}")

def print_cluster_sizes(json_file_path: str):
    """
    Print each cluster and its length from a JSON file containing clusters.
    
    :param json_file_path: Path to the JSON file containing clusters.
    """
    with open(json_file_path, 'r') as file:
        clusters = json.load(file)
    
    for i, cluster in enumerate(clusters):
        print(f"Cluster {i}: Length = {len(cluster)}")

def save_clusters_to_csv(clusters_file: str, csv_file_path: str, output_folder: str):
    """
    Save each cluster from a JSON file to individual CSV files.

    :param clusters_file: Path to the JSON file containing clusters.
    :param csv_file_path: Path to the CSV file containing property data.
    :param output_folder: Path to the folder where CSV files will be saved.
    """
    # Load clusters from JSON file
    with open(clusters_file, 'r') as file:
        clusters = json.load(file)
    
    # Load CSV data
    df = pd.read_csv(csv_file_path)
    
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Iterate over each cluster and save to a CSV file
    for i, cluster in enumerate(clusters, start=1):
        cluster_df = df[df['MLS No'].isin(cluster)]
        output_file = os.path.join(output_folder, f"{i}.csv")
        cluster_df.to_csv(output_file, index=False)
        print(f"Cluster {i} saved to {output_file}")

def calculate_price_spread_score(clusters_file: str, csv_file_path: str, output_file: str):
    """
    Calculate the price spread score for each cluster and save the results to a file.

    :param clusters_file: Path to the JSON file containing clusters.
    :param csv_file_path: Path to the CSV file containing property data.
    :param output_file: Path to the output file for price spread scores.
    """
    # Load clusters from JSON file
    with open(clusters_file, 'r') as file:
        clusters = json.load(file)
    
    # Load CSV data
    df = pd.read_csv(csv_file_path)

    with open(output_file, 'w') as file:
        for i, cluster in enumerate(clusters, start=1):
            score = calculate_iqr(cluster, df)
            file.write(f"Cluster {i}: {score if score is not None else 'N/A'}\n")
            print(f"Cluster {i} processed with score: {score if score is not None else 'N/A'}")

def calculate_iqr(cluster, df):
    cluster_df = df[df['MLS No'].isin(cluster)]
    if 'SP' in cluster_df.columns:
        cluster_df['SP'] = cluster_df['SP'].replace('[\$,]', '', regex=True).astype(float)
        prices = cluster_df['SP'].dropna()
        
        if len(prices) > 1:
            q75, q25 = np.percentile(prices, [75, 25])
            iqr = q75 - q25
            median = np.median(prices)
            normalized_iqr = iqr / median if median != 0 else iqr  # Avoid division by zero
            return normalized_iqr
        else:
            return None
    else:
        print("Column 'SP' not found in the DataFrame.")
        return None

def visualize_cities_with_labels(city_lat_long_file: str, visualization_file: str):
    """
    Visualize cities on a map using folium, with city names as labels.

    :param city_lat_long_file: Path to the JSON file containing city names and their latitude and longitude.
    :param visualization_file: Path to the output HTML file for visualization.
    """
    # Load city data from JSON file
    with open(city_lat_long_file, 'r') as file:
        city_lat_long = json.load(file)
    
    # Create a map centered around the average location
    avg_lat = np.mean([city['latitude'] for city in city_lat_long.values()])
    avg_long = np.mean([city['longitude'] for city in city_lat_long.values()])
    m = folium.Map(location=[avg_lat, avg_long], zoom_start=5)
    
    # Add points and labels to the map
    for city, coords in city_lat_long.items():
        folium.Marker(
            location=(coords['latitude'], coords['longitude']),
            icon=folium.DivIcon(html=f'''
                <div style="font-size: 12px; color: black; background-color: white; padding: 4px 8px; border-radius: 3px; width: auto; text-align: center;">
                    <b>{city}</b>
                </div>
            ''')
        ).add_to(m)
    
    # Save the map to an HTML file
    m.save(visualization_file)

def delete_mls_by_cities(mls_lat_long_file: str, cities: list, mls_data_file: str, output_file: str):
    """
    Delete rows from the MLS data CSV file where the MLS No belongs to the specified cities.

    :param mls_lat_long_file: Path to the CSV file containing MLS No, Latitude, and Longitude.
    :param cities: List of city names to filter out.
    :param mls_data_file: Path to the input MLS data CSV file.
    :param output_file: Path to the output CSV file with filtered data.
    """
    # Load MLS data
    mls_data_df = pd.read_csv(mls_data_file)
    
    # Check if 'City' column exists
    if 'City' not in mls_data_df.columns:
        raise KeyError("The column 'City' does not exist in the MLS data file.")
    
    # Filter MLS Nos that belong to the specified cities
    mls_to_remove = mls_data_df[mls_data_df['City'].isin(cities)]['MLS No'].tolist()
    
    # Load MLS latitude-longitude data
    mls_lat_long_df = pd.read_csv(mls_lat_long_file)
    
    # Remove rows with MLS Nos in the specified cities
    filtered_mls_lat_long_df = mls_lat_long_df[~mls_lat_long_df['MLS No'].isin(mls_to_remove)]
    
    # Save the filtered data to a new CSV file
    filtered_mls_lat_long_df.to_csv(output_file, index=False)

def get_neighborhood_details(clusters_file: str, mls_data_path: str, api_key: str, output_file: str, neighborhoods_list_file: str):
    """
    Get neighborhood details for a random address in each cluster using Google Places API and save to a JSON file.

    :param clusters_file: Path to the JSON file containing clusters.
    :param mls_data_path: Path to the CSV file containing MLS data.
    :param api_key: Google Places API key.
    :param output_file: Path to the output JSON file for neighborhood details.
    :param neighborhoods_list_file: Path to the output file for the list of neighborhood strings.
    """
    # Load clusters from JSON file
    with open(clusters_file, 'r') as file:
        clusters = json.load(file)
    
    # Load MLS data once
    df = pd.read_csv(mls_data_path, low_memory=False)
    address_dict = get_address_strings(df)
    
    neighborhood_details = {}
    neighborhoods_list = []
    
    # Process only the first 3 clusters for testing
    for cluster_index, cluster in enumerate(clusters):
        if not cluster:
            neighborhoods_list.append(None)
            continue
        
        # Select up to 5 random MLS Nos from the cluster
        random_mls_nos = np.random.choice(cluster, min(5, len(cluster)), replace=False)
        neighborhood_found = False
        
        for random_mls_no in random_mls_nos:
            address = address_dict.get(random_mls_no)
            
            if not address:
                print(f"No address found for MLS No: {random_mls_no}")
                continue
            
            # Get place details from Google Places API
            try:
                place_id = get_place_id(address, api_key)
                if place_id:
                    place_details = get_place_details(place_id, api_key)
                    neighborhood = next((comp['long_name'] for comp in place_details['address_components'] if 'neighborhood' in comp['types']), None)
                    
                    if neighborhood:
                        neighborhood_details[random_mls_no] = {
                            'neighborhood': neighborhood,
                            'place_details': place_details
                        }
                        neighborhoods_list.append(neighborhood)
                        print(f"Cluster {cluster_index}: Neighborhood for MLS No {random_mls_no} is {neighborhood}")
                        neighborhood_found = True
                        break
                    else:
                        print(f"No neighborhood found for address: {address}")
                else:
                    print(f"No place ID found for address: {address}")
            
            except Exception as e:
                print(f"Error fetching neighborhood for MLS No {random_mls_no}: {e}")
        
        if not neighborhood_found:
            neighborhoods_list.append(None)
    
    # Save the neighborhood details to a JSON file
    with open(output_file, 'w') as file:
        json.dump(neighborhood_details, file, indent=4)
    
    # Save the list of neighborhood strings to a separate file
    with open(neighborhoods_list_file, 'w') as file:
        json.dump(neighborhoods_list, file, indent=4)

def get_address_strings(df: pd.DataFrame) -> dict:
    """
    Get a dictionary of address strings for all MLS Nos in the dataset.
    
    :param df: DataFrame containing MLS data.
    :return: Dictionary with MLS No as keys and address strings as values.
    """
    address_dict = {}
    for _, row in df.iterrows():
        unit = row['Unit'] if pd.notna(row['Unit']) else ''
        address_string = f"{row['Address']}, {unit}, {row['City']}, CA".replace(', ,', ',')
        address_dict[row['MLS No']] = address_string
    return address_dict

def get_place_id(address: str, api_key: str) -> str:
    """
    Get place ID for a given address using Google Places API.

    :param address: The address to search for.
    :param api_key: Your Google Places API key.
    :return: Place ID if found, else None.
    """
    base_url = "https://maps.googleapis.com/maps/api/place/findplacefromtext/json"
    params = {
        'input': address,
        'inputtype': 'textquery',
        'fields': 'place_id',
        'key': api_key
    }
    response = requests.get(base_url, params=params)
    response_data = response.json()
    
    if response_data['status'] == 'OK' and response_data['candidates']:
        return response_data['candidates'][0]['place_id']
    else:
        print(f"Error fetching place ID: {response_data.get('status', 'Unknown error')}")
        return None

def get_place_details(place_id: str, api_key: str) -> dict:
    """
    Get detailed place information using Google Places API.

    :param place_id: The place ID to get details for.
    :param api_key: Your Google Places API key.
    :return: Dictionary with place details.
    """
    base_url = "https://maps.googleapis.com/maps/api/place/details/json"
    params = {
        'place_id': place_id,
        'fields': 'address_components',
        'key': api_key
    }
    response = requests.get(base_url, params=params)
    response_data = response.json()
    
    if response_data['status'] == 'OK':
        return response_data['result']
    else:
        print(f"Error fetching place details: {response_data.get('status', 'Unknown error')}")
        if 'error_message' in response_data:
            print(f"Error message: {response_data['error_message']}")
        raise Exception(f"Error fetching place details: {response_data.get('status', 'Unknown error')}")

# Example usage:
# address = get_address_string_given_mls_no('123456', 'combined_data.csv')
# print(address)

# Example usage:
# cluster_by_city('data/v3_filtered_cities_ca_only.csv', 'city_clusters.txt')
# Example usage:
# pick_random_rows('combined_data.csv')
# analyze_column('combined_data.csv', 150)
# filter_rows_by_value('combined_data.csv', 150, 'CA', 'ca_only.csv')
# analyze_mls_groups('mls_no_to_study.txt')
# find_duplicates('data/v1_ca_only.csv', [3, 4, 5], 'duplicate_indices.txt')
# check_absolute_duplicates('data/v1_ca_only.csv', 'duplicate_indices.txt')
# sample_duplicates('data/v1_ca_only.csv', 'duplicate_indices_hashmap.txt', 'sampled_duplicates.csv')
# dedupe_by_closing_date('data/v1_ca_only.csv', 'v2_deduped_ca_only.csv')
# count_rows_before_year('data/v3_filtered_cities_ca_only.csv', 2023)
# count_unique_cities('data/v3_filtered_cities_ca_only.csv')
# filter_cities_by_row_count('data/v3_filtered_cities_ca_only.csv', 100, 'data/v4_filtered_cities_ca_only.csv')
# save_city_counts('data/v3_filtered_cities_ca_only.csv', 'city_counts.csv')
# cluster_by_city('data/v3_filtered_cities_ca_only.csv', 'city_clusters.txt')
# print_column_headers('data/v3_filtered_cities_ca_only.csv', [10,11,12,16,18])


# city_clusters = read_city_clusters('city_clusters.json')
# relevant_columns = ['SqFt', 'BR', 'Bth', 'YrBlt', 'Lot SqFt']
# df = load_relevant_columns('data/v3_filtered_cities_ca_only.csv', relevant_columns)
# normalized_data = normalize_data(df)

# # Determine optimal number of clusters
# wcss, silhouette_scores = determine_optimal_clusters(normalized_data)


# city_clusters = read_city_clusters('city_clusters.json')

# # Select one cluster to work with
# one_cluster_indices = city_clusters[0]

# relevant_columns = ['SqFt', 'BR', 'Bth', 'YrBlt', 'Lot SqFt']
# df = load_relevant_columns('data/v3_filtered_cities_ca_only.csv', relevant_columns, one_cluster_indices)
# normalized_data = normalize_data(df)

# # Determine optimal number of clusters
# wcss, silhouette_scores = determine_optimal_clusters(normalized_data)

# # Print WCSS and Silhouette Scores for inspection
# print("WCSS:", wcss)
# print("Silhouette Scores:", silhouette_scores)



# file_path = 'mls_no_to_study.txt'
# # Its a list of lists. print total count of items in each sublist all added up
# analyze_mls_groups(file_path)

# file_path = 'mls_no_to_study.txt'
# mls_data_path = 'combined_data.csv'
# api_key = 

# save_mls_lat_long(file_path, mls_data_path, api_key, 'mls_lat_long.csv')

# dedupe_by_first_column('mls_lat_long.csv', 'mls_lat_long_deduped.csv')

# analyze_cluster_sizes('city_clusters.json')
# cluster_by_city_and_bt('cleaned_data.csv', 'city_bt_clusters.json')
# analyze_cluster_sizes('city_bt_clusters.json')

# count_unique_elements_in_clusters('city_bt_clusters.json')
# analyze_cluster_sizes('city_bt_clusters.json')

# delete_small_clusters('city_bt_clusters.json', 30, 'city_bt_clusters_filtered.json')

# count_unique_elements_in_clusters('city_bt_clusters_filtered.json')
# analyze_cluster_sizes('city_bt_clusters_filtered.json')

# city_bt_clusters.json

# cluster_by_sqft_within_percentage('city_bt_clusters_filtered.json', 'cleaned_data.csv', 5, 'city_bt_sqft_clusters_2.json')



# count_unique_elements_in_clusters('city_bt_sqft_clusters_2.json')
# analyze_cluster_sizes('city_bt_sqft_clusters_2.json')

# delete_small_clusters('city_bt_sqft_clusters_2.json', 50, 'city_bt_sqft_clusters_filtered_2.json')
# count_unique_elements_in_clusters('city_bt_sqft_clusters_filtered_2.json')
# analyze_cluster_sizes('city_bt_sqft_clusters_filtered_2.json')

# cluster_by_bed_bath_ratio('city_bt_sqft_clusters_filtered_2.json', 'cleaned_data.csv', 'city_bt_sqft_br_bth_clusters_2.jsonl')

# count_unique_elements_in_clusters('city_bt_sqft_br_bth_clusters_2.json')
# analyze_cluster_sizes('city_bt_sqft_br_bth_clusters_2.json')
# delete_small_clusters('city_bt_sqft_br_bth_clusters_2.json', 30, 'city_bt_sqft_br_bth_clusters_filtered_2.json')
# count_unique_elements_in_clusters('city_bt_sqft_br_bth_clusters_filtered_2.json')
# analyze_cluster_sizes('city_bt_sqft_br_bth_clusters_filtered_2.json')

# calculate_price_spread('city_bt_sqft_br_bth_clusters_filtered_2.json', 'cleaned_data.csv', 'price_spreads.txt')
# analyze_price_spreads('price_spreads.txt')

# sort_clusters_by_price_spread('price_spreads.txt', 'city_bt_sqft_br_bth_clusters_filtered_2.json', 'clusters_sorted_by_price_spread.json')
# count_unique_elements_in_clusters('clusters_sorted_by_price_spread.json')
# analyze_cluster_sizes('clusters_sorted_by_price_spread.json')
# create_clusters_with_unique_elements('clusters_sorted_by_price_spread.json', 30000, 'clusters_with_unique_elements.json')

# count_unique_elements_in_clusters('clusters_with_unique_elements.json')
# analyze_cluster_sizes('clusters_with_unique_elements.json')

# google_places_api_key = os.getenv('GOOGLE_PLACES_API_KEY')
# print(google_places_api_key)
# save_mls_lat_long('clusters_with_unique_elements.json', 'combined_data.csv', google_places_api_key, 'mls_lat_long.csv')

# flatten_clusters('clusters_with_unique_elements.json', 'flattened_cluster.json')

# geographical_clustering('mls_lat_long.csv', 5, 'geo_clusters.json')
# visualize_clusters('mls_lat_long.csv', 'geo_clusters.json')
# remove_mls_below_latitude('geo_clusters.json', 'mls_lat_long.csv', 36.20552, 'geo_clusters_filtered.json')
# visualize_clusters('mls_lat_long.csv', 'geo_clusters_filtered.json')

# count_unique_elements_in_clusters('geo_clusters_filtered.json')

# filter_mls_by_latitude('mls_lat_long.csv', 36.20552, 'mls_lat_long_north.csv')
# eps = determine_eps('mls_lat_long_north.csv', 50)
# print(eps)
# geographical_clustering_with_dbscan('mls_lat_long_north.csv', 0.001, 50, 'geo_clusters_filtered_north_dbscan.json')

# count_unique_elements_in_clusters('geo_clusters_filtered_north_dbscan.json')
# analyze_cluster_sizes('geo_clusters_filtered_north_dbscan.json')

# visualize_clusters_with_circles('mls_lat_long_north.csv', 'geo_clusters_filtered_north_dbscan.json', 'geo_clusters_filtered_north_dbscan_visualization.html')

# delete_small_clusters('geo_clusters_filtered_north.json', 20, 'geo_clusters_filtered_north_large.json')

# count_unique_elements_in_clusters('geo_clusters_filtered_north_large.json')
# analyze_cluster_sizes('geo_clusters_filtered_north_large.json')

# cluster_by_bt('geo_clusters_filtered_north_large.json', 'cleaned_data.csv', 'bt_clusters.json')

# count_unique_elements_in_clusters('bt_clusters.json')
# analyze_cluster_sizes('bt_clusters.json')

# delete_small_clusters('bt_clusters.json', 20, 'bt_clusters_filtered.json')

# count_unique_elements_in_clusters('bt_clusters_filtered.json')
# analyze_cluster_sizes('bt_clusters_filtered.json')

# count_unique_elements_in_clusters('bt_clusters_filtered.json')
# analyze_cluster_sizes('bt_clusters_filtered.json')
# cluster_by_bed_bath_ratio('bt_clusters_filtered.json', 'cleaned_data.csv', 'bt_br_bth_clusters.json')
# count_unique_elements_in_clusters('bt_br_bth_clusters.json')
# analyze_cluster_sizes('bt_br_bth_clusters.json')    

# delete_small_clusters('bt_br_bth_clusters.json', 15, 'bt_br_bth_clusters_filtered.json')

# count_unique_elements_in_clusters('bt_br_bth_clusters_filtered.json')
# analyze_cluster_sizes('bt_br_bth_clusters_filtered.json')

# visualize_clusters_with_circles('mls_lat_long_north.csv', 'bt_br_bth_clusters_filtered.json', 'bt_br_bth_clusters_visualization.html')


# calculate_price_spread('bt_br_bth_clusters_filtered.json', 'cleaned_data.csv', 'price_spreads_2.txt')
# analyze_price_spreads('price_spreads_2.txt')

# sort_clusters_by_price_spread('price_spreads_2.txt', 'bt_br_bth_clusters_filtered.json', 'clusters_sorted_by_price_spread_2.json')
# count_unique_elements_in_clusters('clusters_sorted_by_price_spread_2.json')
# analyze_cluster_sizes('clusters_sorted_by_price_spread_2.json')

# get_top_clusters('clusters_sorted_by_price_spread_2.json', 20, 'top_20_clusters.json')
# visualize_top_clusters('mls_lat_long.csv', 'top_20_clusters.json', 'top_20_clusters_map.html')

# values = get_unique_values('cleaned_data.csv', 'BT')
# print(values)


# create_initial_cluster('cleaned_data.csv', 'initial_cluster.json')
# filter_clusters_by_column_value('initial_cluster.json', 'cleaned_data.csv', 'BT', 'DE', 'only_de_mls_no_cluster.json')

# cluster_by_city('only_de_mls_no_cluster.json', 'cleaned_data.csv', 'de_city_clusters.json')

# count_unique_elements_in_clusters('de_city_clusters.json')
# analyze_cluster_sizes('de_city_clusters.json')

# delete_small_clusters('de_city_clusters.json', 50, 'de_city_clusters_filtered.json')

# count_unique_elements_in_clusters('de_city_clusters_filtered.json')
# analyze_cluster_sizes('de_city_clusters_filtered.json')

# cities = ['ACTON',
#  'ADELANTO',
#  'AGOURA HIL',
#  'ALAMEDA',
#  'ALAMO',
#  'ALBANY',
#  'ALHAMBRA',
#  'ALISO VIEJ',
#  'ALPINE',
#  'ALTA LOMA',
#  'ALTADENA',
#  'AMERICNYON',
#  'ANAHEIM',
#  'ANTELOPE',
#  'ANTIOCH',
#  'ANZA',
#  'APPLE VALL',
#  'APTOS',
#  'ARCADIA',
#  'ARLETA',
#  'ARROYO GRA',
#  'ATASCADERO',
#  'ATHERTON',
#  'ATWATER',
#  'AUBURN',
#  'AZUSA',
#  'BAKERSFIEL',
#  'BALDWIN PA',
#  'BANNING',
#  'BARSTOW',
#  'BAY POINT',
#  'BEAUMONT',
#  'BELLFLOWER',
#  'BELMONT',
#  'BENICIA',
#  'BERKELEY',
#  'BEVERLY HI',
#  'BIG BEAR C',
#  'BIG BEAR L',
#  'BLOOMINGTO',
#  'BLYTHE',
#  'BONITA',
#  'BORREGO SP',
#  'BOULDERCRK',
#  'BREA',
#  'BRENTWOOD',
#  'BUENA PARK',
#  'BURBANK',
#  'BURLINGAME',
#  'CA CITY',
#  'CALABASAS',
#  'CALIMESA',
#  'CAMARILLO',
#  'CAMBRIA',
#  'CAMERONPAK',
#  'CAMINO',
#  'CAMPBELL',
#  'CANOGA PAR',
#  'CANYON COU',
#  'CANYON LAK',
#  'CAPITOLA',
#  'CARLSBAD',
#  'CARMEL',
#  'CARMEL VAL',
#  'CARMIC',
#  'CARSON',
#  'CASTAIC',
#  'CASTROVAEY',
#  'CATHEDRAL',
#  'CERES',
#  'CERRITOS',
#  'CHATSWORTH',
#  'CHERRY VAL',
#  'CHICO',
#  'CHINO',
#  'CHINO HILL',
#  'CHOWCHILLA',
#  'CHULA VIST',
#  'CITRUS HEI',
#  'CLAREMONT',
#  'CLAYTON',
#  'CLEARLAKE',
#  'CLEARLOAKS',
#  'CLOVERDALE',
#  'CLOVIS',
#  'COARSEGOLD',
#  'COLFAX',
#  'COLTON',
#  'COMPTON',
#  'CONCORD',
#  'COPPEROPOL',
#  'CORNING',
#  'CORONA',
#  'CORONA DEL',
#  'CORTEMAERA',
#  'COSTA MESA',
#  'COTATI',
#  'COVINA',
#  'CRESTLINE',
#  'CULVER CIT',
#  'CUPERTINO',
#  'CYPRESS',
#  'DALY CITY',
#  'DANA POINT',
#  'DANVILLE',
#  'DAVIS',
#  'DEL MAR',
#  'DESERT HOT',
#  'DIAMONDBAR',
#  'DISCOV BAY',
#  'DIXON',
#  'DOWNEY',
#  'DUARTE',
#  'DUBLIN',
#  'EASTPAALTO',
#  'EASTVALE',
#  'EL CAJON',
#  'EL CERRITO',
#  'EL MONTE',
#  'EL SEGUNDO',
#  'ELDO HILLS',
#  'ELK GROVE',
#  'ELSOBRANTE',
#  'EMERYVILLE',
#  'ENCINITAS',
#  'ENCINO',
#  'ESCONDIDO',
#  'FAIR OAKS',
#  'FAIRFIELD',
#  'FALLBROOK',
#  'FILLMORE',
#  'FOLSOM',
#  'FONTANA',
#  'FORESTHILL',
#  'FORT BRAGG',
#  'FOSTERCITY',
#  'FOUNTAIN V',
#  'FRAZIER PA',
#  'FREMONT',
#  'FRESNO',
#  'FULLERTON',
#  'GALT',
#  'GARDENA',
#  'GARDENGRVE',
#  'GILROY',
#  'GLENDALE',
#  'GLENDORA',
#  'GOLD RIVER',
#  'GRANADA HI',
#  'GRAND TERR',
#  'GRANITE BA',
#  'GRASSVALEY',
#  'GREENBRAE',
#  'GRIDEY',
#  'GROVER BEA',
#  'GUERNVILLE',
#  'HACIENDA H',
#  'HALFMO BAY',
#  'HARBOR CIT',
#  'HAWTHORNE',
#  'HAYWARD',
#  'HEALDSBURG',
#  'HELENDALE',
#  'HEMET',
#  'HERCULES',
#  'HERMOSA BE',
#  'HESPERIA',
#  'HIDVALAKE',
#  'HIGHLAND',
#  'HILLSBOROU',
#  'HOLLISTER',
#  'HOMELAND',
#  'HUGHSON',
#  'HUNTINGTNB',
#  'HUNTINGTNP',
#  'INDIO',
#  'INGLEWOOD',
#  'IONE',
#  'IRVINE',
#  'JACKSON',
#  'JOSHUA TRE',
#  'JURUPAVALL',
#  'KELSEYVILL',
#  'KING CITY',
#  'LA CANADAF',
#  'LA CRESCEN',
#  'LA HABRA',
#  'LA JOLLA',
#  'LA MESA',
#  'LA MIRADA',
#  'LA PALMA',
#  'LA PUENTE',
#  'LA QUINTA',
#  'LA VERNE',
#  'LADERA RAN',
#  'LAFAYETTE',
#  'LAGUNA BEA',
#  'LAGUNA HIL',
#  'LAGUNA NIG',
#  'LAGUNA WOO',
#  'LAKE ARROW',
#  'LAKE ELSIN',
#  'LAKE FORES',
#  'LAKEPORT',
#  'LAKESIDE',
#  'LAKEWOOD',
#  'LANCASTER',
#  'LARKSPUR',
#  'LATHROP',
#  'LAWNDALE',
#  'LEMON GROV',
#  'LINCOLN',
#  'LINDA',
#  'LITTLEROCK',
#  'LIVE OAK',
#  'LIVERMORE',
#  'LODI',
#  'LOMA LINDA',
#  'LOMITA',
#  'LOMPOC',
#  'LONG BEACH',
#  'LOOMIS',
#  'LOS ALAMIT',
#  'LOS ALTOS',
#  'LOS ANGELE',
#  'LOS BANOS',
#  'LOS GATOS',
#  'LOS OSOS',
#  'LUCERNE',
#  'LUCERNE VA',
#  'LYNWOOD',
#  'MADERA',
#  'MAGALIA',
#  'MALIBU',
#  'MANHATTAN',
#  'MANTECA',
#  'MARINA',
#  'MARINA DEL',
#  'MARIPOSA',
#  'MARTINEZ',
#  'MARYSVILLE',
#  'MENIFEE',
#  'MENLO PARK',
#  'MERCED',
#  'MILLBRAE',
#  'MILLVALLEY',
#  'MILPITAS',
#  'MISSION VI',
#  'MODESTO',
#  'MONROVIA',
#  'MONTCLAIR',
#  'MONTEBELLO',
#  'MONTEREY',
#  'MONTEREY P',
#  'MOORPARK',
#  'MORAGA',
#  'MORENO VAL',
#  'MORGANHILL',
#  'MORRO BAY',
#  'MOUNTAVIEW',
#  'MOUNTHOUSE',
#  'MURRIETA',
#  'NAPA',
#  'NATIONAL C',
#  'NEVADACITY',
#  'NEWARK',
#  'NEWBURY PA',
#  'NEWHALL',
#  'NEWMAN',
#  'NEWPORT BE',
#  'NEWPORT CO',
#  'NIPOMO',
#  'NORCO',
#  'NORTH HIGH',
#  'NORTH HILL',
#  'NORTH HOLL',
#  'NORTHRIDGE',
#  'NORWALK',
#  'NOVATO',
#  'OAKDALE',
#  'OAKHURST',
#  'OAKLAND',
#  'OAKLEY',
#  'OCEANSIDE',
#  'OJAI',
#  'OLIVEHURST',
#  'ONTARIO',
#  'ORANGE',
#  'ORANGEVALE',
#  'ORINDA',
#  'ORLAND',
#  'OROVILLE',
#  'OXNARD',
#  'Other',
#  'PACIFIC PA',
#  'PACIFICA',
#  'PACIFICGRE',
#  'PACOIMA',
#  'PALM DESER',
#  'PALM SPRIN',
#  'PALMDALE',
#  'PALO ALTO',
#  'PALOS VERD',
#  'PANORAMA C',
#  'PARADISE',
#  'PARAMOUNT',
#  'PASADENA',
#  'PASO ROBLE',
#  'PATTERSON',
#  'PEBBLEBECH',
#  'PENN VALLE',
#  'PERRIS',
#  'PETALUMA',
#  'PHELAN',
#  'PICO RIVER',
#  'PIEDMONT',
#  'PINEMTNCLU',
#  'PINOLE',
#  'PINON HILL',
#  'PIONEER',
#  'PISMO BEAC',
#  'PITTSBURG',
#  'PLACENTIA',
#  'PLACERVILL',
#  'PLAYA DEL',
#  'PLEASAHILL',
#  'PLEASANTON',
#  'PLUMAS LAK',
#  'POLLOCKPIS',
#  'POMONA',
#  'PORT HUENE',
#  'PORTER RAN',
#  'POWAY',
#  'RAMONA',
#  'RANCHO COR',
#  'RANCHO CUC',
#  'RANCHO MIR',
#  'RANCHO MUR',
#  'RANCHO PAL',
#  'RANCHO SAN',
#  'RANCHOMISS',
#  'RANCHOSANT',
#  'RED BLUFF',
#  'REDLANDS',
#  'REDONDO BE',
#  'REDWOODCIY',
#  'RESEDA',
#  'RIALTO',
#  'RICHMOND',
#  'RIO LINDA',
#  'RIO VISTA',
#  'RIPON',
#  'RIVERBANK',
#  'RIVERSIDE',
#  'ROCKLIN',
#  'ROHNERTPAK',
#  'ROLLINGHIE',
#  'ROSAMOND',
#  'ROSEMEAD',
#  'ROSEVILLE',
#  'ROWLAND HE',
#  'RUNNING SP',
#  'RossmoorSC',
#  'SACRAMENTO',
#  'SALIDA',
#  'SALINAS',
#  'SAN BERNAR',
#  'SAN BRUNO',
#  'SAN CARLOS',
#  'SAN CLEMEN',
#  'SAN DIEGO',
#  'SAN DIMAS',
#  'SAN FERNAN',
#  'SAN GABRIE',
#  'SAN JACINT',
#  'SAN JOSE',
#  'SAN JUAN C',
#  'SAN MARCOS',
#  'SAN MARINO',
#  'SAN MATEO',
#  'SAN PABLO',
#  'SAN PEDRO',
#  'SAN RAFAEL',
#  'SAN RAMON',
#  'SANANSELMO',
#  'SANFRCISCO',
#  'SANLEANDRO',
#  'SANLORENZO',
#  'SANLUBISPO',
#  'SANTA ANA',
#  'SANTA BARB',
#  'SANTA CLAR',
#  'SANTA CRUZ',
#  'SANTA MARI',
#  'SANTA MONI',
#  'SANTA PAUL',
#  'SANTA ROSA',
#  'SANTACLARA',
#  'SANTEE',
#  'SARATOGA',
#  'SAUGUS',
#  'SAUSALITO',
#  'SCOTTSVAEY',
#  'SEAL BEACH',
#  'SEASIDE',
#  'SEBASTOPOL',
#  'SHERMAN OA',
#  'SHINGLESPG',
#  'SIERRA MAD',
#  'SIGNAL HIL',
#  'SIMI VALLE',
#  'SOLANA BEA',
#  'SONOMA',
#  'SONORA',
#  'SOUTH GATE',
#  'SOUTH PASA',
#  'SOUTH SAN',
#  'SPRING VAL',
#  'ST. HELENA',
#  'STANTON',
#  'STEVENSON',
#  'STOCKTON',
#  'STUDIO CIT',
#  'SUISUNCITY',
#  'SUN CITY',
#  'SUN VALLEY',
#  'SUNLAND',
#  'SUNNYVALE',
#  'SYLMAR',
#  'TARZANA',
#  'TEHACHAPI',
#  'TEMECULA',
#  'TEMPLE CIT',
#  'TEMPLETON',
#  'THE SEA RA',
#  'THOUSAND O',
#  'TIBURON',
#  'TOLUCA LAK',
#  'TOPANGA',
#  'TORRANCE',
#  'TRABUCO CA',
#  'TRACY',
#  'TUJUNGA',
#  'TURLOCK',
#  'TUSTIN',
#  'TWENTYNINE',
#  'TWIN PEAKS',
#  'UKIAH',
#  'UNION CITY',
#  'UPLAND',
#  'VACAVILLE',
#  'VALENCIA',
#  'VALLEJO',
#  'VALLEY CEN',
#  'VALLEY VIL',
#  'VALLEYSPGS',
#  'VAN NUYS',
#  'VENICE',
#  'VENTURA',
#  'VICTORVILL',
#  'VISALIA',
#  'VISTA',
#  'WALNUT',
#  'WALNUTCREK',
#  'WATSONVILL',
#  'WEST HILLS',
#  'WEST HOLLY',
#  'WESTCOVINA',
#  'WESTLAKE V',
#  'WESTMINSTE',
#  'WESTSAENTO',
#  'WHITTIER',
#  'WILDOMAR',
#  'WILLITS',
#  'WILLOWS',
#  'WILMINGTON',
#  'WINCHESTER',
#  'WINDSOR',
#  'WINNETKA',
#  'WINTERS',
#  'WOODLAND',
#  'WOODLAND H',
#  'WOODSIDE',
#  'WRIGHTWOOD',
#  'YORBA LIND',
#  'YUBA CITY',
#  'YUCAIPA',
#  'YUCCA VALL']

# get_city_lat_long(cities, 'city_lat_long.json', google_places_api_key)

# get_cities_with_latitude_above('city_lat_long.json', 36.20552)

# create_initial_cluster('cleaned_data.csv', 'initial_cluster.json')
# filter_clusters_by_column_values('initial_cluster.json', 'cleaned_data.csv', 'BT', ['DE'], 'only_detached_cluster.json')
# filter_clusters_by_column_values('only_detached_cluster.json', 'cleaned_data.csv', 'State', ['CA'], 'only_detached_ca_cluster.json')

# cities_north = ['ACTON', 'ALAMEDA', 'APTOS', 'ATHERTON', 'ATWATER', 'BALDWIN PA', 'BAY POINT', 'BENICIA', 'BERKELEY', 'BLOOMINGTO', 'BOULDERCRK', 'BRENTWOOD', 'BURLINGAME', 'CANYON COU', 'CAPITOLA', 'CARMEL', 'CARMEL VAL', 'CASTROVAEY', 'CHOWCHILLA', 'CITRUS HEI', 'CLEARLAKE', 'CLOVERDALE', 'COARSEGOLD', 'COLFAX', 'COPPEROPOL', 'CORTEMAERA', 'COTATI', 'CUPERTINO', 'DALY CITY', 'DANVILLE', 'DAVIS', 'DISCOV BAY', 'DUBLIN', 'EL CERRITO', 'ELDO HILLS', 'ELK GROVE', 'ELSOBRANTE', 'EMERYVILLE', 'FAIR OAKS', 'FAIRFIELD', 'FOLSOM', 'FORESTHILL', 'FORT BRAGG', 'FOSTERCITY', 'FRAZIER PA', 'FREMONT', 'FRESNO', 'GALT', 'GILROY', 'GOLD RIVER', 'GRANITE BA', 'GRASSVALEY', 'GREENBRAE', 'GUERNVILLE', 'HALFMO BAY', 'HAYWARD', 'HEALDSBURG', 'HILLSBOROU', 'HUGHSON', 'KELSEYVILL', 'KING CITY', 'LA CRESCEN', 'LAKE FORES', 'LAKEPORT', 'LANCASTER', 'LATHROP', 'LIVERMORE', 'LODI', 'LOS ALTOS', 'LOS BANOS', 'LUCERNE', 'LUCERNE VA', 'MAGALIA', 'MANHATTAN', 'MANTECA', 'MARIPOSA', 'MARYSVILLE', 'MENLO PARK', 'MERCED', 'MILLBRAE', 'MILLVALLEY', 'MILPITAS', 'MODESTO', 'MONTCLAIR', 'MONTEREY', 'MORAGA', 'MORGANHILL', 'MOUNTAVIEW', 'NEVADACITY', 'NEWARK', 'NEWBURY PA', 'NORWALK', 'NOVATO', 'OAKDALE', 'OAKHURST', 'OAKLAND', 'OLIVEHURST', 'ONTARIO', 'ORANGEVALE', 'ORINDA', 'ORLAND', 'OROVILLE', 'PALO ALTO', 'PENN VALLE', 'PETALUMA', 'PINOLE', 'PITTSBURG', 'PLACERVILL', 'PLEASAHILL', 'PLEASANTON', 'PLUMAS LAK', 'RANCHO COR', 'RANCHO MUR', 'RED BLUFF', 'REDWOODCIY', 'RICHMOND', 'RIO LINDA', 'RIO VISTA', 'RIPON', 'ROCKLIN', 'ROHNERTPAK', 'ROSEVILLE', 'SACRAMENTO', 'SALIDA', 'SALINAS', 'SAN BRUNO', 'SAN CARLOS', 'SAN JOSE', 'SAN MARINO', 'SAN MATEO', 'SAN PABLO', 'SAN RAFAEL', 'SAN RAMON', 'SANANSELMO', 'SANFRCISCO', 'SANLEANDRO', 'SANTA CLAR', 'SANTA CRUZ', 'SANTA ROSA', 'SANTACLARA', 'SARATOGA', 'SAUGUS', 'SAUSALITO', 'SEBASTOPOL', 'SONOMA', 'STOCKTON', 'SUISUNCITY', 'SUN VALLEY', 'SUNNYVALE', 'TIBURON', 'TRACY', 'TURLOCK', 'UKIAH', 'UNION CITY', 'VACAVILLE', 'VALENCIA', 'VALLEJO', 'VENICE', 'VISALIA', 'WALNUTCREK', 'WATSONVILL', 'WESTMINSTE', 'WILLITS', 'WINNETKA', 'WINTERS', 'YUBA CITY']

# filter_clusters_by_column_values('only_detached_ca_cluster.json', 'cleaned_data.csv', 'City', cities_north, 'only_detached_north_ca_cluster.json')

# count_unique_elements_in_clusters('only_detached_north_ca_cluster.json')
# analyze_cluster_sizes('only_detached_north_ca_cluster.json')

# cluster_by_city('only_detached_north_ca_cluster.json', 'cleaned_data.csv', 'only_detached_north_ca_grouped_by_city_cluster.json')

# count_unique_elements_in_clusters('only_detached_north_ca_grouped_by_city_cluster.json')
# analyze_cluster_sizes('only_detached_north_ca_grouped_by_city_cluster.json')

# delete_small_clusters('only_detached_north_ca_grouped_by_city_cluster.json', 50, 'only_detached_north_ca_grouped_by_city_cluster_filtered.json')
# get_value_counts_in_clusters('only_detached_north_ca_grouped_by_city_cluster_filtered.json', 'cleaned_data.csv', 'City')



# extract_lot_sqft_from_clusters('only_detached_north_ca_grouped_by_city_cluster_filtered.json', 'cleaned_data.csv', 'mls_lot_sqft.json')



# count_unique_elements_in_clusters('only_detached_north_ca_grouped_by_city_cluster_filtered.json')
# analyze_cluster_sizes('only_detached_north_ca_grouped_by_city_cluster_filtered.json')

# cluster_by_sqft_within_percentage('only_detached_north_ca_grouped_by_city_cluster_filtered.json', 'cleaned_data.csv', 10, 'only_detached_north_ca_grouped_by_city_cluster_filtered_sqft.json')

# count_unique_elements_in_clusters('only_detached_north_ca_grouped_by_city_cluster_filtered_sqft.json')
# analyze_cluster_sizes('only_detached_north_ca_grouped_by_city_cluster_filtered_sqft.json')

# delete_small_clusters('only_detached_north_ca_grouped_by_city_cluster_filtered_sqft.json', 30, 'only_detached_north_ca_grouped_by_city_cluster_filtered_sqft_filtered.json')

# count_unique_elements_in_clusters('only_detached_north_ca_grouped_by_city_cluster_filtered_sqft_filtered.json')
# analyze_cluster_sizes('only_detached_north_ca_grouped_by_city_cluster_filtered_sqft_filtered.json')

# calculate_price_spread('only_detached_north_ca_grouped_by_city_cluster_filtered_sqft_filtered.json', 'cleaned_data.csv', 'price_spreads.txt')
# analyze_price_spreads('price_spreads.txt')

# get_top_clusters('price_spreads.txt', 20, 'top_20_clusters.json')

# limit_cluster_size('only_detached_north_ca_grouped_by_city_cluster_filtered_sqft_filtered.json', 1000, 'first_pass_clusters.json')

# sort_clusters_by_price_spread('price_spreads.txt', 'first_pass_clusters.json', 'first_pass_clusters_sorted_by_price_spread.json')
# get_top_clusters('first_pass_clusters_sorted_by_price_spread.json', 50, 'top_20_clusters_first_pass.json')

# count_unique_elements_in_clusters('top_20_clusters_first_pass.json')
# analyze_cluster_sizes('top_20_clusters_first_pass.json')

# visualize_top_clusters('mls_lat_long.csv', 'top_20_clusters.json', 'top_20_clusters_map.html')

# save_mls_lat_long('top_50_clusters_first_pass.json', 'cleaned_data.csv', google_places_api_key, 'mls_lat_long.csv')


# mls_nos = get_column_values('mls_lat_long_north.csv', 0)
# create_initial_cluster_from_list(mls_nos, 'initial_cluster.json')


# flatten_clusters('top_50_clusters_first_pass.json', 'flattened_cluster.json')

# filter_mls_by_latitude('mls_lat_long.csv', 36.20552, 'mls_lat_long_north.csv')

# eps = determine_eps('mls_lat_long_north.csv', 50)
# print(eps)
# geographical_clustering_with_dbscan('mls_lat_long_north.csv', 0.01, 50, 'geo_clusters_filtered_north_dbscan.json')

# count_unique_elements_in_clusters('geo_clusters_filtered_north_dbscan.json')
# analyze_cluster_sizes('geo_clusters_filtered_north_dbscan.json')

# print_cluster_sizes('geo_clusters_filtered_north_dbscan.json')

# visualize_clusters_with_unique_colors('mls_lat_long_north.csv', 'geo_clusters.json', 'geo_clusters.html')

# refine_large_clusters_with_dbscan('geo_clusters_filtered_north_dbscan.json', 'mls_lat_long_north.csv', 150, 0.001, 30, 'geo_clusters_filtered_north_dbscan_refined.json')
# refine_large_clusters_with_kmeans('geo_clusters.json', 'mls_lat_long_north.csv', 150, 60, 'geo_clusters_refined.json')

# count_unique_elements_in_clusters('geo_clusters_filtered_north_dbscan_refined.json')
# analyze_cluster_sizes('geo_clusters_filtered_north_dbscan_refined.json')

# visualize_clusters_with_unique_colors('mls_lat_long_north.csv', 'geo_clusters_refined.json', 'geo_clusters_refined.html')

# count_unique_elements_in_clusters('geo_clusters_refined.json')
# analyze_cluster_sizes('geo_clusters_refined.json')

# delete_small_clusters('geo_clusters_refined.json', 30, 'geo_clusters_refined.json_filtered.json')

# count_unique_elements_in_clusters('geo_clusters_refined.json_filtered.json')
# analyze_cluster_sizes('geo_clusters_refined.json_filtered.json')

# cluster_by_sqft_within_percentage('geo_clusters_refined.json', 'cleaned_data.csv', 15, 'geo_clusters_broken_by_lotsqft.json')

# count_unique_elements_in_clusters('geo_clusters_broken_by_lotsqft.json')
# analyze_cluster_sizes('geo_clusters_broken_by_lotsqft.json')

# delete_small_clusters('geo_clusters_broken_by_lotsqft.json', 20, 'geo_clusters_broken_by_lotsqft_filtered.json')

# count_unique_elements_in_clusters('geo_clusters_broken_by_lotsqft_filtered.json')
# analyze_cluster_sizes('geo_clusters_broken_by_lotsqft_filtered.json')




# calculate_price_spread('geo_clusters_broken_by_lotsqft_filtered.json', 'cleaned_data.csv', 'price_spreads.txt')
# analyze_price_spreads('price_spreads.txt')

# sort_clusters_by_price_spread('price_spreads.txt', 'geo_clusters_broken_by_lotsqft_filtered.json', 'clusters_sorted_by_price_spread.json')
# count_unique_elements_in_clusters('clusters_sorted_by_price_spread.json')
# analyze_cluster_sizes('clusters_sorted_by_price_spread.json')

# get_top_clusters('clusters_sorted_by_price_spread.json', 20, 'top_20_clusters.json')
# calculate_price_spread_score('top_20_clusters.json', 'cleaned_data.csv', 'results/leaderboard.txt')
# calculate_price_spread('top_20_clusters.json', 'cleaned_data.csv', 'results/price_spreads_top_20.txt')

# get_top_clusters('clusters_sorted_by_price_spread.json', 50, 'top_50_clusters.json')
# calculate_price_spread_score('top_50_clusters.json', 'cleaned_data.csv', 'results/leaderboard.txt')


# visualize_top_clusters('mls_lat_long.csv', 'top_20_clusters.json', 'top_20_clusters_map.html')

# get_top_clusters('clusters_sorted_by_price_spread.json', 50, 'top_50_clusters.json')
# visualize_top_clusters('mls_lat_long.csv', 'top_50_clusters.json', 'top_50_clusters_map.html')


# dbscan_cluster_visualization = 'results/1_dbscan_cluster_visualization.html'
# kmeans_refined_cluster_visualization = 'results/2_kmeans_refined_cluster_visualization.html'
# lotsqft_clusters = 'results/3_lotsqft_clusters.html'
# lotsqft_clusters_filtered = 'results/4_lotsqft_clusters_filtered.html'
# top_20_visualization = 'results/5_top_20_visualization.html'
# top_50_visualization = 'results/6_top_50_visualization.html'

# visualize_clusters_with_unique_colors('mls_lat_long_north.csv', 'geo_clusters_broken_by_lotsqft.json', lotsqft_clusters)
# visualize_clusters_with_unique_colors('mls_lat_long_north.csv', 'geo_clusters_broken_by_lotsqft_filtered.json', lotsqft_clusters_filtered)
# visualize_top_clusters('mls_lat_long_north.csv', 'top_20_clusters.json', top_20_visualization)
# visualize_top_clusters('mls_lat_long_north.csv', 'top_50_clusters.json', top_50_visualization)


# save_clusters_to_csv('top_50_clusters.json', 'cleaned_data.csv', 'results/mls_data')

# calculate_price_spread_score('top_50_clusters.json', 'cleaned_data.csv', 'results/leaderboard.txt')


# pick_random_rows('cleaned_data.csv', 50, 'sample.csv')

# visualize_cities_with_labels('city_lat_long.json', 'city_lat_long.html')

# cities_to_ignore = ['SANFRCISCO', 'SACRAMENTO']


# flatten_clusters('top_50_clusters_first_pass.json', 'flattened_cluster.json')
# filter_clusters_by_column_values('flattened_cluster.json', 'cleaned_data.csv', 'City', cities_to_ignore, 'flattened_cluster_filtered.json')


# filter_mls_by_latitude('mls_lat_long.csv', 36.20552, 'mls_lat_long_north.csv')

# delete_mls_by_cities('mls_lat_long_north.csv', ['SANFRCISCO', 'SACRAMENTO'], 'cleaned_data.csv', 'lat_long.csv')

# geographical_clustering_with_dbscan('lat_long.csv', 0.01, 50, 'cluster.json')
# visualize_clusters_with_unique_colors('lat_long.csv', 'cluster.json', 'geo_clusters.html')
# count_unique_elements_in_clusters('geo_clusters_filtered_north_dbscan.json')
# analyze_cluster_sizes('geo_clusters_filtered_north_dbscan.json')

# print_cluster_sizes('geo_clusters_filtered_north_dbscan.json')

# visualize_clusters_with_unique_colors('mls_lat_long_north.csv', 'geo_clusters.json', 'geo_clusters.html')

# refine_large_clusters_with_dbscan('geo_clusters_filtered_north_dbscan.json', 'mls_lat_long_north.csv', 150, 0.001, 30, 'geo_clusters_filtered_north_dbscan_refined.json')
# refine_large_clusters_with_kmeans('cluster.json', 'lat_long.csv', 150, 60, 'clusters_refined.json')

# count_unique_elements_in_clusters('clusters_refined.json')
# analyze_cluster_sizes('clusters_refined.json')

# visualize_clusters_with_unique_colors('mls_lat_long_north.csv', 'clusters_refined.json', 'geo_clusters_refined.html')

# count_unique_elements_in_clusters('geo_clusters_refined.json')
# analyze_cluster_sizes('geo_clusters_refined.json')

# delete_small_clusters('geo_clusters_refined.json', 20, 'geo_clusters_refined.json_filtered.json')

# count_unique_elements_in_clusters('geo_clusters_refined.json_filtered.json')
# analyze_cluster_sizes('geo_clusters_refined.json_filtered.json')

# cluster_by_sqft_within_percentage('clusters_refined.json', 'cleaned_data.csv', 15, 'refined_clusters_broken.json')

# count_unique_elements_in_clusters('refined_clusters_broken.json')
# analyze_cluster_sizes('refined_clusters_broken.json')

# visualize_clusters_with_unique_colors('mls_lat_long_north.csv', 'refined_clusters_broken.json', 'refined_clusters_broken.html')

# delete_small_clusters('refined_clusters_broken.json', 20, 'refined_clusters_broken.json')

# count_unique_elements_in_clusters('refined_clusters_broken.json')
# analyze_cluster_sizes('refined_clusters_broken.json')




# calculate_price_spread('refined_clusters_broken.json', 'cleaned_data.csv', 'price_spreads.txt')
# analyze_price_spreads('price_spreads.txt')

# # sort_clusters_by_price_spread('price_spreads.txt', 'refined_clusters_broken.json', 'clusters_sorted.json')
# count_unique_elements_in_clusters('clusters_sorted.json')
# analyze_cluster_sizes('clusters_sorted.json')

# get_top_clusters('clusters_sorted.json', 20, 'top_20_clusters.json')
# # calculate_price_spread_score('top_20_clusters.json', 'cleaned_data.csv', 'results/leaderboard.txt')
# # calculate_price_spread('top_20_clusters.json', 'cleaned_data.csv', 'results/price_spreads_top_20.txt')

# get_top_clusters('clusters_sorted.json', 50, 'top_50_clusters.json')
# calculate_price_spread_score('top_50_clusters.json', 'cleaned_data.csv', 'results/leaderboard.txt')


# visualize_top_clusters('mls_lat_long.csv', 'top_20_clusters.json', 'top_20_clusters_map.html')

# # get_top_clusters('clusters_sorted_by_price_spread.json', 50, 'top_50_clusters.json')
# visualize_top_clusters('mls_lat_long.csv', 'top_50_clusters.json', 'top_50_clusters_map.html')


# save_clusters_to_csv('top_50_clusters.json', 'cleaned_data.csv', 'results/mls_data')

# pick_random_rows('cleaned_data.csv', 50, 'sample.csv')

# analyze_sell_prices('top_50_clusters.json', 'cleaned_data.csv', 'results/price_spreads_top_50.txt')

generate_markers_json('lat_long.csv', 'top_50_clusters.json', 'results/markers.json')

# get_neighborhood_details('top_50_clusters.json', 'cleaned_data.csv', google_places_api_key, 'results/neighborhood_details.json', 'results/neighborhoods.json')

# update_mls_lat_long('top_50_clusters.json', 'cleaned_data.csv', google_places_api_key, 'mls_lat_long.csv')