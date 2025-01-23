# -*- coding: utf-8 -*-
"""
# Data Source: https://realestate.co.jp/en/rent
# Resource: https://github.com/ShoKosaka/Suumo/blob/master/Scraping.ipynb
# Resource: https://github.com/coco2525/tokyo-rental-forecasting/blob/main/scripts/data_processing.py
"""
################################################################

from bs4 import BeautifulSoup
import requests
import pandas as pd
import sqlite3
import time
import re
import concurrent.futures
from functools import partial
from datetime import datetime
from tqdm import tqdm

def create_database():
    """Create SQLite database and table"""
    conn = sqlite3.connect('tokyo_rent.db')
    cursor = conn.cursor()
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS properties (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        room_type TEXT,
        category TEXT,
        street TEXT,
        city TEXT,
        prefecture TEXT,
        cost REAL,
        size REAL,
        deposit REAL,
        key_money REAL,
        floor INTEGER,
        year INTEGER,
        station TEXT,
        minute INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    conn.commit()
    return conn

def scrape_page(url, pbar):
    """Scrape a single page with progress tracking"""
    try:
        page_num = int(url.split('page=')[-1])
        current_time = datetime.now().strftime("%H:%M:%S")
        print(f"[{current_time}] Processing page {page_num}")
        
        result = requests.get(url)
        if result.status_code != 200:
            print(f"Failed to retrieve page: {url}")
            pbar.update(1)
            return None
            
        soup = BeautifulSoup(result.content, 'html.parser')
        summary = soup.find_all("div", class_="listing-body")
        
        if not summary:
            pbar.update(1)
            return None
            
        page_data = {
            'room_type': [], 'category': [], 'street': [], 'city': [], 'prefecture': [],
            'cost': [], 'size': [], 'deposit': [], 'key_money': [], 'floor': [],
            'year': [], 'station': [], 'minute': []
        }
        
        print(f"Found {len(summary)} listings on page {page_num}")
        
        for item in summary:
            # Initialize all fields with None for this listing
            current_listing = {
                'room_type': None, 'category': None, 'street': None, 
                'city': None, 'prefecture': None, 'cost': None, 
                'size': None, 'deposit': None, 'key_money': None, 
                'floor': None, 'year': None, 'station': None, 'minute': None
            }
            
            # Get room_type and category
            name = item.find("span", class_="text-semi-strong")
            if name:
                name_text = name.get_text(separator=" ", strip=True)
                parts = name_text.split(' ', 1)
                if len(parts) == 2:
                    current_listing['room_type'] = parts[0].strip()
                    current_listing['category'] = parts[1].strip()
                else:
                    current_listing['room_type'] = name_text
            
            # Get address
            address = item.find("span", itemprop="address")
            if address:
                address_text = address.get_text(separator=", ", strip=True).replace('in ', '')
                try:
                    street_part, city_part, prefecture_part = address_text.split(', ')
                    current_listing['street'] = street_part.strip()
                    current_listing['city'] = city_part.strip()
                    current_listing['prefecture'] = prefecture_part.strip()
                except ValueError:
                    print(f"Error processing address: {address_text}")

            # Process other fields
            for x in item.find_all("div", class_="listing-item"):
                text = x.get_text(strip=True)

                # Get cost
                if "Monthly Costs" in text:
                    cost_text = text.replace("Monthly Costs", "").replace("¥", "").replace(",", "").strip()
                    cost_value = re.findall(r'\d+', cost_text)
                    if cost_value:
                        current_listing['cost'] = float(cost_value[0])

                # Get size
                if "Size" in text:
                    size_text = text.replace("Size", "").replace("m²", "").strip()
                    try:
                        current_listing['size'] = float(size_text)
                    except ValueError:
                        pass

                # Get deposit
                if "Deposit" in text:
                    deposit_text = text.replace("Deposit", "").replace("¥", "").replace(",", "").strip()
                    try:
                        current_listing['deposit'] = float(deposit_text)
                    except ValueError:
                        pass

                # Get key_money
                if "Key Money" in text:
                    key_money_text = text.replace("Key Money", "").replace("¥", "").replace(",", "").strip()
                    try:
                        current_listing['key_money'] = float(key_money_text)
                    except ValueError:
                        pass

                # Get floor
                if "Floor" in text:
                    floor_text = text.replace("Floor", "").replace("F", "").strip()
                    floor_number = re.findall(r'\d+', floor_text)
                    if floor_number:
                        current_listing['floor'] = int(floor_number[0])

                # Get year
                if "Year Built" in text:
                    year_text = text.replace("Year Built", "").strip()
                    if year_text.isdigit():
                        current_listing['year'] = int(year_text)

                # Get station and minute
                if "Nearest Station" in text:
                    nearest_station_text = text.replace("Nearest Station", "").strip()
                    if "(" in nearest_station_text and "min. walk)" in nearest_station_text:
                        try:
                            name_part, minute_part = nearest_station_text.split("(", 1)
                            current_listing['station'] = name_part.replace(" Station", "").strip()
                            current_listing['minute'] = int(minute_part.replace("min. walk)", "").strip())
                        except ValueError:
                            print(f"Error processing nearest station text: {nearest_station_text}")

            # Add all fields from current_listing to page_data
            for key in page_data:
                page_data[key].append(current_listing[key])

        time.sleep(1)  # Rate limiting
        pbar.update(1)
        print(f"Completed processing page {page_num}")
        return page_data
        
    except Exception as e:
        print(f"Error processing page {url}: {str(e)}")
        pbar.update(1)
        return None

def main():
    start_time = time.time()
    print(f"Starting scraping job at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create database connection
    conn = create_database()
    
    # Initial URL and page fetching
    url = 'https://realestate.co.jp/en/rent?order=index_ranking-desc&page=1'
    print("Fetching initial page to determine total pages...")
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')
    
    # Get the number of pages
    body = soup.find("body")
    pages = body.find_all("li", class_="pagination-last")
    
    # Initialize page number
    page_number = 1
    if pages:
        last_page = pages[0].find("a")
        if last_page:
            href = last_page.get("href")
            page_number = int(href.split("page=")[-1])
            print(f"Found {page_number} total pages to process")
    
    # Generate URLs
    base_url = 'https://realestate.co.jp/en/rent'
    urls = [f"{base_url}?order=index_ranking-desc&page={i}" for i in range(1, page_number + 1)]
    
    # Create progress bar
    pbar = tqdm(total=len(urls), desc="Overall Progress", unit="page")
    
    # Use ThreadPoolExecutor for concurrent scraping
    print(f"\nStarting concurrent processing with {min(5, len(urls))} workers")
    all_data = {
        'room_type': [], 'category': [], 'street': [], 'city': [], 'prefecture': [],
        'cost': [], 'size': [], 'deposit': [], 'key_money': [], 'floor': [],
        'year': [], 'station': [], 'minute': []
    }
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        scrape_func = partial(scrape_page, pbar=pbar)
        results = list(executor.map(scrape_func, urls))
    
    # Combine results from all pages
    total_listings = 0
    for result in results:
        if result:
            for key in all_data:
                all_data[key].extend(result[key])
            total_listings += len(result['room_type'])
    
    # Convert to DataFrame
    print("\nCreating DataFrame...")
    realestate_df = pd.DataFrame(all_data)
    
    # Convert empty strings to None (NULL in SQLite)
    realestate_df = realestate_df.replace(r'^\s*$', None, regex=True)
    
    # Convert specific columns to appropriate types
    numeric_columns = ['cost', 'size', 'deposit', 'key_money']
    integer_columns = ['floor', 'year', 'minute']
    
    for col in numeric_columns:
        realestate_df[col] = pd.to_numeric(realestate_df[col], errors='coerce')
    
    for col in integer_columns:
        realestate_df[col] = pd.to_numeric(realestate_df[col], errors='coerce').astype('Int64')
    
    # Save to SQLite
    print("\nSaving to SQLite database...")
    realestate_df.to_sql('properties', conn, if_exists='append', index=False)
    
    # Generate data quality report
    cursor = conn.cursor()
    
    print("\nData Quality Report:")
    
    # Get total rows
    cursor.execute("SELECT COUNT(*) FROM properties")
    total_rows = cursor.fetchone()[0]
    print(f"Total rows: {total_rows}")
    
    # Get NULL counts for each column
    for column in realestate_df.columns:
        cursor.execute(f"SELECT COUNT(*) FROM properties WHERE {column} IS NULL")
        null_count = cursor.fetchone()[0]
        cursor.execute(f"SELECT COUNT(*) FROM properties WHERE {column} IS NOT NULL")
        non_null = cursor.fetchone()[0]
        print(f"{column}: {non_null} non-null values, {null_count} NULL values")
    
    # Generate some basic statistics
    print("\nBasic Statistics:")
    cursor.execute("""
    SELECT 
        AVG(cost) as avg_cost,
        MIN(cost) as min_cost,
        MAX(cost) as max_cost,
        AVG(size) as avg_size,
        COUNT(DISTINCT prefecture) as prefecture_count,
        COUNT(DISTINCT city) as city_count
    FROM properties
    WHERE cost IS NOT NULL
    """)
    stats = cursor.fetchone()
    print(f"Average Cost: ¥{stats[0]:,.2f}")
    print(f"Min Cost: ¥{stats[1]:,.2f}")
    print(f"Max Cost: ¥{stats[2]:,.2f}")
    print(f"Average Size: {stats[3]:.2f} m²")
    print(f"Number of Prefectures: {stats[4]}")
    print(f"Number of Cities: {stats[5]}")
    
    # Close database connection
    conn.close()
    
    # Print summary
    end_time = time.time()
    duration = end_time - start_time
    print(f"\nScraping job completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total time elapsed: {duration:.2f} seconds")
    print(f"Total pages processed: {page_number}")
    print(f"Total listings collected: {total_listings}")
    print(f"Average time per page: {duration/page_number:.2f} seconds")
    print(f"Data saved to: tokyo_rent.db")
    
    pbar.close()

if __name__ == "__main__":
    main()