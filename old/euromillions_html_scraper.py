#!/usr/bin/env python3
"""
EuroMillions HTML Scraper
Extracts draw results from resultados_euromilhoes.html file
"""

import re
from bs4 import BeautifulSoup
from datetime import datetime
import csv
import os

class EuroMillionsHTMLScraper:
    def __init__(self, html_file_path):
        self.html_file_path = html_file_path
        self.results = []
    
    def parse_date(self, date_string):
        """Parse date from format 'Tuesday - 19/08/2025' to datetime object"""
        try:
            # Extract the date part after the dash
            date_part = date_string.split(' - ')[-1].strip()
            # Parse DD/MM/YYYY format
            return datetime.strptime(date_part, '%d/%m/%Y')
        except Exception as e:
            print(f"Error parsing date '{date_string}': {e}")
            return None
    
    def extract_draw_data(self):
        """Extract all draw data from the HTML file"""
        print("Loading HTML file...")
        with open(self.html_file_path, 'r', encoding='utf-8') as file:
            html_content = file.read()
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Find all date headers
        date_elements = soup.find_all('div', class_='c-resultado-sorteo__cabecera--euromillones')
        print(f"Found {len(date_elements)} date elements")
        
        for date_elem in date_elements:
            try:
                # Extract date
                date_text = date_elem.get_text().strip()
                if not date_text:
                    continue
                
                draw_date = self.parse_date(date_text)
                if not draw_date:
                    continue
                
                # Find the parent container to get the draw ID
                parent = date_elem.find_parent('a')
                if not parent:
                    continue
                
                draw_id = parent.get('data-draw-id')
                if not draw_id:
                    continue
                
                # Find main numbers using the draw ID
                main_numbers_id = f"qa_resultado-combination-actMainNumbers-EMIL-{draw_id}"
                main_ul = soup.find('ul', id=main_numbers_id)
                
                if not main_ul:
                    print(f"Could not find main numbers for draw ID: {draw_id}")
                    continue
                
                # Extract main numbers (should be 5)
                main_numbers = []
                main_lis = main_ul.find_all('li', class_='c-resultado-sorteo__combinacion-li--euromillones')
                for li in main_lis:
                    number = li.get_text().strip()
                    if number.isdigit():
                        main_numbers.append(int(number))
                
                # Find star numbers using the draw ID
                stars_id = f"qa_resultado-combination-actMainNumbers-EMIL-stars-{draw_id}"
                stars_ul = soup.find('ul', id=stars_id)
                
                star_numbers = []
                if stars_ul:
                    star_lis = stars_ul.find_all('li', class_='c-resultado-sorteo__estrellas-li')
                    for li in star_lis:
                        number = li.get_text().strip()
                        if number.isdigit():
                            star_numbers.append(int(number))
                
                # Validate the draw (should have 5 main numbers and 2 stars)
                if len(main_numbers) == 5 and len(star_numbers) == 2:
                    draw_result = {
                        'date': draw_date,
                        'main_numbers': sorted(main_numbers),  # Keep in original order for now
                        'star_numbers': sorted(star_numbers),  # Keep in original order for now
                        'draw_id': draw_id
                    }
                    self.results.append(draw_result)
                    print(f"Extracted draw: {draw_date.strftime('%d/%m/%Y')} - {main_numbers} + {star_numbers}")
                else:
                    print(f"Invalid draw data for {draw_date}: {len(main_numbers)} main, {len(star_numbers)} stars")
                    
            except Exception as e:
                print(f"Error processing draw: {e}")
                continue
        
        # Sort results by date (oldest first)
        self.results.sort(key=lambda x: x['date'])
        print(f"\nTotal valid draws extracted: {len(self.results)}")
        return self.results
    
    def save_to_csv(self, output_file='scraped_euromillions_results.csv'):
        """Save results to CSV file"""
        if not self.results:
            print("No results to save")
            return
        
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['Date', 'Main1', 'Main2', 'Main3', 'Main4', 'Main5', 'Star1', 'Star2']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for result in self.results:
                writer.writerow({
                    'Date': result['date'].strftime('%Y-%m-%d'),
                    'Main1': result['main_numbers'][0],
                    'Main2': result['main_numbers'][1],
                    'Main3': result['main_numbers'][2],
                    'Main4': result['main_numbers'][3],
                    'Main5': result['main_numbers'][4],
                    'Star1': result['star_numbers'][0],
                    'Star2': result['star_numbers'][1]
                })
        
        print(f"Results saved to {output_file}")
    
    def show_summary(self):
        """Display summary of extracted data"""
        if not self.results:
            print("No results found")
            return
        
        print(f"\n=== EXTRACTION SUMMARY ===")
        print(f"Total draws: {len(self.results)}")
        print(f"Date range: {self.results[0]['date'].strftime('%d/%m/%Y')} to {self.results[-1]['date'].strftime('%d/%m/%Y')}")
        
        print(f"\nFirst 3 draws:")
        for i, result in enumerate(self.results[:3]):
            main = result['main_numbers']
            stars = result['star_numbers']
            print(f"{i+1}. {result['date'].strftime('%d/%m/%Y')}: {main[0]:02d}-{main[1]:02d}-{main[2]:02d}-{main[3]:02d}-{main[4]:02d} + {stars[0]:02d}-{stars[1]:02d}")
        
        print(f"\nLast 3 draws:")
        for i, result in enumerate(self.results[-3:]):
            main = result['main_numbers']
            stars = result['star_numbers']
            print(f"{len(self.results)-2+i}. {result['date'].strftime('%d/%m/%Y')}: {main[0]:02d}-{main[1]:02d}-{main[2]:02d}-{main[3]:02d}-{main[4]:02d} + {stars[0]:02d}-{stars[1]:02d}")

def main():
    html_file = 'resultados_euromilhoes.html'
    
    if not os.path.exists(html_file):
        print(f"Error: {html_file} not found in current directory")
        return
    
    print("EuroMillions HTML Scraper")
    print("=" * 25)
    
    scraper = EuroMillionsHTMLScraper(html_file)
    
    # Extract data
    results = scraper.extract_draw_data()
    
    if results:
        # Show summary
        scraper.show_summary()
        
        # Save to CSV
        scraper.save_to_csv()
        print(f"\nScraping completed successfully!")
    else:
        print("No valid draws found in the HTML file")

if __name__ == "__main__":
    main()