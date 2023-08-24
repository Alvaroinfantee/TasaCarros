import requests
from bs4 import BeautifulSoup
import h5py
import numpy as np
from PIL import Image
from io import BytesIO

url = 'https://www.montatediunave.com/'
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}

def scrape_table(url, h5file, group):
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')
    cars = soup.select('#main-container > div > section > div > div:nth-child(3) > div > div')

    for idx, car in enumerate(cars):
        name = car.select_one('div > p > span').text
        year = car.select_one('div > div > span').text
        price = car.select_one('div > div > strong').text

        # Extract the numerical value of the price
        if 'US$' in price:
            price_value = float(price.replace('US$', '').replace(',', ''))
            # Convert the price from USD to DOP
            price_in_dop = price_value * 56
        else:
            price_in_dop = float(price.replace('RD$', '').replace(',', ''))

        # Extract the image URL
        image_url = car.select_one('div > a > img')['src']
        image_response = requests.get(image_url)
        image = Image.open(BytesIO(image_response.content))
        image_array = np.array(image)

        # Create a subgroup for each car
        car_group = group.create_group(str(idx))
        car_group.create_dataset('Image', data=image_array)
        car_group.attrs['Name'] = name
        car_group.attrs['Year'] = int(year)
        car_group.attrs['Price'] = price_in_dop

# Create an HDF5 file to store the images and metadata
with h5py.File('car_data.h5', 'w') as h5file:
    # Iterate over multiple pages
    for i in range(1, 100):  # Change the range according to the number of pages
        url_page = url + '?page=' + str(i)
        group = h5file.create_group('page_' + str(i))
        scrape_table(url_page, h5file, group)
