# TokyoRentingBirdview


## Project Overview

TokyoRentingBirdview aims to help newcomers in Japan understand the factors influencing rental prices, enabling them to make informed decisions about their living arrangements. The project delves into the specific relationships between price and various features, such as:
 - What factors contribute to low initial deposits?
 - How does the distance to the nearest station and proximity to downtown affect rental prices differently, and by how much?
 - etc.. name your relasion
By balancing walkability and property location, you can find the best value for your budget. Or discover low-deposit options that meet your needs.

---

## Key Features
- **Data Collection**: Scraped data from 30,883 rental properties across 2,235 pages using `BeautifulSoup`.
- **Data Preprocessing**:
  - Removed irrelevant features and handled missing values using SQLite and Pandas.
  - Created new variables, including building age and dummy variables
  - handling categorical and numric values globally
- **Visualization**:
  - Rental trends and correlations visualized using bar graphs, box plots, and heatmaps.
- **Model Development**:
  - A feature selection model achieved an R² of 86% of accuracy.
- **Web Deployment**:
  - User-friendly prediction app deployed via Gradio.

---




---

## Data and Methods

### Data Source
- Data scraped from [Real Estate Japan](https://realestate.co.jp/en/rent).

### Preprocessing
- improved scapping performance by using muliti thered scapping, now whole website can be capped under 1 hour.
- Added dummy variables for categorical data and standardized numerical features.

### Visualization
- **corrlation**:
  - Correlation between features and price.
- **WEBUI**:
  - a gardio interface.


## Model Performance
| Metric       | Value (JPY) |
|--------------|-------------|
| **R²**       | 0.86        |



## Web Application
The project includes a deployed web interface that enables users to:
- Input property features like building age, square footage, and proximity to stations.
- Obtain correlations between features..

## Applications
- **First-time Renters**: Understand market situation for planning purposes.
- **Investors and Agents**: Strategically evaluate rental properties.


---

## Future Enhancements
1. Selecting "best match" houses by combining the dataset with Google Maps API.
2. Explore alternative machine learning models for added functionality.
3. Expand the dataset to include Suumo data


---

## Credit & addtional resources

- ["fair price" checker](https://www.landandhome.co.jp/rent/price_checker/)
- @coco2525 remixed scapping code and readme.md