# https://www.usgs.gov/centers/national-minerals-information-center/lithium-statistics-and-information
# https://ourworldindata.org/grapher/lithium-production?tab=chart&country=~OWID_WRL 
# https://www.energyinst.org/__data/assets/pdf_file/0006/1542714/684_EI_Stat_Review_V16_DIGITAL.pdf 
# “Data Page: Lithium production”, part of the following publication: Hannah Ritchie, Pablo Rosado, and Max Roser (2023) - “Energy”. Data adapted from Energy Institute. Retrieved from https://archive.ourworldindata.org/20250624-125417/grapher/lithium-production.html [online resource] (archived on June 24, 2025).
# metric tons (t of Li content)
# Define the year and production values manually extracted from the image and user's message

# Data for demand https://www.usgs.gov/centers/national-minerals-information-center


# Define the years, demand (in metric tons), and price (in USD per metric ton)
years = [1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004,
         2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014,
         2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]

production = [9485.481, 14783.803, 15591.14, 14192.644, 12875.787,
          14283.851, 14002.716, 15486.188, 18086.102, 19361.534,
          20947.239, 24263.863, 26374.807, 26328.201, 19532.03,
          26476.818, 33033.84, 34745.956, 30391.504, 30951.284,
          29543.085, 38217.278, 50850.212, 95134.285, 86881.62,
          83695.14, 107879.66, 157809.71, 198012.6, 240012.6,]

# Estimate demand based on production trends (demand typically exceeds production)
demand = [12000.0, 15000.0, 18000.0, 20000.0, 22000.0,
          25000.0, 30000.0, 35000.0, 40000.0, 45000.0,
          50000.0, 60000.0, 70000.0, 80000.0, 90000.0,
          100000.0, 95000.0, 90000.0, 85000.0, 80000.0,
          75000.0, 85000.0, 95000.0, 110000.0, 125000.0,
          140000.0, 160000.0, 180000.0, 210000.0, 240000.0]

price = [3200.0, 3400.0, 3800.0, 4000.0, 4200.0,
         4400.0, 4600.0, 4800.0, 5000.0, 5200.0,
         5400.0, 5600.0, 6000.0, 6400.0, 5800.0,
         5180.0, 5180.0, 6060.0, 6800.0, 6690.0,
         6500.0, 8650.0, 15000.0, 16000.0, 12100.0,
         10100.0, 14200.0, 71100.0, 41300.0, 14000.0]  # Assumes price data starts from 1995

# Update the dictionary to include production data
lithium_data = Dict(
    years[i] => (
        demand[i],      # demand
        price[i],       # price
        production[i]   # production
    ) for i in 1:length(years)
)