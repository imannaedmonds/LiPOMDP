# https://www.usgs.gov/centers/national-minerals-information-center/lithium-statistics-and-information
# https://ourworldindata.org/grapher/lithium-production?tab=chart&country=~OWID_WRL 
# https://www.energyinst.org/__data/assets/pdf_file/0006/1542714/684_EI_Stat_Review_V16_DIGITAL.pdf 
# “Data Page: Lithium production”, part of the following publication: Hannah Ritchie, Pablo Rosado, and Max Roser (2023) - “Energy”. Data adapted from Energy Institute. Retrieved from https://archive.ourworldindata.org/20250624-125417/grapher/lithium-production.html [online resource] (archived on June 24, 2025).
# measured in thousand of tonnes
# Define the year and production values manually extracted from the image and user's message

# Data with 

# Define the years, demand (in thousand metric tons), and price (in USD per metric ton)
years = [1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004,
         2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014,
         2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]

production = [9485.481, 14783.803, 15591.14, 14192.644, 12875.787,
          14283.851, 14002.716, 15486.188, 18086.102, 19361.534,
          20947.239, 24263.863, 26374.807, 26328.201, 19532.03,
          26476.818, 33033.84, 34745.956, 30391.504, 30951.284,
          29543.085, 38217.278, 50850.212, 95134.285, 86881.62,
          83695.14, 107879.66, 157809.71, 198012.6, 201012.6,]

price = [3200.0, 3400.0, 3800.0, 4000.0, 4200.0,
         4400.0, 4600.0, 4800.0, 5000.0, 5200.0,
         5400.0, 5600.0, 6000.0, 6400.0, 5800.0,
         5180.0, 5180.0, 6060.0, 6800.0, 6690.0,
         6500.0, 8650.0, 15000.0, 16000.0, 12100.0,
         10100.0, 14200.0, 71100.0, 41300.0, 14000.0]  # Assumes price data starts from 1995

# Create the dictionary
lithium_data = Dict(years[i] => (demand[i], price[i]) for i in 1:length(years))



# Normalize time (t=0 at year 1995)
t = years - 1995

# --- Linear model fitting ---
def linear_model(t, p0, alpha_slope):
    return p0 + alpha_slope * t

popt_linear, _ = curve_fit(linear_model, t, price)
p0_linear, alpha_slope = popt_linear

# --- Exponential model fitting ---
def exponential_model(t, p0, lambda_exp):
    return p0 * np.exp(lambda_exp * t)

popt_exp, _ = curve_fit(exponential_model, t, price, p0=[price[0], 0.1])
p0_exp, lambda_exp = popt_exp

(p0_linear, alpha_slope), (p0_exp, lambda_exp)
Result
((-3008.4134227824147, 945.5763321773571),
 (58.40860272188521, 0.24080763725507343))