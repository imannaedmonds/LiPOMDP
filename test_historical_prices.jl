using Distributions

# Include the historical data
include("src/historical_data.jl")

# Test the historical price model
function test_historical_prices()
    println("=== HISTORICAL PRICE ANALYSIS ===")
    
    # Simulate historical prices for years 1995-2010 (15 years)
    for t in 1:15
        current_year = 1995 + t - 1
        
        if haskey(lithium_data, current_year)
            historical_price = lithium_data[current_year][2]  # Price is the second element
            
            # Add 5% noise
            noise_std = historical_price * 0.05
            noise = rand(Normal(0, noise_std))
            final_price = historical_price + noise
            
            println("Year $current_year: Historical = $(round(historical_price, digits=2)), Final = $(round(final_price, digits=2))")
        else
            println("Year $current_year: No historical data available")
        end
    end
    
    println()
    println("=== HISTORICAL PRICE RANGE ===")
    prices = [lithium_data[year][2] for year in keys(lithium_data)]
    println("Min price: $(minimum(prices))")
    println("Max price: $(maximum(prices))")
    println("Average price: $(round(mean(prices), digits=2))")
    println("Price range: $(maximum(prices) - minimum(prices))")
end

test_historical_prices()
