
#TODO-Mansur where should I put the hisotrical data
# Historical data access functions
function get_historical_price(pomdp::LiPOMDP, year::Int64)
    if haskey(pomdp.hist_data, year)
        return pomdp.hist_data[year][2]  # price
    else
        @warn "No historical price data for year $year"
        return pomdp.p₀
    end
end

function get_historical_demand(pomdp::LiPOMDP, year::Int64)
    if haskey(pomdp.hist_data, year)
        return pomdp.hist_data[year][1]  # demand
    else
        @warn "No historical demand data for year $year"
        return 0.0
    end
end

function get_historical_production(pomdp::LiPOMDP, year::Int64)
    if haskey(pomdp.hist_data, year)
        return pomdp.hist_data[year][3]  # production
    else
        @warn "No historical production data for year $year"
        return 0.0
    end
end

# Helper function to get all historical data for a year
function get_historical_data(pomdp::LiPOMDP, year::Int64)
    if haskey(pomdp.hist_data, year)
        demand, price, production = pomdp.hist_data[year]
        return (
            demand=demand,
            price=price,
            production=production
        )
    else
        @warn "No historical data for year $year"
        return (
            demand=0.0,
            price=pomdp.p₀,
            production=0.0
        )
    end
end


