using Distributions

# Test the exponential price model
function test_exponential_prices()
    println("=== EXPONENTIAL PRICE ANALYSIS (FIXED) ===")
    
    p₀ = 8_000.0  # New higher base price
    λ_exp = 0.15   # New higher growth rate
    
    println("Base price: $p₀")
    println("Growth rate: $λ_exp")
    println()
    
    for t in 1:15
        base_price = p₀ * exp(λ_exp * t)
        # Add 10% noise
        noise_std = base_price * 0.1
        noise = rand(Normal(0, noise_std))
        final_price = base_price + noise
        
        println("Year $t: Base = $(round(base_price, digits=2)), Final = $(round(final_price, digits=2))")
    end
    
    println()
    println("=== COMPARISON WITH OTHER MODELS ===")
    
    # Static model
    static_price = 10_355.33
    println("Static price: $static_price")
    
    # Linear model at year 15
    linear_p₀ = 2_381.38
    α_slope = 878.39
    linear_price = linear_p₀ + α_slope * 15
    println("Linear price (year 15): $linear_price")
    
    # Exponential model at year 15 (NEW)
    exp_price = p₀ * exp(λ_exp * 15)
    println("Exponential price (year 15): $exp_price")
    
    println()
    println("=== PROFITABILITY ANALYSIS ===")
    println("Static vs Exponential at year 15: $(exp_price > static_price ? "Exponential wins!" : "Static still wins")")
    println("Exponential is $(round((exp_price/static_price - 1) * 100, digits=1))% higher than static")
end

test_exponential_prices()
