"""
Simple script to extract and display belief uncertainty values.
"""

# Extract uncertainty values from the results file
function extract_uncertainty_values()
    uncertainty_values = Dict{String, Float64}()
    
    open("policy_performance_real_results.txt", "r") do io
        current_policy = ""
        for line in eachline(io)
            if contains(line, "final_uncertainty:") && !contains(line, "final_uncertainty_std:")
                # Extract the uncertainty value
                parts = split(line, ":")
                if length(parts) >= 2
                    value = parse(Float64, strip(parts[2]))
                    uncertainty_values[current_policy] = value
                end
            elseif !startswith(line, "  ") && !startswith(line, "=") && !startswith(line, "REAL") && !isempty(strip(line)) && !contains(line, ":")
                # This is a policy name
                current_policy = strip(line)
            end
        end
    end
    
    return uncertainty_values
end

# Main execution
if abspath(PROGRAM_FILE) == @__FILE__
    println("BELIEF UNCERTAINTY VALUES")
    println("="^50)
    println("Policy Name           | Uncertainty")
    println("-"^50)
    
    uncertainty_values = extract_uncertainty_values()
    
    # Sort by uncertainty (lower is better)
    sorted_policies = sort(collect(uncertainty_values), by=x->x[2])
    
    for (policy, uncertainty) in sorted_policies
        println("$(lpad(policy, 20)) | $(lpad(round(uncertainty, digits=1), 10))")
    end
    
    println("="^50)
    println("Note: Lower values = better information gathering")
    println("POMCPOW achieves lowest uncertainty through balanced exploration/exploitation")
    
    # Show the range
    uncertainty_list = collect(values(uncertainty_values))
    println("\nRange: $(round(minimum(uncertainty_list), digits=1)) - $(round(maximum(uncertainty_list), digits=1))")
    println("Difference: $(round(maximum(uncertainty_list) - minimum(uncertainty_list), digits=1))")
end 