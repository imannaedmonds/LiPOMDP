

# function POMDPs.reward(P)
# end

function update_price(s, price_model_type="linear")
    if price_model_type == "linear"
        new_price = s.p + 5
        return new_price
    end

    # if model_type == "exponenital"
    #     new_price = s.p 
end



#given a state s and action a it outputs the distribution of the next state sp
function POMDPs.transition(P::LiPOMDP, s::State, a::Action)
    
#create scenarios for each action item depending on state
   #how to handle each action type 
   #update all of the properties of sp
   sp = deepcopy(s) #make a copy of the state
   sp.t += 1  # Time always advances
   sp.p = update_price(sp, P.price_model_type)
  
   #changing price here based on type of pricing model

    # Action: Do Nothing and Explore (type 0 and 1)
    if a.type in (0, 1)
        return Deterministic(sp)
   end 

    # Action: Open Mine (type 2)
   if a.type == 2
        site = a.site
        sp.mine_ready[site] = true
        return Deterministic(sp)
    end

    #Action: Operate (type 3)
    if a.type == 3
        site = a.site
        if sp.mine_ready[site]
            amt = min(P.mine_output, sp.deposits[site])
            sp.deposits[site] -= amt
            sp.v += amt
        end
        return Deterministic(sp)
    end

# Fallback: return current state if action type is unrecognized
    return Deterministic(sp)
end

    #if the process similar then use it as a nindex but if its different then create branch
    # TODO clarify the logic of the transition function — how each action will change the state
    #last line has to be deterministic 
    #each action type will affect the state being updated 
#produce at capacity -- rate capacity