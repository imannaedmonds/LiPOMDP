using Random
using Parameters
using POMDPs
using POMDPTools

# ── tiny helpers shared by the heuristics ────────────────────────────────────
noop(P::LiPOMDP) = Action(site=fill(0, P.n_deposits), type=fill(0, P.n_deposits))

function open_action(P::LiPOMDP, site::Int)
    svec = fill(0, P.n_deposits); tvec = fill(0, P.n_deposits)
    svec[site] = site; tvec[site] = 2
    return Action(site=svec, type=tvec)
end

function explore_action(P::LiPOMDP, site::Int)
    svec = fill(0, P.n_deposits); tvec = fill(0, P.n_deposits)
    svec[site] = site; tvec[site] = 1
    return Action(site=svec, type=tvec)
end

# 2-step heuristic score: r(s,a) + γ * r(sp, noop)
function two_step_q(P::LiPOMDP, s::State, a::Action)
    r1 = POMDPs.reward(P, s, a)
    sp = rand(POMDPs.transition(P, s, a)) # Deterministic in your model
    r2 = POMDPs.reward(P, sp, noop(P))
    return r1 + P.γ*r2
end

# saleable contribution next step if site is opened now (for demand cap)
function next_step_saleable_if_open(P::LiPOMDP, s::State, site::Int)
    sp = rand(POMDPs.transition(P, s, open_action(P, site)))
    _, sellable = calculate_site_production(P, sp, site, sp.company_demand)
    return sellable
end

# Build a single multi-site "open" action that covers ~demand_cover of demand, <= top_k sites
function open_best_until_demand(P::LiPOMDP, s::State;
                                scores::Vector{Tuple{Int,Float64}},
                                demand_cover::Float64=0.9,
                                top_k::Int=typemax(Int))
    sort!(scores, by = x->x[2], rev=true)  # higher score first
    remaining = s.company_demand * max(0.0, min(demand_cover, 1.0))

    chosen = Int[]
    for (site, _) in scores
        if length(chosen) >= top_k; break; end
        if !s.have_opened[site] && s.deposit[site] > 0
            add = next_step_saleable_if_open(P, s, site)
            if add > 0
                push!(chosen, site)
                remaining -= add
                if remaining <= 0
                    break
                end
            end
        end
    end

    svec = fill(0, P.n_deposits); tvec = fill(0, P.n_deposits)
    for i in chosen
        svec[i] = i; tvec[i] = 2
    end
    return Action(site=svec, type=tvec)
end

# ── 1) ProfitMaximizerPolicy (heuristic, demand-aware, multi-site) ───────────
@with_kw struct ProfitMaximizerPolicy2 <: Policy
    pomdp::LiPOMDP
    debug::Bool=false
    top_k::Int=1              # max sites to open at an activation time
    times::Vector{Int}=[1]    # activate only at these integer times
    demand_cover::Float64=0.9 # cover ~90% of current demand
end

function POMDPs.action(policy::ProfitMaximizerPolicy2, s::State)
    P = policy.pomdp
    if !(Int(round(s.time)) in policy.times)
        return noop(P)
    end
    # score each unopened site by two-step Q of opening that site alone
    scores = Tuple{Int,Float64}[]
    for site in 1:P.n_deposits
        if !s.have_opened[site] && s.deposit[site] > 0
            sc = two_step_q(P, s, open_action(P, site))
            push!(scores, (site, sc))
        end
    end
    if isempty(scores)
        return noop(P)
    end
    act = open_best_until_demand(P, s; scores=scores,
                                 demand_cover=policy.demand_cover,
                                 top_k=policy.top_k)
    if policy.debug
        println("[ProfitMaximizer] scores = ", scores)
        println("[ProfitMaximizer] action = ", act)
    end
    return act
end

function POMDPs.action(policy::ProfitMaximizerPolicy2, b::LiBelief)
    s = State(
        deposit=[mean(d) for d in b.deposits_distribution],
        time=b.time, price=mean(b.price_distribution),
        company_demand=mean(b.demand_distribution),
        total_mined=b.total_mined, have_opened=copy(b.have_opened)
    )
    return POMDPs.action(policy, s)
end

# ── 2) EmissionMinimizerPolicy (heuristic, demand-aware, multi-site) ─────────
@with_kw struct EmissionMinimizerPolicy2 <: Policy
    pomdp::LiPOMDP
    debug::Bool=false
    top_k::Int=1
    times::Vector{Int}=[1]
    demand_cover::Float64=0.9
end

# simple per-site “negative emission cost” score for opening now; higher is better
function emission_score_open(P::LiPOMDP, s::State, site::Int)
    sp = rand(POMDPs.transition(P, s, open_action(P, site)))
    # reuse your reward to capture CAPEX/OPEX penalties and CO₂ cost weightings
    # but bias toward “lower emissions” by evaluating immediate open only
    return two_step_q(P, s, open_action(P, site))
end

function POMDPs.action(policy::EmissionMinimizerPolicy2, s::State)
    P = policy.pomdp
    if !(Int(round(s.time)) in policy.times)
        return noop(P)
    end

    # rank by “cleanliness”: lower variable EF first, tiebreak by two-step score
    candidates = Tuple{Int,Float64,Float64}[]  # (site, ef, q2)
    for site in 1:P.n_deposits
        if !s.have_opened[site] && s.deposit[site] > 0
            ef = P.deposit_types[site] ? P.dle_CO2_emissions : P.mining_CO2_emissions
            q2 = emission_score_open(P, s, site)
            push!(candidates, (site, ef, q2))
        end
    end
    if isempty(candidates)
        return noop(P)
    end
    sort!(candidates, by = x->(x[2], -x[3]))  # lowest EF, then best q2

    # convert to (site, score) vector and open until demand is ~covered
    scores = [(site, q2) for (site, _ef, q2) in candidates]
    act = open_best_until_demand(P, s; scores=scores,
                                 demand_cover=policy.demand_cover,
                                 top_k=policy.top_k)
    if policy.debug
        println("[EmissionMinimizer] ranked = ", candidates)
        println("[EmissionMinimizer] action = ", act)
    end
    return act
end

function POMDPs.action(policy::EmissionMinimizerPolicy2, b::LiBelief)
    s = State(
        deposit=[mean(d) for d in b.deposits_distribution],
        time=b.time, price=mean(b.price_distribution),
        company_demand=mean(b.demand_distribution),
        total_mined=b.total_mined, have_opened=copy(b.have_opened)
    )
    return POMDPs.action(policy, s)
end

# ── 3) ExploreOnlyPolicy (unchanged; still a simple heuristic) ───────────────
struct ExploreOnlyPolicy2 <: Policy
    pomdp::LiPOMDP
    debug::Bool
end

ExploreOnlyPolicy2(p::LiPOMDP; debug=false) = ExploreOnlyPolicy2(p, debug)

function POMDPs.action(policy::ExploreOnlyPolicy2, s::State)
    P = policy.pomdp
    svec = fill(0, P.n_deposits); tvec = fill(0, P.n_deposits)
    for i in 1:P.n_deposits
        if !s.have_opened[i]
            svec[i] = i; tvec[i] = 1
        end
    end
    return Action(site=svec, type=tvec)
end

function POMDPs.action(policy::ExploreOnlyPolicy2, b::LiBelief)
    s = rand(MersenneTwister(1234), b)
    return action(policy, s)
end

# ── 4) RandomHeuristicPolicy (unchanged logic, supports multi-site) ──────────
struct RandomHeuristicPolicy2 <: Policy
    pomdp::LiPOMDP
    rng::AbstractRNG
    debug::Bool
end

RandomHeuristicPolicy2(p::LiPOMDP; rng=Random.GLOBAL_RNG, debug=false) =
    RandomHeuristicPolicy2(p, rng, debug)

function POMDPs.action(policy::RandomHeuristicPolicy2, s::State)
    P = policy.pomdp
    svec = fill(0, P.n_deposits); tvec = fill(0, P.n_deposits)
    for i in 1:P.n_deposits
        if !s.have_opened[i]
            a = rand(policy.rng, 0:2)
            if a > 0
                svec[i] = i; tvec[i] = a
            end
        end
    end
    return Action(site=svec, type=tvec)
end

function POMDPs.action(policy::RandomHeuristicPolicy2, b::LiBelief)
    s = rand(policy.rng, b)
    return action(policy, s)
end
