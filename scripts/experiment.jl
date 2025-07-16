    using LitPOMDP  
    using POMDPs
    using Random


    pomdp = LiPOMDP(
        demand=[1.0, 2.0],
        mine_output=1.2,
        price_model_type = 3,
    )
    rng = MersenneTwister(1234)

    s0 = State(
        deposits=[1.0, 2.0],
        have_mined=[true, false],
        mine_ready=[false, false]
    )



    ###TODO: test initiation action and observation

    a0=Action(
        site=0,
        type=1
    )


    prob_s1=transition(pomdp, s0, a0)
    s1 = rand(rng, prob_s1)

    r = reward(pomdp, s0, a0)

    o_dist = observation(pomdp, s0, a0, s1)
    o_array = rand(rng, o_dist)
    o1 = Observation(deposits=o_array[1:end-1], price=o_array[end])



    sp, o, r = gen(pomdp, s0, a0, rng)

    o_dist = observation(pomdp, s0, a0, s1)
    o_array = rand(rng, o_dist)
    o1 = Observation(deposits=o_array[1:end-1], price=o_array[end])
    sp, o, r = gen(pomdp, s0, a0, rng)


    