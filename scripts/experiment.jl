using LitPOMDP

pomdp = LiPOMDP(
    demand=[1.0, 2.0],
    mine_output=1.2
)

s0 = State(
    deposits=[1.0],
    have_mined=[true]
)

###TODO: test initiation action and observation
