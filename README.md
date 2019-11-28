# OptionPricing
A option pricing repo including American Option, European Option, spread, binary and barrier.

# Reference
Options, Futures and Derivatives - John C.Hull

# Binomial Tree
Vanilla American Option included.

# Black-Scholes
Include analytical solution for:

    vanilla: euro call option - vanilla_european_call()
             euro put  option - vanilla_european_put()

    spread: bull call option - bull_call()
            bull put option - bull_put()
            bear call option - bear_call()
            bear put option - bear_put()

    binary:  cash call option - bi_cashcall()
             cash put  option - bi_cashput()
             asset call option - bi_assetcall()
             asset put  option - bi_assetput()

    barrier: down-and-out-call option - down_call(knock='out')
             down-and-in-call  option - down_call(knock='in')
             up-and-out-call   option - up_call(knock='out')
              up-and-in-call   option - up_call(knock='in')
              up-and-out-put   option - up_put(knock='out')
              up-and-in-put    option - up_put(knock='in')
             down-and-out-put  option - down_put(knock='out')
             down-and-in-put   option - down_put(knock='in')
