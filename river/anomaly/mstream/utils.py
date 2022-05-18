

def counts_to_anom(tot, cur, cur_t):
    cur_mean = tot / cur_t
    sqerr = pow(max(0, cur - cur_mean), 2)
    a = sqerr / cur_mean
    b = sqerr / (cur_mean * max(1, cur_t - 1))
    return a + b
