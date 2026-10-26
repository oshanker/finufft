from decimal import Decimal, getcontext

def calculate_expression(t) -> Decimal:
    """
    Calculates the expression (t / 2pi) * (ln(t / 2pi) - 1) - 1/8 
    using arbitrary-precision decimals.
    
    :param t_str: The big decimal value of 't' as a string to preserve precision.
    :param precision: The number of significant digits for the calculation context.
    :return: The result as a Decimal object.
    """
    
    # 2. Define Pi with high precision matching the set context
    # Expand or compress this constant based on your desired precision length
    pi = Decimal('3.14159265358979323846264338327950288419716939937510')
    
    # 3. Convert input and setup constant fractions

    two_pi = Decimal('2') * pi
    one_eighth = Decimal('1') / Decimal('8')
    
    # 4. Break down the expression steps
    fraction = t / two_pi
    ln_fraction = fraction.ln()  # Decimal has a built-in natural log function
    gram_incr = two_pi/ln_fraction

    # 5. Compute the final formula
    result = fraction * (ln_fraction - Decimal('1')) - one_eighth
    
    return result, gram_incr

# --- Example Usage ---
if __name__ == "__main__":
    # Always pass large numbers as a string to avoid standard float rounding errors
    # 1. Set the precision context for the decimal operations
    precision = 60
    getcontext().prec = precision
    large_t_base = "1.0E12" 
    # 243.53396107975615984
    base_limit = Decimal("243.53396107975615984")
    t = Decimal(large_t_base) + base_limit
    output, gram_incr = calculate_expression(t)
    print(f" gram_incr { gram_incr} ")
    print(f"base_limit {base_limit} n: {output}")
    incr_limit =  Decimal("60000")*gram_incr 
    upperlimit = incr_limit + base_limit
    t = Decimal(large_t_base) + upperlimit
    output, _ = calculate_expression(t)
    print(f"upper_limit {upperlimit} n: {output}")

    upperlimit = Decimal("14859.4767721")
    t = Decimal(large_t_base) + upperlimit
    output, _ = calculate_expression(t)
    print(f"upper_limit {upperlimit} n: {output}")

    
