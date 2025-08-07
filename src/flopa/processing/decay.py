

def fit_decay_curve(decay_curve, time_axis, model):
    print(f"--- MOCK: Fitting decay with '{model}' model ---")
    if model == "1-Component Exponential":
        return {'tau1 (ns)': 2.5, 'amplitude1': 1.0, 'chi-squared': 1.2}
    elif model == "2-Component Exponential":
        return {'tau1 (ns)': 1.0, 'amp1': 0.4, 'tau2 (ns)': 3.5, 'amp2': 0.6, 'chi-squared': 1.05}
    return {}