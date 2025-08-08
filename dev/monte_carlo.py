import numpy as np
from scipy.stats import beta


class MonteCarloSimulator:
    def __init__(self, n_simulations, final_baseline_trend, event_params,
                 class_share_param, product_share_param, gross_price_param, gtn_param):
        self.n_simulations = n_simulations
        self.final_baseline_trend = final_baseline_trend
        self.event_params = event_params
        self.param_dict = {
            "class_share": class_share_param,
            "product_share": product_share_param,
            "gross_price": gross_price_param,
            "gtn": gtn_param
        }

    def sample_from_distribution(
    self,
    dist_type,
    low=None,
    high=None,
    mean=None,
    std=None,
    base=None,
    mode=None,
    probs=None
    ):
        mode = mode or base

        if dist_type == 'uniform':
            if low is None or high is None:
                raise ValueError("Uniform requires 'low' and 'high'")
            return np.random.uniform(low, high)

        elif dist_type == 'normal':
            if mean is None or std is None:
                if low is not None and high is not None:
                    mean = (low + high) / 2
                    std = abs(high - low) / 4
                else:
                    raise ValueError("Normal requires 'mean/std' or 'low/high'")
            if std < 0:
                raise ValueError("Standard deviation must be >= 0")
            return np.random.normal(mean, std)

        elif dist_type == 'triangular':
            if low is None or high is None or mode is None:
                raise ValueError("Triangular requires 'low', 'high', and 'mode/base'")
            if not (low <= mode <= high):
                raise ValueError("Triangular requires low <= mode <= high")
            return np.random.triangular(low, mode, high)

        elif dist_type == 'discrete_uniform':
            if low is None or high is None or mode is None:
                raise ValueError("Discrete uniform requires 'low', 'high', and 'mode'")

            # Default equal probabilities
            probs = probs or [1/3, 1/3, 1/3]
            
            # Validate probability length
            if len(probs) != 3:
                raise ValueError("Discrete uniform 'probs' must be a list of three values for low, mode, and high")

            return np.random.choice([low, mode, high], p=probs)


        elif dist_type == 'beta_pert':
            if low is None or high is None or mode is None:
                raise ValueError("Beta-PERT requires 'low', 'high', and 'mode/base'")
            # Calculate alpha and beta using PERT formula
            mean = (low + 4 * mode + high) / 6
            alpha = ((mean - low) * (2 * mode - low - high)) / ((mode - mean) * (high - low)) if mode != mean else 2.0
            beta_val = alpha * (high - mean) / (mean - low) if (mean - low) != 0 else 2.0
            sample = beta.rvs(alpha, beta_val)
            return low + sample * (high - low)

        else:
            raise ValueError(f"Unsupported distribution type: {dist_type}")
        
        
    def run(self):
        all_results = []

        for _ in range(self.n_simulations):
            # --- Sample event factors (always consolidated) ---
            event_factors = sum(self.sample_from_distribution(**event) for event in self.event_params)

            # --- Sample all parameters (no lists expected) ---
            sampled_values = {}
            for key, param in self.param_dict.items():
                if isinstance(param, list):
                    raise ValueError("SKU-level split provided, but SKU loop is disabled.")
                sampled_values[key] = self.sample_from_distribution(**param)

            # --- Compute net sales ---
            cs = sampled_values["class_share"]
            ps = sampled_values["product_share"]
            gp = sampled_values["gross_price"]
            gtn = sampled_values["gtn"]

            net_sales = self.final_baseline_trend * (1 + event_factors) * cs * ps * gp * (1 - gtn)

            # --- Store full result ---
            all_results.append({
                "event_factors": round(event_factors, 4),
                "class_share": round(cs, 4),
                "product_share": round(ps, 4),
                "gross_price": round(gp, 4),
                "gtn": round(gtn, 4),
                "net_sales": round(net_sales, 2)
            })

        net_sales_array = np.array([result["net_sales"] for result in all_results])
        summary = {
            "mean": round(float(np.mean(net_sales_array)), 2),
            "median": round(float(np.median(net_sales_array)), 2),
            "std": round(float(np.std(net_sales_array)), 2),
            "p5": round(float(np.percentile(net_sales_array, 5)), 2),
            "p10": round(float(np.percentile(net_sales_array, 10)), 2),
            "p25": round(float(np.percentile(net_sales_array, 25)), 2),
            "p50": round(float(np.percentile(net_sales_array, 50)), 2),
            "p75": round(float(np.percentile(net_sales_array, 75)), 2),
            "p90": round(float(np.percentile(net_sales_array, 90)), 2),
            "p95": round(float(np.percentile(net_sales_array, 95)), 2),
            "n_simulations": self.n_simulations
        }

        return {
            "summary": summary,
            "all_results": all_results
        }
