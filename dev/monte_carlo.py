import numpy as np

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

    def sample_from_distribution(self, dist_type, low=None, high=None, mean=None, std=None, base=None, mode=None):
        mode = base  # Support legacy key

        if dist_type == 'uniform':
            if low is None or high is None:
                raise ValueError("Uniform distribution requires 'low' and 'high'")
            return np.random.uniform(low, high)

        elif dist_type == 'normal':
            # Infer mean/std from bounds if not provided
            if mean is None or std is None:
                if low is not None and high is not None:
                    mean = (low + high) / 2
                    std = abs(high - low) / 4  # Make sure it's positive
                else:
                    raise ValueError("Normal distribution requires 'mean/std' or 'low/high'")
            if std < 0:
                raise ValueError(f"Standard deviation must be >= 0, got {std}")
            return np.random.normal(mean, std)

        elif dist_type == 'triangular':
            if low is None or high is None or mode is None:
                raise ValueError("Triangular distribution requires 'low', 'high', and 'mode/base'")
            if not (low <= mode <= high):
                raise ValueError(f"Triangular requires low <= mode <= high, got low={low}, mode={mode}, high={high}")
            return np.random.triangular(low, mode, high)

        elif dist_type == 'discrete_uniform':
            if low is None or high is None or mode is None:
                raise ValueError("Discrete uniform requires 'low', 'high', and 'mode/base'")
            return np.random.choice([low, mode, high])

        else:
            raise ValueError(f"Unsupported distribution type: {dist_type}")





    def run(self):
        # Detect how many SKUs by finding the first split param
        sku_split_start = None
        n_skus = 1
        for i, key in enumerate(self.param_dict):
            if isinstance(self.param_dict[key], list):
                sku_split_start = key
                n_skus = len(self.param_dict[key])
                break

        all_results = []

        for _ in range(self.n_simulations):
            # --- Event factors are always consolidated ---
            event_factors = sum(self.sample_from_distribution(**event) for event in self.event_params)

            sampled_params = {}

            # --- Step 1: Sample consolidated values BEFORE split ---
            split_started = False
            for key, param in self.param_dict.items():
                if isinstance(param, list):
                    split_started = True
                    break
                sampled_params[key] = self.sample_from_distribution(**param)

            # --- Step 2: SKU Loop ---
            sku_results = []
            for i in range(n_skus):
                cs = sampled_params['class_share'] if 'class_share' in sampled_params else \
                    self.sample_from_distribution(**self.param_dict['class_share'][i])
                ps = sampled_params['product_share'] if 'product_share' in sampled_params else \
                    self.sample_from_distribution(**self.param_dict['product_share'][i])
                gp = sampled_params['gross_price'] if 'gross_price' in sampled_params else \
                    self.sample_from_distribution(**self.param_dict['gross_price'][i])
                gtn = sampled_params['gtn'] if 'gtn' in sampled_params else \
                    self.sample_from_distribution(**self.param_dict['gtn'][i])

                net_sales = self.final_baseline_trend * (1 + event_factors) * cs * ps * gp * (1 - gtn)
                sku_results.append(net_sales)

            total_sales = sum(sku_results)
            all_results.append(total_sales)

        all_results = np.array(all_results)
        summary = {
            "mean": round(float(np.mean(all_results)), 2),
            "median": round(float(np.median(all_results)), 2),
            "std": round(float(np.std(all_results)), 2),
            "p5": round(float(np.percentile(all_results, 5)), 2),
            "p10": round(float(np.percentile(all_results, 10)), 2),
            "p25": round(float(np.percentile(all_results, 25)), 2),
            "p50": round(float(np.percentile(all_results, 50)), 2),
            "p75": round(float(np.percentile(all_results, 75)), 2),
            "p90": round(float(np.percentile(all_results, 90)), 2),
            "p95": round(float(np.percentile(all_results, 95)), 2),
            "n_simulations": self.n_simulations
        }

        return {
            "summary": summary,
            "all_results": [round(float(x), 2) for x in all_results]
        }
