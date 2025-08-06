
import importlib.util
import sys
import os

# Dynamically import monte-carlo.py as monte_carlo
spec = importlib.util.spec_from_file_location("monte_carlo", os.path.join(os.path.dirname(__file__), "monte-carlo.py"))
monte_carlo = importlib.util.module_from_spec(spec)
sys.modules["monte_carlo"] = monte_carlo
spec.loader.exec_module(monte_carlo)


def test_monte_carlo_sku_split():
    n_simulations = 1000
    final_baseline_trend = 1000
    # Example: SKU split for product_share_param, consolidated for others
    event_params = [
        {"dist_type": "uniform", "low": 0.01, "high": 0.05} for _ in range(3)
    ]
    class_share_param = {"dist_type": "uniform", "low": 0.2, "high": 0.3}
    product_share_param = [
        {"dist_type": "uniform", "low": 0.5, "high": 0.6},
        {"dist_type": "uniform", "low": 0.6, "high": 0.7}
    ]
    gross_price_param = {"dist_type": "normal", "low": 9, "high": 11}
    gtn_param = {"dist_type": "uniform", "low": 0.1, "high": 0.2}

    result = monte_carlo.monte_carlo_sku_split(
        n_simulations,
        final_baseline_trend,
        event_params,
        class_share_param,
        product_share_param,
        gross_price_param,
        gtn_param
    )
    for sku, res in result.items():
        print(f"{sku} Summary:", res["summary"])
        print(f"{sku} All results (first 5):", res["all_results"][:5])


# def test_monte_carlo_simulation():
#     n_simulations = 1000
#     final_baseline_trend = 1000
#     event_params = [
#         {"dist_type": "uniform", "low": 0.01, "high": 0.05} for _ in range(3)
#     ]
#     class_share_param = {"dist_type": "uniform", "low": 0.2, "high": 0.3}
#     product_share_param = {"dist_type": "uniform", "low": 0.5, "high": 0.6}
#     gross_price_param = {"dist_type": "normal", "mean": 10, "std": 1}
#     gtn_param = {"dist_type": "uniform", "low": 0.1, "high": 0.2}

#     result = monte_carlo.monte_carlo_simulation(
#         n_simulations,
#         final_baseline_trend,
#         event_params,
#         class_share_param,
#         product_share_param,
#         gross_price_param,
#         gtn_param
#     )
#     print("Summary:", result["summary"])
#     print("All results (first 5):", result["all_results"][:5])

if __name__ == "__main__":
    test_monte_carlo_sku_split()
