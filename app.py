from flask import Flask, request, jsonify, render_template
import numpy as np
from scipy.stats import t
from statistics import stdev
from scipy import stats

app = Flask(__name__)

def two_sample(a, b, alternative):
    xbar1 = np.mean(a)
    xbar2 = np.mean(b)

    sd1 = stdev(a)
    sd2 = stdev(b)

    n1 = len(a)
    n2 = len(b)

    df = n1 + n2 - 2
    se = np.sqrt((sd1**2) / n1 + (sd2**2) / n2)

    tcal = ((xbar1 - xbar2) - 0) / se

    if alternative == "two-sided":
        p_value = 2 * (1 - t.cdf(abs(tcal), df))
    elif alternative == "left":
        p_value = t.cdf(tcal, df)
    elif alternative == "right":
        p_value = 1 - t.cdf(tcal, df)
    else:
        return {"error": "Invalid alternative hypothesis"}

    scipy_result = stats.ttest_ind(a, b, alternative=alternative, equal_var=False)

    return {
        "mean_sample1": float(xbar1),
        "mean_sample2": float(xbar2),
        "t_statistic": float(tcal),
        "p_value_manual": float(p_value),
        "scipy_t_statistic": float(scipy_result.statistic),
        "scipy_p_value": float(scipy_result.pvalue)
    }

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ttest", methods=["POST"])
def ttest():
    try:
        data = request.get_json()

        sample1 = list(map(float, data.get("sample1")))
        sample2 = list(map(float, data.get("sample2")))
        alternative = data.get("alternative", "two-sided")

        if len(sample1) < 2 or len(sample2) < 2:
            return jsonify({"error": "Each sample must contain at least 2 values"}), 400

        result = two_sample(sample1, sample2, alternative)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)