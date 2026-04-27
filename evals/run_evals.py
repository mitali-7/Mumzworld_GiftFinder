import requests
import json

API_URL = "http://127.0.0.1:8000/find-gifts"

with open("test_cases.json") as f:
    test_cases = json.load(f)

results = []

for i, test in enumerate(test_cases):
    query = test["query"]

    try:
        response = requests.post(API_URL, json={"query": query})
        data = response.json()

        success = True

        # Basic checks
        if not query.strip():
            success = response.status_code == 400
        elif test["type"] == "invalid":
            success = data.get("out_of_scope", False)
        elif test["type"] == "edge":
            success = len(data.get("recommendations", [])) == 0 or data.get("uncertainty_note")
        else:
            success = len(data.get("recommendations", [])) > 0

    except Exception as e:
        success = False

    results.append({
        "query": query,
        "success": success
    })

# Print results
for r in results:
    print(f"{r['query']} → {'PASS' if r['success'] else 'FAIL'}")

print("\nSummary:")
print(f"{sum(r['success'] for r in results)}/{len(results)} passed")