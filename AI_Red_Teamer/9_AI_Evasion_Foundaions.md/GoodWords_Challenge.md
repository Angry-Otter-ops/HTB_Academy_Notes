## Quick start

1. first, export your BASE_URL variable to make your life easier. 

```
export BASE_URL='http://94.237.52.164:49058'

```

2. Check the status of the instance. 
```
curl -s "94.237.52.164:49058/health"
```
Results:

```
{"status":"healthy"}
```
3. Retrieve the challenge parameters to understand the base spam message and constraints
```
curl -s "export BASE_URL='http://94.237.52.164:49058'
/challenge" | jq
```

4. Test the predication endpoint to confirm base message is classified as spam with high probability.

```
curl -s -X POST "http://STMIP:STMPO/predict" \
  -H 'content-type: application/json' \
  -d '{"text": "England v Macedonia - dont miss the goals/team news. Txt ur national team to 87077 eg ENGLAND to 87077 Try:WALES, SCOTLAND 4txt/ú1.20 POBOXox36504W45WQ 16+"}'
```

Results: 
```
{
  "base_message": "England v Macedonia - dont miss the goals/team news. Txt ur national team to 87077 eg ENGLAND to 87077 Try:WALES, SCOTLAND 4txt/ú1.20 POBOXox36504W45WQ 16+",
  "max_added_words": 25,
  "target_label": "ham"
}
```
5. Another baseline test  
```
curl -s -X POST "http://94.237.52.164:49058/predict" -H 'content-type: application/json' -d '{"text": "GET RICH QUICK!!! FREE MONEY NOW!!! CLICK HERE FOR INSTANT CASH!!!"}'
```
Results:

```
{"label":"spam","spam_probability":0.8994947430284234}  
```
6. Create a Python script to retrieve the challenge and print baseline label probablilty. 

```
import os, requests
host = os.getenv("BASE_URL", "http://127.0.0.1:8080")
ch = requests.get(f"{host}/challenge", timeout=10).json()
base = ch["base_message"]
res = requests.post(f"{host}/predict", json={"text": base}, timeout=15).json()
print(res)
```
Results:

```
Base spam probability: 0.9999999983049008
```
7. Create a Python script to measure single word impacts. 
```
import os, requests, random, numpy as np
random.seed(1337); np.random.seed(1337)
host = os.getenv("BASE_URL", "http://127.0.0.1:8080")
ch = requests.get(f"{host}/challenge", timeout=10).json()
base, budget = ch["base_message"], int(ch["max_added_words"])

def predict(t):
    return requests.post(f"{host}/predict", json={"text": t}, timeout=15).json()

base_p = predict(base)["spam_probability"]
vocab = ["please","thanks","meeting","tomorrow","coffee","home","support","good","great","safe"]
imp = []
for w in vocab:
    p2 = predict(base + " " + w)["spam_probability"]
    imp.append((w, base_p - p2))
imp.sort(key=lambda x: x[1], reverse=True)
top = [w for w, d in imp if d > 0][: max(2*budget, 20)]

aug = base
for i, w in enumerate(top, 1):
    if i > budget: break
    aug = aug + " " + w
    lab = predict(aug)["label"]
    if lab == "ham":
        break

print(requests.post(f"{host}/submit", json={"augmented_text": aug}, timeout=15).json())
```
Results:
```
Base message: 'England v Macedonia - dont miss the goals/team news. Txt ur national team to 87077 eg ENGLAND to 87077 Try:WALES, SCOTLAND 4txt/ú1.20 POBOXox36504W45WQ 16+'
Budget: 25


Final message: 'England v Macedonia - dont miss the goals/team news. Txt ur national team to 87077 eg ENGLAND to 87077 Try:WALES, SCOTLAND 4txt/ú1.20 POBOXox36504W45WQ 16+ later home morning fine nice lunch meeting happy'
Words used: 8
Final classification: ham (spam prob: 0.1077)
```

8. submit the augmented message to obtain the flag
```  
curl -s -X POST "$BASE_URL/submit" \
  -H 'content-type: application/json' \
  -d '{"augmented_text": "England v Macedonia - dont miss the goals/team news. Txt ur national team to 87077 eg ENGLAND to 87077 Try:WALES, SCOTLAND 4txt/ú1.20 POBOXox36504W45WQ 16+ later home morning fine nice lunch meeting happy"}' | jq
```
Results:

```
{
  "result": "success",
  "details": {
    "label": "ham",
    "spam_probability": 0.10773723362627051,
    "words_added": 8
  },
  "flag": "HTB{FLAG}"
}
```