1. Verify instance is running and set the base URL

```
export BASE_URL='http://94.237.56.25:49051'
curl -s "$BASE_URL/health" | jq

```

Results:
```
{
  "service": "skills_assessment_lab",
  "status": "healthy"
}
```

2. Retrieve white-box challenge data

```
curl -s "$BASE_URL/challenge/whitebox" | jq
```

Response: Stored in white_box.json

3. Download complete model bundle for analysis

```
curl -s "$BASE_URL/model/download" -o model.pkl
```
