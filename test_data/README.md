The data here contains \~47K issues and PRs retrieved from the `apache/arrow` GitHub repo on 13th June 2025. 

Only a subset of fields are retained:

- 'url'
- 'title'
- 'created_at'
- 'user_login'
- 'labels'
- 'closed_at'
- 'pull_request'
- 'body'
- 'state'

It should not be used for deployed items, but can be used for testing.

Load into a Python session
``` py
with gzip.open("test_data/issues_min.json.gz", "rt", encoding="utf-8") as f:
    data = json.load(f)
```
