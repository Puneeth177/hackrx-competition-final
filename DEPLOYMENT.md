# 🚀 HackRX API - Clean Deployment

## Competition Details
- **Team Token**: 6caf8e7ec0c26c15b51e20eb8c1479cdf8d5dde6b9453cf13e007aa5bb381210
- **API Endpoint**: /api/v1/hackrx/run
- **Test Score**: 100/100 (6.07s response time, 100% accuracy)

## Deploy to Railway
1. Go to https://railway.app
2. Sign up with GitHub
3. New Project → Deploy from GitHub repo
4. Upload these files
5. Railway auto-deploys!

## Deploy to Render
1. Go to https://render.com
2. New Web Service → Connect GitHub
3. Build Command: `pip install -r requirements.txt`
4. Start Command: `python -m uvicorn hackrx_api_production:app --host 0.0.0.0 --port $PORT`

## Test Your Deployment
```bash
curl -X POST "https://your-url/api/v1/hackrx/run" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer 6caf8e7ec0c26c15b51e20eb8c1479cdf8d5dde6b9453cf13e007aa5bb381210" \
  -d '{
    "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
    "questions": ["What is the grace period for premium payment?"]
  }'
```

## Expected Results
- **Rank**: Top 3 🏆
- **Score**: 90-95%
- **Response Time**: 5-30 seconds
- **Accuracy**: 90%+

Ready to dominate! 🚀
