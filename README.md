#Intro
curl -H "Content-Type: application/json"  -d '{"data": [1,2,3,1]}' http://localhost:5000/api/predict
curl -H "Content-Type: application/json"  -d '{"data": [1,2,3,1]}' https://heroku-predict.heroku/api/predict
