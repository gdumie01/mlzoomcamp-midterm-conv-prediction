import requests
import json

visitor = {
    'Age': 32,
    'Gender' : 'Male',
    'Location': 'Islamabad',
    'LeadSource': 'Social Media',
    'TimeSpent (minutes)' : 60,
    'PagesViewed': 15,
    'LeadStatus': 'Hot',
    'EmailSent': 4,
    'DeviceType': 'Desktop',
    'ReferralSource': 'Facebook',
    'FormSubmissions': 1,
    'Downloads': 2,
    'CTR_ProductPage': 0.1,
    'ResponseTime (hours)': 14,
    'FollowUpEmails': 4,
    'SocialMediaEngagement': 115,
    'PaymentHistory': 'Good'
}

url = 'http://localhost:9696/predict'
response = requests.post(url, json=visitor)
result = response.json()

print(json.dumps(result, indent=2))