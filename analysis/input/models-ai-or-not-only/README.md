1) SH Labs

curl -X POST https://shlabs.music/api/v1/detect \
  -H "X-API-Key: $SHLABS_API_KEY$" \
  -H "Content-Type: application/json" \
  -d '{"audioUrl": "https://example.com/audio.mp3"}'

2) Letssubmit

import requests

api_url = 'https://api.letssubmit.com/analyze_song'
api_key = '$LETSSUBMIT_API_KEY'
file_url = 'https://example.com/path/to/your/file.mp3'

headers = {
    'Authorization': f'Bearer {api_key}',
    'Content-Type': 'application/json'
}

data = {
    'file_url': file_url
}

response = requests.post(api_url, headers=headers, json=data)

if response.status_code == 200:
    print('Success:', response.json())
else:
    print('Error:', response.status_code, response.text)

API Documentation
Welcome to the AI Music Checker API documentation. Everything you need to integrate our API into your applications.

Base URL
All API requests use this base URL:

https://api.letssubmit.com
Authentication
Include your API key in the Authorization header:

Authorization: Bearer $LETSSUBMIT_API_KEY
Endpoints
Analyze Song
Determine if a song is AI-generated.

By File (MP3 or WAV)
POST /analyze_song
Content-Type: application/json

{
    "file_url": "https://example.com/path/to/your/file.mp3"
}
By Streaming URL
Supports Spotify, YouTube, SoundCloud, and Deezer URLs:

POST /analyze_song
Content-Type: application/json

{
    "url": "https://open.spotify.com/track/{track_id}"
}
Note: spotify_url is still accepted as a backward-compatible alias for url.

Response
{
    "ai_probability": 85
}
Error Responses
The API returns standard HTTP status codes:

# 401 Unauthorized — Invalid or expired API key
{"error": "Invalid API key"}

# 400 Bad Request — Missing required fields
{"error": "Provide file_url (direct audio) or url (Spotify, YouTube, SoundCloud, Deezer)"}

# 405 Method Not Allowed — Wrong HTTP method
{"error": "Method not allowed"}
Code Examples
Using cURL
curl -X POST https://api.letssubmit.com/analyze_song \
     -H "Authorization: Bearer $LETSSUBMIT_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{"file_url": "https://example.com/path/to/your/file.mp3"}'
Using JavaScript (Fetch)
const apiUrl = 'https://api.letssubmit.com/analyze_song';
const apiKey = '$LETSSUBMIT_API_KEY';
const fileUrl = 'https://example.com/path/to/your/file.mp3';

const data = {
    file_url: fileUrl
};

fetch(apiUrl, {
    method: 'POST',
    headers: {
        'Authorization': `Bearer ${apiKey}`,
        'Content-Type': 'application/json'
    },
    body: JSON.stringify(data)
})
.then(response => response.json())
.then(data => {
    console.log('Success:', data);
})
.catch(error => {
    console.error('Error:', error);
});
Using Python
import requests

api_url = 'https://api.letssubmit.com/analyze_song'
api_key = '$LETSSUBMIT_API_KEY'
file_url = 'https://example.com/path/to/your/file.mp3'

headers = {
    'Authorization': f'Bearer {api_key}',
    'Content-Type': 'application/json'
}

data = {
    'file_url': file_url
}

response = requests.post(api_url, headers=headers, json=data)

if response.status_code == 200:
    print('Success:', response.json())
else:
    print('Error:', response.status_code, response.text)
Using PHP
<?php

$apiUrl = 'https://api.letssubmit.com/analyze_song';
$apiKey = '$LETSSUBMIT_API_KEY';
$fileUrl = 'https://example.com/path/to/your/file.mp3';

$data = [
    'file_url' => $fileUrl
];

$options = [
    'http' => [
        'header'  => [
            "Authorization: Bearer $apiKey",
            "Content-Type: application/json"
        ],
        'method'  => 'POST',
        'content' => json_encode($data),
    ],
];

$context  = stream_context_create($options);
$result = file_get_contents($apiUrl, false, $context);

echo $result;

?>