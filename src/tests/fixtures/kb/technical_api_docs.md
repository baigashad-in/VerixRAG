# VerixAPI Developer Documentation

## Rate Limits

Rate limits are enforced per API key: 1000 requests per minute for the
Standard plan, 5000 requests per minute for the Business plan, and
unlimited for Enterprise.

## Create User

POST /api/v2/users creates a new user account. The password must be at
least 12 characters and include uppercase, lowercase, number, and
special character.
